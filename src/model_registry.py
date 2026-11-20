"""One shared copy of each neural model per process (local or remote cloud GPU)."""
import json
import threading
import urllib.error
import urllib.request
import numpy as np
import config

_lock = threading.Lock()
_cache = {}


def _get(key, factory):
    with _lock:
        if key not in _cache:
            _cache[key] = factory()
        return _cache[key]


def _remote_headers(content_type: bool = False) -> dict:
    """
    Headers for every call to the cloud GPU worker.

    The worker can be reached over a public Cloudflare tunnel, so it optionally
    requires a shared secret. Set ``REMOTE_MODEL_TOKEN`` in .env to the same value as
    the Colab worker's ``WORKER_TOKEN``. When the worker has no token configured this
    header is simply ignored.
    """
    headers = {"User-Agent": "NewsRec-Client"}
    if content_type:
        headers["Content-Type"] = "application/json"
    token = getattr(config, "REMOTE_MODEL_TOKEN", "").strip()
    if token:
        headers["X-Worker-Token"] = token
    return headers


# ─── Remote Cloud GPU Clients ────────────────────────────────────────────────

class RemoteEmbedderClient:
    """Wrapper that proxies .encode() calls to a remote GPU server over HTTP with local fallback."""

    def __init__(self, base_url: str, model_name: str = None):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.max_seq_length = getattr(config, "EMBED_MAX_SEQ_LEN", 256)
        self._local_fallback = None
        self.last_used_remote = True

    def _get_local(self):
        if self._local_fallback is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[ModelRegistry] Loading local embedder fallback ({self.model_name})...")
            m = SentenceTransformer(
                self.model_name,
                revision=getattr(config, "EMBEDDING_REVISION", None),
                device=device,
            )
            m.max_seq_length = self.max_seq_length
            self._local_fallback = m
        return self._local_fallback

    def encode(self, texts, batch_size: int = 32, normalize_embeddings: bool = True, show_progress_bar: bool = False, **kwargs) -> np.ndarray:
        is_single = isinstance(texts, str)
        batch = [texts] if is_single else list(texts)
        if not batch:
            return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)

        try:
            results = []
            chunk_size = 128  # Batch network requests to prevent timeout on huge payloads
            for i in range(0, len(batch), chunk_size):
                sub_batch = batch[i:i + chunk_size]
                payload = json.dumps({"texts": sub_batch}).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.base_url}/embed",
                    data=payload,
                    headers=_remote_headers(content_type=True)
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    results.extend(data["vectors"])

            arr = np.asarray(results, dtype=np.float32)
            # Verify dimension to ensure remote worker is not returning incompatible vectors
            if arr.ndim == 2 and arr.shape[1] != config.EMBEDDING_DIM:
                raise ValueError(
                    f"Remote embedder returned dimension {arr.shape[1]}, expected {config.EMBEDDING_DIM} "
                    f"({self.model_name}). Incompatible remote worker."
                )
            self.last_used_remote = True
            return arr[0] if is_single else arr
        except Exception as e:
            self.last_used_remote = False
            print(f"[ModelRegistry Warning] Remote embedder call failed ({e}). Falling back to local model.")
            return self._get_local().encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=show_progress_bar,
                **kwargs
            )


class RemoteCrossEncoderClient:
    """Wrapper that proxies .predict() calls to a remote GPU server over HTTP with local fallback."""

    def __init__(self, base_url: str, model_name: str = None):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name or config.RERANKER_MODEL
        self.max_length = getattr(config, "RERANK_MAX_LEN", 512)
        self._local_fallback = None
        self.last_used_remote = True

    def _get_local(self):
        if self._local_fallback is None:
            from sentence_transformers import CrossEncoder
            print(f"[ModelRegistry] Loading local cross-encoder fallback ({self.model_name})...")
            self._local_fallback = CrossEncoder(
                self.model_name,
                revision=getattr(config, "RERANKER_REVISION", None),
                max_length=self.max_length,
            )
        return self._local_fallback

    def predict(self, pairs, batch_size: int = 32, show_progress_bar: bool = False, activation_fn=None, **kwargs) -> np.ndarray:
        if not pairs:
            return np.array([], dtype=np.float32)

        try:
            pair_list = [[p[0], p[1]] for p in pairs]
            results = []
            chunk_size = 128
            for i in range(0, len(pair_list), chunk_size):
                sub_pairs = pair_list[i:i + chunk_size]
                payload = json.dumps({"pairs": sub_pairs}).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.base_url}/rerank",
                    data=payload,
                    headers=_remote_headers(content_type=True)
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    results.extend(data["logits"])

            self.last_used_remote = True
            return np.asarray(results, dtype=np.float32)
        except Exception as e:
            self.last_used_remote = False
            print(f"[ModelRegistry Warning] Remote cross-encoder call failed ({e}). Falling back to local model.")
            return self._get_local().predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                activation_fn=activation_fn,
                **kwargs
            )


def _check_remote_available(url: str) -> bool:
    """Quick 4-second ping to verify the remote GPU worker is live."""
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/health", headers=_remote_headers())
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("status") == "ok"
    except Exception:
        return False


# ─── Public Model Accessors ──────────────────────────────────────────────────

def get_embedder(model_name=None):
    model_name = model_name or config.EMBEDDING_MODEL
    remote_url = getattr(config, "REMOTE_MODEL_URL", "").strip()

    def build():
        if remote_url:
            print(f"[ModelRegistry] Checking remote cloud GPU worker at {remote_url}...")
            if _check_remote_available(remote_url):
                print(f"[ModelRegistry] [OK] Connected to Remote Cloud GPU Worker! Embeddings offloaded to GPU.")
                return RemoteEmbedderClient(remote_url, model_name)
            else:
                print(f"[ModelRegistry] [!] Remote worker at {remote_url} not responding. Falling back to local model.")

        # Local fallback
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(
            model_name,
            revision=getattr(config, "EMBEDDING_REVISION", None),
            device=device,
        )
        model.max_seq_length = config.EMBED_MAX_SEQ_LEN
        return model

    return _get(("embed", model_name, remote_url), build)


def get_cross_encoder(model_name=None):
    model_name = model_name or config.RERANKER_MODEL
    remote_url = getattr(config, "REMOTE_MODEL_URL", "").strip()

    def build():
        if remote_url:
            if _check_remote_available(remote_url):
                print(f"[ModelRegistry] [OK] Cross-Encoder reranking offloaded to Remote Cloud GPU!")
                return RemoteCrossEncoderClient(remote_url, model_name)
            else:
                print(f"[ModelRegistry] [!] Remote worker at {remote_url} not responding. Falling back to local cross-encoder.")

        # Local fallback
        from sentence_transformers import CrossEncoder
        return CrossEncoder(
            model_name,
            revision=getattr(config, "RERANKER_REVISION", None),
            max_length=config.RERANK_MAX_LEN,
        )

    return _get(("rerank", model_name, remote_url), build)