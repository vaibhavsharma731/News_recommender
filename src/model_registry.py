"""One shared copy of each neural model per process (local or remote cloud GPU)."""
import json
import os
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


# ─── Remote Cloud GPU Clients ────────────────────────────────────────────────

class RemoteEmbedderClient:
    """Wrapper that proxies .encode() calls to a remote GPU server over HTTP."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.max_seq_length = getattr(config, "EMBED_MAX_SEQ_LEN", 256)

    def encode(self, texts, batch_size: int = 32, normalize_embeddings: bool = True, show_progress_bar: bool = False, **kwargs) -> np.ndarray:
        is_single = isinstance(texts, str)
        batch = [texts] if is_single else list(texts)
        if not batch:
            return np.zeros((0, 768), dtype=np.float32)

        results = []
        chunk_size = 128  # Batch network requests to prevent timeout on huge payloads
        for i in range(0, len(batch), chunk_size):
            sub_batch = batch[i:i + chunk_size]
            payload = json.dumps({"texts": sub_batch}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/embed",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results.extend(data["vectors"])

        arr = np.asarray(results, dtype=np.float32)
        return arr[0] if is_single else arr


class RemoteCrossEncoderClient:
    """Wrapper that proxies .predict() calls to a remote GPU server over HTTP."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.max_length = getattr(config, "RERANK_MAX_LEN", 160)

    def predict(self, pairs, batch_size: int = 32, show_progress_bar: bool = False, activation_fn=None, **kwargs) -> np.ndarray:
        if not pairs:
            return np.array([], dtype=np.float32)

        pair_list = [[p[0], p[1]] for p in pairs]
        results = []
        chunk_size = 128
        for i in range(0, len(pair_list), chunk_size):
            sub_pairs = pair_list[i:i + chunk_size]
            payload = json.dumps({"pairs": sub_pairs}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/rerank",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results.extend(data["logits"])

        return np.asarray(results, dtype=np.float32)


def _check_remote_available(url: str) -> bool:
    """Quick 4-second ping to verify the remote GPU worker is live."""
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/health", headers={"User-Agent": "NewsRec-Client"})
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
                print(f"[ModelRegistry] 🚀 Connected to Remote Cloud GPU Worker! Embeddings offloaded to GPU.")
                return RemoteEmbedderClient(remote_url)
            else:
                print(f"[ModelRegistry] ⚠️ Remote worker at {remote_url} not responding. Falling back to local model.")

        # Local fallback
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)
        model.max_seq_length = config.EMBED_MAX_SEQ_LEN
        return model

    return _get(("embed", model_name, remote_url), build)


def get_cross_encoder(model_name=None):
    model_name = model_name or config.RERANKER_MODEL
    remote_url = getattr(config, "REMOTE_MODEL_URL", "").strip()

    def build():
        if remote_url:
            if _check_remote_available(remote_url):
                print(f"[ModelRegistry] 🚀 Cross-Encoder reranking offloaded to Remote Cloud GPU!")
                return RemoteCrossEncoderClient(remote_url)
            else:
                print(f"[ModelRegistry] ⚠️ Remote worker at {remote_url} not responding. Falling back to local cross-encoder.")

        # Local fallback
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name, max_length=config.RERANK_MAX_LEN)

    return _get(("rerank", model_name, remote_url), build)