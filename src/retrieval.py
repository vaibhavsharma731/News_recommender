"""
Step 2 - The search engines.

  BM25Index    keyword search (great for names and exact terms)
  DenseIndex   semantic search with neural embeddings (understands meaning)
  TfidfIndex   light fallback for semantic search if the neural model is unavailable

Both semantic indexes offer the same three methods, so the rest of the code
does not care which one is in use:
  scores_for_text(text, short)   similarity of every article to a text query
  scores_for_doc(position)       similarity of every article to one article
  similarity_matrix(positions)   article-to-article similarity (used for diversity)
"""
import hashlib
import math
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np
import torch

# Use all available CPU cores for PyTorch matrix operations
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
    torch.set_num_threads(max(1, os.cpu_count() or 4))
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

import config

import difflib

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Retain English stopwords but preserve essential negations for news sentiment/context
_NEGATIONS = {"no", "not", "nor", "neither", "never"}
_EFFECTIVE_STOP_WORDS = ENGLISH_STOP_WORDS - _NEGATIONS

# Common irregular / non-standard news verbs and nouns for fast mapping
_LEMMA_MAP = {
    "countries": "country",
    "cities": "city",
    "parties": "party",
    "authorities": "authority",
    "companies": "company",
    "travels": "travel",
    "travelled": "travel",
    "travelling": "travel",
    "traveler": "travel",
    "travelers": "travel",
    "traveller": "travel",
    "travellers": "travel",
    "visited": "visit",
    "visiting": "visit",
    "visits": "visit",
    "written": "write",
    "writers": "writer",
    "writing": "write",
    "writes": "write",
    "died": "die",
    "dying": "die",
    "dies": "die",
    "elections": "election",
    "arrested": "arrest",
    "arrests": "arrest",
    "killed": "kill",
    "killing": "kill",
    "kills": "kill",
    "accused": "accuse",
    "accuses": "accuse",
    "accusing": "accuse",
    "protested": "protest",
    "protesting": "protest",
    "protests": "protest",
    "attacks": "attack",
    "attacked": "attack",
    "attacking": "attack",
    "polluted": "pollute",
    "polluting": "pollute",
    "pollution": "pollute",
}

# Protected words that end in 's' or 'es' that should NOT be stripped (e.g. proper nouns, terms)
_PROTECTED_WORDS = {
    "delhi", "paris", "congress", "aiims", "baps", "press", "address", "process", 
    "access", "species", "series", "business", "status", "crisis", "campus",
    "focus", "bonus", "virus", "plus", "tennis", "bengaluru", "bengal"
}


def _lightweight_lemmatize(word: str) -> str:
    """
    Fast rule-based English lemmatization and inflection normalization.
    Preserves named entities, Indian proper nouns, and known technical terms.
    """
    if len(word) <= 3 or word in _PROTECTED_WORDS:
        return word

    if word in _LEMMA_MAP:
        return _LEMMA_MAP[word]

    # -ies -> -y (countries -> country, bodies -> body)
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    # -ing suffix: visiting -> visit, testing -> test, breathing -> breathe
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        # double consonant cleanup: running -> run, swimming -> swim
        if len(stem) > 3 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
            return stem[:-1]
        if stem.endswith(("at", "iz", "us", "iv")):
            return stem + "e"
        return stem

    # -ed suffix: walked -> walk, launched -> launch
    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        if len(stem) > 3 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
            return stem[:-1]
        if stem.endswith(("at", "iz", "us", "iv", "nc", "sc", "rv")):
            return stem + "e"
        return stem

    # -es suffix: beaches -> beach, boxes -> box, dishes -> dish
    if word.endswith("es") and len(word) > 4:
        if word.endswith(("shes", "ches", "sses", "xes", "zes")):
            return word[:-2]
        return word[:-1]

    # standard -s plural: parks -> park, books -> book, authors -> author
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]

    return word


def tokenize(text: str) -> list:
    """
    lowercase -> extract words -> drop stopwords (preserving negations) -> lemmatize inflections.
    Applied symmetrically to both article indexing and queries.
    """
    tokens = []
    for word in TOKEN_RE.findall(text.lower()):
        if len(word) < 2 or word in _EFFECTIVE_STOP_WORDS:
            continue
        lemma = _lightweight_lemmatize(word)
        tokens.append(lemma)
    return tokens


def _normalize_repeats(w: str) -> str:
    """Compress repeated characters: 'akkhhlesh' -> 'akhlesh', 'modiii' -> 'modi'."""
    return re.sub(r"(.)\1+", r"\1", w)


# ----------------------------------------------------------------------------
class BM25Index:
    """
    BM25F field-aware keyword ranking with built-in Indian entity fuzzy query expansion.

    Fields and Weights:
      - Title: weight 3.0, b=0.75
      - Description: weight 1.5, b=0.75
      - Body: weight 1.0, b=0.75

    BM25F calculates effective term frequency:
      tf_tilde(t, d) = sum_f [ w_f * tf_(f, t, d) / (1 - b_f + b_f * (len_f / avg_len_f)) ]
    Then applies standard term saturation:
      score = sum_t [ weight_t * IDF(t) * tf_tilde * (k1 + 1) / (tf_tilde + k1) ]
    """

    FUZZY_MIN_RATIO = 0.70       # SequenceMatcher similarity cutoff for corrections
    FUZZY_WEIGHT = 0.85          # Score weight for fuzzy matched terms

    # Field-aware configurations
    FIELD_WEIGHTS = {
        "title": 3.0,
        "description": 1.5,
        "body": 1.0,
    }
    FIELD_B = {
        "title": 0.75,
        "description": 0.75,
        "body": 0.75,
    }

    def __init__(self, docs_or_fields, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # Parse inputs into structured fields
        self.docs_fields = []
        self._parse_inputs(docs_or_fields)
        self.n = len(self.docs_fields)

        # Compute field lengths and average field lengths
        self.field_lens = {
            f: np.array([len(doc[f]) for doc in self.docs_fields], dtype=np.float32)
            for f in self.FIELD_WEIGHTS
        }
        self.avg_field_lens = {
            f: float(self.field_lens[f].mean()) if self.n and self.field_lens[f].mean() > 0 else 1.0
            for f in self.FIELD_WEIGHTS
        }

        # Keep legacy combined docs representation for length and evidence compatibility
        self.docs = [
            doc["title"] + doc["description"] + doc["body"]
            for doc in self.docs_fields
        ]
        self.lengths = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if self.n else 1.0

        # Build field-aware inverted index:
        # term -> [(doc_pos, tf_tilde), ...]
        # postings_by_pos: term -> {doc_pos: tf_tilde} for O(1) evidence lookup (FIX #2)
        self.postings = defaultdict(list)
        self.postings_by_pos = defaultdict(dict)
        doc_frequency = Counter()

        for pos, doc in enumerate(self.docs_fields):
            # Calculate term frequencies per field
            f_counts = {
                f: Counter(doc[f]) for f in self.FIELD_WEIGHTS
            }
            unique_terms = set(f_counts["title"].keys()) | set(f_counts["description"].keys()) | set(f_counts["body"].keys())

            for term in unique_terms:
                doc_frequency[term] += 1
                # Calculate normalized effective term frequency tf_tilde
                tf_tilde = 0.0
                for f, weight in self.FIELD_WEIGHTS.items():
                    count = f_counts[f].get(term, 0)
                    if count > 0:
                        b_val = self.FIELD_B[f]
                        len_norm = 1.0 - b_val + b_val * (self.field_lens[f][pos] / self.avg_field_lens[f])
                        tf_tilde += weight * (count / max(0.01, len_norm))
                tf_val = float(tf_tilde)
                self.postings[term].append((pos, tf_val))
                self.postings_by_pos[term][pos] = tf_val

        # Rare words get a high IDF weight, common words a low one
        self.idf = {
            term: math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for term, df in doc_frequency.items()
        }

        # Fast character 2-gram index over vocabulary for rapid candidate retrieval
        self._vocab = set(self.postings.keys())
        self._ngram_to_vocab: dict[str, set] = defaultdict(set)
        for word in self._vocab:
            w_pad = f"#{word}#"
            for i in range(len(w_pad) - 1):
                self._ngram_to_vocab[w_pad[i:i+2]].add(word)

    def _parse_inputs(self, docs_or_fields):
        """Standardizes input into list of dicts with tokenized title, description, and body."""
        # Check if pandas DataFrame
        if hasattr(docs_or_fields, "iterrows"):
            for _, row in docs_or_fields.iterrows():
                self.docs_fields.append({
                    "title": tokenize(str(row.get("title", "") or "")),
                    "description": tokenize(str(row.get("description", "") or "")),
                    "body": tokenize(str(row.get("body", "") or "")),
                })
            return

        # Check list items
        for item in docs_or_fields:
            if isinstance(item, dict):
                self.docs_fields.append({
                    "title": tokenize(str(item.get("title", "") or "")),
                    "description": tokenize(str(item.get("description", "") or "")),
                    "body": tokenize(str(item.get("body", "") or "")),
                })
            elif isinstance(item, (tuple, list)) and len(item) == 3:
                self.docs_fields.append({
                    "title": tokenize(str(item[0] or "")),
                    "description": tokenize(str(item[1] or "")),
                    "body": tokenize(str(item[2] or "")),
                })
            else:
                # String fallback (legacy text_bm25 or flat string)
                # Parse title/desc/body if separated by periods, or fallback to body
                text = str(item or "")
                parts = [p.strip() for p in text.split(". ") if p.strip()]
                if len(parts) >= 3:
                    # heuristic fallback
                    self.docs_fields.append({
                        "title": tokenize(parts[0]),
                        "description": tokenize(parts[1]),
                        "body": tokenize(" ".join(parts[2:])),
                    })
                elif len(parts) == 2:
                    self.docs_fields.append({
                        "title": tokenize(parts[0]),
                        "description": tokenize(parts[1]),
                        "body": [],
                    })
                else:
                    self.docs_fields.append({
                        "title": tokenize(text),
                        "description": [],
                        "body": [],
                    })

    def find_best_vocab_match(self, token: str, cutoff: float = 0.70) -> tuple[str | None, float]:
        """Find the closest vocabulary term using 2-gram candidate filtering + SequenceMatcher."""
        if token in self._vocab:
            return token, 1.0
        norm = _normalize_repeats(token)
        if norm in self._vocab:
            return norm, 0.95

        t_pad = f"#{token}#"
        n_pad = f"#{norm}#"
        ngrams = {t_pad[i:i+2] for i in range(len(t_pad) - 1)} | {n_pad[i:i+2] for i in range(len(n_pad) - 1)}
        candidates: Counter = Counter()
        for ng in ngrams:
            for w in self._ngram_to_vocab.get(ng, ()):
                if abs(len(w) - len(token)) <= 3 or abs(len(w) - len(norm)) <= 3:
                    candidates[w] += 1

        if not candidates:
            return None, 0.0

        best_word, best_ratio, best_count = None, 0.0, 0
        for cand in [w for w, _ in candidates.most_common(60)]:
            r1 = difflib.SequenceMatcher(None, token, cand).ratio()
            r2 = difflib.SequenceMatcher(None, norm, cand).ratio()
            r = max(r1, r2)
            if r >= cutoff:
                cnt = len(self.postings.get(cand, []))
                # Prefer significantly higher ratio, or if within 0.03 ratio, prefer much more frequent term
                if (r > best_ratio + 0.03) or (abs(r - best_ratio) <= 0.03 and cnt > best_count) or (r > best_ratio and best_word is None):
                    best_ratio = r
                    best_word = cand
                    best_count = cnt
        return best_word, best_ratio

    def normalize_query(self, query: str) -> tuple[str, list[tuple[str, str]], list[tuple[str, float]]]:
        """
        Processes raw user query into:
          - clean_query: rewritten string with typos corrected (for Dense Embeddings & Reranker)
          - corrections: list of (original, corrected) for UI display ('Did you mean...')
          - expanded_terms: list of (vocab_term, weight) for BM25 multi-term scoring
        """
        raw_tokens = tokenize(query)
        if not raw_tokens:
            return query, [], []

        corrected_tokens: list[str] = []
        corrections: list[tuple[str, str]] = []
        expanded_terms: list[tuple[str, float]] = []

        i = 0
        while i < len(raw_tokens):
            # Check if adjacent pair of tokens should merge (e.g. 'akkh' + 'lesh' -> 'akhilesh')
            if i + 1 < len(raw_tokens):
                t1, t2 = raw_tokens[i], raw_tokens[i+1]
                # Only attempt merge if at least one token is unknown or weak
                if t1 not in self._vocab or t2 not in self._vocab:
                    merged = t1 + t2
                    match_word, score = self.find_best_vocab_match(merged, cutoff=0.74)
                    if match_word and score >= 0.80:
                        corrected_tokens.append(match_word)
                        corrections.append((f"{t1} {t2}", match_word))
                        expanded_terms.append((match_word, self.FUZZY_WEIGHT * score))
                        i += 2
                        continue

            token = raw_tokens[i]
            if token in self._vocab:
                corrected_tokens.append(token)
                expanded_terms.append((token, 1.0))
            else:
                match_word, score = self.find_best_vocab_match(token, cutoff=self.FUZZY_MIN_RATIO)
                if match_word:
                    corrected_tokens.append(match_word)
                    corrections.append((token, match_word))
                    expanded_terms.append((match_word, self.FUZZY_WEIGHT * score))
                else:
                    corrected_tokens.append(token)
                    expanded_terms.append((token, 1.0))
            i += 1

        clean_query = " ".join(corrected_tokens)
        return clean_query, corrections, expanded_terms

    def scores(self, query: str) -> np.ndarray:
        """BM25F score for every article with fuzzy entity matching."""
        result = np.zeros(self.n)
        _, _, expanded_terms = self.normalize_query(query)
        for term, weight in expanded_terms:
            idf = self.idf.get(term, 0.0)
            if idf <= 0.0:
                continue
            for pos, tf_tilde in self.postings.get(term, []):
                # Standard BM25F term saturation on the field-weighted normalized frequency
                result[pos] += weight * idf * (tf_tilde * (self.k1 + 1.0)) / (tf_tilde + self.k1)
        return result

    def query_expand(self, query: str) -> list[str]:
        """
        Return the list of corrected vocabulary terms.
        Used by UI to display 'Auto-corrected: akkhhlesh -> akhilesh'.
        """
        _, corrections, _ = self.normalize_query(query)
        return [f"{orig} -> {corr}" for orig, corr in corrections]




# ----------------------------------------------------------------------------
class DenseIndex:
    """Neural embeddings: every article becomes a vector; similar meaning = close vectors."""

    name = "dense"

    def __init__(self, texts, model_name=config.EMBEDDING_MODEL, vectors=None, use_disk_cache=True):
        from src.model_registry import get_embedder

        texts = list(texts)
        self.model = get_embedder(model_name)
        if vectors is not None:
            stored = np.asarray(vectors, dtype=np.float32)
            if stored.ndim == 2 and stored.shape[0] == len(texts):
                self.vectors = stored
                return
            print("[DenseIndex] Stored embeddings do not match article count; rebuilding.")
        self.vectors = self._load_or_build(texts, model_name, use_disk_cache=use_disk_cache)

    def _load_or_build(self, texts, model_name, use_disk_cache=True) -> np.ndarray:
        max_seq_len = getattr(config, "EMBED_MAX_SEQ_LEN", 256)
        if not use_disk_cache:
            self.model.max_seq_length = max_seq_len
            return self.model.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model_slug = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", model_name)
        master_cache_file = config.CACHE_DIR / f"text_embedding_cache_{model_slug}_len{max_seq_len}.npz"
        legacy_cache_file = config.CACHE_DIR / "text_embedding_cache.npz"

        # Load persistent incremental hash cache if available
        cache_map = {}
        legacy_map = {}
        if master_cache_file.exists():
            try:
                data = np.load(master_cache_file)
                cached_hashes = data["hashes"]
                cached_vecs = data["vectors"]
                for h, v in zip(cached_hashes, cached_vecs):
                    cache_map[str(h)] = v
            except Exception as e:
                print(f"[DenseIndex] Warning: Failed to read master cache ({e}), rebuilding.")
        elif legacy_cache_file.exists() and model_name == "BAAI/bge-base-en-v1.5":
            # Transparent migration: read legacy cache keyed on sha256(text)
            try:
                data = np.load(legacy_cache_file)
                legacy_hashes = [str(h) for h in data["hashes"]]
                legacy_vecs = data["vectors"]
                legacy_map = dict(zip(legacy_hashes, legacy_vecs))
                print(f"[DenseIndex] Found legacy cache with {len(legacy_map)} vectors; migrating to model-keyed cache.")
            except Exception as e:
                print(f"[DenseIndex] Warning: Failed to read legacy cache ({e}).")
                legacy_map = {}

        # Content key includes model name + max_seq_length + text content
        def _content_key(t: str) -> str:
            return hashlib.sha256(f"{model_name}|{max_seq_len}|{t}".encode("utf-8")).hexdigest()

        text_hashes = [_content_key(t) for t in texts]
        vectors = [None] * len(texts)
        missing_indices = []
        missing_texts = []

        for idx, (t, h) in enumerate(zip(texts, text_hashes)):
            if h in cache_map:
                vectors[idx] = cache_map[h]
            elif legacy_map and (legacy_h := hashlib.sha256(t.encode("utf-8")).hexdigest()) in legacy_map:
                vec = legacy_map[legacy_h]
                vectors[idx] = vec
                cache_map[h] = vec
            else:
                missing_indices.append(idx)
                missing_texts.append(t)

        if missing_texts:
            print(f"[DenseIndex] {len(texts) - len(missing_texts)}/{len(texts)} articles found in cache.")
            print(f"[DenseIndex] Encoding {len(missing_texts)} new articles (model={model_name}, max_len={max_seq_len})...")
            self.model.max_seq_length = max_seq_len
            new_vecs = self.model.encode(
                missing_texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            for idx, vec in zip(missing_indices, new_vecs):
                vectors[idx] = vec
                cache_map[text_hashes[idx]] = vec

            # Update persistent master cache
            try:
                all_hashes = np.array(list(cache_map.keys()))
                all_vecs = np.array(list(cache_map.values()))
                np.savez_compressed(master_cache_file, hashes=all_hashes, vectors=all_vecs)
                print(f"[DenseIndex] Updated cache: {len(cache_map)} total vectors saved to {master_cache_file.name}")
            except Exception as e:
                print(f"[DenseIndex] Warning: Could not save master cache: {e}")
        else:
            print(f"[DenseIndex] All {len(texts)} articles found in incremental cache.")
            # If we migrated from legacy, persist the new master cache file once
            if not master_cache_file.exists() and cache_map:
                try:
                    all_hashes = np.array(list(cache_map.keys()))
                    all_vecs = np.array(list(cache_map.values()))
                    np.savez_compressed(master_cache_file, hashes=all_hashes, vectors=all_vecs)
                    print(f"[DenseIndex] Migrated {len(cache_map)} vectors saved to {master_cache_file.name}")
                except Exception as e:
                    print(f"[DenseIndex] Warning: Could not save migrated master cache: {e}")

        return np.asarray(vectors, dtype=np.float32)


    def encode_query(self, text: str, short: bool = False) -> tuple[np.ndarray, float]:
        """Encodes query text into a normalized embedding vector with precise timing."""
        if short:
            text = config.QUERY_PREFIX + text
        t0 = time.perf_counter()
        query_vec = self.model.encode(text, normalize_embeddings=True)
        embed_ms = (time.perf_counter() - t0) * 1000
        return query_vec, embed_ms

    def scan_matrix(self, query_vec: np.ndarray) -> tuple[np.ndarray, float]:
        """Scans the corpus vector matrix with dot product with precise timing."""
        t0 = time.perf_counter()
        scores = self.vectors @ query_vec
        scan_ms = (time.perf_counter() - t0) * 1000
        return scores, scan_ms

    def scores_for_text(self, text: str, short: bool = False) -> np.ndarray:
        if short:
            text = config.QUERY_PREFIX + text
        query_vec = self.model.encode(text, normalize_embeddings=True)
        return self.vectors @ query_vec

    def scores_for_doc(self, pos: int) -> np.ndarray:
        return self.vectors @ self.vectors[pos]

    def get_vector(self, pos: int) -> np.ndarray:
        return self.vectors[pos]

    def scores_for_vector(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return self.vectors @ vec

    def similarity_matrix(self, positions) -> np.ndarray:
        sub = self.vectors[positions]
        return sub @ sub.T


# ----------------------------------------------------------------------------
class TfidfIndex:
    """Light fallback: no downloads, no GPU. Matches words, not meaning."""

    name = "tfidf"

    def __init__(self, texts):
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(texts)   # rows are already L2-normalised

    def get_vector(self, pos: int):
        return self.matrix[pos]

    def scores_for_text(self, text: str, short: bool = False) -> np.ndarray:
        query_vec = self.vectorizer.transform([text])
        return (self.matrix @ query_vec.T).toarray().ravel()

    def scores_for_doc(self, pos: int) -> np.ndarray:
        return (self.matrix @ self.matrix[pos].T).toarray().ravel()

    def scores_for_vector(self, vec) -> np.ndarray:
        return (self.matrix @ vec.T).toarray().ravel()

    def similarity_matrix(self, positions) -> np.ndarray:
        sub = self.matrix[positions]
        return (sub @ sub.T).toarray()



# ----------------------------------------------------------------------------
def reciprocal_rank_fusion(rankings, k: int = config.RRF_K) -> dict:
    """
    Merge several ranked lists into one.
    An article near the top of ANY list gets points: 1 / (k + rank).
    Articles that appear high in BOTH lists win. Works without comparing raw scores.
    """
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, pos in enumerate(ranking, start=1):
            fused[pos] += 1.0 / (k + rank)
    return dict(fused)


def select_quota_candidates(
    bm25_rank: list,
    dense_rank: list,
    n_cand: int = config.N_CANDIDATES,
    k_rrf: int = config.RRF_K,
    rrf_quota: int = 12,
    dense_quota: int = 5,
    bm25_quota: int = 3,
    strong_dense_cutoff: int = 5,
    strong_bm25_cutoff: int = 5,
) -> list:
    """
    Quota-governed candidate selection for article-to-article recommendations.

    Shared symmetrically by both live serving (Recommender) and batch ingestion
    (_precompute_for_article).

    Guarantees:
      1. RRF top rankings form the core pool (default 12).
      2. Up to 5 strong Dense-only candidates (ranks 1..strong_dense_cutoff) that
         have no BM25 overlap are guaranteed entry to the Cross-Encoder.
      3. Up to 3 strong BM25-only candidates (ranks 1..strong_bm25_cutoff).
      4. Any unfilled quota positions are safely backfilled by the next highest RRF candidates.
      5. Final candidate count never exceeds n_cand and contains zero duplicates.
    """
    if n_cand <= 0:
        return []

    # Proportional scaling if n_cand is configured smaller than 20
    if n_cand < (rrf_quota + dense_quota + bm25_quota):
        scale = n_cand / 20.0
        rrf_quota = max(1, int(round(12 * scale)))
        dense_quota = max(1, int(round(5 * scale)))
        bm25_quota = max(1, int(round(3 * scale)))

    fused = reciprocal_rank_fusion([bm25_rank, dense_rank], k=k_rrf)
    rrf_sorted = sorted(fused, key=fused.get, reverse=True)

    bm25_set = set(bm25_rank)
    dense_set = set(dense_rank)

    selected = []
    seen = set()

    # 1. Top RRF quota
    for pos in rrf_sorted:
        if len(selected) >= rrf_quota or len(selected) >= n_cand:
            break
        if pos not in seen:
            selected.append(pos)
            seen.add(pos)

    # 2. Strong Dense-only quota (restricted to strong dense ranks 1..strong_dense_cutoff)
    dense_added = 0
    for pos in dense_rank[:strong_dense_cutoff]:
        if dense_added >= dense_quota or len(selected) >= n_cand:
            break
        if pos not in bm25_set and pos not in seen:
            selected.append(pos)
            seen.add(pos)
            dense_added += 1

    # 3. Strong BM25-only quota (restricted to strong BM25 ranks 1..strong_bm25_cutoff)
    bm25_added = 0
    for pos in bm25_rank[:strong_bm25_cutoff]:
        if bm25_added >= bm25_quota or len(selected) >= n_cand:
            break
        if pos not in dense_set and pos not in seen:
            selected.append(pos)
            seen.add(pos)
            bm25_added += 1

    # 4. Backfill remaining positions from next best RRF
    if len(selected) < n_cand:
        for pos in rrf_sorted:
            if len(selected) >= n_cand:
                break
            if pos not in seen:
                selected.append(pos)
                seen.add(pos)

    return selected[:n_cand]
