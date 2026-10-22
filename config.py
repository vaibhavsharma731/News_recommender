"""
Central settings. Every number you may want to tune lives here,
so you never have to hunt through the code.
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

# Dataset sources to load and merge:
# Loads both articles.csv and articles_2.csv by default if present.
# Can also be customized via DATA_CSV environment variable (comma-separated).
_env_csvs = os.getenv("DATA_CSV", "")
if _env_csvs:
    DATA_PATHS = [DATA_DIR / name.strip() for name in _env_csvs.split(",") if name.strip()]
else:
    _default_files = ["articles.csv", "articles_2.csv", "articles_clean_train_80.csv"]
    DATA_PATHS = [DATA_DIR / f for f in _default_files if (DATA_DIR / f).exists()]
    if not DATA_PATHS:
        DATA_PATHS = [p for p in DATA_DIR.glob("*.csv") if "uploads" not in str(p)]
    if not DATA_PATHS:
        DATA_PATHS = [DATA_DIR / "articles.csv"]

DATA_PATH = DATA_PATHS[0]                    # primary path for backward compatibility
CACHE_DIR = ROOT / "data" / "cache"          # embeddings are saved here after the first run

# --- Environment & HuggingFace authentication ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Token is now loaded from .env if present.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Set the environment variable NEWSREC_NEURAL=0 to skip them and use the light TF-IDF mode.
USE_NEURAL_MODELS = os.getenv("NEWSREC_NEURAL", "1") == "1"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_REVISION = os.getenv("NEWSREC_EMBED_REVISION", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a")
RERANKER_MODEL = os.getenv("NEWSREC_RERANKER", "BAAI/bge-reranker-base")
RERANKER_REVISION = os.getenv("NEWSREC_RERANK_REVISION", "2cfc18c9415c912f9d8155881c133215df768a70")
EMBED_MAX_SEQ_LEN = 256   # same for ingestion AND live search, so vectors are consistent
RERANK_MAX_LEN = 512        # total cross-encoder token limit (query + candidate combined)
RERANK_PER_ARTICLE_TOKENS = 256  # deliberate 50/50 split for article-to-article mode

# Optional: Remote Cloud GPU Worker (e.g. Google Colab + Cloudflare Tunnel).
# When set to a valid URL (e.g. https://xxx.trycloudflare.com), embeddings and rerankings
# run remotely on free cloud GPU instead of local CPU. If empty, runs locally.
REMOTE_MODEL_URL = os.getenv("REMOTE_MODEL_URL", "").strip().rstrip("/")

# --- Ingestion / precompute ---
PRECOMPUTE_CANDIDATES = 20  # fused candidates sent to the reranker during ingestion (matches live N_CANDIDATES)
PRECOMPUTE_COMMIT_EVERY = 5 # progress update + DB commit every N articles

# --- Incremental recommendation maintenance ---
# When a new CSV is ingested:
#   1. the NEW articles are precomputed against the WHOLE database, and
#   2. for every EXISTING ("old") article that a new article is close to, we
#      re-evaluate (O-centric) whether the new article deserves a slot in its top-K.
# Old articles that no new article matches are left completely untouched.
# Set to False to only precompute the uploaded batch (old lists stay frozen).
INCREMENTAL_RECOMMENDATIONS = True

# bge models work better when a SHORT search query gets this prefix (articles do not need it)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SHORT_QUERY_WORDS = 12                        # fewer words than this = "short query"

# --- Token Budget Settings (Replaces retired word counts) ---
TEXT_RECIPE_VERSION = "v2_sentence_token_budget"
EMBED_SPECIAL_TOKENS = 2           # [CLS] and [SEP]
EMBED_TOKEN_BUDGET = EMBED_MAX_SEQ_LEN - EMBED_SPECIAL_TOKENS  # 254 content tokens

RERANK_PAIR_SPECIAL_TOKENS = 4    # <s> query </s></s> passage </s>
# In article-to-article mode, split 50/50: 256 per side minus 2 special tokens = 254 content tokens
RERANK_TOKEN_BUDGET = (RERANK_MAX_LEN // 2) - 2

# Word-based settings retired in favor of token-based budgeting
BODY_WORDS_EMBED = None
BODY_WORDS_RERANK = None

BOILERPLATE_MIN_DOCS = 4    # a sentence repeated in >= N articles is treated as boilerplate

# --- Retrieval pipeline ---
N_RETRIEVE = 50             # candidates taken from each search (BM25, semantic)
N_CANDIDATES = 20           # candidates kept after fusion, sent to the reranker
RRF_K = 60                  # standard Reciprocal Rank Fusion constant
TOP_K = 10                  # default number of recommendations

# --- Diversity (MMR): 1.0 = pure relevance, lower = more variety ---
# OLD (hand-picked): MMR_LAMBDA = 0.75
MMR_LAMBDA = 0.91           # MEASURED via run_experiments.py random search (config 82e2663f)
MMR_THETA = 0.76            # MEASURED: near-dup similarity threshold for MMR penalty

# --- Personalization Settings ---
MAX_USER_HISTORY = 10       # track last N read articles
USER_PROFILE_WEIGHT = 0.35  # blend weight for user history profile vector

# --- News Metadata Boosts (MEASURED Multipliers) ---
# OLD (hand-picked): SECTION_BOOST_MULT = 1.08
# OLD (hand-picked): SUBSECTION_BOOST_MULT = 1.05
# OLD (hand-picked): POPULARITY_BOOST_MULT = 1.06
SECTION_BOOST_MULT = 1.072    # MEASURED: 7.2% boost for same section
SUBSECTION_BOOST_MULT = 1.09  # MEASURED: 9% boost for same subsection
POPULARITY_BOOST_MULT = 1.044 # MEASURED: up to 4.4% boost for popular articles (log-scaled)

# --- Dynamic Recency Half-lives (in days) ---
SECTION_HALF_LIVES = {
    "live_blog": 0.25,      # 6 hours
    "sports": 1.0,          # 1 day
    "entertainment": 2.0,   # 2 days
    "business": 3.0,        # 3 days
    "brand-story": 4.0,     # 4 days
    "default": 2.0
}
# OLD (hand-picked): RECENCY_MAX_BOOST_MULT = 1.10
RECENCY_MAX_BOOST_MULT = 1.104 # MEASURED: up to 10.4% boost for fresh articles
BOOST_MODE = "source_date_recency"  # MEASURED: recency relative to source article date, not corpus max

# --- "No good match" detection & Non-Relatable Story Floor ---
# Query-level gate: uses raw cross-encoder probability, not max-normalized blend.
MIN_RAW_RERANK_THRESHOLD = 0.12   # If top candidate raw rerank probability < 0.12, flag low confidence
ARTICLE_DISCARD_FLOOR = 0.05      # Filter out completely unrelated candidates from results

MIN_RELEVANCE = {
    "reranker": MIN_RAW_RERANK_THRESHOLD,   # raw reranker probability threshold (0-1)
    "dense": 0.50,                          # cosine similarity of bge embeddings
    "tfidf": 0.08,                          # cosine similarity of TF-IDF vectors (light mode)
}


def get_pipeline_version() -> str:
    """
    Compute a 16-character fingerprint hash of all models, parameters,
    and text recipes governing recommendation scores.
    Any change here auto-triggers full recompute on next ingestion.
    """
    import hashlib
    tokens = [
        f"embed_model={EMBEDDING_MODEL}",
        f"embed_rev={EMBEDDING_REVISION}",
        f"rerank_model={RERANKER_MODEL}",
        f"rerank_rev={RERANKER_REVISION}",
        f"embed_len={EMBED_MAX_SEQ_LEN}",
        f"rerank_len={RERANK_MAX_LEN}",
        f"rerank_per_side={RERANK_PER_ARTICLE_TOKENS}",
        f"mmr_lambda={MMR_LAMBDA}",
        f"mmr_theta={MMR_THETA}",
        f"recipe_ver={TEXT_RECIPE_VERSION}",
        # Boost multipliers (changing any of these invalidates stored scores)
        f"sec_boost={SECTION_BOOST_MULT}",
        f"subsec_boost={SUBSECTION_BOOST_MULT}",
        f"pop_boost={POPULARITY_BOOST_MULT}",
        f"rec_boost={RECENCY_MAX_BOOST_MULT}",
        f"boost_mode={BOOST_MODE}",
    ]
    return hashlib.sha256("|".join(tokens).encode("utf-8")).hexdigest()[:16]


