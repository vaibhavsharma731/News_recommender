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
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_REVISION = os.getenv("NEWSREC_EMBED_REVISION", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
EMBEDDING_DIM = 384           # Native vector dimensions for BGE-small
RERANKER_MODEL = os.getenv("NEWSREC_RERANKER", "BAAI/bge-reranker-base")
RERANKER_REVISION = os.getenv("NEWSREC_RERANK_REVISION", "2cfc18c9415c912f9d8155881c133215df768a70")
EMBED_MAX_SEQ_LEN = 256   # same for ingestion AND live search, so vectors are consistent
RERANK_MAX_LEN = 512        # total cross-encoder token limit (query + candidate combined)
RERANK_PER_ARTICLE_TOKENS = 256  # deliberate 50/50 split for article-to-article mode

# Optional: Remote Cloud GPU Worker (e.g. Google Colab + Cloudflare Tunnel).
# When set to a valid URL (e.g. https://xxx.trycloudflare.com), embeddings and rerankings
# run remotely on free cloud GPU instead of local CPU. If empty, runs locally.
REMOTE_MODEL_URL = os.getenv("REMOTE_MODEL_URL", "").strip().rstrip("/")
# Optional shared secret for that worker, matching its WORKER_TOKEN Colab secret.
# Only /health is left open; without this token /embed and /rerank return 401.
REMOTE_MODEL_TOKEN = os.getenv("REMOTE_MODEL_TOKEN", "").strip()

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

# --- Quota-based candidate selection (FIX #13) ---------------------------------
# These values were previously hardcoded at two call sites (recommender.py and
# pipeline.py) and would drift apart silently — the same pattern that caused the
# double-sigmoid bug. They are now a single source of truth here.
# They are included in get_pipeline_version() so tuning any of them automatically
# invalidates precomputed recommendations.
CANDIDATE_RRF_QUOTA        = 12   # top RRF-fused slots (always filled first)
CANDIDATE_DENSE_QUOTA      = 5    # strong Dense-only slots (rank 1..STRONG_DENSE_CUTOFF)
CANDIDATE_BM25_QUOTA       = 3    # strong BM25-only slots  (rank 1..STRONG_BM25_CUTOFF)
CANDIDATE_STRONG_DENSE_CUTOFF = 5 # a "strong" dense hit must be in top-N dense ranks
CANDIDATE_STRONG_BM25_CUTOFF  = 5 # a "strong" BM25 hit must be in top-N BM25 ranks

# --- Relevance blend weights (must sum to 1.0) ---
# Single source of truth for the reranker/BM25/dense mix used by BOTH the
# ingestion pipeline (db/pipeline.py) and the live path (src/recommender.py).
# OLD: 0.70/0.20/0.10 — dense was underweighted (ranked ~40th of 200 in tuning)
# NEW (Change 16, 2026-10-27): 9/15 top configs from 200-experiment random search
#   had w_dense >= 0.20; current 0.10 was the primary gap. BM25 trimmed slightly
#   to reduce keyword bias on metaphoric headlines (e.g. Kapil Sharma jail quote).
WEIGHT_RERANK = 0.55        # cross-encoder (bge-reranker-base)
WEIGHT_BM25 = 0.15          # lexical keyword match
WEIGHT_DENSE = 0.30         # semantic embedding (bge-small-en-v1.5 384d)

# Live hybrid pipeline weights (pure in-memory retrieval without cross-encoder).
# Kept independent of precomputation weights so they do not invalidate pipeline_version.
LIVE_WEIGHT_DENSE = 0.60    # semantic dense vector percentile (60%)
LIVE_WEIGHT_BM25 = 0.40     # lexical BM25 self-normalized score (40%)

# Live-only recency injection: when method="hybrid", articles published within this
# many days of the source article (and in the same section) are added to the candidate
# pool even if BM25 and Dense did not retrieve them. This guarantees recency boost
# can actually surface them. Never used by the precomputed pipeline.
RECENCY_INJECT_DAYS = 7

# --- Diversity (MMR): 1.0 = pure relevance, lower = more variety ---
# Article-to-article: the MEASURED values, shared by BOTH the ingestion pipeline
# (db/pipeline.py) and the live path (src/recommender.py).
# OLD (hand-picked): MMR_LAMBDA = 0.75
MMR_LAMBDA = 0.91           # MEASURED via run_experiments.py random search (config 82e2663f)
MMR_THETA = 0.76            # MEASURED: near-dup similarity threshold for MMR penalty

# Free-text search mode. Search queries return topically looser results than
# article-to-article, so it prefers relevance (higher lambda) and only penalises
# very tight near-duplicates (higher theta). Previously unnamed literals in _mmr().
MMR_LAMBDA_SEARCH = 0.88
MMR_THETA_SEARCH = 0.85

# --- Personalization Settings ---
MAX_USER_HISTORY = 10       # track last N read articles
USER_PROFILE_WEIGHT = 0.35  # blend weight for user history profile vector

# --- News Metadata Boosts (Asymmetric Strategy E) ---
SECTION_BOOST_MULT = 1.100        # Legacy default / focused fallback
SUBSECTION_BOOST_MULT = 1.050     # Legacy default / focused fallback
SECTION_BOOST_FOCUSED = 1.100     # 10% boost for cohesive editorial beats
SECTION_BOOST_UMBRELLA = 1.020    # 2% dampened boost for catch-all / heterogeneous sections
SUBSECTION_BOOST_FOCUSED = 1.050  # 5% boost for matching subsection in focused beats
SUBSECTION_BOOST_UMBRELLA = 1.020 # 2% dampened boost for matching subsection in umbrella beats

# Section Classifications for Asymmetric Boosting (Strategy E — full 56-section audit 2026-11-07)
# Every distinct section in the PostgreSQL articles table is listed in exactly one set.
# Unknown future sections fall back to UMBRELLA (1.020) multipliers — the safer conservative default.
UMBRELLA_SECTIONS = {
    # Primary geographic catch-alls (heterogeneous topic mix — section match is weak signal)
    "cities", "india", "world", "miscellaneous",
    # Standalone city sections (same logic as cities subsections)
    "delhi", "mumbai", "chandigarh", "pune",
    # National news alias / synonym
    "india-news",
    # Generic multi-beat news formats and daily digests
    "news-today", "news-archive", "news-video", "3-things",
    # Multi-category video and multimedia feeds
    "video-detail",
    # Sponsored / multi-topic commercial content
    "brand-story", "in-collaboration",
}
FOCUSED_SECTIONS = {
    # Sports
    "sports", "sports-gallery",
    # Entertainment & Cinema
    "entertainment", "bollywood", "movies", "entertainment-gallery", "casually-obsessed",
    # Lifestyle, Health, Food & Parenting
    "lifestyle", "lifestyle-gallery", "health-wellness", "health-and-wellness",
    "food", "parenting", "love-matters", "love-matters-with-leeza-mangaldas",
    # Legal & Justice
    "legal-news", "legal",
    # Education, Exams & Knowledge
    "explained", "education", "upsc-current-affairs", "research",
    # Politics & Governance
    "political-pulse", "delhi-confidential",
    # Business & Economy
    "business",
    # Technology & Science
    "technology",
    # Opinion, Analysis & Dialogue
    "opinion", "fresh-take", "idea-exchange", "express-exclusive",
    # Culture, Features & Long-form
    "books-and-literature", "travel", "long-reads", "eye", "express-sunday-eye",
    "express-investigations", "people", "horoscope", "puzzles-and-games",
    "e-p-unny-cartoons-gallery",
    # Trending & Viral (deliberate editorial vertical — kept as Focused)
    "trending",
}
POPULARITY_BOOST_MULT = 1.015 # Log-scaled virality nudge
RECENCY_MAX_BOOST_MULT = 1.040 # Max recency boost (source-relative decay)

# --- Dynamic Recency Half-lives (in days) ---
# Tailored to Indian Express editorial categories and news cycle velocity:
# - Breaking / Live: 6 hours (0.25d)
# - Trending / Social Buzz: 1 day (1.0d)
# - Sports: 1.5 days (36h)
# - City / Regional / National hard news: 2.5 days (60h)
# - World / Entertainment: 3.0 days (72h)
# - Business / Technology / Political Analysis: 4.0 days
# - Legal & Courts: 5.0 days
# - In-depth / Flagship (Explained, Opinion, UPSC, Education, Lifestyle): 7.0 days
# - Books & Literary: 10.0 days
SECTION_HALF_LIVES = {
    # Real-time / Live Coverage
    "live_blog": 0.25,                  # 6 hours
    "trending": 1.0,                    # 24 hours (social buzz, viral)
    "news-today": 1.0,
    "puzzles-and-games": 1.0,

    # Fast-moving Sports
    "sports": 1.5,                      # 36 hours (match results & post-game)

    # Daily Hard News & Regional Coverage
    "cities": 2.5,                      # 60 hours (civic, local crime, municipal policies)
    "delhi": 2.5,
    "mumbai": 2.5,
    "chandigarh": 2.5,
    "pune": 2.5,
    "india": 2.5,                       # National developments & governance

    # Broad Interest & Beats
    "world": 3.0,                       # Geopolitics, global summits & conflicts
    "entertainment": 3.0,               # Reviews, weekend box-office, cinema
    "horoscope": 2.0,                   # Daily/weekly astrology

    # Analytical, Economic & Specialized Beats
    "business": 4.0,                    # Earnings, RBI policy, macroeconomics, markets
    "technology": 4.0,                  # Tech launches, AI trends, gadgets
    "political-pulse": 4.0,             # Political insider reporting & party analysis
    "delhi-confidential": 4.0,

    # Legal & Judicial (high reference duration across hearings)
    "legal-news": 5.0,                  # Supreme Court & HC verdicts, bail orders, PILs
    "legal": 5.0,
    "brand-story": 5.0,                 # Sponsored editorial campaigns
    "in-collaboration": 5.0,

    # The Indian Express Flagships & In-Depth / Evergreen
    "explained": 7.0,                   # IE Flagship: Contextual backgrounders, evergreen explainers
    "opinion": 7.0,                     # Editorial columns, intellectual debates, policy essays
    "editorials": 7.0,
    "columns": 7.0,
    "idea-exchange": 7.0,
    "express-exclusive": 7.0,           # Investigative journalism & exclusive scoops
    "long-reads": 7.0,
    "eye": 7.0,
    "express-sunday-eye": 7.0,
    "education": 7.0,                   # Exams, results, admissions guides
    "upsc-current-affairs": 7.0,        # UPSC study material & current affairs analysis
    "upsc-essentials": 7.0,
    "lifestyle": 7.0,                   # Health, wellness, relationships, food
    "health-wellness": 7.0,
    "health-and-wellness": 7.0,
    "fitness": 7.0,

    # Long-form / Evergreen Literary
    "books-and-literature": 10.0,       # Book reviews, author discussions

    # General fallback for unclassified articles
    "default": 3.0                      # 3 days for standard reporting
}

BOOST_MODE = "source_date_recency"  # MEASURED: recency relative to source article date, not corpus max


def _has_date(value) -> bool:
    """True if ``value`` is a usable timestamp (None / NaT / NaN are not)."""
    if value is None:
        return False
    try:
        return not value != value   # NaT and NaN are the only values unequal to themselves
    except Exception:
        return False


def section_half_life(section: str) -> float:
    """Section decay in days; tolerant of hyphen / underscore / case variants."""
    sec = str(section or "").lower().strip()
    return (
        SECTION_HALF_LIVES.get(sec)
        or SECTION_HALF_LIVES.get(sec.replace("_", "-"))
        or SECTION_HALF_LIVES.get(sec.replace("-", "_"))
        or SECTION_HALF_LIVES.get("default", 3.0)
    )


def recency_boost(cand_date, ref_date, section: str, is_live: bool = False) -> float:
    """
    Recency multiplier for one candidate, relative to ``ref_date``.

    ``ref_date`` is chosen by the caller from ``BOOST_MODE``:
      - "source_date_recency" (the MEASURED winner): the SOURCE article's date, so
        a candidate is rewarded for being fresh *relative to what it is recommended
        next to*.
      - "multiplicative" (rejected): the corpus max date, so every candidate is
        scored on absolute freshness.

    Shared by db/pipeline.py and src/recommender.py so the ingested and live paths
    cannot drift apart.
    """
    if not _has_date(cand_date) or not _has_date(ref_date):
        return 1.0
    days_old = max(0.0, (ref_date - cand_date).total_seconds() / 86400.0)
    half_life = section_half_life("live_blog" if is_live else section)
    return 1.0 + (RECENCY_MAX_BOOST_MULT - 1.0) * (0.5 ** (days_old / half_life))


def recency_ref_date(source_date, corpus_max_date):
    """
    Pick the recency reference date according to BOOST_MODE.

    "source_date_recency" (MEASURED winner) -> the source article's date.
    "multiplicative" (rejected variant)     -> the corpus max date.
    Falls back to the corpus max date when the source has no usable date.
    """
    if BOOST_MODE == "source_date_recency" and _has_date(source_date):
        return source_date
    return corpus_max_date

# --- "No good match" detection & Non-Relatable Story Floor ---
# Query-level gate: uses raw cross-encoder probability, not max-normalized blend.
MIN_RAW_RERANK_THRESHOLD = 0.12   # If top candidate raw rerank probability < 0.12, flag low confidence

MIN_RELEVANCE = {
    "reranker": MIN_RAW_RERANK_THRESHOLD,   # raw reranker probability threshold (0-1)
    "dense": 0.50,                          # cosine similarity of bge embeddings
    "tfidf": 0.08,                          # cosine similarity of TF-IDF vectors (light mode)
}

# --- 4. Graceful degradation ------------------------------------------------
# When an isolated article has FEWER than FALLBACK_MIN_RELATED related stories, the
# API attaches a SEPARATE, clearly labelled "Trending in <section>" list.
FALLBACK_MIN_RELATED = 3    # related items below this -> offer the fallback list
FALLBACK_LIMIT = 5          # cap on fallback items


def fallback_window_days(section: str) -> float:
    """
    Recency window around the SOURCE date for the fallback list.

    Three section half-lives, floored at one week: "trending alongside this story",
    not "newest in the database". Reuses the same decay table as the recency boost,
    so sport does not get a two-week window and books do not get a two-day one.
    """
    return max(7.0, 3.0 * section_half_life(section))


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
        f"embed_dim={EMBEDDING_DIM}",
        f"rerank_model={RERANKER_MODEL}",
        f"rerank_rev={RERANKER_REVISION}",
        f"embed_len={EMBED_MAX_SEQ_LEN}",
        f"rerank_len={RERANK_MAX_LEN}",
        f"rerank_per_side={RERANK_PER_ARTICLE_TOKENS}",
        f"mmr_lambda={MMR_LAMBDA}",
        f"mmr_theta={MMR_THETA}",
        f"recipe_ver={TEXT_RECIPE_VERSION}",
        # Relevance blend (changing these invalidates stored scores)
        f"w_rerank={WEIGHT_RERANK}",
        f"w_bm25={WEIGHT_BM25}",
        f"w_dense={WEIGHT_DENSE}",
        # Boost multipliers (changing any of these invalidates stored scores)
        f"sec_boost={SECTION_BOOST_MULT}",
        f"subsec_boost={SUBSECTION_BOOST_MULT}",
        f"sec_boost_focused={SECTION_BOOST_FOCUSED}",
        f"sec_boost_umbrella={SECTION_BOOST_UMBRELLA}",
        f"subsec_boost_focused={SUBSECTION_BOOST_FOCUSED}",
        f"subsec_boost_umbrella={SUBSECTION_BOOST_UMBRELLA}",
        f"umbrella_sections_hash={hashlib.sha256(','.join(sorted(UMBRELLA_SECTIONS)).encode('utf-8')).hexdigest()[:8]}",
        f"focused_sections_hash={hashlib.sha256(','.join(sorted(FOCUSED_SECTIONS)).encode('utf-8')).hexdigest()[:8]}",
        f"pop_boost={POPULARITY_BOOST_MULT}",
        f"rec_boost={RECENCY_MAX_BOOST_MULT}",
        f"boost_mode={BOOST_MODE}",
        # The cross-encoder output transform itself. This token exists because a
        # bug where score() applied sigmoid() to an already-probability model made
        # every stored score meaningless - the fingerprint must move when the
        # transform moves, or stale rows would look valid.
        f"rerank_output=probability",
        # Blend scale normalization (v2 fixed scale: self-score BM25 + cosine percentile)
        f"blend_scale=v2_fixed_scale",
        # Signal consensus agreement boost/demote — include the actual thresholds
        # so tuning any of them correctly invalidates precomputed rows.
        f"signal_agreement=consensus_v5_clean_monotonic",
        f"consensus_strong_rank=5",
        f"consensus_mod_rank=15",
        f"consensus_ce_strong=0.55",
        f"consensus_ce_mod=0.35",
        f"consensus_fluke_kw=0.40",
        f"consensus_fluke_drift=0.35",
        f"consensus_mismatch_damping=0.75",
        f"consensus_tier1=1.15",
        f"consensus_tier2=1.10",
        f"consensus_tier3=1.08",
        f"consensus_dual_retrieval=1.04",
        f"consensus_single_signal=0.80",
        # FIX #13: candidate quota constants
        f"cand_rrf_quota={CANDIDATE_RRF_QUOTA}",
        f"cand_dense_quota={CANDIDATE_DENSE_QUOTA}",
        f"cand_bm25_quota={CANDIDATE_BM25_QUOTA}",
        f"cand_strong_dense_cutoff={CANDIDATE_STRONG_DENSE_CUTOFF}",
        f"cand_strong_bm25_cutoff={CANDIDATE_STRONG_BM25_CUTOFF}",
    ]
    return hashlib.sha256("|".join(tokens).encode("utf-8")).hexdigest()[:16]


