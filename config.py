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

# --- Relevance blend weights (must sum to 1.0) ---
# Single source of truth for the reranker/BM25/dense mix used by BOTH the
# ingestion pipeline (db/pipeline.py) and the live path (src/recommender.py).
# OLD: article-to-article 0.55/0.30/0.15, text search 0.65/0.35
WEIGHT_RERANK = 0.70        # cross-encoder (bge-reranker-base)
WEIGHT_BM25 = 0.20          # lexical keyword match
WEIGHT_DENSE = 0.10         # semantic embedding (bge-base-en-v1.5)

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

# --- News Metadata Boosts (MEASURED Multipliers) ---
# OLD (hand-picked): SECTION_BOOST_MULT = 1.08
# OLD (hand-picked): SUBSECTION_BOOST_MULT = 1.05
# OLD (hand-picked): POPULARITY_BOOST_MULT = 1.06
SECTION_BOOST_MULT = 1.072    # MEASURED: 7.2% boost for same section
SUBSECTION_BOOST_MULT = 1.09  # MEASURED: 9% boost for same subsection
POPULARITY_BOOST_MULT = 1.044 # MEASURED: up to 4.4% boost for popular articles (log-scaled)

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
# OLD (hand-picked): RECENCY_MAX_BOOST_MULT = 1.10
RECENCY_MAX_BOOST_MULT = 1.104 # MEASURED: up to 10.4% boost for fresh articles
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

# --- Relevance Admission (the 4 limiter techniques) --------------------------
# A fixed list of TOP_K is the reason non-relatable stories appear: when a source
# article has only 2 true matches, ranks 3-10 are filled with whatever noise the
# reranker had left. These three gates cut the list down to genuinely related
# articles and let the UI degrade gracefully instead of padding.
#
# All thresholds are compared against the RAW cross-encoder probability, which is
# now a true sigmoid output in [0,1] (see src/reranker.py Reranker._to_probability).
# Before that fix every score was trapped in [0.5, 0.731], so floors were inert.
#
# 1. Hard Admission Floor - absolute minimum probability to be admitted.
#    0.40 keeps clearly-related pairs and drops the 0.5-plateau noise.
RELEVANCE_FLOOR = 0.40
# 2. Elbow / cliff-drop - cut when a rank's probability collapses versus the one
#    above it. 0.45 means "kept score is less than 45% of its predecessor".
ELBOW_RATIO = 0.45
#    Below this many survivors the list is too thin to trust the elbow cut, so
#    only the hard floor applies (never let the cliff detector empty a good list).
ELBOW_MIN_KEEP = 4
# 3. Lexical / entity support gate - applied BEFORE the reranker. A candidate
#    with no shared BM25 term AND no shared entity is blocked outright, no matter
#    how close it sits in embedding space.
LEXICAL_GATE = True
ENTITY_GATE = True

MIN_RELEVANCE = {
    "reranker": MIN_RAW_RERANK_THRESHOLD,   # raw reranker probability threshold (0-1)
    "dense": 0.50,                          # cosine similarity of bge embeddings
    "tfidf": 0.08,                          # cosine similarity of TF-IDF vectors (light mode)
}

# --- 4. Graceful degradation ------------------------------------------------
# When the gates above leave FEWER than FALLBACK_MIN_RELATED related stories, the
# API attaches a SEPARATE, clearly labelled "Trending in <section>" list. Those
# items are popularity, not relevance: they carry no score, no confidence and no
# evidence, are never mixed into the related list, and never enter evaluation.
# They exist so a thin list degrades to something honest instead of being padded.
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


def admission_cut(prob_scores, entities=None, bypass=None) -> dict:
    """
    Apply techniques 1 and 2 to a ranked list and report what happened.

    ``prob_scores`` must be in final rank order (best first) holding the raw
    cross-encoder probability. ``entities`` optionally flags, per position,
    whether the candidate shares an entity with the source (technique 3's
    entity half). ``bypass`` marks positions that are never cut - used for
    editorial Also-Read links, which are ground truth rather than inference.

    Returns ``{"keep": [positions], "n_kept": n, "cut": "floor"|"entity"|"elbow"|None,
    "cut_at": i, "floor": RELEVANCE_FLOOR, "elbow_ratio": ELBOW_RATIO}``. ``keep`` is an
    explicit position list, not a prefix length, because protected entries after the
    cut point are rescued rather than silently discarded -- editorial Also-Read links
    are ground truth and the evaluation would be wrong without them.

    This is the single implementation used by BOTH db/pipeline.py and
    src/recommender.py, so ingested and live lists cannot drift.
    """
    n = len(prob_scores)
    result = {"keep": list(range(n)), "n_kept": n, "cut": None, "cut_at": None,
              "floor": RELEVANCE_FLOOR, "elbow_ratio": ELBOW_RATIO}
    if n == 0:
        return result

    protected = [bool(bypass[i]) if bypass else False for i in range(n)]

    def stop(i, why):
        keep = list(range(i)) + [j for j in range(i, n) if protected[j]] or [0]
        result.update(keep=keep, n_kept=len(keep), cut=why, cut_at=i)
        return result

    # Technique 1: hard relevance floor. Protected positions never trigger it.
    for i in range(n):
        if not protected[i] and prob_scores[i] < RELEVANCE_FLOOR:
            return stop(i, "floor")

    # Technique 3 (entity half): veto an entity-less candidate that follows a
    # related one. Never allowed to empty the list.
    if entities is not None and n > ELBOW_MIN_KEEP:
        for i in range(1, n):
            if protected[i]:
                continue
            if not entities[i] and entities[i - 1]:
                return stop(i, "entity")

    # Technique 2: cliff drop, only trusted on a list long enough to be stable.
    if n > ELBOW_MIN_KEEP:
        for i in range(1, n):
            prev, cur = prob_scores[i - 1], prob_scores[i]
            if not protected[i] and prev > 1e-9 and cur < prev * ELBOW_RATIO:
                return stop(i, "elbow")

    return result


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
        # Relevance blend (changing these invalidates stored scores)
        f"w_rerank={WEIGHT_RERANK}",
        f"w_bm25={WEIGHT_BM25}",
        f"w_dense={WEIGHT_DENSE}",
        # Boost multipliers (changing any of these invalidates stored scores)
        f"sec_boost={SECTION_BOOST_MULT}",
        f"subsec_boost={SUBSECTION_BOOST_MULT}",
        f"pop_boost={POPULARITY_BOOST_MULT}",
        f"rec_boost={RECENCY_MAX_BOOST_MULT}",
        f"boost_mode={BOOST_MODE}",
        # Relevance admission gates (changing any of these changes which
        # articles survive into the stored top-10)
        f"rel_floor={RELEVANCE_FLOOR}",
        f"elbow_ratio={ELBOW_RATIO}",
        f"elbow_min_keep={ELBOW_MIN_KEEP}",
        f"lexical_gate={LEXICAL_GATE}",
        f"entity_gate={ENTITY_GATE}",
        # The cross-encoder output transform itself. This token exists because a
        # bug where score() applied sigmoid() to an already-probability model made
        # every stored score meaningless - the fingerprint must move when the
        # transform moves, or stale rows would look valid.
        f"rerank_output=probability",
    ]
    return hashlib.sha256("|".join(tokens).encode("utf-8")).hexdigest()[:16]


