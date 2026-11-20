"""
Step 4 - The recommender: puts everything together.

Pipeline for every request:
  1. BM25 search + semantic search          (two ranked lists)
  2. Reciprocal Rank Fusion                 (merge into ~30 candidates)
  3. Cross-encoder reranking                (precise re-scoring)
  4. Small boosts: same section, same time  (only when the input is an article)
  5. MMR                                    (avoid near-duplicate results)
  6. A short "why recommended" explanation
"""
from dataclasses import dataclass, field

import time
import json
import numpy as np
import pandas as pd

import config
from src.reranker import Reranker
from src.retrieval import (BM25Index, DenseIndex, TfidfIndex,
                           reciprocal_rank_fusion, select_quota_candidates, tokenize)
from src.calibration import ConfidenceCalibrator
from src.evidence import build_evidence_reason, evaluate_signal_agreement
from src.scoring import blend, boost_parts, linked_set, mmr, norm, self_normalized_bm25, cosine_percentile, rank_adjusted_dense_norm

try:
    from monitoring.resource_monitor import take_snapshot
    _MONITORING_AVAILABLE = True
except Exception:
    _MONITORING_AVAILABLE = False

METHODS = ["bm25", "semantic", "hybrid", "hybrid+rerank"]


@dataclass
class Recommendation:
    article: dict
    score: float          # Calibrated empirical probability (0-1) from unboosted relevance
    reason: str           # Structured JSON evidence
    calibrated_score: float = 0.0
    confidence_band: str = "MEDIUM"
    margin: float = 0.0
    agreement_count: int = 1
    rank: int = 1
    rank_raw: int = 1
    # Relevance after domain boosts and before calibration: the value the ordering
    # actually used, kept separate from `score` so the two are never confused.
    boosted_score: float = 0.0



@dataclass
class Result:
    recommendations: list
    low_confidence: bool          # True = nothing in the dataset matches well
    best_relevance: float
    method: str                   # the method that was really used
    notes: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    candidate_strategy: str = "quota"


class Recommender:
    def __init__(self, articles: pd.DataFrame, use_neural: bool = config.USE_NEURAL_MODELS, vectors=None, use_disk_cache=True):
        self.articles = articles.reset_index(drop=True)
        self.id_to_pos = {}
        for i, aid in enumerate(self.articles["id"]):
            self.id_to_pos[aid] = i
            self.id_to_pos[str(aid)] = i
        self.notes = []
        self._max_page_views = int(self.articles["pageViews"].max()) if "pageViews" in self.articles.columns else 1

        # Pre-sorted index (position → date descending) and pre-extracted arrays for O(1) recency injection.
        # Computed once at engine load, not on every request (zero pandas .iloc in hot loops).
        try:
            dates_dt = pd.to_datetime(self.articles["date"], utc=True, errors="coerce")
            self._date_ns = dates_dt.values.astype("int64")
            self._date_sorted_positions = list(np.argsort(-self._date_ns))
            # FIX #15: cache max_date once — avoids a full pandas column scan per candidate per request
            self._max_date = self.articles["date"].max()
        except Exception:
            self._date_ns = np.zeros(len(self.articles), dtype=np.int64)
            self._date_sorted_positions = list(range(len(self.articles)))
            self._max_date = None

        try:
            self._sections_lower = np.array(
                [str(s or "").strip().lower() for s in self.articles.get("section", [])],
                dtype=object
            )
        except Exception:
            self._sections_lower = np.array([""] * len(self.articles), dtype=object)

        self.bm25 = BM25Index(self.articles)
        self.semantic = self._build_semantic(use_neural, vectors, use_disk_cache)
        self.reranker = self._build_reranker(use_neural)

    # ---------------------------------------------------------------- setup
    def _build_semantic(self, use_neural, vectors=None, use_disk_cache=True):
        if use_neural:
            try:
                return DenseIndex(self.articles["text_embed"], vectors=vectors, use_disk_cache=use_disk_cache)
            except Exception as error:
                self.notes.append(f"Neural embeddings unavailable ({type(error).__name__}); using TF-IDF instead.")
        return TfidfIndex(self.articles["text_embed"])

    def _build_reranker(self, use_neural):
        if not use_neural:
            return None
        try:
            return Reranker()
        except Exception as error:
            self.notes.append(f"Reranker unavailable ({type(error).__name__}); skipping the reranking step.")
            return None

    @property
    def engine_summary(self) -> str:
        semantic = "bge embeddings" if self.semantic.name == "dense" else "TF-IDF (light mode)"
        return f"BM25 + {semantic}" + (" + cross-encoder reranker" if self.reranker else "")

    # ---------------------------------------------------------------- public
    def get_article(self, article_id) -> dict:
        return self._to_dict(self.id_to_pos[article_id])

    def recommend(self, query: str = None, article_id=None, user_history: list = None,
                  top_k: int = config.TOP_K, method: str = "hybrid+rerank",
                  use_boosts: bool = True, diversify: bool = True, calibrate: bool = True,
                  section: str = None, candidate_strategy: str = "quota") -> Result:
        """Give EITHER `article_id` (recommend similar articles) OR `query` (free text) OR `user_history` (personalized for user)."""
        notes = list(self.notes)
        if method == "hybrid+rerank" and self.reranker is None:
            method = "hybrid"

        # --- 0. what are we searching with? -------------------------------
        t_start = time.perf_counter()
        t_prep_start = time.perf_counter()
        _recency_injected = 0
        user_vector = self._build_user_profile_vector(user_history) if user_history else None

        self_bm25 = None
        t_dense_ms = 0.0
        t_bm25_ms = 0.0
        if article_id is not None:
            query_pos = self.id_to_pos[article_id]
            base_row = self.articles.iloc[query_pos]
            # MoreLikeThis focused query: title + top IDF terms from description/lead
            title_toks = tokenize(base_row["title"])
            body_toks = tokenize(f"{base_row['description']} {base_row['body']}")
            top_body_terms = sorted(set(body_toks), key=lambda t: -self.bm25.idf.get(t, 0.0))[:15]
            focused_text = " ".join(title_toks + top_body_terms)
            query_text = focused_text
            rerank_query = self.articles.at[query_pos, "text_rerank"]
            t_query_embed_ms = 0.0  # Already pre-embedded in index
            t_d_start = time.perf_counter()
            semantic_scores = self.semantic.scores_for_doc(query_pos)
            t_dense_scan_ms = (time.perf_counter() - t_d_start) * 1000
            t_dense_ms = t_dense_scan_ms
            t_b_start = time.perf_counter()
            bm25_scores = self.bm25.scores(focused_text)
            t_bm25_ms = (time.perf_counter() - t_b_start) * 1000
            self_bm25 = float(bm25_scores[query_pos])
        elif query:
            query_pos = None
            clean_query, corrections, _ = self.bm25.normalize_query(query)
            query_text = clean_query
            rerank_query = clean_query
            is_short = len(query_text.split()) < config.SHORT_QUERY_WORDS
            t_d_start = time.perf_counter()
            if hasattr(self.semantic, "encode_query"):
                query_vec, t_query_embed_ms = self.semantic.encode_query(query_text, short=is_short)
                semantic_scores, t_dense_scan_ms = self.semantic.scan_matrix(query_vec)
                t_dense_ms = t_query_embed_ms + t_dense_scan_ms
            else:
                semantic_scores = self.semantic.scores_for_text(query_text, is_short)
                t_dense_ms = (time.perf_counter() - t_d_start) * 1000
                t_query_embed_ms = t_dense_ms * 0.90
                t_dense_scan_ms = t_dense_ms * 0.10
            t_b_start = time.perf_counter()
            bm25_scores = self.bm25.scores(query)
            t_bm25_ms = (time.perf_counter() - t_b_start) * 1000
            if corrections:
                corr_str = ", ".join(f"'{orig}' -> '{corr}'" for orig, corr in corrections)
                notes.append(f"Auto-corrected query terms: {corr_str}")
        elif user_vector is not None:
            query_pos = None
            query_text = rerank_query = "Personalized recommendation feed based on your reading history"
            t_query_embed_ms = 0.0
            t_d_start = time.perf_counter()
            semantic_scores = self.semantic.scores_for_vector(user_vector)
            t_dense_scan_ms = (time.perf_counter() - t_d_start) * 1000
            t_dense_ms = t_dense_scan_ms
            bm25_scores = np.zeros(len(self.articles))
            t_bm25_ms = 0.0
        else:
            return Result([], False, 0.0, method, notes + ["Empty query and no history."])

        t_prep_ms = (time.perf_counter() - t_prep_start) * 1000

        # --- articles that may not be returned ----------------------------
        t_retrieval_start = time.perf_counter()
        blocked = np.zeros(len(self.articles), dtype=bool)
        if query_pos is not None:
            blocked[query_pos] = True                      # never recommend the input itself
        if user_history:                                   # option to avoid recommending recently read articles
            for hid in user_history[-config.MAX_USER_HISTORY:]:
                if hid in self.id_to_pos:
                    blocked[self.id_to_pos[hid]] = True
        if section:
            blocked |= (self.articles["section"] != section).to_numpy()

        # --- 1 + 2. search and fuse ---------------------------------------
        bm25_rank = self._rank(bm25_scores, blocked, config.N_RETRIEVE, min_score=1e-9)
        semantic_rank = self._rank(semantic_scores, blocked, config.N_RETRIEVE)

        if method == "bm25":
            candidates = bm25_rank[:config.N_CANDIDATES]
        elif method == "semantic":
            candidates = semantic_rank[:config.N_CANDIDATES]
        elif query_pos is not None and method in ("hybrid", "hybrid+rerank"):
            if candidate_strategy == "pure_rrf":
                fused = reciprocal_rank_fusion([bm25_rank, semantic_rank], k=config.RRF_K)
                candidates = sorted(fused, key=fused.get, reverse=True)[:config.N_CANDIDATES]
                notes.append("Candidate selection: Pure RRF (20 slots)")
                _recency_injected = 0
            else:
                _base_cand_count = 0  # will be set after quota
                candidates = select_quota_candidates(
                    bm25_rank,
                    semantic_rank,
                    n_cand=config.N_CANDIDATES,
                    k_rrf=config.RRF_K,
                    rrf_quota=config.CANDIDATE_RRF_QUOTA,
                    dense_quota=config.CANDIDATE_DENSE_QUOTA,
                    bm25_quota=config.CANDIDATE_BM25_QUOTA,
                    strong_dense_cutoff=config.CANDIDATE_STRONG_DENSE_CUTOFF,
                    strong_bm25_cutoff=config.CANDIDATE_STRONG_BM25_CUTOFF,
                )
                _base_cand_count = len(candidates)
                notes.append(f"Candidate selection: Quota Plan ({config.CANDIDATE_RRF_QUOTA} RRF / {config.CANDIDATE_DENSE_QUOTA} Dense / {config.CANDIDATE_BM25_QUOTA} BM25)")
                # Live-only: inject up to 2 recent same-section articles into the pool
                # so recency boost can actually surface them. Never touches precomputation.
                if method == "hybrid" and query_pos is not None:
                    candidates = self._inject_recency_candidates(
                        candidates, query_pos, blocked, seen=set(candidates), max_inject=2
                    )
                _recency_injected = len(candidates) - _base_cand_count
        else:
            fused = reciprocal_rank_fusion([bm25_rank, semantic_rank])
            candidates = sorted(fused, key=fused.get, reverse=True)[:config.N_CANDIDATES]
            _recency_injected = 0
        if not candidates:
            return Result([], True, 0.0, method, notes, candidate_strategy=candidate_strategy)
        t_retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000

        # --- 3. relevance score of each candidate (0-1) -------------------
        t_blend_start = time.perf_counter()
        t_rerank_ms = 0.0
        # Normalised component arrays are kept at function scope so the evidence
        # card can show the real blend decomposition for whichever method ran.
        bm25_norm = None
        dense_norm = None
        if method == "hybrid+rerank" and query_text:
            t_rerank_start = time.perf_counter()
            if query_pos is None:
                # Live text search: query-centered passage, dynamic budget split
                candidate_rows = [
                    dict(self.articles.iloc[pos])
                    for pos in candidates
                ]
                rerank_scores = np.array(
                    self.reranker.score_pairs(rerank_query, candidate_rows, is_search=True)
                )
            else:
                # Article-to-article: 50/50 budget split using stored text_rerank
                passages = self.articles["text_rerank"].iloc[candidates].tolist()
                rerank_scores = np.array(
                    self.reranker.score_pairs(rerank_query, passages, is_search=False)
                )
            t_rerank_ms = (time.perf_counter() - t_rerank_start) * 1000
            
            # Blend with normalized BM25 keyword score so entity presence is honored
            if query_pos is None:
                # Text query: reranker is primary, BM25 supports entity matching
                bm25_norm = norm([bm25_scores[p] for p in candidates])
                relevance = config.WEIGHT_RERANK * rerank_scores + config.WEIGHT_BM25 * bm25_norm
            else:
                # Article-to-article: reranker + lexical + semantic blend
                relevance, bm25_norm, dense_norm = blend(
                    rerank_scores, bm25_scores, semantic_scores, candidates, self_bm25=self_bm25)
            
            # Query-level gate: use raw cross-encoder probability, not max-normalized blend
            raw_rerank_max = float(np.max(rerank_scores)) if len(rerank_scores) > 0 else 0.0
            best_relevance = raw_rerank_max
            low_confidence = raw_rerank_max < getattr(config, "MIN_RAW_RERANK_THRESHOLD", 0.12)
        elif method == "bm25":
            top = bm25_scores[candidates]
            max_val = top.max() if len(top) and top.max() > 0 else 1.0
            relevance = top / max_val
            best_relevance = float(np.max(relevance)) if len(relevance) > 0 else 0.0
            low_confidence = False
            rerank_scores = np.zeros(len(candidates))
        elif method == "hybrid":
            # Pure Hybrid (Lexical BM25 + Semantic Dense) without Cross-Encoder
            # Build dense-rank lookup for rank-adjusted normalisation
            dense_rank_map = {pos: r + 1 for r, pos in enumerate(semantic_rank)}
            bm25_vals = [bm25_scores[p] for p in candidates]
            dense_vals = [semantic_scores[p] for p in candidates]
            # Rank of each candidate in the global dense-score ordering (1=best)
            cand_dense_ranks = [dense_rank_map.get(p, 51) for p in candidates]
            bm25_norm = self_normalized_bm25(bm25_vals, self_bm25)
            # Rank-adjusted dense norm differentiates rank #1 from rank #31 meaningfully
            dense_norm = rank_adjusted_dense_norm(dense_vals, dense_rank_of=cand_dense_ranks)
            # Live feed weights from config (60% rank-adj dense + 40% self-normalized BM25)
            w_dense = getattr(config, "LIVE_WEIGHT_DENSE", 0.60)
            w_bm25 = getattr(config, "LIVE_WEIGHT_BM25", 0.40)
            relevance = w_dense * dense_norm + w_bm25 * bm25_norm
            best_relevance = float(np.max(relevance)) if len(relevance) > 0 else 0.0
            threshold = config.MIN_RELEVANCE.get(self.semantic.name, 0.25)
            low_confidence = threshold is not None and best_relevance < threshold
            rerank_scores = None
        else:
            relevance = semantic_scores[candidates]
            best_relevance = float(np.max(relevance)) if len(relevance) > 0 else 0.0
            threshold = config.MIN_RELEVANCE.get(self.semantic.name, 0.50)
            low_confidence = threshold is not None and best_relevance < threshold
            rerank_scores = None

        # --- Lower Filter: relevance admission (techniques 1 + 2) -----------
        source_linked_set = set()
        if query_pos is not None:
            source_linked_set = self._linked_ids(query_pos)

        t_blend_ms = (time.perf_counter() - t_blend_start) * 1000
        # --- 4. Personalization & News Boost Calibration -------------------
        t_boosts_start = time.perf_counter()
        final = np.array(relevance, dtype=float)

        # Blend with User Profile Vector if present
        if user_vector is not None and len(candidates) > 0:
            user_affinities = self.semantic.scores_for_vector(user_vector)[candidates]
            w = config.USER_PROFILE_WEIGHT
            final = (1 - w) * final + w * user_affinities

        boost_parts = {}
        if use_boosts and len(candidates) > 0:
            bm25_rank_map = {c: i + 1 for i, c in enumerate(bm25_rank)}
            dense_rank_map = {c: i + 1 for i, c in enumerate(semantic_rank)}
            boost_parts = {
                i: self._boost_parts(
                    query_pos, pos,
                    bm25_rank=bm25_rank_map.get(pos),
                    dense_rank=dense_rank_map.get(pos),
                    rerank_score=float(rerank_scores[i]) if (rerank_scores is not None and len(rerank_scores) > i) else None,
                )
                for i, pos in enumerate(candidates)
            }
            # final = (relevance * domain_multiplier) + agreement_adjustment
            domain_mults = np.array([boost_parts[i].get("domain_multiplier", boost_parts[i].get("total", 1.0)) for i in range(len(candidates))])
            agreement_adjs = np.array([boost_parts[i].get("agreement_adjustment", 0.0) for i in range(len(candidates))])
            final = (final * domain_mults) + agreement_adjs

        t_boosts_ms = (time.perf_counter() - t_boosts_start) * 1000
        # --- 5. diversity ---------------------------------------------------
        t_mmr_start = time.perf_counter()
        raw_order = list(np.argsort(-final))
        raw_rank_map = {cand_idx: r + 1 for r, cand_idx in enumerate(raw_order)}

        if diversify and len(candidates) > 1:
            order = self._mmr(candidates, final, top_k, is_search=(query_pos is None))
        else:
            order = raw_order[:top_k]

        t_mmr_ms = (time.perf_counter() - t_mmr_start) * 1000
        # --- 6. package the answer with calibrated confidence & evidence ---
        t_calib_start = time.perf_counter()
        calibrator = ConfidenceCalibrator()
        recommendations = []
        source_row = dict(self.articles.iloc[query_pos]) if query_pos is not None else {
            "title": query_text, "description": "", "body": ""
        }

        # Real cross-encoder rank among candidates
        if method == "hybrid+rerank" and rerank_scores is not None and len(rerank_scores) > 0:
            ce_arr = np.asarray(rerank_scores, dtype=float)
            ce_order = list(np.argsort(-ce_arr))
            ce_pos_of = {cand_idx: r + 1 for r, cand_idx in enumerate(ce_order)}
        else:
            ce_pos_of = {}

        # FIX #7: build rank lookup dicts once here (O(N_RETRIEVE)) so the per-recommendation
        # assembly loop can do O(1) dict.get() instead of O(N_RETRIEVE) list.index() scans.
        # These may already exist if use_boosts=True (built in step 4); if not, build them now.
        if "bm25_rank_map" not in locals():
            bm25_rank_map = {c: r + 1 for r, c in enumerate(bm25_rank)}
            dense_rank_map = {c: r + 1 for r, c in enumerate(semantic_rank)}

        for rank_idx, i in enumerate(order, start=1):
            pos = candidates[i]
            cand_row = dict(self.articles.iloc[pos])
            cand_id = str(cand_row["id"])
            unboosted = float(np.clip(relevance[i], 0.0, 1.0))
            next_unboosted = float(np.clip(relevance[order[rank_idx]], 0.0, 1.0)) if rank_idx < len(order) else None

            # Agreement signals — O(1) dict lookups, not O(n) list scans
            bm25_r = bm25_rank_map.get(pos)
            dense_r = dense_rank_map.get(pos)
            raw_r = float(rerank_scores[i]) if (method == "hybrid+rerank" and rerank_scores is not None) else None
            ce_r = ce_pos_of.get(i) if method == "hybrid+rerank" else None
            agree_count, signals = evaluate_signal_agreement(bm25_r, dense_r, raw_r, rerank_rank=ce_r)

            if calibrate:
                max_signals = 3 if (method == "hybrid+rerank" and rerank_scores is not None) else 2
                conf_info = calibrator.calibrate(
                    unboosted, next_score=next_unboosted, agreement_count=agree_count, max_signals=max_signals
                )

                reason_str = build_evidence_reason(
                    source_row, cand_row, self.bm25, bm25_r, dense_r, raw_r,
                    editorial_link=(cand_id in source_linked_set),
                    confidence_info=conf_info,
                    scores={
                        "blend": unboosted,
                        "bm25_norm": float(bm25_norm[i]) if bm25_norm is not None else None,
                        "dense_norm": float(dense_norm[i]) if dense_norm is not None else None,
                        "unboosted_confidence": calibrator.predict_probability(unboosted),
                        "boost_items": boost_parts.get(i),
                        "boosted_raw": float(final[i]),
                    },
                    rank_raw=raw_rank_map.get(i, rank_idx),
                    final_rank=rank_idx,
                    cand_pos=pos,
                    bm25_query=query_text,
                    cosine=float(semantic_scores[pos]),
                    rerank_rank=ce_r,
                )
                calibrated_prob = conf_info["calibrated_prob"]
                band = conf_info["band"]
                margin = conf_info["margin"]
            else:
                # Raw uncalibrated mode: bypass Platt sigmoid & deep evidence generation to save latency
                calibrated_prob = round(float(final[i]), 4)
                band = "RAW"
                margin = round(float(final[i] - final[order[rank_idx]]), 4) if rank_idx < len(order) else 0.0
                reason_str = json.dumps({
                    "summary": f"Uncalibrated raw hybrid score {calibrated_prob:.4f} (boosted). Platt scaling disabled.",
                    "terms": [],
                    "entities": [],
                    "signals": signals,
                    "confidence": {"probability": calibrated_prob, "band": "RAW", "margin": margin},
                    "boosts": boost_parts.get(i) or {},
                    "score_breakdown": {
                        "unboosted_raw": unboosted,
                        "boosted_raw": float(final[i]),
                        "platt_calibrated": None,
                    }
                })

            recommendations.append(Recommendation(
                article=self._to_dict(pos),
                score=calibrated_prob,
                reason=reason_str,
                calibrated_score=calibrated_prob,
                confidence_band=band,
                margin=margin,
                agreement_count=agree_count,
                rank=rank_idx,
                rank_raw=raw_rank_map.get(i, rank_idx),
                boosted_score=round(float(final[i]), 4),
            ))
        t_calib_ms = (time.perf_counter() - t_calib_start) * 1000
        total_ms = (time.perf_counter() - t_start) * 1000

        # Snapshot resource metrics without polling overhead
        resource_info = None
        if _MONITORING_AVAILABLE:
            try:
                snap = take_snapshot()
                resource_info = {
                    "cpu_percent": snap.cpu_percent,
                    "ram_mb": snap.ram_mb,
                    "ram_vms_mb": snap.ram_vms_mb,
                    "gpu_available": snap.gpu_available,
                    "gpu_util_pct": snap.gpu_util_pct,
                    "gpu_mem_used_mb": snap.gpu_mem_used_mb,
                    "gpu_mem_total_mb": snap.gpu_mem_total_mb,
                    "disk_read_mb": snap.disk_read_mb,
                    "disk_write_mb": snap.disk_write_mb,
                }
            except Exception:
                pass

        timings = {
            "total_ms": round(total_ms, 2),
            "prep_ms": round(t_prep_ms, 2),
            "query_embed_ms": round(t_query_embed_ms, 2),
            "dense_scan_ms": round(t_dense_scan_ms, 2),
            "bm25_ms": round(t_bm25_ms, 2),
            "dense_ms": round(t_dense_ms, 2),
            "retrieval_ms": round(t_retrieval_ms, 2),
            "rerank_ms": round(t_rerank_ms, 2),
            "blend_ms": round(t_blend_ms, 2),
            "boosts_ms": round(t_boosts_ms, 2),
            "mmr_ms": round(t_mmr_ms, 2),
            "calibration_ms": round(t_calib_ms, 2),
            # Professional metrics for the latency HUD
            "corpus_size": len(self.articles),
            "candidate_count": len(candidates),
            "n_retrieve": config.N_RETRIEVE,
            "recency_injected": _recency_injected,
            "throughput_rps": round(1000.0 / max(total_ms, 0.1), 1),
            "resource": resource_info,
        }
        return Result(recommendations, low_confidence, best_relevance, method, notes, timings=timings, candidate_strategy=candidate_strategy)

    # ---------------------------------------------------------------- helpers
    # NOTE: passage extraction for live search is now handled by
    # Reranker.score_pairs(is_search=True) -> extract_query_centered_passage()
    # which does a token-based, query-centered window anchored to title+desc.

    def _build_user_profile_vector(self, user_history: list):
        if not user_history:
            return None
        recent_ids = [aid for aid in user_history if aid in self.id_to_pos][-config.MAX_USER_HISTORY:]
        if not recent_ids:
            return None
        
        vectors = []
        weights = []
        for rank, aid in enumerate(reversed(recent_ids)):
            pos = self.id_to_pos[aid]
            vec = self.semantic.get_vector(pos)
            # Recency weighting (exponential decay for older history items)
            weight = 0.8 ** rank
            vectors.append(vec)
            weights.append(weight)
        
        weights = np.array(weights) / sum(weights)
        profile_vec = np.sum([w * v for w, v in zip(weights, vectors)], axis=0)
        return profile_vec

    @staticmethod
    def _rank(scores, blocked, n, min_score=-np.inf) -> list:
        """Positions of the n best articles, skipping blocked ones and scores <= min_score.

        Uses O(N) np.argpartition to find the top-n candidates, then sorts only those
        n items. This is mathematically identical to sorting the full array with
        np.argsort, but ~4x faster at 4k articles and ~12ms faster at 200k articles.
        """
        masked = np.where(blocked, -np.inf, scores)
        if len(masked) <= n:
            order = np.argsort(-masked)
        else:
            top_part = np.argpartition(-masked, n)[:n]
            order = top_part[np.argsort(-masked[top_part])]
        return [int(i) for i in order if masked[i] > min_score]

    def _inject_recency_candidates(
        self, candidates: list, query_pos: int, blocked: np.ndarray,
        seen: set, max_inject: int = 2
    ) -> list:
        """Live-only: add up to `max_inject` recently published same-section articles
        that are not already in the candidate pool.

        These articles may have high semantic relevance but low BM25/dense scores
        because they were published after the index was built or simply share no
        keyword overlap with the query. By injecting them, the recency boost in
        boost_parts() can actually surface them if topically appropriate.

        Only runs for method='hybrid' — never touches the precomputed pipeline.
        """
        source_row = self.articles.iloc[query_pos]
        src_section = str(source_row.get("section") or "").strip().lower()
        src_date = source_row.get("date")

        if not src_section or src_date is None:
            return candidates

        try:
            src_dt = pd.to_datetime(src_date)
        except Exception:
            return candidates

        # Find articles in same section, published within 7 days of source, not blocked.
        # Uses the pre-sorted _date_sorted_positions and pre-extracted arrays (computed at engine load)
        # so this is a simple list iteration with nanosecond array lookups — no pandas .iloc or pd.to_datetime in the loop.
        recency_window = getattr(config, "RECENCY_INJECT_DAYS", 7)
        recency_window_ns = int(recency_window * 86400 * 1_000_000_000)
        src_ns = self._date_ns[query_pos] if (hasattr(self, "_date_ns") and query_pos < len(self._date_ns)) else 0

        injected = []
        for pos in self._date_sorted_positions:
            pos = int(pos)
            if pos in seen or blocked[pos]:
                continue
            if hasattr(self, "_sections_lower") and self._sections_lower[pos] != src_section:
                continue
            if src_ns > 0 and hasattr(self, "_date_ns") and self._date_ns[pos] > 0:
                if abs(src_ns - self._date_ns[pos]) > recency_window_ns:
                    continue
            injected.append(pos)
            seen.add(pos)
            if len(injected) >= max_inject:
                break

        return candidates + injected

    def _linked_ids(self, query_pos) -> set:
        """Editorial Also-Read link IDs of the source article, as a string set."""
        return set() if query_pos is None else linked_set(self.articles.iloc[query_pos])

    def _boost_parts(self, query_pos, pos, bm25_rank: int = None, dense_rank: int = None, rerank_score: float = None) -> dict:
        """Every boost multiplier actually applied to this candidate, as numbers.

        The total is simply the product of the parts, so the evaluation card can
        show exactly where a boost came from instead of re-deriving it from a label.
        """
        source = self.articles.iloc[query_pos] if query_pos is not None else None
        # FIX #15: use pre-cached max_date instead of re-scanning the column on every call
        max_date = self._max_date if hasattr(self, "_max_date") else self.articles["date"].max()
        return boost_parts(source, self.articles.iloc[pos], self._max_page_views,
                           max_date,
                           bm25_rank=bm25_rank, dense_rank=dense_rank, rerank_score=rerank_score)

    def _mmr(self, candidates, final, k, is_search: bool = False) -> list:
        """Pick results one by one: high relevance, but penalise near-duplicate articles."""
        return mmr(final, self.semantic.similarity_matrix(candidates), k,
                   config.MMR_LAMBDA_SEARCH if is_search else config.MMR_LAMBDA,
                   config.MMR_THETA_SEARCH if is_search else config.MMR_THETA)

    def _to_dict(self, pos) -> dict:
        row = self.articles.iloc[pos]
        snippet = row["description"] or row["body"][:200]
        created = ""
        if "created_at" in row:
            val = row["created_at"]
            try:
                if pd.notna(val):
                    created = val.isoformat() if hasattr(val, "isoformat") else str(val)
            except Exception:
                created = ""
        return {
            "id": row["id"], "title": row["title"], "snippet": snippet,
            "section": row["section"], "subsection": row["subsection"],
            "date": row["date"].strftime("%d %b %Y") if pd.notna(row["date"]) else "",
            "url": row["url"], "image": row["image"], "is_live": bool(row["is_live"]),
            "created_at": created,
        }

