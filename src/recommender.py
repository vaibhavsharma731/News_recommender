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
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from src.reranker import Reranker
from src.retrieval import (BM25Index, DenseIndex, TfidfIndex,
                           reciprocal_rank_fusion, tokenize)
from src.calibration import ConfidenceCalibrator
from src.evidence import build_evidence_reason, evaluate_signal_agreement

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



@dataclass
class Result:
    recommendations: list
    low_confidence: bool          # True = nothing in the dataset matches well
    best_relevance: float
    method: str                   # the method that was really used
    notes: list = field(default_factory=list)


class Recommender:
    def __init__(self, articles: pd.DataFrame, use_neural: bool = config.USE_NEURAL_MODELS, vectors=None, use_disk_cache=True):
        self.articles = articles.reset_index(drop=True)
        self.id_to_pos = {}
        for i, aid in enumerate(self.articles["id"]):
            self.id_to_pos[aid] = i
            self.id_to_pos[str(aid)] = i
        self.notes = []
        self._max_page_views = int(self.articles["pageViews"].max()) if "pageViews" in self.articles.columns else 1

        self.bm25 = BM25Index(self.articles["text_bm25"])
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
                  use_boosts: bool = True, diversify: bool = True, section: str = None) -> Result:
        """Give EITHER `article_id` (recommend similar articles) OR `query` (free text) OR `user_history` (personalized for user)."""
        notes = list(self.notes)
        if method == "hybrid+rerank" and self.reranker is None:
            method = "hybrid"

        # --- 0. what are we searching with? -------------------------------
        user_vector = self._build_user_profile_vector(user_history) if user_history else None

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
            semantic_scores = self.semantic.scores_for_doc(query_pos)
            bm25_scores = self.bm25.scores(focused_text)
        elif query:
            query_pos = None
            clean_query, corrections, _ = self.bm25.normalize_query(query)
            query_text = clean_query
            rerank_query = clean_query
            is_short = len(query_text.split()) < config.SHORT_QUERY_WORDS
            semantic_scores = self.semantic.scores_for_text(query_text, is_short)
            bm25_scores = self.bm25.scores(query)
            if corrections:
                corr_str = ", ".join(f"'{orig}' -> '{corr}'" for orig, corr in corrections)
                notes.append(f"Auto-corrected query terms: {corr_str}")
        elif user_vector is not None:
            query_pos = None
            query_text = rerank_query = "Personalized recommendation feed based on your reading history"
            semantic_scores = self.semantic.scores_for_vector(user_vector)
            bm25_scores = np.zeros(len(self.articles))
        else:
            return Result([], False, 0.0, method, notes + ["Empty query and no history."])

        # --- articles that may not be returned ----------------------------
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
        else:
            fused = reciprocal_rank_fusion([bm25_rank, semantic_rank])
            candidates = sorted(fused, key=fused.get, reverse=True)[:config.N_CANDIDATES]
        if not candidates:
            return Result([], True, 0.0, method, notes)

        # --- 3. relevance score of each candidate (0-1) -------------------
        if method == "hybrid+rerank" and query_text:
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
            
            # Blend with normalized BM25 keyword score so entity presence is honored
            bm25_cand = bm25_scores[candidates]
            bm25_max = bm25_cand.max() if len(bm25_cand) and bm25_cand.max() > 0 else 1.0
            bm25_norm = bm25_cand / bm25_max
            if query_pos is None:
                # Text query: semantic reranker is primary, BM25 supports entity matching
                relevance = 0.65 * rerank_scores + 0.35 * bm25_norm
            else:
                # Article-to-article: reranker + semantic + keyword blend
                sem_cand = semantic_scores[candidates]
                sem_max = sem_cand.max() if len(sem_cand) and sem_cand.max() > 0 else 1.0
                sem_norm = sem_cand / sem_max
                relevance = 0.55 * rerank_scores + 0.30 * bm25_norm + 0.15 * sem_norm
            
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
        else:
            relevance = semantic_scores[candidates]
            best_relevance = float(np.max(relevance)) if len(relevance) > 0 else 0.0
            threshold = config.MIN_RELEVANCE.get(self.semantic.name, 0.50)
            low_confidence = threshold is not None and best_relevance < threshold
            rerank_scores = np.zeros(len(candidates))

        # --- Lower Filter: Filter out non-relatable stories below discard floor ---
        discard_floor = getattr(config, "ARTICLE_DISCARD_FLOOR", 0.05)
        source_linked_set = set()
        if query_pos is not None:
            source_row_data = self.articles.iloc[query_pos]
            raw_links = source_row_data.get("linked_ids", [])
            if isinstance(raw_links, str):
                try:
                    import json
                    raw_links = json.loads(raw_links)
                except Exception:
                    raw_links = []
            source_linked_set = set(str(x) for x in raw_links)

        if method == "hybrid+rerank" and len(candidates) > 1:
            valid_idx = [
                idx for idx in range(len(candidates))
                if rerank_scores[idx] >= discard_floor or str(self.articles.iloc[candidates[idx]]["id"]) in source_linked_set
            ]
            if valid_idx and len(valid_idx) < len(candidates):
                candidates = [candidates[idx] for idx in valid_idx]
                relevance = relevance[valid_idx]
                rerank_scores = rerank_scores[valid_idx]

        # --- 4. Personalization & News Boost Calibration -------------------
        final = np.array(relevance, dtype=float)

        # Blend with User Profile Vector if present
        if user_vector is not None and len(candidates) > 0:
            user_affinities = self.semantic.scores_for_vector(user_vector)[candidates]
            w = config.USER_PROFILE_WEIGHT
            final = (1 - w) * final + w * user_affinities

        if use_boosts and len(candidates) > 0:
            boost_multipliers = np.array([self._boost_mult(query_pos, pos) for pos in candidates])
            final = final * boost_multipliers

        # --- 5. diversity ---------------------------------------------------
        raw_order = list(np.argsort(-final))
        raw_rank_map = {cand_idx: r + 1 for r, cand_idx in enumerate(raw_order)}

        if diversify and len(candidates) > 1:
            order = self._mmr(candidates, final, top_k, is_search=(query_pos is None))
        else:
            order = raw_order[:top_k]

        # --- 6. package the answer with calibrated confidence & evidence ---
        calibrator = ConfidenceCalibrator()
        recommendations = []
        source_row = dict(self.articles.iloc[query_pos]) if query_pos is not None else {
            "title": query_text, "description": "", "body": ""
        }

        for rank_idx, i in enumerate(order, start=1):
            pos = candidates[i]
            cand_row = dict(self.articles.iloc[pos])
            cand_id = str(cand_row["id"])
            unboosted = float(np.clip(relevance[i], 0.0, 1.0))
            next_unboosted = float(np.clip(relevance[order[rank_idx]], 0.0, 1.0)) if rank_idx < len(order) else None

            # Agreement signals
            bm25_r = list(bm25_rank).index(pos) + 1 if pos in bm25_rank else 999
            dense_r = list(semantic_rank).index(pos) + 1 if pos in semantic_rank else 999
            raw_r = float(rerank_scores[i]) if method == "hybrid+rerank" else None
            agree_count, signals = evaluate_signal_agreement(bm25_r, dense_r, raw_r)

            conf_info = calibrator.calibrate(unboosted, next_score=next_unboosted, agreement_count=agree_count)

            raw_r_val = raw_rank_map.get(i, rank_idx)
            shift_note = f"relevance-only rank #{raw_r_val}, final rank #{rank_idx} (promoted +{raw_r_val - rank_idx})" if rank_idx < raw_r_val else (
                f"relevance-only rank #{raw_r_val}, final rank #{rank_idx} (demoted -{rank_idx - raw_r_val} by MMR diversity)" if rank_idx > raw_r_val else
                f"relevance-only rank #{raw_r_val}, final rank #{rank_idx} (matches relevance)"
            )

            reason_str = build_evidence_reason(
                source_row=source_row,
                cand_row=cand_row,
                bm25_index=self.bm25,
                bm25_rank=bm25_r,
                dense_rank=dense_r,
                rerank_score=raw_r,
                editorial_link=(cand_id in source_linked_set),
                confidence_info=conf_info,
                score_breakdown={
                    "reranker_prob": round(raw_r, 4) if raw_r is not None else None,
                    "bm25_rank": bm25_r,
                    "dense_rank": dense_r,
                    "blend": round(unboosted, 4),
                    "final": round(conf_info["calibrated_prob"], 4)
                },
                rank_relevance={
                    "rank_raw": raw_r_val,
                    "final_rank": rank_idx,
                    "shift": raw_r_val - rank_idx,
                    "description": shift_note
                },
                method_ranks={
                    "bm25": bm25_r,
                    "dense": dense_r,
                    "reranker": round(raw_r, 3) if raw_r is not None else None,
                    "agreement_count": agree_count,
                    "status": "All three methods agree" if agree_count >= 3 else (f"{agree_count} of 3 methods agree" if agree_count >= 2 else "Single method lead")
                },
                margin_cutoff=conf_info.get("margin")
            )

            recommendations.append(Recommendation(
                article=self._to_dict(pos),
                score=conf_info["calibrated_prob"],
                reason=reason_str,
                calibrated_score=conf_info["calibrated_prob"],
                confidence_band=conf_info["band"],
                margin=conf_info["margin"],
                agreement_count=agree_count,
                rank=rank_idx,
                rank_raw=raw_rank_map.get(i, rank_idx),
            ))
        return Result(recommendations, low_confidence, best_relevance, method, notes)

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
        """Positions of the n best articles, skipping blocked ones and scores <= min_score."""
        masked = np.where(blocked, -np.inf, scores)
        order = np.argsort(-masked)[:n]
        return [int(i) for i in order if masked[i] > min_score]

    def _boost_mult(self, query_pos, pos) -> float:
        b = self.articles.iloc[pos]
        mult = 1.0
        if query_pos is not None:
            a = self.articles.iloc[query_pos]
            # Editorial "Also Read" link boost (human curator gold standard)
            if "linked_ids" in a and isinstance(a["linked_ids"], list):
                if b["id"] in a["linked_ids"]:
                    mult *= 1.40
            if a["section"] and a["section"] == b["section"]:
                mult *= config.SECTION_BOOST_MULT
            if a["subsection"] and a["subsection"] == b["subsection"]:
                mult *= config.SUBSECTION_BOOST_MULT

        # Popularity boost: log-scaled so viral articles get a mild boost, not dominance
        page_views = b.get("pageViews", 0)
        if page_views > 0:
            log_views = math.log1p(page_views)
            log_max = math.log1p(self._max_page_views)
            pop_factor = 1.0 + (config.POPULARITY_BOOST_MULT - 1.0) * (log_views / log_max)
            mult *= pop_factor

        if pd.notna(b["date"]):
            max_date = self.articles["date"].max()
            days_old = max(0.0, (max_date - b["date"]).total_seconds() / 86400.0)
            sec = "live_blog" if b["is_live"] else str(b["section"]).lower()
            half_life = config.SECTION_HALF_LIVES.get(sec, config.SECTION_HALF_LIVES["default"])
            recency_mult = 1.0 + (config.RECENCY_MAX_BOOST_MULT - 1.0) * (0.5 ** (days_old / half_life))
            mult *= recency_mult
        return mult

    def _mmr(self, candidates, final, k, is_search: bool = False) -> list:
        """Pick results one by one: high relevance, but penalise near-duplicate articles."""
        similarity = self.semantic.similarity_matrix(candidates)
        selected, remaining = [], list(range(len(candidates)))
        threshold = 0.85 if is_search else config.MMR_THETA  # MEASURED: was 0.80
        lam = 0.88 if is_search else config.MMR_LAMBDA

        while remaining and len(selected) < k:
            def mmr_score(i):
                # Only penalize if similarity exceeds threshold (true near-duplicate wire copy)
                high_sims = [similarity[i][j] for j in selected if similarity[i][j] > threshold]
                redundancy = max(high_sims, default=0.0)
                return lam * final[i] - (1 - lam) * redundancy

            best = max(remaining, key=mmr_score)
            selected.append(best)
            remaining.remove(best)
        return selected

    def _explain(self, query_text, query_pos, pos, relevance_score: float = 0.0, user_history: list = None) -> str:
        """Build a short human-readable reason combining neural semantics and keyword topics."""
        parts = []

        # 1. Neural Semantic Context Match
        if relevance_score >= 0.85:
            parts.append(f"Strong semantic match ({int(relevance_score * 100)}% context similarity)")
        elif relevance_score >= 0.55:
            parts.append(f"Semantic context match ({int(relevance_score * 100)}%)")

        # 2. Shared Topics / Keywords
        if query_text:
            shared = set(tokenize(query_text)) & set(self.bm25.docs[pos])
            shared = sorted(shared, key=lambda t: -self.bm25.idf[t])[:4]
            if shared:
                parts.append("shared topics: " + ", ".join(shared))
            elif not any("semantic" in p.lower() for p in parts):
                parts.append("thematic conceptual match")

        # 3. Section and Temporal Affinity
        if query_pos is not None:
            a, b = self.articles.iloc[query_pos], self.articles.iloc[pos]
            if a["section"] and a["section"] == b["section"]:
                parts.append(f"same section ({b['section']})")
            if pd.notna(a["date"]) and pd.notna(b["date"]):
                days = abs((a["date"] - b["date"]).days)
                if days <= 1:
                    parts.append("published around the same time")
        elif user_history:
            parts.append("matches your recent reading interest")
        return "; ".join(parts) if parts else "High semantic and topic relevance"

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

