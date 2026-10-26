"""
End-to-end ingestion and precomputation pipeline:
CSV -> Cleaning -> PostgreSQL -> Dense Embeddings (pgvector) -> BM25 + pgvector Hybrid Retrieval (RRF Top 30) -> Cross-Encoder Rerank -> News Boosts & MMR -> Top 10 Recommendations with Reasons
"""
import io
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import numpy as np
import pandas as pd

import config
from .connection import SessionLocal, init_db
from .models import Article, ArticleRecommendation, IngestionJob
from .repository import ArticleRepository
from src.retrieval import BM25Index, tokenize, reciprocal_rank_fusion
from src.model_registry import get_embedder
from src.reranker import Reranker
from src.data_loader import clean_dataframe
from src.calibration import ConfidenceCalibrator
from src.evidence import build_evidence_reason, evaluate_signal_agreement, extract_shared_entities
from src.scoring import blend, boost_parts, json_ids, linked_set, mmr


def _fmt_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def _titles_summary(items, limit: int = 4) -> str:
    """'“Title A”, “Title B” (+3 more)' for a list of {"id","title"} dicts."""
    titles = [a["title"] for a in items[:limit] if a.get("title")]
    if not titles:
        return ""
    more = f" (+{len(items) - len(titles)} more)" if len(items) > len(titles) else ""
    return ", ".join(f"“{t[:45]}”" for t in titles) + more


def _focused_query(row, bm25) -> str:
    """
    The query used to represent an article when searching for its neighbours:
    its title tokens plus its 15 most distinctive (highest-IDF) body terms.
    Shared by single-article precompute and by the reverse-push so both use
    exactly the same notion of "what this article is about".
    """
    title_toks = tokenize(str(row["title"]))
    body_toks = tokenize(f"{row['description']} {row['body']}")
    top_terms = sorted(set(body_toks), key=lambda t: -bm25.idf.get(t, 0.0))[:15]
    return " ".join(title_toks + top_terms)


def _load_full_corpus(db):
    """
    Load EVERY stored article (already-existing + just-ingested) into pipeline
    records plus an aligned 768-d embedding matrix.

    Doing this is what lets a newly uploaded article retrieve neighbours from the
    whole database instead of only from the batch it arrived in.
    """
    arts = db.query(Article).order_by(Article.date.desc().nulls_last()).all()
    records = ArticleRepository.corpus_rows(arts)
    vectors = [a.embedding if a.embedding is not None else [0.0] * 768 for a in arts]
    emb = np.asarray(vectors, dtype=np.float32) if vectors else np.zeros((0, 768), dtype=np.float32)
    return records, emb, {r["id"]: pos for pos, r in enumerate(records)}


def _gate_candidates(row, cands, records, bm25_scores):
    """
    Technique 3 - lexical / entity support gate.

    Drop any candidate that shares NEITHER a BM25 keyword NOR a named entity with
    the source article. Never returns an empty list: if everything would be
    gated out the original candidates are returned unchanged, because a genuinely
    isolated article still deserves whatever the reranker thinks is closest.

    Editorial Also-Read links are admitted unconditionally - they are the ground
    truth the whole evaluation is built on.
    """
    if not (config.LEXICAL_GATE or config.ENTITY_GATE):
        return cands

    source_linked = linked_set(row)

    src_ents = set()
    if config.ENTITY_GATE:
        src_ents = {e.lower() for e in extract_shared_entities(row.get("text_rerank", ""), "", limit=24)}

    kept = []
    for c in cands:
        if str(records[c]["id"]) in source_linked:
            kept.append(c)
            continue
        if config.LEXICAL_GATE and bm25_scores[c] > 1e-9:
            kept.append(c)
            continue
        if src_ents:
            c_ents = {e.lower() for e in extract_shared_entities(records[c].get("text_rerank", ""), "", limit=24)}
            if src_ents & c_ents:
                kept.append(c)
    return kept or cands


def _rank_and_build(
    source_row, cand_positions, rerank_vec, records, emb, bm25,
    bm25_all, dense_all, bm25_query, bm25_pos_of, dense_pos_of,
    max_views, max_date, current_ver,
):
    """
    Blend -> boosts -> MMR -> admission gate -> evidence rows.

    The single ranking implementation, called by BOTH the forward precompute and
    the reverse push. They used to be two hand-maintained copies and had already
    drifted (one clipped the dense term and one did not, and reverse-pushed rows
    were missing part of the evidence). ``bm25_pos_of`` / ``dense_pos_of`` map a
    corpus position to its real rank so an agreement signal is never invented.
    """
    relevance, bm25_norm, dense_norm = blend(
        rerank_vec, bm25_all, dense_all, cand_positions)

    parts = [boost_parts(source_row, records[c], max_views, max_date) for c in cand_positions]
    boosted = relevance * np.array([p["total"] for p in parts])

    # Relevance order before diversity: what every rank shift is measured against.
    raw_order = list(np.argsort(-boosted))
    raw_rank = {i: r + 1 for r, i in enumerate(raw_order)}

    # A pool of TOP_K or fewer needs no diversity pass: relevance order is final.
    order = raw_order if len(cand_positions) <= config.TOP_K else mmr(
        boosted, emb[cand_positions] @ emb[cand_positions].T, config.TOP_K)

    # Relevance admission gate (techniques 1-3): trim to genuinely related
    # articles instead of padding to a fixed TOP_K. Editorial Also-Read links
    # bypass every gate - they are the ground truth the evaluation is built on.
    linked = linked_set(source_row)
    src_text = source_row.get("text_rerank", "")
    linked_flags = [str(records[cand_positions[r]]["id"]) in linked for r in order]
    entity_flags = [
        linked_flags[i] or bool(extract_shared_entities(
            src_text, records[cand_positions[r]].get("text_rerank", ""), limit=1))
        for i, r in enumerate(order)
    ]
    gate = config.admission_cut([float(rerank_vec[r]) for r in order], entity_flags, linked_flags)
    order = [order[i] for i in gate["keep"]]

    calibrator = ConfidenceCalibrator()
    recs = []
    for rank, rel in enumerate(order, start=1):
        pos = cand_positions[rel]
        cand_id = str(records[pos]["id"])
        unboosted = float(np.clip(relevance[rel], 0.0, 1.0))
        nxt = float(np.clip(relevance[order[rank]], 0.0, 1.0)) if rank < len(order) else None
        raw = float(rerank_vec[rel])

        # Real corpus ranks for the agreement signals.
        bm25_pos, dense_pos = bm25_pos_of.get(pos, 999), dense_pos_of.get(pos, 999)
        agree, _ = evaluate_signal_agreement(bm25_pos, dense_pos, raw)
        conf = calibrator.calibrate(unboosted, next_score=nxt, agreement_count=agree)

        recs.append({
            "recommended_article_id": cand_id,
            "rank": rank,
            "rank_raw": raw_rank.get(rel, rank),
            "score": conf["calibrated_prob"],
            "boosted_score": round(float(boosted[rel]), 4),
            "rerank_score": round(raw, 6),
            "pipeline_version": current_ver,
            "method": "neural_hybrid_rerank",
            "reason": build_evidence_reason(
                source_row, records[pos], bm25, bm25_pos, dense_pos, raw,
                editorial_link=(cand_id in linked), confidence_info=conf,
                scores={
                    "blend": unboosted,
                    "bm25_norm": float(bm25_norm[rel]),
                    "dense_norm": float(dense_norm[rel]),
                    "unboosted_confidence": calibrator.predict_probability(unboosted),
                    "boost_items": parts[rel],
                    "boosted_raw": float(boosted[rel]),
                },
                rank_raw=raw_rank.get(rel, rank), final_rank=rank,
                cand_pos=pos, bm25_query=bm25_query,
                cosine=float(dense_all[pos]), admission=gate,
            ),
        })
    return recs


def _precompute_for_article(
    idx, records, bm25, reranker, emb, max_views, max_date, n_retrieve, n_cand
):
    """
    Compute the top recommendations for ONE article against the full corpus.

    Returns a tuple ``(recs, candidates)``:
      * recs       -> rows ready for persistence (each carries a raw rerank_score)
      * candidates -> the fused candidate positions, reused by the reverse-push
                      so we know which existing articles this one is close to.
    """
    row = records[idx]

    # A. Keyword candidates. The query string is kept so the evidence card can
    # show which terms actually matched (bm25_term_hits).
    bm25_query = _focused_query(row, bm25)
    bm25_scores = bm25.scores(bm25_query)
    bm25_scores[idx] = -np.inf
    bm25_rank = [
        int(p) for p in np.argsort(-bm25_scores)[:n_retrieve] if bm25_scores[p] > 1e-9
    ]

    # B. Semantic candidates
    dense_scores = emb @ emb[idx]
    dense_scores[idx] = -np.inf
    dense_rank = [int(p) for p in np.argsort(-dense_scores)[:n_retrieve]]

    # C. Reciprocal-rank fusion, then the lexical/entity gate (technique 3): a
    # candidate with no shared keyword AND no shared entity is noise however close
    # it sits in embedding space, so it never reaches the cross-encoder.
    fused = reciprocal_rank_fusion([bm25_rank, dense_rank], k=config.RRF_K)
    cands = sorted(fused, key=fused.get, reverse=True)[:n_cand]
    if not cands:
        return [], []
    cands = _gate_candidates(row, cands, records, bm25_scores)

    # D. Cross-encoder reranking (article-to-article: 50/50 budget split)
    rerank_scores = np.asarray(
        reranker.score_pairs(
            row["text_rerank"],
            [records[c]["text_rerank"] for c in cands],
            is_search=False,
            batch_size=32,
        ),
        dtype=np.float32,
    )

    # E-H. Blend -> boosts -> MMR -> gate -> evidence.
    recs = _rank_and_build(
        row, cands, rerank_scores, records, emb, bm25, bm25_scores, dense_scores,
        bm25_query,
        {c: i + 1 for i, c in enumerate(bm25_rank)},
        {c: i + 1 for i, c in enumerate(dense_rank)},
        max_views, max_date, config.get_pipeline_version(),
    )
    return recs, cands


def _reverse_push_for_old(
    db, new_positions, candidate_map, records, emb, bm25, reranker,
    max_views, max_date, new_id_set, id_to_pos, progress_cb=None,
):
    """
    Let newly ingested articles flow into the recommendation lists of the EXISTING
    articles they are genuinely close to.

    Why this is correct (and cheap):
      * For each new article N we already know its full-corpus candidates
        (``candidate_map``). By (approximate) similarity symmetry, an old article O
        can only be displaced by N if N is one of O's close neighbours too — so the
        candidate set found from N's side is exactly the set worth checking.
      * We reuse O's *stored* rerank scores for its current entries (no cross-encoder
        needed for the old list) if they match the current pipeline version, and
        re-score if the model/configuration changed.
      * We then merge, re-apply boosts + MMR, and keep the top-K.

    Old articles that no new article matches are never touched.

    ``progress_cb(done, total, title)`` (optional) is called as each existing article
    is processed so the caller can stream progress to the Editor log.

    Returns a list of dicts ``{"id", "title"}`` describing every existing article whose
    list was refreshed — so the caller can report exactly which old articles changed.
    """
    # 1. Reverse adjacency: old article position -> new articles close to it.
    reverse = defaultdict(list)
    for npos in new_positions:
        for cpos in candidate_map.get(npos, []):
            if str(records[cpos]["id"]) not in new_id_set:   # keep only OLD articles
                reverse[cpos].append(npos)
    if not reverse:
        return []

    total = len(reverse)
    report_every = max(1, total // 12)   # lively, but no more than ~12 log lines
    refreshed: List[Dict[str, Any]] = []
    current_ver = config.get_pipeline_version()

    for done, (opos, new_neighbors) in enumerate(reverse.items(), start=1):
        if progress_cb and (done % report_every == 0 or done == total):
            progress_cb(done, total, str(records[opos].get("title", "")))

        old_id = str(records[opos]["id"])
        stored = ArticleRepository.get_precomputed_rows(db, old_id, limit=config.TOP_K)

        # 2. Seed the merged pool with O's current entries (reusing their rerank scores
        #    only if they were computed under the current pipeline version).
        cand_positions, rerank_scores, seen = [], [], set()
        for r in stored:
            p = id_to_pos.get(str(r["recommended_article_id"]))
            if p is None or p in seen:
                continue
            seen.add(p)
            cand_positions.append(p)
            same_ver = (r.get("pipeline_version") == current_ver)
            rerank_scores.append(r.get("rerank_score") if same_ver else None)

        # 3. Add the new neighbours (deduped); they still need an O-centric score.
        for npos in dict.fromkeys(new_neighbors):
            if npos != opos and npos not in seen:
                seen.add(npos)
                cand_positions.append(npos)
                rerank_scores.append(None)
        if not cand_positions:
            continue

        # 4. Score only the candidates that have no reusable score (new + legacy/version-mismatched rows).
        missing = [i for i, s in enumerate(rerank_scores) if s is None]
        if missing:
            scored = reranker.score_pairs(
                records[opos]["text_rerank"],
                [records[cand_positions[i]]["text_rerank"] for i in missing],
                is_search=False,
                batch_size=32,
            )
            for k, i in enumerate(missing):
                rerank_scores[i] = float(scored[k])
        rerank_vec = np.asarray(rerank_scores, dtype=np.float32)

        # 5. Real corpus ranks for the agreement signals. The old code passed
        #    `rel_pos + 1` for both BM25 and dense, which invented agreement.
        bm25_query_old = _focused_query(records[opos], bm25)
        bm25_all = bm25.scores(bm25_query_old)
        dense_all = emb @ emb[opos]

        # 6. Blend -> boosts -> MMR -> gate -> evidence, exactly as the forward
        #    path does, then persist.
        recs = _rank_and_build(
            records[opos], cand_positions, rerank_vec, records, emb, bm25,
            bm25_all, dense_all, bm25_query_old,
            {int(p): i + 1 for i, p in enumerate(np.argsort(-bm25_all))},
            {int(p): i + 1 for i, p in enumerate(np.argsort(-dense_all))},
            max_views, max_date, current_ver,
        )
        ArticleRepository.save_recommendations(db, old_id, recs, commit=False)
        refreshed.append({"id": old_id, "title": str(records[opos].get("title", ""))})
    return refreshed




def run_ingestion_pipeline(
    csv_source: Union[str, Path, io.BytesIO],
    job_id: str,
    limit: Optional[int] = None,
    recompute_all_recommendations: bool = False,
):
    """
    CSV -> clean -> PostgreSQL -> embeddings -> full-corpus BM25+dense -> RRF -> rerank
        -> boosts -> MMR -> top-10 saved.

    Recommendation maintenance is INCREMENTAL by default:
      * the uploaded (new) articles are precomputed against the WHOLE database, and
      * existing articles that a new article is close to get a reverse-push update.

    Pass ``recompute_all_recommendations=True`` to rebuild every article's list
    from scratch (correct but slower — the "Option A" full rebuild).
    """
    db = SessionLocal()
    t_job = time.time()
    try:
        # Idempotent: ensures the pgvector extension, tables and the rerank_score
        # column all exist before we start writing (safe on a fresh database).
        init_db()
        upd = lambda **kw: ArticleRepository.update_job(db, job_id, **kw)
        current_version = config.get_pipeline_version()
        upd(pipeline_version=current_version)

        # Compare with previous completed ingestion job
        last_job = db.query(IngestionJob).filter(
            IngestionJob.status == "completed",
            IngestionJob.id != job_id
        ).order_by(IngestionJob.created_at.desc()).first()

        if last_job and last_job.pipeline_version and last_job.pipeline_version != current_version:
            if not recompute_all_recommendations:
                print(
                    f"[Pipeline] Notice: Pipeline configuration changed ({last_job.pipeline_version} -> {current_version}). "
                    f"Forcing full recommendation recompute to maintain score scale integrity."
                )
                recompute_all_recommendations = True

        # 1. Parse CSV
        upd(status="processing", message="STAGE:PARSE|Parsing and validating CSV records...")
        raw_df = pd.read_csv(csv_source)
        if limit and limit > 0:
            raw_df = raw_df.head(limit)
        upd(
            total_articles=len(raw_df),
            message=f"STAGE:CLEAN|Cleaning and normalizing {len(raw_df)} articles...",
        )

        # 2. Clean records
        df = clean_dataframe(raw_df)
        n = len(df)
        if n == 0:
            raise ValueError("CSV contained no usable articles.")

        records = df.to_dict("records")
        ids = [str(r["id"]) for r in records]

        # Check existing state in PostgreSQL for validation and skip-unchanged optimization
        existing_rows = db.query(Article.id, Article.text_embed, Article.embedding).filter(
            Article.id.in_(ids)
        ).all()
        existing_map = {r[0]: (r[1], r[2]) for r in existing_rows}

        existing_recs = db.query(ArticleRecommendation.article_id).filter(
            ArticleRecommendation.article_id.in_(list(existing_map.keys())),
            ArticleRecommendation.pipeline_version == current_version
        ).distinct().all()
        has_current_recs_set = {r[0] for r in existing_recs}

        new_ids = set(ids) - set(existing_map.keys())
        changed_ids = {
            r["id"] for r in records
            if r["id"] in existing_map and existing_map[r["id"]][0] != r["text_embed"]
        }
        needs_emb_ids = new_ids | changed_ids | {
            aid for aid, (txt, emb) in existing_map.items() if emb is None
        }

        dup_count = len(raw_df) - len(df)
        val_summary = (
            f"{len(raw_df)} parsed -> {len(df)} unique articles "
            f"({dup_count} dupes dropped, {len(new_ids)} new, {len(changed_ids)} modified, "
            f"{len(existing_map) - len(changed_ids)} unchanged)"
        )
        print(f"[Pipeline] Validation: {val_summary}")
        upd(total_articles=n, message=f"STAGE:STORAGE|Validation: {val_summary}. Saving to DB...")

        # 3. Persist articles with batched INSERT ... ON CONFLICT
        ArticleRepository.bulk_upsert_articles_fast(db, [{
            "id": str(r["id"]),
            "title": str(r["title"]),
            "description": str(r["description"]),
            "body": str(r["body"]),
            "section": str(r["section"]),
            "subsection": str(r["subsection"]),
            "date": r["date"].to_pydatetime() if pd.notna(r["date"]) else None,
            "url": str(r["url"]),
            "image": str(r["image"]),
            "is_live": bool(r["is_live"]),
            "page_views": int(r["page_views"]),
            "unique_users": int(r["unique_users"]),
            "linked_ids": json_ids(r["linked_ids"]),
            "text_embed": str(r["text_embed"]),
            "text_bm25": str(r["text_bm25"]),
            "text_rerank": str(r["text_rerank"]),
            "embed_token_count": int(r.get("embed_token_count", 0)),
            "rerank_token_count": int(r.get("rerank_token_count", 0)),
            "is_truncated": bool(r.get("is_truncated", False)),
            "recipe_version": str(r.get("recipe_version", config.TEXT_RECIPE_VERSION)),
        } for r in records])

        # 4. Build and persist embeddings (only for new, modified, or missing-vector articles)
        if needs_emb_ids:
            upd(message=f"STAGE:EMBEDDING|Embedding {len(needs_emb_ids)} articles (skipping {len(ids) - len(needs_emb_ids)} unchanged)...")
            embedder = get_embedder()
            emb_records = [r for r in records if r["id"] in needs_emb_ids]
            emb_ids = [r["id"] for r in emb_records]
            emb_texts = [r["text_embed"] for r in emb_records]

            chunks, chunk_size, t0 = [], 128, time.time()
            for i in range(0, len(emb_texts), chunk_size):
                chunks.append(embedder.encode(
                    emb_texts[i:i + chunk_size],
                    batch_size=32,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ))
                done = min(i + chunk_size, len(emb_texts))
                eta = (time.time() - t0) / done * (len(emb_texts) - done)
                upd(
                    processed_articles=int(0.40 * done),
                    message=f"STAGE:EMBEDDING|Embedded {done}/{len(emb_texts)} articles (ETA {_fmt_eta(eta)})...",
                )
            emb = np.vstack(chunks).astype(np.float32)
            ArticleRepository.bulk_set_embeddings(db, emb_ids, emb)
        else:
            print(f"[Pipeline] All {len(ids)} articles have up-to-date embeddings. Skipping embedding stage.")
            upd(message="STAGE:EMBEDDING|All embeddings up-to-date. Skipped embedding step.")

        # 5. Build a FULL-corpus index (already-stored articles + the new ones).
        upd(message="STAGE:INDEX|Building full-corpus BM25 index...")
        full_records, full_emb, id_to_pos = _load_full_corpus(db)
        n_total = len(full_records)
        if n_total == 1:
            print("[Pipeline] Only one article in the corpus; no neighbours to precompute.")
        bm25 = BM25Index([r["text_bm25"] for r in full_records])

        # 6. Determine articles needing recommendations
        new_id_set = set(ids)
        if recompute_all_recommendations:
            target_positions = list(range(n_total))                       # Option A: full rebuild
        else:
            # Only articles that are brand new, had text changes, or don't have valid current recommendations
            needs_recs_ids = new_ids | changed_ids | (set(ids) - has_current_recs_set)
            target_positions = [id_to_pos[i] for i in needs_recs_ids if i in id_to_pos]

        # Positions of genuinely new or changed articles that need to reverse-push into older articles
        new_or_changed_positions = [id_to_pos[i] for i in (new_ids | changed_ids) if i in id_to_pos]

        # Early exit if everything is already precomputed (perfectly idempotent re-upload)
        if not target_positions and not new_or_changed_positions:
            upd(
                status="completed",
                processed_articles=n,
                refreshed_articles=0,
                message=f"STAGE:COMPLETE|All {n} articles and recommendations already up-to-date ({val_summary}).",
            )
            print(f"[Pipeline] Idempotent upload: no articles needed embedding or recommendation recalculation.")
            return job_id

        # 6a. Rerank candidates and save recommendations in batches
        upd(message="STAGE:RERANK|Loading cross-encoder (first run downloads ~1 GB, can take minutes)...")
        reranker = Reranker()

        max_views = int(max((r["page_views"] for r in full_records), default=0)) or 1
        max_date = max((r["date"] for r in full_records if pd.notna(r["date"])), default=None)
        n_retrieve = min(config.N_RETRIEVE, max(n_total - 1, 0))
        n_cand = min(config.N_CANDIDATES, max(n_total - 1, 0))

        candidate_map: Dict[int, List[int]] = {}
        # Distinct EXISTING articles that any new article is close to. This grows as
        # the precompute loop runs, which lets us estimate the reverse-update
        # workload (and therefore an honest, whole-job time-left) BEFORE that pass.
        reverse_seen: set = set()
        track_reverse = (not recompute_all_recommendations) and config.INCREMENTAL_RECOMMENDATIONS
        failed, t0 = 0, time.time()
        total_targets = max(len(target_positions), 1)

        for done, pos in enumerate(target_positions, start=1):
            try:
                recs, cands = _precompute_for_article(
                    pos, full_records, bm25, reranker, full_emb,
                    max_views, max_date, n_retrieve, n_cand,
                )
            except Exception as art_err:  # one bad article must not terminate the entire job
                failed += 1
                recs, cands = [], []
                print(f"[Pipeline] Skipped article {full_records[pos]['id']}: {art_err}")

            if recs:
                ArticleRepository.save_recommendations(
                    db, str(full_records[pos]["id"]), recs, commit=False
                )
            candidate_map[pos] = cands
            if track_reverse:
                for cpos in cands:
                    if str(full_records[cpos]["id"]) not in new_id_set:
                        reverse_seen.add(cpos)

            if done % config.PRECOMPUTE_COMMIT_EVERY == 0 or done == total_targets:
                per = (time.time() - t0) / done
                eta = per * (total_targets - done)

                # Estimate the upcoming reverse pass so the displayed ETA covers the
                # WHOLE job, not just this loop. Extrapolate the distinct old articles
                # seen so far across the full upload.
                est_rev = len(reverse_seen)
                if track_reverse:
                    if done < total_targets:
                        est_rev = int(round(est_rev * total_targets / done))
                    eta += per * est_rev
                eta_txt = f"ETA {_fmt_eta(eta)}"
                if track_reverse and est_rev:
                    eta_txt += f" (incl. ~{_fmt_eta(per * est_rev)} reverse)"

                # update_job commits and therefore commits pending recommendation writes too.
                # Reserve the last 5% of the bar for the reverse-update pass (step 6b).
                upd(
                    processed_articles=int(0.40 * n + 0.55 * n * done / total_targets),
                    message=(
                        f"STAGE:RERANK|Precomputed {done}/{total_targets} "
                        f"({per:.1f}s/article, {eta_txt})"
                    ),
                )
        db.commit()

        # 6b. Reverse pass: let the new articles flow into the lists of the EXISTING
        #     articles they are close to (skipped during a full rebuild).
        refreshed_infos: List[Dict[str, Any]] = []
        if track_reverse and new_or_changed_positions:
            rev_total = len(reverse_seen)
            upd(message=(
                f"STAGE:REVERSE|Re-scoring {rev_total} existing article(s) "
                f"affected by this upload..."
            ))

            t_rev = time.time()

            def _rev_progress(done, total, title):
                snippet = (title[:48] + "…") if len(title) > 48 else title
                per_rev = (time.time() - t_rev) / max(done, 1)
                eta_rev = per_rev * (total - done)
                # Walk the bar through its reserved final 5% (95% -> 100%) and show
                # a live time-left for this, the last, step of the job.
                upd(
                    processed_articles=int(0.95 * n + 0.05 * n * done / max(total, 1)),
                    message=(
                        f"STAGE:REVERSE|Re-scoring existing articles ({done}/{total}, "
                        f"ETA {_fmt_eta(eta_rev)}) — checking “{snippet}”..."
                    ),
                )

            refreshed_infos = _reverse_push_for_old(
                db, new_or_changed_positions, candidate_map, full_records, full_emb, bm25,
                reranker, max_views, max_date, new_id_set, id_to_pos,
                progress_cb=_rev_progress,
            )
            db.commit()

            count = len(refreshed_infos)
            upd(refreshed_articles=count)
            if refreshed_infos:
                upd(message=(
                    f"STAGE:REVERSE|Updated precomputed recommendations for {count} "
                    f"existing article(s): {_titles_summary(refreshed_infos)}"
                ))
            else:
                upd(message="STAGE:REVERSE|No existing articles needed refreshing for this upload.")
            print(f"[Pipeline] Reverse pass refreshed {count} existing article(s).")

        # 7. Refresh live engine directly in this process (no HTTP request needed).
        upd(message="STAGE:INDEX|Refreshing live search engine...")
        try:
            from api.server import reload_engine_from_postgres
            reload_engine_from_postgres()
        except Exception as reload_err:
            print(f"[Pipeline] Engine reload skipped: {reload_err}")

        note = f" ({failed} skipped)" if failed else ""
        extra = ""
        if refreshed_infos:
            extra = (f" Reverse-updated {len(refreshed_infos)} existing article(s): "
                     f"{_titles_summary(refreshed_infos)}.")
        upd(
            status="completed",
            processed_articles=n,
            refreshed_articles=len(refreshed_infos),
            message=f"STAGE:COMPLETE|Processed {n} articles{note} in {_fmt_eta(time.time() - t_job)}.{extra}",
        )

    except Exception as exc:
        print(f"[Pipeline Error] {traceback.format_exc()}")
        try:
            db.rollback()  # ensure the failure status can be written after a DB error
            ArticleRepository.update_job(
                db,
                job_id,
                status="failed",
                message="Processing encountered an error",
                error=str(exc)[:2000],
            )
        except Exception as inner:
            print(f"[Pipeline] Could not record failure: {inner}")
    finally:
        db.close()
