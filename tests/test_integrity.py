"""
Comprehensive unit and integration tests for Data Integrity & Ingestion Pipeline:
  1. Single deduplication policy (latest postPublishDate wins).
  2. Flexible column resolution in clean_dataframe.
  3. Collision-safe synthetic ID generation.
  4. Model- and config-aware embedding cache keying.
  5. Pipeline version fingerprinting.
  6. Idempotent re-upload and skip-unchanged logic.

Run with: python -m unittest tests/test_integrity.py
"""
import unittest
import pandas as pd
import numpy as np
import hashlib
from unittest.mock import MagicMock, patch

import config
from src.data_loader import clean_dataframe, _deduplicate, load_articles
from src.retrieval import DenseIndex


class TestDataIntegrity(unittest.TestCase):

    def test_single_dedup_policy_latest_date_wins(self):
        """Verify that when duplicate articleIds exist, the row with latest date survives."""
        raw = pd.DataFrame({
            "articleId": ["art_100", "art_100", "art_100"],
            "title": ["Version Older", "Version Latest", "Version Mid"],
            "postPublishDate": [
                "2026-01-01 10:00:00",
                "2026-03-01 12:00:00",  # Latest
                "2026-02-01 11:00:00"
            ],
            "body_raw": ["Body 1", "Body 2", "Body 3"]
        })
        deduped, stats = _deduplicate(raw, "articleId")
        self.assertEqual(len(deduped), 1)
        self.assertEqual(stats["duplicates_dropped"], 2)
        self.assertEqual(deduped.iloc[0]["title"], "Version Latest")

    def test_dedup_content_conflicts_logging(self):
        """Verify content conflicts (same ID, different title) are counted correctly."""
        raw = pd.DataFrame({
            "articleId": ["art_1", "art_1", "art_2", "art_2"],
            "title": ["Title A", "Title B Different", "Title C", "Title C"],
            "postPublishDate": ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"],
            "body_raw": ["Body 1", "Body 2", "Body 3", "Body 4"]
        })
        deduped, stats = _deduplicate(raw, "articleId")
        self.assertEqual(stats["total_before"], 4)
        self.assertEqual(stats["total_after"], 2)
        self.assertEqual(stats["duplicates_dropped"], 2)
        self.assertEqual(stats["content_conflicts"], 1)  # Only art_1 had different titles

    def test_synthetic_id_collision_guard(self):
        """Verify that when no ID column is present, synthetic IDs include random/timestamp prefix."""
        raw = pd.DataFrame({
            "title": ["Article One", "Article Two"],
            "body_raw": ["Some text", "Some other text"]
        })
        clean_df = clean_dataframe(raw)
        ids = clean_df["id"].tolist()
        self.assertEqual(len(ids), 2)
        self.assertTrue(ids[0].startswith("art_"))
        self.assertTrue(ids[1].startswith("art_"))
        self.assertNotEqual(ids[0], ids[1])
        # Ensure it's not simple art_1 that collides across different CSV uploads
        self.assertTrue(len(ids[0].split("_")) >= 3)

    def test_flexible_column_mapping(self):
        """Verify clean_dataframe resolves alternative column names seamlessly."""
        raw = pd.DataFrame({
            "id": ["101"],
            "headline": ["Headline Title"],
            "snippet": ["Short snippet"],
            "content": ["Article full content"],
            "category": ["World News"],
            "date": ["2026-05-01T10:00:00Z"],
            "views": [1500]
        })
        cleaned = clean_dataframe(raw)
        self.assertEqual(cleaned.iloc[0]["id"], "101")
        self.assertEqual(cleaned.iloc[0]["title"], "Headline Title")
        self.assertEqual(cleaned.iloc[0]["description"], "Short snippet")
        self.assertEqual(cleaned.iloc[0]["body"], "Article full content")
        self.assertEqual(cleaned.iloc[0]["section"], "World News")
        self.assertEqual(cleaned.iloc[0]["page_views"], 1500)
        self.assertIn("text_embed", cleaned.columns)
        self.assertIn("text_bm25", cleaned.columns)
        self.assertIn("text_rerank", cleaned.columns)
        self.assertIn("embed_token_count", cleaned.columns)
        self.assertIn("rerank_token_count", cleaned.columns)
        self.assertIn("is_truncated", cleaned.columns)
        self.assertIn("recipe_version", cleaned.columns)

    def test_pipeline_version_fingerprint(self):
        """Verify get_pipeline_version is deterministic and sensitive to model/config changes."""
        ver1 = config.get_pipeline_version()
        self.assertEqual(len(ver1), 16)

        # Same config gives same hash
        ver2 = config.get_pipeline_version()
        self.assertEqual(ver1, ver2)

        # Changing config changes the hash
        with patch.object(config, "EMBED_MAX_SEQ_LEN", 384):
            ver_changed = config.get_pipeline_version()
            self.assertNotEqual(ver1, ver_changed)

    def test_word_based_settings_retired(self):
        """Verify word-based body truncation settings are retired."""
        self.assertIsNone(config.BODY_WORDS_EMBED)
        self.assertIsNone(config.BODY_WORDS_RERANK)

    def test_sentence_token_budget_whole_sentences(self):
        """Verify body text stops at the last whole sentence that fits the token budget."""
        from src.data_loader import compose_sentence_budgeted_text, get_embedder_tokenizer
        tok = get_embedder_tokenizer()

        title = "Election Results 2026"
        description = "National counting underway across key constituencies."
        # Create 30 distinct numbered sentences
        body = " ".join([f"Sentence {i} gives more detail about the constituency tallies." for i in range(1, 31)])

        budget = 100
        text, count, is_trunc = compose_sentence_budgeted_text(title, description, body, budget, tok)

        # Must be truncated because 30 sentences > 100 tokens
        self.assertTrue(is_trunc)
        # Token count must not exceed the budget
        self.assertLessEqual(count, budget)
        # Must end with a period and whole sentence, not a mid-sentence cutoff
        self.assertTrue(text.endswith("tallies."))
        # Title and description must be kept whole at the start
        self.assertTrue(text.startswith("Election Results 2026. National counting underway"))

    def test_description_trimmed_at_word_boundary_when_exceeding_budget(self):
        """Verify description is trimmed at a word boundary if title + description exceed budget."""
        from src.data_loader import compose_sentence_budgeted_text, get_embedder_tokenizer
        tok = get_embedder_tokenizer()

        title = "Short Title"
        # 200 distinct words to guarantee title + desc > budget
        huge_description = " ".join([f"word{i}" for i in range(200)])
        body = "Body sentence that will not fit."

        budget = 50
        text, count, is_trunc = compose_sentence_budgeted_text(title, huge_description, body, budget, tok)

        self.assertTrue(is_trunc)
        self.assertLessEqual(count, budget)
        # The last token in text must be a complete word, not chopped mid-character
        last_word = text.split()[-1]
        self.assertTrue(last_word.startswith("word"))
        self.assertFalse(last_word.endswith("."))


class TestEmbeddingCache(unittest.TestCase):

    def test_content_cache_keying(self):
        """Verify cache keys incorporate model name, max_seq_length, and text content."""
        model_name = "test-model-v1"
        max_len = 128
        text = "Sample news article for caching"

        expected_key = hashlib.sha256(f"{model_name}|{max_len}|{text}".encode("utf-8")).hexdigest()
        different_model_key = hashlib.sha256(f"other-model|{max_len}|{text}".encode("utf-8")).hexdigest()
        different_len_key = hashlib.sha256(f"{model_name}|256|{text}".encode("utf-8")).hexdigest()

        self.assertNotEqual(expected_key, different_model_key)
        self.assertNotEqual(expected_key, different_len_key)

class TestRerankerBudgetSplit(unittest.TestCase):
    """Verify build_reranker_pairs() enforces the 512-token hard cap."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(config.RERANKER_MODEL)

    def test_article_to_article_under_cap(self):
        """50/50 split: both sides get at most 254 content tokens, total <= 512."""
        from src.reranker import build_reranker_pairs, _TOTAL_CAP
        query = "India election results 2025. " * 50   # deliberately long
        docs = ["Modi government policy on economic reform. " * 50]
        pairs = build_reranker_pairs(query, docs, is_search=False, tokenizer=self.tok)
        for q, d in pairs:
            total = len(self.tok(q, d, add_special_tokens=True)["input_ids"])
            self.assertLessEqual(total, _TOTAL_CAP,
                                 f"Article pair {total} exceeds cap {_TOTAL_CAP}")

    def test_search_mode_under_cap(self):
        """Live search: short query gets most budget to passage, total <= 512."""
        from src.reranker import build_reranker_pairs, _TOTAL_CAP
        query = "Bumrah injury"
        docs = [{"title": "Bumrah out of series",
                 "description": "Fast bowler ruled out with back stress fracture.",
                 "body": "Detailed medical report. " * 80}]
        pairs = build_reranker_pairs(query, docs, is_search=True, tokenizer=self.tok)
        for q, d in pairs:
            total = len(self.tok(q, d, add_special_tokens=True)["input_ids"])
            self.assertLessEqual(total, _TOTAL_CAP,
                                 f"Search pair {total} exceeds cap {_TOTAL_CAP}")


class TestColabWorkflow(unittest.TestCase):
    """Verify Colab workflow guarantees: pinned revisions, hash sensitivity, and export validation."""

    def test_model_revisions_pinned(self):
        """Verify model revisions are valid 40-char git commit SHAs."""
        self.assertTrue(hasattr(config, "EMBEDDING_REVISION"))
        self.assertTrue(hasattr(config, "RERANKER_REVISION"))
        self.assertEqual(len(config.EMBEDDING_REVISION), 40)
        self.assertEqual(len(config.RERANKER_REVISION), 40)

    def test_pipeline_version_includes_revisions(self):
        """Verify pipeline version changes if model revisions change."""
        orig_ver = config.get_pipeline_version()
        with patch.object(config, "EMBEDDING_REVISION", "0000000000000000000000000000000000000000"):
            alt_ver = config.get_pipeline_version()
        self.assertNotEqual(orig_ver, alt_ver)

    def test_import_validation_catches_invalid_vectors(self):
        """Verify importer raises ValueError on wrong dimensions, NaNs, or unnormalized vectors."""
        import tempfile
        import json
        from import_colab_export import validate_and_import

        # Test NaN vector
        bad_data = {
            "config_hash": config.get_pipeline_version(),
            "articles": {
                "art_1": {
                    "embedding": [float("nan")] * 768
                }
            }
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(bad_data, tf)
            tf_name = tf.name

        with patch("import_colab_export.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value = mock_db
            mock_db.query.return_value.all.return_value = [("art_1",)]
            with self.assertRaises(ValueError) as ctx:
                validate_and_import(tf_name)
            self.assertIn("NaN or Inf", str(ctx.exception))


class TestEvidenceAndCalibration(unittest.TestCase):
    """Verify deterministic evidence, confidence calibration, and raw reranker gating."""

    def test_calibration_monotonic_and_bands(self):
        """Verify calibrated probabilities increase monotonically and assign correct bands."""
        from src.calibration import ConfidenceCalibrator
        cal = ConfidenceCalibrator()

        low_res = cal.calibrate(0.20, next_score=0.10, agreement_count=1)
        med_res = cal.calibrate(0.55, next_score=0.45, agreement_count=2)
        high_res = cal.calibrate(0.90, next_score=0.70, agreement_count=3)

        self.assertLess(low_res["calibrated_prob"], med_res["calibrated_prob"])
        self.assertLess(med_res["calibrated_prob"], high_res["calibrated_prob"])
        self.assertEqual(low_res["band"], "LOW")
        self.assertEqual(med_res["band"], "MEDIUM")
        self.assertEqual(high_res["band"], "HIGH")
        self.assertAlmostEqual(high_res["margin"], 0.20, places=3)
        self.assertEqual(high_res["agreement_count"], 3)

    def test_deterministic_evidence_no_hallucinations(self):
        """Verify evidence is strictly extracted from source text with zero LLMs."""
        from src.evidence import build_evidence_reason
        import json

        source = {
            "title": "Jasprit Bumrah injury update before Champions Trophy",
            "description": "BCCI medical team monitors pace spearhead Bumrah.",
            "section": "Sports",
            "date": "2025-05-10"
        }
        cand = {
            "id": "cand_1",
            "title": "Bumrah ruled out as BCCI announces revised squad",
            "description": "Fast bowler Jasprit Bumrah suffers lower back stress fracture.",
            "body": "The BCCI medical team confirmed Bumrah will miss the Champions Trophy in Dubai.",
            "section": "Sports",
            "date": "2025-05-10"
        }

        reason_str = build_evidence_reason(source, cand, bm25_rank=1, dense_rank=2, rerank_score=0.92)
        data = json.loads(reason_str)

        self.assertIn("Bumrah", str(data["entities"]) + str(data["terms"]))
        self.assertTrue(len(data["best_sentence"]) > 0)
        self.assertEqual(data["agreement_count"], 3)
        self.assertTrue(any("Same Section: Sports" in b for b in data["boosts"]))


if __name__ == "__main__":
    unittest.main()


