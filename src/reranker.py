"""
Step 3 - The reranker (cross-encoder).

Search engines compare the query and each article separately.
A cross-encoder reads the query and one article TOGETHER and scores how
well they match, which is slower but much more precise. We only run it
on the ~30 best candidates, so it stays fast.
"""
import numpy as np

import config


class Reranker:
    def __init__(self, model_name=config.RERANKER_MODEL):
        from sentence_transformers import CrossEncoder   # imported here so light mode works without it
        from torch import nn

        self.model = CrossEncoder(model_name, max_length=512)
        self._identity = nn.Identity()

    def score(self, query: str, documents: list) -> np.ndarray:
        """Return a relevance probability (0-1) for each document."""
        pairs = [(query, doc) for doc in documents]
        logits = np.asarray(self.model.predict(pairs, activation_fn=self._identity), dtype=float)
        return 1.0 / (1.0 + np.exp(-logits))
