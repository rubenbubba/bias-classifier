"""Sanity tests for the inference layer. Skipped if a model isn't trained yet."""
from __future__ import annotations

import pytest

from src.paths import TFIDF_DIR


@pytest.mark.skipif(
    not (TFIDF_DIR / "pipeline.joblib").exists(),
    reason="TF-IDF model not trained yet; run python -m src.models.tfidf_baseline",
)
def test_predict_text_returns_valid_label():
    from src.inference.classify import predict_text
    from src.paths import LABELS

    pred = predict_text(
        "The senator argued that increased government spending is necessary to support working families "
        "and called for stronger labor protections in the next budget cycle.",
        backend="tfidf",
    )
    assert pred.label in LABELS
    assert abs(sum(pred.probs.values()) - 1.0) < 1e-3
    assert all(0 <= p <= 1 for p in pred.probs.values())
