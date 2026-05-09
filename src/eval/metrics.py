"""Cross-dataset evaluation: run a trained model on BABE + SemEval to gauge generalization.

    python -m src.eval.metrics tfidf
    python -m src.eval.metrics distilbert

BABE has sentence-level binary `biased/not` labels — we map our 3-class output to
{biased = left|right, not_biased = center} for a rough comparison. SemEval's
by-article split is binary hyperpartisan vs. not, treated similarly.
"""
from __future__ import annotations

import sys
from typing import Literal

from datasets import load_dataset
from sklearn.metrics import classification_report

from src.inference.classify import predict_text

Backend = Literal["tfidf", "distilbert", "roberta", "ensemble"]


def _3class_to_binary(label: str) -> int:
    """left/right -> 1 (biased), center -> 0 (not biased)."""
    return 0 if label == "center" else 1


def eval_babe(backend: Backend) -> None:
    print("=== BABE (sentence-level biased/not) ===")
    try:
        ds = load_dataset("mediabiasgroup/BABE", split="test")
    except Exception as e:
        print(f"  [skip] failed to load BABE: {e}")
        return

    text_col = "text" if "text" in ds.column_names else ds.column_names[0]
    label_col = next(
        (c for c in ds.column_names if c.lower() in ("label", "labels", "label_bias")),
        None,
    )
    if label_col is None:
        print(f"  [skip] no label column found in {ds.column_names}")
        return

    y_true, y_pred = [], []
    for row in ds:
        # BABE labels: 0 = "Non-biased", 1 = "Biased". We accept ints or strings.
        raw = row[label_col]
        gold = 1 if (str(raw).lower().startswith("bias") or raw == 1) else 0
        pred_label = predict_text(row[text_col], backend=backend).label
        y_true.append(gold)
        y_pred.append(_3class_to_binary(pred_label))

    print(
        classification_report(
            y_true, y_pred, target_names=["not_biased", "biased"], digits=4, zero_division=0
        )
    )


def eval_semeval(backend: Backend) -> None:
    print("=== SemEval 2019 hyperpartisan (by-article) ===")
    try:
        ds = load_dataset(
            "SemEvalWorkshop/hyperpartisan_news_detection", "byarticle", split="train"
        )
    except Exception as e:
        print(f"  [skip] failed to load SemEval: {e}")
        return

    text_col = "text" if "text" in ds.column_names else "article"
    label_col = "hyperpartisan" if "hyperpartisan" in ds.column_names else "label"

    y_true, y_pred = [], []
    for row in ds:
        gold = 1 if bool(row[label_col]) else 0
        pred_label = predict_text(row[text_col], backend=backend).label
        y_true.append(gold)
        y_pred.append(_3class_to_binary(pred_label))

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["not_hyperpartisan", "hyperpartisan"],
            digits=4,
            zero_division=0,
        )
    )


def main() -> None:
    valid = ("tfidf", "distilbert", "roberta", "ensemble")
    if len(sys.argv) < 2 or sys.argv[1] not in valid:
        print(f"usage: python -m src.eval.metrics [{' | '.join(valid)}]")
        sys.exit(1)
    backend: Backend = sys.argv[1]  # type: ignore[assignment]
    eval_babe(backend)
    print()
    eval_semeval(backend)


if __name__ == "__main__":
    main()
