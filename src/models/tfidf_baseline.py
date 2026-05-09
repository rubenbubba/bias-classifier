"""TF-IDF + Logistic Regression baseline.

Train and evaluate:
    python -m src.models.tfidf_baseline                  # media split (default)
    python -m src.models.tfidf_baseline --split random   # random split

Saves the fitted pipeline to models/tfidf/pipeline.joblib (media) or
models/tfidf_random/pipeline.joblib (random).
"""
from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from src.data.clean import clean_for_modeling
from src.paths import LABELS, ensure_dirs, parquet_paths, tfidf_dir


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_for_modeling,
                    ngram_range=(1, 2),
                    max_features=50_000,
                    min_df=5,
                    max_df=0.9,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _load(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[df["text"].str.len() > 0].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["media", "random"], default="media")
    args = parser.parse_args()

    ensure_dirs()
    train_path, val_path, test_path = parquet_paths(args.split)
    train = _load(train_path)
    val = _load(val_path)
    test = _load(test_path)
    print(f"split={args.split} train={len(train)} val={len(val)} test={len(test)}")

    pipe = build_pipeline()
    pipe.fit(train["text"], train["label_id"])

    for name, df in [("val", val), ("test", test)]:
        preds = pipe.predict(df["text"])
        print(f"\n=== {name} ===")
        print(
            classification_report(
                df["label_id"], preds, target_names=LABELS, digits=4, zero_division=0
            )
        )
        cm = confusion_matrix(df["label_id"], preds, labels=list(range(len(LABELS))))
        print("confusion matrix (rows=true, cols=pred):")
        print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())

    out_dir = tfidf_dir(args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = out_dir / "pipeline.joblib"
    joblib.dump(pipe, pipeline_path)
    print(f"\nSaved pipeline -> {pipeline_path}")

    vec: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    feats = np.array(vec.get_feature_names_out())
    print("\nTop 15 features per class:")
    for class_idx, label in enumerate(LABELS):
        if clf.coef_.shape[0] == 1:
            coefs = clf.coef_[0]
            sign = 1 if class_idx == 1 else -1
            top = np.argsort(sign * coefs)[-15:][::-1]
        else:
            top = np.argsort(clf.coef_[class_idx])[-15:][::-1]
        print(f"  {label}: {', '.join(feats[top])}")


if __name__ == "__main__":
    sys.exit(main())
