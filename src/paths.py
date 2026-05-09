"""Centralized filesystem paths so every module agrees on layout."""
from __future__ import annotations
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

ARTICLE_BIAS_DIR = RAW / "article-bias-prediction"
ALLSIDES_CSV = RAW / "allsides.csv"
MBFC_TSV = RAW / "mbfc_acl2018.tsv"

# Default (media split) parquet paths — kept for backward compatibility.
TRAIN_PARQUET = PROCESSED / "train.parquet"
VAL_PARQUET = PROCESSED / "val.parquet"
TEST_PARQUET = PROCESSED / "test.parquet"
CREDIBILITY_CSV = DATA / "credibility_lookup.csv"

MODELS = ROOT / "models"
TFIDF_DIR = MODELS / "tfidf"           # legacy: media-split TF-IDF
DISTILBERT_DIR = MODELS / "distilbert" # legacy: media-split DistilBERT

LABELS = ["left", "center", "right"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

Split = Literal["media", "random"]


def parquet_paths(split: Split = "media") -> tuple[Path, Path, Path]:
    """Return (train, val, test) parquet paths for the requested split.

    Media split keeps the legacy filenames so existing models still load.
    Random split uses a `_random` suffix.
    """
    if split == "media":
        return TRAIN_PARQUET, VAL_PARQUET, TEST_PARQUET
    suffix = "_random"
    return (
        PROCESSED / f"train{suffix}.parquet",
        PROCESSED / f"val{suffix}.parquet",
        PROCESSED / f"test{suffix}.parquet",
    )


def tfidf_dir(split: Split = "media") -> Path:
    return TFIDF_DIR if split == "media" else MODELS / "tfidf_random"


def roberta_dir(split: Split = "media") -> Path:
    return MODELS / f"roberta_{split}"


def ensure_dirs() -> None:
    for p in (RAW, PROCESSED, MODELS, TFIDF_DIR, DISTILBERT_DIR):
        p.mkdir(parents=True, exist_ok=True)
