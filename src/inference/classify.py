"""End-to-end: URL -> bias prediction + outlet credibility lookup.

Backends:
- "tfidf"      -> models/tfidf/pipeline.joblib (sklearn Pipeline)
- "distilbert" -> models/distilbert/ (HF model dir, legacy)
- "roberta"    -> models/roberta_media/ (HF model dir)
- "ensemble"   -> averages tfidf + roberta probabilities

All expose the same predict_text(text) -> BiasPrediction interface.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from src.data.clean import clean_for_modeling
from src.inference.scrape import ScrapedArticle, scrape
from src.paths import (
    CREDIBILITY_CSV,
    DISTILBERT_DIR,
    ID2LABEL,
    LABELS,
    TFIDF_DIR,
    roberta_dir,
)

Backend = Literal["tfidf", "distilbert", "roberta", "ensemble"]
TFIDF_PIPELINE = TFIDF_DIR / "pipeline.joblib"
ROBERTA_DIR = roberta_dir("media")


@dataclass
class BiasPrediction:
    label: str
    probs: dict[str, float]


@dataclass
class CredibilityInfo:
    domain: str
    outlet: str | None
    bias_outlet: str | None
    credibility: str | None
    source: str


@dataclass
class ClassificationResult:
    article: ScrapedArticle
    bias: BiasPrediction
    credibility: CredibilityInfo
    backend: Backend

    def to_dict(self) -> dict:
        return {
            "article": {
                "url": self.article.url,
                "domain": self.article.domain,
                "title": self.article.title,
                "text": self.article.text,
                "authors": self.article.authors,
                "publish_date": self.article.publish_date,
            },
            "bias": asdict(self.bias),
            "credibility": asdict(self.credibility),
            "backend": self.backend,
        }


@lru_cache(maxsize=1)
def _credibility_table() -> pd.DataFrame:
    if not CREDIBILITY_CSV.exists():
        return pd.DataFrame(columns=["domain", "outlet", "bias_outlet", "credibility", "source"])
    df = pd.read_csv(CREDIBILITY_CSV)
    df["domain"] = df["domain"].astype(str).str.lower()
    return df.set_index("domain", drop=False)


def _lookup_credibility(domain: str) -> CredibilityInfo:
    df = _credibility_table()
    domain = (domain or "").lower()
    if domain in df.index:
        row = df.loc[domain]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return CredibilityInfo(
            domain=domain,
            outlet=row.get("outlet") if pd.notna(row.get("outlet")) else None,
            bias_outlet=row.get("bias_outlet") if pd.notna(row.get("bias_outlet")) else None,
            credibility=row.get("credibility") if pd.notna(row.get("credibility")) else None,
            source=str(row.get("source") or "unknown"),
        )
    return CredibilityInfo(
        domain=domain, outlet=None, bias_outlet=None, credibility=None, source="unknown"
    )


@lru_cache(maxsize=1)
def _load_tfidf():
    if not TFIDF_PIPELINE.exists():
        raise FileNotFoundError(
            f"TF-IDF model not found at {TFIDF_PIPELINE}. "
            "Run: python -m src.models.tfidf_baseline"
        )
    return joblib.load(TFIDF_PIPELINE)


def _load_hf_model(model_dir: Path):
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"HuggingFace model not found at {model_dir}. "
            "Train first or unzip the trained model into this directory."
        )
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tok, model, device


@lru_cache(maxsize=1)
def _load_distilbert():
    return _load_hf_model(DISTILBERT_DIR)


@lru_cache(maxsize=1)
def _load_roberta():
    return _load_hf_model(ROBERTA_DIR)


def _predict_tfidf(text: str) -> np.ndarray:
    pipe = _load_tfidf()
    return pipe.predict_proba([text])[0]


def _predict_hf(loader, text: str) -> np.ndarray:
    import torch

    tok, model, device = loader()
    enc = tok(
        clean_for_modeling(text),
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    return torch.softmax(logits, dim=-1).cpu().numpy()


def _to_prediction(probs: np.ndarray) -> BiasPrediction:
    label_id = int(np.argmax(probs))
    return BiasPrediction(
        label=ID2LABEL[label_id],
        probs={ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
    )


def predict_text(text: str, *, backend: Backend = "tfidf") -> BiasPrediction:
    if backend == "tfidf":
        return _to_prediction(_predict_tfidf(text))
    if backend == "distilbert":
        return _to_prediction(_predict_hf(_load_distilbert, text))
    if backend == "roberta":
        return _to_prediction(_predict_hf(_load_roberta, text))
    if backend == "ensemble":
        tfidf_p = _predict_tfidf(text)
        roberta_p = _predict_hf(_load_roberta, text)
        # Weighted toward TF-IDF: it outperforms RoBERTa on the media-split test.
        return _to_prediction(0.65 * tfidf_p + 0.35 * roberta_p)
    raise ValueError(f"Unknown backend: {backend}")


def predict_url(url: str, *, backend: Backend = "tfidf") -> ClassificationResult:
    article = scrape(url)
    bias = predict_text(article.text, backend=backend)
    cred = _lookup_credibility(article.domain)
    return ClassificationResult(article=article, bias=bias, credibility=cred, backend=backend)


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m src.inference.classify <url> [tfidf|distilbert|roberta|ensemble]")
        sys.exit(1)
    url = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 else "tfidf"
    result = predict_url(url, backend=backend)
    out = result.to_dict()
    out["article"]["text"] = out["article"]["text"][:500] + "..."
    print(json.dumps(out, indent=2, default=str))
