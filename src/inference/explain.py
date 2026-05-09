"""Per-prediction explanations.

- TF-IDF: per-class top n-grams ranked by `coefficient * tfidf_value`. Fully transparent.
- DistilBERT: token attributions via `transformers-interpret`'s SequenceClassificationExplainer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.paths import ID2LABEL


@dataclass
class TokenAttribution:
    token: str
    score: float


def explain_tfidf(text: str, *, top_k: int = 15) -> dict[str, list[TokenAttribution]]:
    """Returns per-class top contributing n-grams from the fitted TF-IDF + LogReg pipeline."""
    from src.inference.classify import _load_tfidf

    pipe = _load_tfidf()
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    feats = np.array(vec.get_feature_names_out())

    x = vec.transform([text])  # (1, n_features) sparse
    x_dense = np.asarray(x.todense()).ravel()
    nonzero = np.nonzero(x_dense)[0]
    if nonzero.size == 0:
        return {label: [] for label in ID2LABEL.values()}

    out: dict[str, list[TokenAttribution]] = {}
    for class_idx, label in ID2LABEL.items():
        coefs = clf.coef_[class_idx] if clf.coef_.shape[0] > 1 else clf.coef_[0]
        contributions = coefs[nonzero] * x_dense[nonzero]
        order = np.argsort(contributions)[::-1][:top_k]
        out[label] = [
            TokenAttribution(token=str(feats[nonzero[i]]), score=float(contributions[i]))
            for i in order
        ]
    return out


def _explain_hf(loader, text: str, max_tokens: int) -> list[TokenAttribution]:
    from transformers_interpret import SequenceClassificationExplainer

    from src.data.clean import clean_for_modeling

    tok, model, _device = loader()
    explainer = SequenceClassificationExplainer(model, tok)

    cleaned = clean_for_modeling(text)
    encoded = tok(cleaned, truncation=True, max_length=max_tokens, return_tensors=None)
    truncated = tok.decode(encoded["input_ids"], skip_special_tokens=True)

    attributions = explainer(truncated)
    return [TokenAttribution(token=t, score=float(s)) for t, s in attributions]


def explain_distilbert(text: str, *, max_tokens: int = 80) -> list[TokenAttribution]:
    """Token-level attributions via integrated gradients (DistilBERT)."""
    from src.inference.classify import _load_distilbert

    return _explain_hf(_load_distilbert, text, max_tokens)


def explain_roberta(text: str, *, max_tokens: int = 80) -> list[TokenAttribution]:
    """Token-level attributions via integrated gradients (RoBERTa)."""
    from src.inference.classify import _load_roberta

    return _explain_hf(_load_roberta, text, max_tokens)
