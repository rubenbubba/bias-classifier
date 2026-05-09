"""Streamlit UI: paste a URL, see bias + credibility + explanation.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.classify import predict_url  # noqa: E402
from src.inference.explain import explain_tfidf  # noqa: E402
from src.paths import DISTILBERT_DIR, LABELS, TFIDF_DIR, roberta_dir  # noqa: E402

ROBERTA_MEDIA_DIR = roberta_dir("media")

st.set_page_config(page_title="News Bias & Credibility Classifier", layout="wide")
st.title("News Bias & Credibility Classifier")
st.caption(
    "Trained article-level bias classifier + outlet-level credibility lookup. "
    "Paste a news URL to score it."
)

tfidf_ready = (TFIDF_DIR / "pipeline.joblib").exists()
distilbert_ready = (DISTILBERT_DIR / "config.json").exists()
roberta_ready = (ROBERTA_MEDIA_DIR / "config.json").exists()

available_backends = []
if tfidf_ready:
    available_backends.append("tfidf")
if roberta_ready:
    available_backends.append("roberta")
if distilbert_ready:
    available_backends.append("distilbert")
if tfidf_ready and roberta_ready:
    available_backends.append("ensemble")

with st.sidebar:
    st.header("Settings")
    if not available_backends:
        st.error(
            "No trained models found. Run `python -m src.models.tfidf_baseline` "
            "and/or `python -m src.models.transformer` first."
        )
        backend = "tfidf"
    else:
        backend = st.radio(
            "Model",
            available_backends,
            help=(
                "tfidf: transparent baseline. "
                "roberta: stronger transformer. "
                "ensemble: averages tfidf + roberta probabilities."
            ),
        )

    st.markdown(
        "**Bias** comes from the trained classifier on article text. "
        "**Credibility** comes from a static AllSides + MBFC lookup keyed by the outlet's domain. "
        "Unknown domains return `unknown` rather than a guess."
    )

url = st.text_input(
    "Article URL",
    placeholder="https://www.example.com/news/some-article",
)
go = st.button("Classify", type="primary", disabled=not url or not available_backends)

if go and url and available_backends:
    with st.spinner("Scraping article..."):
        try:
            result = predict_url(url, backend=backend)  # type: ignore[arg-type]
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

    article = result.article
    bias = result.bias
    cred = result.credibility

    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader(article.title or "(no title)")
        meta_parts = [article.domain]
        if article.authors:
            meta_parts.append("by " + ", ".join(article.authors))
        if article.publish_date:
            meta_parts.append(article.publish_date)
        st.caption(" — ".join(meta_parts))

        with st.expander("Extracted article text", expanded=False):
            st.write(article.text)

    with right_col:
        st.markdown("### Bias prediction")
        st.metric("Predicted lean", bias.label.upper())
        prob_df = pd.DataFrame(
            {"label": LABELS, "probability": [bias.probs.get(l, 0.0) for l in LABELS]}
        )
        st.bar_chart(prob_df.set_index("label"))

        st.markdown("### Credibility (outlet-level)")
        if cred.source == "unknown":
            st.warning(f"No rating on file for `{cred.domain}`.")
        else:
            st.metric("Credibility", (cred.credibility or "n/a").upper())
            st.write(
                {
                    "outlet": cred.outlet,
                    "outlet bias (rated)": cred.bias_outlet,
                    "source": cred.source,
                }
            )

        if cred.bias_outlet and cred.bias_outlet != bias.label:
            st.info(
                f"Article reads as **{bias.label}**, but the outlet is rated "
                f"**{cred.bias_outlet}** overall. Worth reading critically."
            )

    st.markdown("---")
    st.markdown("### Why this prediction?")

    if backend in ("tfidf", "ensemble"):
        attributions = explain_tfidf(article.text, top_k=15)
        st.markdown("**TF-IDF n-gram contributions (per class)**")
        cols = st.columns(len(LABELS))
        for col, label in zip(cols, LABELS):
            with col:
                st.markdown(f"**{label.upper()}**")
                items = attributions.get(label, [])
                if not items:
                    st.write("_no contributing features_")
                else:
                    st.dataframe(
                        pd.DataFrame(
                            [(a.token, a.score) for a in items],
                            columns=["n-gram", "contribution"],
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )

    if backend in ("roberta", "ensemble", "distilbert"):
        try:
            if backend == "distilbert":
                from src.inference.explain import explain_distilbert

                explain_fn = explain_distilbert
                label = "DistilBERT"
            else:
                from src.inference.explain import explain_roberta

                explain_fn = explain_roberta
                label = "RoBERTa"

            with st.spinner(f"Computing {label} token attributions (slow)..."):
                attributions = explain_fn(article.text, max_tokens=80)
            st.markdown(f"**{label} integrated-gradients attributions**")
            st.dataframe(
                pd.DataFrame(
                    [(a.token, a.score) for a in attributions],
                    columns=["token", "attribution"],
                ),
                hide_index=True,
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Explanation unavailable: {e}")
