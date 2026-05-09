# News Bias & Credibility Classifier

A text-classification project that scores news articles on the political-bias axis (**left / center / right**) and looks up the publishing outlet's **credibility rating**, returning both — with explanations — through a Streamlit UI.

The trained model is the project's contribution: it is more **transparent and explainable** than calling an LLM. Three classifiers are provided side-by-side so the user can compare:

- **TF-IDF + Logistic Regression** — fast, fully transparent (per-class top n-grams act as direct explanations).
- **Fine-tuned RoBERTa-base** — stronger on familiar outlets, weaker on unseen ones; explanations via integrated gradients.
- **Weighted ensemble** (0.65 TF-IDF + 0.35 RoBERTa) — averages softmax probabilities from both.

Credibility is **not** trained. It comes from a static lookup table built from AllSides + MBFC outlet ratings, joined on domain.

## Quickstart

```bash
pip install -r requirements.txt

# Data + lookup
python -m src.data.download
python -m src.data.preprocess                 # media-based (outlet-disjoint) split, the headline split
python -m src.data.preprocess --split random  # random split, for the comparison number
python -m src.data.build_credibility_lookup

# Train
python -m src.models.tfidf_baseline                # ~3 min on CPU
python -m src.models.tfidf_baseline --split random # ~3 min on CPU

# RoBERTa is GPU-only in practice; we trained on Colab.
# See notebooks/train_roberta_colab.ipynb — runs both splits in ~60 min on a free T4.
# Unzip the resulting roberta_media.zip and roberta_random.zip into models/.

# Run the UI
streamlit run app/streamlit_app.py
```

## Datasets

| Dataset | Purpose | Size | Source |
|---|---|---|---|
| Article Bias Prediction (Baly et al., 2020) | Bias training (left/center/right) | 34,737 articles | https://github.com/ramybaly/Article-Bias-Prediction |
| BABE | Cross-domain eval (binary bias) | 4,121 sentences | `mediabiasgroup/BABE` (HF) |
| SemEval 2019 Hyperpartisan | Cross-domain eval (binary hyperpartisan) | 645 by-article | `SemEvalWorkshop/hyperpartisan_news_detection` (HF) |
| AllSides ratings | Outlet-level bias lookup | ~547 outlets | https://raw.githubusercontent.com/favstats/AllSideR/master/data/allsides_data.csv |
| MBFC ratings (scraped) | Outlet-level credibility lookup | ~1.5K outlets | https://github.com/ramybaly/News-Media-Reliability |

## Splits

The Article Bias Prediction dataset provides two evaluation splits, both used here:

- **Media-based (outlet-disjoint)**: outlets in test ≠ outlets in train. The headline number for the writeup, because it measures whether the model has learned bias signals that generalize to unseen outlets.
- **Random**: outlets shared between train and test. Easier; mostly a sanity check that the task is learnable at all.

The gap between the two scores is the project's central finding.

## Results (test set, macro F1)

| Model | Random split | Media split |
|---|---|---|
| Random chance | 0.33 | 0.33 |
| **TF-IDF + LogReg** | (run locally) | **0.567** |
| DistilBERT (initial run) | — | 0.376 |
| RoBERTa-base | **0.796** | 0.365 |
| Ensemble (65/35 TF-IDF/RoBERTa, naive) | — | 0.398 |

Counterintuitive headline: TF-IDF beats RoBERTa on the outlet-disjoint split. The transformer overfits to outlet-style features that don't transfer.

## Project layout

```
src/
  data/        download.py, preprocess.py, build_credibility_lookup.py, clean.py
  models/      tfidf_baseline.py, distilbert.py, transformer.py
  inference/   scrape.py, classify.py, explain.py
  eval/        metrics.py
app/
  streamlit_app.py
notebooks/
  train_distilbert_colab.ipynb   (initial DistilBERT run)
  train_roberta_colab.ipynb      (final two-split RoBERTa run)
tests/
  test_scrape.py, test_classify.py, demo_urls.txt
```

## Notes and caveats

- **English-only**: all training data is English (US political press). Pasting a non-English URL produces meaningless output.
- **Paywalled outlets** (e.g. NYT, WSJ) block scrapers; the UI surfaces a clear error in those cases. Reuters, AP, Fox, MSNBC, CNN, Politico, Daily Wire, etc. scrape fine.
- **Unknown domains**: credibility lookup returns `unknown` rather than guessing. Bias is still predicted, since that comes from the article text.
- **Cleaner**: a small regex pass (`src/data/clean.py`) strips outlet brand names ("cnn", "fox news") and site chrome ("story highlights", "more videos must watch") before tokenization. Honorifics and titles are deliberately preserved — "President Trump" vs "Trump" carries bias signal.
