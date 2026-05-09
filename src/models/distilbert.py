"""Fine-tune distilbert-base-uncased on the article-level bias splits.

    python -m src.models.distilbert
    python -m src.models.distilbert --epochs 4 --batch-size 8

Saves the best checkpoint (by val macro-F1) to models/distilbert/.
GPU strongly recommended; the script will fall back to CPU but training will be slow.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data.clean import clean_for_modeling
from src.paths import (
    DISTILBERT_DIR,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    TEST_PARQUET,
    TRAIN_PARQUET,
    VAL_PARQUET,
    ensure_dirs,
)

MODEL_NAME = "distilbert-base-uncased"


def _load(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[df["text"].str.len() > 0].reset_index(drop=True)


def _to_hf_ds(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(
        df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
        preserve_index=False,
    )


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    ensure_dirs()
    train = _load(TRAIN_PARQUET)
    val = _load(VAL_PARQUET)
    test = _load(TEST_PARQUET)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        cleaned = [clean_for_modeling(t) for t in batch["text"]]
        return tokenizer(cleaned, truncation=True, max_length=args.max_length)

    train_ds = _to_hf_ds(train).map(tokenize, batched=True, remove_columns=["text"])
    val_ds = _to_hf_ds(val).map(tokenize, batched=True, remove_columns=["text"])
    test_ds = _to_hf_ds(test).map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    use_fp16 = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=str(DISTILBERT_DIR / "_runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=use_fp16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )

    trainer.train()

    # Final eval on the held-out test split.
    test_logits = trainer.predict(test_ds)
    preds = np.argmax(test_logits.predictions, axis=-1)
    print("\n=== test ===")
    print(
        classification_report(
            test_logits.label_ids, preds, target_names=LABELS, digits=4, zero_division=0
        )
    )

    trainer.save_model(str(DISTILBERT_DIR))
    tokenizer.save_pretrained(str(DISTILBERT_DIR))
    print(f"\nSaved best model -> {DISTILBERT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
