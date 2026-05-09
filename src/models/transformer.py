"""Generalized transformer fine-tuner for the bias classification task.

Defaults to roberta-base on the media-based split, with strong regularization
and head+tail truncation. Same script handles both splits and both base models.

    python -m src.models.transformer                                # roberta-base, media split
    python -m src.models.transformer --split random                 # roberta-base, random split
    python -m src.models.transformer --model distilbert-base-uncased

Saves to models/{model_short_name}_{split}/, e.g. models/roberta_media/.

GPU strongly recommended; CPU will work but takes hours. Designed primarily for
the Colab notebook, which uses the same logic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data.clean import clean_for_modeling
from src.paths import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    MODELS,
    ensure_dirs,
    parquet_paths,
    roberta_dir,
)


def _short_name(model: str) -> str:
    """roberta-base -> roberta, distilbert-base-uncased -> distilbert."""
    return model.split("/")[-1].split("-")[0].lower()


def _output_dir(model: str, split: str) -> Path:
    short = _short_name(model)
    if short == "roberta":
        return roberta_dir(split)
    return MODELS / f"{short}_{split}"


def _load(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[df["text"].str.len() > 0].reset_index(drop=True)


def _to_hf_ds(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(
        df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
        preserve_index=False,
    )


def _head_tail_tokenize(texts, tokenizer, max_length: int):
    """Take first half + last half of tokens; bias signals appear at lead AND conclusion."""
    half = max_length // 2
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id or 0

    input_ids_batch = []
    attention_batch = []
    for t in texts:
        enc = tokenizer(t, add_special_tokens=False, truncation=False)
        ids = enc["input_ids"]
        if len(ids) <= max_length - 2:
            chosen = ids
        else:
            chosen = ids[: half - 1] + ids[-(half - 1) :]
        # Re-add CLS/SEP appropriate for the model.
        if cls_id is not None and sep_id is not None:
            ids_full = [cls_id] + chosen + [sep_id]
        else:
            ids_full = chosen
        attn = [1] * len(ids_full)
        input_ids_batch.append(ids_full)
        attention_batch.append(attn)
    return {"input_ids": input_ids_batch, "attention_mask": attention_batch}


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--split", choices=["media", "random"], default="media")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--truncation",
        choices=["head", "head_tail"],
        default="head_tail",
        help="head_tail keeps lead+conclusion of long articles.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=1)
    args = parser.parse_args()

    ensure_dirs()
    out_dir = _output_dir(args.model, args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path, val_path, test_path = parquet_paths(args.split)
    train = _load(train_path)
    val = _load(val_path)
    test = _load(test_path)
    print(
        f"model={args.model} split={args.split} train={len(train)} val={len(val)} test={len(test)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize_batch(batch):
        cleaned = [clean_for_modeling(t) for t in batch["text"]]
        if args.truncation == "head":
            return tokenizer(cleaned, truncation=True, max_length=args.max_length)
        return _head_tail_tokenize(cleaned, tokenizer, args.max_length)

    train_ds = _to_hf_ds(train).map(tokenize_batch, batched=True, remove_columns=["text"])
    val_ds = _to_hf_ds(val).map(tokenize_batch, batched=True, remove_columns=["text"])
    test_ds = _to_hf_ds(test).map(tokenize_batch, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    use_fp16 = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=str(out_dir / "_runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    test_out = trainer.predict(test_ds)
    preds = np.argmax(test_out.predictions, axis=-1)
    print(f"\n=== test ({args.split}) ===")
    print(
        classification_report(
            test_out.label_ids, preds, target_names=LABELS, digits=4, zero_division=0
        )
    )
    cm = confusion_matrix(test_out.label_ids, preds, labels=list(range(len(LABELS))))
    print("confusion matrix:")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\nSaved best model -> {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
