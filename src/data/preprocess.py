"""Read the Article-Bias-Prediction repo dump and produce clean parquet splits.

    python -m src.data.preprocess                # media-based split (default)
    python -m src.data.preprocess --split random # random split

Output columns: text, label, label_id, outlet, domain, article_id.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import tldextract
from tqdm import tqdm

from src.paths import (
    ARTICLE_BIAS_DIR,
    LABEL2ID,
    LABELS,
    PROCESSED,
    ensure_dirs,
    parquet_paths,
)

WHITESPACE_RE = re.compile(r"\s+")


def _find_jsons_dir(repo: Path) -> Path:
    """The articles may be loose JSON files or packed in an archive. Materialize them."""
    direct = repo / "data" / "jsons"
    if direct.is_dir() and any(direct.glob("*.json")):
        return direct

    # Search for an archive named like jsons.* and extract it.
    for archive in repo.rglob("jsons.*"):
        if archive.suffix in {".bz2", ".gz", ".xz"} or archive.name.endswith(
            (".tar.bz2", ".tar.gz", ".tar.xz")
        ):
            print(f"[extract] {archive} -> {archive.parent}")
            with tarfile.open(archive) as tf:
                tf.extractall(archive.parent)
        elif archive.suffix == ".zip":
            print(f"[extract] {archive} -> {archive.parent}")
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(archive.parent)

    if direct.is_dir() and any(direct.glob("*.json")):
        return direct

    raise FileNotFoundError(
        f"Could not locate article JSON files. Looked under {repo}/data/jsons. "
        "If the dataset shipped as an archive, extract it manually."
    )


def _find_split_file(repo: Path, split_kind: str, fold: str) -> Path:
    """Locate a split file like data/splits/{media,random}/{train,valid,test}.{tsv,csv}."""
    candidates = [
        repo / "data" / "splits" / split_kind / f"{fold}.tsv",
        repo / "data" / "splits" / split_kind / f"{fold}.csv",
        repo / "splits" / split_kind / f"{fold}.tsv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find {split_kind}/{fold} split. Tried: {candidates}"
    )


def _read_split(path: Path) -> list[str]:
    """Each split file is one ID per line, possibly with extra whitespace-separated cols."""
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(line.split()[0])
    return ids


def _normalize_text(s: str) -> str:
    return WHITESPACE_RE.sub(" ", s or "").strip()


def _domain_of(url: str | None, source: str | None) -> str:
    if url:
        ext = tldextract.extract(url)
        if ext.domain:
            return ".".join(p for p in [ext.domain, ext.suffix] if p)
    return (source or "").lower().strip()


def _label_to_id(raw) -> int | None:
    """The dataset stores either a string or an int label. Normalize to {0,1,2}."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw if raw in (0, 1, 2) else None
    s = str(raw).strip().lower()
    if s in LABEL2ID:
        return LABEL2ID[s]
    if s in {"0", "1", "2"}:
        return int(s)
    return None


def _build_split(jsons_dir: Path, ids: list[str], desc: str) -> pd.DataFrame:
    rows = []
    missing = 0
    bad_label = 0
    for art_id in tqdm(ids, desc=desc):
        path = jsons_dir / f"{art_id}.json"
        if not path.exists():
            missing += 1
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            missing += 1
            continue

        text = _normalize_text(obj.get("content") or obj.get("content_original") or "")
        if not text:
            continue

        label_id = _label_to_id(obj.get("bias"))
        if label_id is None:
            label_id = _label_to_id(obj.get("bias_text"))
        if label_id is None:
            bad_label += 1
            continue

        rows.append(
            {
                "article_id": art_id,
                "text": text,
                "label_id": label_id,
                "label": LABELS[label_id],
                "outlet": (obj.get("source") or "").strip(),
                "domain": _domain_of(obj.get("url") or obj.get("source_url"), obj.get("source")),
            }
        )

    if missing or bad_label:
        print(f"  [warn] {desc}: missing={missing} bad_label={bad_label}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["media", "random"], default="media")
    args = parser.parse_args()

    ensure_dirs()
    jsons_dir = _find_jsons_dir(ARTICLE_BIAS_DIR)

    train_path, val_path, test_path = parquet_paths(args.split)
    folds = [("train", train_path), ("valid", val_path), ("test", test_path)]
    for fold, out in folds:
        split_file = _find_split_file(ARTICLE_BIAS_DIR, args.split, fold)
        ids = _read_split(split_file)
        df = _build_split(jsons_dir, ids, desc=f"{args.split}/{fold}")
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        counts = df["label"].value_counts().to_dict()
        print(f"  -> {out.name}: {len(df)} rows, label counts={counts}")

    print(f"\nWrote splits to {PROCESSED}")


if __name__ == "__main__":
    sys.exit(main())
