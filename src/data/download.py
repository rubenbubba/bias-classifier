"""Download every external dataset the project needs into data/raw/.

Run once:
    python -m src.data.download

Idempotent: skips anything already present.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import requests

from src.paths import (
    ALLSIDES_CSV,
    ARTICLE_BIAS_DIR,
    MBFC_TSV,
    RAW,
    ensure_dirs,
)

ARTICLE_BIAS_REPO = "https://github.com/ramybaly/Article-Bias-Prediction.git"
MBFC_REPO = "https://github.com/ramybaly/News-Media-Reliability.git"
MBFC_REPO_DIR = RAW / "News-Media-Reliability"

ALLSIDES_URL = (
    "https://raw.githubusercontent.com/favstats/AllSideR/master/data/allsides_data.csv"
)


def _git_clone(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return
    print(f"[clone] {url} -> {dest}")
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        check=True,
    )


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return
    print(f"[download] {url} -> {dest}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)


def _locate_mbfc_tsv() -> Path | None:
    """The MBFC corpus file lives somewhere under the repo; find the most likely .tsv."""
    if not MBFC_REPO_DIR.exists():
        return None
    candidates = list(MBFC_REPO_DIR.rglob("*.tsv")) + list(MBFC_REPO_DIR.rglob("*.csv"))
    if not candidates:
        return None
    # Prefer files containing "corpus" or "acl2018" in their name.
    candidates.sort(
        key=lambda p: (
            "corpus" not in p.name.lower(),
            "acl2018" not in str(p).lower(),
            -p.stat().st_size,
        )
    )
    return candidates[0]


def main() -> None:
    ensure_dirs()

    # 1) Article Bias Prediction (Baly et al.) — bias training corpus.
    _git_clone(ARTICLE_BIAS_REPO, ARTICLE_BIAS_DIR)

    # 2) AllSides outlet ratings — for credibility/bias lookup.
    _download(ALLSIDES_URL, ALLSIDES_CSV)

    # 3) MBFC outlet ratings (scraped by Baly et al.) — for credibility lookup.
    _git_clone(MBFC_REPO, MBFC_REPO_DIR)
    found = _locate_mbfc_tsv()
    if found and not MBFC_TSV.exists():
        shutil.copy(found, MBFC_TSV)
        print(f"[copy] {found} -> {MBFC_TSV}")
    elif not found:
        print(
            "[warn] could not auto-locate the MBFC TSV inside "
            f"{MBFC_REPO_DIR}. Inspect the repo and copy the right file to "
            f"{MBFC_TSV} manually."
        )

    # 4) BABE + SemEval are pulled lazily by HuggingFace `datasets`
    # the first time eval runs; nothing to do here.
    print("\nDone. Next: python -m src.data.preprocess")


if __name__ == "__main__":
    sys.exit(main())
