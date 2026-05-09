"""Merge AllSides + MBFC outlet ratings into a single domain-keyed lookup table.

Output: data/credibility_lookup.csv with columns
    domain, outlet, bias_outlet, credibility, source

`source` is allsides|mbfc|both. When both disagree, MBFC wins for `credibility`
(it is the more canonical credibility source) and AllSides wins for `bias_outlet`.

    python -m src.data.build_credibility_lookup
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import tldextract

from src.paths import ALLSIDES_CSV, CREDIBILITY_CSV, MBFC_TSV, ensure_dirs

ALLSIDES_BIAS_MAP = {
    "left": "left",
    "lean left": "left",
    "leans left": "left",
    "center": "center",
    "least biased": "center",
    "lean right": "right",
    "leans right": "right",
    "right": "right",
}

MBFC_BIAS_MAP = {
    "left": "left",
    "left-center": "left",
    "leftcenter": "left",
    "center": "center",
    "least-biased": "center",
    "right-center": "right",
    "rightcenter": "right",
    "right": "right",
    "extremeright": "right",
    "extremeleft": "left",
}

MBFC_FACT_MAP = {
    "very-high": "high",
    "very high": "high",
    "high": "high",
    "mostly-factual": "mostly-factual",
    "mostly factual": "mostly-factual",
    "mixed": "mixed",
    "low": "low",
    "very-low": "very-low",
    "very low": "very-low",
}


def _normalize_domain(value: str | None) -> str:
    if not value:
        return ""
    s = str(value).strip().lower()
    if "://" not in s and not s.startswith("www."):
        s = "http://" + s
    ext = tldextract.extract(s)
    if not ext.domain:
        return ""
    return ".".join(p for p in [ext.domain, ext.suffix] if p)


def _normalize_label(value: str | None, table: dict[str, str]) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower().replace("_", "-")
    s = re.sub(r"\s+", " ", s)
    return table.get(s) or table.get(s.replace("-", " "))


def _load_allsides() -> pd.DataFrame:
    if not ALLSIDES_CSV.exists():
        print(f"[warn] AllSides CSV not found at {ALLSIDES_CSV}")
        return pd.DataFrame(columns=["domain", "outlet", "bias_outlet"])

    df = pd.read_csv(ALLSIDES_CSV)
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("news_source") or cols.get("name") or list(df.columns)[0]
    bias_col = cols.get("bias") or cols.get("rating") or "bias"
    url_col = cols.get("url") or cols.get("source_url") or cols.get("link")

    out = pd.DataFrame()
    out["outlet"] = df[name_col].astype(str).str.strip()
    out["bias_outlet"] = df[bias_col].map(lambda x: _normalize_label(x, ALLSIDES_BIAS_MAP))
    out["domain"] = (
        df[url_col].map(_normalize_domain) if url_col else ""
    )
    out = out[out["domain"] != ""].dropna(subset=["bias_outlet"])
    return out.drop_duplicates(subset=["domain"])


def _load_mbfc() -> pd.DataFrame:
    if not MBFC_TSV.exists():
        print(f"[warn] MBFC TSV not found at {MBFC_TSV}")
        return pd.DataFrame(columns=["domain", "outlet", "bias_outlet", "credibility"])

    # Try a few delimiters; the Baly scrape ships TSV but copies vary.
    for sep in ("\t", ","):
        try:
            df = pd.read_csv(MBFC_TSV, sep=sep, on_bad_lines="skip")
            if df.shape[1] > 1:
                break
        except Exception:
            continue
    else:
        print(f"[warn] could not parse {MBFC_TSV}")
        return pd.DataFrame()

    cols = {c.lower(): c for c in df.columns}
    url_col = cols.get("source_url_processed") or cols.get("source_url") or cols.get("url")
    name_col = cols.get("source") or cols.get("name") or cols.get("media")
    bias_col = cols.get("bias") or cols.get("political") or cols.get("leaning")
    fact_col = cols.get("fact") or cols.get("factuality") or cols.get("factual_reporting")

    out = pd.DataFrame()
    out["outlet"] = df[name_col].astype(str).str.strip() if name_col else ""
    out["domain"] = df[url_col].map(_normalize_domain) if url_col else ""
    out["bias_outlet"] = (
        df[bias_col].map(lambda x: _normalize_label(x, MBFC_BIAS_MAP)) if bias_col else None
    )
    out["credibility"] = (
        df[fact_col].map(lambda x: _normalize_label(x, MBFC_FACT_MAP)) if fact_col else None
    )
    out = out[out["domain"] != ""]
    return out.drop_duplicates(subset=["domain"])


def main() -> None:
    ensure_dirs()
    a = _load_allsides()
    m = _load_mbfc()
    print(f"AllSides rows: {len(a)}; MBFC rows: {len(m)}")

    merged = pd.merge(
        a,
        m,
        on="domain",
        how="outer",
        suffixes=("_a", "_m"),
    )

    def _coalesce(row, col_a, col_m):
        va = row.get(col_a)
        vm = row.get(col_m)
        if pd.notna(vm) and vm:
            return vm
        return va if pd.notna(va) else None

    merged["outlet"] = merged.apply(lambda r: _coalesce(r, "outlet_a", "outlet_m"), axis=1)
    # AllSides bias preferred (article-level annotators built on top of it).
    merged["bias_outlet"] = merged.apply(
        lambda r: r["bias_outlet_a"]
        if pd.notna(r.get("bias_outlet_a"))
        else r.get("bias_outlet_m"),
        axis=1,
    )
    merged["credibility"] = merged.get("credibility")

    def _source(row):
        in_a = pd.notna(row.get("bias_outlet_a"))
        in_m = pd.notna(row.get("bias_outlet_m")) or pd.notna(row.get("credibility"))
        if in_a and in_m:
            return "both"
        if in_a:
            return "allsides"
        if in_m:
            return "mbfc"
        return "unknown"

    merged["source"] = merged.apply(_source, axis=1)

    final = merged[["domain", "outlet", "bias_outlet", "credibility", "source"]].copy()
    final = final[final["domain"] != ""].drop_duplicates(subset=["domain"])
    final = final.sort_values("domain").reset_index(drop=True)

    CREDIBILITY_CSV.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(CREDIBILITY_CSV, index=False)
    print(f"Wrote {len(final)} rows -> {CREDIBILITY_CSV}")
    print(final.head(10).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
