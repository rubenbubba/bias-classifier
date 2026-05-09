"""URL -> cleaned article text. Trafilatura first, newspaper3k as fallback."""
from __future__ import annotations

from dataclasses import dataclass

import tldextract
import trafilatura


@dataclass
class ScrapedArticle:
    url: str
    domain: str
    title: str
    text: str
    authors: list[str]
    publish_date: str | None

    def is_empty(self) -> bool:
        return not self.text or len(self.text.split()) < 30


def _domain_of(url: str) -> str:
    ext = tldextract.extract(url)
    return ".".join(p for p in [ext.domain, ext.suffix] if p)


def _via_trafilatura(url: str) -> ScrapedArticle | None:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    extracted = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        with_metadata=True,
        output_format="json",
    )
    if not extracted:
        return None
    import json

    obj = json.loads(extracted)
    return ScrapedArticle(
        url=url,
        domain=_domain_of(url),
        title=(obj.get("title") or "").strip(),
        text=(obj.get("text") or "").strip(),
        authors=[a for a in (obj.get("author") or "").split(";") if a.strip()],
        publish_date=obj.get("date"),
    )


def _via_newspaper(url: str) -> ScrapedArticle | None:
    try:
        from newspaper import Article  # type: ignore
    except ImportError:
        return None
    try:
        art = Article(url)
        art.download()
        art.parse()
    except Exception:
        return None
    return ScrapedArticle(
        url=url,
        domain=_domain_of(url),
        title=(art.title or "").strip(),
        text=(art.text or "").strip(),
        authors=list(art.authors or []),
        publish_date=str(art.publish_date) if art.publish_date else None,
    )


def scrape(url: str) -> ScrapedArticle:
    """Fetch and extract the main article text from a URL.

    Raises ValueError if both extractors fail or return too little text.
    """
    for fn in (_via_trafilatura, _via_newspaper):
        result = fn(url)
        if result is not None and not result.is_empty():
            return result

    raise ValueError(
        f"Could not extract article text from {url}. "
        "The page may be paywalled, JS-rendered, or block scraping."
    )
