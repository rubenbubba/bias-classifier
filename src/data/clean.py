"""Strip outlet names and scraping boilerplate before vectorization.

Without this, the TF-IDF model picks up "cnn", "fox news", "story highlights",
"replay more videos must watch", "mr/mrs" (WSJ stylebook), etc. as top features —
which means it's learning *which outlet wrote this* rather than bias signals.

Both training and inference must apply the same cleaner; import this module from
both sides.
"""
from __future__ import annotations

import re

# Outlet names that appear inside the article body (logos, "told CNN", etc.).
# Lowercased; matched as whole tokens.
OUTLET_NAMES = {
    "cnn", "msnbc", "fox", "foxnews", "fox news",
    "npr", "monitor", "csmonitor",
    "reuters", "ap", "associated press", "bloomberg", "bbc", "afp",
    "nyt", "new york times", "wsj", "wall street journal",
    "washingtonpost", "washington post", "wapo",
    "politico", "axios", "the hill", "atlantic", "slate", "salon",
    "huffpost", "huffington post", "vox", "intercept", "buzzfeed", "vice",
    "breitbart", "newsmax", "dailywire", "daily wire", "daily caller",
    "the federalist", "national review", "thehill", "guardian", "telegraph",
    "usa today", "usatoday",
}

# Multi-word phrases first so the regex builder can match longest first.
_OUTLET_PATTERN = re.compile(
    r"\b(?:" + "|".join(sorted(OUTLET_NAMES, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

# Site-chrome and newsletter cruft that survives extraction.
_BOILERPLATE = [
    r"story highlights?",
    r"replay\s+more\s+videos?\s+must\s+watch",
    r"more\s+videos?\s+must\s+watch",
    r"more\s+videos?",
    r"must\s+watch",
    r"just\s+watched",
    r"\breplay\b",
    r"delivered\s+to\s+(?:your\s+)?inbox",
    r"sign\s+up\s+for\s+(?:our\s+)?newsletter",
    r"subscribe\s+to\s+(?:our\s+)?newsletter",
    r"privacy\s+policy",
    r"by\s+signing\s+up",
    r"you\s+agree\s+to\s+our",
    r"©\s*\d{4}.*?reserved",
    r"all\s+rights\s+reserved",
    r"click\s+here\s+to",
    r"follow\s+us\s+on\s+(?:twitter|facebook)",
    r"read\s+more\s*:",
    r"^\s*\(reuters\)\s*[-—]\s*",
]
_BOILERPLATE_PATTERN = re.compile("|".join(_BOILERPLATE), flags=re.IGNORECASE)

_WHITESPACE = re.compile(r"\s+")


def clean_for_modeling(text: str) -> str:
    """Strip outlet brand mentions and site chrome only.

    Honorifics (Mr./Mrs./Dr.), full titles (President/Senator), and all framing
    language are kept on purpose: how an outlet refers to a person ("President
    Trump" vs "Trump", "Dr. Fauci" vs "Fauci") is itself a bias signal we want
    the model to use.

    Lowercases at the end — used as TfidfVectorizer.preprocessor= which
    overrides sklearn's default lowercasing.
    """
    if not text:
        return ""
    s = _BOILERPLATE_PATTERN.sub(" ", text)
    s = _OUTLET_PATTERN.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip().lower()
    return s
