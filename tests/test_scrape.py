"""Smoke tests for the URL scraper. Skipped if no network."""
from __future__ import annotations

import socket

import pytest

from src.inference.scrape import scrape


def _has_network() -> bool:
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _has_network(), reason="needs network")
@pytest.mark.parametrize(
    "url",
    [
        "https://apnews.com/",
        "https://www.reuters.com/",
    ],
)
def test_scrape_homepage_returns_something_or_raises(url):
    """Homepages aren't articles, but the call should either return text or raise ValueError."""
    try:
        result = scrape(url)
        assert isinstance(result.text, str)
    except ValueError:
        pass  # expected for non-article pages
