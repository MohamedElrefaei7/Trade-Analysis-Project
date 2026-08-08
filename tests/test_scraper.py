"""
test_scraper.py — enforcement tests for clients/scraper.py::wci_scraper()'s
distinction between "no new posts" (legitimate success, 0) and "posts
found but none parsed" (a parser/format-drift defect, must raise).

No DB or network access needed: `latest_ts` and `_fetch_wci_posts` are
monkeypatched, so `wci_scraper()`'s `Session()` context managers never
issue a query.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from clients.scraper import WCIParseFailure, wci_scraper


@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    monkeypatch.setattr("clients.scraper.latest_ts", lambda *a, **k: None)


def test_wci_scraper_returns_zero_when_no_posts_found(monkeypatch):
    monkeypatch.setattr("clients.scraper._fetch_wci_posts", lambda **kwargs: [])

    assert wci_scraper() == 0


def test_wci_scraper_raises_when_posts_found_but_nothing_parses(monkeypatch):
    posts = [
        {
            "ts": datetime(2026, 8, 7, tzinfo=timezone.utc),
            "title": "Drewry: World Container Index commentary",
            "link": "https://example.com/post-1",
            "body": "This week's shipping newsletter discusses port congestion "
                    "and fuel prices, with no rate figures of any kind.",
        },
    ]
    monkeypatch.setattr("clients.scraper._fetch_wci_posts", lambda **kwargs: posts)

    with pytest.raises(WCIParseFailure):
        wci_scraper()
