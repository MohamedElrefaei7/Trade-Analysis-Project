"""
test_scraper.py — enforcement tests for clients/scraper.py's distinction
between "no new posts" (legitimate success, 0) and a code/network defect
that must not be reported the same way.

Two layers, mirrored fixes:
  - wci_scraper()'s *parse* layer: posts found but none parsed ->
    WCIParseFailure (the original fix this file predates).
  - Both _fetch_bdi_posts()/_fetch_wci_posts()'s *fetch* layer: a network
    or HTTP error before any post is accumulated -> ScraperFetchFailure
    (Part 2.2 — the same shape of bug, one layer down, in both scrapers).

No DB or network access needed: `latest_ts`, `requests.get`, and the
`_fetch_*_posts` helpers are monkeypatched, so nothing here issues a real
query or HTTP request.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import requests

from clients.scraper import (
    ScraperFetchFailure,
    WCIParseFailure,
    _fetch_bdi_posts,
    _fetch_wci_posts,
    bdi_scraper,
    wci_scraper,
)


@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    monkeypatch.setattr("clients.scraper.latest_ts", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _no_polite_delay(monkeypatch):
    # _fetch_bdi_posts/_fetch_wci_posts sleep 1-2s between pages; not
    # reached by any test here (all fail or exit on page 1), but patched
    # out defensively so a future test added to this file doesn't stall.
    monkeypatch.setattr("clients.scraper._polite_delay", lambda *a, **k: None)


class _FakeResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else []

    def json(self):
        return self._payload


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


# ── Part 2.2: fetch-layer network/HTTP failures ─────────────────────────────
#
# Both required, per instruction: a test covering only the raise path would
# pass while the legitimate-zero path was broken into a false alarm, which
# is the failure mode in the other direction.

def test_fetch_bdi_posts_raises_on_network_failure_not_empty_list(monkeypatch):
    """A timeout/connection error on page 1 (nothing accumulated yet) must
    raise ScraperFetchFailure, never silently return []."""
    def _raise(*a, **k):
        raise requests.exceptions.ConnectionError("simulated DNS failure")

    monkeypatch.setattr("clients.scraper.requests.get", _raise)

    with pytest.raises(ScraperFetchFailure):
        _fetch_bdi_posts(latest_ts=None)


def test_fetch_bdi_posts_raises_on_non_200_with_no_posts_yet(monkeypatch):
    """A non-200 response on page 1 must raise, same as a network
    exception — an HTTP 500 is just as much "we got nothing" as a
    timeout is."""
    monkeypatch.setattr(
        "clients.scraper.requests.get", lambda *a, **k: _FakeResponse(503)
    )

    with pytest.raises(ScraperFetchFailure):
        _fetch_bdi_posts(latest_ts=None)


def test_bdi_scraper_returns_zero_when_no_posts_found(monkeypatch):
    """The legitimate-zero path: a real 200 response with zero matching
    posts must still return 0 and must not raise — this is what a
    too-broad raise-on-any-fetch-issue fix would break."""
    monkeypatch.setattr("clients.scraper._fetch_bdi_posts", lambda **kwargs: [])

    assert bdi_scraper() == 0


def test_fetch_wci_posts_raises_on_network_failure_not_empty_list(monkeypatch):
    """Same fetch-layer bug, same fix, in the WCI helper."""
    def _raise(*a, **k):
        raise requests.exceptions.Timeout("simulated timeout")

    monkeypatch.setattr("clients.scraper.requests.get", _raise)

    with pytest.raises(ScraperFetchFailure):
        _fetch_wci_posts(latest_ts=None)


def test_fetch_wci_posts_does_not_raise_once_a_post_was_already_fetched(monkeypatch):
    """A failure on page 2, after page 1 already yielded a real post, must
    not discard that post by raising — the already-fetched data is real
    and more useful returned than thrown away."""
    page_1 = _FakeResponse(
        200,
        [
            {
                "id": 1,
                "date": "2026-08-07T00:00:00",
                "link": "https://example.com/post-1",
                "title": {"rendered": "Drewry: World Container Index update"},
                "content": {"rendered": "<p>$1,500 per 40ft to Rotterdam.</p>"},
            }
        ],
    )
    calls = {"n": 0}

    def _get(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return page_1
        raise requests.exceptions.ConnectionError("simulated failure on page 2")

    monkeypatch.setattr("clients.scraper.requests.get", _get)

    posts = _fetch_wci_posts(latest_ts=None, max_pages=5)
    assert len(posts) == 1
    assert posts[0]["link"] == "https://example.com/post-1"
