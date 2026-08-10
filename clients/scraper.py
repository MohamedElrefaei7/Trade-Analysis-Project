"""
scraper.py — Web scrapers for shipping data with no free API equivalent.

Two scrapers, one file:

  bdi_scraper()       → Baltic Dry Index daily level from Hellenic Shipping
                        News (WordPress REST API + headline parsing)
                        → economic_benchmarks  series_id='BDI:INDEX'

  wci_scraper()       → Drewry World Container Index composite + per-lane spot
                        rates from weekly Drewry commentary on HSN
                        → economic_benchmarks  series_id='WCI:COMPOSITE' etc.

Dependencies (already in requirements.txt):
    pip install beautifulsoup4 lxml

Implementation notes:
  • BDI uses the Hellenic Shipping News WP REST API and parses the index
    level directly out of post titles ("Baltic Dry Index falls to 2665…").
    No headless browser is needed; Investing.com is now Cloudflare-blocked
    and Stooq's CSV endpoint requires a captcha-issued API key.
  • WCI uses the same HSN WP REST API but reads the rendered post BODY for
    each Drewry weekly commentary, then regex-parses the composite ($/FEU)
    and lane-specific rates (Shanghai→Genoa, Shanghai→Rotterdam,
    Shanghai→Los Angeles, Shanghai→New York, Rotterdam→New York).

Usage:
    from clients.scraper import run
    run()                           # both
    run(["bdi"])                    # single scraper by name
"""

import random
import re
import time
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

from .base import Session, latest_ts, logger

# ── Shared constants ──────────────────────────────────────────────────────────

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_HTTP_HEADERS = {
    "User-Agent": _UA,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _polite_delay(lo: float = 2.0, hi: float = 5.0) -> None:
    time.sleep(random.uniform(lo, hi))


class ScraperFetchFailure(RuntimeError):
    """
    Raised when a fetch helper (_fetch_bdi_posts / _fetch_wci_posts) hits a
    network or HTTP error on its very first page — i.e. before accumulating
    any posts at all. Deliberately distinct from "zero new posts": a page-1
    timeout/DNS failure/non-200 response and a genuinely quiet news day both
    end with an empty list unless this is raised, and the scraper functions'
    `if not posts: return 0` can't tell them apart. Letting this propagate
    turns it into a `failed` job_runs row instead of a silent
    `success, rows_written=0` — the same shape of fix WCIParseFailure already
    applies to the parse layer, one layer up.

    Not raised for a failure on page 2+ after earlier pages already yielded
    posts: those posts are real, already-fetched data, and returning them
    (rather than discarding them by raising) is strictly more correct.
    """


# ── Shared DB helpers ─────────────────────────────────────────────────────────

def _insert_bench_rows(session, rows: list[dict]) -> int:
    if not rows:
        return 0
    session.execute(
        text(
            """
            INSERT INTO economic_benchmarks
                (series_id, source, ts, value, unit, frequency, lag_days)
            VALUES
                (:series_id, :source, :ts, :value, :unit,
                 CAST(:frequency AS data_frequency), :lag_days)
            """
        ),
        rows,
    )
    session.commit()
    return len(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Scraper 1 — Baltic Dry Index from Hellenic Shipping News
# ══════════════════════════════════════════════════════════════════════════════
#
# HSN publishes one post per trading day with the official BDI close in the
# title, e.g. "Baltic Dry Index falls to 2665 down 8 points". We hit their
# WordPress REST API (no auth, no JS) and parse the level out of the title.

_BDI_HSN_API = "https://www.hellenicshippingnews.com/wp-json/wp/v2/posts"

# Match the index level after directional words like "to" / "reaching" / "at".
# Allows commas: "1,234".
_BDI_PRIMARY_RE  = re.compile(
    r'(?:\bto\b|\breaching\b|\bat\b|\bof\b|\baround\b|\bnear\b)\s+([\d,]{3,7})',
    re.I,
)
# Fallback: any 3–7 digit number — the index level is filtered out by range.
_BDI_FALLBACK_RE = re.compile(r'([\d,]{3,7})')

# Numbers that follow these words are year/era references, not index levels.
# "Highest since 2023", "last seen in 2008", "compared to 2022 levels", etc.
_BDI_YEAR_CONTEXT_RE = re.compile(
    r'\b(?:since|in|of|from|during|versus|vs\.?|compared\s+to|like|to)\s+([\d,]{3,7})\s*(?:\b|$)(?!\s*points?\b)',
    re.I,
)

# Plausible BDI range. Filters out point-change values (≤ a few hundred) and
# anything obviously non-index. Historical extremes ~600 (2020) to ~5500 (2008).
_BDI_MIN, _BDI_MAX = 500, 20_000


def _parse_int_with_commas(raw: str) -> int | None:
    raw = raw.replace(",", "").strip()
    return int(raw) if raw.isdigit() else None


def _extract_bdi_level(title: str) -> tuple[int | None, bool]:
    """
    Pull the BDI index level out of a HSN post title.

    Returns (value, is_primary_match):
      - value: the parsed index level, or None if not found
      - is_primary_match: True when the primary directional regex matched
        (high confidence); False when the fallback was used (lower confidence)
    """
    m = _BDI_PRIMARY_RE.search(title)
    if m:
        v = _parse_int_with_commas(m.group(1))
        if v is not None and _BDI_MIN <= v <= _BDI_MAX:
            return v, True

    # Collect numbers that are clearly year/era references so we can exclude
    # them from the fallback. "Highest since 2023" should return None, not 2023.
    year_refs: set[int] = set()
    for ym in _BDI_YEAR_CONTEXT_RE.finditer(title):
        yv = _parse_int_with_commas(ym.group(1))
        if yv is not None:
            year_refs.add(yv)

    candidates = [
        v
        for v in (_parse_int_with_commas(x) for x in _BDI_FALLBACK_RE.findall(title))
        if v is not None and _BDI_MIN <= v <= _BDI_MAX and v not in year_refs
    ]
    return (max(candidates) if candidates else None), False


def _fetch_bdi_posts(latest_ts: datetime | None, max_pages: int = 10) -> list[dict]:
    """
    Page through HSN's WP REST API for "Baltic Dry Index" posts.
    Stops paging once we've passed `latest_ts` (no point fetching older history
    we already have). Posts are returned newest-first by the API.
    """
    seen_links: set[str] = set()
    out: list[dict] = []

    for page in range(1, max_pages + 1):
        try:
            resp = requests.get(
                _BDI_HSN_API,
                headers=_HTTP_HEADERS,
                params={
                    "search": "Baltic Dry Index",
                    "per_page": 100,
                    "page": page,
                    "_fields": "id,date,link,title",
                    "orderby": "date",
                    "order": "desc",
                },
                timeout=20,
            )
            if resp.status_code != 200:
                logger.warning("BDI: HSN API page %d HTTP %d", page, resp.status_code)
                if not out:
                    raise ScraperFetchFailure(
                        f"BDI: HSN API returned HTTP {resp.status_code} on page {page} "
                        "with no posts fetched yet"
                    )
                break
            posts = resp.json()
        except ScraperFetchFailure:
            raise
        except Exception as exc:
            logger.warning("BDI: HSN API error on page %d — %s", page, exc)
            if not out:
                raise ScraperFetchFailure(
                    f"BDI: HSN API request failed on page {page} with no posts fetched yet — {exc}"
                ) from exc
            break

        if not posts:
            break

        passed_latest = False
        for p in posts:
            link = p.get("link") or ""
            if link in seen_links:
                continue
            seen_links.add(link)
            title = (p.get("title") or {}).get("rendered") or ""
            if "Baltic Dry Index" not in title:
                continue
            try:
                ts = datetime.fromisoformat(p["date"]).replace(tzinfo=timezone.utc)
            except (KeyError, ValueError):
                continue
            if latest_ts and ts <= latest_ts:
                passed_latest = True
                continue
            out.append({"ts": ts, "title": title, "link": link})

        if passed_latest:
            # Posts are date-desc, so once we cross latest_ts we're done.
            break
        _polite_delay(1.0, 2.0)

    return out


def bdi_scraper() -> int:
    """
    Scrape Baltic Dry Index daily closing values from Hellenic Shipping News.

    Approach:
      1. Hit HSN's WordPress REST API for posts matching "Baltic Dry Index".
      2. For each post, parse the index level from the title.
      3. Dedupe by UTC date, drop qualitative headlines with no number, and
         insert anything newer than the most recent stored timestamp.

    Returns the number of new rows inserted.
    """
    with Session() as session:
        latest = latest_ts(session, "economic_benchmarks", "series_id", "BDI:INDEX")
    logger.info("BDI: latest stored ts = %s", latest)

    posts = _fetch_bdi_posts(latest_ts=latest)
    if not posts:
        logger.info("BDI: no new posts (already up-to-date or API empty)")
        return 0

    # One value per UTC date. Primary-regex matches beat fallback-only matches;
    # within the same confidence tier, keep the latest timestamp.
    by_date: dict[date, dict] = {}
    skipped_qualitative = 0
    for p in posts:
        value, is_primary = _extract_bdi_level(p["title"])
        if value is None:
            skipped_qualitative += 1
            logger.debug("BDI: no numeric value in %r", p["title"])
            continue
        d = p["ts"].date()
        existing = by_date.get(d)
        if existing is None:
            by_date[d] = {"ts": p["ts"], "value": value, "title": p["title"],
                          "primary": is_primary}
        elif is_primary and not existing["primary"]:
            # A primary match always supersedes a fallback match.
            by_date[d] = {"ts": p["ts"], "value": value, "title": p["title"],
                          "primary": is_primary}
        elif is_primary == existing["primary"] and p["ts"] > existing["ts"]:
            # Same confidence: keep the later post.
            by_date[d] = {"ts": p["ts"], "value": value, "title": p["title"],
                          "primary": is_primary}

    if skipped_qualitative:
        logger.info("BDI: skipped %d qualitative headlines", skipped_qualitative)

    rows = [
        {
            "series_id": "BDI:INDEX",
            "source": "hellenicshippingnews",
            "ts": entry["ts"],
            "value": float(entry["value"]),
            "unit": "index",
            "frequency": "daily",
            "lag_days": 0,
        }
        for entry in by_date.values()
    ]

    with Session() as session:
        inserted = _insert_bench_rows(session, rows)

    logger.info("BDI: %d new rows inserted (covered %d distinct dates)",
                inserted, len(by_date))
    return inserted


# ══════════════════════════════════════════════════════════════════════════════
# Scraper 2 — Drewry World Container Index (WCI) from HSN weekly commentary
# ══════════════════════════════════════════════════════════════════════════════
#
# Hellenic Shipping News re-publishes Drewry's weekly WCI commentary in full,
# typically with the post title "Drewry: World Container Index …". The post
# body contains both the composite index ($X,XXX per 40ft container) and
# per-lane spot rates for the major Drewry routes.
#
# We hit HSN's WordPress REST API with `_fields=content` so the rendered body
# HTML is included alongside the post metadata — one HTTP request per page of
# 100 posts, no per-article fetches needed.

_WCI_HSN_API = "https://www.hellenicshippingnews.com/wp-json/wp/v2/posts"

# Plausible $/40ft-container range. Floor of $700 drops surcharges, point-of-X
# values, and percentage figures occasionally caught by greedy regex windows.
# Drewry's lane all-time low is ~$1,000/FEU but we keep some safety margin.
_WCI_MIN, _WCI_MAX = 700, 30_000

# Lane prices are always quoted as "$X,XXX per 40ft container" in Drewry's
# commentary. Anchoring on `per 40ft` avoids capturing surcharge / point-value
# numbers that lack that unit phrase.
_PER_FEU_TAIL = r'\s*per\s*40ft'

# Composite-index patterns. Tried in order; first match wins.
_WCI_COMPOSITE_PATTERNS = [
    re.compile(
        r'(?:WCI|World Container Index)[^.]{1,300}?(?:\bto\b|\bat\b)\s*\$([\d,]{3,7})' + _PER_FEU_TAIL,
        re.I,
    ),
    re.compile(
        r'(?:WCI|World Container Index)[^.]{1,200}?\$([\d,]{3,7})' + _PER_FEU_TAIL,
        re.I,
    ),
]

# Joint-mention patterns: Drewry frequently bundles the two transpacific lanes
# in one sentence ("from Shanghai to New York and Los Angeles … $A and $B").
_WCI_JOINT_NY_LA = re.compile(
    r'New York and Los Angeles[^.]{0,200}?\$([\d,]{3,7})\s+and\s+\$([\d,]{3,7})',
    re.I,
)
_WCI_JOINT_LA_NY = re.compile(
    r'Los Angeles and New York[^.]{0,200}?\$([\d,]{3,7})\s+and\s+\$([\d,]{3,7})',
    re.I,
)

# Per-lane patterns. Each lane has a list of regexes tried in order. We use
# negative-lookahead on the SH-NY pattern so it doesn't accidentally grab the
# joint "New York and Los Angeles" prefix (handled above instead).
# All patterns require the `per 40ft` tail so they don't catch surcharges.
_WCI_LANE_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    ("WCI:SH-GEN", [
        re.compile(r'Shanghai\s*(?:to|[-–—])\s*Genoa[^.]{0,200}?\$([\d,]{3,7})' + _PER_FEU_TAIL, re.I),
        re.compile(r'\brates\s+to\s+Genoa[^.]{0,150}?\$([\d,]{3,7})' + _PER_FEU_TAIL, re.I),
    ]),
    ("WCI:SH-RTM", [
        re.compile(r'Shanghai\s*(?:to|[-–—])\s*Rotterdam[^.]{0,200}?\$([\d,]{3,7})' + _PER_FEU_TAIL, re.I),
        re.compile(r'\brates\s+to\s+Rotterdam[^.]{0,150}?\$([\d,]{3,7})' + _PER_FEU_TAIL, re.I),
    ]),
    ("WCI:SH-NY", [
        re.compile(
            r'Shanghai\s*(?:to|[-–—])\s*New York(?!\s+and\s+Los Angeles)[^.]{0,200}?\$([\d,]{3,7})' + _PER_FEU_TAIL,
            re.I,
        ),
    ]),
    ("WCI:SH-LA", [
        re.compile(r'Shanghai\s*(?:to|[-–—])\s*Los Angeles[^.]{0,200}?\$([\d,]{3,7})' + _PER_FEU_TAIL, re.I),
        re.compile(r'(?:those|spot rates|rates)\s+to\s+Los Angeles[^.]{0,150}?\$([\d,]{3,7})' + _PER_FEU_TAIL, re.I),
    ]),
    ("WCI:TRANS", [
        re.compile(r'Rotterdam\s*(?:to|[-–—])\s*New York[^.]{0,200}?\$([\d,]{3,7})' + _PER_FEU_TAIL, re.I),
        re.compile(r'Transatlantic[\s\S]{0,300}?\$([\d,]{3,7})' + _PER_FEU_TAIL, re.I),
    ]),
]

# Friendly unit string for every WCI series we store.
_WCI_UNIT = "USD/40ft"


def _is_wci_post(title: str) -> bool:
    """True if the post title looks like a Drewry weekly WCI commentary."""
    if not title:
        return False
    if "World Container Index" in title:
        return True
    upper = title.upper()
    return "DREWRY" in upper and "WCI" in upper


def _wci_clean_value(raw: str) -> float | None:
    try:
        v = float(raw.replace(",", ""))
    except (ValueError, AttributeError):
        return None
    if not (_WCI_MIN <= v <= _WCI_MAX):
        return None
    return v


def _extract_wci_values(body: str) -> dict[str, float]:
    """
    Pull the composite + per-lane spot rates out of one Drewry commentary body.

    Returns a dict mapping series_id (e.g. "WCI:COMPOSITE", "WCI:SH-GEN") to
    its $/40ft value. Lanes that aren't reported in this week's commentary are
    simply missing from the dict — they'll be filled in by other weeks.
    """
    out: dict[str, float] = {}

    # Composite first.
    for pat in _WCI_COMPOSITE_PATTERNS:
        m = pat.search(body)
        if not m:
            continue
        v = _wci_clean_value(m.group(1))
        if v is not None:
            out["WCI:COMPOSITE"] = v
            break

    # Joint NY+LA — handles "Shanghai to NY and LA … $A and $B, respectively".
    j = _WCI_JOINT_NY_LA.search(body)
    if j:
        a, b = _wci_clean_value(j.group(1)), _wci_clean_value(j.group(2))
        if a is not None:
            out["WCI:SH-NY"] = a
        if b is not None:
            out["WCI:SH-LA"] = b
    j = _WCI_JOINT_LA_NY.search(body)
    if j and "WCI:SH-LA" not in out:
        a, b = _wci_clean_value(j.group(1)), _wci_clean_value(j.group(2))
        if a is not None:
            out["WCI:SH-LA"] = a
        if b is not None:
            out["WCI:SH-NY"] = b

    # Individual lane patterns — skip any series already filled by joint match.
    for sid, patterns in _WCI_LANE_PATTERNS:
        if sid in out:
            continue
        for pat in patterns:
            m = pat.search(body)
            if not m:
                continue
            v = _wci_clean_value(m.group(1))
            if v is not None:
                out[sid] = v
                break

    return out


def _fetch_wci_posts(latest_ts: datetime | None, max_pages: int = 5) -> list[dict]:
    """
    Page through HSN's WP REST API for Drewry WCI commentary posts.

    Stops early once we cross `latest_ts`. The `_fields=...content` query gets
    us the rendered body HTML alongside metadata — no per-article HTTP fetch.
    """
    seen_links: set[str] = set()
    out: list[dict] = []

    for page in range(1, max_pages + 1):
        try:
            resp = requests.get(
                _WCI_HSN_API,
                headers=_HTTP_HEADERS,
                params={
                    "search": "Drewry World Container Index",
                    "per_page": 100,
                    "page": page,
                    "_fields": "id,date,link,title,content",
                    "orderby": "date",
                    "order": "desc",
                },
                timeout=25,
            )
            if resp.status_code != 200:
                logger.warning("WCI: HSN API page %d HTTP %d", page, resp.status_code)
                if not out:
                    raise ScraperFetchFailure(
                        f"WCI: HSN API returned HTTP {resp.status_code} on page {page} "
                        "with no posts fetched yet"
                    )
                break
            posts = resp.json()
        except ScraperFetchFailure:
            raise
        except Exception as exc:
            logger.warning("WCI: HSN API error on page %d — %s", page, exc)
            if not out:
                raise ScraperFetchFailure(
                    f"WCI: HSN API request failed on page {page} with no posts fetched yet — {exc}"
                ) from exc
            break

        if not posts:
            break

        passed_latest = False
        for p in posts:
            link = p.get("link") or ""
            if link in seen_links:
                continue
            seen_links.add(link)

            title = (p.get("title") or {}).get("rendered") or ""
            if not _is_wci_post(title):
                continue
            try:
                ts = datetime.fromisoformat(p["date"]).replace(tzinfo=timezone.utc)
            except (KeyError, ValueError):
                continue
            if latest_ts and ts <= latest_ts:
                passed_latest = True
                continue

            body_html = (p.get("content") or {}).get("rendered") or ""
            if not body_html:
                continue
            body_text = BeautifulSoup(body_html, "lxml").get_text(separator=" ")
            out.append({"ts": ts, "title": title, "link": link, "body": body_text})

        if passed_latest:
            break
        _polite_delay(1.0, 2.0)

    return out


class WCIParseFailure(RuntimeError):
    """
    Raised when HSN returned WCI commentary posts but none of them yielded
    a single parseable rate. Deliberately distinct from "zero new posts":
    zero posts is the world having no news (a legitimate, expected 0);
    zero parsed rates from N>0 posts means _extract_wci_values's regexes
    stopped matching HSN's current format — a code defect, not a data
    gap, and one the freshness-based Health tab can't see (a post exists,
    so nothing looks stale, right up until every WCI series is frozen).
    Letting this propagate is what turns it into a `failed` job_runs row
    instead of a silent `success, rows_written=0`.
    """


def wci_scraper() -> int:
    """
    Scrape Drewry World Container Index spot rates from Hellenic Shipping News.

    Approach:
      1. Hit HSN's WP REST API for posts matching "Drewry World Container Index"
         and pull the rendered body inline via _fields=content.
      2. For each commentary, regex out the composite ($/FEU) and lane-specific
         rates (Shanghai→Genoa, →Rotterdam, →Los Angeles, →New York, plus
         Rotterdam→New York Transatlantic).
      3. Insert anything newer than the latest stored WCI:COMPOSITE timestamp.

    Series stored:
        WCI:COMPOSITE    — Drewry composite (8-lane average)
        WCI:SH-GEN       — Shanghai → Genoa
        WCI:SH-RTM       — Shanghai → Rotterdam
        WCI:SH-LA        — Shanghai → Los Angeles
        WCI:SH-NY        — Shanghai → New York
        WCI:TRANS        — Rotterdam → New York (Transatlantic benchmark)

    Lanes that aren't reported in a given week's commentary are skipped for
    that week — Drewry doesn't always quote every route. The composite has
    the highest yield (~95% of posts).

    Returns the total number of rows inserted across all series. Raises
    WCIParseFailure — never returns 0 — if posts were found but none of
    them parsed to any lane; see that class's docstring for why.
    """
    with Session() as session:
        latest = latest_ts(session, "economic_benchmarks", "series_id", "WCI:COMPOSITE")
    logger.info("WCI: latest stored ts = %s", latest)

    posts = _fetch_wci_posts(latest_ts=latest)
    if not posts:
        logger.info("WCI: no new commentary posts (already up-to-date)")
        return 0

    rows: list[dict] = []
    by_post_lanes: list[int] = []

    for p in posts:
        extracted = _extract_wci_values(p["body"])
        by_post_lanes.append(len(extracted))
        if not extracted:
            logger.debug("WCI: no parseable rates in %s", p["link"])
            continue
        for sid, val in extracted.items():
            rows.append({
                "series_id": sid,
                "source": "drewry_via_hsn",
                "ts": p["ts"],
                "value": float(val),
                "unit": _WCI_UNIT,
                "frequency": "weekly",
                "lag_days": 0,
            })
        logger.info("WCI: %s — %d/6 lanes", p["ts"].date(), len(extracted))

    if not rows:
        raise WCIParseFailure(
            f"WCI: {len(posts)} post(s) fetched but 0 parseable rates across "
            f"all of them — HSN's commentary format likely changed; see "
            f"_extract_wci_values / _WCI_COMPOSITE_PATTERNS / _WCI_LANE_PATTERNS"
        )

    with Session() as session:
        inserted = _insert_bench_rows(session, rows)

    avg_lanes = sum(by_post_lanes) / max(1, len(by_post_lanes))
    logger.info(
        "WCI: %d new rows inserted across %d posts (avg %.1f lanes/post)",
        inserted, len(posts), avg_lanes,
    )
    return inserted


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

_SCRAPERS = {
    "bdi": bdi_scraper,
    "wci": wci_scraper,
}


def run(targets: list[str] | None = None) -> None:
    """
    Run all scrapers (or a named subset) in sequence.

    Args:
        targets: list of scraper names — "bdi", "wci".
                 Defaults to both.

    Example:
        run()          # both
        run(["bdi"])   # skip WCI
    """
    names = targets if targets is not None else list(_SCRAPERS)
    unknown = set(names) - set(_SCRAPERS)
    if unknown:
        raise ValueError(f"Unknown scraper name(s): {unknown}. "
                         f"Valid: {set(_SCRAPERS)}")

    total = 0
    for name in names:
        logger.info("▶ Starting scraper: %s", name)
        try:
            n = _SCRAPERS[name]()
            total += n
            logger.info("✓ %s complete — %d rows", name, n)
        except Exception as exc:
            logger.error("✗ %s failed: %s", name, exc, exc_info=True)

    logger.info("Scraper run complete — %d total rows across %d scrapers",
                total, len(names))


if __name__ == "__main__":
    run()
