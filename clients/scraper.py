"""
scraper.py — Web scrapers for shipping data with no free API equivalent.

Three scrapers, one file:

  bdi_scraper()       → Baltic Dry Index daily level from Hellenic Shipping
                        News (WordPress REST API + headline parsing)
                        → economic_benchmarks  series_id='BDI:INDEX'

  wci_scraper()       → Drewry World Container Index composite + per-lane spot
                        rates from weekly Drewry commentary on HSN
                        → economic_benchmarks  series_id='WCI:COMPOSITE' etc.

  port_la_scraper()   → Port of Los Angeles monthly TEU throughput
                        → port_daily_summary   port_unlocode='USLAX'

Dependencies (already in requirements.txt):
    pip install playwright beautifulsoup4 lxml
    playwright install chromium

Implementation notes:
  • BDI uses the Hellenic Shipping News WP REST API and parses the index
    level directly out of post titles ("Baltic Dry Index falls to 2665…").
    No headless browser is needed; Investing.com is now Cloudflare-blocked
    and Stooq's CSV endpoint requires a captcha-issued API key.
  • WCI uses the same HSN WP REST API but reads the rendered post BODY for
    each Drewry weekly commentary, then regex-parses the composite ($/FEU)
    and lane-specific rates (Shanghai→Genoa, Shanghai→Rotterdam,
    Shanghai→Los Angeles, Shanghai→New York, Rotterdam→New York).
  • Port of LA historical pages require JS for the stats table — Playwright used.

Usage:
    from clients.scraper import run
    run()                           # all three
    run(["bdi"])                    # single scraper by name
    run(["bdi", "port_la"])         # subset
"""

import random
import re
import time
import urllib.robotparser
from datetime import date, datetime, timezone
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
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


# ── Robots.txt guard ──────────────────────────────────────────────────────────

def _robots_allows(url: str) -> bool:
    """Return True if robots.txt permits fetching this URL."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp.can_fetch(_UA, url)
    except Exception as exc:
        logger.warning("Could not read %s: %s — proceeding", robots_url, exc)
        return True


# ── Playwright browser factory ────────────────────────────────────────────────

def _new_browser(pw):
    """Launch Chromium with flags that reduce bot-detection fingerprint."""
    browser = pw.chromium.launch(
        headless=True,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
        ],
    )
    return browser


def _new_context(browser):
    return browser.new_context(
        user_agent=_UA,
        viewport={"width": 1280, "height": 800},
        locale="en-US",
        timezone_id="America/New_York",
    )


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
                break
            posts = resp.json()
        except Exception as exc:
            logger.warning("BDI: HSN API error on page %d — %s", page, exc)
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
                break
            posts = resp.json()
        except Exception as exc:
            logger.warning("WCI: HSN API error on page %d — %s", page, exc)
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

    Returns the total number of rows inserted across all series.
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
        logger.warning("WCI: %d posts found but no parseable rates", len(posts))
        return 0

    with Session() as session:
        inserted = _insert_bench_rows(session, rows)

    avg_lanes = sum(by_post_lanes) / max(1, len(by_post_lanes))
    logger.info(
        "WCI: %d new rows inserted across %d posts (avg %.1f lanes/post)",
        inserted, len(posts), avg_lanes,
    )
    return inserted


# ══════════════════════════════════════════════════════════════════════════════
# Scraper 3 — Port of Los Angeles monthly TEU throughput
# ══════════════════════════════════════════════════════════════════════════════

_PORT_LA_BASE = "https://portoflosangeles.org"
_PORT_LA_STATS_URL = (
    f"{_PORT_LA_BASE}/Business/statistics/Container-Statistics"
    "/Historical-TEU-Statistics-{year}"
)

_MONTH_NUM = {
    "january": 1,  "february": 2,  "march": 3,    "april": 4,
    "may": 5,      "june": 6,      "july": 7,      "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# Table column indices (0-based) in the historical TEU statistics page.
# Verified against live page: table[1] on the historical pages.
# Row structure: [Month, Loaded Imports, Empty Imports, Total Imports,
#                 Loaded Exports, Empty Exports, Total Exports, Total TEUs,
#                 Prior Year Change]
_COL_MONTH         = 0
_COL_LOADED_IMP    = 1
_COL_EMPTY_IMP     = 2
_COL_TOTAL_IMP     = 3
_COL_LOADED_EXP    = 4
_COL_EMPTY_EXP     = 5
_COL_TOTAL_EXP     = 6
_COL_TOTAL_TEU     = 7


def _int_teu(raw: str) -> int | None:
    """Parse a TEU value like "812,000.25" → 812000."""
    try:
        return int(float(raw.strip().replace(",", "")))
    except (ValueError, AttributeError):
        return None


def _scrape_port_la_year(page, year: int) -> list[dict]:
    """
    Navigate to the Port of LA historical TEU page for one year and return
    a list of port_daily_summary row dicts for months with data.

    The page requires JavaScript to render the statistics table.
    """
    url = _PORT_LA_STATS_URL.format(year=year)
    logger.info("Port LA: fetching %s", url)

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=25_000)
        page.wait_for_timeout(3_000)
    except PWTimeout:
        logger.error("Port LA: timeout loading %s", url)
        return []
    except Exception as exc:
        logger.error("Port LA: error loading %s — %s", url, exc)
        return []

    soup = BeautifulSoup(page.content(), "lxml")
    tables = soup.find_all("table")

    # The stats table is table[1] (table[0] is a search widget)
    stats_table = None
    for t in tables:
        rows = t.find_all("tr")
        if len(rows) >= 6:   # must have at least a few month rows
            headers = [th.get_text(strip=True).lower()
                       for th in rows[0].find_all(["th", "td"])]
            if any("import" in h or "teu" in h or "export" in h
                   for h in headers):
                stats_table = t
                break

    if not stats_table:
        logger.warning(
            "Port LA: stats table not found on %s — "
            "check table structure. Headers of all tables: %s",
            url,
            [[c.get_text(strip=True)[:20] for c in t.find_all(["th","td"])[:5]]
             for t in tables],
        )
        return []

    rows_out: list[dict] = []
    data_rows = stats_table.find_all("tr")[1:]   # skip header row

    for tr in data_rows:
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) <= _COL_TOTAL_TEU:
            continue

        month_name = cells[_COL_MONTH].lower().strip()
        month_num  = _MONTH_NUM.get(month_name)
        if not month_num:
            continue

        total_teu   = _int_teu(cells[_COL_TOTAL_TEU])
        total_imp   = _int_teu(cells[_COL_TOTAL_IMP])
        total_exp   = _int_teu(cells[_COL_TOTAL_EXP])

        if total_teu is None or total_teu == 0:
            continue  # future month with no data yet

        rows_out.append(
            {
                "port_unlocode": "USLAX",
                "date": date(year, month_num, 1),   # first of the month
                "container_count": total_teu,
                "arrivals": total_imp,
                "departures": total_exp,
            }
        )

    _polite_delay(1.5, 3.0)
    return rows_out


def port_la_scraper(years_back: int = 2) -> int:
    """
    Scrape Port of LA monthly TEU container throughput into port_daily_summary.

    Navigates to the Port of LA historical TEU statistics pages (one per year):
        portoflosangeles.org/Business/statistics/Container-Statistics/
            Historical-TEU-Statistics-<year>

    Table columns captured:
        Total Imports  → arrivals
        Total Exports  → departures
        Total TEUs     → container_count

    Data is monthly; date is stored as the 1st of each month.
    Uses ON CONFLICT DO UPDATE so re-running is idempotent.

    Returns the number of rows upserted.
    """
    if not _robots_allows(_PORT_LA_BASE + "/business/statistics"):
        logger.warning("Port LA: robots.txt disallows scraping — skipping")
        return 0

    current_year = datetime.now().year
    target_years = [current_year - i for i in range(years_back)]

    all_rows: list[dict] = []

    with sync_playwright() as pw:
        browser = _new_browser(pw)
        ctx = _new_context(browser)
        page = ctx.new_page()

        for year in target_years:
            year_rows = _scrape_port_la_year(page, year)
            all_rows.extend(year_rows)
            logger.info("Port LA: %d — %d months scraped", year, len(year_rows))

        browser.close()

    if not all_rows:
        logger.warning("Port LA: no data extracted")
        return 0

    upserted = 0
    with Session() as session:
        for row in all_rows:
            session.execute(
                text(
                    """
                    INSERT INTO port_daily_summary
                        (port_unlocode, date, container_count, arrivals, departures)
                    VALUES
                        (:port_unlocode, :date, :container_count,
                         :arrivals, :departures)
                    ON CONFLICT (port_unlocode, date) DO UPDATE SET
                        container_count = EXCLUDED.container_count,
                        arrivals        = EXCLUDED.arrivals,
                        departures      = EXCLUDED.departures
                    """
                ),
                row,
            )
            upserted += 1
        session.commit()

    logger.info("Port LA: %d rows upserted into port_daily_summary", upserted)
    return upserted


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

_SCRAPERS = {
    "bdi":     bdi_scraper,
    "wci":     wci_scraper,
    "port_la": port_la_scraper,
}


def run(targets: list[str] | None = None) -> None:
    """
    Run all scrapers (or a named subset) in sequence.

    Args:
        targets: list of scraper names — "bdi", "wci", "port_la".
                 Defaults to all three.

    Example:
        run()                   # all three
        run(["bdi", "port_la"]) # skip WCI
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
