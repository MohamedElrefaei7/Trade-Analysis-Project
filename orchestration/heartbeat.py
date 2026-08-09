"""
heartbeat.py — Phase 3's monitoring closer: two independent freshness
checks, one for every job in tasks.py::JOBS, one for the AIS feed itself.

Job-overdue check (check_jobs):
    Reads `max_age` per job from worker/cadences.py::CADENCES — this
    module defines no thresholds of its own. `wci-weekly` once read as
    stale for two and a half months because a separate freshness check
    kept its own copy of "should run weekly" out of sync with the real
    cadence (see CLAUDE.md § 11); duplicating the threshold here would
    reproduce exactly that failure mode, and the two tables of the same
    fact would drift silently in the direction that produces false
    confidence. Every job in CADENCES is checked — there is no opt-out
    list.

    "Last success" means the most recent job_runs row with
    status = 'success', by finished_at — not the most recent row of any
    status. A job that has failed every night for a week has recent
    job_runs rows and no recent successes; keying off MAX(started_at)
    regardless of status would report it healthy off a stale `running` or
    `failed` row.

    Three disjoint outcomes per job, in priority order (a job lands in
    exactly one):
      1. never_succeeded — no `success` row exists at all. This is *not*
         folded into overdue: it's the most alarming state a job can be
         in, and a naive LEFT JOIN silently drops it. `heartbeat` itself
         will be in this state on its very first run, as will any newly
         added job — that's correct, not a bug to suppress.
      2. stale_running — the job's most recent `running` row started
         longer ago than its own max_age. This means the process died
         mid-run: the @job decorator writes the `running` row before the
         call and only updates it to a terminal status on return, so a
         killed process leaves an alertable stale row behind instead of
         silence (see orchestration/jobs.py). Distinct from both overdue
         and failed — surfaced separately, not collapsed into either.
      3. overdue — has a success on record, but it's older than max_age.

AIS freshness check (check_ais):
    The AIS daemon (ais/main.py) writes nothing to job_runs by design
    (see CLAUDE.md § 12) — a process reporting its own liveness is the
    same failure mode Prefect had. This queries MAX(ts) FROM positions
    directly and alerts if it hasn't advanced within AIS_STALE_THRESHOLD.
    Justification is empirical, not theoretical: aisstream.io went into
    silent failure on 2026-08-05 — connection accepted, subscription
    accepted, zero messages delivered — undiscovered until 2026-08-08
    while debugging something unrelated. Every layer above the data
    itself reported healthy; only MAX(ts) disagreed.

run_all() ties both checks together and is registered as the `heartbeat`
@job in tasks.py — which means it writes its own job_runs rows, and
therefore checks itself: if the heartbeat stops running, its own last
success ages out and the next run that does happen reports it. That does
NOT cover the case where the heartbeat never runs again at all — nothing
internal to the scheduler can detect that. Closing that specific gap is
Phase 11's UptimeRobot check against /api/health, not this commit.

The return value (the `rows_written` the @job decorator records) is the
count of problems found — overdue + never_succeeded + stale_running jobs,
plus one if AIS is stale — not rows written to any table. This is the
second job, after alerts-nightly, where the number isn't a row count (see
CLAUDE.md § 10). Zero is the healthy state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import text

from alerts.builder import _maybe_post_slack
from clients.base import Session, logger
from worker.cadences import CADENCES

# Provider-outage detector, not a data-volume check — a feed running at
# ~6% of normal volume still advances MAX(ts) every few seconds and
# passes this cleanly; that's a separate, still-open problem (see
# CONTEXT.md), not what this threshold is for.
AIS_STALE_THRESHOLD = timedelta(hours=6)


@dataclass(frozen=True)
class JobHealth:
    overdue: list[tuple[str, timedelta]]
    never_succeeded: list[str]
    stale_running: list[tuple[str, timedelta]]

    @property
    def problem_count(self) -> int:
        return len(self.overdue) + len(self.never_succeeded) + len(self.stale_running)


@dataclass(frozen=True)
class AisHealth:
    stale: bool
    age: timedelta | None
    max_ts: datetime | None


def _last_successes(session) -> dict[str, datetime]:
    rows = session.execute(
        text(
            "SELECT job_name, MAX(finished_at) AS last_success "
            "FROM job_runs WHERE status = 'success' GROUP BY job_name"
        )
    ).fetchall()
    return {r.job_name: r.last_success for r in rows}


def _latest_running_started_at(session) -> dict[str, datetime]:
    rows = session.execute(
        text(
            "SELECT job_name, MAX(started_at) AS latest_running "
            "FROM job_runs WHERE status = 'running' GROUP BY job_name"
        )
    ).fetchall()
    return {r.job_name: r.latest_running for r in rows}


def check_jobs(session) -> JobHealth:
    """
    Returns the set of jobs whose last success is older than their own
    max_age (plus the never-succeeded and stale-running states), reading
    thresholds only from worker.cadences.CADENCES.
    """
    now = datetime.now(timezone.utc)
    last_success = _last_successes(session)
    latest_running = _latest_running_started_at(session)

    overdue: list[tuple[str, timedelta]] = []
    never_succeeded: list[str] = []
    stale_running: list[tuple[str, timedelta]] = []

    for job_name, cadence in CADENCES.items():
        success_at = last_success.get(job_name)
        if success_at is None:
            never_succeeded.append(job_name)
            continue

        running_since = latest_running.get(job_name)
        if running_since is not None and (now - running_since) > cadence.max_age:
            stale_running.append((job_name, now - running_since))
            continue

        age = now - success_at
        if age > cadence.max_age:
            overdue.append((job_name, age))

    return JobHealth(overdue=overdue, never_succeeded=never_succeeded, stale_running=stale_running)


def check_ais(session) -> AisHealth:
    """
    Alerts when MAX(ts) FROM positions hasn't advanced within
    AIS_STALE_THRESHOLD, whatever the AIS daemon believes about itself.
    No rows at all counts as stale, not skipped — same reasoning as
    never_succeeded above.
    """
    row = session.execute(text("SELECT MAX(ts) FROM positions")).fetchone()
    max_ts = row[0] if row else None
    if max_ts is None:
        return AisHealth(stale=True, age=None, max_ts=None)

    age = datetime.now(timezone.utc) - max_ts
    return AisHealth(stale=age > AIS_STALE_THRESHOLD, age=age, max_ts=max_ts)


def _fmt_age(age: timedelta) -> str:
    hours = age.total_seconds() / 3600
    if hours < 48:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def _problem_dicts(job_health: JobHealth, ais: AisHealth) -> list[dict]:
    """
    Shapes every problem found into the {severity, alert_type, subject}
    dicts alerts.builder._maybe_post_slack() already knows how to digest
    and post — reused rather than writing a second Slack poster (see
    CLAUDE.md § Monitoring).
    """
    problems: list[dict] = []

    for name in job_health.never_succeeded:
        problems.append({
            "severity": "critical",
            "alert_type": "job_never_succeeded",
            "subject": f"heartbeat: {name} has never recorded a successful run",
        })

    for name, age in job_health.stale_running:
        problems.append({
            "severity": "critical",
            "alert_type": "job_stale_running",
            "subject": (
                f"heartbeat: {name} has been 'running' for {_fmt_age(age)} "
                f"(max_age={_fmt_age(CADENCES[name].max_age)}) — likely died mid-run"
            ),
        })

    for name, age in job_health.overdue:
        problems.append({
            "severity": "warning",
            "alert_type": "job_overdue",
            "subject": (
                f"heartbeat: {name} last succeeded {_fmt_age(age)} ago "
                f"(max_age={_fmt_age(CADENCES[name].max_age)})"
            ),
        })

    if ais.stale:
        age_desc = "no positions recorded at all" if ais.age is None else f"{_fmt_age(ais.age)} ago"
        problems.append({
            "severity": "critical",
            "alert_type": "ais_stale",
            "subject": (
                f"heartbeat: AIS feed stale — MAX(ts) from positions was {age_desc} "
                f"(threshold={_fmt_age(AIS_STALE_THRESHOLD)})"
            ),
        })

    return problems


def _post_to_slack(problems: list[dict]) -> None:
    """
    Degrades gracefully in every direction: no SLACK_WEBHOOK_URL logs a
    WARNING and returns; a failed post is caught here regardless of what
    _maybe_post_slack itself catches internally, so a Slack outage can
    never turn into this job's own `failed` row — that would make the
    monitoring's own failure state indistinguishable from the thing it
    monitors failing.
    """
    if not os.environ.get("SLACK_WEBHOOK_URL"):
        logger.warning(
            "heartbeat: SLACK_WEBHOOK_URL not set — %d problem(s) logged above only, not posted",
            len(problems),
        )
        return
    try:
        _maybe_post_slack(problems, date.today())
    except Exception as exc:
        logger.warning("heartbeat: Slack post failed, continuing (%s)", exc)


def run_all() -> int:
    """
    Runs both checks and posts a Slack digest if anything's wrong.
    Returns the count of problems found — see module docstring for why
    this is the one number in this job that isn't a row count.
    """
    with Session() as session:
        job_health = check_jobs(session)
        ais_health = check_ais(session)

    problems = _problem_dicts(job_health, ais_health)

    if not problems:
        logger.info("heartbeat: all clear — %d jobs checked, AIS fresh", len(CADENCES))
        return 0

    logger.warning("heartbeat: %d problem(s) found", len(problems))
    for p in problems:
        logger.warning("  [%s] %s", p["alert_type"], p["subject"])

    _post_to_slack(problems)
    return len(problems)


if __name__ == "__main__":
    run_all()
