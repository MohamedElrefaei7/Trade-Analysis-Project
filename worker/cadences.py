"""
cadences.py — single source of truth for how each job in
orchestration/tasks.py::JOBS is scheduled, and for how stale it's allowed
to get before it's overdue. worker/main.py reads this to register jobs;
the eventual Phase 11 heartbeat imports it too, rather than keeping its
own copy of expected ages — the two facts (trigger timing, overdue
threshold) must never be able to drift apart. See CLAUDE.md § Scheduling.

All timestamps and triggers are UTC, explicitly — see Cadence.trigger's
construction below; never rely on a library or container default.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, timezone

from apscheduler.triggers.base import BaseTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

UTC = timezone.utc


@dataclass(frozen=True)
class Cadence:
    trigger: BaseTrigger
    interval: timedelta
    max_age: timedelta
    misfire_grace_time: int  # seconds — see worker/main.py's registration loop


CADENCES: dict[str, Cadence] = {
    "port-call-refresh": Cadence(
        trigger=IntervalTrigger(hours=2, timezone=UTC),
        interval=timedelta(hours=2),
        max_age=timedelta(hours=2.5),  # ~1.25x interval
        misfire_grace_time=int(timedelta(hours=1).total_seconds()),
    ),
    "bdi-daily": Cadence(
        trigger=CronTrigger(hour=18, minute=30, timezone=UTC),
        interval=timedelta(hours=24),
        max_age=timedelta(hours=30),  # ~1.25x interval
        misfire_grace_time=int(timedelta(hours=12).total_seconds()),
    ),
    "wci-weekly": Cadence(
        trigger=CronTrigger(day_of_week="fri", hour=9, minute=0, timezone=UTC),
        interval=timedelta(days=7),
        max_age=timedelta(days=8.75),  # ~1.25x interval
        misfire_grace_time=int(timedelta(days=3).total_seconds()),
    ),
    "normalizer-nightly": Cadence(
        trigger=CronTrigger(hour=23, minute=30, timezone=UTC),
        interval=timedelta(hours=24),
        max_age=timedelta(hours=30),
        misfire_grace_time=int(timedelta(hours=12).total_seconds()),
    ),
    "targets-nightly": Cadence(
        trigger=CronTrigger(hour=23, minute=45, timezone=UTC),
        interval=timedelta(hours=24),
        max_age=timedelta(hours=30),
        misfire_grace_time=int(timedelta(hours=12).total_seconds()),
    ),
    "signals-nightly": Cadence(
        trigger=CronTrigger(hour=23, minute=55, timezone=UTC),
        interval=timedelta(hours=24),
        max_age=timedelta(hours=30),
        misfire_grace_time=int(timedelta(hours=12).total_seconds()),
    ),
    "models-nightly": Cadence(
        trigger=CronTrigger(hour=0, minute=5, timezone=UTC),
        interval=timedelta(hours=24),
        max_age=timedelta(hours=30),
        misfire_grace_time=int(timedelta(hours=12).total_seconds()),
    ),
    "alerts-nightly": Cadence(
        trigger=CronTrigger(hour=0, minute=15, timezone=UTC),
        interval=timedelta(hours=24),
        max_age=timedelta(hours=30),
        misfire_grace_time=int(timedelta(hours=12).total_seconds()),
    ),
}
