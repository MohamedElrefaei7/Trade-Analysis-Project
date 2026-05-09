"""
alerts — nightly edge-triggered alerter.

Reads from features, signals, and predictions; writes to the alerts table.
Optionally posts a digest to a Slack incoming webhook if SLACK_WEBHOOK_URL
is set in the environment. Edge-triggered: each alert fires once on the day
the threshold is crossed, not every day the value stays beyond it.
"""

from .builder import run_all

__all__ = ["run_all"]
