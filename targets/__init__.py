"""
targets — prediction targets for the modeling layer.

Mirrors the shape of `normalizer`: a single `run_all()` orchestrator that
reads from `features`, computes forward returns, and upserts into the
`targets` table. The modeling layer joins feature rows at date `t` to a
target value that spans `t → t + horizon_days`.
"""

from .builder import run_all

__all__ = ["run_all"]
