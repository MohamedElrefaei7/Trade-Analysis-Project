"""
models — boring-but-honest predictive layer.

One ElasticNet per (target, horizon) trained on lag-adjusted feature z-scores.
Walk-forward cross-validation with `gap=horizon_days` prevents target overlap
between train and test folds. Historical predictions are out-of-sample (true
walk-forward); live predictions on dates past the last realised target use a
final fit on all realised data.

The point isn't accuracy — it's collapsing N features into one number per
target per day that the alerter can threshold against.
"""

from .trainer import run_all

__all__ = ["run_all"]
