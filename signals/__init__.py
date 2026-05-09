"""
signals — nightly lead-lag relationship sweep.

For every (feature, target, window) triple, finds the lag at which Pearson
correlation peaks, then stores the top-N strongest pairs along with Spearman
correlation and a Granger-causality p-value. The downstream alerter reads
this table to surface "what's working right now" without having to recompute
the sweep on demand.
"""

from .builder import run_all

__all__ = ["run_all"]
