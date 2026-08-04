"""
normalizer — Step 7 transformation layer.

Reads raw ingest tables (positions, port_calls, economic_benchmarks,
port_daily_summary) and produces analysis-ready rows in `features`.
Never mutates raw tables.

Execution order (feature_builder.run_all enforces this):

    1. port_resolver        — fix port names / backfill UN/LOCODE
    2. vessel_normalizer    — smooth AIS, detect arrivals/departures
    3. port_summary_builder — roll port_calls up into port_daily_summary
    4. time_aligner         — resample everything to daily
    5. seasonal_adjuster    — subtract STL seasonal component
    6. feature_builder      — assemble final features table (last)

Quick-start:
    from normalizer import run_all
    run_all()
"""

from .feature_builder import run_all

__all__ = ["run_all"]
