"""
normalizer — Step 7 transformation layer.

Reads raw ingest tables (positions, flight_events, port_calls,
economic_benchmarks, port_daily_summary) and produces analysis-ready rows
in `features`. Never mutates raw tables.

Execution order (feature_builder.run_all enforces this):

    1. port_resolver        — fix port names / backfill UN/LOCODE
    2. vessel_normalizer    — smooth AIS, detect arrivals/departures
    3. time_aligner         — resample everything to daily
    4. lag_adjuster         — shift series by publication lag
    5. seasonal_adjuster    — subtract STL seasonal component
    6. feature_builder      — assemble final features table (last)

Quick-start:
    from normalizer import run_all
    run_all()
"""

from .feature_builder import run_all

__all__ = ["run_all"]
