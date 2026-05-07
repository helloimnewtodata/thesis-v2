"""
Small point-in-time universe smoke test.

This is intentionally separate from updated_main_test.py. It samples RICs from
STOXX membership intervals, fetches data with warmup, computes the current
feature set, builds a monthly panel, and adds simple point-in-time eligibility
flags.

Run:
    python -B point_in_time_smoke_test.py
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import refinitiv.data as rd

import main as main_pipeline
import src.data_fetch as data_fetch
from config import (
    CHUNK_SIZE as DEFAULT_CHUNK_SIZE,
    DISPLAY_START_DATE,
    SECTOR_DUMMIES,
    SECTOR_DUMMY_NAMES,
    SLEEP_BETWEEN_CHUNKS as DEFAULT_SLEEP_BETWEEN_CHUNKS,
)
from src.features import compute_all_features
from updated_main_test import drop_extra_valuation_columns, reorder_master_columns


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INTERVALS_PATH = PROJECT_ROOT / "data" / "stoxx600_membership_intervals.csv"
DEFAULT_MASTER_OUTPUT = PROJECT_ROOT / "data" / "diagnostics" / "point_in_time_smoke_master.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "data" / "diagnostics" / "point_in_time_smoke_eligibility_by_month.csv"
DEFAULT_SAMPLE_OUTPUT = PROJECT_ROOT / "data" / "diagnostics" / "point_in_time_smoke_sample_universe.csv"

WARMUP_CALENDAR_DAYS = 750
DEFAULT_END_DATE = "2013-01-31"
CORE_HISTORY_FEATURES = ["MOM_12M", "Beta_252d", "-IdioVol"]


def default_fetch_start(display_start):
    return (
        datetime.strptime(display_start, "%Y-%m-%d")
        - timedelta(days=WARMUP_CALENDAR_DAYS)
    ).strftime("%Y-%m-%d")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--display-start", default=DISPLAY_START_DATE)
    parser.add_argument("--fetch-start", default=None)
    parser.add_argument("--end", default=DEFAULT_END_DATE)
    parser.add_argument("--membership-intervals", default=str(DEFAULT_INTERVALS_PATH))
    parser.add_argument("--output", default=str(DEFAULT_MASTER_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT))
    parser.add_argument("--sample-output", default=str(DEFAULT_SAMPLE_OUTPUT))
    parser.add_argument("--chunk-size", type=int, default=min(DEFAULT_CHUNK_SIZE, 10))
    parser.add_argument("--sleep-between-chunks", type=float, default=DEFAULT_SLEEP_BETWEEN_CHUNKS)
    return parser.parse_args()


def apply_runtime_overrides(fetch_start, end, display_start, chunk_size, sleep_between_chunks):
    data_fetch.PARAMS_DAILY["SDate"] = fetch_start
    data_fetch.PARAMS_DAILY["EDate"] = end
    data_fetch.PARAMS_EURIBOR["SDate"] = fetch_start
    data_fetch.PARAMS_EURIBOR["EDate"] = end
    data_fetch.CHUNK_SIZE = chunk_size
    data_fetch.SLEEP_BETWEEN_CHUNKS = sleep_between_chunks
    main_pipeline.DISPLAY_START_DATE = display_start


def _resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_overlapping_intervals(path, display_start, end):
    path = _resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Membership interval file not found: {path}. "
            "Run `python -B src/build_memberships.py` first."
        )

    intervals = pd.read_csv(path)
    required = {"RIC", "MemberStart", "MemberEnd"}
    missing = required - set(intervals.columns)
    if missing:
        raise ValueError(f"Membership interval file is missing columns: {sorted(missing)}")

    intervals["MemberStart"] = pd.to_datetime(intervals["MemberStart"])
    intervals["MemberEnd"] = pd.to_datetime(intervals["MemberEnd"])
    display_start = pd.to_datetime(display_start)
    end = pd.to_datetime(end)

    intervals = intervals.loc[
        (intervals["MemberStart"] <= end) & (intervals["MemberEnd"] >= display_start)
    ].copy()
    if intervals.empty:
        raise ValueError("No membership intervals overlap the requested smoke-test window.")
    return intervals


def sample_universe_from_intervals(intervals, sample_size, seed):
    rics = sorted(intervals["RIC"].dropna().unique().tolist())
    if sample_size is None or sample_size <= 0 or sample_size >= len(rics):
        sample = rics
    else:
        sample = sorted(random.Random(seed).sample(rics, sample_size))
    sample_intervals = intervals.loc[intervals["RIC"].isin(sample)].copy()
    return sample, sample_intervals


def add_membership_and_eligibility_flags(master, intervals):
    out = master.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["IsIndexMemberAtT"] = False

    for ric, ric_intervals in intervals.groupby("RIC"):
        ric_mask = out["Instrument"].eq(ric)
        if not ric_mask.any():
            continue
        member_mask = pd.Series(False, index=out.index)
        for interval in ric_intervals.itertuples(index=False):
            member_mask |= (
                ric_mask
                & (out["Date"] >= interval.MemberStart)
                & (out["Date"] <= interval.MemberEnd)
            )
        out.loc[member_mask, "IsIndexMemberAtT"] = True

    missing_core = [col for col in CORE_HISTORY_FEATURES if col not in out.columns]
    if missing_core:
        raise ValueError(f"Core history features are missing from master: {missing_core}")

    out["HasCoreHistory"] = out[CORE_HISTORY_FEATURES].notna().all(axis=1)

    out = out.sort_values(["Instrument", "Date"]).copy()
    next_date = out.groupby("Instrument")["Date"].shift(-1)
    out["target"] = out.groupby("Instrument")["Excess_Return"].shift(-1)
    expected_next_date = out["Date"] + pd.offsets.MonthEnd(1)
    out["HasNextMonthReturn"] = next_date.eq(expected_next_date) & out["target"].notna()

    out["EligibleAtT"] = out["IsIndexMemberAtT"] & out["HasCoreHistory"]
    out["EligibleForBacktest"] = out["EligibleAtT"] & out["HasNextMonthReturn"]
    return out.sort_values(["Instrument", "Date"]).reset_index(drop=True)


def eligibility_summary(master):
    return (
        master.groupby("Date")
        .agg(
            Rows=("Instrument", "size"),
            IndexMembers=("IsIndexMemberAtT", "sum"),
            WithCoreHistory=("HasCoreHistory", "sum"),
            EligibleAtT=("EligibleAtT", "sum"),
            WithNextMonthReturn=("HasNextMonthReturn", "sum"),
            EligibleForBacktest=("EligibleForBacktest", "sum"),
        )
        .reset_index()
    )


def print_eligibility_totals(master):
    print("\nEligibility totals")
    for col in [
        "IsIndexMemberAtT",
        "HasCoreHistory",
        "EligibleAtT",
        "HasNextMonthReturn",
        "EligibleForBacktest",
    ]:
        print(f"  {col:<24} {int(master[col].sum()):>8} / {len(master):<8}")


def main():
    args = parse_args()
    fetch_start = args.fetch_start or default_fetch_start(args.display_start)
    output_path = _resolve_path(args.output)
    summary_output = _resolve_path(args.summary_output)
    sample_output = _resolve_path(args.sample_output)

    apply_runtime_overrides(
        fetch_start,
        args.end,
        args.display_start,
        args.chunk_size,
        args.sleep_between_chunks,
    )

    intervals = load_overlapping_intervals(
        args.membership_intervals,
        args.display_start,
        args.end,
    )
    universe, sample_intervals = sample_universe_from_intervals(
        intervals,
        args.sample_size,
        args.seed,
    )

    print("Point-in-time smoke test")
    print(f"  sample RICs          : {len(universe)}")
    print(f"  seed                 : {args.seed}")
    print(f"  membership intervals : {args.membership_intervals}")
    print(f"  fetch window         : {fetch_start} -> {args.end}")
    print(f"  visible master start : {args.display_start}")
    print(f"  chunk size           : {args.chunk_size}")

    sample_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"RIC": universe}).to_csv(sample_output, index=False)

    rd.open_session()
    try:
        print("\nFetching stock fundamentals...")
        df_stocks = data_fetch.fetch_stock_fundamentals(universe)

        print("Fetching quarterly EPS/NI fallback data...")
        df_quarterly_eps = data_fetch.fetch_quarterly_eps_for_pe(universe)

        print("Fetching quarterly P/CF fallback data...")
        df_quarterly_pcf = data_fetch.fetch_quarterly_pcf_for_pcf(universe)

        print("Fetching index data...")
        df_index, df_index_fundamentals = data_fetch.fetch_index_data()

        print("Fetching EURIBOR...")
        df_euribor = data_fetch.fetch_euribor()
    finally:
        rd.close_session()

    print("\nComputing daily features...")
    df_features, df_idx = compute_all_features(
        df_stocks,
        df_index,
        df_index_fundamentals,
        df_euribor,
        SECTOR_DUMMIES,
        SECTOR_DUMMY_NAMES,
        df_quarterly_eps=df_quarterly_eps,
        df_quarterly_pcf=df_quarterly_pcf,
    )

    print("Building monthly smoke master with the same post-processing as MASTER_DF_1...")
    master = main_pipeline.build_master_dataframe(df_features, df_idx)
    master = drop_extra_valuation_columns(master)
    master = add_membership_and_eligibility_flags(master, sample_intervals)
    master = reorder_master_columns(master)
    summary = eligibility_summary(master)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path, index=False)
    summary.to_csv(summary_output, index=False)

    print_eligibility_totals(master)
    print("\nSaved")
    print(f"  sample  : {sample_output}")
    print(f"  master  : {output_path}")
    print(f"  summary : {summary_output}")


if __name__ == "__main__":
    main()
