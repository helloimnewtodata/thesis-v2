"""
Build point-in-time STOXX Europe 600 membership intervals.

Outputs:
    data/stoxx600_membership_intervals.csv
        RIC-level index membership intervals. MemberStart and MemberEnd are
        inclusive dates for monthly filtering.

    data/stoxx600_realistic_universe.csv
        Unique RIC list for all stocks that are members at least once during
        the sample window.

    data/diagnostics/stoxx600_membership_event_summary.csv
        Per-RIC diagnostics: join/leaver counts, interval counts, active flags.

    data/diagnostics/stoxx600_membership_events.csv
        Cleaned raw joiner/leaver events used to build intervals.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import refinitiv.data as rd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_survivor_stoxx_universe import (
    INDEX_RIC,
    _fetch_snapshot_rics,
    fetch_index_change_events,
)


DEFAULT_START_DATE = "2010-01-01"
DEFAULT_END_DATE = "2026-04-30"
DEFAULT_INTERVALS_PATH = PROJECT_ROOT / "data" / "stoxx600_membership_intervals.csv"
DEFAULT_UNIVERSE_PATH = PROJECT_ROOT / "data" / "stoxx600_realistic_universe.csv"
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "data" / "diagnostics" / "stoxx600_membership_events.csv"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "diagnostics" / "stoxx600_membership_event_summary.csv"


def _classify_change(change_value) -> str | None:
    change = str(change_value).strip().lower()
    if "join" in change:
        return "Joiner"
    if "leav" in change:
        return "Leaver"
    return None


def _prepare_events(events: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    clean = events.copy()
    clean["ChangeDate"] = pd.to_datetime(clean["ChangeDate"], errors="coerce")
    clean["RIC"] = clean["RIC"].astype("string").str.strip()
    clean["ChangeTypeClean"] = clean["ChangeType"].map(_classify_change)
    clean = clean.dropna(subset=["RIC", "ChangeDate", "ChangeTypeClean"])
    clean = clean.loc[
        (clean["ChangeDate"] > start_dt) & (clean["ChangeDate"] <= end_dt)
    ].copy()

    # If Refinitiv gives same-day leave+join for one RIC, process the leave
    # first so the state machine records a closed interval and a re-entry.
    clean["EventOrder"] = clean["ChangeTypeClean"].map({"Leaver": 0, "Joiner": 1})
    clean = clean.sort_values(["ChangeDate", "RIC", "EventOrder"]).reset_index(drop=True)
    return clean


def build_membership_intervals(
    start_snapshot: set[str],
    events: pd.DataFrame,
    start_date=DEFAULT_START_DATE,
    end_date=DEFAULT_END_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert start-date snapshot + joiner/leaver events into membership intervals.

    MemberStart and MemberEnd are inclusive for monthly membership tests:
        MemberStart <= month_end <= MemberEnd

    An open membership at the end of the sample is closed at end_date with
    EndSource='OpenAtSampleEnd'.
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    events = _prepare_events(events, start_dt, end_dt)

    active = {
        str(ric).strip(): {
            "MemberStart": start_dt,
            "StartSource": "StartSnapshot",
        }
        for ric in start_snapshot
        if pd.notna(ric) and str(ric).strip()
    }

    intervals = []
    anomalies = []

    def close_interval(ric, end_date_value, end_source):
        state = active.pop(ric)
        intervals.append(
            {
                "RIC": ric,
                "MemberStart": state["MemberStart"],
                "MemberEnd": end_date_value,
                "StartSource": state["StartSource"],
                "EndSource": end_source,
            }
        )

    for row in events.itertuples(index=False):
        ric = str(row.RIC).strip()
        change_date = pd.to_datetime(row.ChangeDate)
        change_type = row.ChangeTypeClean

        if change_type == "Leaver":
            if ric in active:
                close_interval(ric, change_date, "Leaver")
            else:
                anomalies.append(
                    {
                        "RIC": ric,
                        "Date": change_date,
                        "Issue": "LeaverWithoutActiveMembership",
                    }
                )
            continue

        if change_type == "Joiner":
            if ric in active:
                anomalies.append(
                    {
                        "RIC": ric,
                        "Date": change_date,
                        "Issue": "JoinerWhileAlreadyActive",
                    }
                )
            else:
                active[ric] = {
                    "MemberStart": change_date,
                    "StartSource": "Joiner",
                }

    for ric in sorted(active):
        close_interval(ric, end_dt, "OpenAtSampleEnd")

    intervals_df = pd.DataFrame(intervals)
    if not intervals_df.empty:
        intervals_df["MemberStart"] = pd.to_datetime(intervals_df["MemberStart"])
        intervals_df["MemberEnd"] = pd.to_datetime(intervals_df["MemberEnd"])
        intervals_df = intervals_df.sort_values(["RIC", "MemberStart", "MemberEnd"])
        intervals_df["IntervalId"] = intervals_df.groupby("RIC").cumcount() + 1
        intervals_df["IsActiveAtSampleEnd"] = (
            intervals_df["EndSource"] == "OpenAtSampleEnd"
        )
        intervals_df = intervals_df[
            [
                "RIC",
                "IntervalId",
                "MemberStart",
                "MemberEnd",
                "StartSource",
                "EndSource",
                "IsActiveAtSampleEnd",
            ]
        ].reset_index(drop=True)

    anomalies_df = pd.DataFrame(anomalies)
    summary_df = summarize_memberships(intervals_df, events, start_snapshot, anomalies_df)
    return intervals_df, events, summary_df


def summarize_memberships(
    intervals: pd.DataFrame,
    events: pd.DataFrame,
    start_snapshot: set[str],
    anomalies: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if intervals.empty:
        return pd.DataFrame()

    join_counts = (
        events.loc[events["ChangeTypeClean"] == "Joiner"]
        .groupby("RIC")
        .size()
        .rename("JoinEvents")
    )
    leaver_counts = (
        events.loc[events["ChangeTypeClean"] == "Leaver"]
        .groupby("RIC")
        .size()
        .rename("LeaverEvents")
    )
    interval_counts = intervals.groupby("RIC").size().rename("IntervalCount")
    first_start = intervals.groupby("RIC")["MemberStart"].min().rename("FirstMemberStart")
    last_end = intervals.groupby("RIC")["MemberEnd"].max().rename("LastMemberEnd")
    active_end = (
        intervals.groupby("RIC")["IsActiveAtSampleEnd"].max().rename("IsActiveAtSampleEnd")
    )

    summary = pd.concat(
        [join_counts, leaver_counts, interval_counts, first_start, last_end, active_end],
        axis=1,
    ).fillna({"JoinEvents": 0, "LeaverEvents": 0, "IntervalCount": 0})
    summary[["JoinEvents", "LeaverEvents", "IntervalCount"]] = summary[
        ["JoinEvents", "LeaverEvents", "IntervalCount"]
    ].astype(int)
    summary["IsInStartSnapshot"] = summary.index.isin(start_snapshot)
    summary["HasMultipleIntervals"] = summary["IntervalCount"] > 1

    if anomalies is not None and not anomalies.empty:
        anomaly_counts = anomalies.groupby("RIC").size().rename("AnomalyCount")
        summary = summary.join(anomaly_counts, how="left")
    else:
        summary["AnomalyCount"] = 0
    summary["AnomalyCount"] = summary["AnomalyCount"].fillna(0).astype(int)

    return summary.reset_index().rename(columns={"index": "RIC"}).sort_values("RIC")


def write_outputs(
    intervals: pd.DataFrame,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    intervals_path=DEFAULT_INTERVALS_PATH,
    universe_path=DEFAULT_UNIVERSE_PATH,
    events_path=DEFAULT_EVENTS_PATH,
    summary_path=DEFAULT_SUMMARY_PATH,
) -> None:
    intervals_path = Path(intervals_path)
    universe_path = Path(universe_path)
    events_path = Path(events_path)
    summary_path = Path(summary_path)

    for path in [intervals_path, universe_path, events_path, summary_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    intervals.to_csv(intervals_path, index=False)
    events.drop(columns=["EventOrder"], errors="ignore").to_csv(events_path, index=False)
    summary.to_csv(summary_path, index=False)

    universe = pd.DataFrame({"RIC": sorted(intervals["RIC"].dropna().unique())})
    universe.to_csv(universe_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--index-ric", default=INDEX_RIC)
    parser.add_argument("--intervals-output", default=str(DEFAULT_INTERVALS_PATH))
    parser.add_argument("--universe-output", default=str(DEFAULT_UNIVERSE_PATH))
    parser.add_argument("--events-output", default=str(DEFAULT_EVENTS_PATH))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_PATH))
    return parser.parse_args()


def main():
    args = parse_args()
    print("Building point-in-time STOXX membership intervals")
    print(f"  index    : {args.index_ric}")
    print(f"  window   : {args.start_date} -> {args.end_date}")

    rd.open_session()
    try:
        start_snapshot = _fetch_snapshot_rics(args.start_date)
        events = fetch_index_change_events(
            args.start_date,
            args.end_date,
            index_ric=args.index_ric,
        )
    finally:
        rd.close_session()

    intervals, clean_events, summary = build_membership_intervals(
        start_snapshot,
        events,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    write_outputs(
        intervals,
        clean_events,
        summary,
        intervals_path=args.intervals_output,
        universe_path=args.universe_output,
        events_path=args.events_output,
        summary_path=args.summary_output,
    )

    print("\nOutputs")
    print(f"  intervals: {args.intervals_output} ({len(intervals)} rows)")
    print(f"  universe : {args.universe_output} ({intervals['RIC'].nunique()} RICs)")
    print(f"  events   : {args.events_output} ({len(clean_events)} rows)")
    print(f"  summary  : {args.summary_output} ({len(summary)} rows)")

    if not summary.empty:
        print("\nDiagnostics")
        print(f"  start snapshot RICs     : {len(start_snapshot)}")
        print(f"  RICs ever in membership : {intervals['RIC'].nunique()}")
        print(f"  multiple-interval RICs  : {int(summary['HasMultipleIntervals'].sum())}")
        print(f"  active at sample end    : {int(summary['IsActiveAtSampleEnd'].sum())}")
        print(f"  anomaly RICs            : {int((summary['AnomalyCount'] > 0).sum())}")


if __name__ == "__main__":
    main()
