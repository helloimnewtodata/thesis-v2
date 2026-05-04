"""
Sanity check for look-ahead risk in Refinitiv accounting time series.

Why this exists:
    Accounting variables can cause look-ahead bias if Refinitiv backfills a
    newly reported value to the fiscal period-end date. Example: if FY2020
    EBITDA is published on 2021-02-15 but already appears on 2020-12-31 in a
    daily panel, a model formed at 2021-01-31 would see future information.

What the test checks:
    The script fetches accounting fields directly from Refinitiv, including
    both:
        Date            = the daily panel date on which the value is observed
        period_end_date = the fiscal/quarterly period the value refers to

    It then records only true value changes within each Instrument + field
    pair. The first non-missing observation is ignored because it is just the
    starting value in the requested sample, not an update event.

How to read the output:
    period_lag_days = Date - period_end_date

    Good sign:
        period_lag_days is positive, with changes occurring after period end
        and around reporting/announcement dates.

    Warning sign:
        many changes occur on fiscal period-end-like dates, especially
        12-31 / 03-31 / 06-30 / 09-30, or period_lag_days <= 0.

Interpretation caveat:
    This is an empirical sanity check on sampled Refinitiv daily fields, not a
    legal guarantee that the database is fully point-in-time. If the test shows
    positive lags and no systematic period-end updates, using the latest daily
    value available at month-end is much easier to defend than applying a blunt
    3-6 month lag to all accounting variables.

Run:
    python src/sanity_check_lookahead.py
    python src/sanity_check_lookahead.py --sample-size 20 --seed 42 --output data/diagnostics/lagged_feature_changes.csv
    python src/sanity_check_lookahead.py --ticker SAPG.DE --start 2019-01-01 --end 2022-12-31
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import refinitiv.data as rd

DEFAULT_TICKER = "NESTE.HE"
DEFAULT_START = "2019-01-01"
DEFAULT_END = "2022-12-31"
DEFAULT_SAMPLE_SIZE = 20
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = PROJECT_ROOT / "data" / "stoxx600_universe.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_stoxx_universe import save_stoxx_universe_snapshot

FIELD_SPECS = [
    {
        "field": "TR.EPSFRActValue(Period=LTM)",
        "period_field": "TR.EPSFRActValue.periodenddate",
        "label": "Earnings Per Share Reported - Actual",
    },
    {
        "field": "TR.EBITDAActValue",
        "period_field": "TR.EBITDAActValue.periodenddate",
        "label": "EBITDA - Actual",
    },
    {
        "field": "TR.F.NetCashFlowOp(Period=LTM)",
        "period_field": "TR.F.NetCashFlowOp.periodenddate",
        "label": "Net Cash Flow from Operating Activities",
    },
    {
        "field": "TR.F.NetCFOpPerShr(Period=LTM)",
        "period_field": "TR.F.NetCFOpPerShr.periodenddate",
        "label": "Cash Flow from Operations per Share",
    },
    {
        "field": "TR.TotalDebt",
        "period_field": "TR.TotalDebt.periodenddate",
        "label": "Total Debt",
    },
    {
        "field": "TR.F.ShHoldEqCom",
        "period_field": "TR.F.ShHoldEqCom.periodenddate",
        "label": "Shareholders Equity - Common",
    },
    {
        "field": "TR.F.PriceToBookValuePerShr",
        "period_field": "TR.F.PriceToBookValuePerShr.periodenddate",
        "label": "Price to Book Value per Share",
    },
    {
        "field": "TR.F.ReturnAvgComEqPct",
        "period_field": "TR.F.ReturnAvgComEqPct.periodenddate",
        "label": "Return on Average Common Equity - %",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check when Refinitiv accounting fields update in a daily panel."
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help=f"Run one ticker only, e.g. {DEFAULT_TICKER}. If omitted, sample from STOXX 600.",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    return parser.parse_args()


def pick_random_universe(snapshot: pd.DataFrame, size: int, seed: int | None = None) -> list[str]:
    """Same random-sample logic as main_test.py, with an optional seed."""
    rics = snapshot["RIC"].dropna().unique().tolist()
    if seed is not None:
        random.seed(seed)
    if len(rics) <= size:
        return rics
    return random.sample(rics, size)


def _period_end_column(df: pd.DataFrame, value_col: str) -> str | None:
    candidates = [
        col
        for col in df.columns
        if col not in {"Instrument", "Date", value_col}
        and ("period" in col.lower() or "date" in col.lower())
    ]
    if not candidates:
        return None
    return candidates[0]


def fetch_one_accounting_field(
    universe: list[str],
    spec: dict[str, str],
    start: str,
    end: str,
) -> pd.DataFrame:
    params = {"SDate": start, "EDate": end, "Frq": "D", "Curn": "EUR"}
    fields = ["TR.PriceClose.date", spec["period_field"], spec["field"]]

    try:
        df = rd.get_data(universe=universe, fields=fields, parameters=params)
        period_status = "OK"
    except Exception as exc:
        print(
            f"Period-date field failed for {spec['label']} ({spec['period_field']}): "
            f"{type(exc).__name__}: {str(exc)[:120]}"
        )
        df = rd.get_data(
            universe=universe,
            fields=["TR.PriceClose.date", spec["field"]],
            parameters=params,
        )
        period_status = "MISSING"

    if "Date" not in df.columns:
        raise RuntimeError(f"Refinitiv response did not include Date. Columns: {list(df.columns)}")

    value_col = spec["label"] if spec["label"] in df.columns else None
    if value_col is None:
        non_id_cols = [c for c in df.columns if c not in {"Instrument", "Date"}]
        period_like = [c for c in non_id_cols if "period" in c.lower() or "date" in c.lower()]
        value_candidates = [c for c in non_id_cols if c not in period_like]
        if not value_candidates:
            raise RuntimeError(f"Could not identify value column for {spec['field']}: {list(df.columns)}")
        value_col = value_candidates[-1]

    period_col = _period_end_column(df, value_col)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    out = df[["Instrument", "Date", value_col]].copy()
    out = out.rename(columns={value_col: "value"})
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["field"] = spec["label"]
    out["refinitiv_field"] = spec["field"]
    out["period_status"] = period_status

    if period_col is not None:
        out["period_end_date"] = pd.to_datetime(df[period_col], errors="coerce")
    else:
        out["period_end_date"] = pd.NaT

    return (
        out.dropna(subset=["Date"])
        .sort_values(["Instrument", "field", "Date"])
        .reset_index(drop=True)
    )


def fetch_daily_accounting(universe: list[str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for spec in FIELD_SPECS:
        print(f"  - {spec['label']}")
        frames.append(fetch_one_accounting_field(universe, spec, start, end))
    return pd.concat(frames, ignore_index=True)


def find_changes(df: pd.DataFrame) -> pd.DataFrame:
    base = df.loc[df["value"].notna()].copy()
    base = base.sort_values(["Instrument", "field", "Date"])
    grouped = base.groupby(["Instrument", "field"], sort=False)
    base["previous_value"] = grouped["value"].shift()
    changed = base["previous_value"].notna() & base["value"].ne(base["previous_value"])

    out = base.loc[changed].copy()
    out["month_end_like"] = out["Date"].dt.is_month_end
    out["quarter_end_like"] = out["Date"].dt.is_quarter_end
    out["calendar_fy_end_like"] = out["Date"].dt.strftime("%m-%d").eq("12-31")
    out["period_lag_days"] = (out["Date"] - out["period_end_date"]).dt.days
    out["period_month_end_like"] = out["period_end_date"].dt.is_month_end
    out["period_quarter_end_like"] = out["period_end_date"].dt.is_quarter_end
    return out[
        [
            "Instrument",
            "field",
            "refinitiv_field",
            "Date",
            "period_end_date",
            "period_lag_days",
            "value",
            "previous_value",
            "month_end_like",
            "quarter_end_like",
            "calendar_fy_end_like",
            "period_month_end_like",
            "period_quarter_end_like",
            "period_status",
        ]
    ]


def summarize_by_field(changes: pd.DataFrame) -> pd.DataFrame:
    if changes.empty:
        return pd.DataFrame()

    return (
        changes.groupby("field")
        .agg(
            n_changes=("Date", "count"),
            n_month_end_like=("month_end_like", "sum"),
            n_quarter_end_like=("quarter_end_like", "sum"),
            n_calendar_fy_end_like=("calendar_fy_end_like", "sum"),
            n_with_period_end=("period_end_date", "count"),
            median_period_lag_days=("period_lag_days", "median"),
            min_period_lag_days=("period_lag_days", "min"),
            max_period_lag_days=("period_lag_days", "max"),
            first_change=("Date", "min"),
            last_change=("Date", "max"),
        )
        .assign(
            share_quarter_end_like=lambda x: x["n_quarter_end_like"] / x["n_changes"],
            share_calendar_fy_end_like=lambda x: x["n_calendar_fy_end_like"] / x["n_changes"],
        )
        .sort_values("field")
    )


def summarize_by_instrument(changes: pd.DataFrame) -> pd.DataFrame:
    if changes.empty:
        return pd.DataFrame()

    return (
        changes.groupby("Instrument")
        .agg(
            n_changes=("Date", "count"),
            n_quarter_end_like=("quarter_end_like", "sum"),
            n_calendar_fy_end_like=("calendar_fy_end_like", "sum"),
            n_with_period_end=("period_end_date", "count"),
            median_period_lag_days=("period_lag_days", "median"),
        )
        .assign(
            share_quarter_end_like=lambda x: x["n_quarter_end_like"] / x["n_changes"],
            share_calendar_fy_end_like=lambda x: x["n_calendar_fy_end_like"] / x["n_changes"],
        )
        .sort_values(
            ["n_quarter_end_like", "n_calendar_fy_end_like", "Instrument"],
            ascending=[False, False, True],
        )
    )


def main() -> None:
    args = parse_args()

    rd.open_session()
    try:
        if args.ticker:
            universe = [args.ticker]
            print(f"Using single ticker: {args.ticker}")
        else:
            snapshot, output_path = save_stoxx_universe_snapshot(output_path=UNIVERSE_PATH)
            universe = pick_random_universe(snapshot, args.sample_size, seed=args.seed)
            print(f"Universe snapshot: {output_path} ({len(snapshot)} RICs)")
            print(f"Random sample ({len(universe)} RICs):")
            for ric in universe:
                print(f"  - {ric}")

        print(f"\nFetching accounting fields: {args.start} -> {args.end}")
        df = fetch_daily_accounting(universe, args.start, args.end)
    finally:
        rd.close_session()

    changes = find_changes(df).sort_values(["Instrument", "field", "Date"])
    if changes.empty:
        raise SystemExit("No accounting value changes found after first observations.")

    field_summary = summarize_by_field(changes)
    instrument_summary = summarize_by_instrument(changes)
    suspicious = changes.loc[changes["quarter_end_like"] | changes["calendar_fy_end_like"]]

    print("\nField summary:")
    print(field_summary.to_string())

    print("\nInstrument summary:")
    print(instrument_summary.to_string())

    print("\nPeriod-end-like change dates:")
    if suspicious.empty:
        print("None found.")
    else:
        print(suspicious.to_string(index=False))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        changes.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
