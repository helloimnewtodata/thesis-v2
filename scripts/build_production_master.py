"""
Build the first lean production master from a PIT regime master.

Default input:
    data/02_preprocessed/PIT_MASTER_DF_1_JM2.csv

Default output:
    data/02_preprocessed/MASTER_DF_PROD_JM2.csv

The script:
    1. keeps only rows where PIT eligibility flags are all true
    2. recomputes monthly returns, monthly excess returns and next-month target
    3. converts index P/E and P/B into E/P and B/M so the direction matches
       stock-level valuation features
    4. writes only identifiers, ML features and the precomputed target

It deliberately leaves residual feature NaNs in place. The current model layer
drops incomplete feature rows inside each train/test split; keeping the gaps in
the master makes data availability visible instead of silently shrinking the
panel during this build step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "02_preprocessed" / "PIT_MASTER_DF_1_JM2.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "02_preprocessed" / "MASTER_DF_PROD_JM2.csv"
DEFAULT_RF_SOURCE = PROJECT_ROOT / "data" / "02_preprocessed" / "MASTER_DF_1.csv"

PIT_FLAGS = ["IsIndexMemberAtT", "HasCoreHistory", "HasNextMonthReturn"]

INDEX_PE_SOURCE = "Index_Calculated PE Ratio"
INDEX_PB_SOURCE = "Index_Calculated Price to Book"
INDEX_EP_FEATURE = "Index_E/P"
INDEX_BM_FEATURE = "Index_B/M"

MODEL_FEATURE_COLUMNS = [
    # Valuation
    "E/P_ff",
    "1/P/B",
    "-P/S",
    "-P/CF_ff",
    "DivYield_12M",
    # Quality
    "OperatingProfitability",
    "BookToMarket",
    "-Debt/MktCap",
    # Momentum
    "MOM_1M",
    "MOM_12M",
    "RSI_30d",
    "Stock_vs_Sector_12M_1M",
    "Stock_vs_Sector_1M",
    "Hurst",
    # Risk
    "-Vol_30d",
    "-Beta_252d",
    "-IdioVol",
    # Standalone
    "log_MktCap",
    # Sector controls
    "Sector_Financials",
    "Sector_Industrials_Materials",
    "Sector_Consumer",
    "Sector_Health_Care",
    "Sector_Technology_Communication",
    # Market/index controls
    "Index_E/P",
    "Index_B/M",
    "Index_Calculated Index Dividend Yield",
    "Index_Index_MOM_1M",
    "Index_Index_MOM_12M",
    # Regime probabilities. In PIT_MASTER_DF_1_JM2 these are JM2 probabilities,
    # merged into the same schema as the HMM probabilities.
    "Bull_Prob",
    "Bear_Prob",
    "Transition_Prob",
    "Bull_Prob_MeanWindow",
    "Bear_Prob_MeanWindow",
    "Transition_Prob_MeanWindow",
]

STOCK_CONTINUOUS_WINSOR_COLUMNS = [
    "E/P_ff",
    "1/P/B",
    "-P/S",
    "-P/CF_ff",
    "DivYield_12M",
    "OperatingProfitability",
    "BookToMarket",
    "-Debt/MktCap",
    "MOM_1M",
    "MOM_12M",
    "RSI_30d",
    "Stock_vs_Sector_12M_1M",
    "Stock_vs_Sector_1M",
    "-Vol_30d",
    "-Beta_252d",
    "-IdioVol",
    "log_MktCap",
]

OUTPUT_COLUMNS = [
    "Instrument",
    "Date",
    *MODEL_FEATURE_COLUMNS,
    "target",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--rf-source",
        default=str(DEFAULT_RF_SOURCE),
        help="CSV containing Date and Rf_monthly. Defaults to MASTER_DF_1.csv.",
    )
    parser.add_argument(
        "--drop-feature-nans",
        action="store_true",
        help="Also drop rows with NaN in any production model feature.",
    )
    parser.add_argument(
        "--exclude-feature",
        action="append",
        default=[],
        help=(
            "Feature column to exclude from the production output. "
            "Can be passed multiple times, e.g. --exclude-feature BookToMarket."
        ),
    )
    parser.add_argument(
        "--winsorize-stock-continuous",
        action="store_true",
        help=(
            "Winsorize only stock-level continuous features by month. "
            "Does not touch Hurst, index features, target, sector dummies or regime columns."
        ),
    )
    parser.add_argument("--winsor-lower", type=float, default=0.01)
    parser.add_argument("--winsor-upper", type=float, default=0.99)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_monthly_rf(path: Path) -> pd.DataFrame:
    rf = pd.read_csv(path, usecols=["Date", "Rf_monthly"], parse_dates=["Date"])
    rf["Date"] = rf["Date"] + pd.offsets.MonthEnd(0)
    rf = (
        rf.drop_duplicates(subset="Date", keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if rf["Rf_monthly"].isna().any():
        raise ValueError(f"Rf_monthly has NaNs in {path}")
    return rf


def coerce_bool_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = [col for col in PIT_FLAGS if col not in out.columns]
    if missing:
        raise ValueError(f"Missing PIT eligibility flags: {missing}")

    for col in PIT_FLAGS:
        if out[col].dtype == bool:
            continue
        out[col] = out[col].astype(str).str.lower().isin({"true", "1", "yes", "y"})
    return out


def recompute_monthly_target(df: pd.DataFrame, rf_monthly: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"]) + pd.offsets.MonthEnd(0)
    out["Price Close"] = pd.to_numeric(out["Price Close"], errors="coerce")

    out = out.drop(columns=["Rf_monthly"], errors="ignore")
    out = out.merge(rf_monthly, on="Date", how="left")
    if out["Rf_monthly"].isna().any():
        missing_dates = out.loc[out["Rf_monthly"].isna(), "Date"].drop_duplicates().dt.date.tolist()
        raise ValueError(f"Missing Rf_monthly for dates: {missing_dates[:10]}")

    out = out.sort_values(["Instrument", "Date"]).copy()
    out["Monthly_Return"] = out.groupby("Instrument")["Price Close"].pct_change(fill_method=None)
    out["Excess_Return"] = out["Monthly_Return"] - out["Rf_monthly"]
    out["target"] = out.groupby("Instrument")["Excess_Return"].shift(-1)
    return out


def add_aligned_index_valuation_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in [INDEX_PE_SOURCE, INDEX_PB_SOURCE] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing source columns for index valuation transforms: {missing}")

    out = df.copy()
    out[INDEX_EP_FEATURE] = 1 / pd.to_numeric(out[INDEX_PE_SOURCE], errors="coerce")
    out[INDEX_BM_FEATURE] = 1 / pd.to_numeric(out[INDEX_PB_SOURCE], errors="coerce")
    return out


def winsorize_stock_continuous_by_month(
    df: pd.DataFrame,
    feature_columns: list[str],
    lower: float,
    upper: float,
) -> pd.DataFrame:
    if not 0 <= lower < upper <= 1:
        raise ValueError(f"Invalid winsor bounds: lower={lower}, upper={upper}")

    out = df.copy()
    columns = [
        col
        for col in STOCK_CONTINUOUS_WINSOR_COLUMNS
        if col in feature_columns and col in out.columns
    ]
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        bounds = out.groupby("Date")[col].quantile([lower, upper]).unstack()
        bounds = bounds.rename(columns={lower: "_winsor_lo", upper: "_winsor_hi"})
        out = out.merge(bounds, left_on="Date", right_index=True, how="left")
        out[col] = out[col].clip(lower=out["_winsor_lo"], upper=out["_winsor_hi"])
        out = out.drop(columns=["_winsor_lo", "_winsor_hi"])
    return out


def build_production_master(
    input_path: Path,
    output_path: Path,
    rf_source_path: Path,
    drop_feature_nans: bool,
    exclude_features: list[str] | None = None,
    winsorize_stock_continuous: bool = False,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> pd.DataFrame:
    exclude_features = exclude_features or []
    unknown_excludes = [col for col in exclude_features if col not in MODEL_FEATURE_COLUMNS]
    if unknown_excludes:
        raise ValueError(f"Cannot exclude unknown production features: {unknown_excludes}")
    feature_columns = [col for col in MODEL_FEATURE_COLUMNS if col not in set(exclude_features)]
    output_columns = ["Instrument", "Date", *feature_columns, "target"]

    df = pd.read_csv(input_path, parse_dates=["Date"])
    df = coerce_bool_flags(df)
    rf_monthly = load_monthly_rf(rf_source_path)
    df = recompute_monthly_target(df, rf_monthly)
    df = add_aligned_index_valuation_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    eligible = df[PIT_FLAGS].all(axis=1)
    prod = df.loc[eligible].copy()

    # Recomputed target is required for supervised ML/evaluation.
    prod = prod.dropna(subset=["target"])

    missing_output_cols = [col for col in output_columns if col not in prod.columns]
    if missing_output_cols:
        raise ValueError(f"Missing expected production columns: {missing_output_cols}")

    prod = prod[output_columns].sort_values(["Date", "Instrument"]).reset_index(drop=True)

    if winsorize_stock_continuous:
        prod = winsorize_stock_continuous_by_month(
            prod,
            feature_columns=feature_columns,
            lower=winsor_lower,
            upper=winsor_upper,
        )
        print(
            "Winsorized stock-level continuous features by month: "
            f"{winsor_lower:.1%}/{winsor_upper:.1%}"
        )

    if drop_feature_nans:
        before = len(prod)
        prod = prod.dropna(subset=feature_columns + ["target"]).reset_index(drop=True)
        print(f"Dropped feature-NaN rows: {before - len(prod):,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prod.to_csv(output_path, index=False)
    return prod


def print_summary(prod: pd.DataFrame, input_path: Path, output_path: Path):
    print("Production master built")
    print(f"  input       : {input_path}")
    print(f"  output      : {output_path}")
    print(f"  rows        : {len(prod):,}")
    print(f"  columns     : {len(prod.columns):,}")
    print(f"  dates       : {prod['Date'].min().date()} -> {prod['Date'].max().date()}")
    print(f"  instruments : {prod['Instrument'].nunique():,}")

    feature_columns = [col for col in MODEL_FEATURE_COLUMNS if col in prod.columns]
    na = (prod[feature_columns + ["target"]].isna().mean() * 100).sort_values(ascending=False)
    na = na[na > 0]
    if not na.empty:
        print("\nResidual feature/target NaN share (%):")
        print(na.round(2).to_string())
    else:
        print("\nResidual feature/target NaNs: none")


def main():
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    rf_source_path = resolve_path(args.rf_source)
    prod = build_production_master(
        input_path=input_path,
        output_path=output_path,
        rf_source_path=rf_source_path,
        drop_feature_nans=args.drop_feature_nans,
        exclude_features=args.exclude_feature,
        winsorize_stock_continuous=args.winsorize_stock_continuous,
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
    )
    print_summary(prod, input_path, output_path)


if __name__ == "__main__":
    main()
