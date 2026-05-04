"""
Updated smoke run for the current feature pipeline.

This combines the small random-universe workflow from main_test.py with the
monthly master construction from main.py:
    - random STOXX 600 sample
    - refreshed stock/index/EURIBOR/quarterly EPS fetches
    - updated valuation features, including legacy comparison columns
    - Hurst merge via main.build_master_dataframe
    - HMM merge via data/01_raw/outputs/hmm_regimes_monthly_no_lookahead_ml.csv
    - final output: data/02_preprocessed/updated_smoke.csv

Run:
    python updated_main_test.py
"""

import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import refinitiv.data as rd

import main as main_pipeline
import src.data_fetch as data_fetch
from config import (
    DISPLAY_START_DATE,
    END_DATE,
    FETCH_START_DATE,
    SECTOR_DUMMIES,
    SECTOR_DUMMY_NAMES,
)
from main import HMM_ML_PATH
from src.features import compute_all_features


PROJECT_ROOT = Path(__file__).resolve().parent
SURVIVOR_UNIVERSE_PATH = PROJECT_ROOT / "data" / "stoxx600_survivor_universe.csv"
RAW_OUTPUT_DIR = PROJECT_ROOT / "data" / "01_raw"
PREPROCESSED_OUTPUT_DIR = PROJECT_ROOT / "data" / "02_preprocessed"
UPDATED_SMOKE_OUTPUT = PREPROCESSED_OUTPUT_DIR / "updated_smoke_KONGO.csv"

SMOKE_UNIVERSE_SIZE = 10
WARMUP_CALENDAR_DAYS = 600


def default_fetch_start(display_start):
    return (
        datetime.strptime(display_start, "%Y-%m-%d")
        - timedelta(days=WARMUP_CALENDAR_DAYS)
    ).strftime("%Y-%m-%d")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=SMOKE_UNIVERSE_SIZE)
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--display-start", default=DISPLAY_START_DATE)
    parser.add_argument("--fetch-start", default=None, help="Defaults to display-start minus 450 days.")
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--output", default=str(UPDATED_SMOKE_OUTPUT))
    parser.add_argument(
        "--universe-path",
        default=str(SURVIVOR_UNIVERSE_PATH),
        help="CSV with a RIC column. Defaults to the STOXX 600 survivor universe.",
    )
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--sleep-between-chunks", type=float, default=3.0)
    return parser.parse_args()


def apply_runtime_overrides(fetch_start, end, display_start, chunk_size, sleep_between_chunks):
    data_fetch.PARAMS_DAILY["SDate"] = fetch_start
    data_fetch.PARAMS_DAILY["EDate"] = end
    data_fetch.PARAMS_EURIBOR["SDate"] = fetch_start
    data_fetch.PARAMS_EURIBOR["EDate"] = end
    data_fetch.CHUNK_SIZE = chunk_size
    data_fetch.SLEEP_BETWEEN_CHUNKS = sleep_between_chunks

    # build_master_dataframe reads DISPLAY_START_DATE from the main module.
    main_pipeline.DISPLAY_START_DATE = display_start


def print_summary(name, df, max_cols=80):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"shape: {df.shape}")
    print("columns:")
    for column in list(df.columns)[:max_cols]:
        print(f"  - {column}")
    if len(df.columns) > max_cols:
        print(f"  ... {len(df.columns) - max_cols} more columns")
    print("\nhead:")
    print(df.head())


def prepare_export_df(name, df):
    export_df = df.copy()

    if name in {"index_fundamentals"}:
        if export_df.index.name != "Date":
            export_df.index.name = "Date"
        export_df = export_df.reset_index()

    if "Date" in export_df.columns:
        export_df["Date"] = pd.to_datetime(export_df["Date"], errors="coerce").dt.date
    if "QDate" in export_df.columns:
        export_df["QDate"] = pd.to_datetime(export_df["QDate"], errors="coerce").dt.date

    return export_df


def save_outputs(prefix, output_dir, datasets):
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    for name, df in datasets.items():
        path = output_dir / f"{prefix}_{name}.csv"
        prepare_export_df(name, df).to_csv(path, index=False)
        saved_paths[name] = path
    return saved_paths


def load_universe(path):
    """Load a RIC universe from CSV and return a clean one-column DataFrame."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Universumitiedostoa ei löydy: {path}")

    universe = pd.read_csv(path)
    if "RIC" not in universe.columns:
        raise ValueError(f"Universumitiedostossa pitää olla RIC-sarake: {path}")

    universe = (
        universe[["RIC"]]
        .dropna()
        .drop_duplicates()
        .sort_values("RIC")
        .reset_index(drop=True)
    )
    if universe.empty:
        raise ValueError(f"Universumitiedosto on tyhjä: {path}")
    return universe, path


def pick_random_universe(snapshot, size, seed=None):
    """Same random-sample logic as main_test.py: no fixed seed by default."""
    rics = snapshot["RIC"].dropna().unique().tolist()
    if seed is not None:
        random.seed(seed)
    if len(rics) <= size:
        return rics
    return random.sample(rics, size)


def nan_report(df, feature_cols):
    report = []
    n = len(df)
    for col in feature_cols:
        if col not in df.columns:
            report.append((col, "MISSING", "-"))
            continue
        n_nan = int(df[col].isna().sum())
        pct = (n_nan / n * 100) if n > 0 else 0
        report.append((col, n_nan, f"{pct:.1f}%"))
    return report


def print_nan_report(df, feature_cols, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"{'Feature':<45} {'NaN':>10} {'% NaN':>8}")
    print("-" * 65)
    for col, n_nan, pct in nan_report(df, feature_cols):
        print(f"{col:<45} {str(n_nan):>10} {pct:>8}")


def main():
    args = parse_args()
    fetch_start = args.fetch_start or default_fetch_start(args.display_start)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    apply_runtime_overrides(
        fetch_start,
        args.end,
        args.display_start,
        args.chunk_size,
        args.sleep_between_chunks,
    )

    print(f"Warmup-fetch: {fetch_start} -> {args.end}")
    print(f"Master-paneelin näkyvä alku: {args.display_start}")
    print(f"Sample size: {args.sample_size}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Sleep between chunks: {args.sleep_between_chunks}s")
    print(f"Universe source: {args.universe_path}")
    print(f"HMM ML-safe input: {HMM_ML_PATH}")

    if not HMM_ML_PATH.exists():
        raise FileNotFoundError(
            f"HMM-featuret puuttuvat: {HMM_ML_PATH}. Aja ensin `python src/HMM.py`."
        )

    rd.open_session()
    try:
        snapshot, universe_path = load_universe(args.universe_path)
        universe = pick_random_universe(snapshot, args.sample_size, seed=args.seed)

        print(f"\nUniversumitiedosto: {universe_path}")
        print(f"Satunnaiset {len(universe)} RIC:iä tähän ajoon:")
        for ric in universe:
            print(f"  - {ric}")

        print("\nHaetaan osakedata...")
        df_stocks = data_fetch.fetch_stock_fundamentals(universe)

        print("Haetaan kvartaalitason EPS/NI fallback-data...")
        df_quarterly_eps = data_fetch.fetch_quarterly_eps_for_pe(universe)

        print("Haetaan indeksidata...")
        df_index, df_index_fundamentals = data_fetch.fetch_index_data()

        print("Haetaan EURIBOR...")
        df_euribor = data_fetch.fetch_euribor()
    finally:
        rd.close_session()

    raw_paths = save_outputs(
        "updated_smoke",
        RAW_OUTPUT_DIR,
        {
            "stocks": df_stocks,
            "quarterly_eps": df_quarterly_eps,
            "index": df_index,
            "index_fundamentals": df_index_fundamentals,
            "euribor": df_euribor,
        },
    )

    print("\nTallennetut raaka-CSV:t:")
    for name, path in raw_paths.items():
        print(f"  - {name}: {path}")

    print_summary("df_stocks (raw)", df_stocks)
    print_summary("df_quarterly_eps (raw)", df_quarterly_eps)

    print("\n" + "=" * 80)
    print("Lasketaan päivätason featuret...")
    print("=" * 80)
    df_features, df_idx = compute_all_features(
        df_stocks,
        df_index,
        df_index_fundamentals,
        df_euribor,
        SECTOR_DUMMIES,
        SECTOR_DUMMY_NAMES,
        df_quarterly_eps=df_quarterly_eps,
    )

    daily_feature_cols = [
        "E/P",
        "E/P_legacy",
        "E/P_ff",
        "P/E",
        "P/E_legacy",
        "P/E_ff",
        "1/P/B",
        "-P/S",
        "-P/CF",
        "-P/CF_op",
        "-P/CF_ps",
        "-P/CF_refinitiv",
        "DivYield_12M",
        "OperatingProfitability",
        "BookToMarket",
        "-Debt/MktCap",
        "MOM_1M",
        "MOM_12M",
        "Stock_vs_Sector_12M_1M",
        "Stock_vs_Sector_1M",
        "Sector_Group",
        "RSI_30d",
        "-Vol_30d",
        "-Beta_252d",
        "-IdioVol",
        "log_MktCap",
        "Excess_Return",
    ]
    print_nan_report(df_features, daily_feature_cols, "NaN-raportti päivätason featureistä")

    print("\n" + "=" * 80)
    print("Rakennetaan kuukausitason master + Hurst + HMM...")
    print("=" * 80)
    df_master = main_pipeline.build_master_dataframe(df_features, df_idx, hmm_path=HMM_ML_PATH)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_master.to_csv(output_path, index=False)

    master_feature_cols = daily_feature_cols + [
        "Hurst",
        "Hurst_Raw_DFA",
        "HMM_Regime",
        "Bull_Prob",
        "Bear_Prob",
        "Transition_Prob",
        "Bull_Prob_MeanWindow",
        "Bear_Prob_MeanWindow",
        "Transition_Prob_MeanWindow",
        "RegimeLabel",
    ]
    print_nan_report(df_master, master_feature_cols, "NaN-raportti updated_smoke masterista")

    print_summary("updated_smoke master", df_master)
    print(f"\nUpdated smoke saved: {output_path}")


if __name__ == "__main__":
    main()
