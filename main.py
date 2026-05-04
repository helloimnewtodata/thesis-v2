"""
Pääorkesteri: ajaa koko pipelinen data fetch → features → monthly master.
"""

from pathlib import Path

import pandas as pd
from config import (
    DISPLAY_START_DATE,
    SECTOR_DUMMIES,
    SECTOR_DUMMY_NAMES,
)
from src.data_fetch import (
    open_session,
    fetch_stock_fundamentals,
    fetch_index_data,
    fetch_euribor,
)
from src.features import compute_all_features

try:
    from src.hurst import compute_hurst_dfa
except ImportError as exc:
    compute_hurst_dfa = None
    HURST_IMPORT_ERROR = exc
else:
    HURST_IMPORT_ERROR = None


MASTER_OUTPUT_PATH = Path("data/02_preprocessed/master_features_monthly.csv")
INDEX_OUTPUT_PATH = Path("data/02_preprocessed/df_index_features.csv")
HMM_ML_PATH = Path("data/01_raw/outputs/hmm_regimes_monthly_no_lookahead_ml.csv")


def load_universe(path="data/stoxx600_universe.csv"):
    """Lataa STOXX 600 -universumin RIC-lista CSV:stä."""
    df = pd.read_csv(path)
    return df["RIC"].tolist()


def _ensure_date_column(df):
    df = df.copy()
    if "Date" not in df.columns:
        df = df.reset_index()
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def to_monthly_stock_panel(df_features):
    """One row per Instrument-month using the last available stock observation."""
    df = _ensure_date_column(df_features).sort_values(["Instrument", "Date"]).copy()
    df["_Month"] = df["Date"].dt.to_period("M")

    monthly = (
        df.groupby(["Instrument", "_Month"], group_keys=False)
        .tail(1)
        .copy()
    )
    monthly["StockSourceDate"] = monthly["Date"]
    monthly["Date"] = monthly["_Month"].dt.to_timestamp("M")
    return monthly.drop(columns=["_Month"]).reset_index(drop=True)


def to_monthly_index_features(df_idx):
    """One row per month using the last available index observation."""
    idx = _ensure_date_column(df_idx).sort_values("Date").copy()
    idx["_Month"] = idx["Date"].dt.to_period("M")

    monthly = idx.groupby("_Month", group_keys=False).tail(1).copy()
    monthly["IndexSourceDate"] = monthly["Date"]
    monthly["Date"] = monthly["_Month"].dt.to_timestamp("M")
    monthly = monthly.drop(columns=["_Month"]).reset_index(drop=True)

    rename_map = {
        col: f"Index_{col}"
        for col in monthly.columns
        if col not in {"Date", "IndexSourceDate"}
    }
    return monthly.rename(columns=rename_map)


def build_hurst_panel(df_features):
    if compute_hurst_dfa is None:
        raise ImportError(
            "Hurst-featuret vaativat antropy-paketin. Asenna riippuvuudet "
            "esim. `pip install -r requirements.txt`."
        ) from HURST_IMPORT_ERROR

    hurst = compute_hurst_dfa(df_features)
    hurst = hurst.rename(
        columns={
            "raw_dfa_hurst": "Hurst_Raw_DFA",
            "hurst_rank": "Hurst",
        }
    )
    hurst["Date"] = pd.to_datetime(hurst["Date"]) + pd.offsets.MonthEnd(0)
    return hurst[["Date", "Instrument", "Hurst_Raw_DFA", "Hurst"]]


def load_hmm_panel(path=HMM_ML_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"HMM-featuret puuttuvat: {path}. Aja ensin `python src/HMM.py`."
        )

    hmm = pd.read_csv(path)
    hmm["Date"] = pd.to_datetime(hmm["Date"])
    return hmm


def build_master_dataframe(df_features, df_idx, hmm_path=HMM_ML_PATH):
    master = to_monthly_stock_panel(df_features)

    hurst = build_hurst_panel(df_features)
    master = master.merge(
        hurst,
        on=["Date", "Instrument"],
        how="left",
    )

    idx_monthly = to_monthly_index_features(df_idx)
    master = master.merge(idx_monthly, on="Date", how="left")

    hmm = load_hmm_panel(hmm_path)
    master = master.merge(hmm, on="Date", how="left")

    display_start = pd.to_datetime(DISPLAY_START_DATE)
    master = master.loc[master["Date"] >= display_start]
    return master.sort_values(["Instrument", "Date"]).reset_index(drop=True)


def main():
    # 0. Avaa Refinitiv-sessio
    open_session()

    # 1. Lataa universumi
    universe = load_universe()
    print(f"Universumi: {len(universe)} osaketta")

    # 2. Hae data
    print("Haetaan osakkeiden fundamentaalit...")
    df_stocks = fetch_stock_fundamentals(universe)

    print("Haetaan indeksidata...")
    df_index, df_index_fundamentals = fetch_index_data()

    print("Haetaan EURIBOR...")
    df_euribor = fetch_euribor()

    # 3. Laske featuret
    print("Lasketaan featuret...")
    df_features, df_idx = compute_all_features(
        df_stocks, df_index, df_index_fundamentals, df_euribor,
        SECTOR_DUMMIES, SECTOR_DUMMY_NAMES,
    )

    # 4. Rakenna yksi kuukausitason master-dataframe
    print("Rakennetaan master dataframe...")
    df_master = build_master_dataframe(df_features, df_idx)

    # 5. Tallenna
    print("Tallennetaan...")
    MASTER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_master.to_csv(MASTER_OUTPUT_PATH, index=False)
    df_idx.to_csv(INDEX_OUTPUT_PATH, index=True)
    print(f"Master rows: {len(df_master):,}")
    print(f"Saved: {MASTER_OUTPUT_PATH}")
    print(f"Saved: {INDEX_OUTPUT_PATH}")
    print("Valmis!")

    # 6. Mallinnus (TODO)
    # results = run_walk_forward(df_master)


if __name__ == "__main__":
    main()
