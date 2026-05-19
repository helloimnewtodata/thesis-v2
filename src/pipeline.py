"""
Kirjastomoduuli master-paneelin rakennukseen: data fetch → features → monthly master.

Julkinen rajapinta (importoidaan scripts/-skripteistä):
    HMM_ML_PATH              — vakio: HMM-regiimitiedoston oletuspolku
    to_monthly_stock_panel   — päivätason osakedata → kuukausipaneeli
    build_master_dataframe   — yhdistä features + HMM kuukausimasteriin

Ei oma entry point — ajetaan scripts/updated_main_test.py tai scripts/PIT_universe.py kautta.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from config import DISPLAY_START_DATE

try:
    from src.hurst import compute_hurst_dfa
except ImportError as exc:
    compute_hurst_dfa = None
    HURST_IMPORT_ERROR = exc
else:
    HURST_IMPORT_ERROR = None


HMM_ML_PATH = Path("data/01_raw/outputs/hmm_regimes_monthly_no_lookahead_ml.csv")
HMM_PROBABILITY_COLUMNS = [
    "Date",
    "Bull_Prob",
    "Bear_Prob",
    "Transition_Prob",
    "Bull_Prob_MeanWindow",
    "Bear_Prob_MeanWindow",
    "Transition_Prob_MeanWindow",
]


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
    """
    One row per Instrument-month using the last available stock observation.

    The daily feature pipeline keeps Daily_Return and Rf_daily because they are
    needed for rolling risk features. At the monthly modelling frequency,
    Excess_Return is overwritten as the realised monthly stock return minus the
    compounded monthly risk-free return. models.add_forward_return then shifts
    this column by one month to form the next-month target.
    """
    df = _ensure_date_column(df_features).sort_values(["Instrument", "Date"]).copy()
    df["_Month"] = df["Date"].dt.to_period("M")

    monthly = (
        df.groupby(["Instrument", "_Month"], group_keys=False)
        .tail(1)
        .copy()
    )
    monthly["StockSourceDate"] = monthly["Date"]
    monthly["Date"] = monthly["_Month"].dt.to_timestamp("M")

    if "Price Close" in monthly.columns:
        monthly["Monthly_Return"] = monthly.groupby("Instrument")["Price Close"].pct_change()

    if "Rf_daily" in df.columns:
        rf_daily = (
            df[["_Month", "Date", "Rf_daily"]]
            .dropna(subset=["Rf_daily"])
            .drop_duplicates(subset=["Date"], keep="last")
            .sort_values("Date")
        )
        rf_monthly = (
            rf_daily.groupby("_Month")["Rf_daily"]
            .apply(lambda x: float(np.prod(1.0 + x.to_numpy(dtype=float)) - 1.0))
            .rename("Rf_monthly")
            .reset_index()
        )
        monthly = monthly.merge(rf_monthly, on="_Month", how="left")

    if {"Monthly_Return", "Rf_monthly"}.issubset(monthly.columns):
        monthly["Excess_Return"] = monthly["Monthly_Return"] - monthly["Rf_monthly"]

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
    missing = [col for col in HMM_PROBABILITY_COLUMNS if col not in hmm.columns]
    if missing:
        raise ValueError(f"HMM probability -sarakkeet puuttuvat tiedostosta {path}: {missing}")
    return hmm[HMM_PROBABILITY_COLUMNS].copy()


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


