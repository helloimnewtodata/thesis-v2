"""
Diagnostiikka-ajo: Beta_252d ja -IdioVol vain 2010-01-31 → 2013-01-31.

Tarkoitus tarkistaa että nostettu WARMUP_CALENDAR_DAYS poistaa NaN:t
paneelin alusta ENNEN täyden masterin uudelleenrakennusta.

Hakee Refinitivistä vain hinnat (osakkeet + .STOXXR), laskee Daily_Return,
Beta_252d, -IdioVol ja resamplaa kuukauden lopuksi. Tulostaa
NaN-osuudet kuukausittain ja tallentaa pienen taulukon CSV:ksi.

Ajo:
    python beta_idiosync_diagnostics.py
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import refinitiv.data as rd

import src.data_fetch as data_fetch
from src.features import (
    compute_beta,
    compute_daily_return,
    compute_idiosyncratic_vol,
)
from src.pipeline import to_monthly_stock_panel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SURVIVOR_UNIVERSE_PATH = PROJECT_ROOT / "data" / "survivor_universe.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "diagnostics" / "beta_idiovol_warmup_check.csv"

DISPLAY_START = "2010-01-01"
END_DATE = "2013-01-31"
WARMUP_CALENDAR_DAYS = 750


def fetch_start_date(display_start: str, warmup_days: int) -> str:
    return (
        datetime.strptime(display_start, "%Y-%m-%d")
        - timedelta(days=warmup_days)
    ).strftime("%Y-%m-%d")


def load_universe() -> list[str]:
    df = pd.read_csv(SURVIVOR_UNIVERSE_PATH)
    return df["RIC"].dropna().unique().tolist()


def main():
    fetch_start = fetch_start_date(DISPLAY_START, WARMUP_CALENDAR_DAYS)
    print(f"Fetch window : {fetch_start} → {END_DATE}")
    print(f"Display start: {DISPLAY_START}  (warmup {WARMUP_CALENDAR_DAYS} cal days)")

    data_fetch.PARAMS_DAILY["SDate"] = fetch_start
    data_fetch.PARAMS_DAILY["EDate"] = END_DATE

    rd.open_session()

    universe = load_universe()
    print(f"Universe     : {len(universe)} RICs")

    print("\nFetching stock prices ...")
    df_prices = data_fetch._fetch_in_chunks(
        universe,
        ["TR.PriceClose.date", "TR.PriceClose"],
        data_fetch.PARAMS_DAILY,
    )
    df_prices = data_fetch._coerce_numeric_columns(df_prices)
    print(f"  rows: {len(df_prices):,}")

    print("Fetching index prices ...")
    df_index, _ = data_fetch.fetch_index_data()
    print(f"  rows: {len(df_index):,}")

    print("\nComputing Daily_Return, Beta_252d, -IdioVol ...")
    df_prices["Date"] = pd.to_datetime(df_prices["Date"])
    df_prices = compute_daily_return(df_prices)
    df_prices = compute_beta(df_prices, df_index)
    df_prices = compute_idiosyncratic_vol(df_prices, df_index, window=252)

    monthly = to_monthly_stock_panel(df_prices)
    monthly = monthly.loc[monthly["Date"] >= pd.Timestamp(DISPLAY_START)].copy()

    cols = ["Instrument", "Date", "Beta_252d", "-IdioVol"]
    monthly = monthly[[c for c in cols if c in monthly.columns]].sort_values(
        ["Instrument", "Date"]
    )

    print("\nNaN-osuus kuukausittain:")
    nan_by_month = (
        monthly.groupby("Date")[["Beta_252d", "-IdioVol"]]
        .apply(lambda g: g.isna().mean() * 100)
        .round(1)
    )
    print(nan_by_month.to_string())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(monthly):,} rows → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
