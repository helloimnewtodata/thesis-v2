"""Quick diagnostic: does .STOXXR come back from fetch_index_data with PIT params?"""
from datetime import datetime, timedelta
import pandas as pd
import refinitiv.data as rd

import src.data_fetch as data_fetch

DISPLAY_START = "2010-01-01"
END = "2026-04-30"
WARMUP = 750

fetch_start = (datetime.strptime(DISPLAY_START, "%Y-%m-%d") - timedelta(days=WARMUP)).strftime("%Y-%m-%d")
data_fetch.PARAMS_DAILY["SDate"] = fetch_start
data_fetch.PARAMS_DAILY["EDate"] = END

rd.open_session()
try:
    df_index, _ = data_fetch.fetch_index_data()
finally:
    rd.close_session()

print(f"\nTotal rows: {len(df_index):,}")
print("\nRows per Instrument:")
print(df_index.groupby("Instrument").size())

stoxxr = df_index.loc[df_index["Instrument"] == ".STOXXR"]
print(f"\n.STOXXR rows: {len(stoxxr):,}")
if not stoxxr.empty:
    print(f"  date range : {stoxxr['Date'].min()} → {stoxxr['Date'].max()}")
    print(f"  Price Close NaN: {stoxxr['Price Close'].isna().sum()} / {len(stoxxr)}")
    print(f"  Price Close dtype: {stoxxr['Price Close'].dtype}")
    print(stoxxr.head())
