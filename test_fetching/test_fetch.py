"""
Testiskripti: hakee dataa yhdelle osakkeelle ja tulostaa mitä saatiin.
Aja tämä Refinitiv Workspace -ympäristössä.
"""

import refinitiv.data as rd
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

rd.open_session()

# --- 1. Osakkeen fundamentaalit (sama kuin data_fetch.py) ---
print("=" * 60)
print("1. OSAKKEEN FUNDAMENTAALIT")
print("=" * 60)

universe = ['NESTE.HE']

fields = [
    "TR.PriceClose.date",
    "TR.PriceClose",
    "TR.TotalReturn1D",
    "TR.CompanyMarketCap",
    "TR.Volume",
    "TR.SharesOutstanding",
    "TR.TotalDebt",
    "TR.PriceToSalesPerShare",
    "TR.PriceToCFPerShare",
    "TR.PriceToBVPerShare",
    "TR.GICSSector",
    "TR.GICSSubIndustry",
    "TR.SmartNetDebtToMarketCap",
    "TR.FwdPtoEPSSmartEst",
    "TR.DPSActValue",
    "TR.F.DivPerShare",
    "TR.DPSActValueYield",
    # Uudet: Quality-ryhmä
    "TR.F.ReturnAvgComEqPct",     # ROE
    "TR.F.GrossProfitTotAssets",   # Operating Profitability
]

params = {
    "SDate": "2024-01-01",
    "EDate": "2026-03-05",
    "Frq": "D",
    "Curn": "EUR",
}

df_stock = rd.get_data(universe=universe, fields=fields, parameters=params)

print(f"\nShape: {df_stock.shape}")
print(f"\nSarakkeet:\n{df_stock.columns.tolist()}")
print(f"\nEnsimmäiset rivit:")
print(df_stock.head())
print(f"\nPuuttuvat arvot:")
print(df_stock.isna().sum())
print(f"\nDtypes:")
print(df_stock.dtypes)

# --- 2. Indeksidata ---
print("\n" + "=" * 60)
print("2. INDEKSIDATA (.STOXX, .STOXXR)")
print("=" * 60)

df_index = rd.get_data(
    universe=[".STOXX", ".STOXXR"],
    fields=["TR.PriceClose.date", "TR.PriceClose"],
    parameters=params,
)

print(f"\nShape: {df_index.shape}")
print(df_index.head())

# --- 3. Indeksin fundamentaalit ---
print("\n" + "=" * 60)
print("3. INDEKSIN FUNDAMENTAALIT (.STOXX)")
print("=" * 60)

df_idx_fund = rd.get_history(
    universe=[".STOXX"],
    fields=[
        "TR.Index_PE_RTRS",
        "TR.Index_PRICE_TO_BOOK_RTRS",
        "TR.Index_DIV_YLD_RTRS",
        "TR.PriceClose",
    ],
    start="2024-01-01",
    end="2026-03-05",
    interval="1D",
)

print(f"\nShape: {df_idx_fund.shape}")
print(df_idx_fund.head())

# --- 4. EURIBOR ---
print("\n" + "=" * 60)
print("4. EURIBOR 3KK")
print("=" * 60)

df_euribor = rd.get_data(
    universe=["EURIBOR3MD="],
    fields=["TR.FIXINGVALUE", "TR.FIXINGVALUE.date"],
    parameters={"SDate": "2024-01-01", "EDate": "2026-03-05", "Frq": "D"},
)

print(f"\nShape: {df_euribor.shape}")
print(df_euribor.head())

# --- Tallenna raakadata tarkastelua varten ---
df_stock.to_csv("data/01_raw/test_stock.csv", index=False)
df_index.to_csv("data/01_raw/test_index.csv", index=False)
df_euribor.to_csv("data/01_raw/test_euribor.csv", index=False)

print("\n" + "=" * 60)
print("VALMIS! Tiedostot tallennettu data/01_raw/ -kansioon.")
print("=" * 60)
