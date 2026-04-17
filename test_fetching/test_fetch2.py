"""
Testiskripti v2: hakee dataa yhdelle osakkeelle (NESTE.HE) ja tekee
kaikki samat preprosessoinnit kuin FINAL-notebookeissa.
Aja VS Codessa Jupyter-kernelillä tai CodeBookissa.
"""

import refinitiv.data as rd
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

rd.open_session()

# =====================================================================
# 1. OSAKKEEN FUNDAMENTAALIT (DIVIDEND_YIELD_FINAL.ipynb)
# =====================================================================
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
    "TR.PtoEPSMeanEst",
    "TR.F.CF",
    "TR.DPSActValue",
    "TR.F.DivPerShare",
    "TR.DPSActValueYield",
]

params = {
    "SDate": "2024-01-01",
    "EDate": "2026-03-05",
    "Frq": "D",
    "Curn": "EUR",
}

df_stock = rd.get_data(universe=universe, fields=fields, parameters=params)

# Preprosessointi: Div Yield manuaalinen (DIVIDEND_YIELD_FINAL)
df_stock['Div_Yield_Manual'] = (df_stock['Dividend Per Share - Actual'] / df_stock['Price Close']) * 100

# Erillinen Dividend Yield -haku ja merge (DIVIDEND_YIELD_FINAL)
df_dy = rd.get_data(
    universe=universe,
    fields=["TR.DividendYield.date", "TR.DividendYield"],
    parameters=params,
)
df_stock = pd.merge(
    df_stock, df_dy,
    left_on=['Date', 'Instrument'],
    right_on=['Date', 'Instrument'],
    how='left',
)

print(f"\nShape: {df_stock.shape}")
print(f"\nSarakkeet:\n{df_stock.columns.tolist()}")
print(f"\nHead:")
print(df_stock.head())
print(f"\nPuuttuvat arvot:")
print(df_stock.isna().sum())

# =====================================================================
# 2. TUOTOT JA BETA (RETURNS_FINAL + RETURNS_AND_BETA_FINAL)
# =====================================================================
print("\n" + "=" * 60)
print("2. TUOTOT JA BETA")
print("=" * 60)

# Osakkeen tuotot
df_returns = rd.get_data(
    universe=['NESTE.HE'],
    fields=["TR.PriceClose.date", "TR.PriceClose", "TR.TotalReturn1D"],
    parameters=params,
)

# TotalReturn1D tulee prosentteina → muunna desimaaliksi (RETURNS_FINAL)
df_returns['Daily Total Return'] = df_returns['Daily Total Return'] / 100

# Manuaalinen return ilman osinkoja (RETURNS_FINAL)
df_returns['Daily Return excl. dividends'] = df_returns['Price Close'].pct_change()

# Log-tuotot (RETURNS_FINAL)
df_returns['Log Daily Total Return'] = np.log(1 + df_returns['Daily Total Return'])
df_returns['Log Daily Return excl. dividends'] = np.log(1 + df_returns['Daily Return excl. dividends'])

# Indeksituotot (RETURNS_FINAL)
df_index_prices = rd.get_data(
    universe=[".STOXX", ".STOXXR"],
    fields=["TR.PriceClose.date", "TR.PriceClose"],
    parameters=params,
)

# Date ilman kellonaikaa + duplikaattien poisto (RETURNS_FINAL)
df_index_prices['Date'] = pd.to_datetime(df_index_prices['Date']).dt.date
df_index_prices = df_index_prices.drop_duplicates(subset=['Date', 'Instrument'], keep='last')

# Indeksin tuotot per instrument (RETURNS_FINAL)
df_index_prices['Daily Return'] = df_index_prices.groupby('Instrument')['Price Close'].pct_change()
df_index_prices['Daily Log Return'] = np.log(1 + df_index_prices['Daily Return'])

# Pivot (RETURNS_FINAL)
df_pivot = df_index_prices.pivot(index='Date', columns='Instrument', values=['Price Close', 'Daily Return', 'Daily Log Return'])

# Beta (RETURNS_AND_BETA_FINAL)
df_returns['Date'] = pd.to_datetime(df_returns['Date']).dt.date

beta_df = pd.merge(
    df_returns[['Date', 'Daily Total Return']].rename(columns={'Daily Total Return': 'NESTE'}),
    df_index_prices.loc[df_index_prices['Instrument'] == '.STOXXR', ['Date', 'Daily Return']].rename(columns={'Daily Return': 'STOXXR'}),
    on='Date',
    how='inner',
)

cov = beta_df['NESTE'].rolling(252).cov(beta_df['STOXXR'])
var = beta_df['STOXXR'].rolling(252).var()
beta_df['Beta_252'] = cov / var

print(f"\ndf_returns shape: {df_returns.shape}")
print(df_returns.head())
print(f"\ndf_index_prices shape: {df_index_prices.shape}")
print(df_index_prices.head())
print(f"\ndf_pivot shape: {df_pivot.shape}")
print(df_pivot.head())
print(f"\nbeta_df (viimeiset rivit):")
print(beta_df.tail())

# =====================================================================
# 3. RSI JA VOLATILITEETTI (RSI_AND_VOL_FINAL)
# =====================================================================
print("\n" + "=" * 60)
print("3. RSI JA VOLATILITEETTI")
print("=" * 60)

df_rsi = rd.get_data(
    universe=['NESTE.HE'],
    fields=["TR.PriceClose.date", "TR.PriceClose"],
    parameters=params,
)

# Daily return (RSI_AND_VOL_FINAL)
df_rsi['Daily_Return'] = df_rsi['Price Close'].pct_change()

# Vol 30d annualisoitu (RSI_AND_VOL_FINAL)
df_rsi['Vol_30d'] = df_rsi['Daily_Return'].rolling(30).std() * np.sqrt(252)

# RSI 30d (RSI_AND_VOL_FINAL — täsmälleen sama logiikka)
delta = df_rsi['Price Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(30).mean()
avg_loss = loss.rolling(30).mean()
rs = avg_gain / avg_loss
df_rsi['RSI_30d'] = 100 - (100 / (1 + rs))

print(f"\ndf_rsi shape: {df_rsi.shape}")
print(df_rsi.tail(10))

# =====================================================================
# 4. INDEKSIN FUNDAMENTAALIT (INDEX_FINAL)
# =====================================================================
print("\n" + "=" * 60)
print("4. INDEKSIN FUNDAMENTAALIT (.STOXX)")
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

# Index log return (INDEX_FINAL)
df_idx_fund['Index_Log_Return'] = np.log(df_idx_fund['Price Close'] / df_idx_fund['Price Close'].shift(1))

print(f"\nShape: {df_idx_fund.shape}")
print(df_idx_fund.head())
print(f"\nPuuttuvat arvot:")
print(df_idx_fund.isna().sum())

# =====================================================================
# 5. EURIBOR (EURIBOR_FINAL)
# =====================================================================
print("\n" + "=" * 60)
print("5. EURIBOR 3KK")
print("=" * 60)

df_euribor = rd.get_data(
    universe=["EURIBOR3MD="],
    fields=["TR.FIXINGVALUE", "TR.FIXINGVALUE.date"],
    parameters={"SDate": "2024-01-01", "EDate": "2026-03-05", "Frq": "D"},
)

print(f"\nShape: {df_euribor.shape}")
print(df_euribor.head())

# =====================================================================
# YHTEENVETO
# =====================================================================
print("\n" + "=" * 60)
print("YHTEENVETO: kaikki DataFramet")
print("=" * 60)
print(f"df_stock       : {df_stock.shape}  — fundamentaalit + div yield")
print(f"df_returns     : {df_returns.shape}  — tuotot (incl/excl osingot, log)")
print(f"df_index_prices: {df_index_prices.shape}  — .STOXX/.STOXXR hinnat + tuotot")
print(f"df_pivot       : {df_pivot.shape}  — indeksit pivotoituna")
print(f"beta_df        : {beta_df.shape}  — beta 252d")
print(f"df_rsi         : {df_rsi.shape}  — Vol 30d, RSI 30d")
print(f"df_idx_fund    : {df_idx_fund.shape}  — indeksin P/E, P/B, DY, log return")
print(f"df_euribor     : {df_euribor.shape}  — 3kk EURIBOR")

# =====================================================================
# TALLENNA CSV:t (nimessä "2" erottamista varten)
# =====================================================================
print("\n" + "=" * 60)
print("TALLENNETAAN CSV-TIEDOSTOT")
print("=" * 60)

df_stock.to_csv("data/01_raw/df_stock2.csv", index=False)
df_returns.to_csv("data/01_raw/df_returns2.csv", index=False)
df_index_prices.to_csv("data/01_raw/df_index_prices2.csv", index=False)
beta_df.to_csv("data/01_raw/df_beta2.csv", index=False)
df_rsi.to_csv("data/01_raw/df_rsi2.csv", index=False)
df_idx_fund.to_csv("data/01_raw/df_idx_fund2.csv", index=True)
df_euribor.to_csv("data/01_raw/df_euribor2.csv", index=False)

print("Tallennettu data/01_raw/ -kansioon:")
print("  df_stock2.csv")
print("  df_returns2.csv")
print("  df_index_prices2.csv")
print("  df_beta2.csv")
print("  df_rsi2.csv")
print("  df_idx_fund2.csv")
print("  df_euribor2.csv")
print("Valmis!")
