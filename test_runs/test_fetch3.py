"""
Kaksi fetchiä: osakedata + makrodata → yhdistetty paneeli.
Rolling-arvot (Beta 252d, Vol 30d, RSI 30d) vaativat warmup-periodin,
joten dataa haetaan aikaisemmin kuin näytettävä väli.
Aja VS Codessa Jupyter-kernelillä.
"""

import refinitiv.data as rd
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

rd.open_session()

# =====================================================================
# ASETUKSET
# =====================================================================
# Näytettävä väli
NAYTA_ALKU = "2024-01-01"
NAYTA_LOPPU = "2026-03-05"

# Warmup: 252 kaupankäyntipäivää (beta) + marginaali → ~1.5 vuotta taaksepäin
HAKU_ALKU = "2022-07-01"
HAKU_LOPPU = NAYTA_LOPPU

parametrit = {
    "SDate": HAKU_ALKU,
    "EDate": HAKU_LOPPU,
    "Frq": "D",
    "Curn": "EUR",
}

# =====================================================================
# FETCH 1: OSAKEDATA
# =====================================================================
print("=" * 60)
print(f"FETCH 1: OSAKEDATA ({HAKU_ALKU} → {HAKU_LOPPU})")
print("=" * 60)

universe_osakkeet = ['NESTE.HE']

kentat_osakkeet = [
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

df_osake = rd.get_data(universe=universe_osakkeet, fields=kentat_osakkeet, parameters=parametrit)

# --- Duplikaattien poisto (sama Instrument + Date) ---
df_osake['Date'] = pd.to_datetime(df_osake['Date']).dt.date
df_osake = df_osake.drop_duplicates(subset=['Instrument', 'Date'], keep='first')
df_osake.reset_index(drop=True, inplace=True)

# --- GICS Sector & Sub-Industry: tulee vain ensimmäiselle riville → täytä kaikki rivit ---
# Refinitiv voi palauttaa tyhjät arvoina None, '', whitespace → korvataan kaikki NaN:ksi
for col in ['GICS Sector Name', 'GICS Sub-Industry Name']:
    df_osake[col] = df_osake[col].astype(str).str.strip()
    df_osake[col] = df_osake[col].replace({'': np.nan, 'None': np.nan, 'nan': np.nan, '<NA>': np.nan})
    df_osake[col] = df_osake.groupby('Instrument')[col].ffill().bfill()

# --- Tuotot (RETURNS_FINAL) ---
df_osake['Daily Total Return'] = df_osake['Daily Total Return'] / 100
df_osake['Daily Return excl. dividends'] = df_osake['Price Close'].pct_change()
df_osake['Log Daily Total Return'] = np.log(1 + df_osake['Daily Total Return'])
df_osake['Log Daily Return excl. dividends'] = np.log(1 + df_osake['Daily Return excl. dividends'])

# --- Div Yield manuaalinen (DIVIDEND_YIELD_FINAL) ---
df_osake['Div_Yield_Manual'] = (df_osake['Dividend Per Share - Actual'] / df_osake['Price Close']) * 100

# --- Vol 30d annualisoitu (RSI_AND_VOL_FINAL) ---
df_osake['Vol_30d'] = df_osake['Daily Return excl. dividends'].rolling(30).std() * np.sqrt(252)

# --- RSI 30d (RSI_AND_VOL_FINAL — täsmälleen sama logiikka) ---
delta = df_osake['Price Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(30).mean()
avg_loss = loss.rolling(30).mean()
rs = avg_gain / avg_loss
df_osake['RSI_30d'] = 100 - (100 / (1 + rs))

print(f"df_osake shape (warmup mukana): {df_osake.shape}")

# =====================================================================
# FETCH 2: MAKRODATA (indeksi + EURIBOR)
# =====================================================================
print("\n" + "=" * 60)
print(f"FETCH 2: MAKRODATA ({HAKU_ALKU} → {HAKU_LOPPU})")
print("=" * 60)

# --- Indeksin hinnat (.STOXX ja .STOXXR) ---
df_indeksi = rd.get_data(
    universe=[".STOXX", ".STOXXR"],
    fields=["TR.PriceClose.date", "TR.PriceClose"],
    parameters=parametrit,
)
df_indeksi['Date'] = pd.to_datetime(df_indeksi['Date']).dt.date
df_indeksi = df_indeksi.drop_duplicates(subset=['Date', 'Instrument'], keep='last')
df_indeksi['Daily_Return'] = df_indeksi.groupby('Instrument')['Price Close'].pct_change()
df_indeksi['Log_Return'] = np.log(1 + df_indeksi['Daily_Return'])

# --- Indeksin fundamentaalit (.STOXX) (INDEX_FINAL) ---
df_indeksi_funda = rd.get_history(
    universe=[".STOXX"],
    fields=[
        "TR.Index_PE_RTRS",
        "TR.Index_PRICE_TO_BOOK_RTRS",
        "TR.Index_DIV_YLD_RTRS",
        "TR.PriceClose",
    ],
    start=HAKU_ALKU,
    end=HAKU_LOPPU,
    interval="1D",
)
df_indeksi_funda = (
    df_indeksi_funda
    .reset_index()
    .drop_duplicates(subset=['Date'], keep='last')
    .sort_values('Date')
    .set_index('Date')
)
df_indeksi_funda['Index_Log_Return'] = np.log(
    df_indeksi_funda['Price Close'] / df_indeksi_funda['Price Close'].shift(1)
)

# --- EURIBOR (EURIBOR_FINAL) ---
df_euribor = rd.get_data(
    universe=["EURIBOR3MD="],
    fields=["TR.FIXINGVALUE", "TR.FIXINGVALUE.date"],
    parameters={"SDate": HAKU_ALKU, "EDate": HAKU_LOPPU, "Frq": "D"},
)
df_euribor['Date'] = pd.to_datetime(df_euribor['Date']).dt.date
df_euribor = df_euribor.drop_duplicates(subset=['Date'], keep='last')
df_euribor['Rf_daily'] = df_euribor['Fixing Value'] / 100 / 252

# --- Yhdistä makro yhdeksi taulukoksi ---
df_stoxxr = df_indeksi.loc[df_indeksi['Instrument'] == '.STOXXR', ['Date', 'Daily_Return']].copy()
df_stoxxr.rename(columns={'Daily_Return': 'STOXXR_Return'}, inplace=True)

df_indeksi_funda_flat = df_indeksi_funda.reset_index()
df_indeksi_funda_flat['Date'] = pd.to_datetime(df_indeksi_funda_flat['Date']).dt.date
df_indeksi_funda_flat.rename(columns={
    'Calculated PE Ratio': 'Index_PE',
    'Calculated Price to Book': 'Index_PB',
    'Calculated Index Dividend Yield': 'Index_DY',
    'Price Close': 'Index_Price',
}, inplace=True)

df_makro = pd.merge(df_stoxxr, df_euribor[['Date', 'Rf_daily']], on='Date', how='outer')
df_makro = pd.merge(df_makro, df_indeksi_funda_flat[['Date', 'Index_PE', 'Index_PB', 'Index_DY', 'Index_Price', 'Index_Log_Return']], on='Date', how='outer')
df_makro.sort_values('Date', inplace=True)

print(f"df_makro shape: {df_makro.shape}")

# =====================================================================
# BETA 252d (RETURNS_AND_BETA_FINAL)
# =====================================================================
print("\n" + "=" * 60)
print("BETA 252d")
print("=" * 60)

df_osake['Date_dt'] = pd.to_datetime(df_osake['Date']).dt.date

beta_merge = pd.merge(
    df_osake[['Date_dt', 'Daily Total Return']].rename(columns={'Date_dt': 'Date', 'Daily Total Return': 'Stock_Return'}),
    df_stoxxr,
    on='Date',
    how='inner',
)
cov = beta_merge['Stock_Return'].rolling(252).cov(beta_merge['STOXXR_Return'])
var = beta_merge['STOXXR_Return'].rolling(252).var()
beta_merge['Beta_252d'] = cov / var

df_osake = pd.merge(
    df_osake,
    beta_merge[['Date', 'Beta_252d']].rename(columns={'Date': 'Date_dt'}),
    on='Date_dt',
    how='left',
)

print(f"Beta laskettu. Ensimmäinen ei-NaN beta: {beta_merge['Beta_252d'].first_valid_index()}")

# =====================================================================
# YHDISTETTY PANEELI
# =====================================================================
print("\n" + "=" * 60)
print("YHDISTETTY PANEELI (ennen leikkausta)")
print("=" * 60)

df_paneeli = pd.merge(
    df_osake,
    df_makro,
    left_on='Date_dt',
    right_on='Date',
    how='left',
    suffixes=('', '_makro'),
)

# Excess return
df_paneeli['Excess_Return'] = df_paneeli['Daily Return excl. dividends'] - df_paneeli['Rf_daily']

# Siivoa
df_paneeli.drop(columns=['Date_dt', 'Date_makro'], errors='ignore', inplace=True)

print(f"Ennen leikkausta: {df_paneeli.shape}")
print(f"NaN-rivit Beta_252d: {df_paneeli['Beta_252d'].isna().sum()}")
print(f"NaN-rivit Vol_30d:   {df_paneeli['Vol_30d'].isna().sum()}")
print(f"NaN-rivit RSI_30d:   {df_paneeli['RSI_30d'].isna().sum()}")

# =====================================================================
# LEIKKAA: näytä vain haluttu väli
# =====================================================================
print("\n" + "=" * 60)
print(f"LOPULLINEN PANEELI ({NAYTA_ALKU} → {NAYTA_LOPPU})")
print("=" * 60)

df_paneeli['Date'] = pd.to_datetime(df_paneeli['Date'])
df_lopullinen = df_paneeli.loc[df_paneeli['Date'] >= NAYTA_ALKU].copy()
df_lopullinen.reset_index(drop=True, inplace=True)

print(f"\nShape: {df_lopullinen.shape}")

print(f"\nKaikki sarakkeet ({len(df_lopullinen.columns)} kpl):")
for i, col in enumerate(df_lopullinen.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\nEnsimmäiset rivit:")
print(df_lopullinen.head())

print(f"\nViimeiset rivit:")
print(df_lopullinen.tail())

print(f"\nPuuttuvat arvot:")
print(df_lopullinen.isna().sum())

print(f"\nPerustilastot:")
print(df_lopullinen.describe())

# Tallenna
df_lopullinen.to_csv("data/01_raw/df_paneeli2.csv", index=False)
df_makro.to_csv("data/01_raw/df_makro2.csv", index=False)
print(f"\nTallennettu: df_paneeli2.csv, df_makro2.csv → data/01_raw/")
print(f"Warmup-data ({HAKU_ALKU} → {NAYTA_ALKU}) käytetty vain rolling-laskentaan, ei näy lopullisessa.")
