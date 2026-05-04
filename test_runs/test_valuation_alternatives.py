"""
Valuation-ryhmän vaihtoehdot: REF (Refinitiv) vs. SELF (itse laskettu).
Month-end data. Kaikki kombinaatiot rinnakkain CSV:ssä.

AJO: venv-v3/bin/python test_valuation_alternatives.py
"""

import refinitiv.data as rd
import pandas as pd
import numpy as np

rd.open_session()

PARAMS = {"SDate": "2008-10-08", "EDate": "2025-12-31", "Frq": "D", "Curn": "EUR"}
df_smoke = pd.read_csv("data/01_raw/smoke_stocks.csv")
RICS = sorted(df_smoke["Instrument"].dropna().unique().tolist())[:10]
print(f"RICs ({len(RICS)}): {RICS}\n")

fields = [
    "TR.PriceClose.date",
    "TR.PriceClose",
    "TR.CompanyMarketCap",
    "TR.SharesOutstanding",
    # E/P
    "TR.PE",
    "TR.EPSActValue",
    "TR.F.NetIncAfterTax",
    "TR.EPSFRActValue",
    # P/S
    "TR.PriceToSalesPerShare",
    "TR.F.TotRevenue",
    # P/CF
    "TR.PriceToCFPerShare",
    "TR.F.PricetoCFPPerShr",
    "TR.F.CF",
    "TR.F.NetCashFlowOp",
    # DivYield
    "TR.DividendYield",
    "TR.DPSActValue",
    # P/B
    "TR.PriceToBVPerShare",
]

print("Haetaan data...")
df = rd.get_data(RICS, fields, parameters=PARAMS)
rd.close_session()

# --- Nimetään sarakkeet itse (Refinitiv antaa vaihtelevia nimiä) ---
# Kenttien järjestys vastaa fields-listaa, +2 koska Instrument ja Date tulevat ensin
rename = {}
expected = [
    "Price", "MktCap", "Shares",
    "RAW_PE_ref", "RAW_EPS", "RAW_NetIncome", "RAW_EPS_FR",
    "RAW_PS_ref", "RAW_Revenue",
    "RAW_PCF_ref", "RAW_PCF_fiscal_ref", "RAW_CF", "RAW_OpCF",
    "RAW_DY_ref", "RAW_DPS",
    "RAW_PB_ref",
]
data_cols = [c for c in df.columns if c not in ("Instrument", "Date")]
print(f"\nSarakkeet Refinitivistä ({len(data_cols)}):")
for i, (orig, new) in enumerate(zip(data_cols, expected)):
    rename[orig] = new
    nn = df[orig].notna().sum()
    print(f"  {orig:<60} → {new:<20} non-NaN: {nn:,}")

# Jos sarake puuttuu kokonaan (Refinitiv ei palauttanut), expected on pidempi
if len(data_cols) < len(expected):
    print(f"\n  VAROITUS: {len(expected) - len(data_cols)} kenttää puuttuu!")

df = df.rename(columns=rename)

# --- Siivous ---
df["Date"] = pd.to_datetime(df["Date"])
for col in df.columns:
    if col not in ("Instrument", "Date"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --- ffill fundamentaalit per osake ---
df = df.sort_values(["Instrument", "Date"])
for col in df.columns:
    if col not in ("Instrument", "Date", "Price"):
        df[col] = df.groupby("Instrument")[col].ffill()

# --- Month-end ---
df["YM"] = df["Date"].dt.to_period("M")
me = df.groupby(["Instrument", "YM"]).tail(1).copy()
N = len(me)
print(f"\nRaakadata: {len(df):,} riviä → Month-end: {N:,}\n")

# =============================================================================
# Laske KAIKKI kombinaatiot
# =============================================================================
P = me["Price"]
MC = me["MktCap"]
SH = me["Shares"]

# ---------- E/P (isompi = halvempi) ----------
ep = {}
ep["REF_EP_1÷PE"]          = 1.0 / me["RAW_PE_ref"]
ep["SELF_EP_EPS÷Price"]    = me["RAW_EPS"] / P
ep["SELF_EP_EPSfr÷Price"]  = me["RAW_EPS_FR"] / P
ep["SELF_EP_NI÷MktCap"]    = me["RAW_NetIncome"] / MC
ep["SELF_EP_NIperSh÷Price"] = (me["RAW_NetIncome"] / SH) / P

# ---------- P/E (isompi = kalliimpi) ----------
pe = {}
pe["REF_PE"]                = me["RAW_PE_ref"].copy()
pe["SELF_PE_Price÷EPS"]     = P / me["RAW_EPS"]
pe["SELF_PE_Price÷EPSfr"]   = P / me["RAW_EPS_FR"]
pe["SELF_PE_MktCap÷NI"]     = MC / me["RAW_NetIncome"]
pe["SELF_PE_Price÷NIperSh"] = P / (me["RAW_NetIncome"] / SH)

# ---------- P/S ----------
ps = {}
ps["REF_PS"]                = me["RAW_PS_ref"].copy()
ps["SELF_PS_MktCap÷Rev"]    = MC / me["RAW_Revenue"]
ps["SELF_PS_Price÷RevPerSh"] = P / (me["RAW_Revenue"] / SH)

# ---------- P/CF ----------
pcf = {}
pcf["REF_PCF"]               = me["RAW_PCF_ref"].copy()
pcf["REF_PCF_fiscal"]        = me["RAW_PCF_fiscal_ref"].copy()
pcf["SELF_PCF_MktCap÷CF"]    = MC / me["RAW_CF"]
pcf["SELF_PCF_MktCap÷OpCF"]  = MC / me["RAW_OpCF"]
pcf["SELF_PCF_Price÷CFperSh"] = P / (me["RAW_CF"] / SH)
pcf["SELF_PCF_Price÷OpCFperSh"] = P / (me["RAW_OpCF"] / SH)

# ---------- DivYield ----------
dy = {}
dy["REF_DY"]           = me["RAW_DY_ref"] / 100
dy["SELF_DY_DPS÷Price"] = me["RAW_DPS"] / P

# ---------- 1/P/B ----------
pb = {}
pb["REF_invPB"] = 1.0 / me["RAW_PB_ref"]

# =============================================================================
# NaN-raportti
# =============================================================================
groups = [
    ("E/P", ep),
    ("P/E", pe),
    ("P/S", ps),
    ("P/CF", pcf),
    ("DivYield", dy),
    ("1/P/B", pb),
]

print(f"\n{'=' * 90}")
print(f"  MONTH-END NaN-VERTAILU  (N = {N:,})")
print(f"{'=' * 90}")

for label, grp in groups:
    if not grp:
        continue
    print(f"\n  {label}:")
    print(f"  {'Vaihtoehto':<30} {'non-NaN':>8} {'%NaN':>7} {'%Inf':>7} {'median':>12}")
    print(f"  {'-'*68}")
    for name, series in grp.items():
        clean = series.replace([np.inf, -np.inf], np.nan)
        nn = clean.notna().sum()
        pct_nan = (1 - nn / N) * 100
        n_inf = np.isinf(series).sum()
        pct_inf = n_inf / N * 100
        med = clean.median()
        med_str = f"{med:.4f}" if pd.notna(med) else "N/A"
        print(f"  {name:<30} {nn:>8,} {pct_nan:>6.1f}% {pct_inf:>6.1f}% {med_str:>12}")

# =============================================================================
# Korrelaatiot
# =============================================================================
print(f"\n{'=' * 90}")
print(f"  KORRELAATIOT (winsorisoidut 1%/99%)")
print(f"{'=' * 90}")

for label, grp in groups:
    if len(grp) < 2:
        continue
    df_corr = pd.DataFrame(grp).replace([np.inf, -np.inf], np.nan)
    for col in df_corr.columns:
        lo, hi = df_corr[col].quantile(0.01), df_corr[col].quantile(0.99)
        df_corr[col] = df_corr[col].clip(lo, hi)
    print(f"\n  {label}:")
    print(df_corr.corr().round(3).to_string())

# =============================================================================
# Tallenna CSV: ID → E/P → P/E → P/S → P/CF → DY → P/B → RAW
# =============================================================================
ordered_cols = ["Instrument", "Date", "YM", col_map["price"], col_map["mktcap"]]

for label, grp in groups:
    for name in grp.keys():
        me[name] = grp[name]
        ordered_cols.append(name)

# RAW-sarakkeet loppuun
raw_cols = [c for c in me.columns if c not in ordered_cols]
for rc in raw_cols:
    new_name = f"RAW_{rc}"
    me = me.rename(columns={rc: new_name})
    ordered_cols.append(new_name)

me = me[[c for c in ordered_cols if c in me.columns]]
me.to_csv("data/01_raw/valuation_alternatives_monthend.csv", index=False)
print(f"\nTallennettu: data/01_raw/valuation_alternatives_monthend.csv")
print(f"Sarakkeet ({len(me.columns)}):")
for i, c in enumerate(me.columns):
    print(f"  {i+1:>2}. {c}")
