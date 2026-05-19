"""
Featurelaskennat osake- ja indeksitasolla.
Jokainen funktio ottaa sisään DataFramen ja palauttaa sen uudella sarakkeella.
"""

import pandas as pd
import numpy as np


# Refinitiv-fundamentals can come back as object/string dtype when a chunk
# fetch returns all-NaN for a column. Downstream arithmetic (e.g. Market Cap
# / Cash Flow) then crashes with "unsupported operand type(s) for /". Coerce
# upstream once so every compute_* function trusts numeric dtypes.
NUMERIC_FUNDAMENTAL_COLUMNS = [
    "Cash Flow",
    "Net Cash Flow from Operating Activities",
    "Cash Flow from Operations per Share",
    "Earnings Per Share - Actual",
    "Earnings Per Share Reported - Actual",
    "EBITDA - Actual",
    "Earnings before Interest Taxes Depreciation & Amortization",
    "Shareholders Equity - Common",
    "Total Debt",
    "Outstanding Shares",
    "Company Market Cap",
    "Price Close",
    "Price To Book Value Per Share (Daily Time Series Ratio)",
    "Price To Sales Per Share (Daily Time Series Ratio)",
    "Price To Cash Flow Per Share (Daily Time Series Ratio)",
    "Price to Book Value per Share",
    "Dividend Per Share - Actual",
    "Return on Average Common Equity - %",
]

QUARTERLY_NUMERIC_COLUMNS = [
    "NetIncome_Q",
    "EPSfr_Q",
    "NetCashFlowOp_Q",
    "NetCFOpPerShr_Q",
]


def _coerce_numeric_columns(df, cols):
    """Cast known numeric columns to float64 in place, mapping unparseable values to NaN."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# VALUATION-ryhmä
# ---------------------------------------------------------------------------

def compute_ep(df):
    """
    Legacy E/P = trailing 12M basic EPS / Price Close.

    Käyttää TR.EPSActValue ("Earnings Per Share - Actual") -kenttää ja summaa
    raportoidut EPS-arvot edeltäneeltä 365 päivältä. Säilytetään
    vertailufeaturena, mutta ei ylikirjoiteta nykyistä LTM-pohjaista E/P:tä.
    """
    df["Date"] = pd.to_datetime(df["Date"])
    eps_col = "Earnings Per Share - Actual"
    eps_reports = (
        df.loc[df[eps_col].notna(), ["Instrument", "Date", eps_col]]
        .sort_values(["Instrument", "Date"])
        .reset_index(drop=True)
    )

    def _ttm(group):
        group = group.set_index("Date").sort_index()
        group["EPS_TTM_legacy"] = group[eps_col].rolling("365D").sum()
        return group.reset_index()

    eps_ttm = eps_reports.groupby("Instrument", group_keys=False).apply(_ttm)
    df = df.sort_values(["Instrument", "Date"])
    parts = []
    for instrument, group in df.groupby("Instrument", group_keys=False):
        ttm_slice = (
            eps_ttm.loc[eps_ttm["Instrument"] == instrument, ["Date", "EPS_TTM_legacy"]]
            .sort_values("Date")
        )
        merged = pd.merge_asof(
            group.sort_values("Date"),
            ttm_slice,
            on="Date",
            direction="backward",
        )
        parts.append(merged)

    df = pd.concat(parts, ignore_index=True)
    df["E/P_legacy"] = df["EPS_TTM_legacy"] / df["Price Close"]
    df["P/E_legacy"] = df["Price Close"] / df["EPS_TTM_legacy"]
    return df


def compute_pe_ltm(df):
    """
    P/E ja E/P käyttäen Refinitivin LTM-kenttää suoraan.

    Lähde: TR.EPSFRActValue(Period=LTM) — fully reported (diluted) EPS,
    laskettuna last twelve months -periodille Refinitivin omalla logiikalla.
    PE_fix.ipynb:n PE_trial: tämä variantti osuu lähimmäs Refinitivin
    omaa raportoitua PE-lukua.

    Ei fallbackia — jos LTM-arvo puuttuu (esim. uusi listaus, datakatkos),
    P/E ja E/P ovat NaN. Käytä compute_pe_ttm_fallback:ia jos tarvitaan
    parempaa kattavuutta.
    """
    eps_col = "Earnings Per Share Reported - Actual"
    df["P/E"] = df["Price Close"] / df[eps_col]
    df["E/P"] = df[eps_col] / df["Price Close"]
    return df


def compute_pe_ttm_fallback(df, df_quarterly_eps):
    """
    P/E_ff ja E/P_ff fallback-ketjulla:
        1. Refinitivin LTM EPSfr (sama kuin compute_pe_ltm)
        2. → oma 4Q rolling TTM EPSfr (kvartaalisarjasta summattu)
        3. → NI/Shares (TTM Net Income / Outstanding Shares)

    df_quarterly_eps: long-format DataFrame ['Instrument', 'QDate',
    'EPSfr_Q', 'NetIncome_Q'] (data_fetch.fetch_quarterly_eps_for_pe).

    Levitys päivätasolle merge_asof(direction="backward") per osake →
    ei look-ahead biasia.

    PE_fix.ipynb:n combine_first-logiikka. Ratkaisee tilanteita joissa
    Refinitivin valmis LTM-arvo on NaN mutta kvartaaliraportit ovat
    saatavilla.
    """
    # 1. Rakenna kvartaalitason 4Q rolling TTM-summat per osake
    q = df_quarterly_eps.sort_values(["Instrument", "QDate"]).copy()
    q["TTM_EPSfr"] = (
        q.groupby("Instrument")["EPSfr_Q"]
        .rolling(4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )
    q["TTM_NetIncome"] = (
        q.groupby("Instrument")["NetIncome_Q"]
        .rolling(4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # 2. Levitä TTM-arvot päivätasolle per osake (merge_asof backward)
    df = df.sort_values(["Instrument", "Date"])
    parts = []
    for instrument, group in df.groupby("Instrument", group_keys=False):
        q_slice = (
            q.loc[q["Instrument"] == instrument, ["QDate", "TTM_EPSfr", "TTM_NetIncome"]]
            .rename(columns={"QDate": "Date"})
            .sort_values("Date")
        )
        if q_slice.empty:
            group = group.copy()
            group["TTM_EPSfr"] = np.nan
            group["TTM_NetIncome"] = np.nan
        else:
            group = pd.merge_asof(
                group.sort_values("Date"),
                q_slice,
                on="Date",
                direction="backward",
            )
        parts.append(group)
    df = pd.concat(parts, ignore_index=True)

    # 3. NI/Shares fallback
    df["TTM_EPS_from_NI"] = df["TTM_NetIncome"] / df["Outstanding Shares"]

    # 4. Combine_first -ketju: LTM → 4Q TTM → NI/Shares
    eps_ltm_col = "Earnings Per Share Reported - Actual"
    eps_chain = (
        df[eps_ltm_col]
        .combine_first(df["TTM_EPSfr"])
        .combine_first(df["TTM_EPS_from_NI"])
    )

    df["P/E_ff"] = df["Price Close"] / eps_chain
    df["E/P_ff"] = eps_chain / df["Price Close"]
    return df


def compute_inv_pb(df):
    """1/P/B = 1 / Price-to-Book."""
    df["1/P/B"] = 1.0 / df["Price To Book Value Per Share (Daily Time Series Ratio)"]
    return df


def compute_neg_ps(df):
    """-P/S."""
    df["-P/S"] = -df["Price To Sales Per Share (Daily Time Series Ratio)"]
    return df


def compute_neg_pcf(df):
    """
    Legacy -P/CF = -(Market Cap / Cash Flow), TR.F.CF-pohjainen.

    Säilytetään rinnakkaisena vertailufeaturena uusien Operating Cash Flow
    -varianttien kanssa. TR.F.CF on fiscal-vuoden cash flow -kenttä, joten se
    forward-fillataan per osake ennen päivittäistä Market Cap -jakolaskua.
    """
    df = df.sort_values(["Instrument", "Date"])
    df["Cash Flow"] = df.groupby("Instrument")["Cash Flow"].ffill()
    df["-P/CF_legacy"] = -(df["Company Market Cap"] / df["Cash Flow"])
    return df


def compute_pcf_op(df):
    """
    Tuotannon -P/CF Operating Cash Flow LTM:llä — kaksi rinnakkaista varianttia.

    -P/CF_op_LTM   = -(Market Cap / Net Cash Flow from Operating Activities (LTM))
    -P/CF_opps_LTM = -(Price       / Cash Flow from Operations per Share     (LTM))

    Vertailussa REF_PCF_daily:hin (pcf_compare_to_ref.py):
      - op_LTM:   median rel diff −0.15%, 32% riveistä <1% poikkeamalla
      - opps_LTM: median rel diff −2.59%, Spearman 0.940 (op:n 0.927)
    Molemmat säilytetään tuotantosarakkeina; valinta tehdään mallin tasolla.

    Lähdekentät (data_fetch.py: TR.F.NetCashFlowOp(Period=LTM) ja
    TR.F.NetCFOpPerShr(Period=LTM)) ovat LTM-arvoja Refinitivin rolling 4Q
    -laskennalla → ei tarvitse omaa ffill-logiikkaa.

    HUOM: jos Operating CF < 0 (tappiolliset firmat), P/CF on negatiivinen
    ja -P/CF positiivinen.
    """
    opcf_col = "Net Cash Flow from Operating Activities"
    opcfps_col = "Cash Flow from Operations per Share"

    df["-P/CF_op_LTM"] = -(df["Company Market Cap"] / df[opcf_col])
    df["-P/CF_opps_LTM"] = -(df["Price Close"] / df[opcfps_col])
    return df


def compute_pcf_ttm_fallback(df, df_quarterly_pcf):
    """
    -P/CF_ff fallback-ketjulla:
        1. Refinitivin LTM operating CF: -P/CF_op_LTM
        2. → oma 4Q rolling TTM operating CF kvartaalisarjasta
        3. → oma 4Q rolling TTM operating CF per share kvartaalisarjasta

    df_quarterly_pcf: long-format DataFrame ['Instrument', 'QDate',
    'NetCashFlowOp_Q', 'NetCFOpPerShr_Q'] (data_fetch.fetch_quarterly_pcf_for_pcf).

    Levitys päivätasolle merge_asof(direction="backward") per osake →
    ei look-ahead biasia. Legacy TR.F.CF -pohjaista -P/CF:ää ei käytetä
    fallbackina, koska se on eri cash flow -määritelmä.
    """
    q = df_quarterly_pcf.sort_values(["Instrument", "QDate"]).copy()
    q["TTM_NetCashFlowOp"] = (
        q.groupby("Instrument")["NetCashFlowOp_Q"]
        .rolling(4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )
    q["TTM_NetCFOpPerShr"] = (
        q.groupby("Instrument")["NetCFOpPerShr_Q"]
        .rolling(4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df = df.sort_values(["Instrument", "Date"])
    parts = []
    for instrument, group in df.groupby("Instrument", group_keys=False):
        q_slice = (
            q.loc[
                q["Instrument"] == instrument,
                ["QDate", "TTM_NetCashFlowOp", "TTM_NetCFOpPerShr"],
            ]
            .rename(columns={"QDate": "Date"})
            .sort_values("Date")
        )
        if q_slice.empty:
            group = group.copy()
            group["TTM_NetCashFlowOp"] = np.nan
            group["TTM_NetCFOpPerShr"] = np.nan
        else:
            group = pd.merge_asof(
                group.sort_values("Date"),
                q_slice,
                on="Date",
                direction="backward",
            )
        parts.append(group)
    df = pd.concat(parts, ignore_index=True)

    df["-P/CF_op_TTM"] = -(df["Company Market Cap"] / df["TTM_NetCashFlowOp"])
    df["-P/CF_opps_TTM"] = -(df["Price Close"] / df["TTM_NetCFOpPerShr"])
    df["-P/CF_ff"] = (
        df["-P/CF_op_LTM"]
        .combine_first(df["-P/CF_op_TTM"])
        .combine_first(df["-P/CF_opps_TTM"])
    )
    return df


def compute_pcf_refinitiv(df):
    """
    Refinitivin oma P/CF -ratio (TR.PriceToCFPerShare, Daily Time Series).

    Vertailufeature laskettujen P/CF_op / P/CF_ps -varianttien rinnalle —
    sama logiikka kuin PCF_test2.ipynb:n viimeisessä taulussa (REF_PCF).
    CLAUDE.md mainitsee että tällä on ~7% NaN-kattokatkos mid-cap-osakkeissa,
    joten se ei ole päämittari mutta toimii diagnostiikkana.
    """
    ref_col = "Price To Cash Flow Per Share (Daily Time Series Ratio)"
    df["P/CF_refinitiv"] = df[ref_col]
    df["-P/CF_refinitiv"] = -df[ref_col]
    return df


def compute_dividend_yield_trailing(df):
    """
    Trailing 12M Dividend Yield = 12kk kumulatiivinen DPS / edellisen kuun lopun hinta.
    Käytetään merge_asof:ia look-ahead biasin estoon (CLAUDE.md konventio).
    """
    df["Date"] = pd.to_datetime(df["Date"])

    # Kuukauden lopun hinta per osake
    df_month_end = (
        df.groupby("Instrument")
        .resample("ME", on="Date")["Price Close"]
        .last()
        .reset_index()
        .rename(columns={"Price Close": "Price_MonthEnd"})
    )

    # Trailing 12kk kumulatiivinen DPS per osake
    df_month_end_dps = (
        df.groupby("Instrument")
        .resample("ME", on="Date")["Dividend Per Share - Actual"]
        .last()
        .reset_index()
    )
    df_month_end_dps["DPS_12M"] = (
        df_month_end_dps
        .groupby("Instrument")["Dividend Per Share - Actual"]
        .rolling(12, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # Yhdistä ja laske yield edellisen kuun hinnalla
    df_month_end = df_month_end.merge(
        df_month_end_dps[["Instrument", "Date", "DPS_12M"]],
        on=["Instrument", "Date"],
        how="left",
    )
    df_month_end["Price_Prev"] = df_month_end.groupby("Instrument")["Price_MonthEnd"].shift(1)
    df_month_end["DivYield_12M"] = df_month_end["DPS_12M"] / df_month_end["Price_Prev"]

    # merge_asof: jokaiselle päivälle viimeisin saatavilla oleva kuukausiarvo
    df = df.sort_values(["Instrument", "Date"])
    result_parts = []
    for instrument, group in df.groupby("Instrument"):
        monthly = df_month_end.loc[
            df_month_end["Instrument"] == instrument,
            ["Date", "DivYield_12M"],
        ].sort_values("Date")
        merged = pd.merge_asof(
            group.sort_values("Date"),
            monthly,
            on="Date",
            direction="backward",
        )
        result_parts.append(merged)

    df = pd.concat(result_parts, ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# QUALITY-ryhmä
# ---------------------------------------------------------------------------

# Debt / Market Cap 
def compute_neg_debt_to_mktcap(df):
    """
    -Debt/MktCap = -(Total Debt / Market Cap).

    Total Debt on kvartaaliraportti → forward-fill per osake jotta
    saadaan päivittäinen arvo. Market Cap on jo päivittäinen.

    Korvattu aiempi SmartNetDebtToMarketCap-kenttä — syynä 44% NaN
    mid-cap-hännästä. Molemmat tarvittavat raakakentät (TotalDebt,
    CompanyMarketCap) ovat jo fetchattuja, joten oma laskenta on
    ilmainen kattavuuden parannus ja on linjassa CLAUDE.md:n
    Quality-ryhmän määrityksen kanssa ("-Debt/MktCap", ei
    "-Net Debt/MktCap").
    """
    df = df.sort_values(["Instrument", "Date"])
    df["Total Debt"] = df.groupby("Instrument")["Total Debt"].ffill()
    df["-Debt/MktCap"] = -(df["Total Debt"] / df["Company Market Cap"])
    return df

# Book-to-Market 
def compute_book_to_market(df):
    """Book-to-Market = 1 / P/B (Refinitiv field TR.F.PriceToBookValuePerShr)."""
    df = df.sort_values(["Instrument", "Date"])

    # Forward-fill annual P/B across monthly observations
    df["Price to Book Value per Share"] = (
        df.groupby("Instrument")["Price to Book Value per Share"].ffill(limit=12)
    )

    # Invert to get B/M, with safety against zero/negative values
    pb = df["Price to Book Value per Share"].replace(0, float("nan"))
    df["BookToMarket"] = 1 / pb
    df["BookToMarket"] = df["BookToMarket"].where(df["BookToMarket"] > 0)

    return df

# "Return On Average Common Equity %" ja "Gross Profit / Total Assets" ???

# Operating Profitablity
# Operating Profitablity
def compute_operating_profitability(df):
    """
    Compute Operating Profitability = EBITDA(t) / Common Shareholders' Equity(t-1).

    The denominator is lagged one fiscal year relative to the numerator. This
    follows the Fama and French (2015) RMW factor construction and the European
    asset pricing literature (Hanauer et al., 2022). The lag corrects for the
    mechanical distortion where retained earnings from period t are already
    reflected in period-t equity through the accounting identity. Without the
    lag, highly profitable firms have their ratios compressed more than less
    profitable firms, biasing the cross-sectional ranking that the factor is
    meant to capture.

    EBITDA construction (with fallback for coverage):
        Primary:  TR.EBITDAActValue ("EBITDA - Actual")  — I/B/E/S adjusted
        Fallback: TR.F.EBITDA ("Earnings before Interest Taxes Depreciation & Amortization") — fundamentals pipeline

    The primary source uses I/B/E/S analyst-normalized EBITDA, which provides
    consistent treatment of one-time items across firms. Where the primary
    is unavailable (typically smaller firms or earlier periods with weaker
    analyst coverage), the function falls back to the fundamentals-pipeline
    EBITDA computed from reported line items. The two definitions differ in
    their treatment of one-offs but capture the same economic concept; the
    fallback applies to a minority of observations and meaningfully improves
    coverage in the European universe.

    Equity field:
        TR.F.ShHoldEqCom ("Shareholders Equity - Common")

    Both EBITDA sources and the equity field are reported annually. On the
    daily panel, each annual value is forward-filled within firm for up to
    252 trading days (one fiscal year) to populate non-reporting days. The
    cap prevents stale values from propagating beyond a fiscal year for
    firms that have stopped reporting.

    The fiscal-year lag is implemented as shift(252) on the daily panel.
    Each annual value occupies ~252 consecutive trading-day rows after the
    forward-fill, so a 252-day shift moves the denominator back exactly one
    fiscal year (numerator from FY t, denominator from FY t-1).

    Negative common equity (rare, but occurs after large writedowns or
    accumulated losses) is treated as missing rather than allowed to flip
    the sign of OP. A negative-equity firm with positive EBITDA would
    otherwise produce a negative OP, which is not meaningful as a
    profitability signal.

    References:
        Fama, E. F., & French, K. R. (2015). A five-factor asset pricing
        model. Journal of Financial Economics, 116(1), 1-22.
    """
    # Sort required before any groupby().shift(); wrong order produces incorrect lag values.
    df = df.sort_values(["Instrument", "Date"])

    # EBITDA with fallback: use the I/B/E/S adjusted value where populated,
    # fall back to the fundamentals-pipeline EBITDA where the primary is NaN.
    primary_ebitda = df.get("EBITDA - Actual", pd.Series(np.nan, index=df.index))
    fallback_ebitda = df.get(
        "Earnings before Interest Taxes Depreciation & Amortization",
        df.get("EBITDA", pd.Series(np.nan, index=df.index)),
    )
    ebitda_combined = primary_ebitda.where(primary_ebitda.notna(), fallback_ebitda)

    # Forward-fill the combined EBITDA across non-reporting days, capped at one fiscal year.
    ebitda_filled = ebitda_combined.groupby(df["Instrument"]).ffill(limit=252)

    # Same forward-fill logic for equity, capped at one fiscal year.
    equity_filled = df.groupby("Instrument")["Shareholders Equity - Common"].ffill(limit=252)

    # shift(252) on the daily panel moves the denominator back one full fiscal year.
    # Negative or zero equity → NaN. A negative denominator would flip the sign
    # of OP, producing a meaningless signal. Affects ~1% of firm-day observations.
    equity_lag = equity_filled.groupby(df["Instrument"]).shift(252)
    equity_lag = equity_lag.where(equity_lag > 0)

    df["OperatingProfitability"] = ebitda_filled / equity_lag
    return df


# ---------------------------------------------------------------------------
# MOMENTUM-ryhmä
# ---------------------------------------------------------------------------

# Momentum 1M: viimeisen ~21 kaupankäyntipäivän tuotto
def compute_mom_1m(df):
    """1M Momentum: viimeisen ~21 kaupankäyntipäivän tuotto."""
    df["MOM_1M"] = df.groupby("Instrument")["Price Close"].pct_change(21)
    return df

# Momentum 12M: tuotto päivistä t-252 → t-21 (skip t-1)
def compute_mom_12m(df):
    """
    12M Momentum (skip t-1): tuotto päivistä t-252 → t-21.
    Jegadeesh & Titman (1993), Gu et al. (2020).
    """
    df["Price_lag21"] = df.groupby("Instrument")["Price Close"].shift(21)
    df["Price_lag252"] = df.groupby("Instrument")["Price Close"].shift(252)
    df["MOM_12M"] = (df["Price_lag21"] - df["Price_lag252"]) / df["Price_lag252"]
    df.drop(columns=["Price_lag21", "Price_lag252"], inplace=True)
    return df

# GICS 11 → 6 ryhmän aggregointi (eurooppalainen STOXX 600 -universumi).
# Korvaa aiemman 11-sektorin leave-one-out -benchmarkin tiheämmillä ryhmillä,
# jotta tail-sektorit (esim. Real Estate, Utilities) eivät jää 1-osakkeen
# kokoisiksi pienissä otoksissa. Mappi on tutkimusasetelman päätös, ei
# perustu Fama-French US -industry-luokitukseen.
GICS_TO_SECTOR_GROUP = {
    "Financials":             "Financials",
    "Industrials":            "Industrials & Materials",
    "Materials":              "Industrials & Materials",
    "Consumer Discretionary": "Consumer",
    "Consumer Staples":       "Consumer",
    "Health Care":            "Health Care",
    "Information Technology": "Technology & Communication",
    "Communication Services": "Technology & Communication",
    "Energy":                 "Real Assets & Utilities",
    "Utilities":              "Real Assets & Utilities",
    "Real Estate":            "Real Assets & Utilities",
}


def compute_sector_group(df):
    """
    Lisää Sector_Group-sarakkeen GICS_TO_SECTOR_GROUP-mapilla.
    Tuntematon GICS-sektori → NaN.
    """
    df["Sector_Group"] = df["GICS Sector Name"].map(GICS_TO_SECTOR_GROUP)
    return df

_SECTOR_DUMMY_GROUPS = [
    "Financials",
    "Industrials & Materials",
    "Consumer",
    "Health Care",
    "Technology & Communication",
]

def compute_sector_dummies(df, sectors=None, names=None):
    """5 binääristä Sector_Group-dummyä; referenssiryhmä Real Assets & Utilities.

    Parametrit sectors ja names hyväksytään yhteensopivuuden takia mutta
    jätetään huomiotta — dummyt lasketaan aina _SECTOR_DUMMY_GROUPS-listasta.
    """
    for group in _SECTOR_DUMMY_GROUPS:
        col = "Sector_" + group.replace(" & ", "_").replace(" ", "_")
        df[col] = (df["Sector_Group"] == group).astype(int)
    return df

# Stock vs Sector Return Momentum: osakkeen momentum miinus saman Sector_Groupin
# muiden osakkeiden equal-weighted leave-one-out -keskiarvo. Tuotetaan kaksi
# varianttia: 12-1M (Stock_vs_Sector_12M_1M) ja 1M (Stock_vs_Sector_1M).
def compute_stock_vs_sector_return(df):
    """
    Stock vs. Sector return: osakekohtainen momentum miinus Sector_Groupin
    leave-one-out equal-weighted keskiarvo. Tuottaa kaksi feature-saraketta:

        Stock_vs_Sector_12M_1M = MOM_12M - sektoriryhmän_LOO_avg(MOM_12M)
        Stock_vs_Sector_1M  = MOM_1M   - sektoriryhmän_LOO_avg(MOM_1M)

    Eristää osakekohtaisen momentumin sektoriryhmän momentumista
    (Moskowitz & Grinblatt 1999). Leave-one-out estää että iso osake
    benchmarkkaa itseään mekaanisesti omaa keskiarvoaan vasten.

    Sektoriryhmittelyssä yhden osakkeen ryhmän osakkeet → NaN
    (nimittäjä n−1 = 0).

    Vaatii että compute_sector_group, compute_mom_12m ja compute_mom_1m
    on ajettu ennen tätä.
    """
    df = df.sort_values(["Instrument", "Date"])

    for mom_col, out_col in [
        ("MOM_12M", "Stock_vs_Sector_12M_1M"),
        ("MOM_1M", "Stock_vs_Sector_1M"),
    ]:
        stats = (
            df.groupby(["Date", "Sector_Group"])[mom_col]
            .agg(sector_sum="sum", sector_n="count")
            .reset_index()
        )
        df = df.merge(stats, on=["Date", "Sector_Group"], how="left")
        sector_avg_excl = (
            (df["sector_sum"] - df[mom_col])
            / (df["sector_n"] - 1).replace(0, float("nan"))
        )
        df[out_col] = df[mom_col] - sector_avg_excl
        df.drop(columns=["sector_sum", "sector_n"], inplace=True)

    return df

# Relative Strength Index (RSI)
def compute_rsi(df, window=30):
    """
    RSI 30d — sama laskenta kuin RSI_AND_VOL_FINAL.ipynb:ssä.
    """
    def _rsi(group):
        delta = group["Price Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        group[f"RSI_{window}d"] = 100 - (100 / (1 + rs))
        return group

    df = df.groupby("Instrument", group_keys=False).apply(_rsi)
    return df

def compute_daily_return(df):
    """
    Osakekohtainen päivätuotto (Price Close -pohjainen, ilman osinkoja).
    Tarpeellinen Beta-, IdioVol- ja Excess_Return -laskennoille.
    """
    df["Daily_Return"] = df.groupby("Instrument")["Price Close"].pct_change()
    return df

# ---------------------------------------------------------------------------
# RISK-ryhmä
# ---------------------------------------------------------------------------

def _prepare_market_returns(df_index):
    idx = df_index.loc[df_index["Instrument"] == ".STOXXR"].copy()
    idx["Date"] = pd.to_datetime(idx["Date"]).dt.normalize()
    idx["Price Close"] = pd.to_numeric(idx["Price Close"], errors="coerce")
    idx = idx.dropna(subset=["Date", "Price Close"])
    if len(idx) < 2:
        raise ValueError(
            ".STOXXR benchmark series has fewer than 2 valid price rows; "
            "Beta_252d and -IdioVol cannot be computed."
        )
    idx = (
        idx.drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    idx["Index_Return"] = idx["Price Close"].pct_change()
    return idx[["Date", "Index_Return"]]


def _valid_stock_market_return_pairs(group, idx):
    stock = group[["Date_dt", "Daily_Return"]].rename(columns={"Date_dt": "Date"})
    paired = pd.merge(stock, idx, on="Date", how="inner")
    return (
        paired.dropna(subset=["Daily_Return", "Index_Return"])
        .sort_values("Date")
        .reset_index(drop=True)
    )


def _assign_latest_feature_to_stock_dates(group, feature_by_date, feature_col):
    values = np.full(len(group), np.nan, dtype=float)
    feature_by_date = feature_by_date.dropna(subset=[feature_col])
    if feature_by_date.empty:
        return values

    left = pd.DataFrame({
        "_pos": np.arange(len(group)),
        "Date": pd.to_datetime(group["Date_dt"]).to_numpy(),
    }).sort_values("Date")
    right = feature_by_date[["Date", feature_col]].sort_values("Date")

    matched = pd.merge_asof(
        left,
        right,
        on="Date",
        direction="backward",
        tolerance=pd.Timedelta(days=7),
    )
    values[matched["_pos"].to_numpy()] = matched[feature_col].to_numpy()
    return values

def compute_volatility(df, window=30):
    """
    Annualisoitu volatiliteetti 30d — sama kuin RSI_AND_VOL_FINAL.ipynb:ssä.
    """
    def _vol(group):
        daily_ret = group["Price Close"].pct_change()
        group[f"Vol_{window}d"] = daily_ret.rolling(window).std() * np.sqrt(252)
        return group

    df = df.groupby("Instrument", group_keys=False).apply(_vol)
    df[f"-Vol_{window}d"] = -df[f"Vol_{window}d"]
    return df

def compute_beta(df, df_index, window=252):
    """
    Rolling 252d Beta vs. .STOXXR.

    The window is counted on valid stock-market return pairs, not raw stock
    rows. This avoids exchange-calendar mismatches turning an otherwise valid
    252-observation estimate into NaN.
    """
    idx = _prepare_market_returns(df_index)

    df["Date_dt"] = pd.to_datetime(df["Date"]).dt.normalize()

    def _beta(group):
        paired = _valid_stock_market_return_pairs(group, idx)
        cov = paired["Daily_Return"].rolling(window).cov(paired["Index_Return"])
        var = paired["Index_Return"].rolling(window).var()
        paired["Beta_252d"] = cov / var

        group = group.copy()
        group["Beta_252d"] = _assign_latest_feature_to_stock_dates(
            group,
            paired[["Date", "Beta_252d"]],
            "Beta_252d",
        )
        return group

    df = df.groupby("Instrument", group_keys=False).apply(_beta)
    df["-Beta_252d"] = -df["Beta_252d"]
    df.drop(columns=["Date_dt"], inplace=True)
    return df


def compute_idiosyncratic_vol(df, df_index, window=252):
    """
    Idiosyncratic volatility: standard deviation of residuals from a rolling
    CAPM regression of the stock's excess return on the market's excess return.

    Vectorized using the rolling-OLS identity:
        Var(residual) = Var(y) - Cov(y, x)^2 / Var(x)

    This is mathematically equivalent to fitting OLS at each rolling window
    position but runs in O(n) per stock instead of O(n × window). Speedup
    is roughly 100-250× on a 600-stock panel. Both rolling.var and
    rolling.cov default to ddof=1, so the variance and covariance estimators
    are consistent (the loop version mixed ddof=0 and ddof=1, biasing beta).

    Returns the negative of the annualized idiosyncratic volatility,
    consistent with the sign convention used elsewhere in the feature panel.
    """
    idx = _prepare_market_returns(df_index)

    df["Date_dt"] = pd.to_datetime(df["Date"]).dt.normalize()

    def _idio_vol(group):
        paired = _valid_stock_market_return_pairs(group, idx)
        y = paired["Daily_Return"]
        x = paired["Index_Return"]

        # Rolling OLS residual variance via the algebraic identity.
        # All three statistics use pandas default ddof=1 (sample estimators).
        var_y = y.rolling(window).var()
        var_x = x.rolling(window).var()
        cov_yx = y.rolling(window).cov(x)

        resid_var = var_y - (cov_yx ** 2) / var_x
        # Numerical safety: floating-point error can produce tiny negatives
        # when correlation is near 1; the OLS identity guarantees >= 0 in theory.
        resid_var = resid_var.clip(lower=0)
        idio_vol = np.sqrt(resid_var) * np.sqrt(252)
        paired["-IdioVol"] = -idio_vol

        group = group.copy()
        group["-IdioVol"] = _assign_latest_feature_to_stock_dates(
            group,
            paired[["Date", "-IdioVol"]],
            "-IdioVol",
        )
        return group

    df = df.groupby("Instrument", group_keys=False).apply(_idio_vol)
    df.drop(columns=["Date_dt"], inplace=True)
    return df

# ---------------------------------------------------------------------------
# MARKET-ryhmä (indeksitason featuret)
# ---------------------------------------------------------------------------

def compute_index_features(df_index_fundamentals):
    """
    Indeksitason featuret. Säilyttää raa'at Calculated PE / PB / Dividend Yield
    -kentät sellaisenaan ja lisää momentum-featuret samalla logiikalla kuin
    osakkeille (compute_mom_1m / compute_mom_12m).

    Index_MOM_1M    = 21 pörssipäivän pct_change
    Index_MOM_12M   = (Price_lag21 - Price_lag252) / Price_lag252  (skip t-1)

    HUOM: aikaisemman version time-series Z-score -laskenta poistettu — sitä ei
    tarvita vielä tässä vaiheessa. Index_Log_Return jätetty pois samasta syystä.
    """
    df_index_fundamentals = df_index_fundamentals.sort_index().copy()

    df_index_fundamentals["Index_MOM_1M"] = (
        df_index_fundamentals["Price Close"].pct_change(21)
    )

    price_lag21 = df_index_fundamentals["Price Close"].shift(21)
    price_lag252 = df_index_fundamentals["Price Close"].shift(252)
    df_index_fundamentals["Index_MOM_12M"] = (price_lag21 - price_lag252) / price_lag252

    return df_index_fundamentals


# DEPRECATED — vanha time-series Z-score -versio. Säilytetty kommentoituna
# referenssiksi. Korvattu compute_index_features:llä joka pitää raa'at PE/PB/DY
# sellaisenaan ja lisää MOM_1M / MOM_12M -featuret indeksille.
# def compute_index_ts_zscores(df_index_fundamentals):
#     cols = ["Calculated PE Ratio", "Calculated Price to Book", "Calculated Index Dividend Yield"]
#     for col in cols:
#         expanding_mean = df_index_fundamentals[col].expanding().mean()
#         expanding_std = df_index_fundamentals[col].expanding().std()
#         df_index_fundamentals[f"Z_{col}"] = (
#             (df_index_fundamentals[col] - expanding_mean) / expanding_std
#         )
#     df_index_fundamentals["Index_Log_Return"] = np.log(
#         df_index_fundamentals["Price Close"] / df_index_fundamentals["Price Close"].shift(1)
#     )
#     return df_index_fundamentals


# ---------------------------------------------------------------------------
# STANDALONE-featuret
# ---------------------------------------------------------------------------

def compute_log_market_cap(df):
    """log(Market Cap) — Size."""
    df["log_MktCap"] = np.log(df["Company Market Cap"])
    return df


def compute_excess_return(df, df_euribor):
    """
    Excess return = osakkeen tuotto - riskitön korko (EURIBOR päivätasolla).
    Aritmeettinen, Gu et al. (2020).
    """
    df_euribor = df_euribor.copy()
    df_euribor["Date"] = pd.to_datetime(df_euribor["Date"]).dt.date
    df_euribor["Rf_daily"] = df_euribor["Fixing Value"] / 100 / 252

    df["Date_dt"] = pd.to_datetime(df["Date"]).dt.date
    df = df.merge(
        df_euribor[["Date", "Rf_daily"]].rename(columns={"Date": "Date_dt"}),
        on="Date_dt",
        how="left",
    )
    df["Excess_Return"] = df["Daily_Return"] - df["Rf_daily"]
    df.drop(columns=["Date_dt"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Koontifunktio
# ---------------------------------------------------------------------------

def compute_all_features(
    df_stocks,
    df_index,
    df_index_fundamentals,
    df_euribor,
    sectors,
    sector_names,
    df_quarterly_eps=None,
    df_quarterly_pcf=None,
):
    """
    Laskee kaikki featuret yhdellä kutsulla.

    df_quarterly_eps: long-format DataFrame compute_pe_ttm_fallback:ia varten
    (data_fetch.fetch_quarterly_eps_for_pe). Jos None, P/E_ff ja E/P_ff jäävät
    laskematta — vain LTM-pohjainen P/E ja E/P tulevat masteriin.

    df_quarterly_pcf: long-format DataFrame compute_pcf_ttm_fallback:ia varten
    (data_fetch.fetch_quarterly_pcf_for_pcf). Jos None, -P/CF_ff jää
    laskematta — vain LTM-pohjaiset operating CF -variantit tulevat masteriin.
    """

    df = df_stocks.copy()
    df = _coerce_numeric_columns(df, NUMERIC_FUNDAMENTAL_COLUMNS)
    if df_quarterly_eps is not None:
        df_quarterly_eps = _coerce_numeric_columns(
            df_quarterly_eps.copy(), QUARTERLY_NUMERIC_COLUMNS
        )
    if df_quarterly_pcf is not None:
        df_quarterly_pcf = _coerce_numeric_columns(
            df_quarterly_pcf.copy(), QUARTERLY_NUMERIC_COLUMNS
        )

    # Valuation
    df = compute_pe_ltm(df)                  # P/E, E/P  (LTM only, PE_trial)
    df = compute_ep(df)                      # Legacy E/P_legacy, P/E_legacy vertailuun
    if df_quarterly_eps is not None:
        df = compute_pe_ttm_fallback(df, df_quarterly_eps)  # P/E_ff, E/P_ff
    df = compute_inv_pb(df)
    df = compute_neg_ps(df)
    df = compute_neg_pcf(df)                 # Legacy -P/CF vertailuun (TR.F.CF)
    df = compute_pcf_op(df)                  # P/CF_op, P/CF_ps + käänteiset
    if df_quarterly_pcf is not None:
        df = compute_pcf_ttm_fallback(df, df_quarterly_pcf)  # -P/CF_ff
    df = compute_pcf_refinitiv(df)           # P/CF_refinitiv (REF_PCF, TR.PriceToCFPerShare)
    df = compute_dividend_yield_trailing(df)

    # Quality
    df = compute_neg_debt_to_mktcap(df)
    df = compute_book_to_market(df)
    df = compute_operating_profitability(df)

    # Momentum
    df = compute_mom_1m(df)
    df = compute_mom_12m(df)
    df = compute_sector_group(df)              # GICS 11 → 6 ryhmän aggregointi
    df = compute_stock_vs_sector_return(df)    # Stock_vs_Sector_12M_1M + Stock_vs_Sector_1M
    df = compute_rsi(df)
    
    #df = compute_hurst(df)
    # Stock_vs_Sector deaktivoitu — käytä compute_daily_return:ia, jotta Daily_Return-sarake on saatavilla Beta/IdioVol/Excess_Return -laskennoille.
    df = compute_daily_return(df)
   

    # Risk
    df = compute_volatility(df)
    df = compute_beta(df, df_index)
    df = compute_idiosyncratic_vol(df, df_index)

    # Standalone
    df = compute_log_market_cap(df)
    # DEPRECATED: 4 binäärisen GICS-dummyn lisäys. Korvattu Sector_Group:lla
    # (compute_sector_group ajetaan jo Momentum-osiossa). Kutsu jäänyt no-op
    # shimiksi vanhojen callerien yhteensopivuuden takia.
    df = compute_sector_dummies(df, sectors, sector_names)
    df = compute_excess_return(df, df_euribor)

    # Market (indeksitaso)
    df_idx = compute_index_features(df_index_fundamentals)

    return df, df_idx
