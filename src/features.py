"""
Featurelaskennat osake- ja indeksitasolla.
Jokainen funktio ottaa sisään DataFramen ja palauttaa sen uudella sarakkeella.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# VALUATION-ryhmä
# ---------------------------------------------------------------------------

def compute_ep(df):
    """
    E/P = trailing 12M EPS / Price Close.

    Trailing TTM EPS lasketaan summaamalla kaikki osakkeen raportoidut
    EPS-arvot edeltäneiden 365 päivän ajalta. Tämä on robusti sekä
    kvartaali- että puolivuosiraportoiville yrityksille (jälkimmäiset
    yleisiä UK/DE/NO-yrityksillä).

    Korvattu aiempi Forward P/E SmartEstimate -pohjainen laskenta —
    syynä liian korkea NaN-osuus (29%) mid-cap-hännästä joka ei ole
    analyytikkoseurannassa. Trailing-EPS pohjainen E/P on akateemisen
    asset pricing -tutkimuksen standardi (Gu et al. 2020,
    Fama-French -kirjallisuus).

    Kenttälogiikka mirror:oi compute_dividend_yield_trailing:ia
    (merge_asof backward → ei look-ahead biasia).
    """
    df["Date"] = pd.to_datetime(df["Date"])

    # Vain rivit joilla on raportoitu EPS (Refinitiv palauttaa NaN ei-raporttipäiville)
    eps_reports = (
        df.loc[df["Earnings Per Share - Actual"].notna(),
               ["Instrument", "Date", "Earnings Per Share - Actual"]]
        .sort_values(["Instrument", "Date"])
        .reset_index(drop=True)
    )

    # Rolling 365 päivän summa per osake — kalenteripäivät-perusteinen ikkuna
    # käsittelee sekä kvartaali- että puolivuosiraportoivat oikein.
    def _ttm(group):
        group = group.set_index("Date").sort_index()
        group["EPS_TTM"] = group["Earnings Per Share - Actual"].rolling("365D").sum()
        return group.reset_index()

    eps_ttm = eps_reports.groupby("Instrument", group_keys=False).apply(_ttm)

    # merge_asof backward: jokaiselle päivälle viimeisin saatavilla oleva TTM
    df = df.sort_values(["Instrument", "Date"])
    parts = []
    for instrument, group in df.groupby("Instrument", group_keys=False):
        ttm_slice = (
            eps_ttm.loc[eps_ttm["Instrument"] == instrument, ["Date", "EPS_TTM"]]
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

    df["E/P"] = df["EPS_TTM"] / df["Price Close"]
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
    -P/CF = -(Market Cap / Cash Flow).

    Korvattu Refinitivin "Price To Cash Flow Per Share (Daily Time Series Ratio)"
    -kenttä omalla laskennalla. Syy: Refinitivin valmiiksi laskettu ratio jätti
    ~7% riveistä NaN:ksi (todennäköisesti negatiivisen cash flow:n takia tai
    datakatkoista). Omasta laskennasta saamme suoran kontrollin.

    Total Cash Flow (TR.F.CF) on fiscal-vuoden arvo → ffill per osake
    seuraavaan raporttiin asti. Market Cap on päivittäinen.

    HUOM: jos CF < 0, P/CF on negatiivinen ja -P/CF positiivinen.
    Tämä on odotettua käyttäytymistä (tappiolliset firmat saavat korkeamman
    -P/CF-arvon kuin kannattavat, mikä on semanttisesti rikkinäinen signaali).
    Winsorisointi composites.py:ssä (1%/99%) leikkaa pahimmat outlierit.
    Jos tämä osoittautuu ongelmaksi, voidaan lisätä nollaus CF<=0 tapauksessa.
    """
    df = df.sort_values(["Instrument", "Date"])
    df["Cash Flow"] = df.groupby("Instrument")["Cash Flow"].ffill()
    df["-P/CF"] = -(df["Company Market Cap"] / df["Cash Flow"])
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


# ROE ja Operating Profitability tulevat suoraan Refinitivistä:
# "Return On Average Common Equity %" ja "Gross Profit / Total Assets"


# ---------------------------------------------------------------------------
# MOMENTUM-ryhmä
# ---------------------------------------------------------------------------

def compute_mom_1m(df):
    """1M Momentum: viimeisen ~21 kaupankäyntipäivän tuotto."""
    df["MOM_1M"] = df.groupby("Instrument")["Price Close"].pct_change(21)
    return df


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


def compute_hurst(df, window=252):
    """
    Hurst-eksponentti (R/S-analyysi, rolling ikkuna).
    """
    def _hurst_rs(series):
        """R/S-estimaatti yhdelle aikasarjalle."""
        if len(series) < 20 or series.std() == 0:
            return np.nan
        mean = series.mean()
        deviate = (series - mean).cumsum()
        R = deviate.max() - deviate.min()
        S = series.std(ddof=1)
        if S == 0:
            return np.nan
        return np.log(R / S) / np.log(len(series))

    def _rolling_hurst(group):
        log_returns = np.log(group["Price Close"] / group["Price Close"].shift(1))
        group["Hurst"] = log_returns.rolling(window).apply(_hurst_rs, raw=False)
        return group

    df = df.groupby("Instrument", group_keys=False).apply(_rolling_hurst)
    return df


def compute_daily_return(df):
    """
    Osakekohtainen päivätuotto (Price Close -pohjainen, ilman osinkoja).
    Tarpeellinen Beta-, IdioVol- ja Excess_Return -laskennoille.
    """
    df["Daily_Return"] = df.groupby("Instrument")["Price Close"].pct_change()
    return df


# DEACTIVATED: compute_stock_vs_sector_return
# Syy: toteutus oli päivätasolla mutta CLAUDE.md määrittää featuren kuukausitasolla.
# Palautetaan kun kuukausitason toteutus on valmiina. Koodi jätetty viitteeksi alle.
#
# def compute_stock_vs_sector_return(df):
#     """Osakkeen tuotto vs. sektorin keskimääräinen tuotto (kuukausitasolla)."""
#     df = compute_daily_return(df)
#     sector_mean = (
#         df.groupby(["Date", "GICS Sector Name"])["Daily_Return"]
#         .mean()
#         .reset_index()
#         .rename(columns={"Daily_Return": "Sector_Return"})
#     )
#     df = df.merge(sector_mean, on=["Date", "GICS Sector Name"], how="left")
#     df["Stock_vs_Sector"] = df["Daily_Return"] - df["Sector_Return"]
#     return df


# ---------------------------------------------------------------------------
# RISK-ryhmä
# ---------------------------------------------------------------------------

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


def compute_beta(df, df_index):
    """
    Rolling 252d Beta vs. .STOXXR — sama kuin RETURNS_AND_BETA_FINAL.ipynb:ssä.
    """
    # Indeksin päivätuotto
    idx = df_index.loc[df_index["Instrument"] == ".STOXXR"].copy()
    idx["Date"] = pd.to_datetime(idx["Date"]).dt.date
    idx = idx.drop_duplicates(subset=["Date"], keep="last")
    idx["Index_Return"] = idx["Price Close"].pct_change()

    df["Date_dt"] = pd.to_datetime(df["Date"]).dt.date

    def _beta(group):
        # LEFT merge: säilytetään kaikki osakkeen rivit. Jos indeksillä ei
        # ole dataa kyseiselle päivälle (pyhäpäiväepäsymmetria tms.), Index_Return
        # jää NaN:ksi ja rolling cov/var tuottaa NaN Betan siihen kohtaan.
        # Inner-merge rikkoi pituuden (merged < group) → ValueError
        # "Length of values (X) does not match length of index (Y)".
        merged = pd.merge(
            group[["Date_dt", "Daily_Return"]].rename(columns={"Date_dt": "Date"}),
            idx[["Date", "Index_Return"]],
            on="Date",
            how="left",
        )
        cov = merged["Daily_Return"].rolling(252).cov(merged["Index_Return"])
        var = merged["Index_Return"].rolling(252).var()
        group = group.copy()
        group["Beta_252d"] = (cov / var).values
        return group

    df = df.groupby("Instrument", group_keys=False).apply(_beta)
    df["-Beta_252d"] = -df["Beta_252d"]
    df.drop(columns=["Date_dt"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# SUORITUSKYKYVAROITUS — compute_idiosyncratic_vol
# ---------------------------------------------------------------------------
# Nykyinen toteutus on O(n * window) per osake (Python-for-loop joka sovittaa
# OLS:n joka ikkunaan erikseen). 10 osaketta × 16 vuotta × window=252 on vielä
# siedettävä smoke-ajossa, mutta 600 osakkeen × 16 vuoden paneelilla tämä on
# mittaluokkaa minuutteja → kymmeniä minuutteja.
#
# Vektorointi on triviaali rolling-OLS -identiteetillä:
#   Var(residual) = Var(y) - Cov(y, x)^2 / Var(x)
#
# Pandasin rolling.var / rolling.cov hoitaa koko ikkunasliden vektorisoidusti
# O(n) ajassa per osake. Nopeusvaikutus: ~100–250× smoke-ajon yli.
#
# Lisäbonus: alkuperäisessä koodissa on ddof-epäjohdonmukaisuus
# (np.cov → ddof=1, np.var → ddof=0), mikä biasoi betan kertoimella n/(n-1).
# Vektoroidussa versiossa molemmat käyttävät pandasin ddof=1 → korjautuu
# sivutuotteena.
#
# EHDOTETTU VEKTOROITU VERSIO (ota käyttöön ennen 600 osakkeen tuotantoajoa):
#
# def compute_idiosyncratic_vol(df, df_index, window=252):
#     """Idiosynkraattinen volatiliteetti rolling-OLS -identiteetillä."""
#     idx = df_index.loc[df_index["Instrument"] == ".STOXXR"].copy()
#     idx["Date"] = pd.to_datetime(idx["Date"]).dt.date
#     idx = idx.drop_duplicates(subset=["Date"], keep="last")
#     idx["Index_Return"] = idx["Price Close"].pct_change()
#
#     df["Date_dt"] = pd.to_datetime(df["Date"]).dt.date
#
#     def _idio_vol(group):
#         merged = pd.merge(
#             group[["Date_dt", "Daily_Return"]].rename(columns={"Date_dt": "Date"}),
#             idx[["Date", "Index_Return"]],
#             on="Date",
#             how="inner",
#         )
#         y = merged["Daily_Return"]
#         x = merged["Index_Return"]
#
#         # Rolling-OLS-identiteetti (kaikki ddof=1):
#         #   Var(resid) = Var(y) - Cov(y, x)^2 / Var(x)
#         var_y  = y.rolling(window).var()
#         var_x  = x.rolling(window).var()
#         cov_yx = y.rolling(window).cov(x)
#
#         resid_var = var_y - (cov_yx ** 2) / var_x
#         resid_var = resid_var.clip(lower=0)        # pyöristysvirhesuojaus
#         idio_vol = np.sqrt(resid_var) * np.sqrt(252)
#
#         group = group.copy()
#         group["-IdioVol"] = -idio_vol.values
#         return group
#
#     df = df.groupby("Instrument", group_keys=False).apply(_idio_vol)
#     df.drop(columns=["Date_dt"], inplace=True)
#     return df
# ---------------------------------------------------------------------------

def compute_idiosyncratic_vol(df, df_index, window=252):
    """
    Idiosynkraattinen volatiliteetti: CAPM-regressiojäännösten keskihajonta.

    HUOM: hidas toteutus. Katso yllä oleva kommentti vektoroidusta versiosta.
    """
    idx = df_index.loc[df_index["Instrument"] == ".STOXXR"].copy()
    idx["Date"] = pd.to_datetime(idx["Date"]).dt.date
    idx = idx.drop_duplicates(subset=["Date"], keep="last")
    idx["Index_Return"] = idx["Price Close"].pct_change()

    df["Date_dt"] = pd.to_datetime(df["Date"]).dt.date

    def _idio_vol(group):
        # LEFT merge, katso compute_beta — sama pituusongelma muuten.
        merged = pd.merge(
            group[["Date_dt", "Daily_Return"]].rename(columns={"Date_dt": "Date"}),
            idx[["Date", "Index_Return"]],
            on="Date",
            how="left",
        )
        # Rolling CAPM residuaalit
        residuals = []
        for i in range(len(merged)):
            if i < window:
                residuals.append(np.nan)
                continue
            y = merged["Daily_Return"].iloc[i - window:i].values
            x = merged["Index_Return"].iloc[i - window:i].values
            if np.isnan(y).any() or np.isnan(x).any():
                residuals.append(np.nan)
                continue
            beta = np.cov(y, x)[0, 1] / np.var(x) if np.var(x) > 0 else 0
            alpha = np.mean(y) - beta * np.mean(x)
            resid = y - (alpha + beta * x)
            residuals.append(resid.std() * np.sqrt(252))

        group = group.copy()
        group["-IdioVol"] = [-r if r is not np.nan else np.nan for r in residuals]
        return group

    df = df.groupby("Instrument", group_keys=False).apply(_idio_vol)
    df.drop(columns=["Date_dt"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# MARKET-ryhmä (indeksitason featuret)
# ---------------------------------------------------------------------------

def compute_index_ts_zscores(df_index_fundamentals):
    """
    Indeksifaktorien time-series Z-scoret (oman historian perusteella,
    ei cross-sectional). CLAUDE.md konventio.
    """
    cols = [
        "Calculated PE Ratio",
        "Calculated Price to Book",
        "Calculated Index Dividend Yield",
    ]

    for col in cols:
        expanding_mean = df_index_fundamentals[col].expanding().mean()
        expanding_std = df_index_fundamentals[col].expanding().std()
        df_index_fundamentals[f"Z_{col}"] = (
            (df_index_fundamentals[col] - expanding_mean) / expanding_std
        )

    # Index log return
    df_index_fundamentals["Index_Log_Return"] = np.log(
        df_index_fundamentals["Price Close"] / df_index_fundamentals["Price Close"].shift(1)
    )

    return df_index_fundamentals


# ---------------------------------------------------------------------------
# STANDALONE-featuret
# ---------------------------------------------------------------------------

def compute_log_market_cap(df):
    """log(Market Cap) — Size."""
    df["log_MktCap"] = np.log(df["Company Market Cap"])
    return df


def compute_sector_dummies(df, sectors, names):
    """4 binääristä GICS-sektoridummya."""
    for sector, name in zip(sectors, names):
        df[f"Sector_{name}"] = (df["GICS Sector Name"] == sector).astype(int)
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

def compute_all_features(df_stocks, df_index, df_index_fundamentals, df_euribor, sectors, sector_names):
    """Laskee kaikki featuret yhdellä kutsulla."""

    df = df_stocks.copy()

    # Valuation
    df = compute_ep(df)
    df = compute_inv_pb(df)
    df = compute_neg_ps(df)
    df = compute_neg_pcf(df)
    df = compute_dividend_yield_trailing(df)

    # Quality
    df = compute_neg_debt_to_mktcap(df)
    # ROE ja Operating Profitability tulevat suoraan datasta

    # Momentum
    df = compute_mom_1m(df)
    df = compute_mom_12m(df)
    df = compute_rsi(df)
    df = compute_hurst(df)
    # Stock_vs_Sector deaktivoitu — käytä compute_daily_return:ia
    # jotta Daily_Return-sarake on saatavilla Beta/IdioVol/Excess_Return -laskennoille.
    df = compute_daily_return(df)

    # Risk
    df = compute_volatility(df)
    df = compute_beta(df, df_index)
    df = compute_idiosyncratic_vol(df, df_index)

    # Standalone
    df = compute_log_market_cap(df)
    df = compute_sector_dummies(df, sectors, sector_names)
    df = compute_excess_return(df, df_euribor)

    # Market (indeksitaso)
    df_idx = compute_index_ts_zscores(df_index_fundamentals)

    return df, df_idx
