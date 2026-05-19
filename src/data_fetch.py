"""
Refinitiv LSEG -datahaut. API-kutsut käyttävät rd.get_data() samalla
rakenteella kuin alkuperäisissä FINAL-notebookeissa.
"""

import refinitiv.data as rd
import pandas as pd
import numpy as np
import time
import warnings
from pathlib import Path

from config import (
    PARAMS_DAILY, PARAMS_EURIBOR, INDEX_UNIVERSE, EURIBOR_RIC,
    CHUNK_SIZE, SLEEP_BETWEEN_CHUNKS,
)

warnings.filterwarnings('ignore', category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PRICE_CACHE_PATH = PROJECT_ROOT / "data" / "01_raw" / "updated_smoke_index.csv"


def open_session():
    """Avaa Refinitiv-sessio."""
    rd.open_session()


def _fetch_in_chunks(
    universe,
    fields,
    params,
    chunk_size=None,
    max_retries=3,
    retry_sleep=10,
):
    """
    Hakee datan chunkeissa API-rajoitusten vuoksi.
    Sama logiikka kuin FINAL-notebookeissa.
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE

    def _fetch_one_chunk(chunk_universe, label):
        df = None
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                print(
                    f"Refinitiv chunk {label}: "
                    f"{len(chunk_universe)} instruments, attempt {attempt}/{max_retries}"
                )
                df = rd.get_data(
                    universe=chunk_universe,
                    fields=fields,
                    parameters=params,
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt == max_retries:
                    break
                wait = retry_sleep * attempt
                print(f"Chunk failed; retrying in {wait}s...")
                time.sleep(wait)

        if df is not None:
            return df

        if len(chunk_universe) <= 1:
            raise last_exc

        mid = len(chunk_universe) // 2
        left = chunk_universe[:mid]
        right = chunk_universe[mid:]
        print(
            f"Chunk {label} failed after {max_retries} attempts; "
            f"splitting {len(chunk_universe)} instruments into {len(left)} + {len(right)}."
        )
        left_df = _fetch_one_chunk(left, f"{label}.1")
        time.sleep(SLEEP_BETWEEN_CHUNKS)
        right_df = _fetch_one_chunk(right, f"{label}.2")
        return pd.concat([left_df, right_df], ignore_index=True)

    results = []
    for i in range(0, len(universe), chunk_size):
        chunk_universe = universe[i:i + chunk_size]
        label = f"{i // chunk_size + 1}/{(len(universe) + chunk_size - 1) // chunk_size}"
        df = _fetch_one_chunk(chunk_universe, label)
        results.append(df)
        if i + chunk_size < len(universe):
            time.sleep(SLEEP_BETWEEN_CHUNKS)
    return pd.concat(results, ignore_index=True)


# Sarakkeet joita EI saa muuntaa numeeriseksi (säilyvät merkkijonoina/päiväyksinä)
_NON_NUMERIC_COLS = {
    "Instrument",
    "Date",
    "GICS Sector Name",
    "GICS Sub-Industry Name",
}


def _clean_stock_fundamentals(df):
    """
    Siivoa Refinitivin raakapalautus osakefundamentaaleille:

    1. GICS-sarakkeiden normalisointi ja ffill/bfill per osake.
       Refinitiv palauttaa GICS-arvot erillisillä metadata-riveillä
       (Date=NaT, vain GICS täytetty). Kopioidaan arvot osakkeen muille
       riveille ennen kuin metadata-rivit pudotetaan.

    2. Pudota rivit joissa ei ole Datea (metadata-rivien jäänteet).
       Näiden läsnäolo rikkoo merge_asof-kutsut joita features.py käyttää.

    3. Poista duplikaatit (sama Instrument + Date).
    """
    # 1. GICS-sarakkeiden normalisointi + täydennys per osake
    gics_cols = ["GICS Sector Name", "GICS Sub-Industry Name"]
    for col in gics_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(
                {"": np.nan, "None": np.nan, "nan": np.nan, "<NA>": np.nan}
            )
            df[col] = df.groupby("Instrument")[col].ffill().bfill()

    # 2. Pudota rivit ilman päivämäärää
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)

    # 3. Duplikaattien poisto
    df = df.drop_duplicates(subset=["Instrument", "Date"], keep="first").reset_index(drop=True)

    return df


def _coerce_numeric_columns(df, exclude=_NON_NUMERIC_COLS):
    """
    Pakota kaikki ei-poissuljetut sarakkeet numeerisiksi.

    Refinitiv palauttaa puuttuvat arvot satunnaisesti merkkijonoina
    ("", "NaN", "NA", whitespace), mikä aiheuttaa koko sarakkeen
    muuttumisen object-tyyppiseksi ja rikkoo downstream-aritmetiikan
    (esim. 1.0 / df[col] → TypeError: float / str).

    errors='coerce' muuttaa kaikki ei-numeeriset arvot NaN:ksi.
    """
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_stock_fundamentals(universe):
    """
    Hakee osakekohtaiset fundamentaalit — sama kenttälista kuin
    DIVIDEND_YIELD_FINAL.ipynb:ssä.
    """
    fields = [
        "TR.PriceClose.date",
        "TR.PriceClose",
        "TR.TotalReturn1D",
        "TR.CompanyMarketCap",
        "TR.Volume",
        "TR.SharesOutstanding",
        "TR.TotalDebt",                # → -Debt/MktCap (oma laskenta, ffill)
        "TR.F.CF",                     # legacy Cash Flow -kenttä, säilytetty datassa vaikka -P/CF on korvattu
        "TR.PriceToSalesPerShare",
        "TR.PriceToBVPerShare",
        "TR.GICSSector",
        # "TR.GICSSubIndustry",
        "TR.EPSActValue",              # legacy basic EPS, säilytetty (compute_ep deprecated → korvattu PE_trial-versioilla)
        "TR.DPSActValue",
        "TR.DPSActValueYield",
        # Quality-ryhmän featuret
        "TR.F.ReturnAvgComEqPct",      # ROE
        "TR.EBITDAActValue",             # → EV/EBITDA (oma laskenta, MktCap / EBITDA, ffill)
        "TR.F.EBITDA",                   # fallback → OperatingProfitability
        "TR.F.ShHoldEqCom",               # → Book-to-Market (oma laskenta, 1 / P/B, ffill)
        "TR.F.PriceToBookValuePerShr", # → P/B (oma laskenta, Price / (Book Value per Share), ffill)
        # PE_trial (PE_fix.ipynb) — Refinitivin omaa PE-lukua lähinnä osuva variantti
        "TR.EPSFRActValue(Period=LTM)",   # Fully reported (diluted) EPS, LTM. → P/E ja E/P
        # PCF Operating CF -variantit (PCF_test2.ipynb) — korvaa legacy -P/CF (joka käytti TR.F.CF:ää)
        "TR.F.NetCashFlowOp(Period=LTM)",  # Net Cash Flow from Operating Activities, LTM. → P/CF_op
        "TR.F.NetCFOpPerShr(Period=LTM)",  # Cash Flow from Operations per Share, LTM. → P/CF_ps
        "TR.PriceToCFPerShare",            # Refinitivin oma P/CF -ratio (Daily Time Series). → -P/CF_refinitiv
    ]

    df = _fetch_in_chunks(universe, fields, PARAMS_DAILY)
    df = _coerce_numeric_columns(df)
    df = _clean_stock_fundamentals(df)
    return df


def fetch_quarterly_eps_for_pe(universe):
    """
    Hakee kvartaalitason EPS- ja Net Income -sarjat PE_trial:n fallback-ketjua varten.

    Palauttaa long-format DataFramen sarakkeilla
        ['Instrument', 'QDate', 'EPSfr_Q', 'NetIncome_Q'].

    `EPSfr_Q` (TR.EPSFRActValue) ja `NetIncome_Q` (TR.F.NetIncAfterTax) haetaan
    kahtena erillisenä API-kutsuna, koska Refinitiv ei salli kahta eri
    .periodenddate-ankkuria samassa kentälistassa. Kutsut yhdistetään
    (Instrument, QDate) -avaimella outer-mergellä.

    Kvartaalihaku kattaa saman aikavälin kuin daily-haku
    (FETCH_START_DATE → END_DATE), jotta features.py:n
    compute_pe_ttm_fallback voi laskea 4Q rolling TTM:n koko paneelin yli.
    """
    quarterly_params = {
        "SDate": PARAMS_DAILY["SDate"],
        "EDate": PARAMS_DAILY["EDate"],
        "Frq": "FQ",
        "Curn": "EUR",
    }

    df_eps = _fetch_in_chunks(
        universe,
        ["TR.EPSFRActValue.periodenddate", "TR.EPSFRActValue"],
        quarterly_params,
    )
    df_ni = _fetch_in_chunks(
        universe,
        ["TR.F.NetIncAfterTax.periodenddate", "TR.F.NetIncAfterTax"],
        quarterly_params,
    )

    # Refinitivin oletetut palautusnimet:
    #   df_eps: ['Instrument', 'Period End Date', 'Earnings Per Share Reported - Actual']
    #   df_ni:  ['Instrument', 'Period End Date', 'Net Income after Tax']
    # Jos nimet poikkeavat, mukauta tämä rename ensimmäisen ajon jälkeen.
    df_eps = df_eps.rename(columns={
        "Period End Date": "QDate",
        "Earnings Per Share Reported - Actual": "EPSfr_Q",
    })
    df_ni = df_ni.rename(columns={
        "Period End Date": "QDate",
        "Net Income after Tax": "NetIncome_Q",
    })

    df_eps["QDate"] = pd.to_datetime(df_eps["QDate"], errors="coerce")
    df_ni["QDate"] = pd.to_datetime(df_ni["QDate"], errors="coerce")
    df_eps["EPSfr_Q"] = pd.to_numeric(df_eps["EPSfr_Q"], errors="coerce")
    df_ni["NetIncome_Q"] = pd.to_numeric(df_ni["NetIncome_Q"], errors="coerce")

    df = (
        df_eps.merge(df_ni, on=["Instrument", "QDate"], how="outer")
        .dropna(subset=["QDate"])
        .sort_values(["Instrument", "QDate"])
        .reset_index(drop=True)
    )
    return df


def fetch_quarterly_pcf_for_pcf(universe):
    """
    Hakee kvartaalitason operating cash flow -sarjat P/CF fallback-ketjua varten.

    Palauttaa long-format DataFramen sarakkeilla
        ['Instrument', 'QDate', 'NetCashFlowOp_Q', 'NetCFOpPerShr_Q'].

    Kvartaalihaku kattaa saman aikavälin kuin daily-haku
    (FETCH_START_DATE → END_DATE), jotta features.py voi laskea 4Q rolling
    TTM:n koko paneelin yli ja levittää sen päivätasolle ilman look-aheadia.
    """
    quarterly_params = {
        "SDate": PARAMS_DAILY["SDate"],
        "EDate": PARAMS_DAILY["EDate"],
        "Frq": "FQ",
        "Curn": "EUR",
    }

    df_op = _fetch_in_chunks(
        universe,
        ["TR.F.NetCashFlowOp.periodenddate", "TR.F.NetCashFlowOp"],
        quarterly_params,
    )
    df_opps = _fetch_in_chunks(
        universe,
        ["TR.F.NetCFOpPerShr.periodenddate", "TR.F.NetCFOpPerShr"],
        quarterly_params,
    )

    df_op = df_op.rename(columns={
        "Period End Date": "QDate",
        "Net Cash Flow from Operating Activities": "NetCashFlowOp_Q",
    })
    df_opps = df_opps.rename(columns={
        "Period End Date": "QDate",
        "Cash Flow from Operations per Share": "NetCFOpPerShr_Q",
    })

    df_op["QDate"] = pd.to_datetime(df_op["QDate"], errors="coerce")
    df_opps["QDate"] = pd.to_datetime(df_opps["QDate"], errors="coerce")
    df_op["NetCashFlowOp_Q"] = pd.to_numeric(df_op["NetCashFlowOp_Q"], errors="coerce")
    df_opps["NetCFOpPerShr_Q"] = pd.to_numeric(df_opps["NetCFOpPerShr_Q"], errors="coerce")

    df = (
        df_op.merge(df_opps, on=["Instrument", "QDate"], how="outer")
        .dropna(subset=["QDate"])
        .sort_values(["Instrument", "QDate"])
        .reset_index(drop=True)
    )
    return df


def fetch_stock_prices(universe):
    """
    Hakee osakkeiden hinnat ja total return — sama kuin
    RETURNS_FINAL.ipynb:ssä.
    """
    fields = [
        "TR.PriceClose.date",
        "TR.PriceClose",
        "TR.TotalReturn1D",
    ]

    df = _fetch_in_chunks(universe, fields, PARAMS_DAILY)
    df = _coerce_numeric_columns(df)

    # Muunna Total Return desimaaliin (Refinitiv antaa prosentteina)
    df["Daily Total Return"] = df["Daily Total Return"] / 100

    return df


def _valid_index_price_count(df, instrument):
    """Count rows with a real date and price for a single index RIC."""
    if df is None or df.empty:
        return 0
    required = {"Instrument", "Date", "Price Close"}
    if not required.issubset(df.columns):
        return 0

    mask = df["Instrument"].eq(instrument)
    dates = pd.to_datetime(df.loc[mask, "Date"], errors="coerce")
    prices = pd.to_numeric(df.loc[mask, "Price Close"], errors="coerce")
    return int((dates.notna() & prices.notna()).sum())


def _fetch_single_index_price_history(instrument):
    """Fallback for index RICs that rd.get_data returns as a NaT/NaN row."""
    hist = rd.get_history(
        universe=[instrument],
        fields=["TR.PriceClose"],
        start=PARAMS_DAILY["SDate"],
        end=PARAMS_DAILY["EDate"],
        interval="1D",
    )
    hist = hist.reset_index()
    if "Date" not in hist.columns:
        hist = hist.rename(columns={hist.columns[0]: "Date"})

    price_col = None
    for candidate in ["Price Close", "TR.PriceClose", "Close Price"]:
        if candidate in hist.columns:
            price_col = candidate
            break
    if price_col is None:
        non_date_cols = [c for c in hist.columns if c != "Date"]
        if non_date_cols:
            price_col = non_date_cols[0]
    if price_col is None:
        return pd.DataFrame(columns=["Instrument", "Date", "Price Close"])

    hist = hist.rename(columns={price_col: "Price Close"})
    hist["Instrument"] = instrument
    return hist[["Instrument", "Date", "Price Close"]]


def _load_cached_index_prices(instrument):
    if not INDEX_PRICE_CACHE_PATH.exists():
        return pd.DataFrame(columns=["Instrument", "Date", "Price Close"])

    cached = pd.read_csv(
        INDEX_PRICE_CACHE_PATH,
        usecols=["Instrument", "Date", "Price Close"],
    )
    cached = _coerce_numeric_columns(cached)
    cached["Date"] = pd.to_datetime(cached["Date"], errors="coerce")
    cached = cached.loc[cached["Instrument"].eq(instrument)].dropna(
        subset=["Date", "Price Close"]
    )

    start = pd.to_datetime(PARAMS_DAILY["SDate"])
    end = pd.to_datetime(PARAMS_DAILY["EDate"])
    if cached.empty or cached["Date"].min() > start or cached["Date"].max() < end:
        return pd.DataFrame(columns=["Instrument", "Date", "Price Close"])

    return cached.loc[(cached["Date"] >= start) & (cached["Date"] <= end)].copy()


def fetch_index_data():
    """
    Hakee indeksidatan (.STOXX ja .STOXXR) — sama kuin
    RETURNS_FINAL.ipynb:ssä ja INDEX_FINAL.ipynb:ssä.
    """
    # Indeksin hinta- ja tuottodata
    price_fields = [
        "TR.PriceClose.date",
        "TR.PriceClose",
    ]

    # Hae jokainen indeksi erikseen: pitkillä aikaikkunoilla rd.get_data
    # palauttaa yhdistetyssä kutsussa toiselle instrumentille vain yhden NaT/NaN-rivin.
    parts = []
    for ric in INDEX_UNIVERSE:
        last_exc = None
        try:
            part = rd.get_data(
                universe=[ric],
                fields=price_fields,
                parameters=PARAMS_DAILY,
            )
            part = _coerce_numeric_columns(part)
        except Exception as exc:
            last_exc = exc
            print(f"WARNING: {ric} rd.get_data failed: {exc}")
            part = pd.DataFrame(columns=["Instrument", "Date", "Price Close"])

        if _valid_index_price_count(part, ric) < 2:
            print(f"WARNING: {ric} price series was empty from rd.get_data; trying rd.get_history...")
            try:
                part = _fetch_single_index_price_history(ric)
                part = _coerce_numeric_columns(part)
            except Exception as exc:
                last_exc = exc
                print(f"WARNING: {ric} rd.get_history failed: {exc}")
                part = pd.DataFrame(columns=["Instrument", "Date", "Price Close"])

        if _valid_index_price_count(part, ric) < 2:
            print(f"WARNING: {ric} live price fetch failed; trying cache {INDEX_PRICE_CACHE_PATH}...")
            part = _load_cached_index_prices(ric)

        if _valid_index_price_count(part, ric) < 2:
            raise ValueError(
                f"{ric} index price series is missing or all-NaN. "
                "Index features and beta/idiosyncratic volatility cannot be computed."
            ) from last_exc

        parts.append(part)
    df_index = pd.concat(parts, ignore_index=True)
    df_index = _coerce_numeric_columns(df_index)

    df_index["Date"] = pd.to_datetime(df_index["Date"], errors="coerce")
    df_index = df_index.dropna(subset=["Date", "Price Close"])
    df_index = (
        df_index
        .drop_duplicates(subset=["Instrument", "Date"], keep="last")
        .sort_values(["Instrument", "Date"])
        .reset_index(drop=True)
    )

    # Indeksin fundamentaalit (INDEX_FINAL.ipynb)
    index_fields = [
        "TR.Index_PE_RTRS",
        "TR.Index_PRICE_TO_BOOK_RTRS",
        "TR.Index_DIV_YLD_RTRS",
        "TR.PriceClose",
    ]

    df_index_fundamentals = rd.get_history(
        universe=[".STOXX"],
        fields=index_fields,
        start=PARAMS_DAILY["SDate"],
        end=PARAMS_DAILY["EDate"],
        interval="1D",
    )
    df_index_fundamentals = _coerce_numeric_columns(df_index_fundamentals)

    # Kohdistetaan fundamentit indeksin oikeaan kaupankäyntikalenteriin.
    # Jätetään raakahakudatasta pois rivit, joilla fundamentit puuttuvat,
    # jotta backward-match ei pysähdy tyhjään pyhäpäiväriviin.
    fundamental_cols = [
        "Calculated PE Ratio",
        "Calculated Price to Book",
        "Calculated Index Dividend Yield",
    ]
    df_index_fundamentals = (
        df_index_fundamentals
        .reset_index()
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
    )
    df_index_fundamentals["Date"] = pd.to_datetime(df_index_fundamentals["Date"])
    df_index_fundamentals = df_index_fundamentals.loc[
        df_index_fundamentals[fundamental_cols].notna().any(axis=1)
    ].copy()
    df_index_fundamentals.drop(columns=["Price Close"], inplace=True, errors="ignore")

    df_stoxx_calendar = (
        df_index.loc[df_index["Instrument"] == ".STOXX", ["Date", "Price Close"]]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    df_index_fundamentals = pd.merge_asof(
        df_stoxx_calendar,
        df_index_fundamentals,
        on="Date",
        direction="backward",
    ).set_index("Date")

    return df_index, df_index_fundamentals


def trim_warmup(df, start_date, date_col="Date"):
    """
    Leikkaa warmup-rivit pois: palauttaa vain rivit joissa date_col >= start_date.

    Tukee sekä sarake- että indeksipohjaista päivämäärää (esim.
    df_index_fundamentals käyttää DatetimeIndexiä).
    """
    start = pd.to_datetime(start_date)
    df = df.copy()

    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col])
        return df.loc[dates >= start].reset_index(drop=True)

    if isinstance(df.index, pd.DatetimeIndex):
        return df.loc[df.index >= start]

    # Ei löytynyt päivämääräsaraketta/-indeksiä → palauta ennallaan
    return df


def fetch_euribor():
    """
    Hakee 3kk EURIBOR-datan — sama kuin EURIBOR_FINAL.ipynb:ssä.
    """
    fields = [
        "TR.FIXINGVALUE",
        "TR.FIXINGVALUE.date",
    ]

    df = rd.get_data(
        universe=EURIBOR_RIC,
        fields=fields,
        parameters=PARAMS_EURIBOR,
    )
    df = _coerce_numeric_columns(df)

    # Varmistetaan yksi rivi per päivä ennen myöhempiä mergejä.
    df["Date"] = pd.to_datetime(df["Date"])
    df = (
        df
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df
