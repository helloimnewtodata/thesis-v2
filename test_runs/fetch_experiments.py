"""
TILAPÄINEN kenttävariantti-experimentti.

Tavoite: löytää Refinitivin kenttävariantit jotka antavat parhaan kattavuuden
(vähiten NaN:a) seuraaville tunnusluvuille:
    - EPS  (Earnings Per Share)
    - CF   (Cash Flow)
    - DPS  (Dividend Per Share)
    - ROE  (Return On Equity)

Lähestymistapa:
  1. Hakee samat 10 RIC:iä kuin nykyinen smoke_stocks.csv (vertailukelpoinen).
  2. Hakee jokaisen kenttävariantin ERIKSEEN (1 API-kutsu per kenttä) ─ jos
     yksi kenttänimi on virheellinen, muut etenevät silti.
  3. Tulostaa per metriikka taulukon: NaN%, dailyness, tyyppi.
  4. Tallentaa raaka-CSV:t data/01_raw/experiments/ visuaalista tarkastelua varten.

Lopputulos: päätetään mitkä kentät siirretään src/data_fetch.py:hyn ja tämä
tiedosto poistetaan.

AJO:
    venv-v3/bin/python fetch_experiments.py
"""

from pathlib import Path
import pandas as pd
import refinitiv.data as rd

from config import PARAMS_DAILY
from src.data_fetch import _coerce_numeric_columns, _clean_stock_fundamentals

rd.open_session()

PROJECT_ROOT = Path(__file__).resolve().parent
SMOKE_PATH = PROJECT_ROOT / "data" / "01_raw" / "smoke_stocks.csv"
OUT_DIR = PROJECT_ROOT / "data" / "01_raw" / "experiments"


# =============================================================================
# Kenttävariantit per metriikka
# =============================================================================
# Periaate: sekoita "actual" / "fiscal" / "TTM" / "mean" -varintteja, jotta
# näemme mikä antaa parhaan kattavuuden ilman SmartEstimate-rajauksia.
EXPERIMENTS = {
    "EPS": [
        "TR.EPSActValue",                              # nykyinen (kvartaali, ffilled)
        "TR.BasicEPSExclExordItemsCom",                # basic EPS excl extraord
        "TR.BasicEPSInclExordItemsCom",                # basic EPS incl extraord
        "TR.DilutedEPSExclExordItemsCom",              # diluted variant
        "TR.F.EPSBasicBeforeExtra",                    # fiscal basic
        "TR.EPSMean",                                  # analyst mean (forward)
        "TR.EPSMeanEst",                               # mean estimate
        "TR.TtmEpsBasic",                              # TTM (jos olemassa)
        "TR.BasicEPSExclExordItemsComTtm",             # TTM-variantti
    ],
    "CashFlow": [
        "TR.F.CF",                                     # nykyinen (fiscal CF)
        "TR.F.OpCF",                                   # operating CF
        "TR.F.FreeCashFlow",                           # free CF
        "TR.F.CFOpActualValue",                        # actual reported
        "TR.OperatingCFTtm",                           # TTM
        "TR.CFOpToShareTtm",                           # TTM per share
        "TR.PriceToCFPerShare",                        # baseline (poistettu)
    ],
    "DPS": [
        "TR.DPSActValue",                              # nykyinen (declared)
        "TR.F.DivPerShare",                            # fiscal annual
        "TR.DPSCommonGross",                           # gross common
        "TR.DPSCommonGrossActValue",                   # actual gross common
        "TR.AnnualizedDPS",                            # annualized
        "TR.LastDividendValue",                        # most recent declared
        "TR.TtmDPSCommonGross",                        # TTM
    ],
    "ROE": [
        "TR.F.ReturnAvgComEqPct",                      # nykyinen (fiscal annual)
        "TR.F.ReturnOnEquity",                         # simple ROE
        "TR.NetIncomeReturnOnEquityYr0",               # year 0
        "TR.ROETtm",                                   # TTM
        "TR.ROEActValue",                              # actual reported
        "TR.F.NetIncomeReturnOnEquity",                # net income / equity
    ],
}


def get_smoke_universe():
    """Lataa samat RIC:it joita käytettiin smoke_stocks.csv:n luonnissa."""
    if not SMOKE_PATH.exists():
        raise FileNotFoundError(
            f"smoke_stocks.csv puuttuu: {SMOKE_PATH}\n"
            "Aja ensin main_test.py jotta universumi on määritelty."
        )
    df = pd.read_csv(SMOKE_PATH)
    return sorted(df["Instrument"].dropna().unique().tolist())


def fetch_single_field(universe, field):
    """
    Hakee yhden kentän kaikille osakkeille.
    Palauttaa DataFramen tai None jos kenttä on virheellinen.
    """
    try:
        df = rd.get_data(
            universe=universe,
            fields=["TR.PriceClose.date", field],
            parameters=PARAMS_DAILY,
        )
        df = _coerce_numeric_columns(df)
        df = _clean_stock_fundamentals(df)
        return df
    except Exception as e:
        print(f"  X  {field:<48} VIRHE: {type(e).__name__}: {str(e)[:80]}")
        return None


def classify(df, col):
    """
    Luokittele kenttä NaN%:n ja dailyness:n perusteella.
    dailyness = uniikkeja arvoja per osake / ei-NaN rivejä per osake
    """
    nonnan = df[col].notna().sum()
    total = len(df)
    if total == 0:
        return None
    pct_nan = (1 - nonnan / total) * 100
    n_uniq = df.groupby("Instrument")[col].nunique().sum() if nonnan > 0 else 0
    dailyness = n_uniq / nonnan if nonnan > 0 else 0

    if nonnan == 0:
        kind = "EMPTY"
    elif dailyness > 0.5:
        kind = "DAILY"
    elif dailyness > 0.05:
        kind = "mixed"
    else:
        kind = "FFILLED"

    return {
        "non_nan": nonnan,
        "total": total,
        "pct_nan": pct_nan,
        "dailyness": dailyness,
        "kind": kind,
    }


def print_group_table(label, results):
    print(f"\n{'=' * 110}")
    print(f"  {label}")
    print('=' * 110)
    print(f"{'Refinitiv field':<35} {'pandas column':<48} {'non-NaN':>8} {'%NaN':>6} {'daily':>6} {'type':>8}")
    print('-' * 110)
    for r in results:
        print(
            f"{r['field']:<35} {r['col']:<48} {r['non_nan']:>8} "
            f"{r['pct_nan']:>5.1f}% {r['dailyness']:>6.3f} {r['kind']:>8}"
        )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = get_smoke_universe()
    print(f"Universumi: {len(universe)} RIC:iä (sama kuin smoke_stocks.csv)")
    for r in universe:
        print(f"  - {r}")
    print(f"\nAikaväli: {PARAMS_DAILY['SDate']} → {PARAMS_DAILY['EDate']}")
    print(f"Kenttäryhmiä: {list(EXPERIMENTS.keys())}")
    print(f"Kenttiä yhteensä: {sum(len(v) for v in EXPERIMENTS.values())}")

    rd.open_session()
    try:
        for label, fields in EXPERIMENTS.items():
            print(f"\n{'#' * 110}")
            print(f"# Fetch [{label}] — {len(fields)} kenttävarinttia")
            print('#' * 110)

            results = []
            combined = None  # CSV-koonti tämän ryhmän tuloksista

            for field in fields:
                print(f"  -> {field}")
                df = fetch_single_field(universe, field)
                if df is None or df.empty:
                    continue

                data_cols = [c for c in df.columns if c not in {"Instrument", "Date"}]
                for col in data_cols:
                    stats = classify(df, col)
                    if stats is None:
                        continue
                    results.append({"field": field, "col": col, **stats})

                # Liitä combined-tauluun (long format)
                df_long = df.copy()
                df_long["_field"] = field
                if combined is None:
                    combined = df_long
                else:
                    combined = pd.concat([combined, df_long], ignore_index=True)

            # Lajittele kattavuuden mukaan paras ensin
            results.sort(key=lambda r: r["pct_nan"])
            print_group_table(label, results)

            if combined is not None:
                out_path = OUT_DIR / f"experiment_{label}.csv"
                combined.to_csv(out_path, index=False)
                print(f"\nTallennettu: {out_path}")
    finally:
        rd.close_session()

    print(f"\n{'=' * 110}")
    print("Valmis. Tarkasta tulokset yllä ja päätä mitkä kentät siirretään")
    print("src/data_fetch.py:hyn. Raaka-CSV:t ovat data/01_raw/experiments/.")
    print('=' * 110)


if __name__ == "__main__":
    main()
