"""
probe_refinitiv_columns.py

Diagnostiikkaskripti: hakee jokaisen data_fetch.py:n käyttämän TR.-kentän
mini-universumilla, kirjaa Refinitivin palauttaman sarakenimen ja tallentaa
mappingin CSV:nä.

Käyttötarkoitus: ennen täyttä data-hakua (~30+ min) varmistetaan että
oletetut sarakenimet features.py:ssä ja data_fetch.py:n rename-blokeissa
vastaavat sitä mitä Refinitiv todella palauttaa. Erityisesti uudet kentät
TR.EPSFRActValue(Period=LTM), TR.F.NetCashFlowOp(Period=LTM) ja
TR.F.NetCFOpPerShr(Period=LTM) on syytä verifioida ennen tuotantoajoa.

Käyttö:
    python src/probe_refinitiv_columns.py
    python src/probe_refinitiv_columns.py --output my_mapping.csv
    python src/probe_refinitiv_columns.py --tickers NESTE.HE,ASML.AS

Oletukset:
    - 3 osakkeen mini-universumi (NESTE.HE, ASML.AS, SAPG.DE)
    - 2 viikon hakuväli päiväkenttiin, 12 kk kvartaalikenttiin
    - CSV-output data/diagnostics/refinitiv_field_mapping.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import refinitiv.data as rd


# Mini-universumi — 3 isoa STOXX 600 -osaketta, joiden tiedetään raportoivan
# kaikkia tarvittavia kenttiä. Vaihda --tickers-flagilla jos jokin näistä
# ei vastaa.
DEFAULT_UNIVERSE = ["NESTE.HE", "ASML.AS", "SAPG.DE"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "diagnostics" / "refinitiv_field_mapping.csv"

# Lyhyt aikaikkuna riittää sarakenimien selvittämiseen
DAILY_PARAMS = {"SDate": "2024-01-01", "EDate": "2024-01-15", "Frq": "D", "Curn": "EUR"}
QUARTERLY_PARAMS = {"SDate": "2023-01-01", "EDate": "2024-12-31", "Frq": "FQ", "Curn": "EUR"}
INDEX_PARAMS = {"SDate": "2024-01-01", "EDate": "2024-01-15", "Frq": "D", "Curn": "EUR"}


# Päivätaajuiset osakekentät — fetch_stock_fundamentals():n field-lista
DAILY_FIELDS = [
    "TR.PriceClose",
    "TR.TotalReturn1D",
    "TR.CompanyMarketCap",
    "TR.Volume",
    "TR.SharesOutstanding",
    "TR.TotalDebt",
    "TR.F.CF",
    "TR.PriceToSalesPerShare",
    "TR.PriceToBVPerShare",
    "TR.GICSSector",
    "TR.GICSSubIndustry",
    "TR.EPSActValue",
    "TR.DPSActValue",
    "TR.F.DivPerShare",
    "TR.DPSActValueYield",
    "TR.F.ReturnAvgComEqPct",
    "TR.F.GrossProfitTotAssets",
    "TR.EBITDAActValue",
    "TR.F.ShHoldEqCom",
    "TR.F.PriceToBookValuePerShr",
    # Uudet kentät (PE_trial + Operating CF -PCF)
    "TR.EPSFRActValue(Period=LTM)",
    "TR.F.NetCashFlowOp(Period=LTM)",
    "TR.F.NetCFOpPerShr(Period=LTM)",
]

# Kvartaalitason kentät — fetch_quarterly_eps_for_pe():n field-lista
QUARTERLY_FIELDS = [
    "TR.EPSFRActValue",
    "TR.F.NetIncAfterTax",
]

# Indeksikentät — fetch_index_data():n field-lista
INDEX_FIELDS = [
    "TR.Index_PE_RTRS",
    "TR.Index_PRICE_TO_BOOK_RTRS",
    "TR.Index_DIV_YLD_RTRS",
]


def probe_one_field(field, params, universe):
    """
    Hakee yhden TR.-kentän, palauttaa (refinitiv_column_name, status, n_non_null).

    Refinitiv palauttaa joko ['Instrument', <field>] tai ['Instrument', 'Date', <field>]
    (jos .date- tai .periodenddate-kvalifikaattori on mukana, tai jos kenttä on
    ajallisesti vaihteleva). Otetaan viimeinen sarake jonka nimi ei ole
    'Instrument' eikä 'Date'.
    """
    try:
        df = rd.get_data(universe=universe, fields=[field], parameters=params)
        non_id_cols = [c for c in df.columns if c not in ("Instrument", "Date")]
        if not non_id_cols:
            return None, "EMPTY", 0
        col = non_id_cols[-1]
        n_non_null = int(df[col].notna().sum())
        return col, "OK", n_non_null
    except Exception as e:
        return None, f"ERROR: {type(e).__name__}: {str(e)[:80]}", 0


def probe_field_with_periodenddate(field, params, universe):
    """
    Erikoiskäsittely kvartaalikentille jotka tarvitsevat .periodenddate-ankkurin.
    Refinitivin palautus: ['Instrument', 'Period End Date', <field>].
    """
    field_with_date = f"{field}.periodenddate"
    try:
        df = rd.get_data(
            universe=universe,
            fields=[field_with_date, field],
            parameters=params,
        )
        non_id_cols = [c for c in df.columns if c not in ("Instrument",)]
        if len(non_id_cols) < 2:
            return None, None, "EMPTY", 0
        date_col = non_id_cols[0]
        value_col = non_id_cols[-1]
        n_non_null = int(df[value_col].notna().sum())
        return value_col, date_col, "OK", n_non_null
    except Exception as e:
        return None, None, f"ERROR: {type(e).__name__}: {str(e)[:80]}", 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="CSV-output polku (oletus: data/diagnostics/refinitiv_field_mapping.csv)",
    )
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_UNIVERSE),
        help="Pilkulla erotettu lista RIC-tikkereitä (oletus: NESTE.HE,ASML.AS,SAPG.DE)",
    )
    args = parser.parse_args()

    universe = [t.strip() for t in args.tickers.split(",") if t.strip()]
    print(f"Probe-universumi: {universe}")
    print(f"Daily-aikaväli:    {DAILY_PARAMS['SDate']} → {DAILY_PARAMS['EDate']}")
    print(f"Quarterly-aikaväli: {QUARTERLY_PARAMS['SDate']} → {QUARTERLY_PARAMS['EDate']}")
    print()

    rd.open_session()
    try:
        rows = []

        print("=" * 80)
        print("DAILY FIELDS — fetch_stock_fundamentals()")
        print("=" * 80)
        for f in DAILY_FIELDS:
            col, status, n = probe_one_field(f, DAILY_PARAMS, universe)
            rows.append({
                "tr_field": f,
                "type": "daily",
                "refinitiv_column": col,
                "status": status,
                "n_non_null": n,
            })
            print(f"  {f:42s} → {str(col):55s} [{status}, n={n}]")

        print()
        print("=" * 80)
        print("QUARTERLY FIELDS — fetch_quarterly_eps_for_pe()")
        print("=" * 80)
        for f in QUARTERLY_FIELDS:
            value_col, date_col, status, n = probe_field_with_periodenddate(
                f, QUARTERLY_PARAMS, universe
            )
            rows.append({
                "tr_field": f"{f}.periodenddate",
                "type": "quarterly",
                "refinitiv_column": date_col,
                "status": status,
                "n_non_null": "(date column)",
            })
            rows.append({
                "tr_field": f,
                "type": "quarterly",
                "refinitiv_column": value_col,
                "status": status,
                "n_non_null": n,
            })
            print(f"  {f:42s} → value: {str(value_col):40s} date: {str(date_col):25s} [n={n}]")

        print()
        print("=" * 80)
        print("INDEX FIELDS — fetch_index_data() (.STOXX)")
        print("=" * 80)
        for f in INDEX_FIELDS:
            col, status, n = probe_one_field(f, INDEX_PARAMS, [".STOXX"])
            rows.append({
                "tr_field": f,
                "type": "index",
                "refinitiv_column": col,
                "status": status,
                "n_non_null": n,
            })
            print(f"  {f:42s} → {str(col):55s} [{status}, n={n}]")

    finally:
        rd.close_session()

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print()
    print("=" * 80)
    print(f"Saved: {args.output}")
    print("=" * 80)
    n_ok = (out["status"] == "OK").sum()
    n_total = len(out)
    print(f"Onnistuneet: {n_ok}/{n_total}")
    fails = out.loc[out["status"] != "OK"]
    if len(fails) > 0:
        print(f"\nEpäonnistuneet kentät:")
        print(fails[["tr_field", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
