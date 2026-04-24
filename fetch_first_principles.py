"""
FIRST PRINCIPLES: osakekohtaiset tunnusluvut rautalangasta.

Tavoite: hakea Refinitivista RAAKA-KOMPONENTIT joista lasketaan itse:
    - E/P  = Net Income / Market Cap  (tai EPS / Price)
    - P/CF = Market Cap / Cash Flow   (tai Price / CFPS)
    - P/S  = Market Cap / Revenue     (tai Price / SalesPS)
    - DY   = Total Dividends Paid / Market Cap

Miksi itse laskeminen:
    1. Refinitivin valmiit ratiot (TR.PriceToCFPerShare, TR.PriceToSalesPerShare)
       sisaltavat enemman NaN:a kuin itselaskettu versio
    2. Per-share-kentat (TR.EPSActValue, TR.DPSActValue) ovat forward-filled,
       jolloin rolling-sum (TTM) summaa saman arvon moneen kertaan
    3. Total-level-luvut (TR.F.NetIncome, TR.F.TotRevenue) + Market Cap
       antavat suoraan oikean "trailing" arvon ilman per-share-ongelmia

Tama tiedosto:
    1. Testaa mita raaka-komponentteja Refinitivista loytyy
    2. Vertaa NaN-kattavuutta per kentta
    3. Laskee itse E/P, P/CF, P/S, DY ja vertaa Refinitivin valmiisiin ratioihin
    4. Tulostaa yhteenvedon

AJO:
    venv-v3/bin/python fetch_first_principles.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import refinitiv.data as rd

from config import PARAMS_DAILY
from src.data_fetch import _coerce_numeric_columns, _clean_stock_fundamentals

rd.open_session()

PROJECT_ROOT = Path(__file__).resolve().parent
SMOKE_PATH = PROJECT_ROOT / "data" / "01_raw" / "smoke_stocks.csv"
OUT_DIR = PROJECT_ROOT / "data" / "01_raw" / "first_principles"


# =============================================================================
# A) Raaka-komponenttikentät joita haetaan kokeeksi
# =============================================================================
# Nämä ovat TOTAL-LEVEL-lukuja (ei per-share), koska
# ratio = total_luku / market_cap  on luotettavampi kuin per_share / price.

COMPONENT_FIELDS = {
    # --- Hinta & markkina-arvo (päivittäinen, benchmark) ---
    "price_mktcap": [
        "TR.PriceClose",
        "TR.CompanyMarketCap",
        "TR.SharesOutstanding",
    ],

    # --- Tulos (Net Income) — E/P:n laskemiseen ---
    "net_income": [
        "TR.F.NetIncome",                   # fiscal net income (total)
        "TR.F.NetIncAfterTax",              # net income after tax
        "TR.F.NetIncBeforeExtra",           # net income before extraordinary
        "TR.NetIncome",                     # datastream variant
        "TR.NetIncomeMean",                 # analyst consensus (forward)
        "TR.EPSActValue",                   # per-share vertailu
    ],

    # --- Liikevaihto (Revenue) — P/S:n laskemiseen ---
    "revenue": [
        "TR.F.TotRevenue",                  # total revenue (fiscal)
        "TR.Revenue",                       # datastream variant
        "TR.RevenueActValue",               # actual reported
        "TR.RevenueMean",                   # analyst consensus
        "TR.F.SalesPerShare",               # per-share (vertailu)
        "TR.PriceToSalesPerShare",          # Refinitivin valmis P/S (vertailu)
    ],

    # --- Kassavirta (Cash Flow) — P/CF:n laskemiseen ---
    "cash_flow": [
        "TR.F.CF",                          # nykyinen (total CF, fiscal)
        "TR.F.OpCF",                        # operating cash flow
        "TR.F.FreeCashFlow",                # free cash flow
        "TR.F.CashFlowPerShare",            # per-share (vertailu)
        "TR.PriceToCFPerShare",             # Refinitivin valmis P/CF (vertailu)
    ],

    # --- Osingot (Dividends) — DY:n laskemiseen ---
    "dividends": [
        "TR.DPSActValue",                   # nykyinen per-share
        "TR.F.DivPerShare",                 # fiscal per-share
        "TR.F.TotDivPaid",                  # TOTAL dividends paid (koko yhtiö)
        "TR.F.CashDivPaidCommon",           # cash div paid (common)
        "TR.F.CommonDividends",             # common dividends (total)
        "TR.DividendYield",                 # Refinitivin valmis DY (vertailu)
    ],
}


def get_smoke_universe():
    if not SMOKE_PATH.exists():
        raise FileNotFoundError(
            f"smoke_stocks.csv puuttuu: {SMOKE_PATH}\n"
            "Aja ensin main_test.py jotta universumi on maaritelty."
        )
    df = pd.read_csv(SMOKE_PATH)
    return sorted(df["Instrument"].dropna().unique().tolist())


def fetch_single_field(universe, field):
    """Hakee yhden kentan. Palauttaa DataFramen tai None."""
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
        print(f"  X  {field:<45} VIRHE: {type(e).__name__}: {str(e)[:80]}")
        return None


def classify(df, col):
    """NaN%, dailyness, tyyppi."""
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
    return {"non_nan": nonnan, "total": total, "pct_nan": pct_nan,
            "dailyness": dailyness, "kind": kind}


# =============================================================================
# B) Itse-laskenta: haetaan price + mktcap + parhaat komponentit
# =============================================================================

def fetch_base_data(universe):
    """Hakee hinnan, markkina-arvon ja osakemäärän (nämä ovat daily)."""
    df = rd.get_data(
        universe=universe,
        fields=[
            "TR.PriceClose.date",
            "TR.PriceClose",
            "TR.CompanyMarketCap",
            "TR.SharesOutstanding",
        ],
        parameters=PARAMS_DAILY,
    )
    df = _coerce_numeric_columns(df)
    df = _clean_stock_fundamentals(df)
    return df


def fetch_best_components(universe):
    """
    Hakee parhaat komponenttikentät itse-laskentaa varten.
    Nämä valitaan kokeellisten tulosten perusteella.
    """
    # Haetaan kaikki kerralla (tehokkaampi kuin yksi kerrallaan)
    best_fields = [
        "TR.PriceClose.date",
        "TR.PriceClose",
        "TR.CompanyMarketCap",
        # Net Income (E/P)
        "TR.F.NetIncome",
        "TR.EPSActValue",
        # Revenue (P/S)
        "TR.F.TotRevenue",
        "TR.PriceToSalesPerShare",
        # Cash Flow (P/CF)
        "TR.F.CF",
        "TR.PriceToCFPerShare",
        # Dividends (DY)
        "TR.DPSActValue",
        "TR.F.TotDivPaid",
        "TR.DividendYield",
    ]
    try:
        df = rd.get_data(
            universe=universe,
            fields=best_fields,
            parameters=PARAMS_DAILY,
        )
        df = _coerce_numeric_columns(df)
        df = _clean_stock_fundamentals(df)
        return df
    except Exception as e:
        print(f"  Batch fetch VIRHE: {e}")
        return None


def compute_self_ratios(df):
    """
    Laskee tunnusluvut itse raaka-komponenteista.
    Vertaa Refinitivin valmiisiin ratioihin.
    """
    results = {}

    mktcap = df.get("Company Market Cap")
    price = df.get("Price Close")

    # --- E/P = Net Income / Market Cap ---
    net_inc = df.get("Net Income")
    if net_inc is not None and mktcap is not None:
        results["Self_EP"] = net_inc / mktcap
    else:
        print("  ! E/P: Net Income tai Market Cap puuttuu")

    # EPS-pohjainen vertailu
    eps = df.get("Earnings Per Share - Actual Value")
    if eps is not None and price is not None:
        results["Ref_EP_perShare"] = eps / price

    # --- neg P/S = -Market Cap / Revenue ---
    rev = df.get("Total Revenue")
    if rev is not None and mktcap is not None:
        results["Self_negPS"] = -(mktcap / rev)

    ps_ref = df.get("Price To Sales Per Share")
    if ps_ref is not None:
        results["Ref_PS"] = ps_ref

    # --- neg P/CF = -Market Cap / Cash Flow ---
    cf = df.get("Cash Flow")
    if cf is not None and mktcap is not None:
        results["Self_negPCF"] = -(mktcap / cf)

    pcf_ref = df.get("Price To Cash Flow Per Share")
    if pcf_ref is not None:
        results["Ref_PCF"] = pcf_ref

    # --- DY = Total Dividends / Market Cap ---
    totdiv = df.get("Total Dividends Paid")
    if totdiv is not None and mktcap is not None:
        # Total Dividends Paid on yleensä negatiivinen (kassavirta ulos)
        results["Self_DY"] = totdiv.abs() / mktcap

    dy_ref = df.get("Dividend Yield")
    if dy_ref is not None:
        results["Ref_DY"] = dy_ref / 100  # prosenteista desimaaliksi

    dps = df.get("Dividend Per Share - Actual Value")
    if dps is not None and price is not None:
        results["Ref_DY_perShare"] = dps / price

    return results


def compare_coverage(df, ratios):
    """Tulostaa NaN-vertailun: itse laskettu vs. Refinitivin valmis."""
    total = len(df)
    print(f"\n{'=' * 90}")
    print(f"  VERTAILU: itse laskettu vs. Refinitivin valmis  (N = {total:,})")
    print(f"{'=' * 90}")
    print(f"{'Tunnusluku':<25} {'non-NaN':>10} {'%NaN':>8} {'%Inf':>8} {'median':>12}")
    print("-" * 90)

    for name, series in sorted(ratios.items()):
        # Korvaa inf -> NaN laskentaa varten
        clean = series.replace([np.inf, -np.inf], np.nan)
        nonnan = clean.notna().sum()
        pct_nan = (1 - nonnan / total) * 100
        n_inf = np.isinf(series).sum() if series.dtype != object else 0
        pct_inf = (n_inf / total) * 100
        med = clean.median()
        med_str = f"{med:.6f}" if pd.notna(med) else "N/A"
        tag = "<<< SELF" if name.startswith("Self_") else "    ref"
        print(f"{name:<25} {nonnan:>10,} {pct_nan:>7.1f}% {pct_inf:>7.1f}% {med_str:>12}  {tag}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = get_smoke_universe()
    print(f"Universumi: {len(universe)} RIC:ia")
    for r in universe:
        print(f"  - {r}")
    print(f"\nAikaväli: {PARAMS_DAILY['SDate']} -> {PARAMS_DAILY['EDate']}")

    # =========================================================================
    # VAIHE 1: Testaa kaikki komponenttikentat yksitellen
    # =========================================================================
    print("\n" + "#" * 90)
    print("# VAIHE 1: Kenttien saatavuus (yksittäiset haut)")
    print("#" * 90)

    all_results = {}
    for group, fields in COMPONENT_FIELDS.items():
        print(f"\n--- {group.upper()} ({len(fields)} kenttaa) ---")
        group_results = []
        for field in fields:
            print(f"  -> {field}")
            df = fetch_single_field(universe, field)
            if df is None or df.empty:
                group_results.append({
                    "field": field, "col": "-", "non_nan": 0,
                    "total": 0, "pct_nan": 100, "dailyness": 0, "kind": "FAILED"
                })
                continue
            data_cols = [c for c in df.columns if c not in {"Instrument", "Date"}]
            for col in data_cols:
                stats = classify(df, col)
                if stats:
                    group_results.append({"field": field, "col": col, **stats})

        group_results.sort(key=lambda r: r["pct_nan"])
        all_results[group] = group_results

        # Tulosta ryhmätaulukko
        print(f"\n{'Kentta':<45} {'Sarake':<35} {'non-NaN':>8} {'%NaN':>6} {'daily':>6} {'tyyppi':>8}")
        print("-" * 110)
        for r in group_results:
            print(
                f"{r['field']:<45} {r['col']:<35} {r['non_nan']:>8} "
                f"{r['pct_nan']:>5.1f}% {r['dailyness']:>6.3f} {r['kind']:>8}"
            )

    # =========================================================================
    # VAIHE 2: Itse-laskenta ja vertailu
    # =========================================================================
    print("\n" + "#" * 90)
    print("# VAIHE 2: Itse-laskenta vs. Refinitivin valmiit ratiot")
    print("#" * 90)

    print("\nHaetaan kaikki komponentit batch-hakuna...")
    df_all = fetch_best_components(universe)

    if df_all is not None:
        print(f"Haettu {len(df_all):,} rivia, sarakkeet:")
        for c in df_all.columns:
            nn = df_all[c].notna().sum()
            print(f"  {c:<45} non-NaN: {nn:>8,}  ({nn/len(df_all)*100:.1f}%)")

        ratios = compute_self_ratios(df_all)
        compare_coverage(df_all, ratios)

        # Tallenna raaka + lasketut
        df_out = df_all.copy()
        for name, series in ratios.items():
            df_out[name] = series
        out_path = OUT_DIR / "first_principles_comparison.csv"
        df_out.to_csv(out_path, index=False)
        print(f"\nTallennettu: {out_path}")
    else:
        print("Batch-haku epäonnistui — tarkista Refinitiv Workspace -yhteys")

    # =========================================================================
    # VAIHE 3: Suositukset
    # =========================================================================
    print("\n" + "=" * 90)
    print("  SUOSITUKSET")
    print("=" * 90)
    print("""
    E/P:
      - JOS TR.F.NetIncome löytyy -> laske itse: Net Income / Market Cap
      - Ei tarvitse TTM rolling sum -logiikkaa, koska total-level luku
      - Vaihtoehto: TR.EPSActValue / Price (mutta vaatii change-point fix)

    P/S:
      - JOS TR.F.TotRevenue löytyy -> laske itse: -(Market Cap / Revenue)
      - Parempi kattavuus kuin TR.PriceToSalesPerShare

    P/CF:
      - TR.F.CF on jo hyvä (~7% NaN) -> nykyinen compute_neg_pcf on OK
      - JOS TR.F.OpCF löytyy, se voi olla vaihtoehto (Operating CF)

    DY:
      - JOS TR.F.TotDivPaid löytyy -> |Total Div Paid| / Market Cap
      - Ei tarvitse merge_asof + rolling sum -logiikkaa
      - Vaihtoehto: TR.DPSActValue / Price (mutta vaatii change-point fix)
    """)


if __name__ == "__main__":
    main()
