"""
Index-feature auditointi koko aikasarjalle.

Re-fetchaa Refinitivin raakakentät .STOXX-indeksille ja vertaa niitä
MASTER_DF_1.csv:n arvoihin. Raportoi kuukaudet joissa on:
  - epätodennäköisiä arvoja (per-field plausibility range)
  - NaN:eja raakahausta
  - >10 % diffi nykyisen masterin arvoon (Refinitiv on korjannut tai
    alkuperäinen Stage-1-haku epäonnistui)

Auditoitavat raakakentät (haetaan):
    TR.Index_PE_RTRS              → "Calculated PE Ratio"
    TR.Index_PRICE_TO_BOOK_RTRS   → "Calculated Price to Book"
    TR.Index_DIV_YLD_RTRS         → "Calculated Index Dividend Yield"

Auditoitavat johdetut kentät (vain järkevyystarkistus MASTER_DF_1:stä):
    Index_Index_MOM_1M
    Index_Index_MOM_12M

Outputs:
    data/diagnostics/index_features_audit_<timestamp>.csv
        kuukausittainen vertailu (master vs fresh, diff %)
    Konsoliyhteenveto: outlier-kuukaudet, diffit, NaN:t

Ajo (Refinitiv-sessio avoinna):
    python diagnostics/index_features_audit.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

import src.data_fetch as data_fetch
from config import FETCH_START_DATE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = PROJECT_ROOT / "data" / "02_preprocessed" / "MASTER_DF_1.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "diagnostics"

# Plausibility-rajat per kenttä (STOXX 600 -tyyppinen laajapohjainen indeksi)
PLAUSIBILITY = {
    "Calculated PE Ratio":                {"min": 5.0,   "max": 60.0,  "near_zero": 1.0},
    "Calculated Price to Book":           {"min": 0.5,   "max": 5.0,   "near_zero": 0.1},
    "Calculated Index Dividend Yield":    {"min": 1.0,   "max": 8.0,   "near_zero": 0.1},
}

# Master-tiedoston sarakenimet vastaavat raakakenttiä "Index_"-prefixillä
MASTER_COL_MAP = {
    "Calculated PE Ratio":             "Index_Calculated PE Ratio",
    "Calculated Price to Book":        "Index_Calculated Price to Book",
    "Calculated Index Dividend Yield": "Index_Calculated Index Dividend Yield",
}

# Johdetut momentum-kentät (eivät vaadi refetchia, vain järkevyystarkistus)
DERIVED_PLAUSIBILITY = {
    "Index_Index_MOM_1M":  {"min": -0.20, "max": 0.20},
    "Index_Index_MOM_12M": {"min": -0.50, "max": 0.70},
}

DIFF_THRESHOLD_PCT = 10.0  # raportoidaan jos |master - fresh| / |master| > tämä


def fetch_index_fields_fresh(start: str, end: str) -> pd.DataFrame:
    """Refetch .STOXX-indeksin kolme calculated-fieldiä koko aikaväliltä.

    Avaa Refinitiv-session jos ei jo ole avoinna.
    """
    import refinitiv.data as rd

    print(f"Avataan Refinitiv-sessio…")
    data_fetch.open_session()

    print(f"Haetaan .STOXX-indeksin raakakentät {start} → {end}")
    df = rd.get_history(
        universe=[".STOXX"],
        fields=[
            "TR.Index_PE_RTRS",
            "TR.Index_PRICE_TO_BOOK_RTRS",
            "TR.Index_DIV_YLD_RTRS",
        ],
        start=start,
        end=end,
        interval="1D",
    )
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
    return df


def month_end_panel(daily: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Resamplaa päivätaajuus kuukauden lopuksi (viimeinen kaupankäyntipäivä)."""
    out = (
        daily.set_index("Date")[cols]
        .resample("M")
        .last()
        .reset_index()
    )
    out["Date"] = out["Date"] + pd.offsets.MonthEnd(0)
    return out


def load_master_index_panel() -> pd.DataFrame:
    """Lataa master-paneelin kuukausittaiset indeksiarvot (yksi rivi per kk)."""
    print(f"Ladataan {MASTER_PATH}…")
    df = pd.read_csv(MASTER_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    cols = list(MASTER_COL_MAP.values()) + list(DERIVED_PLAUSIBILITY.keys())
    return df.groupby("Date", as_index=False)[cols].first()


def find_anomalies(series: pd.Series, bounds: dict) -> pd.Series:
    """Palauttaa boolean-maskin: True jos arvo on epätodennäköinen."""
    near_zero = bounds.get("near_zero")
    mask = (series < bounds["min"]) | (series > bounds["max"])
    if near_zero is not None:
        mask |= series.abs() < near_zero
    return mask


def main():
    end_date = datetime.now().strftime("%Y-%m-%d")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Refetch raakakentät
    fresh_daily = fetch_index_fields_fresh(FETCH_START_DATE, end_date)
    fresh_monthly = month_end_panel(fresh_daily, list(MASTER_COL_MAP.keys()))
    fresh_monthly = fresh_monthly.rename(
        columns={raw: f"{raw}_fresh" for raw in MASTER_COL_MAP.keys()}
    )

    # 2. Lataa master
    master = load_master_index_panel()

    # 3. Yhdistä master + fresh kuukauden Date-keynä
    audit = master.merge(fresh_monthly, on="Date", how="outer").sort_values("Date")

    # 4. Laske diffit raakakentille
    print("\n" + "=" * 72)
    print("RAAKAKENTÄT — refetch vs master")
    print("=" * 72)

    raw_summary_rows = []
    for raw_field, master_col in MASTER_COL_MAP.items():
        fresh_col = f"{raw_field}_fresh"

        n_master_nan = audit[master_col].isna().sum()
        n_fresh_nan = audit[fresh_col].isna().sum()

        # Outlierit (per kenttä raja-arvotaulukosta)
        bounds = PLAUSIBILITY[raw_field]
        master_outliers = find_anomalies(audit[master_col], bounds)
        fresh_outliers = find_anomalies(audit[fresh_col], bounds)

        # Diff-% (vain kun molemmat eivät ole NaN)
        both = audit[[master_col, fresh_col]].dropna()
        if len(both) > 0:
            diff_abs = (both[master_col] - both[fresh_col]).abs()
            diff_pct = 100 * diff_abs / both[master_col].abs().replace(0, pd.NA)
            big_diffs = (diff_pct > DIFF_THRESHOLD_PCT).sum()
        else:
            big_diffs = 0

        raw_summary_rows.append({
            "field": raw_field,
            "n_months": len(audit),
            "master_NaN": int(n_master_nan),
            "fresh_NaN": int(n_fresh_nan),
            "master_outliers": int(master_outliers.sum()),
            "fresh_outliers": int(fresh_outliers.sum()),
            f"diff_>{DIFF_THRESHOLD_PCT:.0f}%": int(big_diffs),
        })

        print(f"\n→ {raw_field}")
        print(f"   Master NaN: {n_master_nan}   Fresh NaN: {n_fresh_nan}")
        print(f"   Master outliers (out of [{bounds['min']}, {bounds['max']}]): {master_outliers.sum()}")
        print(f"   Fresh  outliers: {fresh_outliers.sum()}")
        print(f"   Diffi master vs fresh > {DIFF_THRESHOLD_PCT:.0f} %: {big_diffs} kk")

        # Listaa outlier-kuukaudet (ensimmäiset 10)
        if master_outliers.sum() > 0 or fresh_outliers.sum() > 0:
            print(f"   Outlier-kuukaudet (näytetään max 10):")
            problem_mask = master_outliers | fresh_outliers
            for _, row in audit.loc[problem_mask, ["Date", master_col, fresh_col]].head(10).iterrows():
                date_str = row["Date"].strftime("%Y-%m-%d")
                m = row[master_col]
                f = row[fresh_col]
                m_str = f"{m:>10.4f}" if pd.notna(m) else "       NaN"
                f_str = f"{f:>10.4f}" if pd.notna(f) else "       NaN"
                print(f"     {date_str}   master={m_str}   fresh={f_str}")

        # Listaa kuukaudet joissa iso diff (ensimmäiset 10)
        if len(both) > 0 and big_diffs > 0:
            print(f"   Suurimmat diffit (näytetään max 10):")
            diffs_with_date = audit[["Date", master_col, fresh_col]].dropna()
            diffs_with_date["diff_pct"] = 100 * (diffs_with_date[master_col] - diffs_with_date[fresh_col]).abs() / diffs_with_date[master_col].abs().replace(0, pd.NA)
            for _, row in diffs_with_date.sort_values("diff_pct", ascending=False).head(10).iterrows():
                date_str = row["Date"].strftime("%Y-%m-%d")
                print(f"     {date_str}   master={row[master_col]:>10.4f}   fresh={row[fresh_col]:>10.4f}   diff={row['diff_pct']:>6.2f} %")

    # 5. Johdetut MOM-kentät (vain järkevyystarkistus, ei refetchia)
    print("\n" + "=" * 72)
    print("JOHDETUT KENTÄT — vain järkevyystarkistus (ei refetchia)")
    print("=" * 72)
    for col, bounds in DERIVED_PLAUSIBILITY.items():
        outliers = find_anomalies(master[col].dropna(), bounds)
        n_nan = master[col].isna().sum()
        print(f"\n→ {col}")
        print(f"   NaN: {n_nan}   Outliers (out of [{bounds['min']}, {bounds['max']}]): {outliers.sum()}")
        if outliers.sum() > 0:
            problem_dates = master.loc[master.index.isin(outliers[outliers].index), ["Date", col]]
            for _, row in problem_dates.head(10).iterrows():
                date_str = row["Date"].strftime("%Y-%m-%d")
                print(f"     {date_str}   {col}={row[col]:.4f}")

    # 6. Tallenna kuukausitason vertailu CSV:nä
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"index_features_audit_{timestamp}.csv"
    audit.to_csv(output_path, index=False)

    # 7. Tallenna yhteenvetotaulu erikseen
    summary_path = OUTPUT_DIR / f"index_features_audit_summary_{timestamp}.csv"
    pd.DataFrame(raw_summary_rows).to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print(f"Tallennettu: {output_path}")
    print(f"Tallennettu: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
