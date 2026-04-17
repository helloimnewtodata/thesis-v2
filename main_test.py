"""
Smoke-versio täydestä pipeline-ajosta:
1. luo universumitiedoston snapshotista
2. ottaa siitä 10 SATUNNAISTA RIC:iä
3. hakee warmup-ikkunan kattavan osake-, indeksi-, indeksifunda- ja EURIBOR-datan
4. tallentaa 4 raaka-CSV:tä (Stage 1) → data/01_raw/
5. laskee featuret (Stage 2)
6. leikkaa warmupin pois
7. tallentaa master-paneelin → data/02_preprocessed/
8. tulostaa shape + NaN-raportin

Tämä on 1:1 sama rakenne kuin main.py tuotannossa, vain 10 osakkeella ja ilman
komposiittisignaaleja (build_composite_signals ajetaan vasta main.py:stä).

HUOM: features.py:n compute_idiosyncratic_vol on Python-loop O(n * window).
10 osakkeella smoke-ajossa siedettävä, mutta vektoroi ennen 600 osakkeen
tuotantoajoa. Ehdotus vektoroidusta versiosta on features.py:n kommenteissa.
"""

import random
from pathlib import Path

import pandas as pd
import refinitiv.data as rd

import src.data_fetch as data_fetch
from src.build_stoxx_universe import save_stoxx_universe_snapshot
from src.features import compute_all_features
from config import (
    DISPLAY_START_DATE,
    FETCH_START_DATE,
    END_DATE,
    SECTOR_DUMMIES,
    SECTOR_DUMMY_NAMES,
)


PROJECT_ROOT = Path(__file__).resolve().parent
UNIVERSE_PATH = PROJECT_ROOT / "data" / "stoxx600_universe.csv"
RAW_OUTPUT_DIR = PROJECT_ROOT / "data" / "01_raw"
PREPROCESSED_OUTPUT_DIR = PROJECT_ROOT / "data" / "02_preprocessed"

SMOKE_UNIVERSE_SIZE = 10


def print_summary(name, df):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"shape: {df.shape}")
    print("columns:")
    for column in df.columns:
        print(f"  - {column}")
    print("\nhead:")
    print(df.head())


def prepare_export_df(name, df):
    export_df = df.copy()

    if name == "index_fundamentals":
        # rd.get_history() palauttaa päivämäärän indexissä, joten nostetaan se
        # sarakkeeksi ennen tallennusta.
        if export_df.index.name != "Date":
            export_df.index.name = "Date"
        export_df = export_df.reset_index()
        if "Date" in export_df.columns:
            export_df["Date"] = pd.to_datetime(export_df["Date"]).dt.date

    return export_df


def save_raw_outputs(datasets):
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    for name, df in datasets.items():
        path = RAW_OUTPUT_DIR / f"smoke_{name}.csv"
        export_df = prepare_export_df(name, df)
        export_df.to_csv(path, index=False)
        saved_paths[name] = path

    return saved_paths


def save_master(df, filename="master_smoke.csv"):
    PREPROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = PREPROCESSED_OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    return path


def pick_random_universe(snapshot, size):
    """Valitse satunnaiset RIC:it — ei seediä, joka ajolla eri näyte."""
    rics = snapshot["RIC"].dropna().unique().tolist()
    if len(rics) <= size:
        return rics
    return random.sample(rics, size)


def nan_report(df, feature_cols):
    """NaN-rivien osuus featuresaraketta kohti — debuggausta varten."""
    report = []
    n = len(df)
    for col in feature_cols:
        if col not in df.columns:
            report.append((col, "MISSING", "-"))
            continue
        n_nan = df[col].isna().sum()
        pct = (n_nan / n * 100) if n > 0 else 0
        report.append((col, n_nan, f"{pct:.1f}%"))
    return report


def main():
    print(f"Warmup-fetch: {FETCH_START_DATE} → {END_DATE}")
    print(f"Master-paneelin näkyvä alku: {DISPLAY_START_DATE}")

    rd.open_session()
    try:
        # 1. Universumi
        snapshot, output_path = save_stoxx_universe_snapshot(output_path=UNIVERSE_PATH)
        universe = pick_random_universe(snapshot, SMOKE_UNIVERSE_SIZE)

        print(f"\nUniversumitiedosto: {output_path}")
        print(f"Satunnaiset {len(universe)} RIC:iä tähän ajoon:")
        for ric in universe:
            print(f"  - {ric}")

        # 2. Stage 1: raakadata (warmupin kanssa)
        print("\nHaetaan osakedata...")
        df_stocks = data_fetch.fetch_stock_fundamentals(universe)

        print("Haetaan indeksidata...")
        df_index, df_index_fundamentals = data_fetch.fetch_index_data()

        print("Haetaan EURIBOR...")
        df_euribor = data_fetch.fetch_euribor()
    finally:
        rd.close_session()

    # 3. Tallenna 4 raaka-CSV:tä
    saved_paths = save_raw_outputs(
        {
            "stocks": df_stocks,
            "index": df_index,
            "index_fundamentals": df_index_fundamentals,
            "euribor": df_euribor,
        }
    )

    print_summary("df_stocks (raw)", df_stocks)
    print_summary("df_index (raw)", df_index)
    print_summary("df_index_fundamentals (raw)", df_index_fundamentals)
    print_summary("df_euribor (raw)", df_euribor)

    print("\nTallennetut raaka-CSV:t (Stage 1):")
    for name, path in saved_paths.items():
        print(f"  - {name}: {path}")

    # 4. Stage 2: featurelaskenta (warmupin kanssa, jotta rolling-ikkunat kypsyvät)
    print("\n" + "=" * 80)
    print("Lasketaan featuret...")
    print("=" * 80)
    df_features, df_idx = compute_all_features(
        df_stocks,
        df_index,
        df_index_fundamentals,
        df_euribor,
        SECTOR_DUMMIES,
        SECTOR_DUMMY_NAMES,
    )

    # 5. Leikkaa warmup pois
    print(f"\nLeikataan warmup pois — master alkaa {DISPLAY_START_DATE}:stä.")
    df_master = data_fetch.trim_warmup(df_features, DISPLAY_START_DATE)
    df_idx_trimmed = data_fetch.trim_warmup(df_idx, DISPLAY_START_DATE)

    print(f"Ennen leikkausta: {df_features.shape}")
    print(f"Leikkauksen jälkeen: {df_master.shape}")

    # 6. Tallenna master
    master_path = save_master(df_master, "master_smoke.csv")
    idx_path = save_master(
        df_idx_trimmed.reset_index() if df_idx_trimmed.index.name == "Date" else df_idx_trimmed,
        "master_smoke_index.csv",
    )
    print(f"\nMaster-paneeli:        {master_path}")
    print(f"Master-indeksifunda:   {idx_path}")

    # 7. NaN-raportti
    feature_cols = [
        "E/P", "1/P/B", "-P/S", "-P/CF", "DivYield_12M",
        "Return On Average Common Equity %", "Gross Profit / Total Assets", "-Debt/MktCap",
        "MOM_1M", "MOM_12M", "RSI_30d", "Hurst",
        "-Vol_30d", "-Beta_252d", "-IdioVol",
        "log_MktCap", "Excess_Return",
    ]
    print("\n" + "=" * 80)
    print("NaN-raportti master-paneelista")
    print("=" * 80)
    print(f"{'Feature':<45} {'NaN':>10} {'% NaN':>8}")
    print("-" * 65)
    for col, n_nan, pct in nan_report(df_master, feature_cols):
        print(f"{col:<45} {str(n_nan):>10} {pct:>8}")

    print_summary("master (Stage 2 output)", df_master)


if __name__ == "__main__":
    main()
