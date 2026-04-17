"""
Pääorkesteri: ajaa koko pipelinen data fetch → features → composites → models.
"""

import pandas as pd
from config import (
    SECTOR_DUMMIES, SECTOR_DUMMY_NAMES, TRAIN_END,
)
from src.data_fetch import (
    open_session,
    fetch_stock_fundamentals,
    fetch_index_data,
    fetch_euribor,
)
from src.features import compute_all_features
from src.composites import build_composite_signals


def load_universe(path="data/stoxx600_universe.csv"):
    """Lataa STOXX 600 -universumin RIC-lista CSV:stä."""
    df = pd.read_csv(path)
    return df["RIC"].tolist()


def main():
    # 0. Avaa Refinitiv-sessio
    open_session()

    # 1. Lataa universumi
    universe = load_universe()
    print(f"Universumi: {len(universe)} osaketta")

    # 2. Hae data
    print("Haetaan osakkeiden fundamentaalit...")
    df_stocks = fetch_stock_fundamentals(universe)

    print("Haetaan indeksidata...")
    df_index, df_index_fundamentals = fetch_index_data()

    print("Haetaan EURIBOR...")
    df_euribor = fetch_euribor()

    # 3. Laske featuret
    print("Lasketaan featuret...")
    df_features, df_idx = compute_all_features(
        df_stocks, df_index, df_index_fundamentals, df_euribor,
        SECTOR_DUMMIES, SECTOR_DUMMY_NAMES,
    )

    # 4. Rakenna komposiittisignaalit
    print("Rakennetaan komposiittisignaalit...")
    df_signals = build_composite_signals(df_features, train_end=TRAIN_END)

    # 5. Liitä Market-signaali indeksitasolta
    # (Market-ryhmän Z-scoret ovat time-series, ei cross-sectional)
    df_idx["Date"] = df_idx.index
    market_cols = ["Date", "Z_Calculated PE Ratio", "Z_Calculated Price to Book", "Z_Calculated Index Dividend Yield"]
    df_signals = df_signals.merge(
        df_idx[market_cols],
        on="Date",
        how="left",
    )
    df_signals["Signal_Market"] = df_signals[
        ["Z_Calculated PE Ratio", "Z_Calculated Price to Book", "Z_Calculated Index Dividend Yield"]
    ].mean(axis=1)

    # 6. Tallenna
    print("Tallennetaan...")
    df_signals.to_csv("data/02_preprocessed/df_features_final.csv", index=False)
    df_idx.to_csv("data/02_preprocessed/df_index_features.csv", index=False)
    print("Valmis!")

    # 7. Mallinnus (TODO)
    # results = run_walk_forward(df_signals)


if __name__ == "__main__":
    main()
