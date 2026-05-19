"""
Korvaa rikkinäiset Index_B/M-arvot tuotantopaneelissa omalla market-cap-
painotetulla harmonisella aggregaatilla universumin osakkeiden P/B-arvoista.

Tausta:
    Refinitivin TR.Index_PRICE_TO_BOOK_RTRS palauttaa ~0.0001 placeholder-
    arvon STOXX-indeksille joillain kuukausilla (havaittu 2026-Q1 alkaen).
    Tämä räjäyttää johdetun Index_B/M = 1/(P/B) -saraketta 10 000+ -tasolle.
    Refinitivin uudelleenhaku ei korjaa (vahvistettu diagnostics/
    index_features_audit.py:llä).

Algoritmi:
    1. Tunnistaa kuukaudet joissa Index_B/M > BROKEN_BM_THRESHOLD
    2. Per rikkinäinen kuukausi:
         MktCap_i = exp(log_MktCap_i)
         own_PB   = Σ MktCap_i / Σ (MktCap_i × (1/P/B)_i)
         own_BM   = 1 / own_PB
       Tämä on matemaattisesti yhtä kuin Σ MktCap / Σ BookValue, joka
       on indeksitason P/B:n oikea aggregaatti.
    3. Korvaa Index_B/M-arvo niissä riveissä saman kuukauden sisällä.
    4. Tallentaa output _fixed.csv-suffiksilla; alkuperäinen säilyy.

Lähdetiedosto: data/02_preprocessed/MASTER_DF_PROD_JM2_nonnan_winsor.csv
Output:        data/02_preprocessed/MASTER_DF_PROD_JM2_nonnan_winsor_fixed.csv

Ei Refinitiv-kutsuja, ei muita lähdetiedostoja. Idempotentti — jos ajetaan
uudestaan, sama tulos (uutta korjaa ei tarvita kun Index_B/M ei enää ylitä
kynnystä).

Ajo:
    python scripts/fix_index_pb_outliers.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROD_IN  = PROJECT_ROOT / "data" / "02_preprocessed" / "MASTER_DF_PROD_JM2_nonnan_winsor.csv"
PROD_OUT = PROJECT_ROOT / "data" / "02_preprocessed" / "MASTER_DF_PROD_JM2_nonnan_winsor_fixed.csv"

# Legitiimi Index_B/M on tyypillisesti ~0.4-0.8; rikkinäiset arvot >10000.
BROKEN_BM_THRESHOLD = 10.0


def main():
    if not PROD_IN.exists():
        raise FileNotFoundError(f"Lähde puuttuu: {PROD_IN}")

    prod = pd.read_csv(PROD_IN, parse_dates=["Date"])

    broken_dates = sorted(prod.loc[prod["Index_B/M"] > BROKEN_BM_THRESHOLD, "Date"].unique())
    if not broken_dates:
        print(f"Ei rikkinäisiä kuukausia (kynnys Index_B/M > {BROKEN_BM_THRESHOLD}). Mitään ei korjattu.")
        return

    print(f"Tunnistettu {len(broken_dates)} rikkinäistä kuukautta. Korjataan…\n")
    print(f"{'Kuukausi':<14}{'old Index_B/M':>16}{'new Index_B/M':>16}{'own P/B':>12}{'stocks':>10}")
    print("-" * 68)

    for date in broken_dates:
        rows = prod["Date"] == date
        block = prod.loc[rows].copy()
        block["MktCap"] = np.exp(block["log_MktCap"])

        numer = block["MktCap"].sum()
        denom = (block["MktCap"] * block["1/P/B"]).sum()
        if denom <= 0:
            raise ValueError(f"Aggregaatin nimittäjä ≤ 0 kuukaudelle {date}; ei voi korjata.")

        own_pb = numer / denom
        own_bm = 1.0 / own_pb
        old_bm = prod.loc[rows, "Index_B/M"].iloc[0]

        prod.loc[rows, "Index_B/M"] = own_bm

        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        print(f"{date_str:<14}{old_bm:>16.4f}{own_bm:>16.4f}{own_pb:>12.4f}{len(block):>10}")

    PROD_OUT.parent.mkdir(parents=True, exist_ok=True)
    prod.to_csv(PROD_OUT, index=False)

    print(f"\n  Korjattu {len(broken_dates)} kk.")
    print(f"  Tallennettu: {PROD_OUT}")
    print(f"  Alkuperäinen säilytetty: {PROD_IN}")


if __name__ == "__main__":
    main()
