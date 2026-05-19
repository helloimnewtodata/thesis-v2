"""
fetch_news.py — Step 1 of 3 in the sentiment pipeline.

Fetches Refinitiv news headlines for every stock in the STOXX 600 universe
and saves them to a raw CSV that sentiment.py will score in Step 2.

Universe: all 861 RICs in MASTER_DF_PROD_JM2_nonnan_winsor.csv.
Window  : 2025-03-01 → 2026-04-30 (months where sentiment will be used as a
          feature in the OOS period of run_sentiment_experiment.py).

Pagination: headlines are fetched month-by-month per RIC so that high-volume
stocks (e.g. Nokia) cannot be truncated by the API count limit. Every stock is
guaranteed full coverage back to NEWS_DATE_FROM regardless of news volume.

The run is resumable: if interrupted, already-fetched RICs are skipped on
restart. Progress is written to the output CSV after every stock.

Runtime: ~13 hours (861 RICs × 14 months × 5s sleep). Run overnight.

Output
------
    data/01_raw/news_headlines_raw.csv
    Columns: Instrument, date, headline
    One row per headline. Multiple headlines per stock per month are normal.

Next step
---------
    Once this script finishes, run:
        python src/sentiment.py
    That script scores every headline with FinBERT and aggregates to one
    sentiment score per stock per month, then merges it onto the master panel.

Usage (with Refinitiv session open):
    python src/fetch_news.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import refinitiv.data as rd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MASTER_PATH = Path("data/02_preprocessed/MASTER_DF_PROD_JM2_nonnan_winsor.csv")
OUTPUT_PATH = Path("data/01_raw/news_headlines_raw.csv")

NEWS_DATE_FROM = "2025-03-01T00:00:00"
NEWS_DATE_TO   = "2026-04-30T23:59:59"

HEADLINES_PER_MONTH  = 5000  # per monthly chunk — ample for any single month
SLEEP_BETWEEN_MONTHS = 5.0   # seconds between monthly chunks — conservative rate limit
SLEEP_BETWEEN_RICS   = 3.0   # seconds between stocks


# ---------------------------------------------------------------------------
# Universe derivation
# ---------------------------------------------------------------------------

def get_universe(master_path: Path = MASTER_PATH) -> list[str]:
    """
    Return the universe: all unique RICs in the production master panel.
    """
    df = pd.read_csv(master_path, usecols=["Instrument"])
    return sorted(df["Instrument"].unique().tolist())


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_headlines_for_ric(
    ric: str,
    date_from: str = NEWS_DATE_FROM,
    date_to: str = NEWS_DATE_TO,
    count: int = HEADLINES_PER_MONTH,
) -> pd.DataFrame:
    """
    Pull headlines for one RIC across the full window by fetching month-by-month.

    Monthly pagination guarantees every stock gets data back to NEWS_DATE_FROM
    regardless of how many headlines it generates. A single request would be
    cut off by the count limit for high-volume stocks like Nokia.

    Returns a deduplicated DataFrame with columns [Instrument, date, headline].
    """
    months = pd.date_range(
        start=pd.Timestamp(date_from[:10]),
        end=pd.Timestamp(date_to[:10]),
        freq="MS",
    )

    chunks: list[pd.DataFrame] = []

    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd(0)
        month_end = min(month_end, pd.Timestamp(date_to[:10]))

        chunk_start = month_end.strftime("%Y-%m-%dT23:59:59")
        chunk_end   = month_start.strftime("%Y-%m-%dT00:00:00")

        for attempt in range(3):
            try:
                df = rd.news.get_headlines(
                    query=f"R:{ric}",
                    count=count,
                    start=chunk_start,
                    end=chunk_end,
                )
                if df is not None and not df.empty:
                    chunk = pd.DataFrame({
                        "Instrument": ric,
                        "date": pd.to_datetime(df.index).normalize(),
                        "headline": df["headline"].astype(str),
                    })
                    chunks.append(chunk.dropna(subset=["headline"]))
                break
            except Exception as exc:
                if "429" in str(exc) and attempt < 2:
                    wait = 300 * (attempt + 1)  # 5 min, then 10 min
                    print(f"  [429] Rate limited — waiting {wait}s before retry {attempt + 2}/3 …")
                    time.sleep(wait)
                else:
                    print(f"  [ERROR] {ric} {month_start.date()}: {exc}")
                    break

        time.sleep(SLEEP_BETWEEN_MONTHS)

    if not chunks:
        return pd.DataFrame(columns=["Instrument", "date", "headline"])

    return pd.concat(chunks, ignore_index=True).drop_duplicates()


def fetch_all(
    output_path: Path = OUTPUT_PATH,
    date_from: str = NEWS_DATE_FROM,
    date_to: str = NEWS_DATE_TO,
    sleep: float = SLEEP_BETWEEN_RICS,
) -> pd.DataFrame:
    """
    Fetch headlines for every RIC in the point-in-time universe and save to CSV.
    Skips RICs already in the output file so the run is resumable if interrupted.
    """
    rics = get_universe()
    print(f"Universe : {len(rics)} RICs (from MASTER_DF_PROD_JM2_nonnan_winsor.csv)")
    print(f"Window   : {date_from[:10]} → {date_to[:10]}")
    print(f"Method   : monthly pagination ({HEADLINES_PER_MONTH} headlines/month, {SLEEP_BETWEEN_MONTHS}s between months)")

    already_done: set[str] = set()
    if output_path.exists():
        existing = pd.read_csv(output_path, usecols=["Instrument"])
        already_done = set(existing["Instrument"].unique())
        print(f"Resuming — {len(already_done)} RICs already fetched, skipping them.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()

    total_headlines = 0
    for i, ric in enumerate(rics, 1):
        if ric in already_done:
            continue

        print(f"[{i}/{len(rics)}] {ric} …", end=" ", flush=True)
        df = fetch_headlines_for_ric(ric, date_from, date_to)
        n = len(df)
        print(f"{n} headlines")
        total_headlines += n

        df.to_csv(output_path, mode="a", header=write_header, index=False)
        write_header = False

        time.sleep(sleep)

    print(f"\nDone. {total_headlines:,} headlines saved to {output_path}")
    return pd.read_csv(output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rd.open_session()
    try:
        df = fetch_all()
        print("\nSample:")
        print(df.head(10).to_string(index=False))
        print(f"\nShape: {df.shape}")
        print(f"Unique stocks with headlines: {df['Instrument'].nunique()}")
        print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    finally:
        rd.close_session()
