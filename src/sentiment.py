"""
sentiment.py — Step 2 of 3 in the sentiment pipeline.

Scores every news headline from fetch_news.py with FinBERT and aggregates
to one sentiment value per stock per month, then merges onto the master panel.

Model: ProsusAI/finbert — BERT fine-tuned on financial news text.
Score: P(positive) - P(negative) in [-1, +1].
  +1 = maximally positive (e.g. record earnings beat)
  -1 = maximally negative (e.g. fraud investigation)
   0 = neutral or balanced (e.g. routine management appointment)

FinBERT is used instead of general-purpose sentiment models (e.g. VADER)
because financial language carries domain-specific meaning: words like
"beat", "miss", "writedown" need financial calibration to be scored correctly.

Aggregation: mean score across all headlines in a given stock-month. Stocks
with no headlines that month receive Sentiment = NaN and Headline_Count = 0.
NaN is handled natively by the tree models in run_sentiment_experiment.py
(which imputes 0 = neutral before training).

Input
-----
    data/01_raw/news_headlines_raw.csv          (from fetch_news.py)
    data/02_preprocessed/MASTER_DF_PROD_JM2_nonnan_winsor.csv

Output
------
    data/02_preprocessed/MASTER_DF_PROD_JM2_SENTIMENT.csv
    Same as the master panel with two extra columns:
        Sentiment      — mean FinBERT score for that stock-month (NaN if no news)
        Headline_Count — number of headlines scored (0 if no news)

Next step
---------
    Once this script finishes, run:
        python run_sentiment_experiment.py
    That script runs three model variants (A: no sentiment, B: news dummy,
    C: full sentiment) over the 13-month OOS window and produces a comparison
    table and performance regression to test whether sentiment adds value.

Usage:
    python src/sentiment.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# ProsusAI/finbert is a BERT model fine-tuned on financial news text.
# It outputs three class probabilities: positive, negative, neutral.
# We use it because general-purpose sentiment models (e.g. VADER) are not
# calibrated for financial language where words like "beat" or "miss" carry
# specific directional meaning.
MODEL_NAME   = "ProsusAI/finbert"

HEADLINES_PATH = Path("data/01_raw/news_headlines_raw.csv")
MASTER_PATH    = Path("data/02_preprocessed/MASTER_DF_PROD_JM2_nonnan_winsor.csv")
OUTPUT_PATH    = Path("data/02_preprocessed/MASTER_DF_PROD_JM2_SENTIMENT.csv")

# Module-level globals so the model is loaded once and reused across calls.
# Loading FinBERT takes ~5 seconds; we do not want to repeat that on every
# function call.
_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSequenceClassification | None = None
_device: torch.device | None = None
_pos_idx: int | None = None
_neg_idx: int | None = None


def _load_model() -> None:
    """Load FinBERT into memory (once). Subsequent calls are no-ops."""
    global _tokenizer, _model, _device, _pos_idx, _neg_idx
    if _model is not None:
        return
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    _model.eval()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device)

    # Map label names to their indices dynamically.
    # We do NOT hard-code indices (e.g. 0=positive) because the ordering in
    # id2label can vary between model checkpoints and would silently break
    # the scoring if it ever changes.
    label_map = _model.config.id2label
    _pos_idx = next(k for k, v in label_map.items() if v == "positive")
    _neg_idx = next(k for k, v in label_map.items() if v == "negative")


# ---------------------------------------------------------------------------
# Phase 1 — scoring
# ---------------------------------------------------------------------------

def score_headline(text: str) -> float:
    """
    Score a single headline with FinBERT.

    Returns a composite score in [-1, +1]:
        score = P(positive) - P(negative)

    +1 means maximally positive sentiment (e.g. record earnings beat).
    -1 means maximally negative sentiment (e.g. fraud investigation).
     0 means neutral or balanced (e.g. routine management appointment).
    """
    _load_model()
    inputs = _tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(_device)
    with torch.no_grad():
        outputs = _model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    return float(probs[_pos_idx] - probs[_neg_idx])


def score_headlines_batch(
    texts: list[str], batch_size: int = 32
) -> np.ndarray:
    """
    Score a list of headlines efficiently using batched inference.

    Batching is ~10-30x faster than scoring one headline at a time because
    the GPU (or CPU SIMD) processes multiple sequences in parallel.

    Returns a 1-D numpy array of composite scores, one per input headline,
    in the same order as the input list.
    """
    _load_model()
    all_scores: list[float] = []
    batches = range(0, len(texts), batch_size)
    for i in tqdm(batches, desc="Scoring headlines", unit="batch"):
        batch = texts[i : i + batch_size]
        # padding=True pads shorter sequences to the longest in the batch.
        # truncation=True cuts sequences longer than max_length=512 tokens.
        inputs = _tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(_device)
        with torch.no_grad():
            outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        composites = probs[:, _pos_idx] - probs[:, _neg_idx]
        all_scores.extend(composites.tolist())
    return np.array(all_scores)


# ---------------------------------------------------------------------------
# Phase 2 — aggregation to stock-month level
# ---------------------------------------------------------------------------

def aggregate_to_stock_month(
    df: pd.DataFrame,
    ric_col: str = "Instrument",
    date_col: str = "date",
    text_col: str = "headline",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Score all headlines and aggregate to one value per (Instrument, month-end).

    The aggregation logic:
        1. Strip timezone info from dates (Refinitiv returns tz-aware timestamps;
           the master panel uses naive dates — they must match for the merge).
        2. Snap each headline date to its month-end (e.g. March 15 → March 31).
        3. Drop trivially short headlines (len <= 5) that would produce noise.
        4. Score all remaining headlines with FinBERT in one batched pass.
        5. Group by (Instrument, month-end) and take the MEAN score.

    Mean is used rather than median or sum because it is the standard
    aggregation in the news sentiment literature (e.g. Tetlock 2007,
    Garcia 2013) and is consistent with the equal-weighted treatment of
    news flow in our pipeline.

    Returns a DataFrame with columns:
        Instrument      — stock RIC
        Date            — month-end date
        Sentiment       — mean FinBERT score across all headlines that month
        Headline_Count  — number of headlines scored (useful as a signal for
                          news attention / media coverage intensity)
    """
    df = df.copy()

    # Defensive timezone handling: strip tz if present, leave naive as-is.
    df[date_col] = pd.to_datetime(df[date_col])
    if df[date_col].dt.tz is not None:
        df[date_col] = df[date_col].dt.tz_localize(None)

    # Snap to month-end so headlines align with the master panel's Date column.
    df["_month_end"] = df[date_col] + pd.offsets.MonthEnd(0)

    # Drop blank or near-blank headlines before scoring to avoid wasting
    # compute on garbage strings that Refinitiv occasionally returns.
    df = df[df[text_col].str.len() > 5].copy()

    scores = score_headlines_batch(df[text_col].tolist(), batch_size=batch_size)
    df["_score"] = scores

    agg = (
        df.groupby([ric_col, "_month_end"])["_score"]
        .agg(Sentiment="mean", Headline_Count="count")
        .reset_index()
        .rename(columns={ric_col: "Instrument", "_month_end": "Date"})
    )
    return agg


def build_sentiment_feature(
    headlines_df: pd.DataFrame,
    master_df: pd.DataFrame,
    ric_col: str = "Instrument",
    date_col: str = "date",
    text_col: str = "headline",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Build the Sentiment feature column and attach it to the master panel.

    Calls aggregate_to_stock_month, then left-joins the result onto the
    master panel on (Instrument, Date).

    Left-join preserves every row in the master panel:
        - Stock-months WITH headlines → Sentiment = mean FinBERT score
        - Stock-months WITHOUT headlines → Sentiment = NaN

    NaN handling downstream:
        Tree models (LightGBM, XGBoost) handle NaN natively — they learn
        a separate split direction for missing values, so no imputation is
        needed. This is the intended behaviour: NaN means "no news this month",
        which is itself informative.

    The pre-2025 period (the vast majority of the panel) will be all NaN
    because we only have headlines from March 2025 onwards

    Returns master_df with two additional columns: Sentiment, Headline_Count.
    """
    sentiment = aggregate_to_stock_month(
        headlines_df,
        ric_col=ric_col,
        date_col=date_col,
        text_col=text_col,
        batch_size=batch_size,
    )

    master_df = master_df.copy()
    master_df["Date"] = pd.to_datetime(master_df["Date"])

    merged = master_df.merge(
        sentiment[["Instrument", "Date", "Sentiment", "Headline_Count"]],
        on=["Instrument", "Date"],
        how="left",
    )

    # Stocks with no headlines that month logically have 0 articles, not NaN.
    merged["Headline_Count"] = merged["Headline_Count"].fillna(0).astype(int)

    # Diagnostics — verify the merge worked as expected before saving.
    coverage = merged["Sentiment"].notna().mean()
    print(f"Sentiment coverage : {coverage:.1%} of stock-months have at least one headline")
    print(f"NaN stock-months   : {merged['Sentiment'].isna().sum():,}")
    sentiment_dates = merged.loc[merged["Sentiment"].notna(), "Date"]
    if len(sentiment_dates) > 0:
        print(f"Sentiment dates    : {sentiment_dates.min().date()} → {sentiment_dates.max().date()}")

    return merged


# ---------------------------------------------------------------------------
# Production entry point
# ---------------------------------------------------------------------------

def run_production(
    headlines_path: Path = HEADLINES_PATH,
    master_path: Path = MASTER_PATH,
    output_path: Path = OUTPUT_PATH,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Full pipeline: load raw headlines + master panel, score, aggregate, save.

    This is the function to call once fetch_news.py has finished pulling
    headlines. The output CSV is a drop-in replacement for MASTER_DF_1.csv
    with two extra columns (Sentiment, Headline_Count) and is used as input
    to Pipeline B (best model + sentiment feature).
    """
    print(f"Loading headlines from {headlines_path} …")
    headlines_df = pd.read_csv(headlines_path, parse_dates=["date"])
    print(f"  {len(headlines_df):,} headlines for {headlines_df['Instrument'].nunique()} stocks")
    print(f"  Date range: {headlines_df['date'].min().date()} → {headlines_df['date'].max().date()}")

    print(f"\nLoading master panel from {master_path} …")
    master_df = pd.read_csv(master_path, parse_dates=["Date"])
    print(f"  {len(master_df):,} stock-month rows, {master_df['Instrument'].nunique()} stocks")

    print(f"\nDevice: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Loading {MODEL_NAME} …\n")

    result = build_sentiment_feature(headlines_df, master_df, batch_size=batch_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\nSaved augmented master to {output_path}")
    print(f"Columns: {result.columns.tolist()}")

    return result


if __name__ == "__main__":
    run_production()
