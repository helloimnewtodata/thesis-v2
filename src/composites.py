"""
Komposiittisignaalien rakentaminen: cross-sectional Z-score + aggregointi
(Z-score averaging tai PCA ryhmäkohtaisesti).
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Ryhmädefinitiot: feature-sarakkeen nimi → ryhmä
VALUATION_FEATURES = ["E/P", "1/P/B", "-P/S", "-P/CF", "DivYield_12M"]
QUALITY_FEATURES = ["Return On Average Common Equity %", "Gross Profit / Total Assets", "-Debt/MktCap"]
MOMENTUM_FEATURES = ["MOM_1M", "MOM_12M", "RSI_30d", "Hurst", "Stock_vs_Sector"]
RISK_FEATURES = ["-Vol_30d", "-Beta_252d", "-IdioVol"]
MARKET_FEATURES = ["Z_Calculated PE Ratio", "Z_Calculated Price to Book", "Z_Calculated Index Dividend Yield"]


def cross_sectional_zscore(df, features, date_col="Date"):
    """
    Cross-sectional Z-score: jokaiselle päivälle erikseen
    (keskiarvo=0, keskihajonta=1 poikkileikkauksessa).
    """
    for feat in features:
        if feat not in df.columns:
            continue
        df[f"z_{feat}"] = df.groupby(date_col)[feat].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    return df


def winsorise(df, features, limits=(0.01, 0.99), date_col="Date"):
    """Winsorisoi featuret per poikkileikkaus (1%/99%)."""
    for feat in features:
        col = f"z_{feat}"
        if col not in df.columns:
            continue
        df[col] = df.groupby(date_col)[col].transform(
            lambda x: x.clip(x.quantile(limits[0]), x.quantile(limits[1]))
        )
    return df


def aggregate_zscore_avg(df, features, signal_name):
    """Z-score averaging: komposiittisignaali = z-scorien keskiarvo."""
    z_cols = [f"z_{f}" for f in features if f"z_{f}" in df.columns]
    df[signal_name] = df[z_cols].mean(axis=1)
    return df


def aggregate_pca(df, features, signal_name, date_col="Date", train_end="2016-12-31"):
    """
    PCA-aggregointi: sovitetaan vain train-dataan, transformoidaan koko aineisto.
    CLAUDE.md: sign alignment → winsorisation → standardisation → fit train → transform test.
    """
    z_cols = [f"z_{f}" for f in features if f"z_{f}" in df.columns]

    train_mask = pd.to_datetime(df[date_col]) <= pd.to_datetime(train_end)
    train_data = df.loc[train_mask, z_cols].dropna()

    scaler = StandardScaler()
    scaler.fit(train_data)

    pca = PCA(n_components=1)
    pca.fit(scaler.transform(train_data))

    # Transformoi koko aineisto
    valid_mask = df[z_cols].notna().all(axis=1)
    df[signal_name] = np.nan
    if valid_mask.any():
        scaled = scaler.transform(df.loc[valid_mask, z_cols])
        df.loc[valid_mask, signal_name] = pca.transform(scaled).flatten()

    return df


def build_composite_signals(df, method_per_group=None, train_end="2016-12-31"):
    """
    Rakentaa 5 komposiittisignaalia. method_per_group on dict:
        {"Valuation": "zscore", "Quality": "pca", ...}
    Oletus: zscore averaging kaikille.
    """
    if method_per_group is None:
        method_per_group = {
            "Valuation": "zscore",
            "Quality": "zscore",
            "Momentum": "zscore",
            "Risk": "zscore",
            "Market": "zscore",
        }

    groups = {
        "Valuation": VALUATION_FEATURES,
        "Quality": QUALITY_FEATURES,
        "Momentum": MOMENTUM_FEATURES,
        "Risk": RISK_FEATURES,
    }

    # Cross-sectional Z-score ja winsorisaatio kaikille osakefeatureille
    all_features = VALUATION_FEATURES + QUALITY_FEATURES + MOMENTUM_FEATURES + RISK_FEATURES
    df = cross_sectional_zscore(df, all_features)
    df = winsorise(df, all_features)

    # Aggregoi per ryhmä
    for group_name, features in groups.items():
        method = method_per_group.get(group_name, "zscore")
        signal_name = f"Signal_{group_name}"

        if method == "pca":
            df = aggregate_pca(df, features, signal_name, train_end=train_end)
        else:
            df = aggregate_zscore_avg(df, features, signal_name)

    # Market-signaali tulee indeksitasolta (ei cross-sectional),
    # se liitetään erikseen main.py:ssä

    return df
