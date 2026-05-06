"""
models.py — Walk-forward model training and OOS evaluation.

Models
------
Linear : OLS, LASSO, Ridge, ElasticNet
Tree   : Random Forest, LightGBM, XGBoost
Neural : Feedforward NN (placeholder — architecture TBD)

Training protocol (two phases)
-------------------------------
Phase 1 — hyperparameter selection (runs ONCE before the OOS loop):
    Fixed train window  [TRAIN_START, TRAIN_END]     -> fit candidates
    Fixed val window    [VAL_START,   VAL_END]        -> select best params
    Hyperparameters are locked after Phase 1 and never re-searched in Phase 2.

Phase 2 — expanding-window OOS prediction (monthly retraining):
    For each month t in [OOS_START, OOS_END]:
        - Retrain on ALL data from TRAIN_START up to month t-1  (expanding)
        - Use Phase-1 hyperparameters (no re-tuning -> no future information)
        - Predict the full stock cross-section at month t

Target
------
Next-month excess return: features known at month-end t predict the return
realised at t+1.  Call add_forward_return(df) once on the full panel before
passing it to run_walk_forward().
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

from config import TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END

# ======================================================================
# Feature column lists
# ======================================================================

# All 18 features fed directly into the models (no composite signal aggregation).
FEATURE_COLS = [
    # Valuation
    "E/P_ff", "1/P/B", "-P/S", "-P/CF_ff", "DivYield_12M",
    # Quality
    "OperatingProfitability", "BookToMarket", "-Debt/MktCap",
    # Momentum
    "MOM_1M", "MOM_12M", "RSI_30d",
    "Stock_vs_Sector_12M_1M", "Stock_vs_Sector_1M",
    # Risk
    "-Vol_30d", "-Beta_252d", "-IdioVol",
    # Standalone size factor
    "log_MktCap",
]

# Appended automatically if present in the dataframe (produced by hurst.py / HMM.py).
OPTIONAL_FEATURE_COLS = ["Hurst", "HMM_Regime"]

TARGET_COL = "Excess_Return"
TRAIN_START = "2010-01-01"


# ======================================================================
# Data preparation
# ======================================================================

def add_forward_return(df, target_col=TARGET_COL):
    """
    Append a 'target' column = next-month excess return per stock.

    Features at month-end t predict the return realised at t+1.
    shift(-1) per Instrument moves the return one row earlier so that
    each feature row carries the *future* return it should predict.
    No look-ahead bias: the target value is only revealed after t.

    Call this once on the full panel before any train/val/test split.
    """
    df = df.sort_values(["Instrument", "Date"]).copy()
    df["target"] = df.groupby("Instrument")[target_col].shift(-1)
    return df


def _resolve_features(df, feature_cols):
    """Return only those cols that exist in df; warn about the rest."""
    present = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(present)
    if missing:
        warnings.warn(f"Feature columns missing from df (skipped): {sorted(missing)}")
    return present


def prepare_Xy(df, feature_cols, target_col="target"):
    """
    Extract (X, y, instruments, dates) from a panel slice.

    Rows where any feature or the target is NaN are dropped — sklearn,
    LightGBM, and XGBoost all require complete cases.

    Returns
    -------
    X           : np.ndarray  (n_rows, n_features)
    y           : np.ndarray  (n_rows,)
    instruments : pd.Series   stock identifiers aligned with X/y rows
    dates       : pd.Series   month-end dates aligned with X/y rows
    """
    cols = _resolve_features(df, feature_cols)
    needed = cols + [target_col, "Instrument", "Date"]
    subset = df[needed].dropna(subset=cols + [target_col])
    X = subset[cols].to_numpy(dtype=float)
    y = subset[target_col].to_numpy(dtype=float)
    return X, y, subset["Instrument"].reset_index(drop=True), subset["Date"].reset_index(drop=True)


def prepare_X(df, feature_cols):
    """
    Extract X for prediction only (no target column required).

    Stocks with any NaN feature are excluded from the prediction for
    that month — they will be absent from the output DataFrame.
    """
    cols = _resolve_features(df, feature_cols)
    subset = df[cols + ["Instrument", "Date"]].dropna(subset=cols)
    X = subset[cols].to_numpy(dtype=float)
    return X, subset["Instrument"].reset_index(drop=True), subset["Date"].reset_index(drop=True)


# ======================================================================
# Shared helpers
# ======================================================================

def _mse(y_true, y_pred):
    """Mean squared error — used as the val-set selection criterion."""
    return float(np.mean((y_true - y_pred) ** 2))


def _fit_scaler(X_train):
    """Fit a StandardScaler on X_train only — never touches val or test data."""
    return StandardScaler().fit(X_train)


# ======================================================================
# OLS  (no tuning — serves as the linear baseline)
# ======================================================================

def tune_ols(*_):
    """OLS has no hyperparameters."""
    return {}


def fit_ols(X_train, y_train, params):
    """
    Ordinary least squares with StandardScaler.

    Scaling ensures all coefficients are on the same unit so they can be
    meaningfully compared across features; it does not affect predictions.
    Returns (scaler, model).
    """
    scaler = _fit_scaler(X_train)
    model = LinearRegression()
    model.fit(scaler.transform(X_train), y_train)
    return scaler, model


# ======================================================================
# LASSO  (L1 — sparsity-inducing regularisation)
# ======================================================================

def tune_lasso(X_train, y_train, X_val, y_val):
    """
    Search alpha (L1 penalty strength) on the val set.
    Larger alpha -> more coefficients pushed to exactly zero.
    """
    alphas = [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    scaler = _fit_scaler(X_train)
    Xs, Xv = scaler.transform(X_train), scaler.transform(X_val)
    best_alpha, best_mse = None, np.inf
    for alpha in alphas:
        m = Lasso(alpha=alpha, max_iter=10_000)
        m.fit(Xs, y_train)
        mse = _mse(y_val, m.predict(Xv))
        if mse < best_mse:
            best_mse, best_alpha = mse, alpha
    return {"alpha": best_alpha}


def fit_lasso(X_train, y_train, params):
    scaler = _fit_scaler(X_train)
    model = Lasso(max_iter=10_000, **params)
    model.fit(scaler.transform(X_train), y_train)
    return scaler, model


# ======================================================================
# Ridge  (L2 — shrinkage without zeroing coefficients)
# ======================================================================

def tune_ridge(X_train, y_train, X_val, y_val):
    """Search alpha on the val set."""
    alphas = [1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]
    scaler = _fit_scaler(X_train)
    Xs, Xv = scaler.transform(X_train), scaler.transform(X_val)
    best_alpha, best_mse = None, np.inf
    for alpha in alphas:
        m = Ridge(alpha=alpha)
        m.fit(Xs, y_train)
        mse = _mse(y_val, m.predict(Xv))
        if mse < best_mse:
            best_mse, best_alpha = mse, alpha
    return {"alpha": best_alpha}


def fit_ridge(X_train, y_train, params):
    scaler = _fit_scaler(X_train)
    model = Ridge(**params)
    model.fit(scaler.transform(X_train), y_train)
    return scaler, model


# ======================================================================
# ElasticNet  (L1 + L2 blend)
# ======================================================================

def tune_elastic_net(X_train, y_train, X_val, y_val):
    """
    Joint search over alpha and l1_ratio on the val set.
    l1_ratio=1 -> pure LASSO; l1_ratio=0 -> pure Ridge.
    """
    alphas    = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0]
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    scaler = _fit_scaler(X_train)
    Xs, Xv = scaler.transform(X_train), scaler.transform(X_val)
    best, best_mse = {}, np.inf
    for alpha in alphas:
        for l1 in l1_ratios:
            m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=10_000)
            m.fit(Xs, y_train)
            mse = _mse(y_val, m.predict(Xv))
            if mse < best_mse:
                best_mse = mse
                best = {"alpha": alpha, "l1_ratio": l1}
    return best


def fit_elastic_net(X_train, y_train, params):
    scaler = _fit_scaler(X_train)
    model = ElasticNet(max_iter=10_000, **params)
    model.fit(scaler.transform(X_train), y_train)
    return scaler, model


# ======================================================================
# Random Forest
# ======================================================================

def tune_rf(X_train, y_train, X_val, y_val):
    """
    Tune max_depth and min_samples_leaf on val MSE.

    n_estimators is fixed at 300 — enough for prediction variance to
    stabilise without dominating runtime. No feature scaling needed:
    tree splits are invariant to monotone transformations.
    """
    grid = [
        {"max_depth": d, "min_samples_leaf": leaf}
        for d    in [3, 5, 8, None]   # None = fully grown trees
        for leaf in [20, 50, 100]
    ]
    best, best_mse = {}, np.inf
    for params in grid:
        m = RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=42, **params
        )
        m.fit(X_train, y_train)
        mse = _mse(y_val, m.predict(X_val))
        if mse < best_mse:
            best_mse, best = mse, params
    return best


def fit_rf(X_train, y_train, params):
    model = RandomForestRegressor(
        n_estimators=300, n_jobs=-1, random_state=42, **params
    )
    model.fit(X_train, y_train)
    # Tree models need no scaler; return None to match the (scaler, model) interface.
    return None, model


# ======================================================================
# LightGBM
# ======================================================================

def tune_lgbm(X_train, y_train, X_val, y_val):
    """
    Tune learning_rate, max_depth, and num_leaves on the val set.

    Early stopping determines the optimal number of trees for each
    config. The winning n_estimators is stored in the returned params
    so Phase 2 trains a fixed tree count (no early stopping needed).
    """
    grid = [
        {"learning_rate": lr, "max_depth": d, "num_leaves": nl}
        for lr in [0.01, 0.05]
        for d  in [3, 5]
        for nl in [15, 31]
    ]
    best, best_mse = {}, np.inf
    for params in grid:
        m = lgb.LGBMRegressor(n_estimators=1000, random_state=42, verbose=-1, **params)
        m.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)],
        )
        mse = _mse(y_val, m.predict(X_val))
        if mse < best_mse:
            best_mse = mse
            # Lock in the tree count found by early stopping for Phase 2
            best = {**params, "n_estimators": m.best_iteration_}
    return best


def fit_lgbm(X_train, y_train, params):
    """Refit with fixed hyperparameters — no early stopping in Phase 2."""
    model = lgb.LGBMRegressor(random_state=42, verbose=-1, **params)
    model.fit(X_train, y_train)
    return None, model


# ======================================================================
# XGBoost
# ======================================================================

def tune_xgb(X_train, y_train, X_val, y_val):
    """
    Tune learning_rate, max_depth, and subsample on the val set.
    Same early-stopping logic as LightGBM — best_iteration stored for Phase 2.
    """
    grid = [
        {"learning_rate": lr, "max_depth": d, "subsample": sub}
        for lr  in [0.01, 0.05]
        for d   in [3, 5]
        for sub in [0.7, 1.0]
    ]
    best, best_mse = {}, np.inf
    for params in grid:
        m = xgb.XGBRegressor(
            n_estimators=1000,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=40,
            **params,
        )
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        mse = _mse(y_val, m.predict(X_val))
        if mse < best_mse:
            best_mse = mse
            best = {**params, "n_estimators": m.best_iteration}
    return best


def fit_xgb(X_train, y_train, params):
    """Refit with fixed hyperparameters — no early stopping in Phase 2."""
    model = xgb.XGBRegressor(random_state=42, verbosity=0, **params)
    model.fit(X_train, y_train)
    return None, model


# ======================================================================
# Neural Network (placeholder)
# ======================================================================

def tune_nn(*_):
    raise NotImplementedError(
        "Define the NN architecture in fit_nn() before running. "
        "Suggested starting point: 3 hidden layers [256, 128, 64], "
        "ReLU activations, dropout 0.3, Adam optimiser."
    )


def fit_nn(X_train, y_train, params):
    raise NotImplementedError("NN architecture not defined yet — fill in fit_nn().")


# ======================================================================
# Model registry
# Maps model name -> (tune_fn, fit_fn)
# ======================================================================

MODEL_REGISTRY = {
    "OLS":        (tune_ols,         fit_ols),
    "LASSO":      (tune_lasso,       fit_lasso),
    "Ridge":      (tune_ridge,       fit_ridge),
    "ElasticNet": (tune_elastic_net, fit_elastic_net),
    "RF":         (tune_rf,          fit_rf),
    "LGBM":       (tune_lgbm,        fit_lgbm),
    "XGB":        (tune_xgb,         fit_xgb),
    "NN":         (tune_nn,          fit_nn),
}

# Default run excludes NN until it is implemented
DEFAULT_MODELS = ["OLS", "LASSO", "Ridge", "ElasticNet", "RF", "LGBM", "XGB"]


# ======================================================================
# Walk-forward engine
# ======================================================================

def run_walk_forward(
    df,
    feature_cols=None,
    models=None,
    train_start=TRAIN_START,
    train_end=TRAIN_END,
    val_start=VAL_START,
    val_end=VAL_END,
    oos_start=OOS_START,
    oos_end=OOS_END,
    verbose=True,
):
    """
    Two-phase walk-forward prediction loop.

    Phase 1  (hyperparameter selection — runs once)
    ------------------------------------------------
    Train: [train_start, train_end]
    Val  : [val_start,   val_end]
    Each model's tune_* function searches its grid and returns best_params.
    Hyperparameters are locked after this phase.

    Phase 2  (OOS prediction — monthly retraining)
    -----------------------------------------------
    For each month t in [oos_start, oos_end]:
      1. Expanding train set: all data from train_start up to t-1.
      2. Refit each model with Phase-1 hyperparameters (no re-tuning).
      3. Predict the full cross-section at t.
      4. Collect (Date, Instrument, y_true, y_pred_*) for that month.

    Parameters
    ----------
    df           : Monthly panel. Must contain 'target' column; if absent,
                   add_forward_return(df) is called automatically.
    feature_cols : Features to use. Defaults to FEATURE_COLS plus any
                   OPTIONAL_FEATURE_COLS that exist in df.
    models       : List of model names, "default" (all except NN), or "all".
    verbose      : Print Phase-1 tuning results and Phase-2 monthly progress.

    Returns
    -------
    DataFrame with columns:
        Date, Instrument, y_true, y_pred_OLS, y_pred_LASSO, ..., y_pred_XGB
    One row per (stock, OOS month) where predictions are available.
    """
    # ── setup ──────────────────────────────────────────────────────────
    if "target" not in df.columns:
        df = add_forward_return(df)

    if feature_cols is None:
        extras = [c for c in OPTIONAL_FEATURE_COLS if c in df.columns]
        feature_cols = FEATURE_COLS + extras

    if models is None or models == "default":
        models = DEFAULT_MODELS
    elif models == "all":
        models = list(MODEL_REGISTRY.keys())

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # ── Phase 1: hyperparameter selection ─────────────────────────────
    hp_train = (df["Date"] >= train_start) & (df["Date"] <= train_end)
    hp_val   = (df["Date"] >= val_start)   & (df["Date"] <= val_end)

    X_tr, y_tr, _, _ = prepare_Xy(df[hp_train], feature_cols)
    X_vl, y_vl, _, _ = prepare_Xy(df[hp_val],   feature_cols)

    if verbose:
        print("=" * 60)
        print("Phase 1: hyperparameter selection")
        print(f"  Train : {train_start} -> {train_end}  ({len(y_tr):,} obs)")
        print(f"  Val   : {val_start} -> {val_end}  ({len(y_vl):,} obs)")
        print("=" * 60)

    best_params = {}
    active_models = []

    for name in models:
        tune_fn, _ = MODEL_REGISTRY[name]
        try:
            if verbose:
                print(f"  Tuning {name} ...", end="  ")
            params = tune_fn(X_tr, y_tr, X_vl, y_vl)
            best_params[name] = params
            active_models.append(name)
            if verbose:
                print(f"best params: {params}")
        except NotImplementedError as exc:
            print(f"\n  Skipping {name}: {exc}")

    # ── Phase 2: expanding OOS loop ────────────────────────────────────
    oos_periods = (
        df.loc[(df["Date"] >= oos_start) & (df["Date"] <= oos_end), "Date"]
        .dt.to_period("M")
        .unique()
        .sort_values()
    )

    if verbose:
        print(f"\nPhase 2: OOS prediction")
        print(f"  Period : {oos_start} -> {oos_end}  ({len(oos_periods)} months)")
        print(f"  Models : {active_models}")
        print("=" * 60)

    records = []
    for period in oos_periods:
        # month-end timestamp for this OOS month
        t = period.to_timestamp("M")

        # Expanding train: everything strictly before month t
        exp_train = (df["Date"] >= train_start) & (df["Date"] < t)
        is_test   = df["Date"].dt.to_period("M") == period

        if exp_train.sum() == 0 or is_test.sum() == 0:
            continue

        X_exp, y_exp, _, _    = prepare_Xy(df[exp_train], feature_cols)
        X_test, inst_test, _  = prepare_X(df[is_test],    feature_cols)

        if verbose:
            print(f"  {period}  |  train: {len(y_exp):,} obs  |  test: {len(inst_test)} stocks")

        # True forward returns for this month (used for evaluation later)
        y_true_map = (
            df.loc[is_test, ["Instrument", "target"]]
            .dropna(subset=["target"])
            .set_index("Instrument")["target"]
        )

        preds = {}
        for name in active_models:
            _, fit_fn = MODEL_REGISTRY[name]
            try:
                scaler, model = fit_fn(X_exp, y_exp, best_params[name])
                # Tree models return scaler=None; linear/NN models return a fitted scaler
                X_in = scaler.transform(X_test) if scaler is not None else X_test
                preds[f"y_pred_{name}"] = model.predict(X_in)
            except Exception as exc:
                warnings.warn(f"{name} failed at {period}: {exc}")
                preds[f"y_pred_{name}"] = np.full(len(inst_test), np.nan)

        month_df = pd.DataFrame(
            {"Date": t, "Instrument": inst_test.values, **preds}
        )
        month_df["y_true"] = month_df["Instrument"].map(y_true_map)
        records.append(month_df)

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


# ======================================================================
# Evaluation
# ======================================================================

def r2_oos(y_true, y_pred, y_benchmark):
    """
    Out-of-sample R2  (Campbell & Thompson 2008; Gu et al. 2020).

    R2_OOS = 1 - sum(y_t - y_hat_t)^2 / sum(y_t - y_bar_{t-1})^2

    The denominator uses the expanding historical mean y_bar_{t-1} as the
    naive benchmark — i.e., "predict last period's average return".
    A positive R2_OOS means the model beats this benchmark in MSE terms.

    Parameters
    ----------
    y_true      : Realised excess returns.
    y_pred      : Model predictions.
    y_benchmark : Expanding historical mean of y_true, precomputed before
                  each observation (no look-ahead into the OOS window).
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_benchmark) ** 2)
    return float("nan") if ss_tot == 0 else float(1.0 - ss_res / ss_tot)


def evaluate_all(predictions_df):
    """
    Compute OOS metrics for every y_pred_* column in predictions_df.

    Benchmark: expanding mean of pooled y_true, sorted by Date (Gu et al. 2020).
    This is the "predict the historical mean" naive strategy.

    Returns a DataFrame (one row per model) with: model, R2_OOS, MSE, MAE.
    Sorted descending by R2_OOS so the best model appears first.
    """
    pred_cols = [c for c in predictions_df.columns if c.startswith("y_pred_")]
    if not pred_cols:
        raise ValueError("No y_pred_* columns found in predictions_df.")

    df = predictions_df.dropna(subset=["y_true"]).sort_values("Date").copy()
    # Expanding historical mean — the benchmark for each row in the OOS window
    df["_benchmark"] = df["y_true"].expanding().mean()

    rows = []
    for col in pred_cols:
        valid = df.dropna(subset=[col])
        if valid.empty:
            continue
        r2  = r2_oos(
            valid["y_true"].to_numpy(),
            valid[col].to_numpy(),
            valid["_benchmark"].to_numpy(),
        )
        mse = float(np.mean((valid["y_true"] - valid[col]) ** 2))
        mae = float(np.mean(np.abs(valid["y_true"] - valid[col])))
        rows.append({
            "model":  col.replace("y_pred_", ""),
            "R2_OOS": round(r2,  4),
            "MSE":    round(mse, 6),
            "MAE":    round(mae, 4),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("R2_OOS", ascending=False)
        .reset_index(drop=True)
    )


# ======================================================================
# Standalone entry point
# ======================================================================

MASTER_PATH     = Path("data/02_preprocessed/master_features_monthly.csv")
PREDICTIONS_OUT = Path("results/predictions_oos.csv")
METRICS_OUT     = Path("results/evaluation_metrics.csv")


def main(models=None, verbose=True):
    """
    Load master panel -> run walk-forward -> save predictions and metrics.

    Parameters
    ----------
    models  : List of model names, "default", or "all". Defaults to DEFAULT_MODELS.
    verbose : Print Phase-1 tuning results and Phase-2 monthly progress.
    """
    print(f"Loading master panel from {MASTER_PATH} ...")
    df = pd.read_csv(MASTER_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    feature_cols = FEATURE_COLS + [c for c in OPTIONAL_FEATURE_COLS if c in df.columns]

    df = add_forward_return(df)

    predictions_df = run_walk_forward(
        df,
        feature_cols=feature_cols,
        models=models,
        verbose=verbose,
    )

    if predictions_df.empty:
        print("No predictions generated — check date ranges and feature coverage.")
        return

    PREDICTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(PREDICTIONS_OUT, index=False)
    print(f"\nPredictions saved : {PREDICTIONS_OUT}  ({len(predictions_df):,} rows)")

    metrics = evaluate_all(predictions_df)
    metrics.to_csv(METRICS_OUT, index=False)
    print(f"Metrics saved     : {METRICS_OUT}")
    print("\n" + metrics.to_string(index=False))


# Tähän lisätään mitä eri malleja syöttää ulos
if __name__ == "__main__":
    main(models=["OLS"])
