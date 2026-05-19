"""
Walk-forward XGBoost hyperparameter optimization for return forecasting.

The model is still trained as a next-month excess-return forecaster. The
validation setup is time-series aware: each validation year is scored using
only training observations dated before that validation period.

Primary selection criterion:
    weighted validation MSE

Secondary diagnostics:
    OOS R2-style validation score, monthly Spearman IC, decile spread,
    prediction dispersion and unique prediction counts.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import ParameterSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = PROJECT_ROOT / "data" / "02_preprocessed" / "MASTER_DF_PROD_JM2_nonnan_winsor.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_HPO_OUT = RESULTS_DIR / "xgb_walkforward_hpo_results.csv"
DEFAULT_BEST_PARAMS_OUT = RESULTS_DIR / "xgb_walkforward_hpo_best_params.json"
DEFAULT_PRED_OUT = RESULTS_DIR / "predictions_oos_prod_jm2_winsor_xgb_hpo.csv"
DEFAULT_METRICS_OUT = RESULTS_DIR / "evaluation_metrics_prod_jm2_winsor_xgb_hpo.csv"
DEFAULT_RANKING_OUT = RESULTS_DIR / "stock_rankings_oos_prod_jm2_winsor_xgb_hpo.csv"
DEFAULT_LATEST_RANKING_OUT = RESULTS_DIR / "stock_ranking_latest_prod_jm2_winsor_xgb_hpo.csv"
DEFAULT_PORTFOLIO_RETURNS_OUT = RESULTS_DIR / "portfolio_returns_prod_jm2_winsor_xgb_hpo.csv"
DEFAULT_PORTFOLIO_PERFORMANCE_OUT = RESULTS_DIR / "portfolio_performance_prod_jm2_winsor_xgb_hpo.csv"


PARAM_GRID = {
    "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
    "max_depth": [2, 3, 4, 5, 6],
    "min_child_weight": [1, 3, 5, 10, 20],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
    "reg_lambda": [0.1, 1, 5, 10, 25],
    "reg_alpha": [0, 0.001, 0.01, 0.1, 1],
    # Return targets are small decimals, so large gamma values can prevent
    # almost all splits and collapse the model to a near-constant predictor.
    "gamma": [0, 0.0001, 0.001, 0.005, 0.01],
}

ANCHOR_PARAMS = [
    {
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 1,
        "reg_alpha": 0,
        "gamma": 0,
    },
    {
        "learning_rate": 0.02,
        "max_depth": 5,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 1,
        "reg_alpha": 0,
        "gamma": 0,
    },
    {
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 1,
        "reg_alpha": 0,
        "gamma": 0,
    },
    {
        "learning_rate": 0.01,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1,
        "reg_alpha": 0,
        "gamma": 0,
    },
    {
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_lambda": 0.1,
        "reg_alpha": 0,
        "gamma": 0,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default=str(DEFAULT_MASTER))
    parser.add_argument("--n-iter", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-estimators", type=int, default=2000)
    parser.add_argument(
        "--min-estimators",
        type=int,
        default=100,
        help="Minimum tree count used when early stopping collapses to a trivial mean predictor.",
    )
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--cv-val-years", default="2014,2015,2016,2017,2018")
    parser.add_argument("--oos-start", default="2019-01-01")
    parser.add_argument("--oos-end", default="2025-12-31")
    parser.add_argument("--mse-tie-tolerance", type=float, default=0.005)
    parser.add_argument(
        "--min-median-unique-predictions",
        type=float,
        default=50,
        help="Reject validation candidates with too few unique monthly predictions unless all candidates fail.",
    )
    parser.add_argument(
        "--min-mean-pred-std",
        type=float,
        default=1e-5,
        help="Reject near-constant validation candidates unless all candidates fail.",
    )
    parser.add_argument("--hpo-output", default=str(DEFAULT_HPO_OUT))
    parser.add_argument("--best-params-output", default=str(DEFAULT_BEST_PARAMS_OUT))
    parser.add_argument("--predictions-output", default=str(DEFAULT_PRED_OUT))
    parser.add_argument("--metrics-output", default=str(DEFAULT_METRICS_OUT))
    parser.add_argument("--ranking-output", default=str(DEFAULT_RANKING_OUT))
    parser.add_argument("--latest-ranking-output", default=str(DEFAULT_LATEST_RANKING_OUT))
    parser.add_argument("--portfolio-returns-output", default=str(DEFAULT_PORTFOLIO_RETURNS_OUT))
    parser.add_argument("--portfolio-performance-output", default=str(DEFAULT_PORTFOLIO_PERFORMANCE_OUT))
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_panel(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, parse_dates=["Date"])
    feature_cols = [c for c in df.columns if c not in {"Instrument", "Date", "target"}]
    if not feature_cols:
        raise ValueError("No feature columns found.")
    if df[feature_cols + ["target"]].isna().any().any():
        raise ValueError("Input has NaNs in model features or target.")
    numeric = df[feature_cols + ["target"]].select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Input has inf/-inf in model features or target.")
    return df.sort_values(["Date", "Instrument"]).reset_index(drop=True), feature_cols


def make_param_candidates(n_iter: int, seed: int) -> list[dict]:
    full_size = int(np.prod([len(v) for v in PARAM_GRID.values()]))
    anchors = [dict(params) for params in ANCHOR_PARAMS]
    if n_iter >= full_size:
        keys = list(PARAM_GRID)
        full = [dict(zip(keys, values)) for values in product(*[PARAM_GRID[k] for k in keys])]
        return dedupe_params([*anchors, *full])

    random_n = max(0, n_iter - len(anchors))
    sampled = list(ParameterSampler(PARAM_GRID, n_iter=random_n, random_state=seed))
    return dedupe_params([*anchors, *sampled])[:n_iter]


def dedupe_params(candidates: list[dict]) -> list[dict]:
    out = []
    seen = set()
    keys = list(PARAM_GRID)
    for params in candidates:
        signature = tuple(params[key] for key in keys)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(params)
    return out


def xgb_model(params: dict, n_estimators: int, seed: int, early_stopping_rounds: int | None = None):
    kwargs = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": int(n_estimators),
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": 0,
        **params,
    }
    if early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    return xgb.XGBRegressor(**kwargs)


def arrays(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return df[feature_cols].to_numpy(dtype=float), df["target"].to_numpy(dtype=float)


def monthly_spearman_ic(scored: pd.DataFrame, pred_col: str = "y_pred") -> pd.Series:
    rows = []
    for date, month in scored.groupby("Date"):
        valid = month[[pred_col, "y_true"]].dropna()
        if len(valid) < 3 or valid[pred_col].nunique() < 2 or valid["y_true"].nunique() < 2:
            continue
        rows.append((date, valid[pred_col].corr(valid["y_true"], method="spearman")))
    return pd.Series(dict(rows), dtype=float)


def decile_spread_returns(scored: pd.DataFrame, pred_col: str = "y_pred", n_deciles: int = 10) -> pd.Series:
    rows = []
    for date, month in scored.groupby("Date"):
        valid = month[["Instrument", pred_col, "y_true"]].dropna().copy()
        if len(valid) < n_deciles:
            continue
        ranks = valid[pred_col].rank(method="first")
        valid["decile"] = pd.qcut(ranks, n_deciles, labels=False) + 1
        long_ret = valid.loc[valid["decile"] == n_deciles, "y_true"].mean()
        short_ret = valid.loc[valid["decile"] == 1, "y_true"].mean()
        rows.append((date, float(long_ret - short_ret)))
    return pd.Series(dict(rows), dtype=float)


def sharpe(monthly_returns: pd.Series) -> float:
    r = monthly_returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(12))


def score_predictions(scored: pd.DataFrame, benchmark_mean: float) -> dict:
    y_true = scored["y_true"].to_numpy(dtype=float)
    y_pred = scored["y_pred"].to_numpy(dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = float(np.sum((y_true - benchmark_mean) ** 2))
    r2 = float("nan") if denom == 0 else float(1 - np.sum((y_true - y_pred) ** 2) / denom)

    ic = monthly_spearman_ic(scored)
    spread = decile_spread_returns(scored)
    by_month = scored.groupby("Date")["y_pred"]

    return {
        "mse": mse,
        "mae": mae,
        "r2_vs_train_mean": r2,
        "mean_monthly_spearman_ic": float(ic.mean()) if not ic.empty else float("nan"),
        "icir": sharpe(ic),
        "mean_decile_spread": float(spread.mean()) if not spread.empty else float("nan"),
        "decile_spread_sharpe": sharpe(spread),
        "mean_pred_std": float(by_month.std().mean()),
        "median_unique_predictions": float(by_month.nunique().median()),
        "n_obs": int(len(scored)),
        "n_months": int(scored["Date"].nunique()),
    }


def cv_folds(df: pd.DataFrame, train_start: str, val_years: Iterable[int]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    train_start_ts = pd.Timestamp(train_start)
    out = []
    for year in val_years:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")
        if (df["Date"] < start).any() and (df["Date"].between(start, end)).any():
            if start <= train_start_ts:
                continue
            out.append((start, end))
    return out


def evaluate_candidate(
    df: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    candidate_id: int,
    folds: list[tuple[pd.Timestamp, pd.Timestamp]],
    train_start: str,
    seed: int,
    max_estimators: int,
    min_estimators: int,
    early_stopping_rounds: int,
) -> dict:
    fold_scores = []
    best_iterations = []
    all_scored = []
    train_start_ts = pd.Timestamp(train_start)

    for fold_id, (val_start, val_end) in enumerate(folds, start=1):
        train_mask = (df["Date"] >= train_start_ts) & (df["Date"] < val_start)
        val_mask = (df["Date"] >= val_start) & (df["Date"] <= val_end)
        train_df = df.loc[train_mask]
        val_df = df.loc[val_mask]

        X_train, y_train = arrays(train_df, feature_cols)
        X_val, y_val = arrays(val_df, feature_cols)

        model = xgb_model(
            params=params,
            n_estimators=max_estimators,
            seed=seed,
            early_stopping_rounds=early_stopping_rounds,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)

        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is None:
            best_iteration = max_estimators - 1
        selected_estimators = max(int(best_iteration) + 1, int(min_estimators))
        best_iterations.append(selected_estimators)

        if selected_estimators != int(best_iteration) + 1:
            model = xgb_model(
                params=params,
                n_estimators=selected_estimators,
                seed=seed,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
        else:
            y_pred = model.predict(X_val)

        scored = val_df[["Date", "Instrument", "target"]].rename(columns={"target": "y_true"}).copy()
        scored["y_pred"] = y_pred
        fold_score = score_predictions(scored, benchmark_mean=float(y_train.mean()))
        fold_score.update(
            {
                "candidate_id": candidate_id,
                "fold_id": fold_id,
                "val_start": val_start.date().isoformat(),
                "val_end": val_end.date().isoformat(),
                "best_iteration": int(best_iteration) + 1,
                "selected_estimators": selected_estimators,
            }
        )
        fold_scores.append(fold_score)
        all_scored.append(scored)

    fold_df = pd.DataFrame(fold_scores)
    scored_all = pd.concat(all_scored, ignore_index=True)
    pooled = score_predictions(scored_all, benchmark_mean=float("nan"))
    total_obs = fold_df["n_obs"].sum()
    weights = fold_df["n_obs"] / total_obs

    result = {
        "candidate_id": candidate_id,
        **params,
        "selected_n_estimators": int(np.median(best_iterations)),
        "mean_best_iteration": float(np.mean(best_iterations)),
        "weighted_mse": float((fold_df["mse"] * weights).sum()),
        "weighted_mae": float((fold_df["mae"] * weights).sum()),
        "mean_r2_vs_train_mean": float(fold_df["r2_vs_train_mean"].mean()),
        "mean_monthly_spearman_ic": pooled["mean_monthly_spearman_ic"],
        "icir": pooled["icir"],
        "mean_decile_spread": pooled["mean_decile_spread"],
        "decile_spread_sharpe": pooled["decile_spread_sharpe"],
        "mean_pred_std": pooled["mean_pred_std"],
        "median_unique_predictions": pooled["median_unique_predictions"],
        "n_cv_obs": int(total_obs),
        "n_cv_months": int(scored_all["Date"].nunique()),
    }
    return result


def select_best(
    hpo_results: pd.DataFrame,
    mse_tie_tolerance: float,
    min_median_unique_predictions: float,
    min_mean_pred_std: float,
) -> tuple[pd.Series, bool]:
    usable = hpo_results.loc[
        (hpo_results["median_unique_predictions"] >= min_median_unique_predictions)
        & (hpo_results["mean_pred_std"] >= min_mean_pred_std)
    ].copy()
    used_degeneracy_filter = True
    if usable.empty:
        usable = hpo_results.copy()
        used_degeneracy_filter = False

    best_mse = usable["weighted_mse"].min()
    near = usable.loc[usable["weighted_mse"] <= best_mse * (1 + mse_tie_tolerance)].copy()
    near = near.sort_values(
        ["weighted_mse", "mean_monthly_spearman_ic", "mean_pred_std", "median_unique_predictions"],
        ascending=[True, False, False, False],
    )
    return near.iloc[0], used_degeneracy_filter


def best_params_from_row(row: pd.Series) -> dict:
    params = {}
    for key in PARAM_GRID:
        value = row[key].item() if hasattr(row[key], "item") else row[key]
        if key == "max_depth":
            params[key] = int(value)
        elif key == "min_child_weight":
            params[key] = float(value)
        else:
            params[key] = float(value)
    return params


def run_oos_predictions(
    df: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    n_estimators: int,
    train_start: str,
    oos_start: str,
    oos_end: str,
    seed: int,
) -> pd.DataFrame:
    train_start_ts = pd.Timestamp(train_start)
    oos_start_ts = pd.Timestamp(oos_start)
    oos_end_ts = pd.Timestamp(oos_end)
    oos_periods = sorted(
        df.loc[(df["Date"] >= oos_start_ts) & (df["Date"] <= oos_end_ts), "Date"]
        .dt.to_period("M")
        .unique()
    )

    records = []
    for period in oos_periods:
        t = period.to_timestamp("M")
        train_mask = (df["Date"] >= train_start_ts) & (df["Date"] < t)
        test_mask = df["Date"].dt.to_period("M").eq(period)
        train_df = df.loc[train_mask]
        test_df = df.loc[test_mask]
        if train_df.empty or test_df.empty:
            continue

        X_train, y_train = arrays(train_df, feature_cols)
        X_test = test_df[feature_cols].to_numpy(dtype=float)
        model = xgb_model(params=params, n_estimators=n_estimators, seed=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        month = test_df[["Date", "Instrument", "target"]].rename(columns={"target": "y_true"}).copy()
        month["y_pred_XGB_HPO"] = y_pred
        records.append(month)
        print(f"  {period} | train={len(train_df):,} test={len(test_df):,}")

    if not records:
        return pd.DataFrame(columns=["Date", "Instrument", "y_true", "y_pred_XGB_HPO"])
    return pd.concat(records, ignore_index=True)


def evaluate_oos(pred: pd.DataFrame) -> pd.DataFrame:
    valid = pred.dropna(subset=["y_true", "y_pred_XGB_HPO"]).sort_values("Date").copy()
    valid["_benchmark"] = valid["y_true"].expanding().mean()
    y = valid["y_true"].to_numpy(dtype=float)
    yhat = valid["y_pred_XGB_HPO"].to_numpy(dtype=float)
    bench = valid["_benchmark"].to_numpy(dtype=float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - bench) ** 2)
    ic = monthly_spearman_ic(valid.rename(columns={"y_pred_XGB_HPO": "y_pred"}))
    spread = decile_spread_returns(valid.rename(columns={"y_pred_XGB_HPO": "y_pred"}))
    by_month = valid.groupby("Date")["y_pred_XGB_HPO"]
    return pd.DataFrame(
        [
            {
                "model": "XGB_HPO",
                "R2_OOS": round(float("nan") if ss_tot == 0 else float(1 - ss_res / ss_tot), 4),
                "MSE": round(float(np.mean((y - yhat) ** 2)), 6),
                "MAE": round(float(np.mean(np.abs(y - yhat))), 4),
                "Mean_Spearman_IC": round(float(ic.mean()), 4),
                "ICIR": round(sharpe(ic), 3),
                "Mean_Decile_Spread": round(float(spread.mean()), 4),
                "Decile_Spread_Sharpe": round(sharpe(spread), 3),
                "Mean_Pred_Std": round(float(by_month.std().mean()), 6),
                "Median_Unique_Predictions": round(float(by_month.nunique().median()), 1),
            }
        ]
    )


def build_rankings(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking = pred.dropna(subset=["y_pred_XGB_HPO"]).copy()
    ranking = ranking.sort_values(["Date", "y_pred_XGB_HPO", "Instrument"], ascending=[True, False, True])
    ranking["rank_xgb_hpo"] = ranking.groupby("Date")["y_pred_XGB_HPO"].rank(method="first", ascending=False).astype(int)
    ranking = ranking[["Date", "rank_xgb_hpo", "Instrument", "y_pred_XGB_HPO", "y_true"]]
    latest_date = ranking["Date"].max()
    latest = ranking.loc[ranking["Date"].eq(latest_date)].sort_values("rank_xgb_hpo").reset_index(drop=True)
    return ranking, latest


def build_portfolio_summary(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = pred.rename(columns={"y_pred_XGB_HPO": "y_pred"}).copy()
    rows = []
    for date, month in scored.groupby("Date"):
        valid = month[["Instrument", "y_pred", "y_true"]].dropna().copy()
        if len(valid) < 10:
            continue
        valid["decile"] = pd.qcut(valid["y_pred"].rank(method="first"), 10, labels=False) + 1
        long_ret = float(valid.loc[valid["decile"] == 10, "y_true"].mean())
        short_ret = float(valid.loc[valid["decile"] == 1, "y_true"].mean())
        rows.append(
            {
                "Date": date,
                "model": "XGB_HPO",
                "long_return": long_ret,
                "short_return": short_ret,
                "portfolio_return": long_ret - short_ret,
                "benchmark_return": float(valid["y_true"].mean()),
                "n_long": int((valid["decile"] == 10).sum()),
                "n_short": int((valid["decile"] == 1).sum()),
                "n_universe": int(len(valid)),
            }
        )
    portfolio_returns = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    r = portfolio_returns["portfolio_return"].dropna()
    b = portfolio_returns["benchmark_return"].dropna()
    performance = pd.DataFrame(
        [
            {
                "model": "XGB_HPO",
                "sharpe": round(sharpe(r), 3),
                "ann_return": round(float(r.mean() * 12), 4),
                "ann_vol": round(float(r.std() * np.sqrt(12)), 4),
                "cum_return": round(float((1 + r).prod() - 1), 4),
                "max_drawdown": round(max_drawdown(r), 4),
                "n_months": int(len(r)),
            },
            {
                "model": "EW_Benchmark",
                "sharpe": round(sharpe(b), 3),
                "ann_return": round(float(b.mean() * 12), 4),
                "ann_vol": round(float(b.std() * np.sqrt(12)), 4),
                "cum_return": round(float((1 + b).prod() - 1), 4),
                "max_drawdown": round(max_drawdown(b), 4),
                "n_months": int(len(b)),
            },
        ]
    )
    return portfolio_returns, performance


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    cumulative = (1 + r).cumprod()
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
    return float(drawdown.min())


def main() -> None:
    args = parse_args()
    master = resolve_path(args.master)
    output_paths = [
        resolve_path(args.hpo_output),
        resolve_path(args.best_params_output),
        resolve_path(args.predictions_output),
        resolve_path(args.metrics_output),
        resolve_path(args.ranking_output),
        resolve_path(args.latest_ranking_output),
        resolve_path(args.portfolio_returns_output),
        resolve_path(args.portfolio_performance_output),
    ]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    df, feature_cols = load_panel(master)
    val_years = [int(x.strip()) for x in args.cv_val_years.split(",") if x.strip()]
    folds = cv_folds(df, args.train_start, val_years)
    candidates = make_param_candidates(args.n_iter, args.seed)

    print(f"Input: {master}")
    print(f"Shape: {df.shape}, features={len(feature_cols)}")
    print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"CV folds: {', '.join(f'{s.date()}..{e.date()}' for s, e in folds)}")
    print(f"Candidates: {len(candidates)} sampled from expanded grid")

    results = []
    for idx, params in enumerate(candidates, start=1):
        print(f"[{idx:03d}/{len(candidates):03d}] {params}")
        result = evaluate_candidate(
            df=df,
            feature_cols=feature_cols,
            params=params,
            candidate_id=idx,
            folds=folds,
            train_start=args.train_start,
            seed=args.seed,
            max_estimators=args.max_estimators,
            min_estimators=args.min_estimators,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        print(
            "    mse={weighted_mse:.6f} r2={mean_r2_vs_train_mean:.4f} "
            "ic={mean_monthly_spearman_ic:.4f} spread={mean_decile_spread:.4f} "
            "pred_std={mean_pred_std:.6f} uniq_med={median_unique_predictions:.1f} "
            "trees={selected_n_estimators}".format(**result)
        )
        results.append(result)

    hpo = pd.DataFrame(results).sort_values("weighted_mse").reset_index(drop=True)
    hpo.to_csv(resolve_path(args.hpo_output), index=False)

    best, used_degeneracy_filter = select_best(
        hpo,
        mse_tie_tolerance=args.mse_tie_tolerance,
        min_median_unique_predictions=args.min_median_unique_predictions,
        min_mean_pred_std=args.min_mean_pred_std,
    )
    best_params = best_params_from_row(best)
    n_estimators = int(best["selected_n_estimators"])
    best_payload = {
        "selection": "lowest weighted validation MSE; close ties inspected with IC and prediction dispersion",
        "mse_tie_tolerance": args.mse_tie_tolerance,
        "min_estimators": args.min_estimators,
        "used_degeneracy_filter": used_degeneracy_filter,
        "min_median_unique_predictions": args.min_median_unique_predictions,
        "min_mean_pred_std": args.min_mean_pred_std,
        "params": best_params,
        "n_estimators": n_estimators,
        "cv_metrics": {
            key: best[key].item() if hasattr(best[key], "item") else best[key]
            for key in [
                "weighted_mse",
                "weighted_mae",
                "mean_r2_vs_train_mean",
                "mean_monthly_spearman_ic",
                "icir",
                "mean_decile_spread",
                "decile_spread_sharpe",
                "mean_pred_std",
                "median_unique_predictions",
            ]
        },
    }
    resolve_path(args.best_params_output).write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    print("\nBest params:")
    print(json.dumps(best_payload, indent=2))

    print("\nRunning final expanding-window OOS predictions...")
    pred = run_oos_predictions(
        df=df,
        feature_cols=feature_cols,
        params=best_params,
        n_estimators=n_estimators,
        train_start=args.train_start,
        oos_start=args.oos_start,
        oos_end=args.oos_end,
        seed=args.seed,
    )
    pred.to_csv(resolve_path(args.predictions_output), index=False)

    metrics = evaluate_oos(pred)
    metrics.to_csv(resolve_path(args.metrics_output), index=False)

    ranking, latest = build_rankings(pred)
    ranking.to_csv(resolve_path(args.ranking_output), index=False)
    latest.to_csv(resolve_path(args.latest_ranking_output), index=False)

    portfolio_returns, portfolio_performance = build_portfolio_summary(pred)
    portfolio_returns.to_csv(resolve_path(args.portfolio_returns_output), index=False)
    portfolio_performance.to_csv(resolve_path(args.portfolio_performance_output), index=False)

    print("\nOOS metrics:")
    print(metrics.to_string(index=False))
    print("\nPortfolio performance:")
    print(portfolio_performance.to_string(index=False))
    print("\nLatest ranking top 25:")
    print(latest.head(25).to_string(index=False))
    print("\nSaved:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
