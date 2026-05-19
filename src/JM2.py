#
# ============================================================
# CONTINUOUS STATISTICAL JUMP MODEL (SOFT JM)
# ============================================================
#
# Tama on JM.py:n sisartiedosto. Erona on, etta hard Jump Model korvataan
# Continuous Statistical Jump Modelilla (Nystrup et al. 2020; Aydinhan et al. 2024):
# regiimi ei ole binaarinen vaan jokaiselle havainnolle paatellaan
# todennakoisyysmuotoinen paino-vektori w_t simplexilta (summa = 1).
#
# 1. Sama datavirta ja samat featuret kuin JM.py:ssa
# Sama lahdedata (STOXX 600, VSTOXX, Saksan 2Y/10Y, ECB credit spread) ja samat
# featuret (stoxx_return, vstoxx, term_spread, credit_spread).
#
# 2. Simplex-hila
# Todennakoisyys-simplex diskretoidaan hilaksi: kaikki w joiden alkiot ovat
# kokonaislukukertoimia 1/G ja summautuvat 1:een. G = JM2_GRID_RESOLUTION.
# Hard JM:n state-vertexit (e_k) ovat hilan kulmapisteita, joten malli yleistaa
# JM.py:n: jos hila on pelkat kulmapisteet, tulos on identtinen hard JM:n kanssa.
#
# 3. Globaali tavoitefunktio
# min_{mu_k, w_t in hila}  sum_t sum_k w_{t,k} ||x_t - mu_k||^2
#                          + sum_t (lambda/2) ||w_t - w_{t-1}||_1
# Sisempi optimointi on sama Viterbi-tyylinen DP kuin JM.py:ssa, mutta "statet"
# ovat hilan pisteita ja siirtorangaistus on (lambda/2)*L1-etaisyys hilapisteiden
# valilla. Kulmapisteille (lambda/2)*||e_j - e_k||_1 = lambda*1{j!=k}, joten skaala
# on yhteensopiva JM.py:n kanssa.
#
# 4. Centroid-paivitys on painotettu keskiarvo
# mu_k = sum_t w_{t,k} x_t / sum_t w_{t,k} (KMeans-tyylinen M-step, mutta pehmein
# painoin). Tyhja state reseedataan satunnaisella havainnolla degeneroitumisen
# valttamiseksi.
#
# 5. Todennakoisyysmuotoinen output
# Toisin kuin hard JM, talta mallilta tulee aidot regiimipainot valille [0, 1]
# jotka summautuvat 1:een -- vertailukelpoiset HMM.py:n posteriorien kanssa.
# Tallennetaan Bull/Bear/Transition_Prob seka niiden STATE_MEAN_WINDOW_DAYS:n
# liukuva keskiarvo. Painot ovat ajallisesti tasoitettuja, koska jump penalty
# vaikuttaa suoraan w_t-sekvenssiin.
#
# 6. State-labelointi
# Sama economic-score -labelointi kuin JM.py:ssa, mutta state-keskiarvot lasketaan
# painotettuina keskiarvoina (sum_t w_{t,k} x_t / sum_t w_{t,k}). Labelointi vain
# training-window'n tiedoilla -> ei tulevien tuottojen vuotoa.
#
# 7. No-lookahead skaalaus, kuukausittainen refit ja ML-safe shift
# Identtiset JM.py:n kanssa: StandardScaler fitataan vain training-window'lle,
# kuukausittainen refit SourceDate->MonthEnd-rakenteella, ja signal month-end ->
# next target month -siirto estaa same-period leakagen.
#
import os
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import refinitiv.data as rd
from src.ecb_cache_utils import ECB_CACHE_DIR, get_ecb_series
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
from matplotlib import pyplot as plt


warnings.filterwarnings("ignore", category=FutureWarning)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "01_raw" / "outputs"
MONTHLY_OUTPUT_PATH = OUTPUT_DIR / "JM2_output.csv"
MONTHLY_ML_OUTPUT_PATH = OUTPUT_DIR / "JM2_output_ml.csv"
FEATURE_PLOT_PATH = OUTPUT_DIR / "JM2_output_features.png"
PROBABILITY_PLOT_PATH = OUTPUT_DIR / "JM2_output_probabilities.png"
LAMBDA_VALIDATION_PATH = OUTPUT_DIR / "JM2_output_lambda_validation.csv"

START_DATE = os.getenv("START_DATE", "2006-01-01")
END_DATE = os.getenv("END_DATE", "2026-04-30")
N_COMPONENTS = int(os.getenv("N_COMPONENTS", "3"))
MIN_TRAIN_OBS = int(os.getenv("MIN_TRAIN_OBS", "756"))
WINDOW_MODE = os.getenv("WINDOW_MODE", "expanding").strip().lower()
ROLLING_WINDOW_OBS = int(os.getenv("ROLLING_WINDOW_OBS", "1260"))

JM_LAMBDA = float(os.getenv("JM_LAMBDA", "10.0"))
JM_TUNE_LAMBDA = os.getenv("JM_TUNE_LAMBDA", "false").strip().lower() == "true"
JM_LAMBDA_GRID = [
    float(value) for value in os.getenv("JM_LAMBDA_GRID", "0.1,0.3,1,3,10,30,100").split(",")
]
JM_MAX_ITER = int(os.getenv("JM_MAX_ITER", "50"))
JM_TOL = float(os.getenv("JM_TOL", "1e-6"))
JM_N_INIT = int(os.getenv("JM_N_INIT", "10"))
JM_RANDOM_STATE = int(os.getenv("JM_RANDOM_STATE", "42"))
STATE_MEAN_WINDOW_DAYS = int(os.getenv("STATE_MEAN_WINDOW_DAYS", "5"))

# Resolution of the probability-simplex grid. G=5 with K=3 -> 21 grid points
# (weights are multiples of 0.2). Higher G = finer probabilities but heavier DP.
JM2_GRID_RESOLUTION = int(os.getenv("JM2_GRID_RESOLUTION", "5"))

REGIME_COLORS = {0: "green", 1: "red", 2: "orange"}
REGIME_LABELS = {0: "Bull", 1: "Bear", 2: "Transition"}

OUTPUT_COLUMN_ORDER = [
    "Date",
    "SourceDate",
    "TrainStart",
    "TrainEnd",
    "TrainObs",
    "WindowMode",
    "RollingWindowObs",
    "JM2_Regime",
    "RegimeLabel",
    "LastRawState",
    "SelectedLambda",
    "GridResolution",
    "JMObjective",
    "ModelConverged",
    "JMIterations",
    "Bull_Prob",
    "Bear_Prob",
    "Transition_Prob",
    "Bull_Prob_MeanWindow",
    "Bear_Prob_MeanWindow",
    "Transition_Prob_MeanWindow",
    "ProbabilityMeanWindowDaysUsed",
    "BullStateTrainMeanReturn",
    "BearStateTrainMeanReturn",
    "TransitionStateTrainMeanReturn",
    "BullStateEconomicScore",
    "BearStateEconomicScore",
    "TransitionStateEconomicScore",
    "NextMonthDate",
]


def fetch_input_data(start_date: str, end_date: str):
    df_equity = rd.get_history(
        universe=[".STOXX", ".V2TX"],
        fields=["TRDPRC_1"],
        start=start_date,
        end=end_date,
        interval="daily",
    )

    df_yields = rd.get_history(
        universe=["DE2YT=RR", "DE10YT=RR"],
        fields=["MID_YLD_1"],
        start=start_date,
        end=end_date,
        interval="daily",
    )

    ecb_aaa, ecb_all = get_ecb_series(start_date, end_date)
    df_credit = (ecb_all["all_yield"] - ecb_aaa["aaa_yield"]).to_frame(name="credit_spread")
    return df_equity, df_yields, df_credit


def build_feature_matrix(df_equity, df_yields, df_credit):
    jm_df = pd.DataFrame(index=df_equity.index)

    stoxx_price = df_equity[".STOXX"].astype(float)
    jm_df["stoxx_return"] = np.log(stoxx_price / stoxx_price.shift(1))
    jm_df["vstoxx"] = df_equity[".V2TX"].astype(float)
    jm_df["term_spread"] = (
        df_yields["DE10YT=RR"].astype(float) - df_yields["DE2YT=RR"].astype(float)
    )
    jm_df = jm_df.join(df_credit, how="left")
    jm_df["credit_spread"] = jm_df["credit_spread"].ffill()

    features = ["stoxx_return", "vstoxx", "term_spread", "credit_spread"]
    jm_df = jm_df[features].dropna().sort_index()
    stoxx_price = stoxx_price.reindex(jm_df.index)
    return jm_df, features, stoxx_price


def get_month_end_observation_dates(jm_df):
    source_dates = pd.DatetimeIndex(jm_df.index)
    month_end_labels = source_dates.to_period("M").to_timestamp(how="end").normalize()
    obs = pd.DataFrame({"MonthEnd": month_end_labels, "SourceDate": source_dates})
    obs = obs.groupby("MonthEnd", as_index=False).tail(1)
    obs = obs.rename(columns={"MonthEnd": "Date"})
    obs = obs[["Date", "SourceDate"]].sort_values("Date").reset_index(drop=True)
    return obs


def zscore_series(values: pd.Series):
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def build_probability_grid(resolution: int, n_components: int) -> np.ndarray:
    """
    Discretise the probability simplex into a grid of weight vectors.

    Returns every w whose entries are integer multiples of 1/resolution and sum
    to 1 (stars-and-bars enumeration). For resolution=5, n_components=3 this is
    21 grid points; the corner points e_k recover the hard-JM state vertices.
    """
    points = []
    for dividers in combinations(range(resolution + n_components - 1), n_components - 1):
        prev = -1
        parts = []
        for d in dividers:
            parts.append(d - prev - 1)
            prev = d
        parts.append(resolution + n_components - 1 - prev - 1)
        points.append(parts)
    grid = np.array(points, dtype=np.float64) / resolution
    return grid


def build_penalty_matrix(grid: np.ndarray, lambda_penalty: float) -> np.ndarray:
    """Transition penalty between grid points: (lambda/2) * L1 distance.

    For corner points this equals lambda * 1{j != k}, so the scale matches the
    hard JM jump penalty in JM.py.
    """
    l1 = np.abs(grid[:, None, :] - grid[None, :, :]).sum(axis=2)
    return 0.5 * lambda_penalty * l1


def compute_loss_matrix(
    X: np.ndarray, centroids: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """loss[t, g] = sum_k grid[g, k] * ||x_t - centroids[k]||^2."""
    diff = X[:, None, :] - centroids[None, :, :]
    sqdist = np.sum(diff * diff, axis=2)  # T x K
    return sqdist @ grid.T  # T x G


def jump_model_dynamic_programming(loss_matrix: np.ndarray, penalty_matrix: np.ndarray):
    """
    Globally optimal Viterbi-style decoder for the continuous-JM objective given
    fixed centroids. Identical structure to JM.py's decoder, but the "states" are
    grid points and the transition penalty is an arbitrary G x G matrix.

        C[0, g] = loss_matrix[0, g]
        C[t, g] = loss_matrix[t, g] + min_j { C[t-1, j] + penalty_matrix[j, g] }
    """
    T, G = loss_matrix.shape
    cost = np.empty((T, G), dtype=np.float64)
    backpointer = np.zeros((T, G), dtype=np.int32)

    cost[0] = loss_matrix[0]

    for t in range(1, T):
        transition_cost = cost[t - 1][:, None] + penalty_matrix
        best_j = transition_cost.argmin(axis=0)
        min_cost = transition_cost[best_j, np.arange(G)]
        cost[t] = loss_matrix[t] + min_cost
        backpointer[t] = best_j

    states = np.empty(T, dtype=np.int32)
    states[T - 1] = int(cost[T - 1].argmin())
    objective = float(cost[T - 1].min())
    for t in range(T - 2, -1, -1):
        states[t] = backpointer[t + 1, states[t + 1]]

    return states, objective


def _update_centroids_soft(
    X: np.ndarray, soft_weights: np.ndarray, K: int, rng: np.random.Generator
) -> np.ndarray:
    """Weighted centroid update; reseed robustly if a state has ~0 total weight."""
    centroids = np.empty((K, X.shape[1]), dtype=np.float64)
    total_weight = soft_weights.sum(axis=0)
    for k in range(K):
        if total_weight[k] > 1e-8:
            centroids[k] = (soft_weights[:, k : k + 1] * X).sum(axis=0) / total_weight[k]
        else:
            idx = int(rng.integers(0, X.shape[0]))
            centroids[k] = X[idx]
    return centroids


def _fit_continuous_jump_model_single_init(
    X: np.ndarray,
    K: int,
    grid: np.ndarray,
    penalty_matrix: np.ndarray,
    max_iter: int,
    tol: float,
    random_state: int,
):
    rng = np.random.default_rng(random_state)

    # KMeans seeds the initial assignment; its hard labels become one-hot soft
    # weights (corner points of the grid) for the first M-step.
    km = KMeans(n_clusters=K, n_init=1, random_state=random_state)
    km.fit(X)
    soft_weights = np.eye(K, dtype=np.float64)[km.labels_.astype(np.int32)]

    prev_objective = np.inf
    objective = np.inf
    converged = False
    iteration_count = 0
    grid_states = None

    for iteration in range(max_iter):
        iteration_count = iteration + 1

        # 1) Centroid update given current soft weights (weighted cluster means)
        centroids = _update_centroids_soft(X, soft_weights, K, rng)

        # 2) Weight sequence update via DP over the simplex grid (globally optimal)
        loss = compute_loss_matrix(X, centroids, grid)
        grid_states, objective = jump_model_dynamic_programming(loss, penalty_matrix)
        new_soft_weights = grid[grid_states]

        if abs(prev_objective - objective) < tol:
            soft_weights = new_soft_weights
            converged = True
            break

        soft_weights = new_soft_weights
        prev_objective = objective

    return {
        "grid_states": grid_states.astype(np.int32),
        "soft_weights": soft_weights,
        "centroids": centroids,
        "objective": float(objective),
        "iterations": int(iteration_count),
        "converged": bool(converged),
    }


def fit_continuous_jump_model(
    X: np.ndarray,
    K: int,
    grid: np.ndarray,
    lambda_penalty: float,
    max_iter: int = None,
    tol: float = None,
    n_init: int = None,
    random_state: int = None,
):
    """Fit the continuous JM with multiple restarts; keep the lowest-objective fit."""
    max_iter = JM_MAX_ITER if max_iter is None else max_iter
    tol = JM_TOL if tol is None else tol
    n_init = JM_N_INIT if n_init is None else n_init
    random_state = JM_RANDOM_STATE if random_state is None else random_state

    penalty_matrix = build_penalty_matrix(grid, lambda_penalty)
    seed_rng = np.random.default_rng(random_state)
    seeds = seed_rng.integers(0, 10**9, size=n_init)

    best = None
    for seed in seeds:
        result = _fit_continuous_jump_model_single_init(
            X, K, grid, penalty_matrix, max_iter, tol, int(seed)
        )
        if best is None or result["objective"] < best["objective"]:
            best = result
    return best


def classify_online_continuous(
    X_new: np.ndarray,
    centroids: np.ndarray,
    grid: np.ndarray,
    penalty_matrix: np.ndarray,
    prev_grid_state: int,
) -> np.ndarray:
    """
    Causal online classifier: for each new observation t, assign the grid point
        g_t = argmin_g { sum_k grid[g,k] ||x_t - mu_k||^2 + penalty(g_{t-1}, g) }
    using only information up to t. Used for the validation half of lambda tuning.
    """
    loss = compute_loss_matrix(X_new, centroids, grid)
    states = np.empty(X_new.shape[0], dtype=np.int32)
    current_prev = int(prev_grid_state)
    for t in range(X_new.shape[0]):
        total = loss[t] + penalty_matrix[current_prev]
        best_g = int(total.argmin())
        states[t] = best_g
        current_prev = best_g
    return states


def make_regime_map(soft_weights: np.ndarray, train_df: pd.DataFrame, features: list):
    """Map raw centroid ids to economic regimes (0=Bull, 1=Bear, 2=Transition).

    State-conditional means are weighted by the soft regime weights, using only
    training-window data, so no future information leaks into labelling. Raises
    if a centroid received ~0 total weight (the caller skips that month).
    """
    weights = np.asarray(soft_weights, dtype=np.float64)
    K = weights.shape[1]
    feat_vals = train_df[features].values
    total_weight = weights.sum(axis=0)

    if np.any(total_weight <= 1e-8):
        n_active = int(np.sum(total_weight > 1e-8))
        raise ValueError(
            f"Continuous JM loysi vain {n_active} aktiivista statea, tarvitaan {K}."
        )

    state_means_arr = np.empty((K, len(features)), dtype=np.float64)
    for k in range(K):
        state_means_arr[k] = (
            weights[:, k : k + 1] * feat_vals
        ).sum(axis=0) / total_weight[k]
    state_means = pd.DataFrame(state_means_arr, columns=features, index=range(K))

    standardized = state_means.apply(zscore_series, axis=0)
    economic_score = (
        standardized["stoxx_return"]
        - standardized["vstoxx"]
        - standardized["credit_spread"]
        + 0.5 * standardized["term_spread"]
    )

    sorted_states = economic_score.sort_values(ascending=False).index.tolist()

    if N_COMPONENTS == 3:
        regime_map = {
            sorted_states[0]: 0,
            sorted_states[-1]: 1,
            sorted_states[1]: 2,
        }
    elif N_COMPONENTS == 2:
        regime_map = {
            sorted_states[0]: 0,
            sorted_states[-1]: 1,
        }
    else:
        raise ValueError(f"N_COMPONENTS={N_COMPONENTS} ei tueta; kayta 2 tai 3.")

    return regime_map, state_means, economic_score


def _to_regime_ordered_weights(soft_weights: np.ndarray, regime_map: dict) -> np.ndarray:
    """Reorder raw-centroid weight columns into regime order (0=Bull,1=Bear,2=Trans)."""
    regime_weights = np.zeros_like(soft_weights)
    for raw_state, regime_id in regime_map.items():
        regime_weights[:, regime_id] = soft_weights[:, raw_state]
    return regime_weights


def select_lambda(train_df: pd.DataFrame, features: list, grid: np.ndarray):
    """Pick lambda via an 80/20 split within the current training window.

    Estimation subwindow = first 80% (used to fit the JM and build the regime map).
    Validation subwindow = last 20%, classified causally with the online classifier.
    Score: validation Sharpe of a 1-day-delayed strategy whose exposure is the
    continuous risk-on weight (1 - Bear_Prob). Ties on Sharpe (within 0.05) are
    broken by lower grid-switch count.
    """
    n = len(train_df)
    n_est = int(np.floor(0.8 * n))
    est_df = train_df.iloc[:n_est]
    val_df = train_df.iloc[n_est:]

    if len(val_df) < 20 or len(est_df) < MIN_TRAIN_OBS:
        return JM_LAMBDA, pd.DataFrame()

    scaler = StandardScaler()
    X_est = scaler.fit_transform(est_df[features].values)
    X_val = scaler.transform(val_df[features].values)
    val_returns = val_df["stoxx_return"].values

    results = []
    for lam in JM_LAMBDA_GRID:
        try:
            fit = fit_continuous_jump_model(X_est, N_COMPONENTS, grid, lam)
            regime_map_est, _, _ = make_regime_map(fit["soft_weights"], est_df, features)
        except Exception:
            results.append(
                {
                    "Lambda": lam,
                    "Sharpe": float("nan"),
                    "MeanReturn": float("nan"),
                    "Volatility": float("nan"),
                    "Switches": -1,
                    "Valid": False,
                }
            )
            continue

        penalty_matrix = build_penalty_matrix(grid, lam)
        prev_grid_state = int(fit["grid_states"][-1])
        val_grid_states = classify_online_continuous(
            X_val, fit["centroids"], grid, penalty_matrix, prev_grid_state
        )
        val_weights_raw = grid[val_grid_states]

        bear_raw_state = next(k for k, v in regime_map_est.items() if v == 1)
        bear_prob = val_weights_raw[:, bear_raw_state]
        # Continuous risk-on exposure rather than a hard risk-on/off switch.
        positions = 1.0 - bear_prob
        # 1-day delay: exposure decided at t-1 trades return at t (no same-bar peeking)
        positions_lagged = np.concatenate([[0.0], positions[:-1]])
        strategy_returns = positions_lagged * val_returns

        mean_ret = float(strategy_returns.mean())
        vol = float(strategy_returns.std(ddof=0))
        if vol == 0 or not np.isfinite(vol):
            sharpe = float("nan")
        else:
            sharpe = float((mean_ret / vol) * np.sqrt(252))

        n_switches = int((val_grid_states[1:] != val_grid_states[:-1]).sum())
        results.append(
            {
                "Lambda": lam,
                "Sharpe": sharpe,
                "MeanReturn": mean_ret,
                "Volatility": vol,
                "Switches": n_switches,
                "Valid": True,
            }
        )

    results_df = pd.DataFrame(results)
    valid = results_df[results_df["Valid"] & results_df["Sharpe"].notna()]
    if valid.empty:
        return JM_LAMBDA, results_df

    max_sharpe = valid["Sharpe"].max()
    sharpe_tol = 0.05
    near_best = valid[valid["Sharpe"] >= max_sharpe - sharpe_tol]
    selected = near_best.sort_values(["Switches", "Lambda"]).iloc[0]
    return float(selected["Lambda"]), results_df


def fit_predict_month_end_regime(train_df: pd.DataFrame, features: list, grid: np.ndarray):
    # StandardScaler is fit ONLY on this month's training window — no look-ahead.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_df[features].values)

    if JM_TUNE_LAMBDA:
        selected_lambda, lambda_diag = select_lambda(train_df, features, grid)
    else:
        selected_lambda = JM_LAMBDA
        lambda_diag = None

    fit = fit_continuous_jump_model(X_scaled, N_COMPONENTS, grid, selected_lambda)

    soft_weights = fit["soft_weights"]
    regime_map, state_means, economic_score = make_regime_map(
        soft_weights, train_df, features
    )

    regime_weights = _to_regime_ordered_weights(soft_weights, regime_map)
    last_probs = regime_weights[-1]
    last_regime = int(np.argmax(last_probs))
    last_raw_state = int(np.argmax(soft_weights[-1]))

    window_days = min(STATE_MEAN_WINDOW_DAYS, len(regime_weights))
    window_mean = regime_weights[-window_days:].mean(axis=0)

    bull_state = next((k for k, v in regime_map.items() if v == 0), None)
    bear_state = next((k for k, v in regime_map.items() if v == 1), None)
    transition_state = next((k for k, v in regime_map.items() if v == 2), None)

    def state_mean_return(state_key):
        if state_key is None or state_key not in state_means.index:
            return float("nan")
        return float(state_means.loc[state_key, "stoxx_return"])

    def state_econ_score(state_key):
        if state_key is None or state_key not in economic_score.index:
            return float("nan")
        return float(economic_score.loc[state_key])

    return {
        "JM2_Regime": last_regime,
        "LastRawState": last_raw_state,
        "SelectedLambda": float(selected_lambda),
        "GridResolution": int(JM2_GRID_RESOLUTION),
        "JMObjective": float(fit["objective"]),
        "ModelConverged": bool(fit["converged"]),
        "JMIterations": int(fit["iterations"]),
        "Bull_Prob": float(last_probs[0]),
        "Bear_Prob": float(last_probs[1]),
        "Transition_Prob": float(last_probs[2]),
        "Bull_Prob_MeanWindow": float(window_mean[0]),
        "Bear_Prob_MeanWindow": float(window_mean[1]),
        "Transition_Prob_MeanWindow": float(window_mean[2]),
        "ProbabilityMeanWindowDaysUsed": int(window_days),
        "BullStateTrainMeanReturn": state_mean_return(bull_state),
        "BearStateTrainMeanReturn": state_mean_return(bear_state),
        "TransitionStateTrainMeanReturn": state_mean_return(transition_state),
        "BullStateEconomicScore": state_econ_score(bull_state),
        "BearStateEconomicScore": state_econ_score(bear_state),
        "TransitionStateEconomicScore": state_econ_score(transition_state),
    }, lambda_diag


def select_train_window(jm_df: pd.DataFrame, source_date: pd.Timestamp) -> pd.DataFrame:
    full_train_df = jm_df.loc[:source_date].copy()
    if WINDOW_MODE == "expanding":
        return full_train_df
    if WINDOW_MODE == "rolling":
        return full_train_df.tail(ROLLING_WINDOW_OBS).copy()
    raise ValueError(
        f"Tuntematon WINDOW_MODE={WINDOW_MODE!r}. Kayta arvoa 'expanding' tai 'rolling'."
    )


def build_no_lookahead_monthly_regimes(jm_df: pd.DataFrame, features: list, grid: np.ndarray):
    month_end_obs = get_month_end_observation_dates(jm_df)
    rows = []
    lambda_diagnostics = []

    total_month_ends = len(month_end_obs)
    month_end_pairs = month_end_obs[["Date", "SourceDate"]].itertuples(index=False, name=None)

    for position, (month_end_raw, source_date_raw) in enumerate(month_end_pairs, start=1):
        month_end = pd.Timestamp(month_end_raw)
        source_date = pd.Timestamp(source_date_raw)
        train_df = select_train_window(jm_df, source_date)

        if len(train_df) < MIN_TRAIN_OBS:
            continue

        try:
            pred, lambda_diag = fit_predict_month_end_regime(train_df, features, grid)
        except Exception as exc:
            print(
                f"JM2 fit epaonnistui kuukaudelle {month_end.date()} "
                f"(source {source_date.date()}): {exc}"
            )
            continue

        rows.append(
            {
                "Date": month_end,
                "SourceDate": source_date,
                "TrainStart": train_df.index.min(),
                "TrainEnd": train_df.index.max(),
                "TrainObs": int(len(train_df)),
                "WindowMode": WINDOW_MODE,
                "RollingWindowObs": ROLLING_WINDOW_OBS if WINDOW_MODE == "rolling" else np.nan,
                **pred,
            }
        )

        if lambda_diag is not None and not lambda_diag.empty:
            tmp = lambda_diag.copy()
            tmp.insert(0, "Date", month_end)
            tmp.insert(1, "SourceDate", source_date)
            lambda_diagnostics.append(tmp)

        if position % 12 == 0 or position == total_month_ends:
            print(
                f"No-lookahead JM2: {position}/{total_month_ends} month-endia kasitelty "
                f"(viimeisin {month_end.strftime('%Y-%m')})"
            )

    monthly = pd.DataFrame(rows)
    if monthly.empty:
        return monthly, pd.DataFrame()

    monthly["RegimeLabel"] = monthly["JM2_Regime"].map(REGIME_LABELS)
    monthly["NextMonthDate"] = monthly["Date"] + pd.offsets.MonthEnd(1)
    monthly = monthly[OUTPUT_COLUMN_ORDER].sort_values("Date").reset_index(drop=True)

    if lambda_diagnostics:
        lambda_df = (
            pd.concat(lambda_diagnostics, ignore_index=True)
            .sort_values(["Date", "Lambda"])
            .reset_index(drop=True)
        )
    else:
        lambda_df = pd.DataFrame()

    return monthly, lambda_df


def build_ml_safe_monthly_regimes(monthly: pd.DataFrame) -> pd.DataFrame:
    ml = monthly.copy()
    ml = ml.rename(columns={"Date": "SignalMonthEnd", "SourceDate": "SignalSourceDate"})
    ml["Date"] = ml["SignalMonthEnd"] + pd.offsets.MonthEnd(1)
    ml["TargetMonth"] = ml["Date"].dt.to_period("M").astype(str)

    lead_cols = ["Date", "TargetMonth", "SignalMonthEnd", "SignalSourceDate"]
    other_cols = [col for col in ml.columns if col not in lead_cols and col != "NextMonthDate"]
    return ml[lead_cols + other_cols].sort_values("Date").reset_index(drop=True)


def plot_month_end_feature_regimes(
    jm_df: pd.DataFrame, stoxx_price: pd.Series, monthly: pd.DataFrame, output_path: Path
):
    monthly_plot = monthly.set_index("SourceDate").sort_index()
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    series_specs = [
        ("STOXX 600", stoxx_price, axes[0]),
        ("VSTOXX", jm_df["vstoxx"], axes[1]),
        ("Term Spread (10Y-2Y)", jm_df["term_spread"], axes[2]),
        ("Sovereign Spread (All-AAA)", jm_df["credit_spread"], axes[3]),
    ]

    for label, full_series, axis in series_specs:
        axis.plot(full_series.index, full_series.values, color="lightgray", linewidth=0.9)
        for regime_id, color in REGIME_COLORS.items():
            mask = monthly_plot["JM2_Regime"] == regime_id
            axis.scatter(
                monthly_plot.index[mask],
                full_series.reindex(monthly_plot.index[mask]),
                c=color,
                s=20,
                alpha=0.9,
                label=REGIME_LABELS[regime_id],
            )
        axis.set_ylabel(label)

    axes[0].legend(loc="upper left", markerscale=1.3)
    axes[0].set_title("No-Lookahead Continuous Jump Model Month-End Regimes")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_probabilities(monthly: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(
        monthly["Date"],
        monthly["Bull_Prob"],
        color=REGIME_COLORS[0],
        linewidth=1.7,
        label="Bull_Prob",
    )
    ax.plot(
        monthly["Date"],
        monthly["Bear_Prob"],
        color=REGIME_COLORS[1],
        linewidth=1.7,
        label="Bear_Prob",
    )
    ax.plot(
        monthly["Date"],
        monthly["Transition_Prob"],
        color=REGIME_COLORS[2],
        linewidth=1.7,
        label="Transition_Prob",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Regime probability")
    ax.set_xlabel("Date")
    ax.set_title(
        "No-Lookahead Continuous Jump Model Month-End Regime Probabilities"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ECB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if WINDOW_MODE not in {"expanding", "rolling"}:
        raise ValueError(
            f"Tuntematon WINDOW_MODE={WINDOW_MODE!r}. Kayta arvoa 'expanding' tai 'rolling'."
        )
    if STATE_MEAN_WINDOW_DAYS < 1:
        raise ValueError("STATE_MEAN_WINDOW_DAYS tulee olla vahintaan 1.")
    if WINDOW_MODE == "rolling" and ROLLING_WINDOW_OBS < MIN_TRAIN_OBS:
        raise ValueError("ROLLING_WINDOW_OBS tulee olla >= MIN_TRAIN_OBS rolling-tilassa.")
    if N_COMPONENTS not in {2, 3}:
        raise ValueError(f"N_COMPONENTS={N_COMPONENTS} ei tueta; kayta 2 tai 3.")
    if JM_LAMBDA < 0:
        raise ValueError("JM_LAMBDA tulee olla >= 0.")
    if JM_N_INIT < 1:
        raise ValueError("JM_N_INIT tulee olla vahintaan 1.")
    if JM_MAX_ITER < 1:
        raise ValueError("JM_MAX_ITER tulee olla vahintaan 1.")
    if any(lam < 0 for lam in JM_LAMBDA_GRID):
        raise ValueError("JM_LAMBDA_GRID arvojen tulee olla >= 0.")
    if JM2_GRID_RESOLUTION < 1:
        raise ValueError("JM2_GRID_RESOLUTION tulee olla vahintaan 1.")

    grid = build_probability_grid(JM2_GRID_RESOLUTION, N_COMPONENTS)

    print(
        "Building no-lookahead Continuous Jump Model monthly regimes "
        f"with {WINDOW_MODE} window from {START_DATE} to {END_DATE} "
        f"(min train obs={MIN_TRAIN_OBS}, state-mean window={STATE_MEAN_WINDOW_DAYS}d, "
        f"K={N_COMPONENTS}, n_init={JM_N_INIT}, tune_lambda={JM_TUNE_LAMBDA}, "
        f"grid_resolution={JM2_GRID_RESOLUTION} -> {len(grid)} grid points)"
    )
    if WINDOW_MODE == "rolling":
        print(f"Rolling window observations: {ROLLING_WINDOW_OBS}")
    if JM_TUNE_LAMBDA:
        print(f"Lambda grid: {JM_LAMBDA_GRID}")
    else:
        print(f"Fixed lambda: {JM_LAMBDA}")

    rd.open_session()
    try:
        df_equity, df_yields, df_credit = fetch_input_data(START_DATE, END_DATE)
        jm_df, features, stoxx_price = build_feature_matrix(df_equity, df_yields, df_credit)

        print(f"Feature matrix: {len(jm_df)} daily observations")
        print(f"Date range: {jm_df.index.min().date()} — {jm_df.index.max().date()}")

        monthly, lambda_df = build_no_lookahead_monthly_regimes(jm_df, features, grid)
        if monthly.empty:
            raise RuntimeError("No-lookahead monthly regime table jaa tyhjaksi.")

        monthly_ml = build_ml_safe_monthly_regimes(monthly)

        monthly.to_csv(MONTHLY_OUTPUT_PATH, index=False)
        monthly_ml.to_csv(MONTHLY_ML_OUTPUT_PATH, index=False)
        plot_month_end_feature_regimes(jm_df, stoxx_price, monthly, FEATURE_PLOT_PATH)
        plot_monthly_probabilities(monthly, PROBABILITY_PLOT_PATH)

        if JM_TUNE_LAMBDA and not lambda_df.empty:
            lambda_df.to_csv(LAMBDA_VALIDATION_PATH, index=False)

        print(f"\nSaved: {MONTHLY_OUTPUT_PATH}")
        print(f"Saved: {MONTHLY_ML_OUTPUT_PATH}")
        print(f"Saved: {FEATURE_PLOT_PATH}")
        print(f"Saved: {PROBABILITY_PLOT_PATH}")
        if JM_TUNE_LAMBDA and not lambda_df.empty:
            print(f"Saved: {LAMBDA_VALIDATION_PATH}")
        print(f"Rows: {len(monthly)}")
        print(
            f"Date range: {monthly['Date'].min().strftime('%Y-%m')} — "
            f"{monthly['Date'].max().strftime('%Y-%m')}"
        )
        print("\nRegime distribution (argmax):")
        print(
            monthly["JM2_Regime"]
            .value_counts()
            .rename({0: "Bull (0)", 1: "Bear (1)", 2: "Transition (2)"})
            .sort_index()
        )
        print("\nHead:")
        print(monthly.head(12).to_string(index=False))
    finally:
        rd.close_session()


if __name__ == "__main__":
    main()
