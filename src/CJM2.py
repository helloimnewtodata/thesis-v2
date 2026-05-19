#
# ============================================================
# CONTINUOUS STATISTICAL JUMP MODEL — GENUINE QP SOLVE (CJM2)
# ============================================================
#
# Toteutus seuraa samaa paperia kuin CJM.py:
#   Aydinhan, Kolm, Mulvey & Shu (2024), "Identifying Patterns in Financial
#   Markets: Extending the Statistical Jump Model for Regime Identification".
#
# Tama on NELJAS JM-toteutus repossa:
#   JM.py   = hard discrete jump model
#   JM2.py  = naiivi continuous -yritys (tavallinen L1) -> romahti hardiksi
#   CJM.py  = paperin mukainen continuous, mutta state-osaongelma ratkaistaan
#             DISKRETOIMALLA simplex hilaksi ja ajamalla Viterbi-DP sen yli
#   CJM2.py = sama tavoitefunktio kuin CJM.py:ssa, mutta state-osaongelma
#             ratkaistaan AITONA jatkuvana QP:na koko simplexin yli (ei hilaa)
#
# 1. Miksi CJM.py:n arvot jaivat "koviksi"
# CJM.py:n hila-DP valitsee joka paivalle TASAN YHDEN hilapisteen. Arvot ovat
# siis aina 1/resolution-monikertoja, ja kun lambda on fit-kustannukseen nahden
# pieni, DP istuu hilan KULMISSA -> output naytti ~100 % kovalta 0/1:lta.
#
# 2. CJM2: jatkuva ratkaisu, ei hilaa
# Tavoitefunktio (paperin yhtalo 5, identtinen CJM.py:n kanssa):
#   min_{Theta,S}  sum_t sum_k s_{t,k} l(y_t, theta_k)
#                  + (lambda/4) sum_{t>0} ||s_{t-1} - s_t||_1^2
# State-osaongelma (Theta kiinnitettyna) on KONVEKSI optimointi yli koko
# todennakoisyys-simplexin: lineaarinen fit-termi + neliojen summa nelioidysta
# L1-rangaistuksesta. Ratkaistaan suoraan jatkuvana (FISTA: kiihdytetty
# projektoitu gradientti, projektio per-paiva simplexille). Optimi voi olla
# missa tahansa simplexin sisalla -> aidot, jatkuva-arvoiset regiimipainot,
# jotka EIVAT snappaa mihinkaan hilaan.
#
# 3. Nelioity L1 sailyy
# Sama syy kuin CJM.py:ssa: tavallinen L1 tekee state-osaongelmasta LP:n, jonka
# optimi on aina simplexin karjessa. Neliointi tekee siita QP:n, jonka optimi
# voi olla sisalla. CJM2 ei muuta rangaistusta — vain ratkaisutavan.
#
# 4. Sileytetty L1 numeriikkaa varten
# ||d||_1 ei ole derivoituva nollassa. Korvataan |d_k| ~ sqrt(d_k^2 + eps)
# (CJM2_SMOOTH_EPS, oletus 1e-4), jolloin koko tavoitefunktio on sileasti
# derivoituva ja FISTA suppenee. eps -> 0 palauttaa tarkan neliodyn L1:n.
#
# 5. Lambda-skaala
# Karkipisteille (lambda/4)*||e_a - e_b||_1^2 = lambda, joten lambda on samalla
# skaalalla kuin hard JM:ssa. Continuous-malli tarvitsee silti SUUREMMAN
# lambdan kuin diskreetti (paperi luku 5-6: paivadatalla continuous-optimaali
# ~10^3 vs. diskreetti ~10^2). Siksi CJM2:n JM_LAMBDA-oletus on 1000 ja
# JM_LAMBDA_GRID ulottuu 3000:een — pieni lambda istuu QP-optimin karkiin
# riippumatta siita ratkaistaanko hilalla vai jatkuvasti.
#
# 6. No-lookahead refit, economic-score -labelointi, ML-safe shift
# Sama thesis-pipeline-rakenne kuin JM.py/CJM.py:ssa: StandardScaler ja fit
# vain training-window'lle, kuukausittainen refit SourceDate->MonthEnd, ja
# signal month-end -> next target month -siirto estaa same-period leakagen.
#
import os
import warnings
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
MONTHLY_OUTPUT_PATH = OUTPUT_DIR / "CJM2_output.csv"
MONTHLY_ML_OUTPUT_PATH = OUTPUT_DIR / "CJM2_output_ml.csv"
FEATURE_PLOT_PATH = OUTPUT_DIR / "CJM2_output_features.png"
PROBABILITY_PLOT_PATH = OUTPUT_DIR / "CJM2_output_probabilities.png"
LAMBDA_VALIDATION_PATH = OUTPUT_DIR / "CJM2_output_lambda_validation.csv"

START_DATE = os.getenv("START_DATE", "2006-01-01")
END_DATE = os.getenv("END_DATE", "2026-04-30")
N_COMPONENTS = int(os.getenv("N_COMPONENTS", "3"))
MIN_TRAIN_OBS = int(os.getenv("MIN_TRAIN_OBS", "756"))
WINDOW_MODE = os.getenv("WINDOW_MODE", "expanding").strip().lower()
ROLLING_WINDOW_OBS = int(os.getenv("ROLLING_WINDOW_OBS", "1260"))

# Continuous-model lambda scale: the genuine QP optimum sits at the simplex
# corners for small lambda exactly as the grid DP did, so the default is on the
# continuous scale (Aydinhan et al. 2024, Sec. 5-6: ~10^3 for daily data).
JM_LAMBDA = float(os.getenv("JM_LAMBDA", "1000.0"))
JM_TUNE_LAMBDA = os.getenv("JM_TUNE_LAMBDA", "false").strip().lower() == "true"
JM_LAMBDA_GRID = [
    float(value)
    for value in os.getenv("JM_LAMBDA_GRID", "30,100,300,1000,3000").split(",")
]
JM_MAX_ITER = int(os.getenv("JM_MAX_ITER", "50"))
JM_TOL = float(os.getenv("JM_TOL", "1e-6"))
JM_N_INIT = int(os.getenv("JM_N_INIT", "10"))
JM_RANDOM_STATE = int(os.getenv("JM_RANDOM_STATE", "42"))
STATE_MEAN_WINDOW_DAYS = int(os.getenv("STATE_MEAN_WINDOW_DAYS", "5"))

# Smoothing of the squared-L1 penalty: |d| ~ sqrt(d^2 + eps). Smaller eps is
# closer to the exact penalty but makes the FISTA inner solve stiffer.
CJM2_SMOOTH_EPS = float(os.getenv("CJM2_SMOOTH_EPS", "1e-4"))
# FISTA inner solver budget for the state subproblem (full training window).
CJM2_INNER_MAX_ITER = int(os.getenv("CJM2_INNER_MAX_ITER", "500"))
CJM2_INNER_TOL = float(os.getenv("CJM2_INNER_TOL", "1e-7"))
# FISTA budget for the causal per-step online classifier (lambda tuning only).
CJM2_ONLINE_MAX_ITER = int(os.getenv("CJM2_ONLINE_MAX_ITER", "200"))

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
    "CJM2_Regime",
    "RegimeLabel",
    "LastRawState",
    "SelectedLambda",
    "SmoothEps",
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
    cjm_df = pd.DataFrame(index=df_equity.index)

    stoxx_price = df_equity[".STOXX"].astype(float)
    cjm_df["stoxx_return"] = np.log(stoxx_price / stoxx_price.shift(1))
    cjm_df["vstoxx"] = df_equity[".V2TX"].astype(float)
    cjm_df["term_spread"] = (
        df_yields["DE10YT=RR"].astype(float) - df_yields["DE2YT=RR"].astype(float)
    )
    cjm_df = cjm_df.join(df_credit, how="left")
    cjm_df["credit_spread"] = cjm_df["credit_spread"].ffill()

    features = ["stoxx_return", "vstoxx", "term_spread", "credit_spread"]
    cjm_df = cjm_df[features].dropna().sort_index()
    stoxx_price = stoxx_price.reindex(cjm_df.index)
    return cjm_df, features, stoxx_price


def get_month_end_observation_dates(cjm_df):
    source_dates = pd.DatetimeIndex(cjm_df.index)
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


def project_rows_to_simplex(V: np.ndarray) -> np.ndarray:
    """Euclidean projection of every row of V onto the probability simplex.

    Standard sort-based algorithm (Held et al. 1974; Wang & Carreira-Perpinan
    2013), fully vectorised over rows. Each returned row is non-negative and
    sums to exactly 1.
    """
    n_rows, K = V.shape
    U = np.sort(V, axis=1)[:, ::-1]
    cssv = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, K + 1)
    cond = U - cssv / ind > 0
    rho = cond.sum(axis=1)
    theta = cssv[np.arange(n_rows), rho - 1] / rho
    return np.maximum(V - theta[:, None], 0.0)


def project_vec_to_simplex(v: np.ndarray) -> np.ndarray:
    """Scalar-row version of project_rows_to_simplex (used by the online solver)."""
    K = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, K + 1)
    cond = u - cssv / ind > 0
    rho = int(np.nonzero(cond)[0][-1])
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


def _state_value(S: np.ndarray, loss: np.ndarray, lam: float, eps: float) -> float:
    """Objective of the state subproblem: linear fit term + smoothed (lambda/4)
    squared-L1 jump penalty between consecutive rows."""
    D = S[:-1] - S[1:]
    h = np.sqrt(D * D + eps)
    H = h.sum(axis=1)
    return float(np.sum(loss * S)) + 0.25 * lam * float(np.sum(H * H))


def _state_value_grad(S: np.ndarray, loss: np.ndarray, lam: float, eps: float):
    """Objective value and its gradient w.r.t. S for the state subproblem.

    The penalty for the pair (t, t+1) is (lambda/4) * pen_eps(S[t] - S[t+1]),
    pen_eps(d) = (sum_k sqrt(d_k^2 + eps))^2. d pen / d d = 2 H d / h, which is
    routed +back to row t and -forward to row t+1.
    """
    D = S[:-1] - S[1:]
    h = np.sqrt(D * D + eps)
    H = h.sum(axis=1, keepdims=True)
    value = float(np.sum(loss * S)) + 0.25 * lam * float(np.sum(H[:, 0] ** 2))

    gpen = 2.0 * H * (D / h)  # d pen / d D, shape (T-1, K)
    G = loss.copy()
    G[:-1] += 0.25 * lam * gpen
    G[1:] -= 0.25 * lam * gpen
    return value, G


def solve_state_subproblem(
    loss: np.ndarray,
    lam: float,
    eps: float,
    max_iter: int,
    tol: float,
    S_init: np.ndarray,
) -> np.ndarray:
    """Solve the (convex) state subproblem over the FULL probability simplex.

    minimise   sum_t loss[t] . s_t  +  (lambda/4) sum_t pen_eps(s_t - s_{t+1})
    subject to s_t on the probability simplex.

    Method: FISTA (accelerated projected gradient) with backtracking line
    search. The whole objective is smooth after the eps-smoothing of the L1
    norm, and convex, so FISTA converges to the global optimum of the
    subproblem. No grid — s_t can land anywhere in the simplex.
    """
    S = S_init.copy()
    Y = S.copy()
    t_acc = 1.0
    L = 1.0
    prev_value = np.inf

    for _ in range(max_iter):
        L = max(L * 0.8, 1e-6)  # allow the step to grow back between iterations
        f_Y, G = _state_value_grad(Y, loss, lam, eps)

        while True:
            S_new = project_rows_to_simplex(Y - G / L)
            diff = S_new - Y
            f_new = _state_value(S_new, loss, lam, eps)
            quad = f_Y + float(np.sum(G * diff)) + 0.5 * L * float(np.sum(diff * diff))
            if f_new <= quad + 1e-9 or L > 1e12:
                break
            L *= 2.0

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_acc * t_acc)) / 2.0
        Y = S_new + ((t_acc - 1.0) / t_new) * (S_new - S)
        S = S_new
        t_acc = t_new

        if abs(prev_value - f_new) < tol * (1.0 + abs(prev_value)):
            break
        prev_value = f_new

    return S


def _update_centroids_soft(
    X: np.ndarray, soft_weights: np.ndarray, K: int, rng: np.random.Generator
) -> np.ndarray:
    """Coordinate-descent step (a): weighted centroid means.

    For the scaled squared-L2 loss the solution is analytic:
        theta_k = sum_t s_{t,k} x_t / sum_t s_{t,k}.
    A state with ~0 total weight is reseeded with a random training observation.
    """
    centroids = np.empty((K, X.shape[1]), dtype=np.float64)
    total_weight = soft_weights.sum(axis=0)
    for k in range(K):
        if total_weight[k] > 1e-8:
            centroids[k] = (soft_weights[:, k : k + 1] * X).sum(axis=0) / total_weight[k]
        else:
            idx = int(rng.integers(0, X.shape[0]))
            centroids[k] = X[idx]
    return centroids


def _compute_loss_matrix(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """loss[t, k] = (1/2) ||x_t - centroids[k]||_2^2 — the scaled squared-L2 loss."""
    diff = X[:, None, :] - centroids[None, :, :]
    return 0.5 * np.sum(diff * diff, axis=2)


def _fit_cjm2_single_init(
    X: np.ndarray,
    K: int,
    lam: float,
    eps: float,
    inner_max_iter: int,
    inner_tol: float,
    outer_max_iter: int,
    outer_tol: float,
    random_state: int,
):
    rng = np.random.default_rng(random_state)

    # K-means++ seeds the initial assignment; its hard labels become one-hot
    # soft weights for the first centroid update.
    km = KMeans(n_clusters=K, n_init=1, random_state=random_state)
    km.fit(X)
    soft_weights = np.eye(K, dtype=np.float64)[km.labels_.astype(np.int32)]

    prev_objective = np.inf
    objective = np.inf
    converged = False
    iteration_count = 0
    centroids = None

    for iteration in range(outer_max_iter):
        iteration_count = iteration + 1

        # (a) Fit centroids given the current soft weights
        centroids = _update_centroids_soft(X, soft_weights, K, rng)

        # (b) Fit the soft state sequence as a genuine continuous QP. The
        #     previous iteration's weights warm-start FISTA, so the inner solve
        #     converges quickly after the first outer iteration.
        loss = _compute_loss_matrix(X, centroids)
        soft_weights = solve_state_subproblem(
            loss, lam, eps, inner_max_iter, inner_tol, soft_weights
        )
        objective = _state_value(soft_weights, loss, lam, eps)

        if abs(prev_objective - objective) < outer_tol:
            converged = True
            break
        prev_objective = objective

    return {
        "soft_weights": soft_weights,
        "centroids": centroids,
        "objective": float(objective),
        "iterations": int(iteration_count),
        "converged": bool(converged),
    }


def fit_cjm2(
    X: np.ndarray,
    K: int,
    lambda_penalty: float,
    eps: float = None,
    inner_max_iter: int = None,
    inner_tol: float = None,
    outer_max_iter: int = None,
    outer_tol: float = None,
    n_init: int = None,
    random_state: int = None,
):
    """Fit the continuous jump model with multiple restarts; keep the lowest-
    objective solution."""
    eps = CJM2_SMOOTH_EPS if eps is None else eps
    inner_max_iter = CJM2_INNER_MAX_ITER if inner_max_iter is None else inner_max_iter
    inner_tol = CJM2_INNER_TOL if inner_tol is None else inner_tol
    outer_max_iter = JM_MAX_ITER if outer_max_iter is None else outer_max_iter
    outer_tol = JM_TOL if outer_tol is None else outer_tol
    n_init = JM_N_INIT if n_init is None else n_init
    random_state = JM_RANDOM_STATE if random_state is None else random_state

    seed_rng = np.random.default_rng(random_state)
    seeds = seed_rng.integers(0, 10**9, size=n_init)

    best = None
    for seed in seeds:
        result = _fit_cjm2_single_init(
            X,
            K,
            lambda_penalty,
            eps,
            inner_max_iter,
            inner_tol,
            outer_max_iter,
            outer_tol,
            int(seed),
        )
        if best is None or result["objective"] < best["objective"]:
            best = result
    return best


def _solve_online_step(
    loss_vec: np.ndarray,
    s_prev: np.ndarray,
    lam: float,
    eps: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Causal per-observation state solve: minimise over s on the simplex
        loss_vec . s + (lambda/4) pen_eps(s_prev - s)
    given the previously assigned soft state s_prev. Convex; solved with FISTA.
    """
    s = s_prev.copy()
    y = s.copy()
    t_acc = 1.0
    L = 1.0
    prev_value = np.inf

    def value(vec):
        d = s_prev - vec
        h = np.sqrt(d * d + eps)
        return float(loss_vec @ vec) + 0.25 * lam * float(h.sum() ** 2)

    for _ in range(max_iter):
        L = max(L * 0.8, 1e-6)
        d = s_prev - y
        h = np.sqrt(d * d + eps)
        H = float(h.sum())
        # d/ds [ loss.s + (lambda/4) H^2 ];  d/ds of (s_prev - s) is -I
        grad = loss_vec - 0.5 * lam * H * (d / h)
        f_y = float(loss_vec @ y) + 0.25 * lam * H * H

        while True:
            s_new = project_vec_to_simplex(y - grad / L)
            diff = s_new - y
            f_new = value(s_new)
            quad = f_y + float(grad @ diff) + 0.5 * L * float(diff @ diff)
            if f_new <= quad + 1e-9 or L > 1e12:
                break
            L *= 2.0

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_acc * t_acc)) / 2.0
        y = s_new + ((t_acc - 1.0) / t_new) * (s_new - s)
        s = s_new
        t_acc = t_new

        if abs(prev_value - f_new) < tol * (1.0 + abs(prev_value)):
            break
        prev_value = f_new

    return s


def classify_online_cjm2(
    X_new: np.ndarray,
    centroids: np.ndarray,
    lam: float,
    eps: float,
    prev_weights: np.ndarray,
) -> np.ndarray:
    """Causal online classifier: for each new observation t solve the per-step
    QP using only information up to t and the previous soft state. Used for the
    validation half of the lambda tuning split — strictly no peeking ahead.
    """
    T = X_new.shape[0]
    K = centroids.shape[0]
    weights = np.empty((T, K), dtype=np.float64)
    s_prev = np.asarray(prev_weights, dtype=np.float64).copy()
    for t in range(T):
        loss_vec = 0.5 * np.sum((centroids - X_new[t]) ** 2, axis=1)
        s_prev = _solve_online_step(
            loss_vec, s_prev, lam, eps, CJM2_ONLINE_MAX_ITER, CJM2_INNER_TOL
        )
        weights[t] = s_prev
    return weights


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
            f"CJM2 loysi vain {n_active} aktiivista statea, tarvitaan {K}."
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


def select_lambda(train_df: pd.DataFrame, features: list):
    """Pick lambda via an 80/20 split within the current training window.

    Estimation subwindow = first 80% (used to fit the CJM2 and build the regime
    map). Validation subwindow = last 20%, classified causally with the online
    per-step QP solver. Score: validation Sharpe of a 1-day-delayed strategy
    whose exposure is the continuous risk-on weight (1 - Bear_Prob). Ties on
    Sharpe (within 0.05) are broken by lower total absolute weight movement.
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
            fit = fit_cjm2(X_est, N_COMPONENTS, lam)
            regime_map_est, _, _ = make_regime_map(fit["soft_weights"], est_df, features)
        except Exception:
            results.append(
                {
                    "Lambda": lam,
                    "Sharpe": float("nan"),
                    "MeanReturn": float("nan"),
                    "Volatility": float("nan"),
                    "WeightMovement": float("nan"),
                    "Valid": False,
                }
            )
            continue

        prev_weights = fit["soft_weights"][-1]
        val_weights_raw = classify_online_cjm2(
            X_val, fit["centroids"], lam, CJM2_SMOOTH_EPS, prev_weights
        )

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

        weight_movement = float(
            np.abs(val_weights_raw[1:] - val_weights_raw[:-1]).sum()
        )
        results.append(
            {
                "Lambda": lam,
                "Sharpe": sharpe,
                "MeanReturn": mean_ret,
                "Volatility": vol,
                "WeightMovement": weight_movement,
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
    selected = near_best.sort_values(["WeightMovement", "Lambda"]).iloc[0]
    return float(selected["Lambda"]), results_df


def fit_predict_month_end_regime(train_df: pd.DataFrame, features: list):
    # StandardScaler is fit ONLY on this month's training window — no look-ahead.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_df[features].values)

    if JM_TUNE_LAMBDA:
        selected_lambda, lambda_diag = select_lambda(train_df, features)
    else:
        selected_lambda = JM_LAMBDA
        lambda_diag = None

    fit = fit_cjm2(X_scaled, N_COMPONENTS, selected_lambda)

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
        "CJM2_Regime": last_regime,
        "LastRawState": last_raw_state,
        "SelectedLambda": float(selected_lambda),
        "SmoothEps": float(CJM2_SMOOTH_EPS),
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


def select_train_window(cjm_df: pd.DataFrame, source_date: pd.Timestamp) -> pd.DataFrame:
    full_train_df = cjm_df.loc[:source_date].copy()
    if WINDOW_MODE == "expanding":
        return full_train_df
    if WINDOW_MODE == "rolling":
        return full_train_df.tail(ROLLING_WINDOW_OBS).copy()
    raise ValueError(
        f"Tuntematon WINDOW_MODE={WINDOW_MODE!r}. Kayta arvoa 'expanding' tai 'rolling'."
    )


def build_no_lookahead_monthly_regimes(cjm_df: pd.DataFrame, features: list):
    month_end_obs = get_month_end_observation_dates(cjm_df)
    rows = []
    lambda_diagnostics = []

    total_month_ends = len(month_end_obs)
    month_end_pairs = month_end_obs[["Date", "SourceDate"]].itertuples(index=False, name=None)

    for position, (month_end_raw, source_date_raw) in enumerate(month_end_pairs, start=1):
        month_end = pd.Timestamp(month_end_raw)
        source_date = pd.Timestamp(source_date_raw)
        train_df = select_train_window(cjm_df, source_date)

        if len(train_df) < MIN_TRAIN_OBS:
            continue

        try:
            pred, lambda_diag = fit_predict_month_end_regime(train_df, features)
        except Exception as exc:
            print(
                f"CJM2 fit epaonnistui kuukaudelle {month_end.date()} "
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
                f"No-lookahead CJM2: {position}/{total_month_ends} month-endia kasitelty "
                f"(viimeisin {month_end.strftime('%Y-%m')})"
            )

    monthly = pd.DataFrame(rows)
    if monthly.empty:
        return monthly, pd.DataFrame()

    monthly["RegimeLabel"] = monthly["CJM2_Regime"].map(REGIME_LABELS)
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
    cjm_df: pd.DataFrame, stoxx_price: pd.Series, monthly: pd.DataFrame, output_path: Path
):
    monthly_plot = monthly.set_index("SourceDate").sort_index()
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    series_specs = [
        ("STOXX 600", stoxx_price, axes[0]),
        ("VSTOXX", cjm_df["vstoxx"], axes[1]),
        ("Term Spread (10Y-2Y)", cjm_df["term_spread"], axes[2]),
        ("Sovereign Spread (All-AAA)", cjm_df["credit_spread"], axes[3]),
    ]

    for label, full_series, axis in series_specs:
        axis.plot(full_series.index, full_series.values, color="lightgray", linewidth=0.9)
        for regime_id, color in REGIME_COLORS.items():
            mask = monthly_plot["CJM2_Regime"] == regime_id
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
    axes[0].set_title(
        "No-Lookahead Continuous Statistical Jump Model (Genuine QP Solve) Month-End Regimes"
    )
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
        "No-Lookahead Continuous Statistical Jump Model (Genuine QP Solve) "
        "Month-End Regime Probabilities"
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
    if CJM2_SMOOTH_EPS <= 0:
        raise ValueError("CJM2_SMOOTH_EPS tulee olla > 0.")
    if CJM2_INNER_MAX_ITER < 1:
        raise ValueError("CJM2_INNER_MAX_ITER tulee olla vahintaan 1.")
    if CJM2_ONLINE_MAX_ITER < 1:
        raise ValueError("CJM2_ONLINE_MAX_ITER tulee olla vahintaan 1.")

    print(
        "Building no-lookahead Continuous Statistical Jump Model (genuine QP solve) "
        f"monthly regimes with {WINDOW_MODE} window from {START_DATE} to {END_DATE} "
        f"(min train obs={MIN_TRAIN_OBS}, state-mean window={STATE_MEAN_WINDOW_DAYS}d, "
        f"K={N_COMPONENTS}, n_init={JM_N_INIT}, tune_lambda={JM_TUNE_LAMBDA}, "
        f"smooth_eps={CJM2_SMOOTH_EPS}, inner_max_iter={CJM2_INNER_MAX_ITER})"
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
        cjm_df, features, stoxx_price = build_feature_matrix(df_equity, df_yields, df_credit)

        print(f"Feature matrix: {len(cjm_df)} daily observations")
        print(f"Date range: {cjm_df.index.min().date()} — {cjm_df.index.max().date()}")

        monthly, lambda_df = build_no_lookahead_monthly_regimes(cjm_df, features)
        if monthly.empty:
            raise RuntimeError("No-lookahead monthly regime table jaa tyhjaksi.")

        monthly_ml = build_ml_safe_monthly_regimes(monthly)

        monthly.to_csv(MONTHLY_OUTPUT_PATH, index=False)
        monthly_ml.to_csv(MONTHLY_ML_OUTPUT_PATH, index=False)
        plot_month_end_feature_regimes(cjm_df, stoxx_price, monthly, FEATURE_PLOT_PATH)
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
            monthly["CJM2_Regime"]
            .value_counts()
            .rename({0: "Bull (0)", 1: "Bear (1)", 2: "Transition (2)"})
            .sort_index()
        )
        # Share of months whose probability vector is genuinely soft (not pinned
        # to a simplex corner). With the continuous QP solve a "hard" month is
        # one where the max probability is essentially 1.
        probs = monthly[["Bull_Prob", "Bear_Prob", "Transition_Prob"]].values
        is_soft = probs.max(axis=1) < 0.999
        print(
            f"\nSoft (non-corner) probability months: {int(is_soft.sum())}/{len(monthly)} "
            f"({100 * is_soft.mean():.1f}%)"
        )
        print("\nHead:")
        print(monthly.head(12).to_string(index=False))
    finally:
        rd.close_session()


if __name__ == "__main__":
    main()
