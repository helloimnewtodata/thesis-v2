#
# ============================================================
# STATISTICAL JUMP MODEL VS. GAUSSIAN HMM
# ============================================================
#
# Tama on rinnakkainen toteutus HMM.py:lle. Latentin regiimin paatteleva malli on
# Gaussian HMM:n sijaan Statistical Jump Model (Bemporad et al. 2018; Nystrup et al. 2020).
#
# 1. Sama datavirta ja samat featuret
# Sama lahdedata (STOXX 600, VSTOXX, Saksan 2Y/10Y, ECB credit spread) ja samat featuret
# (stoxx_return, vstoxx, term_spread, credit_spread) kuin HMM.py:ssa.
#
# 2. Globaali tavoitefunktio
# Jump Model maarittaa centroideja mu_k ja state-sekvenssin s_t minimoimalla:
#     sum_t ||x_t - mu_{s_t}||^2 + lambda * sum_{t>0} 1{s_t != s_{t-1}}
# Sisempi optimointi on Viterbi-tyylinen dynaaminen ohjelmointi: globaalisti optimaalinen
# state-sekvenssi annetuilla centroideilla.
#
# 3. Jump penalty lambda
# Korkeampi lambda → harvempia regiimin vaihtoja (pidemmat regiimit). JM_TUNE_LAMBDA=true
# valitsee lambdan validointi-Sharpen perusteella; oletuksena kaytetaan kiintea JM_LAMBDA
# kuukausittaisen refitin nopeuttamiseksi.
#
# 4. Ei posterior-todennakoisyyksia
# Jump Modelista ei tule Bayes-tyyppisia state-todennakoisyyksia. Tallennetaan hard-state
# indikaattorit (1/0) ja viimeisen STATE_MEAN_WINDOW_DAYS:n hard-state frekvenssit. Naita
# ei saa kasitella todennakoisyyksina, vaikka ne osuvatkin valille [0, 1].
#
# 5. State-labelointi
# Sama economic-score -labelointi kuin HMM.py:ssa: state-keskiarvojen z-score ja
# tuotto/vola/credit/term-spread -pisteytys. Labelointi tehdaan vain training-window'n
# tiedoilla, jotta tulevia tuottoja ei vuoda labelointiin.
#
# 6. No-lookahead skaalaus ja kuukausittainen refit
# StandardScaler fitataan vain kunkin source daten kayttotreniwindow'n perusteella.
# Kuukausittainen refit toimii samalla SourceDate→MonthEnd-rakenteella kuin HMM.py:ssa.
#
# 7. ML-safe shift
# Sama signal month-end → next target month -siirto kuin HMM.py:ssa: estaa same-period
# leakagen kun JM-featuret yhdistetaan seuraavan kuukauden tuottoihin.
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
MONTHLY_OUTPUT_PATH = OUTPUT_DIR / "JM_output.csv"
MONTHLY_ML_OUTPUT_PATH = OUTPUT_DIR / "JM_output_ml.csv"
FEATURE_PLOT_PATH = OUTPUT_DIR / "JM_output_features.png"
STATE_FREQUENCY_PLOT_PATH = OUTPUT_DIR / "JM_output_state_frequencies.png"
LAMBDA_VALIDATION_PATH = OUTPUT_DIR / "JM_output_lambda_validation.csv"

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
    "JM_Regime",
    "RegimeLabel",
    "LastRawState",
    "SelectedLambda",
    "JMObjective",
    "ModelConverged",
    "JMIterations",
    "Bull_StateIndicator",
    "Bear_StateIndicator",
    "Transition_StateIndicator",
    "Bull_StateMeanWindow",
    "Bear_StateMeanWindow",
    "Transition_StateMeanWindow",
    "StateMeanWindowDaysUsed",
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


def jump_model_dynamic_programming(loss_matrix: np.ndarray, lambda_penalty: float):
    """
    Globally optimal Viterbi-style decoder for the JM objective given fixed centroids.

    With loss_matrix[t, k] = ||x_t - mu_k||^2, solves:
        C[0, k] = loss_matrix[0, k]
        C[t, k] = loss_matrix[t, k] + min_j { C[t-1, j] + lambda * 1{j != k} }
    and reconstructs the optimal state path via backpointers.
    """
    T, K = loss_matrix.shape
    cost = np.empty((T, K), dtype=np.float64)
    backpointer = np.zeros((T, K), dtype=np.int32)

    cost[0] = loss_matrix[0]
    penalty = lambda_penalty * (1.0 - np.eye(K, dtype=np.float64))

    for t in range(1, T):
        # transition_cost[j, k] = C[t-1, j] + lambda * 1{j != k}
        transition_cost = cost[t - 1][:, None] + penalty
        best_j = transition_cost.argmin(axis=0)
        min_cost = transition_cost[best_j, np.arange(K)]
        cost[t] = loss_matrix[t] + min_cost
        backpointer[t] = best_j

    states = np.empty(T, dtype=np.int32)
    states[T - 1] = int(cost[T - 1].argmin())
    objective = float(cost[T - 1].min())
    for t in range(T - 2, -1, -1):
        states[t] = backpointer[t + 1, states[t + 1]]

    return states, objective


def compute_jump_objective(
    X: np.ndarray, states: np.ndarray, centroids: np.ndarray, lambda_penalty: float
) -> float:
    diff = X - centroids[states]
    fit_cost = float(np.sum(diff * diff))
    jumps = int(np.sum(states[1:] != states[:-1]))
    return fit_cost + lambda_penalty * jumps


def _update_centroids(
    X: np.ndarray, states: np.ndarray, K: int, rng: np.random.Generator
) -> np.ndarray:
    """Update centroids; reinitialize robustly if a state has no members."""
    centroids = np.empty((K, X.shape[1]), dtype=np.float64)
    for k in range(K):
        mask = states == k
        if mask.any():
            centroids[k] = X[mask].mean(axis=0)
        else:
            # Empty state — reseed with a random training observation to avoid degeneracy
            idx = int(rng.integers(0, X.shape[0]))
            centroids[k] = X[idx]
    return centroids


def _fit_jump_model_single_init(
    X: np.ndarray,
    K: int,
    lambda_penalty: float,
    max_iter: int,
    tol: float,
    random_state: int,
):
    rng = np.random.default_rng(random_state)

    # KMeans seeds the initial state assignment (per-step centroid update is identical to
    # KMeans' M-step, so this is a natural warm start).
    km = KMeans(n_clusters=K, n_init=1, random_state=random_state)
    km.fit(X)
    states = km.labels_.astype(np.int32)

    centroids = _update_centroids(X, states, K, rng)
    prev_objective = np.inf
    objective = np.inf
    converged = False
    iteration_count = 0

    for iteration in range(max_iter):
        iteration_count = iteration + 1

        # 1) Centroid update given current states (closed-form: cluster means)
        centroids = _update_centroids(X, states, K, rng)

        # 2) State sequence update via DP given centroids (globally optimal in s)
        loss = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_states, objective = jump_model_dynamic_programming(loss, lambda_penalty)

        if abs(prev_objective - objective) < tol:
            states = new_states
            converged = True
            break

        states = new_states
        prev_objective = objective

    return {
        "states": states.astype(np.int32),
        "centroids": centroids,
        "objective": float(objective),
        "iterations": int(iteration_count),
        "converged": bool(converged),
    }


def fit_jump_model(
    X: np.ndarray,
    K: int,
    lambda_penalty: float,
    max_iter: int = None,
    tol: float = None,
    n_init: int = None,
    random_state: int = None,
):
    """Fit Jump Model with multiple random restarts; keep the lowest-objective solution."""
    max_iter = JM_MAX_ITER if max_iter is None else max_iter
    tol = JM_TOL if tol is None else tol
    n_init = JM_N_INIT if n_init is None else n_init
    random_state = JM_RANDOM_STATE if random_state is None else random_state

    seed_rng = np.random.default_rng(random_state)
    seeds = seed_rng.integers(0, 10**9, size=n_init)

    best = None
    for seed in seeds:
        result = _fit_jump_model_single_init(
            X, K, lambda_penalty, max_iter, tol, int(seed)
        )
        if best is None or result["objective"] < best["objective"]:
            best = result
    return best


def classify_online_jump_model(
    X_new: np.ndarray, centroids: np.ndarray, lambda_penalty: float, prev_state: int
) -> np.ndarray:
    """
    Causal online classifier: for each new observation t, assign
        s_t = argmin_k { ||x_t - mu_k||^2 + lambda * 1{s_{t-1} != k} }
    using only information up to t and the previously assigned state. Used for the
    validation half of the lambda tuning split — strictly no peeking into future obs.
    """
    K = centroids.shape[0]
    states = np.empty(X_new.shape[0], dtype=np.int32)
    current_prev = int(prev_state)
    for t in range(X_new.shape[0]):
        fit_cost = np.sum((centroids - X_new[t]) ** 2, axis=1)
        switch_cost = lambda_penalty * (np.arange(K) != current_prev).astype(np.float64)
        total = fit_cost + switch_cost
        best_k = int(total.argmin())
        states[t] = best_k
        current_prev = best_k
    return states


def make_regime_map(raw_states: np.ndarray, train_df: pd.DataFrame):
    """Map raw JM state ids to economic regimes (0=Bull, 1=Bear, 2=Transition).

    Uses only training-window state-conditional means, so no future information leaks
    into labelling. Raises if fewer than N_COMPONENTS distinct states were actually
    used by the DP (the caller skips that month).
    """
    state_means = (
        train_df.assign(raw_state=raw_states)
        .groupby("raw_state")[["stoxx_return", "vstoxx", "term_spread", "credit_spread"]]
        .mean()
    )

    if len(state_means) < N_COMPONENTS:
        raise ValueError(
            f"JM loysi vain {len(state_means)} statea kayttoon, tarvitaan {N_COMPONENTS}."
        )

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


def select_lambda(train_df: pd.DataFrame, features: list):
    """Pick lambda via an 80/20 split within the current training window.

    Estimation subwindow = first 80% (used both to fit JM and to build the regime map).
    Validation subwindow = last 20%, classified causally with the online classifier so
    no future info touches state assignments. Score: validation Sharpe of a simple
    1-day-delayed risk-on/risk-off strategy (risk-on when not Bear). Ties on Sharpe
    (within 0.05) are broken by lower switch count.
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
            fit = fit_jump_model(X_est, N_COMPONENTS, lam)
            regime_map_est, _, _ = make_regime_map(fit["states"], est_df)
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

        prev_state = int(fit["states"][-1])
        val_raw_states = classify_online_jump_model(
            X_val, fit["centroids"], lam, prev_state
        )
        # If a raw state never appeared during estimation it has no economic label —
        # default it to Transition (2) so the strategy treats it as risk-on.
        val_mapped = np.array([regime_map_est.get(int(s), 2) for s in val_raw_states])

        positions = (val_mapped != 1).astype(np.float64)
        # 1-day delay: position decided at t-1 trades return at t (no same-bar peeking)
        positions_lagged = np.concatenate([[0.0], positions[:-1]])
        strategy_returns = positions_lagged * val_returns

        mean_ret = float(strategy_returns.mean())
        vol = float(strategy_returns.std(ddof=0))
        if vol == 0 or not np.isfinite(vol):
            sharpe = float("nan")
        else:
            sharpe = float((mean_ret / vol) * np.sqrt(252))

        n_switches = int((val_raw_states[1:] != val_raw_states[:-1]).sum())
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


def fit_predict_month_end_regime(train_df: pd.DataFrame, features: list):
    # StandardScaler is fit ONLY on this month's training window — no full-sample fit,
    # no look-ahead.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_df[features].values)

    if JM_TUNE_LAMBDA:
        selected_lambda, lambda_diag = select_lambda(train_df, features)
    else:
        selected_lambda = JM_LAMBDA
        lambda_diag = None

    fit = fit_jump_model(X_scaled, N_COMPONENTS, selected_lambda)

    raw_states = fit["states"]
    regime_map, state_means, economic_score = make_regime_map(raw_states, train_df)

    last_raw_state = int(raw_states[-1])
    last_regime = int(regime_map[last_raw_state])

    state_indicators = {0: 0, 1: 0, 2: 0}
    state_indicators[last_regime] = 1

    window_days = min(STATE_MEAN_WINDOW_DAYS, len(raw_states))
    window_states = raw_states[-window_days:]
    mapped_window = np.array([regime_map.get(int(s), -1) for s in window_states])
    state_window_freq = {
        0: float((mapped_window == 0).mean()),
        1: float((mapped_window == 1).mean()),
        2: float((mapped_window == 2).mean()),
    }

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
        "JM_Regime": last_regime,
        "LastRawState": last_raw_state,
        "SelectedLambda": float(selected_lambda),
        "JMObjective": float(fit["objective"]),
        "ModelConverged": bool(fit["converged"]),
        "JMIterations": int(fit["iterations"]),
        "Bull_StateIndicator": int(state_indicators[0]),
        "Bear_StateIndicator": int(state_indicators[1]),
        "Transition_StateIndicator": int(state_indicators[2]),
        "Bull_StateMeanWindow": state_window_freq[0],
        "Bear_StateMeanWindow": state_window_freq[1],
        "Transition_StateMeanWindow": state_window_freq[2],
        "StateMeanWindowDaysUsed": int(window_days),
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


def build_no_lookahead_monthly_regimes(jm_df: pd.DataFrame, features: list):
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
            pred, lambda_diag = fit_predict_month_end_regime(train_df, features)
        except Exception as exc:
            print(
                f"JM fit epaonnistui kuukaudelle {month_end.date()} "
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
                f"No-lookahead JM: {position}/{total_month_ends} month-endia kasitelty "
                f"(viimeisin {month_end.strftime('%Y-%m')})"
            )

    monthly = pd.DataFrame(rows)
    if monthly.empty:
        return monthly, pd.DataFrame()

    monthly["RegimeLabel"] = monthly["JM_Regime"].map(REGIME_LABELS)
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
            mask = monthly_plot["JM_Regime"] == regime_id
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
    axes[0].set_title("No-Lookahead Jump Model Month-End Regimes")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_state_frequencies(monthly: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(
        monthly["Date"],
        monthly["Bull_StateMeanWindow"],
        color=REGIME_COLORS[0],
        linewidth=1.7,
        label="Bull_StateMeanWindow",
    )
    ax.plot(
        monthly["Date"],
        monthly["Bear_StateMeanWindow"],
        color=REGIME_COLORS[1],
        linewidth=1.7,
        label="Bear_StateMeanWindow",
    )
    ax.plot(
        monthly["Date"],
        monthly["Transition_StateMeanWindow"],
        color=REGIME_COLORS[2],
        linewidth=1.7,
        label="Transition_StateMeanWindow",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Hard-state frequency")
    ax.set_xlabel("Date")
    ax.set_title(
        "No-Lookahead Jump Model Month-End Hard-State Frequencies "
        f"({STATE_MEAN_WINDOW_DAYS}D rolling — NOT posterior probabilities)"
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

    print(
        "Building no-lookahead Jump Model monthly regimes "
        f"with {WINDOW_MODE} window from {START_DATE} to {END_DATE} "
        f"(min train obs={MIN_TRAIN_OBS}, state-mean window={STATE_MEAN_WINDOW_DAYS}d, "
        f"K={N_COMPONENTS}, n_init={JM_N_INIT}, tune_lambda={JM_TUNE_LAMBDA})"
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

        monthly, lambda_df = build_no_lookahead_monthly_regimes(jm_df, features)
        if monthly.empty:
            raise RuntimeError("No-lookahead monthly regime table jaa tyhjaksi.")

        monthly_ml = build_ml_safe_monthly_regimes(monthly)

        monthly.to_csv(MONTHLY_OUTPUT_PATH, index=False)
        monthly_ml.to_csv(MONTHLY_ML_OUTPUT_PATH, index=False)
        plot_month_end_feature_regimes(jm_df, stoxx_price, monthly, FEATURE_PLOT_PATH)
        plot_monthly_state_frequencies(monthly, STATE_FREQUENCY_PLOT_PATH)

        if JM_TUNE_LAMBDA and not lambda_df.empty:
            lambda_df.to_csv(LAMBDA_VALIDATION_PATH, index=False)

        print(f"\nSaved: {MONTHLY_OUTPUT_PATH}")
        print(f"Saved: {MONTHLY_ML_OUTPUT_PATH}")
        print(f"Saved: {FEATURE_PLOT_PATH}")
        print(f"Saved: {STATE_FREQUENCY_PLOT_PATH}")
        if JM_TUNE_LAMBDA and not lambda_df.empty:
            print(f"Saved: {LAMBDA_VALIDATION_PATH}")
        print(f"Rows: {len(monthly)}")
        print(
            f"Date range: {monthly['Date'].min().strftime('%Y-%m')} — "
            f"{monthly['Date'].max().strftime('%Y-%m')}"
        )
        print("\nRegime distribution:")
        print(
            monthly["JM_Regime"]
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
