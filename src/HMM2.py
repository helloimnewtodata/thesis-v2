#
# ============================================================
# HMM2: TIMING-SHARPE-OPTIMOITU GAUSSIAN HMM
# ============================================================
#
# Variantti HMM.py:lle, jonka muutokset on suunnattu yhteen tavoitteeseen:
# parantaa standalone-timing-Sharpea regime_strategy_benchmark_soft-tyylisessa
# 1-P(Bear) -strategiassa. Ei ML-feature-targetointia.
#
# Erot HMM.py:hin (kaikki konfiguroitavissa env-vaareilla):
#
# 1) N_COMPONENTS = 2 (Bull/Bear) oletuksena.
#    K=3:n Transition on tulkinnallisesti epaselva ja aiheuttaa whip-sawta
#    posterioreissa. Shu et al. -tyyppisessa kirjallisuudessa K=2 on stabiilimpi
#    timingiin. K=3 edelleen tuettu env-vaarilla.
#
# 2) Multi-restart fitting (HMM_N_INIT, oletus 10).
#    HMM.py kaytti yhta kiintea random_state=42. Tama voi loytaa huonon
#    paikallisen minimin. Tassa kokeillaan N satunnaista alkutilaa per refit ja
#    valitaan korkein log-likelihood. Tradeoff: ~N-kertainen compute-aika.
#
# 3) Diagonaalinen kovarianssi (HMM_COVARIANCE_TYPE = "diag" oletus).
#    "full" overfittaa pienissa trainingikkunoissa, koska kovaltiis-parametreja
#    on K * D^2/2 versus K * D. "diag" pakottaa featuret riippumattomiksi state:n
#    sisalla, mika on saannolliseempi.
#
# 4) Sileytetyt low-frequency featuret (FEATURE_SMOOTHING_DAYS, oletus 5).
#    vstoxx / term_spread / credit_spread saavat 5-paivan rolling meanin
#    (taakse, ei look-aheadia). Daily return jaa raaaksi koska se on
#    informatiivisin signaali. Vahentaa posterioreita heiluttavaa kohinaa.
#
# 5) Pidempi posterior-keskiarvoikkuna (POSTERIOR_MEAN_WINDOW_DAYS, oletus 20).
#    HMM.py kaytti 5pv. Pidempi ikkuna sammuttaa whip-sawta strategiatasolla
#    ja antaa softer signaalin Bear-todennakoisyydelle.
#
# Output:
# Saa samat sarakkeet kuin HMM.py (HMM_Regime, Bear_Prob, Bear_Prob_MeanWindow,
# jne.), jotta regime_strategy_benchmark_soft drop-in toimii ilman muutoksia.
# K=2-tilassa Transition_Prob == 0 ja Transition-state-mittarit ovat NaN.
#
import os
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import refinitiv.data as rd
from src.ecb_cache_utils import ECB_CACHE_DIR, get_ecb_series
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
from matplotlib import pyplot as plt


warnings.filterwarnings("ignore", category=FutureWarning)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "01_raw" / "outputs"
MONTHLY_OUTPUT_PATH = OUTPUT_DIR / "hmm2_regimes_monthly_no_lookahead.csv"
MONTHLY_ML_OUTPUT_PATH = OUTPUT_DIR / "hmm2_regimes_monthly_no_lookahead_ml.csv"
FEATURE_PLOT_PATH = OUTPUT_DIR / "hmm2_regimes_monthly_no_lookahead_features.png"
PROBABILITY_PLOT_PATH = OUTPUT_DIR / "hmm2_regimes_monthly_no_lookahead_probabilities.png"

START_DATE = os.getenv("START_DATE", "2006-01-01")
END_DATE = os.getenv("END_DATE", "2026-04-30")
N_COMPONENTS = int(os.getenv("N_COMPONENTS", "2"))
MIN_TRAIN_OBS = int(os.getenv("MIN_TRAIN_OBS", "756"))
HMM_MAX_ITER = int(os.getenv("HMM_MAX_ITER", "500"))
HMM_TOL = float(os.getenv("HMM_TOL", "1e-4"))
HMM_RANDOM_STATE = int(os.getenv("HMM_RANDOM_STATE", "42"))
HMM_N_INIT = int(os.getenv("HMM_N_INIT", "10"))
HMM_COVARIANCE_TYPE = os.getenv("HMM_COVARIANCE_TYPE", "diag").strip().lower()
FEATURE_SMOOTHING_DAYS = int(os.getenv("FEATURE_SMOOTHING_DAYS", "5"))
WINDOW_MODE = os.getenv("WINDOW_MODE", "expanding").strip().lower()
ROLLING_WINDOW_OBS = int(os.getenv("ROLLING_WINDOW_OBS", "1260"))
POSTERIOR_MEAN_WINDOW_DAYS = int(os.getenv("POSTERIOR_MEAN_WINDOW_DAYS", "20"))

REGIME_COLORS = {0: "green", 1: "red", 2: "orange"}
REGIME_LABELS = {0: "Bull", 1: "Bear", 2: "Transition"}


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
    hmm_df = pd.DataFrame(index=df_equity.index)

    stoxx_price = df_equity[".STOXX"].astype(float)
    hmm_df["stoxx_return"] = np.log(stoxx_price / stoxx_price.shift(1))
    hmm_df["vstoxx"] = df_equity[".V2TX"].astype(float)
    hmm_df["term_spread"] = (
        df_yields["DE10YT=RR"].astype(float) - df_yields["DE2YT=RR"].astype(float)
    )
    hmm_df = hmm_df.join(df_credit, how="left")
    hmm_df["credit_spread"] = hmm_df["credit_spread"].ffill()

    # HMM2: sileyta low-frequency featuret takautuvalla rolling meanilla. EI
    # look-aheadia koska rolling(window).mean() kayttaa vain menneita havaintoja.
    # Daily return pidetaan raaaksi - se on informatiivisin Bear-signaali ja sen
    # sileytys lisaisi viivetta liikaa.
    if FEATURE_SMOOTHING_DAYS > 1:
        for col in ("vstoxx", "term_spread", "credit_spread"):
            hmm_df[col] = hmm_df[col].rolling(FEATURE_SMOOTHING_DAYS, min_periods=1).mean()

    features = ["stoxx_return", "vstoxx", "term_spread", "credit_spread"]
    hmm_df = hmm_df[features].dropna().sort_index()
    stoxx_price = stoxx_price.reindex(hmm_df.index)
    return hmm_df, features, stoxx_price


def get_month_end_observation_dates(hmm_df):
    source_dates = pd.DatetimeIndex(hmm_df.index)
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


def make_regime_map(raw_states, train_df):
    """Map raw HMM2 state ids to economic regimes using training-window means.

    K=2: highest economic score -> Bull (0), lowest -> Bear (1).
    K=3: middle -> Transition (2). No look-ahead: uses only training data means.
    """
    state_means = (
        train_df.assign(raw_state=raw_states)
        .groupby("raw_state")[["stoxx_return", "vstoxx", "term_spread", "credit_spread"]]
        .mean()
    )

    if len(state_means) < N_COMPONENTS:
        raise ValueError(
            f"HMM2 loysi vain {len(state_means)} statea, tarvitaan {N_COMPONENTS}."
        )

    standardized = state_means.apply(zscore_series, axis=0)
    economic_score = (
        standardized["stoxx_return"]
        - standardized["vstoxx"]
        - standardized["credit_spread"]
        + 0.5 * standardized["term_spread"]
    )

    sorted_states = economic_score.sort_values(ascending=False).index.tolist()

    if N_COMPONENTS == 2:
        regime_map = {
            sorted_states[0]: 0,
            sorted_states[-1]: 1,
        }
    elif N_COMPONENTS == 3:
        regime_map = {
            sorted_states[0]: 0,
            sorted_states[-1]: 1,
            sorted_states[1]: 2,
        }
    else:
        raise ValueError(f"N_COMPONENTS={N_COMPONENTS} ei tueta; kayta 2 tai 3.")

    return regime_map, state_means, economic_score


def map_raw_posterior_to_regime_probabilities(raw_posterior, regime_map):
    posterior_by_regime = {0: 0.0, 1: 0.0, 2: 0.0}
    for raw_state, prob in enumerate(raw_posterior):
        regime = regime_map.get(raw_state)
        if regime is not None:
            posterior_by_regime[regime] = float(prob)
    return posterior_by_regime


def fit_best_hmm(X_scaled):
    """Multi-restart HMM fit; keep the highest log-likelihood model."""
    seed_rng = np.random.default_rng(HMM_RANDOM_STATE)
    seeds = seed_rng.integers(0, 10**9, size=HMM_N_INIT)

    best_model = None
    best_score = -np.inf
    fit_failures = 0

    for seed in seeds:
        try:
            candidate = GaussianHMM(
                n_components=N_COMPONENTS,
                covariance_type=HMM_COVARIANCE_TYPE,
                n_iter=HMM_MAX_ITER,
                random_state=int(seed),
                tol=HMM_TOL,
            )
            candidate.fit(X_scaled)
            score = candidate.score(X_scaled)
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_model = candidate
        except Exception:
            fit_failures += 1
            continue

    if best_model is None:
        raise RuntimeError(
            f"All {HMM_N_INIT} HMM2 restarts failed ({fit_failures} exceptions)."
        )
    return best_model, fit_failures


def fit_predict_month_end_regime(train_df, features):
    # StandardScaler is fit ONLY on this month's training window — no look-ahead
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_df[features].values)

    model, fit_failures = fit_best_hmm(X_scaled)

    raw_states = model.predict(X_scaled)
    regime_map, state_means, economic_score = make_regime_map(raw_states, train_df)

    last_raw_state = int(raw_states[-1])
    last_regime = int(regime_map[last_raw_state])

    bull_state = next((k for k, v in regime_map.items() if v == 0), None)
    bear_state = next((k for k, v in regime_map.items() if v == 1), None)
    transition_state = next((k for k, v in regime_map.items() if v == 2), None)

    posterior_all_raw = model.predict_proba(X_scaled)
    posterior_last_raw = posterior_all_raw[-1]
    posterior_by_regime = map_raw_posterior_to_regime_probabilities(posterior_last_raw, regime_map)

    posterior_window_days = min(POSTERIOR_MEAN_WINDOW_DAYS, len(posterior_all_raw))
    posterior_mean_raw = posterior_all_raw[-posterior_window_days:].mean(axis=0)
    posterior_mean_by_regime = map_raw_posterior_to_regime_probabilities(
        posterior_mean_raw,
        regime_map,
    )

    def state_return(state_key):
        if state_key is None or state_key not in state_means.index:
            return float("nan")
        return float(state_means.loc[state_key, "stoxx_return"])

    def state_score(state_key):
        if state_key is None or state_key not in economic_score.index:
            return float("nan")
        return float(economic_score.loc[state_key])

    return {
        "HMM_Regime": last_regime,
        "Bull_Prob": posterior_by_regime[0],
        "Bear_Prob": posterior_by_regime[1],
        "Transition_Prob": posterior_by_regime[2],
        "Bull_Prob_MeanWindow": posterior_mean_by_regime[0],
        "Bear_Prob_MeanWindow": posterior_mean_by_regime[1],
        "Transition_Prob_MeanWindow": posterior_mean_by_regime[2],
        "PosteriorMeanWindowDaysUsed": posterior_window_days,
        "LastRawState": last_raw_state,
        "ModelConverged": bool(model.monitor_.converged),
        "TrainLogLikelihood": float(model.score(X_scaled)),
        "MultiRestartFailures": int(fit_failures),
        "NComponents": int(N_COMPONENTS),
        "CovarianceType": HMM_COVARIANCE_TYPE,
        "FeatureSmoothingDays": int(FEATURE_SMOOTHING_DAYS),
        "BullStateTrainMeanReturn": state_return(bull_state),
        "BearStateTrainMeanReturn": state_return(bear_state),
        "TransitionStateTrainMeanReturn": state_return(transition_state),
        "BullStateEconomicScore": state_score(bull_state),
        "BearStateEconomicScore": state_score(bear_state),
        "TransitionStateEconomicScore": state_score(transition_state),
    }


def select_train_window(hmm_df, source_date):
    full_train_df = hmm_df.loc[:source_date].copy()
    if WINDOW_MODE == "expanding":
        return full_train_df
    if WINDOW_MODE == "rolling":
        return full_train_df.tail(ROLLING_WINDOW_OBS).copy()
    raise ValueError(
        f"Tuntematon WINDOW_MODE={WINDOW_MODE!r}. Kayta arvoa 'expanding' tai 'rolling'."
    )


def build_no_lookahead_monthly_regimes(hmm_df, features):
    month_end_obs = get_month_end_observation_dates(hmm_df)
    rows = []

    total_month_ends = len(month_end_obs)
    month_end_pairs = month_end_obs[["Date", "SourceDate"]].itertuples(index=False, name=None)

    for position, (month_end_raw, source_date_raw) in enumerate(month_end_pairs, start=1):
        month_end = pd.Timestamp(month_end_raw)
        source_date = pd.Timestamp(source_date_raw)
        train_df = select_train_window(hmm_df, source_date)

        if len(train_df) < MIN_TRAIN_OBS:
            continue

        try:
            pred = fit_predict_month_end_regime(train_df, features)
        except Exception as exc:
            print(
                f"HMM2 fit epaonnistui kuukaudelle {month_end.date()} "
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

        if position % 12 == 0 or position == total_month_ends:
            print(
                f"No-lookahead HMM2: {position}/{total_month_ends} month-endia kasitelty "
                f"(viimeisin {month_end.strftime('%Y-%m')})"
            )

    monthly = pd.DataFrame(rows)
    if monthly.empty:
        return monthly

    monthly["RegimeLabel"] = monthly["HMM_Regime"].map(REGIME_LABELS)
    monthly["NextMonthDate"] = monthly["Date"] + pd.offsets.MonthEnd(1)
    monthly = monthly.sort_values("Date").reset_index(drop=True)
    return monthly


def build_ml_safe_monthly_regimes(monthly):
    ml = monthly.copy()
    ml = ml.rename(columns={"Date": "SignalMonthEnd", "SourceDate": "SignalSourceDate"})
    ml["Date"] = ml["SignalMonthEnd"] + pd.offsets.MonthEnd(1)
    ml["TargetMonth"] = ml["Date"].dt.to_period("M").astype(str)

    lead_cols = ["Date", "TargetMonth", "SignalMonthEnd", "SignalSourceDate"]
    other_cols = [col for col in ml.columns if col not in lead_cols and col != "NextMonthDate"]
    return ml[lead_cols + other_cols].sort_values("Date").reset_index(drop=True)


def plot_month_end_feature_regimes(hmm_df, stoxx_price, monthly, output_path: Path):
    monthly_plot = monthly.set_index("SourceDate").sort_index()
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    series_specs = [
        ("STOXX 600", stoxx_price, axes[0]),
        (f"VSTOXX (smoothed {FEATURE_SMOOTHING_DAYS}d)", hmm_df["vstoxx"], axes[1]),
        (f"Term Spread (smoothed {FEATURE_SMOOTHING_DAYS}d)", hmm_df["term_spread"], axes[2]),
        (f"Credit Spread (smoothed {FEATURE_SMOOTHING_DAYS}d)", hmm_df["credit_spread"], axes[3]),
    ]

    for label, full_series, axis in series_specs:
        axis.plot(full_series.index, full_series.values, color="lightgray", linewidth=0.9)
        for regime_id, color in REGIME_COLORS.items():
            mask = monthly_plot["HMM_Regime"] == regime_id
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
        f"HMM2 Month-End Regimes (K={N_COMPONENTS}, cov={HMM_COVARIANCE_TYPE}, "
        f"n_init={HMM_N_INIT}, post window={POSTERIOR_MEAN_WINDOW_DAYS}d)"
    )
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_regime_probabilities(monthly, output_path: Path):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(
        monthly["Date"],
        monthly["Bull_Prob_MeanWindow"],
        color=REGIME_COLORS[0],
        linewidth=1.7,
        label="Bull_Prob_MeanWindow",
    )
    ax.plot(
        monthly["Date"],
        monthly["Bear_Prob_MeanWindow"],
        color=REGIME_COLORS[1],
        linewidth=1.7,
        label="Bear_Prob_MeanWindow",
    )
    if N_COMPONENTS >= 3:
        ax.plot(
            monthly["Date"],
            monthly["Transition_Prob_MeanWindow"],
            color=REGIME_COLORS[2],
            linewidth=1.7,
            label="Transition_Prob_MeanWindow",
        )
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Date")
    ax.set_title(
        f"HMM2 Month-End Regime Probabilities ({POSTERIOR_MEAN_WINDOW_DAYS}D Mean, K={N_COMPONENTS})"
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
    if POSTERIOR_MEAN_WINDOW_DAYS < 1:
        raise ValueError("POSTERIOR_MEAN_WINDOW_DAYS tulee olla vahintaan 1.")
    if FEATURE_SMOOTHING_DAYS < 1:
        raise ValueError("FEATURE_SMOOTHING_DAYS tulee olla vahintaan 1.")
    if WINDOW_MODE == "rolling" and ROLLING_WINDOW_OBS < MIN_TRAIN_OBS:
        raise ValueError("ROLLING_WINDOW_OBS tulee olla >= MIN_TRAIN_OBS rolling-tilassa.")
    if N_COMPONENTS not in {2, 3}:
        raise ValueError(f"N_COMPONENTS={N_COMPONENTS} ei tueta; kayta 2 tai 3.")
    if HMM_N_INIT < 1:
        raise ValueError("HMM_N_INIT tulee olla vahintaan 1.")
    if HMM_COVARIANCE_TYPE not in {"diag", "full", "tied", "spherical"}:
        raise ValueError(
            f"HMM_COVARIANCE_TYPE={HMM_COVARIANCE_TYPE!r} ei tueta. "
            "Sallitut: diag, full, tied, spherical."
        )

    print(
        "Building no-lookahead HMM2 monthly regimes "
        f"with {WINDOW_MODE} window from {START_DATE} to {END_DATE} "
        f"(K={N_COMPONENTS}, cov={HMM_COVARIANCE_TYPE}, n_init={HMM_N_INIT}, "
        f"feature smoothing={FEATURE_SMOOTHING_DAYS}d, "
        f"posterior mean window={POSTERIOR_MEAN_WINDOW_DAYS}d, "
        f"min train obs={MIN_TRAIN_OBS})"
    )
    if WINDOW_MODE == "rolling":
        print(f"Rolling window observations: {ROLLING_WINDOW_OBS}")

    rd.open_session()
    try:
        df_equity, df_yields, df_credit = fetch_input_data(START_DATE, END_DATE)
        hmm_df, features, stoxx_price = build_feature_matrix(df_equity, df_yields, df_credit)

        print(f"Feature matrix: {len(hmm_df)} daily observations")
        print(f"Date range: {hmm_df.index.min().date()} — {hmm_df.index.max().date()}")

        monthly = build_no_lookahead_monthly_regimes(hmm_df, features)
        if monthly.empty:
            raise RuntimeError("No-lookahead monthly regime table jaa tyhjaksi.")

        monthly_ml = build_ml_safe_monthly_regimes(monthly)

        monthly.to_csv(MONTHLY_OUTPUT_PATH, index=False)
        monthly_ml.to_csv(MONTHLY_ML_OUTPUT_PATH, index=False)
        plot_month_end_feature_regimes(hmm_df, stoxx_price, monthly, FEATURE_PLOT_PATH)
        plot_monthly_regime_probabilities(monthly, PROBABILITY_PLOT_PATH)

        print(f"\nSaved: {MONTHLY_OUTPUT_PATH}")
        print(f"Saved: {MONTHLY_ML_OUTPUT_PATH}")
        print(f"Saved: {FEATURE_PLOT_PATH}")
        print(f"Saved: {PROBABILITY_PLOT_PATH}")
        print(f"Rows: {len(monthly)}")
        print(
            f"Date range: {monthly['Date'].min().strftime('%Y-%m')} — "
            f"{monthly['Date'].max().strftime('%Y-%m')}"
        )
        print("\nRegime distribution:")
        print(
            monthly["HMM_Regime"]
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
