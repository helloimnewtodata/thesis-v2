# ============================================================
# MSR-BENCHMARK HMM:lle (Hamilton 1989 Markov-Switching Regression)
# ============================================================
#
# Tama skripti peilaa src/HMM.py:n rakennetta tarkalleen, mutta vaihtaa
# GaussianHMM:n tilalle statsmodels.tsa.regime_switching.MarkovRegression:n
# (Hamilton 1989). Tarkoitus on tarjota akateemisesti vahvin suora
# haastaja HMM:lle samassa diskreetin latenttitilan kehikossa.
#
# Erot HMM:aan:
#   - Univariate: malli sovitetaan vain STOXX-tuottosarjaan. Tama on
#     kanoninen Hamilton-asetelma ja vahvistaa tutkimuskysymyksen:
#     riittaako tuottosarja regiimien tunnistamiseen, vai onko HMM:n
#     monimuuttujainen latentti-tila lisaarvoinen?
#   - Regime-riippuva keskiarvo + varianssi (switching_variance=True).
#   - Relabelointi taloudellisen skoorin mukaan kayttaa _kaikkia_
#     4 inputtia (stoxx_return, vstoxx, term_spread, credit_spread)
#     paivakohtaiseen tila-assignmenttiin perustuvana keskiarvona,
#     jotta lopullinen Bull/Bear/Transition-mappi on suoraan
#     vertailukelpoinen HMM:n outputin kanssa.
#
import os
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import refinitiv.data as rd
from src.ecb_cache_utils import ECB_CACHE_DIR, get_ecb_series
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


matplotlib.use("Agg")
from matplotlib import pyplot as plt


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "01_raw" / "outputs"
MONTHLY_OUTPUT_PATH = OUTPUT_DIR / "msr_regimes_monthly_no_lookahead.csv"
MONTHLY_ML_OUTPUT_PATH = OUTPUT_DIR / "msr_regimes_monthly_no_lookahead_ml.csv"
FEATURE_PLOT_PATH = OUTPUT_DIR / "msr_regimes_monthly_no_lookahead_features.png"
PROBABILITY_PLOT_PATH = OUTPUT_DIR / "msr_regimes_monthly_no_lookahead_probabilities.png"

START_DATE = os.getenv("START_DATE", "2006-01-01")
END_DATE = os.getenv("END_DATE", "2026-04-30")
N_COMPONENTS = int(os.getenv("N_COMPONENTS", "3"))
MIN_TRAIN_OBS = int(os.getenv("MIN_TRAIN_OBS", "756"))
MSR_MAX_ITER = int(os.getenv("MSR_MAX_ITER", "500"))
MSR_RANDOM_STATE = int(os.getenv("MSR_RANDOM_STATE", "42"))
WINDOW_MODE = os.getenv("WINDOW_MODE", "expanding").strip().lower()
ROLLING_WINDOW_OBS = int(os.getenv("ROLLING_WINDOW_OBS", "1260"))
POSTERIOR_MEAN_WINDOW_DAYS = int(os.getenv("POSTERIOR_MEAN_WINDOW_DAYS", "5"))

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
    msr_df = pd.DataFrame(index=df_equity.index)

    stoxx_price = df_equity[".STOXX"].astype(float)
    msr_df["stoxx_return"] = np.log(stoxx_price / stoxx_price.shift(1))
    msr_df["vstoxx"] = df_equity[".V2TX"].astype(float)
    msr_df["term_spread"] = (
        df_yields["DE10YT=RR"].astype(float) - df_yields["DE2YT=RR"].astype(float)
    )
    msr_df = msr_df.join(df_credit, how="left")
    msr_df["credit_spread"] = msr_df["credit_spread"].ffill()

    features = ["stoxx_return", "vstoxx", "term_spread", "credit_spread"]
    msr_df = msr_df[features].dropna().sort_index()
    stoxx_price = stoxx_price.reindex(msr_df.index)
    return msr_df, features, stoxx_price


def get_month_end_observation_dates(msr_df):
    source_dates = pd.DatetimeIndex(msr_df.index)
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
    state_means = (
        train_df.assign(raw_state=raw_states)
        .groupby("raw_state")[["stoxx_return", "vstoxx", "term_spread", "credit_spread"]]
        .mean()
    )

    if len(state_means) < 3:
        raise ValueError(f"MSR loysi vain {len(state_means)} statea, tarvitaan 3.")

    standardized = state_means.apply(zscore_series, axis=0)
    economic_score = (
        standardized["stoxx_return"]
        - standardized["vstoxx"]
        - standardized["credit_spread"]
        + 0.5 * standardized["term_spread"]
    )

    sorted_states = economic_score.sort_values(ascending=False).index.tolist()
    regime_map = {
        sorted_states[0]: 0,
        sorted_states[-1]: 1,
        sorted_states[1]: 2,
    }
    return regime_map, state_means, economic_score


def map_raw_posterior_to_regime_probabilities(raw_posterior, regime_map):
    posterior_by_regime = {0: 0.0, 1: 0.0, 2: 0.0}
    for raw_state, prob in enumerate(raw_posterior):
        posterior_by_regime[regime_map[raw_state]] = float(prob)
    return posterior_by_regime


def fit_predict_month_end_regime(train_df, features):
    # Hamiltonin kanoninen asetelma: univariate stoxx_return, regime-riippuva
    # keskiarvo + varianssi. Skaalataan jotta optimointi konvergoi vakaasti.
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(train_df[["stoxx_return"]].values).ravel()

    np.random.seed(MSR_RANDOM_STATE)
    model = MarkovRegression(
        endog=y_scaled,
        k_regimes=N_COMPONENTS,
        trend="c",
        switching_variance=True,
    )
    result = model.fit(maxiter=MSR_MAX_ITER, disp=False)

    # Smoothed marginal probabilities: shape (n_regimes, n_obs).
    smoothed = np.asarray(result.smoothed_marginal_probabilities)
    if smoothed.shape[0] != N_COMPONENTS:
        smoothed = smoothed.T
    # Paivakohtainen kova tila-assignment relabelointia varten.
    raw_states = smoothed.argmax(axis=0)
    regime_map, state_means, economic_score = make_regime_map(raw_states, train_df)

    last_raw_state = int(raw_states[-1])
    last_regime = int(regime_map[last_raw_state])

    bull_state = [k for k, v in regime_map.items() if v == 0][0]
    bear_state = [k for k, v in regime_map.items() if v == 1][0]
    transition_state = [k for k, v in regime_map.items() if v == 2][0]

    # smoothed[regime, day] -> posterior viimeisena paivana per raw regime.
    posterior_last_raw = smoothed[:, -1]
    posterior_by_regime = map_raw_posterior_to_regime_probabilities(posterior_last_raw, regime_map)

    posterior_window_days = min(POSTERIOR_MEAN_WINDOW_DAYS, smoothed.shape[1])
    posterior_mean_raw = smoothed[:, -posterior_window_days:].mean(axis=1)
    posterior_mean_by_regime = map_raw_posterior_to_regime_probabilities(
        posterior_mean_raw,
        regime_map,
    )

    return {
        "MSR_Regime": last_regime,
        "Bull_Prob": posterior_by_regime[0],
        "Bear_Prob": posterior_by_regime[1],
        "Transition_Prob": posterior_by_regime[2],
        "Bull_Prob_MeanWindow": posterior_mean_by_regime[0],
        "Bear_Prob_MeanWindow": posterior_mean_by_regime[1],
        "Transition_Prob_MeanWindow": posterior_mean_by_regime[2],
        "PosteriorMeanWindowDaysUsed": posterior_window_days,
        "LastRawState": last_raw_state,
        "ModelConverged": bool(result.mle_retvals.get("converged", False)),
        "TrainLogLikelihood": float(result.llf),
        "BullStateTrainMeanReturn": float(state_means.loc[bull_state, "stoxx_return"]),
        "BearStateTrainMeanReturn": float(state_means.loc[bear_state, "stoxx_return"]),
        "TransitionStateTrainMeanReturn": float(state_means.loc[transition_state, "stoxx_return"]),
        "BullStateEconomicScore": float(economic_score.loc[bull_state]),
        "BearStateEconomicScore": float(economic_score.loc[bear_state]),
        "TransitionStateEconomicScore": float(economic_score.loc[transition_state]),
    }


def select_train_window(msr_df, source_date):
    full_train_df = msr_df.loc[:source_date].copy()
    if WINDOW_MODE == "expanding":
        return full_train_df
    if WINDOW_MODE == "rolling":
        return full_train_df.tail(ROLLING_WINDOW_OBS).copy()
    raise ValueError(
        f"Tuntematon WINDOW_MODE={WINDOW_MODE!r}. Kayta arvoa 'expanding' tai 'rolling'."
    )


def build_no_lookahead_monthly_regimes(msr_df, features):
    month_end_obs = get_month_end_observation_dates(msr_df)
    rows = []

    total_month_ends = len(month_end_obs)
    month_end_pairs = month_end_obs[["Date", "SourceDate"]].itertuples(index=False, name=None)

    for position, (month_end_raw, source_date_raw) in enumerate(month_end_pairs, start=1):
        month_end = pd.Timestamp(month_end_raw)
        source_date = pd.Timestamp(source_date_raw)
        train_df = select_train_window(msr_df, source_date)

        if len(train_df) < MIN_TRAIN_OBS:
            continue

        try:
            pred = fit_predict_month_end_regime(train_df, features)
        except Exception as exc:
            print(f"MSR fit epaonnistui kuukaudelle {month_end.date()} (source {source_date.date()}): {exc}")
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
                f"No-lookahead MSR: {position}/{total_month_ends} month-endia kasitelty "
                f"(viimeisin {month_end.strftime('%Y-%m')})"
            )

    monthly = pd.DataFrame(rows)
    if monthly.empty:
        return monthly

    monthly["RegimeLabel"] = monthly["MSR_Regime"].map(REGIME_LABELS)
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


def plot_month_end_feature_regimes(msr_df, stoxx_price, monthly, output_path: Path):
    monthly_plot = monthly.set_index("SourceDate").sort_index()
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    series_specs = [
        ("STOXX 600", stoxx_price, axes[0]),
        ("VSTOXX", msr_df["vstoxx"], axes[1]),
        ("Term Spread (10Y-2Y)", msr_df["term_spread"], axes[2]),
        ("Sovereign Spread (All-AAA)", msr_df["credit_spread"], axes[3]),
    ]

    for label, full_series, axis in series_specs:
        axis.plot(full_series.index, full_series.values, color="lightgray", linewidth=0.9)
        for regime_id, color in REGIME_COLORS.items():
            mask = monthly_plot["MSR_Regime"] == regime_id
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
    axes[0].set_title("No-Lookahead MSR Month-End Regimes (Hamilton 1989)")
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
    ax.set_title(f"No-Lookahead MSR Month-End Regime Probabilities ({POSTERIOR_MEAN_WINDOW_DAYS}D Mean)")
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
    if WINDOW_MODE == "rolling" and ROLLING_WINDOW_OBS < MIN_TRAIN_OBS:
        raise ValueError("ROLLING_WINDOW_OBS tulee olla >= MIN_TRAIN_OBS rolling-tilassa.")

    print(
        "Building no-lookahead MSR monthly regimes "
        f"with {WINDOW_MODE} window from {START_DATE} to {END_DATE} "
        f"(min train obs={MIN_TRAIN_OBS}, posterior mean window={POSTERIOR_MEAN_WINDOW_DAYS}d)"
    )
    if WINDOW_MODE == "rolling":
        print(f"Rolling window observations: {ROLLING_WINDOW_OBS}")

    rd.open_session()
    try:
        df_equity, df_yields, df_credit = fetch_input_data(START_DATE, END_DATE)
        msr_df, features, stoxx_price = build_feature_matrix(df_equity, df_yields, df_credit)

        print(f"Feature matrix: {len(msr_df)} daily observations")
        print(f"Date range: {msr_df.index.min().date()} — {msr_df.index.max().date()}")

        monthly = build_no_lookahead_monthly_regimes(msr_df, features)
        if monthly.empty:
            raise RuntimeError("No-lookahead monthly regime table jaa tyhjaksi.")

        monthly_ml = build_ml_safe_monthly_regimes(monthly)

        monthly.to_csv(MONTHLY_OUTPUT_PATH, index=False)
        monthly_ml.to_csv(MONTHLY_ML_OUTPUT_PATH, index=False)
        plot_month_end_feature_regimes(msr_df, stoxx_price, monthly, FEATURE_PLOT_PATH)
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
        print(monthly["MSR_Regime"].value_counts().rename(
            {0: "Bull (0)", 1: "Bear (1)", 2: "Transition (2)"}
        ).sort_index())
        print("\nHead:")
        print(monthly.head(12).to_string(index=False))
    finally:
        rd.close_session()


if __name__ == "__main__":
    main()
