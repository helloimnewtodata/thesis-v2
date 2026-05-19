"""
Merge ML-safe Jump Model regime features into the monthly master panel.

JM regimes are market-wide (one value per month, shared by every stock), so the
merge is on Date only — every Instrument on a given month-end inherits the same
JM features. This mirrors how HMM features are merged in
main.build_master_dataframe (merge on "Date").

Input:  data/02_preprocessed/MASTER_DF_1.csv
        data/01_raw/outputs/JM_output_ml.csv   (produced by `python src/JM.py`)
Output: data/02_preprocessed/MASTER_DF_1_JM.csv

The HMM probability columns are dropped from the output — this file carries the
JM regime features in their place.
"""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = ROOT / "data" / "02_preprocessed" / "MASTER_DF_1.csv"
JM_ML_PATH = ROOT / "data" / "01_raw" / "outputs" / "JM_output_ml.csv"
OUTPUT_PATH = ROOT / "data" / "02_preprocessed" / "MASTER_DF_1_JM.csv"

HMM_COLUMNS = [
    "Bull_Prob",
    "Bear_Prob",
    "Transition_Prob",
    "Bull_Prob_MeanWindow",
    "Bear_Prob_MeanWindow",
    "Transition_Prob_MeanWindow",
]

# ML-relevant JM features, analogous to the HMM probability columns:
#   *_StateIndicator  — hard one-hot of the current month-end regime
#   *_StateMeanWindow — last-N-trading-day hard-state frequencies (NOT posteriors)
JM_FEATURE_COLUMNS = [
    "Bull_StateIndicator",
    "Bear_StateIndicator",
    "Transition_StateIndicator",
    "Bull_StateMeanWindow",
    "Bear_StateMeanWindow",
    "Transition_StateMeanWindow",
]


def main():
    if not JM_ML_PATH.exists():
        raise FileNotFoundError(
            f"JM-featuret puuttuvat: {JM_ML_PATH}. Aja ensin `python src/JM.py`."
        )

    master = pd.read_csv(MASTER_PATH)
    master["Date"] = pd.to_datetime(master["Date"])

    jm = pd.read_csv(JM_ML_PATH)
    jm["Date"] = pd.to_datetime(jm["Date"])

    missing = [col for col in JM_FEATURE_COLUMNS if col not in jm.columns]
    if missing:
        raise ValueError(f"JM-sarakkeet puuttuvat tiedostosta {JM_ML_PATH}: {missing}")

    master = master.drop(columns=[c for c in HMM_COLUMNS if c in master.columns])

    jm_panel = jm[["Date"] + JM_FEATURE_COLUMNS].drop_duplicates(subset="Date")
    merged = master.merge(jm_panel, on="Date", how="left")

    unmatched = merged[JM_FEATURE_COLUMNS[0]].isna().sum()
    if unmatched:
        print(
            f"Varoitus: {unmatched} riville ei loytynyt JM-regiimia "
            f"(Date puuttuu JM-taulusta)."
        )

    merged = merged.sort_values(["Instrument", "Date"]).reset_index(drop=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(merged)}, cols: {len(merged.columns)}")
    print(f"Dropped HMM columns: {[c for c in HMM_COLUMNS if c in master.columns] or HMM_COLUMNS}")
    print(f"Added JM columns: {JM_FEATURE_COLUMNS}")


if __name__ == "__main__":
    main()
