"""
Merge ML-safe Continuous Jump Model regime probabilities into a monthly master.

JM2 regimes are market-wide (one value per month, shared by every stock), so the
merge is on Date only — every Instrument on a given month-end inherits the same
JM2 features. This mirrors how HMM features are merged in
main.build_master_dataframe (merge on "Date").

JM2's probability columns share the HMM column names (Bull_Prob, ...), so the HMM
columns are dropped from the master first and JM2's probabilities take their place.

Usage:
    python merge_jm2_to_master.py
        -> data/02_preprocessed/MASTER_DF_1.csv      + JM2 -> MASTER_DF_1_JM2.csv
    python merge_jm2_to_master.py --master data/02_preprocessed/PIT_MASTER_DF_1.csv
        -> PIT_MASTER_DF_1.csv                       + JM2 -> PIT_MASTER_DF_1_JM2.csv

The output filename defaults to "<master stem>_JM2.csv" next to the master.
Requires data/01_raw/outputs/JM2_output_ml.csv (produced by `python src/JM2.py`).
"""
import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_PATH = ROOT / "data" / "02_preprocessed" / "MASTER_DF_1.csv"
JM2_ML_PATH = ROOT / "data" / "01_raw" / "outputs" / "JM2_output_ml.csv"

HMM_COLUMNS = [
    "Bull_Prob",
    "Bear_Prob",
    "Transition_Prob",
    "Bull_Prob_MeanWindow",
    "Bear_Prob_MeanWindow",
    "Transition_Prob_MeanWindow",
]

# ML-relevant JM2 features — genuine regime probabilities in [0, 1] summing to 1,
# directly analogous to the HMM posterior columns they replace.
JM2_FEATURE_COLUMNS = [
    "Bull_Prob",
    "Bear_Prob",
    "Transition_Prob",
    "Bull_Prob_MeanWindow",
    "Bear_Prob_MeanWindow",
    "Transition_Prob_MeanWindow",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master",
        default=str(DEFAULT_MASTER_PATH),
        help="Path to the monthly master CSV to merge JM2 features into.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to '<master stem>_JM2.csv' next to the master.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    master_path = Path(args.master)
    if not master_path.is_absolute():
        master_path = ROOT / master_path

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
    else:
        output_path = master_path.with_name(f"{master_path.stem}_JM2.csv")

    if not master_path.exists():
        raise FileNotFoundError(f"Master-tiedostoa ei loydy: {master_path}")
    if not JM2_ML_PATH.exists():
        raise FileNotFoundError(
            f"JM2-featuret puuttuvat: {JM2_ML_PATH}. Aja ensin `python src/JM2.py`."
        )

    master = pd.read_csv(master_path)
    master["Date"] = pd.to_datetime(master["Date"])

    jm2 = pd.read_csv(JM2_ML_PATH)
    jm2["Date"] = pd.to_datetime(jm2["Date"])

    missing = [col for col in JM2_FEATURE_COLUMNS if col not in jm2.columns]
    if missing:
        raise ValueError(f"JM2-sarakkeet puuttuvat tiedostosta {JM2_ML_PATH}: {missing}")

    dropped = [c for c in HMM_COLUMNS if c in master.columns]
    master = master.drop(columns=dropped)

    jm2_panel = jm2[["Date"] + JM2_FEATURE_COLUMNS].drop_duplicates(subset="Date")
    merged = master.merge(jm2_panel, on="Date", how="left")

    unmatched = merged[JM2_FEATURE_COLUMNS[0]].isna().sum()
    if unmatched:
        print(
            f"Varoitus: {unmatched} riville ei loytynyt JM2-regiimia "
            f"(Date puuttuu JM2-taulusta)."
        )

    merged = merged.sort_values(["Instrument", "Date"]).reset_index(drop=True)
    merged.to_csv(output_path, index=False)

    print(f"Master in:  {master_path}")
    print(f"Saved:      {output_path}")
    print(f"Rows: {len(merged)}, cols: {len(merged.columns)}")
    print(f"Dropped HMM columns: {dropped or '(none present)'}")
    print(f"Added JM2 columns: {JM2_FEATURE_COLUMNS}")


if __name__ == "__main__":
    main()
