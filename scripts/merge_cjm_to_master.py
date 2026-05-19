"""
Merge ML-safe Continuous Statistical Jump Model regime probabilities into a
monthly master.

CJM regimes are market-wide (one value per month, shared by every stock), so the
merge is on Date only — every Instrument on a given month-end inherits the same
CJM features. This mirrors how HMM features are merged in
main.build_master_dataframe (merge on "Date").

CJM's probability columns share the HMM column names (Bull_Prob, ...), so the HMM
columns are dropped from the master first and CJM's probabilities take their place.

Usage:
    python merge_cjm_to_master.py
        -> data/02_preprocessed/MASTER_DF_1.csv      + CJM -> MASTER_DF_1_CJM.csv
    python merge_cjm_to_master.py --master data/02_preprocessed/PIT_MASTER_DF_1.csv
        -> PIT_MASTER_DF_1.csv                       + CJM -> PIT_MASTER_DF_1_CJM.csv

The output filename defaults to "<master stem>_CJM.csv" next to the master.
Requires data/01_raw/outputs/CJM_output_ml.csv (produced by `python src/CJM.py`).
"""
import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_PATH = ROOT / "data" / "02_preprocessed" / "MASTER_DF_1.csv"
CJM_ML_PATH = ROOT / "data" / "01_raw" / "outputs" / "CJM_output_ml.csv"

HMM_COLUMNS = [
    "Bull_Prob",
    "Bear_Prob",
    "Transition_Prob",
    "Bull_Prob_MeanWindow",
    "Bear_Prob_MeanWindow",
    "Transition_Prob_MeanWindow",
]

# ML-relevant CJM features — genuine regime probabilities in [0, 1] summing to 1,
# directly analogous to the HMM posterior columns they replace.
CJM_FEATURE_COLUMNS = [
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
        help="Path to the monthly master CSV to merge CJM features into.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to '<master stem>_CJM.csv' next to the master.",
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
        output_path = master_path.with_name(f"{master_path.stem}_CJM.csv")

    if not master_path.exists():
        raise FileNotFoundError(f"Master-tiedostoa ei loydy: {master_path}")
    if not CJM_ML_PATH.exists():
        raise FileNotFoundError(
            f"CJM-featuret puuttuvat: {CJM_ML_PATH}. Aja ensin `python src/CJM.py`."
        )

    master = pd.read_csv(master_path)
    master["Date"] = pd.to_datetime(master["Date"])

    cjm = pd.read_csv(CJM_ML_PATH)
    cjm["Date"] = pd.to_datetime(cjm["Date"])

    missing = [col for col in CJM_FEATURE_COLUMNS if col not in cjm.columns]
    if missing:
        raise ValueError(f"CJM-sarakkeet puuttuvat tiedostosta {CJM_ML_PATH}: {missing}")

    dropped = [c for c in HMM_COLUMNS if c in master.columns]
    master = master.drop(columns=dropped)

    cjm_panel = cjm[["Date"] + CJM_FEATURE_COLUMNS].drop_duplicates(subset="Date")
    merged = master.merge(cjm_panel, on="Date", how="left")

    unmatched = merged[CJM_FEATURE_COLUMNS[0]].isna().sum()
    if unmatched:
        print(
            f"Varoitus: {unmatched} riville ei loytynyt CJM-regiimia "
            f"(Date puuttuu CJM-taulusta)."
        )

    merged = merged.sort_values(["Instrument", "Date"]).reset_index(drop=True)
    merged.to_csv(output_path, index=False)

    print(f"Master in:  {master_path}")
    print(f"Saved:      {output_path}")
    print(f"Rows: {len(merged)}, cols: {len(merged.columns)}")
    print(f"Dropped HMM columns: {dropped or '(none present)'}")
    print(f"Added CJM columns: {CJM_FEATURE_COLUMNS}")


if __name__ == "__main__":
    main()
