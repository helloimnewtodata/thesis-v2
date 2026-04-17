from pathlib import Path

import refinitiv.data as rd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_DATE = "2026-04-01"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "stoxx600_universe.csv"


def fetch_stoxx_universe_snapshot(snapshot_date=DEFAULT_SNAPSHOT_DATE):
    """Fetch a STOXX Europe 600 snapshot for one date and return RICs as a DataFrame."""
    snapshot_code = snapshot_date.replace("-", "")
    df = rd.get_data(
        universe=[f"0#.STOXX({snapshot_code})"],
        fields=["TR.PriceClose"],
        parameters={"SDate": snapshot_date, "EDate": snapshot_date},
    )

    snapshot = (
        df[["Instrument"]]
        .rename(columns={"Instrument": "RIC"})
        .dropna()
        .drop_duplicates()
        .sort_values("RIC")
        .reset_index(drop=True)
    )
    return snapshot


def save_stoxx_universe_snapshot(
    snapshot_date=DEFAULT_SNAPSHOT_DATE,
    output_path=DEFAULT_OUTPUT_PATH,
):
    """Fetch the snapshot and save it under data/ in the project root."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = fetch_stoxx_universe_snapshot(snapshot_date=snapshot_date)
    snapshot.to_csv(output_path, index=False)
    return snapshot, output_path


def main():
    rd.open_session()
    try:
        snapshot, output_path = save_stoxx_universe_snapshot()
    finally:
        rd.close_session()

    print(f"Tallennettu {len(snapshot)} RIC:iä tiedostoon {output_path}")


if __name__ == "__main__":
    main()
