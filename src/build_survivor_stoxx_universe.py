"""
Survivor-universumi STOXX Europe 600:lle — point-in-time -metodologialla.

Logiikka (taso 3):
    1. Hae snapshot-jäsenlista start_date:lle (RIC:t jotka olivat indeksissä t=start).
    2. Hae kaikki indeksin Joiner/Leaver -tapahtumat välillä (start_date, end_date]
       Refinitivin TR.IndexJLConstituent* -kentistä.
    3. RIC selviää otokseen jos ja vain jos:
         (a) se oli indeksissä start_date:nä, JA
         (b) sen kohdalle EI osu yhtään LEAVER-tapahtumaa välillä (start_date, end_date].
       Tämä karsii osakkeet jotka tippuivat (vaikka olisivat palanneet myöhemmin —
       välissä on aukko jolloin osake ei ollut indeksissä → ei jatkuva jäsenyys).

HUOM survivorship biasista:
    Tämä menetelmä rajaa otoksen tarkoituksellisesti vain jatkuvasti indeksissä
    olleisiin osakkeisiin → vinouttaa tuottoja ylöspäin (delistautuneet ja
    indeksistä pudonneet jäävät pois). Käytä vain robustness-testeissä tai
    tutkimusasetelmissa jotka vaativat tasapainoisen paneelin koko aikaväliltä.
    Tuotantotason tutkimuksessa point-in-time-paneeli (kuukausittain rebalansoitu
    indeksijäsenyys) on metodologisesti oikeampi.
"""

from pathlib import Path

import pandas as pd
import refinitiv.data as rd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_RIC = ".STOXX"
DEFAULT_START_DATE = "2010-01-01"
DEFAULT_END_DATE = "2026-01-01"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "stoxx600_survivor_universe.csv"


def _fetch_snapshot_rics(snapshot_date):
    """Hakee STOXX 600 -jäsenlistan yhdelle ankkuripäivälle ja palauttaa RIC-setin."""
    snapshot_code = snapshot_date.replace("-", "")
    df = rd.get_data(
        universe=[f"0#.STOXX({snapshot_code})"],
        fields=["TR.PriceClose"],
        parameters={"SDate": snapshot_date, "EDate": snapshot_date},
    )
    rics = (
        df["Instrument"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    return set(rics)


def fetch_index_change_events(start_date, end_date, index_ric=INDEX_RIC):
    """
    Hakee kaikki Joiner/Leaver -tapahtumat indeksille välillä [start, end].

    Refinitivin kentät:
        TR.IndexJLConstituentRIC       -> RIC joka liittyi tai poistui
        TR.IndexJLConstituentRIC.date  -> tapahtuman päivämäärä
        TR.IndexJLConstituentituentChange -> "Joiner" tai "Leaver"
        IC=B (parameters)              -> Joiners ja Leavers

    Palauttaa DataFrame ['RIC', 'ChangeDate', 'ChangeType'].
    """
    df = rd.get_data(
        universe=[index_ric],
        fields=[
            "TR.IndexJLConstituentRIC",
            "TR.IndexJLConstituentRIC.date",
            "TR.IndexJLConstituentituentChange",
        ],
        parameters={
            "SDate": start_date,
            "EDate": end_date,
            "IC": "B",
        },
    )

    if df is None or df.empty:
        return pd.DataFrame(columns=["RIC", "ChangeDate", "ChangeType"])

    # Refinitiv-palautusten sarakenimet voivat vaihdella; mapataan sisältöavainsanoilla
    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc == "date" or "change date" in lc:
            rename_map[col] = "ChangeDate"
        elif "ric" in lc and "constituent" in lc:
            rename_map[col] = "RIC"
        elif lc == "change" or "constituentituentchange" in lc:
            rename_map[col] = "ChangeType"
    df = df.rename(columns=rename_map)

    needed = {"RIC", "ChangeDate", "ChangeType"}
    if not needed.issubset(df.columns):
        raise RuntimeError(
            f"Refinitivin palautuksesta puuttuu odotettuja sarakkeita. "
            f"Saatiin: {df.columns.tolist()}"
        )

    df = df[["RIC", "ChangeDate", "ChangeType"]].copy()
    df["ChangeDate"] = pd.to_datetime(df["ChangeDate"], errors="coerce")
    df["RIC"] = df["RIC"].astype("string").str.strip()
    df["ChangeType"] = df["ChangeType"].astype("string").str.strip()
    df = df.dropna(subset=["RIC", "ChangeDate"])
    df = df[df["RIC"] != ""]
    return df.sort_values(["RIC", "ChangeDate", "ChangeType"]).reset_index(drop=True)


def filter_continuous_members(start_snapshot, events, start_date, end_date):
    """
    Palauttaa setin RIC:eistä jotka olivat indeksissä jatkuvasti
    [start_date, end_date] välillä — eli olivat snapshotissa eikä yhtään
    Leaver-tapahtumaa osu väliin (start_date, end_date].

    Jos osake palasi indeksiin myöhemmin saman ikkunan sisällä, se EI silti
    täytä kriteeriä, koska aukko (out-of-index) tarkoittaa että osakkeen tuottoja
    ei voi käyttää indeksin jäsenenä koko ikkunan aikana.
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    in_window = events[
        (events["ChangeDate"] > start_dt) & (events["ChangeDate"] <= end_dt)
    ]
    leavers_in_window = set(
        in_window.loc[
            in_window["ChangeType"].str.contains("leav", case=False, na=False),
            "RIC",
        ]
    )

    return start_snapshot - leavers_in_window


def fetch_survivor_universe(
    start_date=DEFAULT_START_DATE,
    end_date=DEFAULT_END_DATE,
    index_ric=INDEX_RIC,
):
    """
    Pää-orkesteri: snapshot start_date:lle + Joiner/Leaver -historia +
    jatkuvuusfilter. Palauttaa (DataFrame, diagnostiikka-dict).
    """
    start_snapshot = _fetch_snapshot_rics(start_date)
    events = fetch_index_change_events(start_date, end_date, index_ric=index_ric)
    survivors = filter_continuous_members(start_snapshot, events, start_date, end_date)

    survivors_df = pd.DataFrame({"RIC": sorted(survivors)})

    diagnostics = {
        "n_start_snapshot": len(start_snapshot),
        "n_events_total": len(events),
        "n_joiners_in_window": int(
            events["ChangeType"].str.contains("join", case=False, na=False).sum()
        ),
        "n_leavers_in_window": int(
            events["ChangeType"].str.contains("leav", case=False, na=False).sum()
        ),
        "n_survivors": len(survivors_df),
    }
    return survivors_df, diagnostics


def save_survivor_universe(
    start_date=DEFAULT_START_DATE,
    end_date=DEFAULT_END_DATE,
    output_path=DEFAULT_OUTPUT_PATH,
):
    """Hae survivor-lista ja tallenna CSV:nä."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    survivors, diagnostics = fetch_survivor_universe(start_date, end_date)
    survivors.to_csv(output_path, index=False)
    return survivors, output_path, diagnostics


def main():
    rd.open_session()
    try:
        survivors, output_path, diag = save_survivor_universe()
    finally:
        rd.close_session()

    print(f"\n{'=' * 60}")
    print(f"Survivor-universumi: STOXX 600, {DEFAULT_START_DATE} → {DEFAULT_END_DATE}")
    print("=" * 60)
    print(f"Snapshot-jäsenmäärä {DEFAULT_START_DATE}:    {diag['n_start_snapshot']}")
    print(f"Tapahtumia kokonaisuudessaan ikkunassa:   {diag['n_events_total']}")
    print(f"  Joiners:                                {diag['n_joiners_in_window']}")
    print(f"  Leavers:                                {diag['n_leavers_in_window']}")
    print(f"Jatkuvasti indeksissä koko aikavälin:     {diag['n_survivors']}")
    print(f"\nTallennettu: {output_path}")


if __name__ == "__main__":
    main()
