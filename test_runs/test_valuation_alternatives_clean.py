"""
Raw Refinitiv valuation variable smoke test.

Tavoite:
- hakea kandidaattiluvut mahdollisimman "sellaisenaan" Refinitivistä
- nimetä sarakkeet täsmälleen field codejen mukaan
- välttää omia kombinaatioita tai ratio-laskentaa tässä vaiheessa
- tallentaa daily raw + month-end raw + kattavuusraportti

AJO:
    venv-v3/bin/python test_valuation_alternatives_clean.py
"""

from pathlib import Path

import pandas as pd
import refinitiv.data as rd


PARAMS = {
    "SDate": "2008-10-08",
    "EDate": "2025-12-31",
    "Frq": "D",
    "Curn": "EUR",
}

RICS_SOURCE = Path("data/01_raw/smoke_stocks.csv")
UNIVERSE_FALLBACK = Path("data/stoxx600_universe.csv")
OUTPUT_DIR = Path("data/01_raw")
N_RICS = 10

FIELDS = [
    # Perusankkurit
    "TR.PriceClose",
    "TR.CompanyMarketCap",
    "TR.SharesOutstanding",
    # E/P / P/E -kandidaatit
    "TR.PE",
    "TR.EPSActValue",
    "TR.EPSFRActValue",
    "TR.F.NetIncAfterTax",
    "TR.FwdPtoEPSSmartEst",
    "TR.PtoEPSMeanEst",
    # P/S
    "TR.PriceToSalesPerShare",
    "TR.F.TotRevenue",
    # P/CF
    "TR.PriceToCFPerShare",
    "TR.F.PriceToCFPPerShr",
    "TR.F.CF",
    "TR.F.NetCashFlowOp",
    # Dividend / DPS
    "TR.DividendYield",
    "TR.DPSActValue",
    "TR.F.DivPerShare",
    "TR.DPSActValueYield",
    # P/B
    "TR.PriceToBVPerShare",
    # ROE -kandidaatit
    "TR.F.ReturnAvgComEqPct",
    "TR.ROEActValue",
]


def load_rics():
    if RICS_SOURCE.exists():
        df = pd.read_csv(RICS_SOURCE, usecols=["Instrument"])
        rics = sorted(df["Instrument"].dropna().unique().tolist())[:N_RICS]
        source = str(RICS_SOURCE)
    elif UNIVERSE_FALLBACK.exists():
        df = pd.read_csv(UNIVERSE_FALLBACK, usecols=["RIC"])
        rics = sorted(df["RIC"].dropna().unique().tolist())[:N_RICS]
        source = str(UNIVERSE_FALLBACK)
    else:
        raise FileNotFoundError(
            "Ei löydy RIC-lähdettä. Odotin joko smoke_stocks.csv:tä tai stoxx600_universe.csv:tä."
        )
    return rics, source


def fetch_one_field(universe, field_code):
    df = rd.get_data(
        universe=universe,
        fields=["TR.PriceClose.date", field_code],
        parameters=PARAMS,
    )

    if "Date" not in df.columns:
        raise ValueError(f"Kentälle {field_code} ei tullut Date-saraketta.")

    value_cols = [c for c in df.columns if c not in ("Instrument", "Date")]
    if len(value_cols) != 1:
        raise ValueError(
            f"Kenttä {field_code} palautti odottamattoman sarakemäärän: {value_cols}"
        )

    refinitiv_label = value_cols[0]
    out = df[["Instrument", "Date", refinitiv_label]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out[field_code] = pd.to_numeric(out[refinitiv_label], errors="coerce")
    out = (
        out.drop(columns=[refinitiv_label])
        .dropna(subset=["Date"])
        .drop_duplicates(subset=["Instrument", "Date"], keep="last")
        .sort_values(["Instrument", "Date"])
        .reset_index(drop=True)
    )

    meta = {
        "field_code": field_code,
        "refinitiv_label": refinitiv_label,
        "rows": len(out),
        "non_null": int(out[field_code].notna().sum()),
        "pct_nan": round(out[field_code].isna().mean() * 100, 2),
        "status": "ok",
        "error": "",
    }
    return out, meta


def build_master(universe):
    frames = []
    metadata = []

    for field_code in FIELDS:
        print(f"Haetaan {field_code} ...")
        try:
            field_df, meta = fetch_one_field(universe, field_code)
        except Exception as exc:
            metadata.append(
                {
                    "field_code": field_code,
                    "refinitiv_label": "",
                    "rows": 0,
                    "non_null": 0,
                    "pct_nan": 100.0,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"  -> FAIL: {exc}")
            continue

        frames.append(field_df)
        metadata.append(meta)
        print(
            f"  -> OK: {meta['refinitiv_label']} | rows={meta['rows']:,} | "
            f"non-null={meta['non_null']:,} | %NaN={meta['pct_nan']:.2f}"
        )

    if not frames:
        raise RuntimeError("Yksikään kenttä ei palautunut onnistuneesti.")

    master = frames[0]
    for frame in frames[1:]:
        master = master.merge(frame, on=["Instrument", "Date"], how="outer")

    master = master.sort_values(["Instrument", "Date"]).reset_index(drop=True)
    return master, pd.DataFrame(metadata)


def build_month_end(master):
    me = master.copy()
    me["YM"] = me["Date"].dt.to_period("M")
    me = (
        me.sort_values(["Instrument", "Date"])
        .groupby(["Instrument", "YM"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return me


def coverage_report(df, level_name):
    rows = []
    for col in df.columns:
        if col in ("Instrument", "Date", "YM"):
            continue
        rows.append(
            {
                "level": level_name,
                "column": col,
                "non_null": int(df[col].notna().sum()),
                "pct_nan": round(df[col].isna().mean() * 100, 2),
                "median": df[col].median(skipna=True),
            }
        )
    return pd.DataFrame(rows).sort_values(["pct_nan", "column"]).reset_index(drop=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rics, source = load_rics()
    print(f"RIC-lähde: {source}")
    print(f"RICs ({len(rics)}): {rics}\n")

    rd.open_session()
    try:
        master, field_meta = build_master(rics)
    finally:
        rd.close_session()

    month_end = build_month_end(master)

    daily_cov = coverage_report(master, "daily_raw")
    me_cov = coverage_report(month_end, "month_end_raw")
    coverage = pd.concat([daily_cov, me_cov], ignore_index=True)

    daily_path = OUTPUT_DIR / "valuation_alternatives_clean_daily_raw.csv"
    me_path = OUTPUT_DIR / "valuation_alternatives_clean_monthend_raw.csv"
    meta_path = OUTPUT_DIR / "valuation_alternatives_clean_field_map.csv"
    cov_path = OUTPUT_DIR / "valuation_alternatives_clean_coverage.csv"

    master.to_csv(daily_path, index=False)
    month_end.to_csv(me_path, index=False)
    field_meta.to_csv(meta_path, index=False)
    coverage.to_csv(cov_path, index=False)

    print("\nTallennettu tiedostot:")
    print(f"  - daily raw:   {daily_path}")
    print(f"  - month-end:   {me_path}")
    print(f"  - field map:   {meta_path}")
    print(f"  - coverage:    {cov_path}")

    print("\nMonth-end coverage (paras ensin):")
    print(me_cov.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
