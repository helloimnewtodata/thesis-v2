"""
P/E- ja P/CF-vaihtoehtojen smoke test.

Tavoite:
- hakea vain P/E- ja P/CF-haaran raw-kentät Refinitivistä
- forward-fillata fiscal/actual-luvut per osake
- muodostaa month-end snapshot
- laskea omat vaihtoehtoiset P/E- ja P/CF-luvut
- verrata niitä Refinitivin referenssiratioihin

AJO:
    source venv-v3/bin/activate
    python test_pe_pcf_alternatives_clean.py
"""

from pathlib import Path

import numpy as np
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
    "TR.PriceClose",
    "TR.CompanyMarketCap",
    "TR.SharesOutstanding",
    # P/E
    "TR.PE",
    "TR.EPSActValue",
    "TR.EPSFRActValue",
    "TR.F.NetIncAfterTax",
    # P/CF
    "TR.PriceToCFPerShare",
    "TR.F.PricetoCFPPerShr",
    "TR.F.CF",
    "TR.F.NetCashFlowOp",
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


def forward_fill_fundamentals(df):
    df = df.sort_values(["Instrument", "Date"]).copy()
    skip = {"Instrument", "Date", "TR.PriceClose"}
    fill_cols = [c for c in df.columns if c not in skip]
    for col in fill_cols:
        df[col] = df.groupby("Instrument")[col].ffill()
    return df


def build_month_end(df):
    me = df.copy()
    me["YM"] = me["Date"].dt.to_period("M")
    me = (
        me.sort_values(["Instrument", "Date"])
        .groupby(["Instrument", "YM"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return me


def safe_div(numerator, denominator):
    out = numerator / denominator
    return out.replace([np.inf, -np.inf], np.nan)


def compute_alternatives(me):
    out = me.copy()

    price = out["TR.PriceClose"]
    mktcap = out["TR.CompanyMarketCap"]
    shares = out["TR.SharesOutstanding"]

    # P/E
    out["REF_PE"] = out["TR.PE"]
    out["SELF_PE_Price_div_EPSAct"] = safe_div(price, out["TR.EPSActValue"])
    out["SELF_PE_Price_div_EPSFR"] = safe_div(price, out["TR.EPSFRActValue"])
    out["SELF_PE_MktCap_div_NetInc"] = safe_div(mktcap, out["TR.F.NetIncAfterTax"])
    out["SELF_PE_Price_div_NIperSh"] = safe_div(
        price, safe_div(out["TR.F.NetIncAfterTax"], shares)
    )

    # P/CF
    out["REF_PCF"] = out["TR.PriceToCFPerShare"]
    out["REF_PCF_fiscal"] = out["TR.F.PricetoCFPPerShr"]
    out["SELF_PCF_MktCap_div_CF"] = safe_div(mktcap, out["TR.F.CF"])
    out["SELF_PCF_MktCap_div_OpCF"] = safe_div(mktcap, out["TR.F.NetCashFlowOp"])
    out["SELF_PCF_Price_div_CFperSh"] = safe_div(price, safe_div(out["TR.F.CF"], shares))
    out["SELF_PCF_Price_div_OpCFperSh"] = safe_div(
        price, safe_div(out["TR.F.NetCashFlowOp"], shares)
    )

    return out


def summarize_candidates(df, group_name, columns):
    rows = []
    n = len(df)
    for col in columns:
        s = df[col].replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "group": group_name,
                "column": col,
                "non_null": int(s.notna().sum()),
                "pct_nan": round((1 - s.notna().sum() / n) * 100, 2),
                "median": s.median(skipna=True),
            }
        )
    return pd.DataFrame(rows).sort_values(["pct_nan", "column"]).reset_index(drop=True)


def compare_to_reference(df, reference_col, candidate_cols, group_name):
    rows = []
    ref = df[reference_col].replace([np.inf, -np.inf], np.nan)

    for col in candidate_cols:
        cand = df[col].replace([np.inf, -np.inf], np.nan)
        valid = ref.notna() & cand.notna()
        n_overlap = int(valid.sum())

        if n_overlap == 0:
            rows.append(
                {
                    "group": group_name,
                    "reference": reference_col,
                    "candidate": col,
                    "overlap_n": 0,
                    "corr": np.nan,
                    "median_abs_diff": np.nan,
                    "median_abs_pct_diff": np.nan,
                }
            )
            continue

        ref_valid = ref[valid]
        cand_valid = cand[valid]

        abs_diff = (cand_valid - ref_valid).abs()
        abs_pct_diff = safe_div(abs_diff, ref_valid.abs())

        rows.append(
            {
                "group": group_name,
                "reference": reference_col,
                "candidate": col,
                "overlap_n": n_overlap,
                "corr": cand_valid.corr(ref_valid),
                "median_abs_diff": abs_diff.median(skipna=True),
                "median_abs_pct_diff": abs_pct_diff.median(skipna=True),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["median_abs_pct_diff", "candidate"], na_position="last"
    ).reset_index(drop=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rics, source = load_rics()
    print(f"RIC-lähde: {source}")
    print(f"RICs ({len(rics)}): {rics}\n")

    rd.open_session()
    try:
        daily_raw, field_meta = build_master(rics)
    finally:
        rd.close_session()

    daily_ffill = forward_fill_fundamentals(daily_raw)
    month_end = build_month_end(daily_ffill)
    results = compute_alternatives(month_end)

    pe_cols = [
        "REF_PE",
        "SELF_PE_Price_div_EPSAct",
        "SELF_PE_Price_div_EPSFR",
        "SELF_PE_MktCap_div_NetInc",
        "SELF_PE_Price_div_NIperSh",
    ]
    pcf_cols = [
        "REF_PCF",
        "REF_PCF_fiscal",
        "SELF_PCF_MktCap_div_CF",
        "SELF_PCF_MktCap_div_OpCF",
        "SELF_PCF_Price_div_CFperSh",
        "SELF_PCF_Price_div_OpCFperSh",
    ]

    coverage = pd.concat(
        [
            summarize_candidates(results, "PE", pe_cols),
            summarize_candidates(results, "PCF", pcf_cols),
        ],
        ignore_index=True,
    )

    comparisons = pd.concat(
        [
            compare_to_reference(
                results,
                "REF_PE",
                [c for c in pe_cols if c != "REF_PE"],
                "PE",
            ),
            compare_to_reference(
                results,
                "REF_PCF",
                [c for c in pcf_cols if c != "REF_PCF"],
                "PCF_vs_REF_PCF",
            ),
            compare_to_reference(
                results,
                "REF_PCF_fiscal",
                [c for c in pcf_cols if c != "REF_PCF_fiscal"],
                "PCF_vs_REF_PCF_fiscal",
            ),
        ],
        ignore_index=True,
    )

    raw_path = OUTPUT_DIR / "pe_pcf_alternatives_daily_raw.csv"
    me_path = OUTPUT_DIR / "pe_pcf_alternatives_monthend_raw.csv"
    result_path = OUTPUT_DIR / "pe_pcf_alternatives_monthend_results.csv"
    map_path = OUTPUT_DIR / "pe_pcf_alternatives_field_map.csv"
    coverage_path = OUTPUT_DIR / "pe_pcf_alternatives_coverage.csv"
    comparison_path = OUTPUT_DIR / "pe_pcf_alternatives_comparison.csv"

    daily_raw.to_csv(raw_path, index=False)
    month_end.to_csv(me_path, index=False)
    results.to_csv(result_path, index=False)
    field_meta.to_csv(map_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    comparisons.to_csv(comparison_path, index=False)

    print("\nTallennetut tiedostot:")
    print(f"  - daily raw:     {raw_path}")
    print(f"  - month-end raw: {me_path}")
    print(f"  - results:       {result_path}")
    print(f"  - field map:     {map_path}")
    print(f"  - coverage:      {coverage_path}")
    print(f"  - comparison:    {comparison_path}")

    print("\nCoverage (paras ensin):")
    print(coverage.to_string(index=False))

    print("\nVertailu Refinitiviin:")
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
