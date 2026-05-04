import os
from pathlib import Path

import numpy as np
import pandas as pd
import refinitiv.data as rd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
UNIVERSE_PATH = ROOT / "data" / "stoxx600_universe.csv"
OUTPUT_PATH = SCRIPT_DIR / "pe_pcf_fixed.csv"
MONTHLY_OUTPUT_PATH = SCRIPT_DIR / "pe_pcf_fixed_monthly_end.csv"

START_DATE = os.getenv("START_DATE", "2010-01-01")
END_DATE = os.getenv("END_DATE", "2025-12-31")
STOCK_LIMIT = int(os.getenv("STOCK_LIMIT", "500"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "123"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "25"))
MONTHLY_LOOKBACK_DAYS = int(os.getenv("MONTHLY_LOOKBACK_DAYS", "3"))

PARAMS = {"SDate": START_DATE, "EDate": END_DATE, "Frq": "D", "Curn": "EUR"}


def build_quarterly_params(start_date):
    start_quarter = pd.Timestamp(start_date).to_period("Q")
    current_quarter = pd.Timestamp.today().to_period("Q")
    lookback_quarters = max(19, (current_quarter.ordinal - start_quarter.ordinal) + 4)
    return {"Period": f"FQ-{lookback_quarters}:FQ0", "Frq": "FQ", "Curn": "EUR"}


QUARTERLY_PARAMS = build_quarterly_params(START_DATE)


def coerce_dates_and_numeric(df, numeric_cols):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def drop_missing_dates(df, label):
    missing_dates = df["Date"].isna().sum()
    if missing_dates:
        print(f"Poistetaan {label}-datasta {missing_dates} riviä ilman Date-arvoa")
        df = df.dropna(subset=["Date"])
    return df.reset_index(drop=True)


def chunk_list(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def normalize_response_frame(raw_df, columns, instrument_hint=None):
    df = raw_df.copy()

    if len(df.columns) == 0:
        return pd.DataFrame(columns=columns)

    if len(df.columns) == len(columns):
        df.columns = columns
        return df

    if len(df.columns) == len(columns) - 1 and columns[0] == "Instrument":
        if instrument_hint is None:
            raise ValueError("Instrument-sarake puuttuu eika instrument_hint ole saatavilla")
        df.columns = columns[1:]
        df.insert(0, "Instrument", instrument_hint)
        return df

    raise ValueError(
        f"Odottamaton sarakemaara {len(df.columns)}. Odotettiin {len(columns)} tai {len(columns) - 1}."
    )


def fetch_table_for_stocks(stocks, fields, parameters, columns, label, batch_size=BATCH_SIZE):
    parts = []
    failed_stocks = []
    batches = list(chunk_list(stocks, batch_size))

    for batch_idx, batch in enumerate(batches, start=1):
        try:
            raw = rd.get_data(universe=batch, fields=fields, parameters=parameters)
            part = normalize_response_frame(
                raw,
                columns=columns,
                instrument_hint=batch[0] if len(batch) == 1 else None,
            )
            parts.append(part)
            print(f"{label}: batch {batch_idx}/{len(batches)} valmis ({len(batch)} osaketta)")
        except Exception as exc:
            print(
                f"{label}: batch {batch_idx}/{len(batches)} epaonnistui "
                f"({len(batch)} osaketta), kokeillaan osake kerrallaan: {exc}"
            )
            for stock in batch:
                try:
                    raw = rd.get_data(universe=[stock], fields=fields, parameters=parameters)
                    part = normalize_response_frame(raw, columns=columns, instrument_hint=stock)
                    parts.append(part)
                except Exception as stock_exc:
                    failed_stocks.append(stock)
                    print(f"{label}: osake {stock} epaonnistui: {stock_exc}")

    if failed_stocks:
        print(f"{label}: data puuttui {len(failed_stocks)} osakkeelta: {failed_stocks}")

    if not parts:
        return pd.DataFrame(columns=columns)

    return pd.concat(parts, ignore_index=True)


def merge_asof_by_instrument(left, right, value_cols):
    left = left.sort_values(["Instrument", "Date"]).reset_index(drop=True)
    right = right.sort_values(["Instrument", "Date"]).reset_index(drop=True)

    parts = []
    for instrument in left["Instrument"].dropna().unique():
        left_part = left[left["Instrument"] == instrument].copy().reset_index(drop=True)
        right_part = (
            right[right["Instrument"] == instrument][["Date", *value_cols]]
            .copy()
            .reset_index(drop=True)
        )

        if len(right_part) > 0:
            merged = pd.merge_asof(left_part, right_part, on="Date", direction="backward")
        else:
            merged = left_part
            for col in value_cols:
                merged[col] = np.nan

        parts.append(merged)

    if not parts:
        return left

    return pd.concat(parts, ignore_index=True)


def fetch_pe_frame(stocks):
    pe_cols = [
        "Instrument",
        "Date",
        "Price",
        "MktCap",
        "Shares",
        "REF_PE",
        "EPSAct",
        "EPSfr",
        "NetIncome",
        "EPSAct_LTM",
    ]
    df_pe = fetch_table_for_stocks(
        stocks=stocks,
        fields=[
            "TR.PriceClose.date",
            "TR.PriceClose",
            "TR.CompanyMarketCap",
            "TR.SharesOutstanding",
            "TR.PE",
            "TR.EPSActValue",
            "TR.EPSFRActValue",
            "TR.F.NetIncAfterTax",
            "TR.EPSActValue(Period=LTM)",
        ],
        parameters=PARAMS,
        columns=pe_cols,
        label="PE",
    )
    df_pe = coerce_dates_and_numeric(
        df_pe,
        ["Price", "MktCap", "Shares", "REF_PE", "EPSAct", "EPSfr", "NetIncome", "EPSAct_LTM"],
    )
    df_pe = drop_missing_dates(df_pe, "PE")
    df_pe = df_pe.sort_values(["Instrument", "Date"]).drop_duplicates(
        subset=["Instrument", "Date"],
        keep="last",
    ).reset_index(drop=True)

    df_pe["EPSAct_ff"] = df_pe.groupby("Instrument")["EPSAct"].ffill()
    df_pe["EPSfr_ff"] = df_pe.groupby("Instrument")["EPSfr"].ffill()
    df_pe["NI_ff"] = df_pe.groupby("Instrument")["NetIncome"].ffill()
    df_pe["Shares_ff"] = df_pe.groupby("Instrument")["Shares"].ffill()
    df_pe["EPS_from_NI"] = df_pe["NI_ff"] / df_pe["Shares_ff"]
    df_pe["Implied_EPS_LTM_formula"] = df_pe["EPSAct_LTM"].round(3)
    df_pe["Implied_EPS_REF"] = df_pe["Price"] / df_pe["REF_PE"]

    q1 = fetch_table_for_stocks(
        stocks=stocks,
        fields=["TR.EPSActValue.periodenddate", "TR.EPSActValue"],
        parameters=QUARTERLY_PARAMS,
        columns=["Instrument", "Date", "EPSAct_Q"],
        label="kvartaali-EPSAct",
    )
    q1 = coerce_dates_and_numeric(q1, ["EPSAct_Q"])
    q1 = drop_missing_dates(q1, "kvartaali-EPSAct")

    q2 = fetch_table_for_stocks(
        stocks=stocks,
        fields=["TR.EPSFRActValue.periodenddate", "TR.EPSFRActValue"],
        parameters=QUARTERLY_PARAMS,
        columns=["Instrument", "Date", "EPSfr_Q"],
        label="kvartaali-EPSfr",
    )
    q2 = coerce_dates_and_numeric(q2, ["EPSfr_Q"])
    q2 = drop_missing_dates(q2, "kvartaali-EPSfr")

    q3 = fetch_table_for_stocks(
        stocks=stocks,
        fields=["TR.F.NetIncAfterTax.periodenddate", "TR.F.NetIncAfterTax"],
        parameters=QUARTERLY_PARAMS,
        columns=["Instrument", "Date", "NetIncome_Q"],
        label="kvartaali-NetIncome",
    )
    q3 = coerce_dates_and_numeric(q3, ["NetIncome_Q"])
    q3 = drop_missing_dates(q3, "kvartaali-NetIncome")

    q = (
        q1.merge(q2, on=["Instrument", "Date"], how="outer")
        .merge(q3, on=["Instrument", "Date"], how="outer")
        .sort_values(["Instrument", "Date"])
        .reset_index(drop=True)
    )
    q["TTM_EPSAct"] = (
        q.groupby("Instrument")["EPSAct_Q"]
        .rolling(4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )
    q["TTM_EPSfr"] = (
        q.groupby("Instrument")["EPSfr_Q"]
        .rolling(4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )
    q["TTM_NetIncome"] = (
        q.groupby("Instrument")["NetIncome_Q"]
        .rolling(4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )
    q_ttm = q[
        ["Instrument", "Date", "TTM_EPSAct", "TTM_EPSfr", "TTM_NetIncome"]
    ].dropna(how="all", subset=["TTM_EPSAct", "TTM_EPSfr", "TTM_NetIncome"])
    df_pe = merge_asof_by_instrument(
        df_pe,
        q_ttm,
        ["TTM_EPSAct", "TTM_EPSfr", "TTM_NetIncome"],
    )
    df_pe["TTM_EPS_from_NI"] = df_pe["TTM_NetIncome"] / df_pe["Shares_ff"]

    extra = fetch_table_for_stocks(
        stocks=stocks,
        fields=["TR.PriceClose.date", "TR.EPSFRActValue(Period=LTM)"],
        parameters=PARAMS,
        columns=["Instrument", "Date", "EPSFR_LTM"],
        label="EPSFR_LTM",
    )
    extra = coerce_dates_and_numeric(extra, ["EPSFR_LTM"])
    extra = drop_missing_dates(extra, "EPSFR_LTM")
    extra = extra.sort_values(["Instrument", "Date"]).drop_duplicates(
        subset=["Instrument", "Date"],
        keep="last",
    ).reset_index(drop=True)

    if len(extra) > 0:
        df_pe = df_pe.merge(extra, on=["Instrument", "Date"], how="left")
    else:
        df_pe["EPSFR_LTM"] = np.nan

    df_pe["Implied_EPS_trial"] = (
        df_pe["EPSFR_LTM"]
        .combine_first(df_pe["TTM_EPSfr"])
        .combine_first(df_pe["EPSfr"])
        .combine_first(df_pe["EPSAct"])
        .combine_first(df_pe["Implied_EPS_LTM_formula"])
        .combine_first(df_pe["TTM_EPSAct"])
        .combine_first(df_pe["TTM_EPS_from_NI"])
    )
    df_pe["Implied_EPS_trial_source"] = np.select(
        [
            df_pe["EPSFR_LTM"].notna(),
            df_pe["EPSFR_LTM"].isna() & df_pe["TTM_EPSfr"].notna(),
            df_pe["EPSFR_LTM"].isna() & df_pe["TTM_EPSfr"].isna() & df_pe["EPSfr"].notna(),
            df_pe["EPSFR_LTM"].isna() & df_pe["TTM_EPSfr"].isna() & df_pe["EPSfr"].isna() & df_pe["EPSAct"].notna(),
            df_pe["EPSFR_LTM"].isna() & df_pe["TTM_EPSfr"].isna() & df_pe["EPSfr"].isna() & df_pe["EPSAct"].isna() & df_pe["Implied_EPS_LTM_formula"].notna(),
            df_pe["EPSFR_LTM"].isna() & df_pe["TTM_EPSfr"].isna() & df_pe["EPSfr"].isna() & df_pe["EPSAct"].isna() & df_pe["Implied_EPS_LTM_formula"].isna() & df_pe["TTM_EPSAct"].notna(),
            df_pe["EPSFR_LTM"].isna() & df_pe["TTM_EPSfr"].isna() & df_pe["EPSfr"].isna() & df_pe["EPSAct"].isna() & df_pe["Implied_EPS_LTM_formula"].isna() & df_pe["TTM_EPSAct"].isna() & df_pe["TTM_EPS_from_NI"].notna(),
        ],
        ["EPSFR_LTM", "TTM_EPSfr", "EPSfr", "EPSAct", "Implied_EPS_LTM_formula", "TTM_EPSAct", "TTM_EPS_from_NI"],
        default=None,
    )
    epsfr_rows = int(df_pe["EPSFR_LTM"].notna().sum())
    fallback_rows = int((df_pe["EPSFR_LTM"].isna() & df_pe["Implied_EPS_trial"].notna()).sum())
    print(
        "Kaytossa oleva trial-sarja: ensisijaisesti EPSFR_LTM, fallback-ketjulla notebookin PE-ehdokkaat "
        f"(EPSFR_LTM {epsfr_rows} rivilla, fallback {fallback_rows} rivilla)"
    )

    df_pe["PE_trial"] = df_pe["Price"] / df_pe["Implied_EPS_trial"]

    return df_pe[["Instrument", "Date", "REF_PE", "PE_trial"]]


def fetch_pcf_frame(stocks):
    pcf_cols = [
        "Instrument",
        "Date",
        "Price",
        "MktCap",
        "REF_PCF",
        "OpCF",
        "CFPS_Act",
        "OpCFPS",
    ]
    df_pcf = fetch_table_for_stocks(
        stocks=stocks,
        fields=[
            "TR.PriceClose.date",
            "TR.PriceClose",
            "TR.CompanyMarketCap",
            "TR.PriceToCFPerShare",
            "TR.F.NetCashFlowOp",
            "TR.CFPSActValue",
            "TR.F.NetCFOpPerShr",
        ],
        parameters={**PARAMS, "Period": "LTM"},
        columns=pcf_cols,
        label="PCF",
    )
    df_pcf = coerce_dates_and_numeric(
        df_pcf,
        ["Price", "MktCap", "REF_PCF", "OpCF", "CFPS_Act", "OpCFPS"],
    )
    df_pcf = drop_missing_dates(df_pcf, "PCF")
    df_pcf = df_pcf.sort_values(["Instrument", "Date"]).drop_duplicates(
        subset=["Instrument", "Date"],
        keep="last",
    ).reset_index(drop=True)

    df_pcf["PCF_MktCap_div_OpCF"] = df_pcf["MktCap"] / df_pcf["OpCF"]
    df_pcf["PCF_Price_div_CFPS_Act"] = df_pcf["Price"] / df_pcf["CFPS_Act"]
    df_pcf["PCF_Price_div_OpCFPS"] = df_pcf["Price"] / df_pcf["OpCFPS"]

    return df_pcf[
        [
            "Instrument",
            "Date",
            "REF_PCF",
            "PCF_MktCap_div_OpCF",
            "PCF_Price_div_CFPS_Act",
            "PCF_Price_div_OpCFPS",
        ]
    ]


def build_monthly_end_snapshot(df, stocks, start_date, end_date, lookback_days):
    month_ends = (
        pd.period_range(
            start=pd.Timestamp(start_date).to_period("M"),
            end=pd.Timestamp(end_date).to_period("M"),
            freq="M",
        )
        .to_timestamp(how="end")
        .normalize()
    )

    grid = pd.MultiIndex.from_product(
        [stocks, month_ends],
        names=["Instrument", "Date"],
    ).to_frame(index=False)

    candidates = df.copy()
    candidates["MonthEnd"] = (
        candidates["Date"]
        .dt.to_period("M")
        .dt.to_timestamp(how="end")
        .dt.normalize()
    )
    candidates["LookbackDays"] = (candidates["MonthEnd"] - candidates["Date"]).dt.days
    candidates = candidates[
        candidates["LookbackDays"].between(0, lookback_days, inclusive="both")
    ].copy()
    candidates = candidates.sort_values(
        ["Instrument", "MonthEnd", "Date"],
        ascending=[True, True, False],
    )
    candidates = candidates.drop_duplicates(
        subset=["Instrument", "MonthEnd"],
        keep="first",
    )
    candidates = candidates.rename(columns={"Date": "SourceDate", "MonthEnd": "Date"})

    value_cols = [col for col in df.columns if col not in {"Instrument", "Date"}]
    monthly = grid.merge(
        candidates[["Instrument", "Date", "SourceDate", "LookbackDays", *value_cols]],
        on=["Instrument", "Date"],
        how="left",
    )
    monthly["SourceDate"] = pd.to_datetime(monthly["SourceDate"], errors="coerce")
    monthly["LookbackDays"] = monthly["LookbackDays"].astype("Int64")

    return monthly.sort_values(["Instrument", "Date"]).reset_index(drop=True)


def print_nan_summary(df, cols, label):
    print(f"\n{label} NaN-osuudet:")
    for col in cols:
        pct = df[col].isna().mean() * 100
        print(f"  {col}: {pct:.1f}%")


def main():
    universe = pd.read_csv(UNIVERSE_PATH)
    stock_count = min(STOCK_LIMIT, len(universe))
    stocks = universe["RIC"].sample(stock_count, random_state=RANDOM_STATE).tolist()

    print(
        f"Haetaan {len(stocks)} osaketta aikavalilta {START_DATE} - {END_DATE} "
        f"(batch-koko {BATCH_SIZE}, monthly lookback {MONTHLY_LOOKBACK_DAYS} paivaa)"
    )

    rd.open_session()
    try:
        pe = fetch_pe_frame(stocks)
        pcf = fetch_pcf_frame(stocks)

        df = pe.merge(pcf, on=["Instrument", "Date"], how="outer")
        df = drop_missing_dates(df, "yhdistetty")
        df = df.sort_values(["Instrument", "Date"]).reset_index(drop=True)

        keep_cols = [
            "Instrument",
            "Date",
            "REF_PE",
            "PE_trial",
            "REF_PCF",
            "PCF_MktCap_div_OpCF",
            "PCF_Price_div_CFPS_Act",
            "PCF_Price_div_OpCFPS",
        ]
        df = df[keep_cols]

        monthly_df = build_monthly_end_snapshot(
            df=df,
            stocks=stocks,
            start_date=START_DATE,
            end_date=END_DATE,
            lookback_days=MONTHLY_LOOKBACK_DAYS,
        )

        print(f"\nPaivadata: {len(df)} riviä, {df['Instrument'].nunique()} osaketta")
        print_nan_summary(df, keep_cols[2:], "Paivadata")

        monthly_value_cols = keep_cols[2:]
        monthly_hit_rate = monthly_df["SourceDate"].notna().mean() * 100
        print(
            f"\nKuukausiloppudata: {len(monthly_df)} riviä, "
            f"osumia {monthly_hit_rate:.1f}% riveista"
        )
        print_nan_summary(monthly_df, monthly_value_cols, "Kuukausiloppudata")

        df.to_csv(OUTPUT_PATH, index=False)
        monthly_df.to_csv(MONTHLY_OUTPUT_PATH, index=False)

        print(f"\n{OUTPUT_PATH.name} tallennettu ({len(df)} riviä)")
        print(f"{MONTHLY_OUTPUT_PATH.name} tallennettu ({len(monthly_df)} riviä)")
        print("\nKuukausiloppudatan esikatselu:")
        print(monthly_df.head(20).to_string(index=False))
    finally:
        rd.close_session()


if __name__ == "__main__":
    main()
