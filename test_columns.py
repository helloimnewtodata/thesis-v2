import refinitiv.data as rd
import pandas as pd

rd.open_session()
df = pd.read_csv("data/01_raw/smoke_stocks.csv")
rics = sorted(df["Instrument"].dropna().unique().tolist())[:3]
params = {"SDate": "2020-01-01", "EDate": "2025-12-31", "Frq": "D", "Curn": "EUR"}

# NetIncome-vaihtoehdot
ni_fields = [
    "TR.F.NetIncAfterTax",
    "TR.F.NetIncBeforeExtra",
    "TR.NetIncome",
    "TR.F.NetIncomeToCompany",
    "TR.F.IncBefTax",
]

# OpCF-vaihtoehdot
opcf_fields = [
    "TR.F.CashFlowOps",
    "TR.OperCashFlow",
    "TR.F.NetCashFlowOp",
    "TR.F.CashFlowOpActValue",
]

for label, candidates in [("NET INCOME", ni_fields), ("OPERATING CF", opcf_fields)]:
    print(f"\n--- {label} ---")
    for field in candidates:
        try:
            r = rd.get_data(rics[:1], ["TR.PriceClose.date", field], parameters=params)
            cols = [c for c in r.columns if c not in ("Instrument", "Date")]
            if cols:
                nn = r[cols[0]].notna().sum()
                print(f"  OK  {field:<40} → {cols[0]:<50} non-NaN: {nn}")
            else:
                print(f"  X   {field:<40} → ei datasaraketta")
        except Exception as e:
            print(f"  X   {field:<40} → {type(e).__name__}: {str(e)[:60]}")

rd.close_session()
