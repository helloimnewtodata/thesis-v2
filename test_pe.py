import refinitiv.data as rd
import pandas as pd

rd.open_session()
df = pd.read_csv("data/01_raw/smoke_stocks.csv")
rics = sorted(df["Instrument"].dropna().unique().tolist())[:10]
print(f"RICs: {rics}")

result = rd.get_data(
    rics,
    ["TR.PriceClose.date", "TR.PE"],
    parameters={"SDate": "2008-10-08", "EDate": "2025-12-31", "Frq": "D", "Curn": "EUR"},
)
print(f"Rows: {len(result)}, NaN: {result.iloc[:,2].isna().sum()} ({result.iloc[:,2].isna().mean()*100:.1f}%)")
print(result.head(20))
rd.close_session()
