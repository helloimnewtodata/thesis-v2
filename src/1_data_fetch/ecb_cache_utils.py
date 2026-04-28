import io
import os
import ssl
import subprocess
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "01_raw" / "outputs"
ECB_CACHE_DIR = OUTPUT_DIR / "ecb_cache"
ECB_AAA_CACHE_PATH = ECB_CACHE_DIR / "ecb_aaa_10y.csv"
ECB_ALL_CACHE_PATH = ECB_CACHE_DIR / "ecb_all_10y.csv"

DEFAULT_START_DATE = os.getenv("START_DATE", "2010-01-01")
DEFAULT_END_DATE = os.getenv("END_DATE", "2026-03-05")
DEFAULT_REFRESH_ECB_CACHE = os.getenv("REFRESH_ECB_CACHE", "0") == "1"
DEFAULT_ECB_SSL_VERIFY = os.getenv("ECB_SSL_VERIFY", "1") != "0"


def build_ecb_urls(start_date: str, end_date: str):
    aaa_url = (
        "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y"
        f"?startPeriod={start_date}&endPeriod={end_date}&format=csvdata"
    )
    all_url = (
        "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_C.SV_C_YM.SR_10Y"
        f"?startPeriod={start_date}&endPeriod={end_date}&format=csvdata"
    )
    return aaa_url, all_url


def load_cached_ecb_series(
    cache_path: Path,
    value_col: str,
    start_date: str | None = None,
    end_date: str | None = None,
):
    if not cache_path.exists():
        return None

    df = pd.read_csv(cache_path)
    if "Date" not in df.columns or value_col not in df.columns:
        raise ValueError(f"ECB-cache tiedostosta puuttuu odotetut sarakkeet: {cache_path}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    if start_date is not None and end_date is not None:
        df = df.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)].copy()

    print(f"Loaded ECB cache: {cache_path} ({len(df)} riviä)")
    return df


def download_bytes_with_urlopen(url: str, ssl_verify: bool):
    ssl_context = None
    if not ssl_verify:
        ssl_context = ssl._create_unverified_context()
        print("Warning: ECB_SSL_VERIFY=0, SSL-sertifikaatin verifiointi ohitettu ECB-haussa.")

    with urlopen(url, context=ssl_context) as response:
        return response.read()


def download_bytes_with_curl(url: str, ssl_verify: bool):
    cmd = ["curl", "-fsSL", url]
    if not ssl_verify:
        cmd.insert(1, "-k")
        print("Warning: curl -k kaytossa ECB-haussa.")

    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout


def download_csv_bytes(url: str, ssl_verify: bool):
    first_exc = None

    try:
        return download_bytes_with_urlopen(url, ssl_verify=ssl_verify)
    except Exception as exc:
        first_exc = exc
        print(f"urlopen-ECB-haku epaonnistui: {exc}")

    try:
        return download_bytes_with_curl(url, ssl_verify=ssl_verify)
    except Exception as exc:
        raise RuntimeError(f"Myos curl-pohjainen ECB-haku epaonnistui: {exc}") from first_exc


def download_and_cache_ecb_series(url: str, cache_path: Path, value_col: str, ssl_verify: bool):
    raw = pd.read_csv(io.BytesIO(download_csv_bytes(url, ssl_verify=ssl_verify)))
    df = raw[["TIME_PERIOD", "OBS_VALUE"]].copy()
    df.columns = ["Date", value_col]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Saved ECB cache: {cache_path} ({len(df)} riviä)")
    return df.set_index("Date")


def get_ecb_series(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    refresh_cache: bool = DEFAULT_REFRESH_ECB_CACHE,
    ssl_verify: bool = DEFAULT_ECB_SSL_VERIFY,
):
    aaa_url, all_url = build_ecb_urls(start_date, end_date)

    if not refresh_cache:
        cached_aaa = load_cached_ecb_series(ECB_AAA_CACHE_PATH, "aaa_yield", start_date, end_date)
        cached_all = load_cached_ecb_series(ECB_ALL_CACHE_PATH, "all_yield", start_date, end_date)
        if cached_aaa is not None and cached_all is not None:
            return cached_aaa, cached_all

    try:
        ecb_aaa = download_and_cache_ecb_series(
            aaa_url,
            ECB_AAA_CACHE_PATH,
            "aaa_yield",
            ssl_verify=ssl_verify,
        )
        ecb_all = download_and_cache_ecb_series(
            all_url,
            ECB_ALL_CACHE_PATH,
            "all_yield",
            ssl_verify=ssl_verify,
        )
    except Exception as exc:
        cached_aaa = load_cached_ecb_series(ECB_AAA_CACHE_PATH, "aaa_yield", start_date, end_date)
        cached_all = load_cached_ecb_series(ECB_ALL_CACHE_PATH, "all_yield", start_date, end_date)
        if cached_aaa is not None and cached_all is not None:
            print(f"ECB download epaonnistui, kaytetaan paikallista cachea: {exc}")
            return cached_aaa, cached_all

        helper_script = ROOT / "src" / "1_data_fetch" / "fetch_ecb_cache.py"
        raise RuntimeError(
            "ECB-datan lataus epaonnistui eika paikallista cachea loytynyt. "
            f"Aja ensin {helper_script} tai tallenna tiedostot polkuihin "
            f"{ECB_AAA_CACHE_PATH} ja {ECB_ALL_CACHE_PATH}. "
            "Tarvittaessa ohita sertifikaattitarkistus bootstrap-ajossa asetuksella ECB_SSL_VERIFY=0."
        ) from exc

    ecb_aaa = ecb_aaa.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)].copy()
    ecb_all = ecb_all.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)].copy()
    return ecb_aaa, ecb_all
