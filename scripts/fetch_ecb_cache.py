import os

from src.ecb_cache_utils import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    ECB_AAA_CACHE_PATH,
    ECB_ALL_CACHE_PATH,
    ECB_CACHE_DIR,
    get_ecb_series,
)


START_DATE = os.getenv("START_DATE", DEFAULT_START_DATE)
END_DATE = os.getenv("END_DATE", DEFAULT_END_DATE)
ECB_SSL_VERIFY = os.getenv("ECB_SSL_VERIFY", "1") != "0"


def main():
    ECB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "Fetching ECB cache "
        f"from {START_DATE} to {END_DATE} "
        f"(ssl_verify={'on' if ECB_SSL_VERIFY else 'off'})"
    )

    ecb_aaa, ecb_all = get_ecb_series(
        start_date=START_DATE,
        end_date=END_DATE,
        refresh_cache=True,
        ssl_verify=ECB_SSL_VERIFY,
    )

    print(f"\nSaved: {ECB_AAA_CACHE_PATH} ({len(ecb_aaa)} riviä)")
    print(f"Saved: {ECB_ALL_CACHE_PATH} ({len(ecb_all)} riviä)")


if __name__ == "__main__":
    main()
