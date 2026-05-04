#!/usr/bin/env python3
"""
Fetch historical index constituents with LSEG Tick History REST API.

This uses the DataScope Select / Tick History endpoint:
    /RestApi/v1/Search/HistoricalChainResolution

Credentials are read from environment variables. Either provide a token:
    DSS_TOKEN / LSEG_DSS_TOKEN / DATASCOPE_TOKEN

or provide username and password:
    DSS_USERNAME / LSEG_DSS_USERNAME / DATASCOPE_USERNAME
    DSS_PASSWORD / LSEG_DSS_PASSWORD / DATASCOPE_PASSWORD

Example:
    python3 tick_universe.py --start 2010-01-01 --end 2026-01-01
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "https://selectapi.datascope.lseg.com/RestApi/v1"
DEFAULT_CHAIN_RIC = "0#.STOXX"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"


def _env_first(names: Iterable[str]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return None


def _parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def _snapshot_range(as_of_date: dt.date) -> dict[str, str]:
    start = dt.datetime.combine(as_of_date, dt.time.min)
    end = start + dt.timedelta(days=1)
    return {
        "Start": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "End": end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }


def _auth_header_value(token: str) -> str:
    token = token.strip()
    lower = token.lower()
    if lower.startswith("token ") or lower.startswith("bearer "):
        return token
    return f"Token {token}"


def request_token(
    session: requests.Session,
    base_url: str,
    username: str,
    password: str,
    timeout: int,
) -> str:
    url = f"{base_url.rstrip('/')}/Authentication/RequestToken"
    headers = {
        "Prefer": "respond-async",
        "Content-Type": "application/json; odata.metadata=minimal",
    }
    payload = {"Credentials": {"Username": username, "Password": password}}
    response = session.post(url, json=payload, headers=headers, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            "DSS authentication failed "
            f"(HTTP {response.status_code}): {response.text[:1000]}"
        )
    data = response.json()
    token = data.get("value")
    if not token:
        raise RuntimeError(f"DSS authentication response did not contain a token: {data}")
    return token


def load_token(session: requests.Session, base_url: str, timeout: int) -> str:
    token = _env_first(["DSS_TOKEN", "LSEG_DSS_TOKEN", "DATASCOPE_TOKEN"])
    if token:
        return token

    username = _env_first(["DSS_USERNAME", "LSEG_DSS_USERNAME", "DATASCOPE_USERNAME"])
    password = _env_first(["DSS_PASSWORD", "LSEG_DSS_PASSWORD", "DATASCOPE_PASSWORD"])
    if not username or not password:
        raise RuntimeError(
            "Missing DSS/Tick History credentials. Set either DSS_TOKEN, or "
            "DSS_USERNAME and DSS_PASSWORD. Aliases LSEG_DSS_* and DATASCOPE_* "
            "are also supported."
        )
    return request_token(session, base_url, username, password, timeout)


def request_json(
    session: requests.Session,
    url: str,
    headers: dict[str, str],
    payload: dict,
    timeout: int,
    async_timeout: int,
    poll_seconds: int,
) -> dict:
    response = session.post(url, json=payload, headers=headers, timeout=timeout)
    if response.status_code == 202:
        location = response.headers.get("Location") or response.headers.get("location")
        if not location:
            raise RuntimeError(
                "Async DSS response did not include a Location header: "
                f"{response.text[:1000]}"
            )
        monitor_url = urljoin(url, location)
        deadline = time.monotonic() + async_timeout
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(f"DSS async request did not complete: {monitor_url}")
            time.sleep(poll_seconds)
            response = session.get(monitor_url, headers=headers, timeout=timeout)
            if response.status_code != 202:
                break

    if response.status_code != 200:
        raise RuntimeError(
            "DSS request failed "
            f"(HTTP {response.status_code}): {response.text[:1000]}"
        )
    return response.json()


def fetch_chain_snapshot(
    session: requests.Session,
    base_url: str,
    token: str,
    chain_ric: str,
    as_of_date: dt.date,
    timeout: int,
    async_timeout: int,
    poll_seconds: int,
) -> pd.DataFrame:
    url = f"{base_url.rstrip('/')}/Search/HistoricalChainResolution"
    headers = {
        "Prefer": "respond-async",
        "Content-Type": "application/json; odata.metadata=minimal",
        "Accept-Charset": "UTF-8",
        "Authorization": _auth_header_value(token),
    }
    payload = {
        "Request": {
            "ChainRics": [chain_ric],
            "Range": _snapshot_range(as_of_date),
        }
    }
    data = request_json(
        session=session,
        url=url,
        headers=headers,
        payload=payload,
        timeout=timeout,
        async_timeout=async_timeout,
        poll_seconds=poll_seconds,
    )

    rows: list[dict] = []
    for chain in data.get("value", []):
        for constituent in chain.get("Constituents", []) or []:
            row = dict(constituent)
            row["Date"] = as_of_date.isoformat()
            row["ChainRIC"] = chain.get("Identifier")
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["Date", "ChainRIC", "Identifier"])

    df = pd.DataFrame(rows)
    if "Identifier" in df.columns:
        df = df.drop_duplicates(subset=["Date", "ChainRIC", "Identifier"])
    return df.sort_values(["Date", "Identifier"]).reset_index(drop=True)


def build_checkpoint_dates(start: str, end: str, frequency: str) -> list[dt.date]:
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    if end_date < start_date:
        raise ValueError("--end must be on or after --start")

    dates = {start_date, end_date}
    if frequency == "start-end":
        return sorted(dates)

    freq_map = {
        "month-end": "M",
        "quarter-end": "Q",
        "year-end": "A",
    }
    for ts in pd.date_range(start=start_date, end=end_date, freq=freq_map[frequency]):
        dates.add(ts.date())
    return sorted(dates)


def extract_ric_set(
    snapshot: pd.DataFrame,
    include_dot_rics: bool,
    valid_only: bool,
) -> set[str]:
    if snapshot.empty or "Identifier" not in snapshot.columns:
        return set()

    df = snapshot.copy()
    if valid_only and "Status" in df.columns:
        df = df[df["Status"].astype(str).str.casefold().eq("valid")]

    rics = df["Identifier"].dropna().astype(str).str.strip()
    rics = rics[rics.ne("")]
    if not include_dot_rics:
        rics = rics[~rics.str.startswith(".")]
        rics = rics[~rics.str.startswith("0#")]
    return set(rics.tolist())


def save_outputs(
    snapshots: pd.DataFrame,
    ric_sets: dict[str, set[str]],
    output_dir: Path,
    chain_ric: str,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_chain = chain_ric.replace("#", "").replace(".", "").replace(":", "")
    snapshots_path = output_dir / f"tick_{safe_chain.lower()}_snapshots.csv"
    survivors_path = output_dir / f"tick_{safe_chain.lower()}_survivor_universe.csv"
    diagnostics_path = output_dir / f"tick_{safe_chain.lower()}_diagnostics.csv"

    snapshots.to_csv(snapshots_path, index=False)

    if ric_sets:
        survivors = sorted(set.intersection(*ric_sets.values()))
    else:
        survivors = []
    pd.DataFrame({"RIC": survivors}).to_csv(survivors_path, index=False)

    diagnostics = pd.DataFrame(
        [
            {
                "Date": date,
                "n_constituents": len(rics),
            }
            for date, rics in sorted(ric_sets.items())
        ]
    )
    diagnostics.to_csv(diagnostics_path, index=False)
    return snapshots_path, survivors_path, diagnostics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a STOXX universe from Tick History HistoricalChainResolution."
    )
    parser.add_argument("--chain", default=DEFAULT_CHAIN_RIC)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument(
        "--frequency",
        choices=["start-end", "month-end", "quarter-end", "year-end"],
        default="start-end",
        help=(
            "Checkpoint frequency. start-end is a light access test; month-end "
            "is stronger but still not a proof of uninterrupted membership."
        ),
    )
    parser.add_argument(
        "--date",
        action="append",
        help="Explicit as-of date YYYY-MM-DD. Can be repeated; overrides --frequency.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--base-url",
        default=os.getenv("DSS_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--async-timeout", type=int, default=600)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument(
        "--max-dates",
        type=int,
        default=40,
        help="Safety cap for API calls. Increase intentionally for monthly runs.",
    )
    parser.add_argument(
        "--include-dot-rics",
        action="store_true",
        help="Keep identifiers starting with '.' or '0#' in the filtered RIC sets.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Keep constituents whose Status is not Valid.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.date:
        dates = sorted({_parse_date(value) for value in args.date})
    else:
        dates = build_checkpoint_dates(args.start, args.end, args.frequency)

    if len(dates) > args.max_dates:
        print(
            f"Refusing to call HistoricalChainResolution for {len(dates)} dates "
            f"because --max-dates is {args.max_dates}. Increase --max-dates if "
            "this is intentional.",
            file=sys.stderr,
        )
        return 2

    with requests.Session() as session:
        token = load_token(session, args.base_url, args.timeout)
        snapshots = []
        ric_sets: dict[str, set[str]] = {}

        for as_of_date in dates:
            print(f"Fetching {args.chain} as of {as_of_date}...")
            snapshot = fetch_chain_snapshot(
                session=session,
                base_url=args.base_url,
                token=token,
                chain_ric=args.chain,
                as_of_date=as_of_date,
                timeout=args.timeout,
                async_timeout=args.async_timeout,
                poll_seconds=args.poll_seconds,
            )
            snapshots.append(snapshot)
            ric_sets[as_of_date.isoformat()] = extract_ric_set(
                snapshot,
                include_dot_rics=args.include_dot_rics,
                valid_only=not args.include_invalid,
            )
            print(
                f"  raw rows={len(snapshot)}, "
                f"filtered RICs={len(ric_sets[as_of_date.isoformat()])}"
            )

    all_snapshots = pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame()
    snapshots_path, survivors_path, diagnostics_path = save_outputs(
        snapshots=all_snapshots,
        ric_sets=ric_sets,
        output_dir=args.output_dir,
        chain_ric=args.chain,
    )

    survivors_count = (
        len(set.intersection(*ric_sets.values())) if ric_sets else 0
    )
    print("\nDone")
    print(f"Checkpoints: {len(ric_sets)}")
    print(f"Intersection across selected checkpoints: {survivors_count}")
    print(f"Snapshots:   {snapshots_path}")
    print(f"Survivors:   {survivors_path}")
    print(f"Diagnostics: {diagnostics_path}")
    if args.frequency != "month-end" and not args.date:
        print(
            "\nNote: start-end mode only proves membership at the selected dates, "
            "not uninterrupted membership throughout the whole interval."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
