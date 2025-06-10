#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spacescope Data Fetcher

This module retrieves Filecoin network metrics (storage capacity, gas usage,
economics sector data, circulating supply, and onboarding by method) in date-range
chunks of up to 31 days, then merges them on `stat_date` for downstream analysis.

Usage:
    python spacescope_fetcher.py --start 2025-05-01 --end 2025-06-01 \
        --output ./metrics.csv

Author: Juan Pablo Madrigal Cianci (refactored by ChatGPT)
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import Callable, List, Union
from datetime import date,timedelta
import requests
import pandas as pd

# --- Constants --------------------------------------------------------------
BASE_URL = "https://api.spacescope.io/v2"
ENDPOINTS = {
    'capacity':    "/power/network_storage_capacity",
    'supply':      "/circulating_supply/circulating_supply",
    'sector':      "/economics/sector_economics",
    'gas':         "/gas/daily_gas_usage_in_units",
    'onboarding':  "/power/daily_power_onboarding_by_method",
}

MAX_DAYS = 31
EIB = 1 << 60  # bytes → exbibytes
NANO = 1e-9    # gas units conversion

# --- Logging setup ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Conversion functions --------------------------------------------------

def bytes_to_eib(value: float) -> float:
    """Convert bytes to exbibytes."""
    return value / EIB


def gas_to_units(value: float) -> float:
    """Convert raw gas value to unit scale."""
    return value * NANO

# --- HTTP client -----------------------------------------------------------
class SpacescopeClient:
    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("API token must be provided via SPACESCOPE_TOKEN")
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {token}'})

    def fetch(self, endpoint: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch raw JSON data from an endpoint between two dates.

        Returns:
            pd.DataFrame: DataFrame of the endpoint's `data` list.
        """
        url = f"{BASE_URL}{endpoint}"
        params = {'start_date': start, 'end_date': end}
        logger.info(f"Requesting %s from %s to %s", endpoint, start, end)
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return pd.DataFrame(resp.json().get('data', []))

# --- Chunked fetch ---------------------------------------------------------
def fetch_full(
    client: SpacescopeClient,
    endpoint: str,
    transform: Callable[[pd.DataFrame], pd.DataFrame],
    start_date: Union[str, datetime],
    end_date:   Union[str, datetime]
) -> pd.DataFrame:
    """
    Fetch data in ≤MAX_DAYS-day chunks and apply a transformation to each.

    Accepts start_date/end_date as ISO strings or datetime objects.
    """
    # Normalize inputs
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date)

    if end_date < start_date:
        raise ValueError("end_date must not precede start_date")

    cursor = start_date
    chunks: List[pd.DataFrame] = []

    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=MAX_DAYS - 1), end_date)
        df_raw = client.fetch(
            endpoint,
            start=cursor.date().isoformat(),
            end=chunk_end.date().isoformat()
        )
        if not df_raw.empty:
            chunks.append(transform(df_raw))
        cursor = chunk_end + timedelta(days=1)

    if not chunks:
        return pd.DataFrame(columns=['stat_date'])

    df_all = pd.concat(chunks, ignore_index=True)
    df_all.drop_duplicates(subset=['stat_date'], inplace=True)
    df_all.sort_values('stat_date', inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    return df_all

# --- Transformation functions ----------------------------------------------

def transform_capacity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['stat_date'] = pd.to_datetime(df['stat_date'])
    for c in ['total_qa_bytes_power', 'total_raw_bytes_power', 'baseline_power']:
        if c in df.columns:
            df[c] = df[c].astype(float).apply(bytes_to_eib)
    return df


def transform_supply(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['stat_date'] = pd.to_datetime(df['stat_date'])
    return df


def transform_sector(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['stat_date'] = pd.to_datetime(df['stat_date'])
    return df


def transform_gas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['stat_date'] = pd.to_datetime(df['stat_date'])
    for c in df.columns:
        if c != 'stat_date':
            df[c] = df[c].astype(float).apply(gas_to_units)
    return df


def transform_onboarding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['stat_date'] = pd.to_datetime(df['stat_date'])
    # Convert numeric columns to float; leave method columns as-is
    for c in df.columns:
        if c != 'stat_date' and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(float)
    return df

# --- Merging ---------------------------------------------------------------

def merge_on_date(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge a list of DataFrames on 'stat_date' with an inner join."""
    from functools import reduce
    return reduce(lambda left, right: pd.merge(left, right, on='stat_date', how='inner'), dfs)


def get_all(start,end=None):
    if end==None:
        end=str(date.today() - timedelta(days=1))

    token = os.getenv("SPACESCOPE_TOKEN") or "ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ"
    client = SpacescopeClient(token)
    
    dfs = [
        fetch_full(client, ENDPOINTS['capacity'],   transform_capacity,   start, end),
        fetch_full(client, ENDPOINTS['supply'],     transform_supply,     start, end),
        fetch_full(client, ENDPOINTS['sector'],     transform_sector,     start, end),
        fetch_full(client, ENDPOINTS['gas'],        transform_gas,        start, end),
        fetch_full(client, ENDPOINTS['onboarding'], transform_onboarding, start, end),
    ]
    
    return  merge_on_date(dfs)
    




