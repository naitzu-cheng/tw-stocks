"""Download daily prices from Yahoo Finance and cache them in the database."""
from datetime import datetime, timedelta, timezone
import threading

import pandas as pd
import yfinance as yf
from sqlalchemy import inspect, text

from app.db import engine, table_name_for
from app.markets import alternate_taiwan_symbol, normalize_symbol

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
DEFAULT_PERIOD = "5y"          # 1 year is far too short to test a strategy
REFRESH_AFTER = timedelta(hours=6)

_fetched_at = {}
_lock = threading.Lock()
_batch_lock = threading.Lock()


def _download(symbol, period, interval):
    # auto_adjust=True: prices are adjusted for dividends and splits. Taiwan
    # stocks pay large dividends, so unadjusted closes understate returns.
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    return _clean(df)


def _clean(df):
    df = df[[c for c in PRICE_COLUMNS if c in df.columns]]
    df = df[df["Close"].notna() & (df["Close"] > 0)].copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    return df[~df.index.duplicated(keep="last")].sort_index()


def _download_many(symbols, period):
    """One batched Yahoo request for many tickers (much faster than a loop)."""
    raw = yf.download(symbols, period=period, interval="1d", auto_adjust=True,
                      group_by="ticker", threads=True, progress=False)
    out = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        for sym in symbols:
            if sym in raw.columns.get_level_values(0):
                df = raw[sym].dropna(how="all")
                if not df.empty and "Close" in df:
                    out[sym] = _clean(df)
    elif len(symbols) == 1:
        out[symbols[0]] = _clean(raw)
    return {k: v for k, v in out.items() if not v.empty}


def ensure_fresh_many(symbols, period=DEFAULT_PERIOD):
    """Refresh every stale symbol with one batched download.

    Symbols the batch misses are retried one by one. Returns a list of errors.
    """
    symbols = [normalize_symbol(s) for s in symbols]
    now = datetime.now(timezone.utc)
    with _lock:
        stale = [s for s in symbols
                 if not (_fetched_at.get(s) and now - _fetched_at[s] < REFRESH_AFTER)]
    if not stale:
        return []
    errors = []
    with _batch_lock:
        try:
            got = _download_many(stale, period)
        except Exception as exc:
            got, errors = {}, [f"批次下載失敗：{exc}"]
        for sym in stale:
            df = got.get(sym)
            if df is None:
                try:
                    df = _download(sym, period, "1d")
                except Exception as exc:
                    errors.append(f"{sym}：{exc}")
                    continue
            if df is None or df.empty:
                if not has_data(sym):
                    errors.append(f"{sym}：查無資料")
                continue
            store(sym, df)
            with _lock:
                _fetched_at[sym] = datetime.now(timezone.utc)
    return errors


def store(symbol, df):
    """Merge new rows into the cached table (new values win)."""
    table = table_name_for(symbol)
    existing = get_data(symbol, limit=None)
    if not existing.empty:
        # Re-downloaded adjusted prices supersede the cache, because a new
        # dividend changes every earlier adjusted close.
        if len(df) and df.index[0] <= existing.index[0]:
            combined = df
        else:
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = df
    combined.to_sql(table, con=engine, if_exists="replace", index=True)
    return len(df)


def fetch_and_store(symbol, period=DEFAULT_PERIOD, interval="1d"):
    """Download and store. Returns (resolved_symbol, rows_downloaded)."""
    symbol = normalize_symbol(symbol)
    df = _download(symbol, period, interval)
    if df.empty:
        alt = alternate_taiwan_symbol(symbol)
        if alt:
            alt_df = _download(alt, period, interval)
            if not alt_df.empty:
                symbol, df = alt, alt_df
    if df.empty:
        return symbol, 0
    rows = store(symbol, df)
    with _lock:
        _fetched_at[symbol] = datetime.now(timezone.utc)
    return symbol, rows


def ensure_fresh(symbol, period=DEFAULT_PERIOD):
    """Fetch only when the cache is missing or older than REFRESH_AFTER.

    Returns (resolved_symbol, error_message_or_None).
    """
    symbol = normalize_symbol(symbol)
    with _lock:
        last = _fetched_at.get(symbol)
    if last and datetime.now(timezone.utc) - last < REFRESH_AFTER and has_data(symbol):
        return symbol, None
    try:
        resolved, rows = fetch_and_store(symbol, period=period)
        if rows == 0 and not has_data(resolved):
            return resolved, f"Yahoo Finance 找不到 {symbol} 的資料"
        return resolved, None
    except Exception as exc:  # network errors: fall back to cache
        alt = alternate_taiwan_symbol(symbol)
        if not has_data(symbol) and alt and has_data(alt):
            return alt, f"資料更新失敗，使用快取：{exc}"
        return symbol, f"資料更新失敗，使用快取：{exc}"


def has_data(symbol):
    return inspect(engine).has_table(table_name_for(symbol))


def get_data(symbol, limit=2000):
    symbol = normalize_symbol(symbol)
    table = table_name_for(symbol)
    if not has_data(symbol):
        return pd.DataFrame()
    with engine.connect() as conn:
        df = pd.read_sql(text(f'SELECT * FROM "{table}" ORDER BY date'), conn,
                         index_col="date", parse_dates=["date"])
    return df if limit is None else df.tail(limit)
