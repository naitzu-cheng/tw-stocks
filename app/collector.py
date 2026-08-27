import yfinance as yf
import pandas as pd
from app.db import engine, ensure_table


def fetch_and_store(symbol, period="1y", interval="1d"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df is None or df.empty:
        return 0
    df = df[df["Close"].notna()].copy()
    if df.empty:
        return 0
    df.index.name = "date"
    ensure_table(symbol)
    table_name = symbol.replace('.', '_')
    try:
        existing = pd.read_sql_table(table_name, con=engine, index_col="date",
                                     parse_dates=["date"])
        combined = pd.concat([existing, df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    except Exception:
        combined = df
    combined.to_sql(table_name, con=engine, if_exists="replace", index=True)
    return len(df)


def get_data(symbol, limit=100):
    table = symbol.replace('.', '_')
    try:
        df = pd.read_sql_table(table, con=engine, index_col='date', parse_dates=['date'])
        return df.tail(limit)
    except Exception:
        return pd.DataFrame()
