"""Technical features. Every feature uses only data up to and including day t."""
import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(loss.notna())


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_close).abs(),
                    (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


FEATURE_LABELS = {
    "ret_1": "1 日報酬",
    "ret_5": "5 日報酬（短期動能）",
    "ret_20": "20 日報酬（月動能）",
    "ret_60": "60 日報酬（季動能）",
    "rsi_14": "RSI(14)",
    "macd_hist": "MACD 柱狀體 / 股價",
    "bb_pctb": "布林通道 %B",
    "atr_pct": "ATR(14) / 股價",
    "vol_20": "20 日波動率",
    "vol_ratio": "20 日 / 60 日波動率",
    "volume_z": "成交量 Z 分數",
    "dist_high_120": "距 120 日高點",
    "ma_ratio": "短長均線比",
    "rel_strength_20": "相對大盤 20 日強弱",
    "market_trend": "大盤 / 200 日均線",
}


def build_features(df: pd.DataFrame, benchmark_close: pd.Series = None,
                   fast_window: int = 20, slow_window: int = 60) -> pd.DataFrame:
    close = df["Close"]
    log_close = np.log(close)
    f = pd.DataFrame(index=df.index)
    for n in (1, 5, 20, 60):
        f[f"ret_{n}"] = log_close.diff(n)
    f["rsi_14"] = rsi(close) / 100 - 0.5
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close
    mid = close.rolling(20).mean()
    band = close.rolling(20).std()
    f["bb_pctb"] = (close - (mid - 2 * band)) / (4 * band)
    f["atr_pct"] = atr(df) / close
    daily = log_close.diff()
    f["vol_20"] = daily.rolling(20).std()
    f["vol_ratio"] = f["vol_20"] / daily.rolling(60).std()
    if "Volume" in df and df["Volume"].fillna(0).gt(0).mean() > 0.9:
        log_vol = np.log(df["Volume"].where(df["Volume"] > 0))
        f["volume_z"] = (log_vol - log_vol.rolling(20).mean()) / log_vol.rolling(20).std()
    f["dist_high_120"] = close / close.rolling(120).max() - 1
    f["ma_ratio"] = close.rolling(fast_window).mean() / close.rolling(slow_window).mean() - 1
    if benchmark_close is not None and len(benchmark_close.dropna()) > 200:
        bench = benchmark_close.reindex(df.index).ffill()
        f["rel_strength_20"] = f["ret_20"] - np.log(bench).diff(20)
        f["market_trend"] = bench / bench.rolling(200).mean() - 1
    return f.replace([np.inf, -np.inf], np.nan)
