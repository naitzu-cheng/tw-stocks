import numpy as np
import pandas as pd

from app.collector import get_data
from app.metrics import performance

WEIGHTINGS = {"equal": "等權重", "inverse_vol": "風險平價（波動倒數）"}


def build_portfolio_analysis(symbols, limit=2000, weighting="equal",
                             rebalance_days=21, benchmark_close=None):
    """Long-only portfolio, rebalanced every `rebalance_days`.

    inverse_vol: each stock gets weight proportional to 1 / its 60-day
    volatility, measured *before* the rebalance date (no look-ahead), so a
    volatile stock cannot dominate the portfolio's risk.
    Mixing TW and US stocks: returns are in each stock's own currency
    (FX is ignored) and holidays are forward-filled.
    """
    symbols = [s.strip() for s in symbols if s.strip()]
    if len(symbols) < 2:
        raise ValueError("投資組合至少需要兩支股票")
    if weighting not in WEIGHTINGS:
        raise ValueError("weighting 必須是 equal 或 inverse_vol")
    close_prices = {}
    for symbol in symbols:
        data = get_data(symbol, limit=limit)
        if data is None or data.empty or "Close" not in data:
            raise ValueError(f"找不到 {symbol} 的價格資料")
        close_prices[symbol] = data["Close"]

    closes = pd.DataFrame(close_prices).sort_index().ffill().dropna()
    if len(closes) < 90:
        raise ValueError("投資組合至少需要 90 個共同交易日")
    returns = closes.pct_change().dropna()
    vol60 = returns.rolling(60, min_periods=20).std().shift(1)

    weights = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for i in range(0, len(returns), rebalance_days):
        if weighting == "inverse_vol" and vol60.iloc[i].notna().all():
            inv = 1 / vol60.iloc[i]
            weights.iloc[i] = inv / inv.sum()
        else:
            weights.iloc[i] = 1 / len(symbols)
    # Between rebalances weights drift with prices.
    drifted = []
    w = weights.iloc[0].to_numpy()
    for i in range(len(returns)):
        if weights.iloc[i].notna().all():
            w = weights.iloc[i].to_numpy()
        drifted.append(w)
        grown = w * (1 + returns.iloc[i].to_numpy())
        w = grown / grown.sum()
    weights = pd.DataFrame(drifted, index=returns.index, columns=returns.columns)
    portfolio_returns = (weights * returns).sum(axis=1)

    bench_ret = benchmark_close.pct_change() if benchmark_close is not None else None
    metrics = performance(portfolio_returns, bench_ret)
    vol = returns.std() * np.sqrt(252)
    latest_w = weights.iloc[-1]
    cov = returns.tail(252).cov() * 252
    port_var = float(latest_w @ cov @ latest_w)
    risk_contrib = (latest_w * (cov @ latest_w)) / port_var if port_var > 0 else latest_w * 0
    per_stock = [{"symbol": s, "weight": float(latest_w[s]), "volatility": float(vol[s]),
                  "risk_contribution": float(risk_contrib[s]),
                  "total_return": float(closes[s].iloc[-1] / closes[s].iloc[0] - 1)}
                 for s in symbols]
    metrics.update({
        "portfolio_return": metrics["total_return"],
        "annual_return": metrics["cagr"],
        "weighting": weighting,
        "weighting_label": WEIGHTINGS[weighting],
        "weights": {s: float(latest_w[s]) for s in symbols},
    })
    return {
        "symbols": symbols,
        "metrics": metrics,
        "per_stock": per_stock,
        "equity": (1 + portfolio_returns).cumprod(),
        "correlation": returns.corr(),
    }
