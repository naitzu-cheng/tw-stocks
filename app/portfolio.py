import pandas as pd

from app.collector import get_data


def build_portfolio_analysis(symbols, limit=2000):
    symbols = [symbol.strip() for symbol in symbols if symbol.strip()]
    if len(symbols) < 2:
        raise ValueError("投資組合至少需要兩支股票")
    close_prices = {}
    for symbol in symbols:
        data = get_data(symbol, limit=limit)
        if data is None or data.empty or "Close" not in data:
            raise ValueError(f"找不到 {symbol} 的價格資料，請先收集資料")
        close_prices[symbol] = data["Close"]

    closes = pd.DataFrame(close_prices).sort_index().ffill().dropna()
    if len(closes) < 30:
        raise ValueError("投資組合至少需要 30 個共同交易日")
    returns = closes.pct_change().dropna()
    portfolio_returns = returns.mean(axis=1)
    equity = (1 + portfolio_returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    annual_return = equity.iloc[-1] ** (1 / years) - 1
    volatility = portfolio_returns.std() * (252 ** 0.5)

    return {
        "symbols": symbols,
        "metrics": {
            "start": equity.index[0].strftime("%Y-%m-%d"),
            "end": equity.index[-1].strftime("%Y-%m-%d"),
            "observations": int(len(equity)),
            "portfolio_return": float(equity.iloc[-1] - 1),
            "annual_return": float(annual_return),
            "annual_volatility": float(volatility),
            "max_drawdown": float(drawdown.min()),
            "sharpe": float(annual_return / volatility if volatility else 0),
            "weights": {symbol: 1 / len(symbols) for symbol in symbols},
        },
        "equity": equity,
        "correlation": returns.corr(),
    }
