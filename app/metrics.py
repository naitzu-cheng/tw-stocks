"""Performance statistics, including tests for whether a Sharpe ratio is luck."""
import math

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def performance(returns: pd.Series, benchmark: pd.Series = None,
                risk_free: float = 0.0, positions: pd.Series = None):
    """Summary statistics for a series of daily (net) returns."""
    r = returns.dropna()
    if len(r) < 2:
        raise ValueError("報酬資料太少，無法計算績效")
    equity = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    total = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else -1.0
    vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    excess = r - risk_free / TRADING_DAYS
    sharpe = float(excess.mean() / r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
    downside = r[r < 0]
    downside_dev = math.sqrt((np.minimum(r, 0) ** 2).mean()) * math.sqrt(TRADING_DAYS)
    sortino = float(excess.mean() * TRADING_DAYS / downside_dev) if downside_dev > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
    active = r[r != 0]
    gains, losses = active[active > 0].sum(), -active[active < 0].sum()
    out = {
        "start": r.index[0].strftime("%Y-%m-%d"),
        "end": r.index[-1].strftime("%Y-%m-%d"),
        "days": int(len(r)),
        "total_return": total,
        "cagr": cagr,
        "annual_volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "max_drawdown_days": int(_longest_underwater(drawdown)),
        "win_rate": float((active > 0).mean()) if len(active) else 0.0,
        "profit_factor": float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0,
        "psr": probabilistic_sharpe(r),
        "skew": float(stats.skew(r)),
        "kurtosis": float(stats.kurtosis(r, fisher=False)),
    }
    if positions is not None:
        out["exposure"] = float((positions.reindex(r.index).fillna(0) != 0).mean())
    if benchmark is not None:
        out.update(relative_to(r, benchmark))
    return out


def _longest_underwater(drawdown: pd.Series) -> int:
    longest = current = 0
    for value in drawdown.to_numpy():
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest


def relative_to(returns: pd.Series, benchmark: pd.Series):
    joined = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(joined) < 30:
        return {}
    y, x = joined.iloc[:, 0], joined.iloc[:, 1]
    var = x.var()
    beta = float(y.cov(x) / var) if var > 0 else 0.0
    alpha = float((y.mean() - beta * x.mean()) * TRADING_DAYS)
    active = y - x
    te = active.std() * math.sqrt(TRADING_DAYS)
    return {
        "beta": beta,
        "alpha": alpha,
        "correlation": float(y.corr(x)),
        "information_ratio": float(active.mean() * TRADING_DAYS / te) if te > 0 else 0.0,
    }


def probabilistic_sharpe(returns: pd.Series, benchmark_sharpe: float = 0.0) -> float:
    """Probability that the true (annualised) Sharpe exceeds benchmark_sharpe.

    Bailey & López de Prado (2012). Adjusts for sample length, skewness and fat
    tails: a Sharpe of 1.5 over 6 months is far less convincing than over 10 years.
    """
    r = returns.dropna()
    n = len(r)
    if n < 10 or r.std() == 0:
        return 0.5
    sr = r.mean() / r.std(ddof=1)                      # daily Sharpe
    sr_star = benchmark_sharpe / math.sqrt(TRADING_DAYS)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return 0.5
    z = (sr - sr_star) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Sharpe you'd expect from the *best of n random strategies* (daily units)."""
    if n_trials <= 1:
        return 0.0
    gamma = 0.5772156649
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return sharpe_std * ((1 - gamma) * z1 + gamma * z2)


def deflated_sharpe(returns: pd.Series, trial_sharpes) -> float:
    """PSR measured against the best Sharpe expected purely from trying many
    parameter sets (Bailey & López de Prado, 2014). Low = probably data-mined."""
    trial = np.asarray([s for s in trial_sharpes if np.isfinite(s)], dtype=float)
    if len(trial) < 2:
        return probabilistic_sharpe(returns)
    daily_std = trial.std(ddof=1) / math.sqrt(TRADING_DAYS)
    sr0 = expected_max_sharpe(len(trial), daily_std) * math.sqrt(TRADING_DAYS)
    return probabilistic_sharpe(returns, benchmark_sharpe=sr0)


def binomial_pvalue(hits: int, n: int, p0: float) -> float:
    """One-sided p-value that accuracy hits/n beats chance rate p0."""
    if n <= 0:
        return 1.0
    return float(stats.binomtest(hits, n, p0, alternative="greater").pvalue)
