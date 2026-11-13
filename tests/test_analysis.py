import numpy as np
import pandas as pd
import pytest

from app.analysis import Costs, build_analysis, ma_robustness, run_backtest, walk_forward
from app.markets import normalize_symbol, profile_for
from app.metrics import performance, probabilistic_sharpe
from conftest import make_prices


@pytest.mark.parametrize("raw,expected", [
    ("2330", "2330.TW"), ("2330.tw", "2330.TW"), ("6488.TWO", "6488.TWO"),
    ("00878", "00878.TW"), ("aapl", "AAPL"), ("brk.b", "BRK-B"), ("^GSPC", "^GSPC"),
])
def test_normalize_symbol(raw, expected):
    assert normalize_symbol(raw) == expected


def test_market_profiles():
    assert profile_for("2330.TW").tax_rate == 0.003
    assert profile_for("0050.TW").tax_rate == 0.001
    us = profile_for("AAPL")
    assert us.fee_rate == 0 and us.benchmark == "SPY" and us.currency == "USD"


def test_backtest_uses_next_day_and_charges_tax_on_sell():
    idx = pd.bdate_range("2024-01-01", periods=5)
    close = pd.Series([100, 110, 121, 121, 121], index=idx, dtype=float)
    signal = pd.Series([1, 1, 0, 0, 0], index=idx, dtype=float)
    costs = Costs(fee_rate=0.001, tax_rate=0.003, slippage_rate=0.0)
    bt = run_backtest(close, signal, costs)
    # Signal on day 0 -> earns day 1 return (10%), never day 0's.
    assert bt["position"].tolist() == [0, 1, 1, 0, 0]
    assert bt["cost"].iloc[1] == pytest.approx(0.001)          # buy: fee only
    assert bt["cost"].iloc[3] == pytest.approx(0.004)          # sell: fee + tax
    assert bt["strategy_return"].iloc[1] == pytest.approx(0.10 - 0.001)


def test_perfect_foresight_signal_is_not_rewarded_without_shift():
    """A signal that uses today's return must not capture today's return."""
    df = make_prices(500, seed=3)
    today_up = (df["Close"].pct_change() > 0).astype(float)
    bt = run_backtest(df["Close"], today_up, Costs(0, 0, 0))
    assert performance(bt["strategy_return"])["sharpe"] < 1.5


def test_ma_analysis_metrics(prices, benchmark):
    r = build_analysis(prices, 20, 60, Costs(), benchmark.pct_change())
    m = r["metrics"]
    for key in ("sharpe", "sortino", "max_drawdown", "psr", "beta", "alpha", "exposure"):
        assert key in m
    assert -1 <= m["max_drawdown"] <= 0
    assert m["trades"] == len(m["trade_log"])


def test_robustness_on_random_walk(prices):
    r = ma_robustness(prices, Costs())
    assert r["n_trials"] >= 25
    # On pure noise the deflated Sharpe of an arbitrary choice should not look convincing.
    assert r["chosen"]["deflated_sharpe"] < 0.95


def test_walk_forward_has_purge_gap(prices):
    m = walk_forward(prices, "logistic", horizon=5, compare_models=False)["model_metrics"]
    idx = prices.index
    for fold in m["folds"]:
        gap = idx.get_loc(pd.Timestamp(fold["test_start"])) - idx.get_loc(pd.Timestamp(fold["train_end"]))
        assert gap > 5


def test_walk_forward_finds_no_edge_in_noise(prices):
    m = walk_forward(prices, "logistic", compare_models=False)["model_metrics"]
    assert 0.4 < m["auc"] < 0.6
    assert m["p_value"] > 0.01


def test_walk_forward_detects_real_signal():
    """Positive control: returns that genuinely follow 20-day momentum."""
    rng = np.random.default_rng(1)
    n = 1500
    ret = np.zeros(n)
    noise = rng.normal(0, 0.01, n)
    for t in range(20, n):
        ret[t] = 0.25 * ret[t - 20:t].mean() * 20 / 20 + noise[t] + 0.15 * np.sign(ret[t - 20:t].sum()) * 0.01
    close = 100 * np.exp(np.cumsum(ret))
    idx = pd.bdate_range("2019-01-01", periods=n)
    df = pd.DataFrame({"Open": close, "High": close * 1.005, "Low": close * 0.995,
                       "Close": close, "Volume": 1e6}, index=idx)
    m = walk_forward(df, "logistic", horizon=5, compare_models=False)["model_metrics"]
    assert m["auc"] > 0.55
    assert m["p_value"] < 0.05


def test_psr_increases_with_sample_length():
    rng = np.random.default_rng(0)
    short = pd.Series(rng.normal(0.0008, 0.01, 252))
    long = pd.Series(np.tile(short.to_numpy(), 8))
    assert probabilistic_sharpe(long) > probabilistic_sharpe(short)
