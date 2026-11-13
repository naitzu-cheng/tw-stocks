import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.collector as collector
import app.main as main
from app.screener import regime, screen, validate
from app.universe import MARKETS
from conftest import make_prices


def _universe(n=30, days=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=days)
    cols = {}
    for k in range(n):
        drift = (k - n / 2) * 0.0001          # some strong, some weak stocks
        cols[f"S{k}"] = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, days)))
    return pd.DataFrame(cols, index=idx)


def test_screen_classifies_strong_and_weak():
    closes = _universe()
    table = screen(closes)
    assert len(table) == 30
    assert table["score"].is_monotonic_decreasing
    bulls = table[table["signal"] == "bull"]
    bears = table[table["signal"] == "bear"]
    assert len(bulls) and len(bears)
    assert (bulls["close"] > bulls["ma200"]).all()
    assert (bears["close"] < bears["ma200"]).all()
    assert set(table["signal"]) <= {"bull", "bear", "hot", "neutral"}


def test_regime_and_breadth():
    closes = _universe()
    r = regime(closes.mean(axis=1), closes)
    assert 0 <= r["breadth200"] <= 1
    assert r["regime"] in {"多頭", "空頭", "震盪 / 轉折"}


def test_validate_runs_without_lookahead():
    closes = _universe(days=700)
    v = validate(closes)
    assert v["periods"] >= 10
    assert 0 <= v["hit_rate"] <= 1
    # Strategy equity must start on/after day 252 (momentum warm-up).
    assert v["strategy_equity"].index[0] > closes.index[251]


@pytest.fixture
def client(monkeypatch):
    def fake_one(symbol, period, interval):
        return make_prices(900, seed=abs(hash(symbol)) % 1000)

    def fake_many(symbols, period):
        return {s: fake_one(s, period, "1d") for s in symbols[:-3]}   # force fallback for a few

    monkeypatch.setattr(collector, "_download", fake_one)
    monkeypatch.setattr(collector, "_download_many", fake_many)
    collector._fetched_at.clear()
    main._cache.clear()
    return TestClient(main.app)


@pytest.mark.parametrize("market", list(MARKETS))
def test_market_page(client, market):
    r = client.get(f"/market?market={market}")
    assert r.status_code == 200, r.text[:500]
    assert "市場狀態" in r.text
    j = client.get(f"/market.json?market={market}").json()
    assert len(j["stocks"]) == len(MARKETS[market]["universe"])
    assert j["regime"]["regime"]


def test_dashboard_ml_is_lazy(client):
    page = client.get("/dashboard?symbol=2330")
    assert page.status_code == 200
    assert "/dashboard/ml?" in page.text and "樣本外準確率" not in page.text
    frag = client.get("/dashboard/ml?symbol=2330.TW").json()
    assert "樣本外準確率" in frag["html"] and frag["figure"]["data"]


def test_bad_market(client):
    assert client.get("/market?market=JP").status_code == 400
