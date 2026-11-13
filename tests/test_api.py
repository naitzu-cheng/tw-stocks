import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.collector as collector
import app.main as main
from conftest import make_prices


@pytest.fixture
def client(monkeypatch):
    seeds = {"2330.TW": 1, "0050.TW": 2, "AAPL": 3, "SPY": 4, "6488.TWO": 5, "MU": 6}

    def fake_download(symbol, period, interval):
        if symbol.startswith("ZZZZ") or symbol == "6488.TW":
            return pd.DataFrame()           # unknown ticker / not listed on TWSE
        return make_prices(1300, seed=seeds.get(symbol, sum(map(ord, symbol))))

    monkeypatch.setattr(collector, "_download", fake_download)
    collector._fetched_at.clear()
    main._cache.clear()
    return TestClient(main.app)


@pytest.mark.parametrize("url", [
    "/dashboard?symbol=2330&compare=true",
    "/dashboard?symbol=AAPL&model_name=random_forest",
    "/dashboard?symbol=2330.TW&fast_window=20&slow_window=60&fee_rate=0.001425&slippage_rate=0.0005&model_name=logistic",
    "/portfolio?symbols=2330,AAPL,0050&weighting=inverse_vol",
    "/portfolio?symbols=2330,AAPL&weighting=equal",
    "/risk-dashboard?symbol=AAPL",
    "/guide",
])
def test_pages_render(client, url):
    r = client.get(url)
    assert r.status_code == 200, r.text[:500]
    assert "<html" in r.text


def test_tpex_fallback(client):
    r = client.get("/analysis?symbol=6488")
    assert r.status_code == 200
    assert r.json()["symbol"] == "6488.TWO"


def test_json_endpoints(client):
    a = client.get("/analysis?symbol=AAPL").json()
    assert a["costs"]["fee_rate"] == 0
    m = client.get("/model?symbol=2330&horizon=10").json()
    assert m["horizon"] == 10 and 0 <= m["auc"] <= 1
    r = client.get("/robustness?symbol=2330").json()
    assert r["n_trials"] >= 25
    k = client.get("/risk?symbol=2330").json()
    assert k["lot_size"] == 1000 and k["suggested_shares"] >= 0
    d = client.get("/data?symbol=2330&limit=5").json()
    assert len(d) == 5


def test_bad_input(client):
    assert client.get("/analysis?symbol=2330&fast_window=60&slow_window=20").status_code == 400
    assert client.get("/analysis?symbol=bad;drop").status_code == 400
    assert client.get("/analysis?symbol=ZZZZ").status_code == 404


@pytest.mark.parametrize("url", [
    # Exactly what the dashboard form sends when the cost boxes are left blank.
    "/dashboard?symbol=MU&fast_window=20&slow_window=60&fee_rate=&tax_rate=&slippage_rate=&model_name=logistic&horizon=5&step=63",
    "/dashboard/ml?symbol=AAPL&fee_rate=&tax_rate=&slippage_rate=&fast_window=",
    "/risk-dashboard?symbol=AAPL&capital=&risk_per_trade=0.01&atr_multiple=2&stop_loss_pct=",
    "/portfolio?symbols=&weighting=inverse_vol&rebalance_days=",
    "/analysis?symbol=2330&fee_rate=",
])
def test_blank_form_fields_use_defaults(client, url):
    r = client.get(url)
    assert r.status_code == 200, r.text[:300]
