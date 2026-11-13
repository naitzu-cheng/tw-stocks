import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import tempfile

os.environ["PREFETCH_MARKETS"] = "0"
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tempfile.mkdtemp(), "test.db")


def make_prices(n=1300, seed=0, drift=0.0004, vol=0.015):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    ret = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    vol_ = rng.integers(1_000_000, 5_000_000, n).astype(float)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                       "Volume": vol_}, index=idx)
    df.index.name = "date"
    return df


@pytest.fixture
def prices():
    return make_prices()


@pytest.fixture
def benchmark():
    return make_prices(seed=99)["Close"]
