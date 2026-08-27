import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_analysis(df, fast_window=20, slow_window=60,
                   fee_rate=0.001425, slippage_rate=0.0005):
    if df is None or df.empty or "Close" not in df:
        return None
    if fast_window >= slow_window or fast_window < 2:
        raise ValueError("fast_window must be smaller than slow_window and at least 2")
    if fee_rate < 0 or slippage_rate < 0:
        raise ValueError("fee_rate and slippage_rate cannot be negative")

    prices = df.sort_index().copy()
    prices["return"] = prices["Close"].pct_change().fillna(0)
    prices["fast_ma"] = prices["Close"].rolling(fast_window).mean()
    prices["slow_ma"] = prices["Close"].rolling(slow_window).mean()
    signal = (prices["fast_ma"] > prices["slow_ma"]).astype(float)
    prices["position"] = signal.shift(1).fillna(0)
    prices["turnover"] = prices["position"].diff().abs().fillna(0)
    prices["cost"] = prices["turnover"] * (fee_rate + slippage_rate)
    prices["strategy_return"] = prices["position"] * prices["return"] - prices["cost"]
    prices["buy_hold_equity"] = (1 + prices["return"]).cumprod()
    prices["strategy_equity"] = (1 + prices["strategy_return"]).cumprod()
    position_change = prices["position"].diff().fillna(prices["position"])
    trade_rows = prices.loc[position_change != 0, ["Close", "position"]].copy()
    trade_rows["action"] = trade_rows["position"].map({1.0: "買入", 0.0: "賣出"})
    trade_rows["date"] = trade_rows.index.strftime("%Y-%m-%d")
    trade_rows["price"] = trade_rows["Close"].astype(float)
    trade_log = trade_rows[["date", "action", "price"]].to_dict(orient="records")

    strategy_equity = prices["strategy_equity"]
    drawdown = strategy_equity / strategy_equity.cummax() - 1
    days = max((prices.index[-1] - prices.index[0]).days, 1)
    years = days / 365.25
    annual_return = strategy_equity.iloc[-1] ** (1 / years) - 1
    volatility = prices["strategy_return"].std() * (252 ** 0.5)
    sharpe = annual_return / volatility if volatility else 0

    return {
        "prices": prices,
        "metrics": {
            "start": prices.index[0].strftime("%Y-%m-%d"),
            "end": prices.index[-1].strftime("%Y-%m-%d"),
            "observations": int(len(prices)),
            "strategy_return": float(strategy_equity.iloc[-1] - 1),
            "buy_hold_return": float(prices["buy_hold_equity"].iloc[-1] - 1),
            "annual_return": float(annual_return),
            "annual_volatility": float(volatility),
            "sharpe": float(sharpe),
            "max_drawdown": float(drawdown.min()),
            "trades": int(signal.diff().abs().sum()),
            "total_cost": float(prices["cost"].sum()),
            "trade_log": trade_log,
        },
    }


def build_model_analysis(df, fast_window=20, slow_window=60,
                         fee_rate=0.001425, slippage_rate=0.0005,
                         model_name="random_forest"):
    result = build_analysis(df, fast_window, slow_window, fee_rate, slippage_rate)
    if result is None:
        return None

    prices = result["prices"]
    prices["momentum_5"] = prices["Close"].pct_change(5)
    prices["ma_ratio"] = prices["fast_ma"] / prices["slow_ma"] - 1
    prices["volatility_20"] = prices["return"].rolling(20).std()
    prices["volume_change"] = prices["Volume"].pct_change().replace([float("inf"), -float("inf")], pd.NA)
    prices["next_return"] = prices["Close"].pct_change().shift(-1)
    features = ["momentum_5", "ma_ratio", "volatility_20", "volume_change"]
    model_data = prices.dropna(subset=features + ["next_return"]).copy()
    feature_data = prices.dropna(subset=features).copy()
    if len(model_data) < 80:
        raise ValueError("模型至少需要 80 筆有效資料")
    model_factories = {
        "logistic": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(random_state=42, max_iter=1000)),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=42, n_jobs=-1),
        "gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=0.1, random_state=42),
    }
    model_labels = {
        "logistic": "Logistic Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "HistGradientBoosting",
    }
    if model_name not in model_factories:
        raise ValueError("model_name must be logistic, random_forest, or gradient_boosting")

    split = int(len(model_data) * 0.7)
    train = model_data.iloc[:split]
    test = model_data.iloc[split:].copy()
    if train["next_return"].gt(0).nunique() < 2:
        raise ValueError("訓練資料中的上漲與下跌樣本不足")

    target = train["next_return"].gt(0).astype(int)
    estimators = {}
    for name, factory in model_factories.items():
        estimator = factory()
        estimator.fit(train[features], target)
        estimators[name] = estimator
        test[f"{name}_probability"] = estimator.predict_proba(test[features])[:, 1]
    test["probability"] = test[f"{model_name}_probability"]
    test["position"] = (test["probability"] >= 0.5).astype(float)
    test["turnover"] = test["position"].diff().abs().fillna(test["position"].abs())
    test["strategy_return"] = (test["position"] * test["next_return"]
                                - test["turnover"] * (fee_rate + slippage_rate))
    test["model_equity"] = (1 + test["strategy_return"]).cumprod()
    test["buy_hold_equity"] = (1 + test["next_return"]).cumprod()
    actual = test["next_return"].gt(0)
    accuracy = ((test["probability"] >= 0.5) == actual).mean()
    accuracies = {
        name: float(((test[f"{name}_probability"] >= 0.5) == actual).mean())
        for name in model_factories
    }
    model_return = test["model_equity"].iloc[-1] - 1
    buy_hold_return = test["buy_hold_equity"].iloc[-1] - 1
    model_returns = test["strategy_return"]
    model_volatility = model_returns.std() * (252 ** 0.5)
    model_drawdown = test["model_equity"] / test["model_equity"].cummax() - 1
    model_years = max((test.index[-1] - test.index[0]).days / 365.25, 1 / 365.25)
    model_annual_return = test["model_equity"].iloc[-1] ** (1 / model_years) - 1
    model_sharpe = model_annual_return / model_volatility if model_volatility else 0
    up_rate = actual.mean()

    latest_estimator = model_factories[model_name]()
    latest_estimator.fit(model_data[features], model_data["next_return"].gt(0).astype(int))
    latest = feature_data.iloc[-1]
    latest_probability = latest_estimator.predict_proba(latest[features].to_frame().T)[0, 1]

    result["model_prices"] = test
    result["model_metrics"] = {
        "train_observations": int(len(train)),
        "test_observations": int(len(test)),
        "accuracy": float(accuracy),
        "model_name": model_labels[model_name],
        "model_key": model_name,
        "accuracies": accuracies,
        "baseline_accuracy": float(max(up_rate, 1 - up_rate)),
        "precision": float(precision_score(actual, test["probability"] >= 0.5, zero_division=0)),
        "recall": float(recall_score(actual, test["probability"] >= 0.5, zero_division=0)),
        "model_return": float(model_return),
        "buy_hold_return": float(buy_hold_return),
        "trades": int(test["turnover"].sum()),
        "model_volatility": float(model_volatility),
        "model_max_drawdown": float(model_drawdown.min()),
        "model_sharpe": float(model_sharpe),
        "latest_date": latest.name.strftime("%Y-%m-%d"),
        "latest_probability": float(latest_probability),
        "latest_signal": "持有" if latest_probability >= 0.5 else "空手",
        "fee_rate": float(fee_rate),
        "slippage_rate": float(slippage_rate),
    }
    return result
