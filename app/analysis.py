"""Backtests and walk-forward machine-learning evaluation."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.features import FEATURE_LABELS, build_features
from app.metrics import binomial_pvalue, deflated_sharpe, performance


@dataclass
class Costs:
    fee_rate: float = 0.001425
    tax_rate: float = 0.003
    slippage_rate: float = 0.0005

    def validate(self):
        if min(self.fee_rate, self.tax_rate, self.slippage_rate) < 0:
            raise ValueError("手續費、交易稅與滑價不可為負數")
        return self

    @property
    def round_trip(self):
        return 2 * (self.fee_rate + self.slippage_rate) + self.tax_rate


def run_backtest(close: pd.Series, signal: pd.Series, costs: Costs) -> pd.DataFrame:
    """Long-only backtest.

    signal[t] is decided with information up to the close of day t, so the
    position is held from t to t+1 (shift by one day: no look-ahead).
    Buys pay fee + slippage; sells pay fee + tax + slippage.
    """
    out = pd.DataFrame(index=close.index)
    out["Close"] = close
    out["return"] = close.pct_change().fillna(0)
    out["signal"] = signal.reindex(close.index).fillna(0).clip(0, 1)
    out["position"] = out["signal"].shift(1).fillna(0)
    change = out["position"].diff().fillna(out["position"])
    buys, sells = change.clip(lower=0), (-change).clip(lower=0)
    out["cost"] = (buys * (costs.fee_rate + costs.slippage_rate)
                   + sells * (costs.fee_rate + costs.tax_rate + costs.slippage_rate))
    out["strategy_return"] = out["position"] * out["return"] - out["cost"]
    out["strategy_equity"] = (1 + out["strategy_return"]).cumprod()
    out["buy_hold_equity"] = (1 + out["return"]).cumprod()
    out["trade"] = np.select([change > 0, change < 0], ["買入", "賣出"], "")
    return out


def _trade_log(bt: pd.DataFrame):
    rows = bt[bt["trade"] != ""]
    # The trade happens at the close of the signal day (one day before the position).
    return [{"date": d.strftime("%Y-%m-%d"), "action": a, "price": float(p)}
            for d, a, p in zip(rows.index, rows["trade"], rows["Close"])]


def band_signal(prob: pd.Series, threshold: pd.Series, band: float) -> pd.Series:
    """Enter when prob > threshold + band, exit when prob < threshold - band.

    The gap (hysteresis) stops the position from flipping every time the
    probability wobbles around the threshold, which would burn money on costs.
    """
    state, out = 0.0, []
    for p, t in zip(prob.to_numpy(), threshold.to_numpy()):
        if np.isnan(p):
            out.append(state)
            continue
        if state == 0 and p > t + band:
            state = 1.0
        elif state == 1 and p < t - band:
            state = 0.0
        out.append(state)
    return pd.Series(out, index=prob.index)


def _ma_signal(close, fast, slow):
    return (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float)


def build_analysis(df, fast_window=20, slow_window=60, costs: Costs = None,
                   benchmark_returns: pd.Series = None):
    """Moving-average crossover backtest."""
    if df is None or df.empty or "Close" not in df:
        return None
    if fast_window >= slow_window or fast_window < 2:
        raise ValueError("短均線必須小於長均線，且至少為 2")
    costs = (costs or Costs()).validate()
    prices = df.sort_index()
    if len(prices) < slow_window + 30:
        raise ValueError(f"資料只有 {len(prices)} 筆，長均線 {slow_window} 需要更多資料")
    bt = run_backtest(prices["Close"], _ma_signal(prices["Close"], fast_window, slow_window), costs)
    bt = bt.iloc[slow_window:]  # skip the warm-up period where the MA does not exist yet
    for col in ("Open", "High", "Low", "Volume"):
        if col in prices:
            bt[col] = prices[col]
    bt["fast_ma"] = prices["Close"].rolling(fast_window).mean()
    bt["slow_ma"] = prices["Close"].rolling(slow_window).mean()
    strat = performance(bt["strategy_return"], benchmark_returns, positions=bt["position"])
    hold = performance(bt["return"], benchmark_returns)
    trades = _trade_log(bt)
    metrics = {
        **strat,
        "strategy_return": strat["total_return"],
        "annual_return": strat["cagr"],
        "buy_hold_return": hold["total_return"],
        "buy_hold": hold,
        "observations": int(len(bt)),
        "trades": len(trades),
        "total_cost": float(bt["cost"].sum()),
        "trade_log": trades,
    }
    return {"prices": bt, "metrics": metrics}


FAST_GRID = [5, 10, 20, 30, 50]
SLOW_GRID = [20, 40, 60, 100, 150, 200]


def ma_robustness(df, costs: Costs = None, fast_window=20, slow_window=60,
                  in_sample_ratio=0.6):
    """Test every MA pair, pick the best one in-sample, then check it out-of-sample.

    If the in-sample winner falls apart afterwards, the "best parameters" were
    mostly fitting noise. Also returns the deflated Sharpe of the user's choice.
    """
    costs = (costs or Costs()).validate()
    close = df["Close"].sort_index()
    warm = max(SLOW_GRID)
    if len(close) < warm + 250:
        raise ValueError("參數穩健度檢查至少需要約 2 年資料")
    split = warm + int((len(close) - warm) * in_sample_ratio)
    split_date = close.index[split]
    pairs = [(f, s) for f in FAST_GRID for s in SLOW_GRID if f < s]
    if (fast_window, slow_window) not in pairs:
        pairs.append((fast_window, slow_window))
    rows = []
    for f, s in pairs:
        r = run_backtest(close, _ma_signal(close, f, s), costs)["strategy_return"].iloc[warm:]
        ins, oos = r[r.index < split_date], r[r.index >= split_date]
        rows.append({"fast": f, "slow": s,
                     "sharpe_full": _sharpe(r), "sharpe_in": _sharpe(ins),
                     "sharpe_out": _sharpe(oos), "returns": r})
    table = pd.DataFrame(rows)
    best = table.loc[table["sharpe_in"].idxmax()]
    chosen = table[(table["fast"] == fast_window) & (table["slow"] == slow_window)].iloc[0]
    rank = int((table["sharpe_out"] > table["sharpe_out"].median()).sum())
    heat = table.pivot(index="fast", columns="slow", values="sharpe_full")
    neighbours = table[(table["fast"].between(fast_window / 2, fast_window * 2))
                       & (table["slow"].between(slow_window / 2, slow_window * 2))]
    return {
        "split_date": split_date.strftime("%Y-%m-%d"),
        "n_trials": int(len(table)),
        "heatmap": heat,
        "table": table.drop(columns="returns"),
        "best_in_sample": {"fast": int(best["fast"]), "slow": int(best["slow"]),
                           "sharpe_in": float(best["sharpe_in"]),
                           "sharpe_out": float(best["sharpe_out"])},
        "chosen": {"fast": fast_window, "slow": slow_window,
                   "sharpe_in": float(chosen["sharpe_in"]),
                   "sharpe_out": float(chosen["sharpe_out"]),
                   "deflated_sharpe": deflated_sharpe(chosen["returns"], table["sharpe_full"])},
        "neighbour_sharpe_std": float(neighbours["sharpe_full"].std()) if len(neighbours) > 2 else None,
        "in_out_correlation": float(table["sharpe_in"].corr(table["sharpe_out"], method="spearman")),
        "pct_positive_out": float((table["sharpe_out"] > 0).mean()),
        "rank_above_median": rank,
    }


def _sharpe(r):
    r = r.dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(252)) if sd and sd > 0 else 0.0


MODEL_LABELS = {
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "HistGradientBoosting",
}


def make_model(name):
    # Deliberately strong regularisation: daily stock data is mostly noise, and
    # deep trees memorise that noise.
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=40,
                                      max_features="sqrt", random_state=42, n_jobs=1)
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=3,
                                              min_samples_leaf=40, l2_regularization=1.0,
                                              random_state=42)
    raise ValueError("model_name 必須是 logistic、random_forest 或 gradient_boosting")


def walk_forward(df, model_name="logistic", horizon=5, step=63, min_train=500,
                 threshold=None, band=0.02, costs: Costs = None, benchmark_close: pd.Series = None,
                 fast_window=20, slow_window=60, compare_models=True):
    """Walk-forward (rolling re-training) evaluation of a direction classifier.

    * Target: is the close `horizon` days later higher than today's close?
    * Every `step` days the model is re-trained on all data available *before*
      that date, minus a purge gap of `horizon` days (the last training labels
      would otherwise overlap the test period: label leakage).
    * Each prediction is therefore made by a model that never saw that period.
    * threshold=None: go long when P(up) > the up-rate seen in training, so the
      model has to beat "always long", not just predict the usual drift.
    """
    if model_name not in MODEL_LABELS:
        raise ValueError("model_name 必須是 logistic、random_forest 或 gradient_boosting")
    if not 1 <= horizon <= 60:
        raise ValueError("預測天數 horizon 需介於 1 到 60")
    costs = (costs or Costs()).validate()
    prices = df.sort_index()
    feats = build_features(prices, benchmark_close, fast_window, slow_window)
    feats = feats.dropna(axis=1, thresh=int(len(feats) * 0.5))
    feature_cols = list(feats.columns)
    fwd = prices["Close"].shift(-horizon) / prices["Close"] - 1
    data = feats.assign(fwd=fwd).dropna(subset=feature_cols)
    labelled = data.dropna(subset=["fwd"])
    min_train = min(min_train, int(len(labelled) * 0.5))
    if len(labelled) < 300 or min_train < 200:
        raise ValueError(f"有效資料只有 {len(labelled)} 筆，walk-forward 至少需要約 300 筆（建議 3 年以上）")

    names = list(MODEL_LABELS) if compare_models else [model_name]
    probs = {n: pd.Series(np.nan, index=data.index) for n in names}
    thresholds = pd.Series(np.nan, index=data.index)
    train_acc = {n: [] for n in names}
    folds = []
    starts = list(range(min_train + horizon, len(data), step))
    for start in starts:
        end = min(start + step, len(data))
        train = labelled.iloc[: start - horizon]           # purge gap
        train = train[train.index < data.index[start]]
        y = (train["fwd"] > 0).astype(int)
        if y.nunique() < 2:
            continue
        test = data.iloc[start:end]
        up_rate = float(y.mean())
        thresholds.loc[test.index] = up_rate if threshold is None else threshold
        for n in names:
            model = make_model(n).fit(train[feature_cols], y)
            probs[n].loc[test.index] = model.predict_proba(test[feature_cols])[:, 1]
            train_acc[n].append(float((model.predict(train[feature_cols]) == y).mean()))
        folds.append({"train_end": train.index[-1].strftime("%Y-%m-%d"),
                      "test_start": test.index[0].strftime("%Y-%m-%d"),
                      "test_end": test.index[-1].strftime("%Y-%m-%d"),
                      "train_size": int(len(train))})
    if not folds:
        raise ValueError("walk-forward 沒有可用的測試區段")

    oos_index = probs[model_name].dropna().index
    close = prices["Close"]
    bench_ret = benchmark_close.pct_change() if benchmark_close is not None else None
    comparison = {}
    detail = None
    for n in names:
        p = probs[n].loc[oos_index]
        thr = thresholds.loc[oos_index]
        signal = band_signal(p, thr, band)
        bt = run_backtest(close.loc[oos_index[0]:], signal, costs)
        scored = data.loc[oos_index].dropna(subset=["fwd"])
        actual = (scored["fwd"] > 0).astype(int)
        pred = (p.loc[scored.index] > thr.loc[scored.index]).astype(int)
        hits = int((pred == actual).sum())
        n_scored = len(actual)
        up = float(actual.mean())
        baseline = max(up, 1 - up)
        # Overlapping h-day labels are not independent: use n / horizon samples.
        eff_n = max(n_scored // horizon, 1)
        perf = performance(bt["strategy_return"], bench_ret, positions=bt["position"])
        comparison[n] = {
            "label": MODEL_LABELS[n],
            "accuracy": hits / n_scored,
            "train_accuracy": float(np.mean(train_acc[n])),
            "baseline_accuracy": baseline,
            "auc": float(roc_auc_score(actual, p.loc[scored.index])) if actual.nunique() > 1 else 0.5,
            "p_value": binomial_pvalue(round(hits / n_scored * eff_n), eff_n, baseline),
            "precision": float(actual[pred == 1].mean()) if pred.sum() else 0.0,
            "exposure": float(signal.mean()),
            "sharpe": perf["sharpe"],
            "total_return": perf["total_return"],
            "max_drawdown": perf["max_drawdown"],
            "psr": perf["psr"],
        }
        if n == model_name:
            detail = (bt, perf, p, thr)

    bt, perf, p, thr = detail
    hold = performance(bt["return"], bench_ret)
    bt["probability"] = p
    bt["threshold"] = thr
    # Final model on all labelled data -> today's reading.
    y_all = (labelled["fwd"] > 0).astype(int)
    final = make_model(model_name).fit(labelled[feature_cols], y_all)
    latest_row = data.iloc[[-1]]
    latest_p = float(final.predict_proba(latest_row[feature_cols])[0, 1])
    latest_thr = float(y_all.mean()) if threshold is None else float(threshold)
    importance = _importance(final, feature_cols)
    m = comparison[model_name]
    return {
        "model_prices": bt,
        "model_metrics": {
            **m,
            "model_key": model_name,
            "model_name": MODEL_LABELS[model_name],
            "horizon": horizon,
            "step": step,
            "band": band,
            "folds": folds,
            "n_folds": len(folds),
            "oos_start": oos_index[0].strftime("%Y-%m-%d"),
            "oos_end": oos_index[-1].strftime("%Y-%m-%d"),
            "test_observations": int(len(oos_index)),
            "strategy": perf,
            "buy_hold": hold,
            "model_return": perf["total_return"],
            "buy_hold_return": hold["total_return"],
            "trades": int((bt["trade"] != "").sum()),
            "overfit_gap": m["train_accuracy"] - m["accuracy"],
            "comparison": comparison,
            "features": [FEATURE_LABELS.get(c, c) for c in feature_cols],
            "importance": importance,
            "latest_date": data.index[-1].strftime("%Y-%m-%d"),
            "latest_probability": latest_p,
            "latest_threshold": latest_thr,
            "latest_signal": "偏多（持有）" if latest_p > latest_thr else "中性 / 空手",
            "verdict": _verdict(m, perf, hold),
        },
    }


def _importance(model, cols):
    est = model
    if hasattr(model, "named_steps"):
        est = model.named_steps["logisticregression"]
        values = np.abs(est.coef_[0])
    elif hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    else:
        return []
    total = values.sum() or 1
    pairs = sorted(zip(cols, values / total), key=lambda x: -x[1])
    return [{"feature": FEATURE_LABELS.get(c, c), "weight": float(v)} for c, v in pairs[:8]]


def _verdict(m, perf, hold):
    """Plain-language reading of the walk-forward result."""
    notes = []
    edge = m["accuracy"] - m["baseline_accuracy"]
    if m["p_value"] < 0.05 and m["auc"] > 0.53:
        notes.append(("good", f"準確率比基準高 {edge:+.1%}，AUC {m['auc']:.2f}，統計上不太像純運氣（p={m['p_value']:.3f}）。"))
    else:
        notes.append(("warn", f"準確率與基準差 {edge:+.1%}、AUC {m['auc']:.2f}，p={m['p_value']:.2f}：沒有足夠證據說模型有預測力。"))
    if m["train_accuracy"] - m["accuracy"] > 0.10:
        notes.append(("bad", f"訓練準確率 {m['train_accuracy']:.0%} 遠高於樣本外 {m['accuracy']:.0%}：典型過擬合。"))
    if perf["sharpe"] > hold["sharpe"] and perf["psr"] > 0.9:
        notes.append(("good", f"扣成本後 Sharpe {perf['sharpe']:.2f} 優於買入持有 {hold['sharpe']:.2f}。"))
    else:
        notes.append(("warn", f"扣成本後 Sharpe {perf['sharpe']:.2f}，買入持有 {hold['sharpe']:.2f}；PSR {perf['psr']:.0%}。"))
    return notes
