"""Market summary: regime, breadth and a rule-based trend/momentum screen.

Rules are deliberately simple and transparent (no machine learning):
trend (price vs 50/200-day averages) + cross-sectional momentum are the two
technical effects with the longest research record. The `validate` step
backtests the very same rule so you can see whether it has helped.
"""
import numpy as np
import pandas as pd

from app.analysis import Costs
from app.features import rsi
from app.metrics import performance

TOP_N = 10
LABELS = {"bull": "偏多", "hot": "過熱", "bear": "偏空", "neutral": "中性"}


def _frames(closes: pd.DataFrame):
    f = {
        "ma50": closes.rolling(50).mean(),
        "ma200": closes.rolling(200).mean(),
        "ret_1m": closes / closes.shift(21) - 1,
        "ret_3m": closes / closes.shift(63) - 1,
        # 12-1 momentum: last 12 months excluding the latest month (short-term
        # reversal makes the latest month noisy).
        "mom_12_1": closes.shift(21) / closes.shift(252) - 1,
        "vol_60": closes.pct_change().rolling(60).std() * np.sqrt(252),
        "high_252": closes.rolling(252, min_periods=120).max(),
    }
    f["rsi"] = closes.apply(rsi)
    mom = f["mom_12_1"].fillna(f["ret_3m"])
    trend = ((closes > f["ma200"]).astype(float) + (f["ma50"] > f["ma200"]).astype(float)) / 2
    trend = trend.where(f["ma200"].notna())
    f["trend"] = trend
    f["score"] = 100 * (0.35 * mom.rank(axis=1, pct=True)
                        + 0.25 * f["ret_3m"].rank(axis=1, pct=True)
                        + 0.25 * trend
                        + 0.15 * (-f["vol_60"]).rank(axis=1, pct=True))
    return f


def _classify(close, ma50, ma200, score, rsi_value):
    if any(pd.isna(v) for v in (close, ma50, ma200, score)):
        return "neutral"
    up = close > ma200 and ma50 > ma200
    down = close < ma200 and ma50 < ma200
    stretched = close / ma50 - 1 > 0.15
    if up and (rsi_value >= 75 or stretched):
        return "hot"
    if up and score >= 65:
        return "bull"
    if down and score <= 35:
        return "bear"
    return "neutral"


def _reasons(row):
    out = []
    if row["close"] > row["ma200"]:
        out.append("站上 200 日線")
    else:
        out.append("跌破 200 日線")
    out.append("50 日線在 200 日線之上" if row["ma50"] > row["ma200"] else "50 日線在 200 日線之下")
    if pd.notna(row["mom_rank"]):
        out.append(f"12 個月動能排名前 {max(1, round((1 - row['mom_rank']) * 100))}%")
    if row["rsi"] >= 75:
        out.append(f"RSI {row['rsi']:.0f} 偏熱")
    elif row["rsi"] <= 30:
        out.append(f"RSI {row['rsi']:.0f} 超賣")
    if row["close"] / row["ma50"] - 1 > 0.15:
        out.append(f"高於 50 日線 {row['close'] / row['ma50'] - 1:.0%}")
    return "、".join(out)


def screen(closes: pd.DataFrame):
    """Latest reading for every stock in the universe."""
    closes = closes.sort_index().ffill()
    f = _frames(closes)
    last = closes.index[-1]
    mom_rank = f["mom_12_1"].rank(axis=1, pct=True).loc[last]
    rows = []
    for sym in closes.columns:
        c = closes[sym]
        if c.notna().sum() < 210:
            continue
        row = {
            "symbol": sym, "close": float(c.loc[last]),
            "ret_1d": float(c.pct_change().loc[last]),
            "ret_1m": float(f["ret_1m"].loc[last, sym]),
            "ret_3m": float(f["ret_3m"].loc[last, sym]),
            "mom_12_1": float(f["mom_12_1"].loc[last, sym]),
            "ma50": float(f["ma50"].loc[last, sym]), "ma200": float(f["ma200"].loc[last, sym]),
            "rsi": float(f["rsi"].loc[last, sym]),
            "vol_60": float(f["vol_60"].loc[last, sym]),
            "from_high": float(c.loc[last] / f["high_252"].loc[last, sym] - 1),
            "score": float(f["score"].loc[last, sym]),
            "mom_rank": float(mom_rank.get(sym, np.nan)),
        }
        row["signal"] = _classify(row["close"], row["ma50"], row["ma200"], row["score"], row["rsi"])
        row["signal_label"] = LABELS[row["signal"]]
        row["reasons"] = _reasons(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def regime(index_close: pd.Series, closes: pd.DataFrame):
    """Index trend + breadth (share of the universe above its 200-day average)."""
    closes = closes.sort_index().ffill()
    ma200 = closes.rolling(200).mean()
    ma50 = closes.rolling(50).mean()
    valid = ma200.notna()
    breadth200 = ((closes > ma200) & valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
    breadth50 = ((closes > ma50) & ma50.notna()).sum(axis=1) / ma50.notna().sum(axis=1).replace(0, np.nan)
    out = {"breadth200": float(breadth200.iloc[-1]), "breadth50": float(breadth50.iloc[-1]),
           "breadth_history": breadth200.dropna().tail(500)}
    if index_close is not None and len(index_close.dropna()) > 200:
        ic = index_close.dropna()
        ima200, ima50 = ic.rolling(200).mean(), ic.rolling(50).mean()
        out.update({
            "index_close": float(ic.iloc[-1]), "index_date": ic.index[-1].strftime("%Y-%m-%d"),
            "index_ret_1m": float(ic.iloc[-1] / ic.iloc[-22] - 1),
            "index_ret_3m": float(ic.iloc[-1] / ic.iloc[-64] - 1),
            "index_vs_ma200": float(ic.iloc[-1] / ima200.iloc[-1] - 1),
            "index_ma50_above_200": bool(ima50.iloc[-1] > ima200.iloc[-1]),
            "index_from_high": float(ic.iloc[-1] / ic.tail(252).max() - 1),
            "index_vol_20": float(ic.pct_change().tail(20).std() * np.sqrt(252)),
            "index_history": ic.tail(500), "index_ma200": ima200.tail(500),
        })
        above = out["index_vs_ma200"] > 0
    else:
        above = out["breadth200"] >= 0.5
    b = out["breadth200"]
    if above and b >= 0.55:
        out["regime"], out["regime_level"], out["regime_emoji"] = "多頭", "good", "🐮"
        out["regime_text"] = "大盤在 200 日線之上、多數個股也在長期趨勢之上。趨勢型訊號在這種環境下歷史表現較好。"
    elif not above and b <= 0.45:
        out["regime"], out["regime_level"], out["regime_emoji"] = "空頭", "bad", "🐻"
        out["regime_text"] = "大盤在 200 日線之下、多數個股走弱。這種環境下「偏多」名單也容易失靈，歷史上降低持股或分批是較保守的做法。"
    else:
        out["regime"], out["regime_level"], out["regime_emoji"] = "震盪 / 轉折", "warn", "🐢"
        out["regime_text"] = "大盤與個股方向不一致（例如指數靠少數權值股撐著）。訊號可靠度較低，宜觀察廣度是否跟上。"
    return out


def validate(closes: pd.DataFrame, index_close: pd.Series = None, costs: Costs = None,
             top_n: int = TOP_N, rebalance_days: int = 21):
    """Backtest: every month hold up to `top_n` stocks flagged 偏多 (equal slots,
    empty slots in cash) vs an equal-weight portfolio of the whole universe.

    Uses only data available at each rebalance date. Survivorship bias: the
    universe is *today's* large caps, which flatters every strategy here.
    """
    costs = costs or Costs()
    closes = closes.sort_index().ffill()
    f = _frames(closes)
    start = 252
    if len(closes) < start + 3 * rebalance_days:
        return None
    strat, univ, months, beat = [], [], 0, 0
    held = set()
    dates = list(range(start, len(closes) - 1, rebalance_days))
    for k, i in enumerate(dates):
        j = dates[k + 1] if k + 1 < len(dates) else len(closes) - 1
        t = closes.index[i]
        sig = {s: _classify(closes.at[t, s], f["ma50"].at[t, s], f["ma200"].at[t, s],
                            f["score"].at[t, s], f["rsi"].at[t, s]) for s in closes.columns}
        ranked = f["score"].loc[t].dropna().sort_values(ascending=False)
        new = {s for s in ranked.index if sig[s] == "bull"}
        new = set(list(ranked[ranked.index.isin(new)].index)[:top_n])
        buys, sells = len(new - held) / top_n, len(held - new) / top_n
        cost = buys * (costs.fee_rate + costs.slippage_rate) + \
            sells * (costs.fee_rate + costs.tax_rate + costs.slippage_rate)
        held = new
        window = closes.iloc[i:j + 1]
        rel = window / window.iloc[0]
        avail = [s for s in closes.columns if pd.notna(closes.at[t, s])]
        s_path = (rel[list(held)].sum(axis=1) + (top_n - len(held))) / top_n if held else \
            pd.Series(1.0, index=window.index)
        u_path = rel[avail].mean(axis=1)
        s_ret = s_path.pct_change().iloc[1:]
        if len(s_ret):
            s_ret.iloc[0] -= cost
        strat.append(s_ret)
        univ.append(u_path.pct_change().iloc[1:])
        months += 1
        beat += int(s_path.iloc[-1] * (1 - cost) > u_path.iloc[-1])
    s_ret, u_ret = pd.concat(strat), pd.concat(univ)
    out = {"strategy": performance(s_ret), "universe": performance(u_ret),
           "periods": months, "hit_rate": beat / months if months else 0.0,
           "top_n": top_n, "strategy_equity": (1 + s_ret).cumprod(),
           "universe_equity": (1 + u_ret).cumprod()}
    if index_close is not None:
        ir = index_close.reindex(closes.index).ffill().pct_change().loc[s_ret.index]
        out["index"] = performance(ir)
        out["index_equity"] = (1 + ir).cumprod()
    return out
