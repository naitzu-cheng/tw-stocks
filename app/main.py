import json
from urllib.parse import urlencode
import os
from contextlib import asynccontextmanager
import math
import threading
import time
from pathlib import Path
from typing import Optional

import markdown
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from urllib.parse import quote, urlparse
from fastapi.templating import Jinja2Templates
from plotly.subplots import make_subplots

from app.analysis import (Costs, MODEL_LABELS, build_analysis, ma_robustness,
                          walk_forward)
from app.collector import ensure_fresh, ensure_fresh_many, fetch_and_store, get_data
from app.features import atr
from app.markets import is_taiwan, normalize_symbol, profile_for
from app.portfolio import WEIGHTINGS, build_portfolio_analysis
from app.screener import regime, screen, validate
from app.universe import MARKETS

ROOT = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
@asynccontextmanager
async def lifespan(_app):
    # Fill the market-summary cache in the background so the first visitor waits less.
    if os.environ.get("PREFETCH_MARKETS", "1") == "1":
        threading.Thread(target=_warm_markets, daemon=True).start()
    yield


app = FastAPI(title="TW & US Stocks Quant Lab", lifespan=lifespan)


@app.middleware("http")
async def drop_empty_query_params(request: Request, call_next):
    """HTML forms send blank inputs as `fee_rate=`; treat them as "not given"
    so the default (e.g. the market's automatic cost) applies instead of a 422."""
    raw = request.scope.get("query_string", b"")
    if raw:
        from urllib.parse import parse_qsl, urlencode as _enc
        pairs = parse_qsl(raw.decode("latin-1"), keep_blank_values=True)
        kept = [(k, v) for k, v in pairs if v.strip() != ""]
        if len(kept) != len(pairs):
            request.scope["query_string"] = _enc(kept).encode("latin-1")
    return await call_next(request)

PLOT_LAYOUT = dict(template="plotly_white", margin=dict(l=40, r=20, t=50, b=70),
                   font=dict(family="Nunito, Huninn, PingFang TC, sans-serif", color="#3d3550"),
                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   colorway=["#8b7cf6", "#ff8fab", "#4cc9a6", "#ffb07c", "#6cb8ff", "#f5c84c"],
                   legend=dict(orientation="h", y=-0.06, yanchor="top", x=0))


# ---------------------------------------------------------------- helpers
def clean(value):
    """Make numpy / pandas / inf values JSON-safe."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, pd.DataFrame):
        return clean(value.to_dict(orient="records"))
    return value


def resolve(symbol: str, refresh: bool = True):
    try:
        symbol = normalize_symbol(symbol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    error = None
    if refresh:
        symbol, error = ensure_fresh(symbol)
    return symbol, error


def costs_for(symbol, fee_rate=None, tax_rate=None, slippage_rate=None):
    profile = profile_for(symbol)
    return profile, Costs(
        fee_rate=profile.fee_rate if fee_rate is None else fee_rate,
        tax_rate=profile.tax_rate if tax_rate is None else tax_rate,
        slippage_rate=profile.slippage_rate if slippage_rate is None else slippage_rate,
    )


def benchmark_close(symbol, refresh=True):
    bench = profile_for(symbol).benchmark
    if bench == symbol:
        return None
    if refresh:
        ensure_fresh(bench)
    data = get_data(bench, limit=None)
    return None if data.empty else data["Close"]


def load_prices(symbol):
    df = get_data(symbol, limit=None)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"沒有 {symbol} 的資料，請確認代號（台股如 2330、美股如 AAPL）")
    return df


_cache = {}


def cached(key, fn):
    """Small in-memory cache: model training is the slow part of the dashboard."""
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 6 * 3600:
        return hit[1]
    value = fn()
    _cache[key] = (time.time(), value)
    if len(_cache) > 64:
        _cache.pop(next(iter(_cache)))
    return value


def to_html(fig, first=False):
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn" if first else False,
                       config={"displaylogo": False, "responsive": True})



# ---------------------------------------------------------------- market switch
MARKET_DEFAULTS = {
    "TW": {"symbol": "2330.TW", "portfolio": "2330.TW,2317.TW,2454.TW,0050.TW",
           "chips": ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2881.TW", "0050.TW", "006208.TW", "00878.TW"]},
    "US": {"symbol": "AAPL", "portfolio": "AAPL,MSFT,NVDA,SPY",
           "chips": ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ"]},
}
COOKIE_AGE = 365 * 24 * 3600


def market_of(symbol):
    return "TW" if is_taiwan(symbol) else "US"


def current_market(request: Request):
    value = (request.cookies.get("market") or "").upper()
    return value if value in MARKET_DEFAULTS else "TW"


def last_symbol(request: Request, market: str):
    raw = request.cookies.get(f"last_{market}")
    try:
        sym = normalize_symbol(raw) if raw else None
    except ValueError:
        sym = None
    if sym and market_of(sym) == market:
        return sym
    return MARKET_DEFAULTS[market]["symbol"]


def page(request: Request, name: str, ctx: dict, market: str = None, symbol: str = None):
    """Render a page with the shared navigation context and remember the market."""
    market = market or current_market(request)
    shown = symbol or last_symbol(request, market)
    resp = templates.TemplateResponse(request, name, {
        "mkt": market, "path": request.url.path, "nav_symbol": shown,
        "chips": MARKET_DEFAULTS[market]["chips"], **ctx})
    resp.set_cookie("market", market, max_age=COOKIE_AGE, samesite="lax")
    if symbol:
        resp.set_cookie(f"last_{market_of(symbol)}", symbol, max_age=COOKIE_AGE, samesite="lax")
    return resp


@app.get("/switch-market")
def switch_market(request: Request, to: str = "TW", next: str = "/market"):
    to = to.upper()
    if to not in MARKET_DEFAULTS:
        raise HTTPException(status_code=400, detail="to 必須是 TW 或 US")
    path = urlparse(next).path or "/market"
    if not path.startswith("/") or path.startswith("//"):
        path = "/market"
    if path in ("/dashboard", "/risk-dashboard"):
        url = f"{path}?symbol={quote(last_symbol(request, to))}"
    elif path in ("/market", "/"):
        url = f"/market?market={to}"
    elif path == "/portfolio":
        url = "/portfolio"
    else:
        url = path
    resp = RedirectResponse(url, status_code=303)
    resp.set_cookie("market", to, max_age=COOKIE_AGE, samesite="lax")
    return resp

# ---------------------------------------------------------------- JSON API
@app.get("/")
def root():
    return RedirectResponse("/market")


@app.post("/collect")
def collect(symbol: str = "2330.TW", period: str = "5y", interval: str = "1d"):
    try:
        resolved, rows = fetch_and_store(symbol, period=period, interval=interval)
        return {"symbol": resolved, "rows_saved": rows}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/data")
def data(symbol: str = "2330.TW", limit: int = 100):
    symbol, _ = resolve(symbol, refresh=False)
    df = get_data(symbol, limit=limit)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No data")
    out = df.reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return clean(out.to_dict(orient="records"))


@app.get("/analysis")
def analysis(symbol: str = "2330.TW", fast_window: int = 20, slow_window: int = 60,
             fee_rate: Optional[float] = None, tax_rate: Optional[float] = None,
             slippage_rate: Optional[float] = None, refresh: bool = True):
    symbol, err = resolve(symbol, refresh)
    _, costs = costs_for(symbol, fee_rate, tax_rate, slippage_rate)
    bench = benchmark_close(symbol, refresh)
    try:
        result = build_analysis(load_prices(symbol), fast_window, slow_window, costs,
                                None if bench is None else bench.pct_change())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return clean({"symbol": symbol, "fast_window": fast_window, "slow_window": slow_window,
                  "costs": vars(costs), "refresh_error": err, **result["metrics"]})


@app.get("/robustness")
def robustness(symbol: str = "2330.TW", fast_window: int = 20, slow_window: int = 60,
               fee_rate: Optional[float] = None, tax_rate: Optional[float] = None,
               slippage_rate: Optional[float] = None, refresh: bool = True):
    symbol, _ = resolve(symbol, refresh)
    _, costs = costs_for(symbol, fee_rate, tax_rate, slippage_rate)
    try:
        r = ma_robustness(load_prices(symbol), costs, fast_window, slow_window)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    r.pop("heatmap")
    return clean({"symbol": symbol, **r})


@app.get("/model")
def model(symbol: str = "2330.TW", model_name: str = "logistic", horizon: int = 5,
          step: int = 63, band: float = 0.02, compare: bool = False,
          fast_window: int = 20, slow_window: int = 60,
          fee_rate: Optional[float] = None, tax_rate: Optional[float] = None,
          slippage_rate: Optional[float] = None, refresh: bool = True):
    symbol, _ = resolve(symbol, refresh)
    _, costs = costs_for(symbol, fee_rate, tax_rate, slippage_rate)
    try:
        r = run_model(symbol, model_name, horizon, step, band, compare, fast_window,
                      slow_window, costs, benchmark_close(symbol, refresh))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return clean({"symbol": symbol, **r["model_metrics"]})


def run_model(symbol, model_name, horizon, step, band, compare, fast, slow, costs, bench):
    if step < 5:
        raise ValueError("step（重新訓練間隔）至少 5 天")
    df = load_prices(symbol)
    key = ("model", symbol, str(df.index[-1]), len(df), model_name, horizon, step, band,
           compare, fast, slow, tuple(vars(costs).values()))
    return cached(key, lambda: walk_forward(
        df, model_name, horizon=horizon, step=step, band=band, costs=costs,
        benchmark_close=bench, fast_window=fast, slow_window=slow, compare_models=compare))


@app.get("/risk")
def risk(symbol: str = "2330.TW", capital: float = 1_000_000, risk_per_trade: float = 0.01,
         stop_loss_pct: Optional[float] = None, atr_multiple: float = 2.0,
         refresh: bool = True):
    symbol, _ = resolve(symbol, refresh)
    df = load_prices(symbol)
    if capital <= 0 or risk_per_trade <= 0 or atr_multiple <= 0:
        raise HTTPException(status_code=400, detail="資金與風險參數必須為正數")
    if stop_loss_pct is not None and stop_loss_pct <= 0:
        raise HTTPException(status_code=400, detail="停損比例必須為正數")
    profile = profile_for(symbol)
    close = float(df["Close"].iloc[-1])
    atr_value = float(atr(df).iloc[-1])
    atr_stop_pct = atr_multiple * atr_value / close
    stop_pct = stop_loss_pct if stop_loss_pct is not None else atr_stop_pct
    risk_budget = capital * risk_per_trade
    shares = int(min(risk_budget / (close * stop_pct), capital / close))
    lots = shares // profile.lot_size if profile.lot_size > 1 else None
    vol = float(df["Close"].pct_change().tail(60).std() * math.sqrt(252))
    return clean({
        "symbol": symbol, "market": profile.label, "currency": profile.currency,
        "latest_close": close, "capital": capital, "risk_per_trade": risk_per_trade,
        "atr_14": atr_value, "atr_multiple": atr_multiple, "atr_stop_pct": atr_stop_pct,
        "stop_loss_pct": stop_pct, "stop_method": "固定比例" if stop_loss_pct else f"{atr_multiple} × ATR",
        "stop_price": close * (1 - stop_pct), "risk_budget": risk_budget,
        "suggested_shares": shares, "board_lots": lots, "lot_size": profile.lot_size,
        "position_value": shares * close, "position_weight": shares * close / capital,
        "annual_volatility_60d": vol,
        "note": "教育用途的風險估算，不是下單建議",
    })


# ---------------------------------------------------------------- HTML pages
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, symbol: Optional[str] = None, fast_window: int = 20,
              slow_window: int = 60, fee_rate: Optional[float] = None,
              tax_rate: Optional[float] = None, slippage_rate: Optional[float] = None,
              model_name: str = "logistic", horizon: int = 5, step: int = 63,
              band: float = 0.02, compare: bool = False):
    symbol, refresh_error = resolve(symbol or last_symbol(request, current_market(request)))
    profile, costs = costs_for(symbol, fee_rate, tax_rate, slippage_rate)
    df = load_prices(symbol)
    bench = benchmark_close(symbol)
    bench_ret = None if bench is None else bench.pct_change()
    errors = {}
    try:
        strat = build_analysis(df, fast_window, slow_window, costs, bench_ret)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        robust = ma_robustness(df, costs, fast_window, slow_window)
    except ValueError as e:
        robust, errors["robust"] = None, str(e)

    charts = {"price": to_html(price_figure(strat["prices"], strat["metrics"]["trade_log"]), first=True),
              "strategy": to_html(strategy_figure(strat["prices"], bench))}
    if robust:
        charts["heatmap"] = to_html(heatmap_figure(robust))

    return page(request, "dashboard.html", {
        "symbol": symbol, "profile": profile, "costs": costs,
        "params": {"fast_window": fast_window, "slow_window": slow_window,
                   "fee_rate": fee_rate, "tax_rate": tax_rate, "slippage_rate": slippage_rate,
                   "model_name": model_name, "horizon": horizon, "step": step,
                   "band": band, "compare": compare},
        "models": MODEL_LABELS, "m": strat["metrics"], "robust": robust,
        "charts": charts, "ml_query": urlencode({**dict(request.query_params), "symbol": symbol}),
        "errors": errors, "refresh_error": refresh_error,
        "latest_close": float(df["Close"].iloc[-1]),
        "latest_date": df.index[-1].strftime("%Y-%m-%d"),
    }, market=market_of(symbol), symbol=symbol)


@app.get("/dashboard/ml")
def dashboard_ml(request: Request, symbol: str = "2330.TW", fast_window: int = 20,
                 slow_window: int = 60, fee_rate: Optional[float] = None,
                 tax_rate: Optional[float] = None, slippage_rate: Optional[float] = None,
                 model_name: str = "logistic", horizon: int = 5, step: int = 63,
                 band: float = 0.02, compare: bool = False):
    """Machine-learning tab, loaded only when the user opens it (it is slow)."""
    symbol, _ = resolve(symbol, refresh=False)
    _, costs = costs_for(symbol, fee_rate, tax_rate, slippage_rate)
    try:
        ml = run_model(symbol, model_name, horizon, step, band, compare, fast_window,
                       slow_window, costs, benchmark_close(symbol, refresh=False))
    except ValueError as e:
        return {"html": f'<div class="note warn">{e}</div>', "figure": None}
    html = templates.get_template("ml_panel.html").render(ml=ml["model_metrics"])
    return {"html": html, "figure": json.loads(model_figure(ml["model_prices"]).to_json())}


def price_figure(bt, trades):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=bt.index, open=bt["Open"], high=bt["High"], low=bt["Low"],
                                 close=bt["Close"], name="K 線", increasing_line_color="#f2557a",
                                 decreasing_line_color="#1fa776"), 1, 1)
    fig.add_trace(go.Scatter(x=bt.index, y=bt["fast_ma"], name="短均線",
                             line=dict(width=1.2, color="#ffb07c")), 1, 1)
    fig.add_trace(go.Scatter(x=bt.index, y=bt["slow_ma"], name="長均線",
                             line=dict(width=1.2, color="#6cb8ff")), 1, 1)
    for action, color, marker in (("買入", "#f2557a", "triangle-up"), ("賣出", "#1fa776", "triangle-down")):
        pts = [t for t in trades if t["action"] == action]
        fig.add_trace(go.Scatter(x=[t["date"] for t in pts], y=[t["price"] for t in pts],
                                 mode="markers", name=action,
                                 marker=dict(color=color, symbol=marker, size=11,
                                             line=dict(width=1, color="white"))), 1, 1)
    if "Volume" in bt:
        fig.add_trace(go.Bar(x=bt.index, y=bt["Volume"], name="成交量",
                             marker_color="#d9d2ee", showlegend=False), 2, 1)
    from app.features import rsi
    fig.add_trace(go.Scatter(x=bt.index, y=rsi(bt["Close"]), name="RSI(14)",
                             line=dict(color="#ff8fab", width=1.2)), 3, 1)
    for level in (30, 70):
        fig.add_hline(y=level, line=dict(dash="dot", color="#bbb", width=1), row=3, col=1)
    fig.update_layout(height=720, xaxis_rangeslider_visible=False, **PLOT_LAYOUT)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])],
                     rangeselector=dict(buttons=[dict(count=6, label="6M", step="month"),
                                                 dict(count=1, label="1Y", step="year"),
                                                 dict(count=3, label="3Y", step="year"),
                                                 dict(step="all", label="全部")]), row=1, col=1)
    return fig


def strategy_figure(bt, bench):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=bt.index, y=bt["strategy_equity"], name="均線策略（扣成本）",
                             line=dict(color="#8b7cf6", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=bt.index, y=bt["buy_hold_equity"], name="買入持有",
                             line=dict(color="#b9b2c6", width=1.5)), 1, 1)
    if bench is not None:
        b = bench.reindex(bt.index).ffill()
        fig.add_trace(go.Scatter(x=bt.index, y=b / b.iloc[0], name="大盤基準",
                                 line=dict(color="#f5b84c", width=1.2, dash="dot")), 1, 1)
    for col, name, color in (("strategy_equity", "策略回撤", "#8b7cf6"),
                             ("buy_hold_equity", "持有回撤", "#888")):
        dd = bt[col] / bt[col].cummax() - 1
        fig.add_trace(go.Scatter(x=bt.index, y=dd, name=name, fill="tozeroy",
                                 line=dict(color=color, width=1), showlegend=False), 2, 1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_layout(height=560, title="資產曲線與回撤", **PLOT_LAYOUT)
    return fig


def heatmap_figure(robust):
    heat = robust["heatmap"]
    fig = go.Figure(go.Heatmap(z=heat.values, x=[str(c) for c in heat.columns],
                               y=[str(i) for i in heat.index], colorscale="RdYlGn", zmid=0,
                               text=[["" if np.isnan(v) else f"{v:.2f}" for v in row] for row in heat.values],
                               texttemplate="%{text}",
                               hovertemplate="短 %{y} / 長 %{x}<br>Sharpe %{z:.2f}<extra></extra>"))
    fig.update_layout(height=420, title="各組均線參數的 Sharpe（全期間、扣成本）",
                      xaxis_title="長均線", yaxis_title="短均線", **PLOT_LAYOUT)
    return fig


def model_figure(bt):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=bt.index, y=bt["strategy_equity"], name="模型策略（樣本外）",
                             line=dict(color="#8b7cf6", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=bt.index, y=bt["buy_hold_equity"], name="買入持有",
                             line=dict(color="#b9b2c6", width=1.5)), 1, 1)
    fig.add_trace(go.Scatter(x=bt.index, y=bt["probability"], name="預測上漲機率",
                             line=dict(color="#ff8fab", width=1)), 2, 1)
    fig.add_trace(go.Scatter(x=bt.index, y=bt["threshold"], name="進場門檻",
                             line=dict(color="#f2557a", width=1, dash="dot")), 2, 1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_layout(height=560, title="Walk-forward 樣本外資產曲線", **PLOT_LAYOUT)
    return fig


@app.get("/portfolio", response_class=HTMLResponse)
def portfolio(request: Request, symbols: Optional[str] = None,
              weighting: str = "inverse_vol", rebalance_days: int = 21):
    symbols = symbols or MARKET_DEFAULTS[current_market(request)]["portfolio"]
    try:
        symbol_list = list(dict.fromkeys(normalize_symbol(s) for s in symbols.split(",") if s.strip()))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    resolved, errors = [], []
    for s in symbol_list:
        r, err = ensure_fresh(s)
        resolved.append(r)
        if err:
            errors.append(f"{s}：{err}")
    bench = benchmark_close(resolved[0]) if resolved else None
    try:
        result = build_portfolio_analysis(resolved, limit=None, weighting=weighting,
                                          rebalance_days=max(rebalance_days, 1),
                                          benchmark_close=bench)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    fig = go.Figure(go.Scatter(x=result["equity"].index, y=result["equity"],
                               name=result["metrics"]["weighting_label"],
                               line=dict(color="#8b7cf6", width=2)))
    fig.update_layout(height=460, title="投資組合資產曲線", **PLOT_LAYOUT)
    corr = result["correlation"]
    cfig = go.Figure(go.Heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index),
                                zmin=-1, zmax=1, colorscale="RdBu_r",
                                text=np.round(corr.values, 2), texttemplate="%{text}"))
    cfig.update_layout(height=380, title="日報酬相關係數", **PLOT_LAYOUT)
    markets_used = {market_of(s) for s in resolved}
    return page(request, "portfolio.html", {
        "symbols": ",".join(resolved), "weighting": weighting, "weightings": WEIGHTINGS,
        "rebalance_days": rebalance_days, "m": result["metrics"],
        "per_stock": result["per_stock"], "errors": errors,
        "chart": to_html(fig, first=True), "corr_chart": to_html(cfig),
    }, market=markets_used.pop() if len(markets_used) == 1 else None)


@app.get("/risk-dashboard", response_class=HTMLResponse)
def risk_dashboard(request: Request, symbol: Optional[str] = None, capital: Optional[float] = None,
                   risk_per_trade: float = 0.01, stop_loss_pct: Optional[float] = None,
                   atr_multiple: float = 2.0):
    symbol = symbol or last_symbol(request, current_market(request))
    if capital is None:
        capital = 1_000_000 if market_of(normalize_symbol(symbol)) == "TW" else 30_000
    r = risk(symbol, capital, risk_per_trade, stop_loss_pct, atr_multiple)
    return page(request, "risk.html", {"r": r, "symbol": r["symbol"], "stop_loss_input": stop_loss_pct},
                market=market_of(r["symbol"]), symbol=r["symbol"])


@app.get("/guide", response_class=HTMLResponse)
def guide(request: Request):
    text = (ROOT / "docs" / "GUIDE.md").read_text(encoding="utf-8")
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
    return page(request, "guide.html", {"body": html})


def _pct(value, digits=1):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value * 100:.{digits}f}%"


def _num(value, digits=2):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "∞" if value == float("inf") else "—"
    return f"{value:,.{digits}f}"


templates.env.filters["pct"] = _pct
templates.env.filters["spct"] = lambda v, d=1: ("+" if v is not None and v > 0 else "") + _pct(v, d)
templates.env.filters["num"] = _num
templates.env.filters["plain"] = lambda v: ("%.7f" % v).rstrip("0").rstrip(".") if v else "0"


# ---------------------------------------------------------------- market summary
def market_summary(market: str):
    if market not in MARKETS:
        raise HTTPException(status_code=400, detail="market 必須是 TW 或 US")
    cfg = MARKETS[market]
    symbols = cfg["universe"] + [cfg["index"]]
    errors = ensure_fresh_many(symbols)
    closes = {}
    for s in cfg["universe"]:
        df = get_data(s, limit=None)
        if not df.empty:
            closes[s] = df["Close"]
    if len(closes) < 5:
        raise HTTPException(status_code=502, detail="成分股資料不足，可能是資料來源暫時無法連線：" + "；".join(errors[:3]))
    closes = pd.DataFrame(closes).sort_index()
    idx = get_data(cfg["index"], limit=None)
    index_close = None if idx.empty else idx["Close"]
    key = ("market", market, str(closes.index[-1]), closes.shape)

    def build():
        _, costs = costs_for(cfg["universe"][0])
        return {"screen": screen(closes), "regime": regime(index_close, closes),
                "validate": validate(closes, index_close, costs)}
    result = cached(key, build)
    return cfg, result, errors


@app.get("/market.json")
def market_json(market: str = "TW"):
    cfg, r, errors = market_summary(market.upper())
    reg = {k: v for k, v in r["regime"].items() if not isinstance(v, pd.Series)}
    val = r["validate"]
    val = None if val is None else {k: v for k, v in val.items() if not isinstance(v, pd.Series)}
    return clean({"market": market.upper(), "regime": reg, "validation": val,
                  "stocks": r["screen"].to_dict(orient="records"), "errors": errors})


@app.get("/market", response_class=HTMLResponse)
def market_page(request: Request, market: Optional[str] = None):
    market = (market or current_market(request)).upper()
    cfg, r, errors = market_summary(market)
    reg, table, val = r["regime"], r["screen"], r["validate"]
    charts = {}
    if "index_history" in reg:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                            vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=reg["index_history"].index, y=reg["index_history"],
                                 name=cfg["index_label"], line=dict(color="#8b7cf6", width=2)), 1, 1)
        fig.add_trace(go.Scatter(x=reg["index_ma200"].index, y=reg["index_ma200"],
                                 name="200 日線", line=dict(color="#f5b84c", width=1.5, dash="dot")), 1, 1)
        bh = reg["breadth_history"]
        fig.add_trace(go.Scatter(x=bh.index, y=bh, name="站上 200 日線的比例",
                                 fill="tozeroy", line=dict(color="#ff8fab", width=1)), 2, 1)
        fig.add_hline(y=0.5, line=dict(dash="dot", color="#999", width=1), row=2, col=1)
        fig.update_yaxes(tickformat=".0%", range=[0, 1], row=2, col=1)
        fig.update_layout(height=520, title="大盤與市場廣度", **PLOT_LAYOUT)
        charts["regime"] = to_html(fig, first=True)
    if val:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=val["strategy_equity"].index, y=val["strategy_equity"],
                                 name=f"每月持有偏多前 {val['top_n']} 名", line=dict(color="#8b7cf6", width=2)))
        fig.add_trace(go.Scatter(x=val["universe_equity"].index, y=val["universe_equity"],
                                 name="名單內全部等權重", line=dict(color="#b9b2c6", width=1.5)))
        if "index_equity" in val:
            fig.add_trace(go.Scatter(x=val["index_equity"].index, y=val["index_equity"],
                                     name=cfg["index_label"], line=dict(color="#f5b84c", width=1.2, dash="dot")))
        fig.update_layout(height=420, title="這套規則過去有沒有用？（含交易成本）", **PLOT_LAYOUT)
        charts["validate"] = to_html(fig, first="regime" not in charts)
    groups = {k: table[table["signal"] == k].to_dict(orient="records") for k in ("bull", "hot", "bear")}
    return page(request, "market.html", {
        "market": market, "cfg": cfg, "markets": MARKETS, "reg": reg, "val": val,
        "groups": groups, "rows": table.to_dict(orient="records"), "charts": charts,
        "errors": errors,
        "as_of": reg.get("index_date"),
    }, market=market)


def _warm_markets():
    for m in MARKETS:
        try:
            market_summary(m)
        except Exception:
            pass
