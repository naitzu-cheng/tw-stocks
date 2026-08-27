from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from app.collector import fetch_and_store, get_data
from app.analysis import build_analysis, build_model_analysis
from app.portfolio import build_portfolio_analysis
import plotly.graph_objs as go
import plotly.io as pio

app = FastAPI(title="TW Stocks Mini App")


@app.get("/")
def root():
    return {"message": "TW Stocks Mini App. Use /collect, /data, /plot, /analysis, /model, /risk, /portfolio, /dashboard"}


@app.post("/collect")
def collect(symbol: str = "2330.TW", period: str = "1y", interval: str = "1d"):
    try:
        rows = fetch_and_store(symbol, period=period, interval=interval)
        return {"symbol": symbol, "rows_saved": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data")
def data(symbol: str = "2330.TW", limit: int = 100):
    df = get_data(symbol, limit=limit)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No data")
    return df.tail(limit).reset_index().to_dict(orient="records")


@app.get("/plot", response_class=HTMLResponse)
def plot(symbol: str = "2330.TW"):
    df = get_data(symbol, limit=365)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No data")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name=symbol))
    fig.update_layout(title=f"{symbol} price")
    html = pio.to_html(fig, full_html=False)
    return HTMLResponse(html)


@app.get("/analysis")
def analysis(symbol: str = "2330.TW", limit: int = 2000,
             fast_window: int = 20, slow_window: int = 60,
             fee_rate: float = 0.001425, slippage_rate: float = 0.0005):
    try:
        result = build_analysis(get_data(symbol, limit=limit), fast_window, slow_window,
                                fee_rate, slippage_rate)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if result is None:
        raise HTTPException(status_code=404, detail="No data")
    return {"symbol": symbol, "fast_window": fast_window,
            "slow_window": slow_window, **result["metrics"]}


@app.get("/model")
def model(symbol: str = "2330.TW", limit: int = 2000,
          fast_window: int = 20, slow_window: int = 60,
          fee_rate: float = 0.001425, slippage_rate: float = 0.0005,
          model_name: str = "random_forest"):
    try:
        result = build_model_analysis(get_data(symbol, limit=limit), fast_window,
                                      slow_window, fee_rate, slippage_rate, model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if result is None:
        raise HTTPException(status_code=404, detail="No data")
    return {"symbol": symbol, "fast_window": fast_window,
            "slow_window": slow_window, **result["model_metrics"]}


@app.get("/risk")
def risk(symbol: str = "2330.TW", capital: float = 100000,
         risk_per_trade: float = 0.01, stop_loss_pct: float = 0.05):
    data = get_data(symbol, limit=1)
    if data is None or data.empty or "Close" not in data:
        raise HTTPException(status_code=404, detail="No data")
    if capital <= 0 or risk_per_trade <= 0 or stop_loss_pct <= 0:
        raise HTTPException(status_code=400, detail="capital and risk settings must be positive")
    close = float(data["Close"].iloc[-1])
    risk_budget = capital * risk_per_trade
    shares_by_risk = int(risk_budget / (close * stop_loss_pct))
    shares_by_capital = int(capital / close)
    shares = min(shares_by_risk, shares_by_capital)
    return {"symbol": symbol, "latest_close": close, "capital": capital,
            "risk_per_trade": risk_per_trade, "stop_loss_pct": stop_loss_pct,
            "risk_budget": risk_budget, "suggested_shares": shares,
            "position_value": shares * close,
            "note": "教育用途的固定百分比停損估算，不是下單建議"}


@app.get("/risk-dashboard", response_class=HTMLResponse)
def risk_dashboard(symbol: str = "2330.TW", capital: float = 100000,
                   risk_per_trade: float = 0.01, stop_loss_pct: float = 0.05):
    result = risk(symbol, capital, risk_per_trade, stop_loss_pct)
    return HTMLResponse(f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>風險管理</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:900px;margin:32px auto;padding:0 20px;color:#17202a;background:#f5f7f9}}.panel{{background:white;border:1px solid #dfe5ea;padding:20px;margin-top:18px}}form{{display:flex;gap:12px;align-items:end;flex-wrap:wrap}}label{{display:grid;gap:6px}}input{{padding:9px;border:1px solid #bac5ce;width:120px}}button{{padding:10px 18px;background:#0b6e69;color:white;border:0;border-radius:4px}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:18px}}article{{background:white;padding:16px;border-top:3px solid #0b6e69}}article span{{display:block;color:#66737d;font-size:13px}}strong{{display:block;font-size:22px;margin-top:8px}}.hint{{color:#66737d}}@media(max-width:700px){{.metrics{{grid-template-columns:1fr}}}}</style></head>
<body><h1>風險管理</h1><p class="hint">用固定比例估算單筆交易最多可承擔多少損失與建議股數。</p><form method="get"><label>股票代號<input name="symbol" value="{symbol}"></label><label>總資金<input type="number" name="capital" value="{capital}" min="1"></label><label>單筆風險比例<input type="number" name="risk_per_trade" value="{risk_per_trade}" step="0.001" min="0.001"></label><label>停損比例<input type="number" name="stop_loss_pct" value="{stop_loss_pct}" step="0.01" min="0.001"></label><button type="submit">計算</button></form><div class="metrics"><article><span>最新收盤價</span><strong>{result['latest_close']:.2f}</strong></article><article><span>可承擔損失</span><strong>{result['risk_budget']:.2f}</strong></article><article><span>估算建議股數</span><strong>{result['suggested_shares']}</strong></article><article><span>估算部位金額</span><strong>{result['position_value']:.2f}</strong></article></div><div class="panel"><b>怎麼看？</b><p>這是風險控制的估算，不是預測股價，也不會自動下單。實際交易還要考慮整股／零股、流動性、漲跌停與交易稅。</p></div></body></html>""")


@app.get("/portfolio", response_class=HTMLResponse)
def portfolio(symbols: str = "2330.TW,006208.TW", limit: int = 2000):
    symbol_list = list(dict.fromkeys(
        symbol.strip() for symbol in symbols.split(",") if symbol.strip()
    ))
    try:
        for symbol in symbol_list:
            fetch_and_store(symbol, period="1y", interval="1d")
        result = build_portfolio_analysis(symbol_list, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"資料更新失敗：{e}")
    metrics = result["metrics"]
    fig = go.Figure(go.Scatter(x=result["equity"].index, y=result["equity"], name="等權重投資組合"))
    fig.update_layout(title="等權重投資組合資產曲線", template="plotly_white", height=500)
    chart = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    weights = "".join(f"<li>{symbol}: {weight:.1%}</li>" for symbol, weight in metrics["weights"].items())
    correlation = result["correlation"].round(2).to_html(classes="table")
    return HTMLResponse(f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>投資組合分析</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#17202a;background:#f5f7f9}}.panel,.chart{{background:white;border:1px solid #dfe5ea;padding:18px;margin-top:18px}}form{{display:flex;gap:12px;align-items:end;flex-wrap:wrap}}label{{display:grid;gap:6px}}input{{padding:9px;border:1px solid #bac5ce;width:300px}}button{{padding:10px 18px;background:#0b6e69;color:white;border:0;border-radius:4px}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}article{{background:white;padding:15px;border-top:3px solid #0b6e69}}article span{{display:block;color:#66737d;font-size:13px}}strong{{display:block;font-size:21px;margin-top:8px}}.table{{border-collapse:collapse}}.table td,.table th{{padding:8px;border:1px solid #dfe5ea}}@media(max-width:700px){{.metrics{{grid-template-columns:repeat(2,1fr)}}}}</style></head>
<body><h1>投資組合分析</h1><p>資料期間：{metrics['start']} 至 {metrics['end']}，每支股票等權重配置。</p><form method="get"><label>股票代號（逗號分隔）<input name="symbols" value="{symbols}"></label><button type="submit">更新組合</button></form>
<div class="metrics"><article><span>總報酬</span><strong>{metrics['portfolio_return']:.2%}</strong></article><article><span>年化波動</span><strong>{metrics['annual_volatility']:.2%}</strong></article><article><span>最大回撤</span><strong>{metrics['max_drawdown']:.2%}</strong></article><article><span>Sharpe</span><strong>{metrics['sharpe']:.2f}</strong></article></div><div class="panel"><h2>配置</h2><ul>{weights}</ul><h2>相關係數</h2>{correlation}</div><div class="chart">{chart}</div></body></html>""")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(symbol: str = "2330.TW", fast_window: int = 20, slow_window: int = 60,
              fee_rate: float = 0.001425, slippage_rate: float = 0.0005,
              model_name: str = "random_forest"):
    refresh_error = None
    try:
        fetch_and_store(symbol, period="1y", interval="1d")
    except Exception as e:
        refresh_error = str(e)
    try:
        result = build_model_analysis(get_data(symbol, limit=2000), fast_window,
                                      slow_window, fee_rate, slippage_rate, model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if result is None:
        raise HTTPException(status_code=404, detail="No data")

    prices = result["prices"]
    metrics = result["metrics"]
    model_prices = result["model_prices"]
    model_metrics = result["model_metrics"]

    price_fig = go.Figure(go.Candlestick(x=prices.index, open=prices["Open"],
                                         high=prices["High"], low=prices["Low"],
                                         close=prices["Close"], name="價格"))
    price_fig.update_layout(title="一般 K 線", template="plotly_white", height=560,
                            xaxis_rangeslider_visible=False)
    trade_log = metrics["trade_log"]
    for action, color, marker_symbol in [("買入", "green", "triangle-up"), ("賣出", "red", "triangle-down")]:
        points = [trade for trade in trade_log if trade["action"] == action]
        price_fig.add_trace(go.Scatter(
            x=[trade["date"] for trade in points],
            y=[trade["price"] for trade in points],
            mode="markers",
            marker={"color": color, "symbol": marker_symbol, "size": 12},
            name=action,
            hovertext=[f"{trade['date']}｜{action}｜價格 {trade['price']:.2f}" for trade in points],
            hoverinfo="text",
        ))
    price_chart = pio.to_html(price_fig, full_html=False, include_plotlyjs="cdn")

    strategy_fig = go.Figure()
    strategy_fig.add_trace(go.Scatter(x=prices.index, y=prices["strategy_equity"],
                                      name="均線策略"))
    strategy_fig.add_trace(go.Scatter(x=prices.index, y=prices["buy_hold_equity"],
                                      name="買入持有"))
    equity_by_date = {
        index.strftime("%Y-%m-%d"): value
        for index, value in prices["strategy_equity"].items()
    }
    for action, color, marker_symbol in [("買入", "green", "triangle-up"), ("賣出", "red", "triangle-down")]:
        points = [trade for trade in trade_log if trade["action"] == action]
        strategy_fig.add_trace(go.Scatter(
            x=[trade["date"] for trade in points],
            y=[equity_by_date[trade["date"]] for trade in points],
            mode="markers",
            marker={"color": color, "symbol": marker_symbol, "size": 12},
            name=action,
            hovertext=[f"{trade['date']}｜{action}｜價格 {trade['price']:.2f}" for trade in points],
            hoverinfo="text",
        ))
    strategy_fig.update_layout(title="策略資產曲線（含成本）", template="plotly_white", height=460)
    strategy_chart = pio.to_html(strategy_fig, full_html=False, include_plotlyjs=False)

    model_fig = go.Figure()
    model_fig.add_trace(go.Scatter(x=model_prices.index, y=model_prices["model_equity"],
                                   name="模型策略"))
    model_fig.add_trace(go.Scatter(x=model_prices.index, y=model_prices["buy_hold_equity"],
                                   name="測試集買入持有"))
    model_fig.update_layout(title=f"{model_metrics['model_name']} 模型測試集資產曲線",
                            template="plotly_white", height=460)
    model_chart = pio.to_html(model_fig, full_html=False, include_plotlyjs=False)

    def percent(value):
        return f"{value:.2%}"

    cards = "".join([
        f"<article><span>均線策略報酬</span><strong>{percent(metrics['strategy_return'])}</strong></article>",
        f"<article><span>買入持有報酬</span><strong>{percent(metrics['buy_hold_return'])}</strong></article>",
        f"<article><span>年化波動</span><strong>{percent(metrics['annual_volatility'])}</strong></article>",
        f"<article><span>最大回撤</span><strong>{percent(metrics['max_drawdown'])}</strong></article>",
        f"<article><span>Sharpe</span><strong>{metrics['sharpe']:.2f}</strong></article>",
        f"<article><span>交易成本</span><strong>{percent(metrics['total_cost'])}</strong></article>",
    ])
    trade_rows_html = "".join(
        f"<tr><td>{trade['date']}</td><td class=\"{trade['action']}\">"
        f"{trade['action']}</td><td>{trade['price']:.2f}</td></tr>"
        for trade in trade_log
    )
    refresh_note = (f"<div class=\"warning\">最新資料更新失敗，以下顯示上次快取資料：{refresh_error}</div>"
                    if refresh_error else
                    "<div class=\"success\">已在載入 Dashboard 時更新最近交易資料。</div>")
    return HTMLResponse(f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>TW Stocks Dashboard</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#17202a;background:#f5f7f9}}h1{{margin-bottom:4px}}header{{border-left:5px solid #0b6e69;padding-left:16px}}form,.panel{{background:white;padding:18px;border:1px solid #dfe5ea;margin-top:18px}}form{{display:flex;gap:12px;align-items:end;flex-wrap:wrap}}label{{display:grid;gap:6px;font-size:14px}}input,select{{padding:9px;border:1px solid #bac5ce;border-radius:4px;width:100px}}button{{padding:10px 18px;background:#0b6e69;color:white;border:0;border-radius:4px;cursor:pointer}}.tabs,nav{{display:flex;gap:6px;margin-top:20px;flex-wrap:wrap}}nav a{{padding:8px 12px;background:white;color:#0b6e69;text-decoration:none;border:1px solid #dfe5ea;border-radius:4px}}.tab{{background:#dfe5ea;color:#17202a}}.tab.active{{background:#0b6e69;color:white}}.content{{display:none}}.content.active{{display:block}}section{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:18px 0}}article{{background:white;padding:15px;border-top:3px solid #0b6e69}}article span{{display:block;color:#66737d;font-size:13px}}strong{{display:block;font-size:22px;margin-top:8px}}.chart{{background:white;padding:8px;border:1px solid #dfe5ea;margin-top:18px}}.trade-table{{border-collapse:collapse;width:100%;max-width:520px}}.trade-table td,.trade-table th{{padding:8px;text-align:left;border-bottom:1px solid #dfe5ea}}.買入{{color:#16834b;font-weight:bold}}.賣出{{color:#c0392b;font-weight:bold}}small,.hint{{color:#66737d}}.warning,.success{{padding:12px;margin-top:14px}}.warning{{background:#fff4d6;border-left:4px solid #d99000}}.success{{background:#e4f6ed;border-left:4px solid #16834b}}@media(max-width:800px){{section{{grid-template-columns:repeat(2,1fr)}}}}</style></head>
<body><header><h1>{symbol} 量化研究室</h1><small>{metrics['start']} 至 {metrics['end']}｜資料筆數 {metrics['observations']}</small></header><nav><a href="#overview" onclick="showTab('overview',this)">研究總覽</a><a href="#candles" onclick="showTab('candles',this)">一般 K 線</a><a href="#strategy" onclick="showTab('strategy',this)">策略回測</a><a href="#model" onclick="showTab('model',this)">機器學習</a><a href="/portfolio?symbols={symbol},006208.TW">投資組合</a><a href="/risk-dashboard?symbol={symbol}">風險管理</a><a href="/docs">API 文件</a></nav>{refresh_note}
<form method="get"><label>股票代號<input name="symbol" value="{symbol}"></label><label>短均線<input type="number" name="fast_window" value="{fast_window}" min="2"></label><label>長均線<input type="number" name="slow_window" value="{slow_window}" min="3"></label><label>手續費率<input type="number" name="fee_rate" value="{fee_rate}" step="0.000001" min="0"></label><label>滑價率<input type="number" name="slippage_rate" value="{slippage_rate}" step="0.000001" min="0"></label><label>機器學習模型<select name="model_name"><option value="logistic" {'selected' if model_name == 'logistic' else ''}>Logistic Regression</option><option value="random_forest" {'selected' if model_name == 'random_forest' else ''}>Random Forest</option><option value="gradient_boosting" {'selected' if model_name == 'gradient_boosting' else ''}>HistGradientBoosting</option></select></label><button type="submit">更新研究</button></form>
<div class="tabs"><button class="tab active" onclick="showTab('overview',this)">研究總覽</button><button class="tab" onclick="showTab('candles',this)">一般 K 線</button><button class="tab" onclick="showTab('strategy',this)">策略回測</button><button class="tab" onclick="showTab('model',this)">機器學習</button></div>
<div id="overview" class="content active"><div class="panel"><h2>先看懂這些數字</h2><p><b>報酬</b>：這段歷史期間資產增加多少。<br><b>波動</b>：報酬上下震盪的程度，越高代表風險通常越高。<br><b>最大回撤</b>：從歷史高點跌到低點最深的一次。<br><b>Sharpe</b>：每承擔一單位波動換來多少報酬，僅供比較。</p><div class="warning">這是歷史回測教學工具，不是買賣建議。過去績效不保證未來結果。</div></div><section>{cards}</section></div>
<div id="candles" class="content"><div class="chart">{price_chart}</div></div>
<div id="strategy" class="content"><div class="panel"><h2>均線交叉策略</h2><p class="hint">短均線高於長均線時，下一個交易日持有；已扣除手續費與滑價。綠色三角形是買入，紅色三角形是賣出。</p><h3>實際交易紀錄</h3><table class="trade-table"><tr><th>日期</th><th>動作</th><th>價格</th></tr>{trade_rows_html}</table></div><div class="chart">{strategy_chart}</div></div>
<div id="model" class="content"><div class="panel"><h2>{model_metrics['model_name']} 方向模型</h2><p class="hint">模型使用 5 日動能、均線比例、20 日波動與成交量變化，前 70% 訓練、後 30% 測試。用上方下拉選單切換模型。</p><section><article><span>測試準確率</span><strong>{percent(model_metrics['accuracy'])}</strong></article><article><span>基準準確率</span><strong>{percent(model_metrics['baseline_accuracy'])}</strong></article><article><span>模型報酬</span><strong>{percent(model_metrics['model_return'])}</strong></article><article><span>測試集買入持有</span><strong>{percent(model_metrics['buy_hold_return'])}</strong></article><article><span>模型最大回撤</span><strong>{percent(model_metrics['model_max_drawdown'])}</strong></article><article><span>模型 Sharpe</span><strong>{model_metrics['model_sharpe']:.2f}</strong></article></section><p><b>最新訊號（{model_metrics['latest_date']}）</b>：{model_metrics['latest_signal']}，預測上漲機率 {percent(model_metrics['latest_probability'])}。<br><b>Precision</b>：{percent(model_metrics['precision'])}　<b>Recall</b>：{percent(model_metrics['recall'])}<br><span class="hint">Precision 是預測上漲時真的上漲的比例；Recall 是所有上漲日中被抓到的比例。</span></p><div class="warning">模型是研究實驗，不是買賣訊號。請比較測試集報酬、回撤與交易成本，不要只看準確率。</div></div><div class="chart">{model_chart}</div></div>
<script>function showTab(id,button){{document.querySelectorAll('.content').forEach(function(item){{item.classList.remove('active')}});document.querySelectorAll('.tab').forEach(function(item){{item.classList.remove('active')}});document.getElementById(id).classList.add('active');button.classList.add('active');window.dispatchEvent(new Event('resize'));}}</script></body></html>""")
