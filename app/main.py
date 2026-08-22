from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from app.collector import fetch_and_store, get_data
import plotly.graph_objs as go
import plotly.io as pio

app = FastAPI(title="TW Stocks Mini App")


@app.get("/")
def root():
    return {"message": "TW Stocks Mini App. Use /collect, /data, /plot"}


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
