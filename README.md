TW Stocks Mini App
=================

Quick demo FastAPI app to fetch Taiwan stock data (via yfinance), store to SQLite, and show simple Plotly visualization.

Run locally

1. Create a virtualenv and install:

```bash
cd ~/tw-stocks
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start the app:

```bash
uvicorn app.main:app --reload
```

Endpoints

- `POST /collect?symbol=2330.TW&period=1y&interval=1d` — fetch & store historical data
- `GET /data?symbol=2330.TW&limit=100` — return JSON rows
- `GET /plot?symbol=2330.TW` — HTML Plotly candlestick

Notes

- Database file `stocks.db` will be created in the project root when collecting data.
- Add `.venv` and `stocks.db` to `.gitignore` before pushing to GitHub.
