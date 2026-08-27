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
- `GET /analysis?symbol=2330.TW&fast_window=20&slow_window=60` — strategy statistics
- `GET /model?symbol=2330.TW&model_name=random_forest&fast_window=20&slow_window=60` — machine-learning test statistics
- `GET /risk?symbol=2330.TW&capital=100000&risk_per_trade=0.01&stop_loss_pct=0.05` — position-size estimate
- `GET /risk-dashboard?symbol=2330.TW` — risk management UI
- `GET /portfolio?symbols=2330.TW,006208.TW` — equal-weight portfolio analysis page
- `GET /dashboard?symbol=2330.TW&fast_window=20&slow_window=60` — interactive analysis UI

Quant dashboard

Collect a symbol first, then open the dashboard in a browser:

```bash
curl -X POST "http://127.0.0.1:8000/collect?symbol=2317.TW&period=1y&interval=1d"
open "http://127.0.0.1:8000/dashboard?symbol=2317.TW"
```

The dashboard automatically fetches the latest one year of daily data when it
loads, then reads the local SQLite database. Collection removes empty rows and
updates an existing date instead of creating duplicate dates. Other API endpoints
still read the database without automatically fetching remote data.

The dashboard has four tabs: research overview, ordinary candlesticks, strategy
backtest, and machine learning. The moving-average strategy includes configurable
fees and slippage. The model tab compares Logistic Regression with Random Forest
and HistGradientBoosting to estimate next-day direction from momentum,
moving-average ratio, volatility, and volume change. Use the model dropdown in
the dashboard, or pass `model_name=logistic`, `random_forest`, or
`gradient_boosting` to the API. Data is split in time order (70% train, 30% test) to reduce
look-ahead bias. A more complex model is not automatically a better model, so
compare both models on the unseen test period and against buy-and-hold.

The model view also shows a naive baseline accuracy, precision, recall, model
volatility, maximum drawdown, Sharpe ratio, and the latest probability/signal.
An accuracy of 58% is meaningful only when it beats the baseline by a useful
margin and still produces better risk-adjusted returns after costs.

Portfolio and risk tools

Open `/portfolio?symbols=2330.TW,006208.TW` to compare an equal-weight portfolio,
including its return, volatility, maximum drawdown, Sharpe ratio, and return
correlation. Open `/risk` to estimate a position size from total capital, the
maximum percentage to risk on one trade, and a fixed stop-loss distance. These
are educational estimates, not broker orders or investment advice.

Quant beginner guide

- A return answers: how much did the portfolio gain or lose?
- Volatility describes how much daily returns moved around.
- Maximum drawdown describes the worst fall from a previous peak.
- Sharpe is a rough risk-adjusted comparison; it is not a guarantee of profit.
- Accuracy is only a classification score. Compare model return with buy-and-hold
	on the unseen test period before judging a model.

These are historical experiments, not investment advice. The model is not a
trading recommendation, and this project still needs stronger validation before
real-money use.

Notes

- Database file `stocks.db` will be created in the project root when collecting data.
- Add `.venv` and `stocks.db` to `.gitignore` before pushing to GitHub.
