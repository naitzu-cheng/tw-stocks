台股＋美股量化研究室
==================

FastAPI 網站：用 Yahoo Finance 抓台股與美股日線資料（已調整除權息），做均線回測、
參數穩健度檢查、walk-forward 機器學習、投資組合與風險管理。

📘 **分析方法教學（含過擬合與適用時機）：[docs/GUIDE.md](docs/GUIDE.md)**，網站上也可在 `/guide` 閱讀。

## 本機執行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

打開 <http://127.0.0.1:8000/dashboard?symbol=2330> 或 `?symbol=AAPL`。

測試（使用模擬資料，不需要網路）：

```bash
pytest -q
```

## 支援的代號

| 輸入 | 實際使用 |
|---|---|
| `2330`、`2330.tw` | `2330.TW`（上市） |
| `6488` | 先試 `.TW`，查無資料自動改 `6488.TWO`（上櫃） |
| `AAPL`、`nvda`、`SPY` | 美股 |
| `BRK.B` | `BRK-B` |

市場會自動決定交易成本與比較基準：台股手續費 0.1425%、賣出證交稅 0.3%（ETF 0.1%），基準 0050；
美股零佣金，基準 SPY。所有成本都可以在畫面上覆寫（例如輸入券商折扣後的手續費）。

## 頁面與 API

| 路徑 | 說明 |
|---|---|
| `GET /market?market=TW`（或 `US`） | 大盤總覽：市場狀態、市場廣度、偏多 / 偏空 / 過熱名單、規則回測（首頁） |
| `GET /market.json?market=US` | 同上（JSON） |
| `GET /dashboard?symbol=2330` | 個股研究室：總覽、K 線＋均線＋RSI、均線回測、參數穩健度、walk-forward 機器學習（點開分頁才計算） |
| `GET /portfolio?symbols=2330,0050,AAPL,SPY&weighting=inverse_vol` | 投資組合（等權重或風險平價、定期再平衡、風險貢獻） |
| `GET /risk-dashboard?symbol=2330` | ATR 停損與部位大小 |
| `GET /guide` | 分析方法教學 |
| `GET /analysis` | 均線策略績效（JSON） |
| `GET /robustness` | 均線參數穩健度與 Deflated Sharpe（JSON） |
| `GET /model?model_name=logistic&horizon=5&step=63&compare=true` | Walk-forward 模型結果（JSON） |
| `GET /risk` | 部位大小估算（JSON） |
| `POST /collect?symbol=AAPL&period=10y` | 手動抓取並快取資料 |
| `GET /data?symbol=2330&limit=100` | 快取的原始資料 |

舊網址參數（`fast_window`、`slow_window`、`fee_rate`、`slippage_rate`、`model_name`）仍然可用。

## 程式結構

```
app/
  markets.py    代號正規化、台/美股成本與基準
  collector.py  Yahoo Finance 下載、快取（6 小時內不重抓）
  features.py   技術指標特徵（全部只用當天以前的資料）
  metrics.py    Sharpe/Sortino/Calmar/回撤/Beta/Alpha、PSR、Deflated Sharpe
  analysis.py   回測引擎、均線策略、參數穩健度、walk-forward 模型
  portfolio.py  投資組合
  screener.py   大盤總覽：市場狀態、廣度、趨勢＋動能評分、規則回測
  universe.py   大盤總覽掃描的台股 / 美股名單（可自行修改）
  main.py       FastAPI 路由與圖表
  templates/    頁面
docs/GUIDE.md   教學
tests/          單元與 API 測試（含「隨機資料不應有預測力」與「真訊號應被偵測到」）
```

## 部署到 Render

`render.yaml` 已設定 Python 3.12。Push 到 GitHub 後 Render 會自動部署。

免費方案的檔案系統不是永久的，重啟後 SQLite 快取會清空（網站會自動重抓）。
若要保留資料，可建立 Render PostgreSQL，在 `requirements.txt` 加上 `psycopg2-binary`，
並設定環境變數 `DATABASE_URL`。

免費方案 CPU 很弱，勾選「比較三個模型」會明顯變慢；結果會在記憶體中快取 6 小時。
啟動時會在背景預先抓取大盤總覽的資料（設定環境變數 `PREFETCH_MARKETS=0` 可關閉）。

---

歷史回測與教學用途，不構成投資建議。
