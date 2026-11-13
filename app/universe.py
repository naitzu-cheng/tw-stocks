"""Stock lists scanned by the market summary page.

These are hand-picked large caps (roughly the biggest 0050 / S&P 500
constituents). Index membership changes over time: edit freely.
"""

TW_UNIVERSE = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2881.TW", "2882.TW",
    "2891.TW", "2303.TW", "3711.TW", "2412.TW", "2886.TW", "2884.TW", "1216.TW",
    "2885.TW", "2357.TW", "3231.TW", "2892.TW", "2345.TW", "2880.TW", "5880.TW",
    "2890.TW", "2883.TW", "3008.TW", "2002.TW", "1303.TW", "1301.TW", "2207.TW",
    "6669.TW", "3034.TW", "2327.TW", "4938.TW", "2395.TW", "2301.TW", "3037.TW",
    "2379.TW", "1101.TW", "2603.TW", "2912.TW", "5871.TW",
]

US_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK-B", "JPM",
    "LLY", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "JNJ", "WMT", "NFLX", "ABBV",
    "BAC", "CRM", "ORCL", "AMD", "KO", "PEP", "MRK", "CVX", "ADBE", "TMO", "LIN",
    "MCD", "CSCO",
]

MARKETS = {
    "TW": {"label": "台股", "index": "^TWII", "index_label": "加權指數",
           "universe": TW_UNIVERSE},
    "US": {"label": "美股", "index": "^GSPC", "index_label": "S&P 500",
           "universe": US_UNIVERSE},
}
