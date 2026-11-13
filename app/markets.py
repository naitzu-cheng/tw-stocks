"""Market detection, symbol normalisation and default trading costs.

Taiwan (TWSE / TPEx) and US equities have very different cost structures, so a
backtest that uses the same cost for both is systematically wrong.
"""
from dataclasses import dataclass, asdict
import re


@dataclass(frozen=True)
class MarketProfile:
    market: str            # "TW" or "US"
    currency: str
    benchmark: str         # used for beta / alpha / relative strength
    fee_rate: float        # broker commission, charged on buy and sell
    tax_rate: float        # charged on sells only
    slippage_rate: float   # estimated price impact, each side
    lot_size: int          # shares per board lot
    label: str

    def as_dict(self):
        return asdict(self)


# 台股：手續費 0.1425%（買賣都收，可再乘券商折扣），證交稅賣出 0.3%，ETF 0.1%。
# 美股：多數券商零佣金，賣出另有極小的 SEC/FINRA 費用，這裡以 0 + 滑價近似。
_TW_STOCK = MarketProfile("TW", "TWD", "0050.TW", 0.001425, 0.003, 0.0005, 1000, "台股")
_TW_ETF = MarketProfile("TW", "TWD", "0050.TW", 0.001425, 0.001, 0.0005, 1000, "台股 ETF")
_US = MarketProfile("US", "USD", "SPY", 0.0, 0.0000278, 0.0005, 1, "美股")

_TW_SUFFIX = re.compile(r"\.(TW|TWO)$", re.IGNORECASE)


def normalize_symbol(raw: str) -> str:
    """Turn user input into a Yahoo Finance ticker.

    "2330" -> "2330.TW", "2330.tw" -> "2330.TW", "aapl" -> "AAPL",
    "brk.b" -> "BRK-B" (Yahoo uses a dash for US share classes).
    TPEx (上櫃) stocks need ".TWO"; plain numbers default to ".TW" and the
    collector retries with ".TWO" if TWSE returns nothing.
    """
    symbol = (raw or "").strip().upper()
    if not symbol:
        raise ValueError("請輸入股票代號")
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,20}", symbol):
        raise ValueError(f"股票代號格式不正確：{raw}")
    if _TW_SUFFIX.search(symbol):
        return symbol
    if re.fullmatch(r"\d{4,6}[A-Z]?", symbol):
        return f"{symbol}.TW"
    if re.fullmatch(r"[A-Z]{1,5}\.[A-Z]", symbol):
        return symbol.replace(".", "-")
    return symbol


def is_taiwan(symbol: str) -> bool:
    return bool(_TW_SUFFIX.search(symbol))


def profile_for(symbol: str) -> MarketProfile:
    if is_taiwan(symbol):
        code = symbol.split(".")[0]
        return _TW_ETF if code.startswith("00") else _TW_STOCK
    return _US


def alternate_taiwan_symbol(symbol: str):
    """2330.TW <-> 2330.TWO, used when the first exchange has no data."""
    if symbol.endswith(".TWO"):
        return symbol[:-4] + ".TW"
    if symbol.endswith(".TW"):
        return symbol + "O"
    return None
