import asyncio
import logging

import yfinance as yf

from .cache import ETFCache
from .etf_universe import ETF_UNIVERSE

logger = logging.getLogger(__name__)

_cache = ETFCache()
_SEMAPHORE = asyncio.Semaphore(10)  # cap concurrent yfinance calls


def _parse_holdings(etf_ticker: str) -> list[dict]:
    """Blocking yfinance call — run via asyncio.to_thread."""
    try:
        ticker = yf.Ticker(etf_ticker)
        funds = ticker.funds_data
        if funds is None:
            return []
        top = funds.top_holdings
        if top is None or top.empty:
            return []
        rows = []
        for symbol, row in top.iterrows():
            pct = row.get("holdingPercent", 0.0)
            rows.append(
                {
                    "symbol": str(symbol).upper(),
                    "name": str(row.get("holdingName", "")),
                    "weight_pct": round(float(pct) * 100, 4),
                }
            )
        return rows
    except Exception as exc:
        logger.debug("Failed to fetch holdings for %s: %s", etf_ticker, exc)
        return []


async def _fetch_holdings(etf_ticker: str) -> list[dict]:
    cached = await _cache.get(f"h:{etf_ticker}")
    if cached is not None:
        return cached
    async with _SEMAPHORE:
        holdings = await asyncio.to_thread(_parse_holdings, etf_ticker)
    await _cache.set(f"h:{etf_ticker}", holdings)
    return holdings


async def get_etfs_holding(ticker: str) -> list[dict]:
    """Return every ETF in the universe that holds *ticker*, sorted by weight desc."""
    ticker = ticker.upper().strip()
    tasks = [_fetch_holdings(etf) for etf in ETF_UNIVERSE]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    matches: list[dict] = []
    for etf, holdings in zip(ETF_UNIVERSE, results):
        if isinstance(holdings, Exception) or not holdings:
            continue
        for h in holdings:
            if h["symbol"] == ticker:
                matches.append(
                    {
                        "etf": etf,
                        "holding_name": h["name"],
                        "weight_pct": h["weight_pct"],
                    }
                )
                break

    return sorted(matches, key=lambda x: x["weight_pct"], reverse=True)


async def get_etf_holdings(etf_ticker: str) -> list[dict]:
    """Return the full top-holdings list for a single ETF."""
    return await _fetch_holdings(etf_ticker.upper().strip())


async def get_top_etf_exposure(ticker: str) -> list[dict]:
    """Return the top 5 ETFs by portfolio weight for *ticker*."""
    matches = await get_etfs_holding(ticker)
    return matches[:5]
