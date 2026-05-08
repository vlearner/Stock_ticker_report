from mcp.server.fastmcp import FastMCP

from .tools import get_etf_holdings, get_etfs_holding, get_top_etf_exposure

mcp = FastMCP(
    "mcp-etf-holdings",
    version="0.1.0",
    description="ETF holdings reverse lookup — find which ETFs hold any stock ticker",
)


@mcp.tool()
async def etfs_holding(ticker: str) -> list[dict]:
    """
    Return all ETFs in the universe that hold the given stock ticker.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL" or "NVDA".

    Returns:
        List of {etf, holding_name, weight_pct} dicts sorted by weight descending.
    """
    return await get_etfs_holding(ticker)


@mcp.tool()
async def etf_holdings(etf_ticker: str) -> list[dict]:
    """
    Return the top holdings of a given ETF.

    Args:
        etf_ticker: ETF ticker symbol, e.g. "SPY" or "QQQ".

    Returns:
        List of {symbol, name, weight_pct} dicts for each holding.
    """
    return await get_etf_holdings(etf_ticker)


@mcp.tool()
async def top_etf_exposure(ticker: str) -> list[dict]:
    """
    Return the top 5 ETFs by portfolio weight for a given stock.

    Args:
        ticker: Stock ticker symbol, e.g. "TSLA".

    Returns:
        Up to 5 {etf, holding_name, weight_pct} dicts sorted by weight descending.
    """
    return await get_top_etf_exposure(ticker)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
