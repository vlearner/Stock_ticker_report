from unittest.mock import AsyncMock, patch

import pytest

from mcp_etf_holdings.tools import get_etf_holdings, get_etfs_holding, get_top_etf_exposure

_MOCK_HOLDINGS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "weight_pct": 7.2},
    {"symbol": "MSFT", "name": "Microsoft Corp.", "weight_pct": 6.8},
    {"symbol": "NVDA", "name": "NVIDIA Corp.", "weight_pct": 5.1},
]


@pytest.fixture(autouse=True)
def mock_fetch(tmp_path):
    """Patch _fetch_holdings to return deterministic data without network calls."""
    async def _fake_fetch(etf_ticker: str) -> list[dict]:
        return _MOCK_HOLDINGS if etf_ticker in ("SPY", "QQQ") else []

    with patch("mcp_etf_holdings.tools._fetch_holdings", side_effect=_fake_fetch):
        yield


async def test_get_etfs_holding_found():
    results = await get_etfs_holding("AAPL")
    etfs = [r["etf"] for r in results]
    assert "SPY" in etfs
    assert "QQQ" in etfs


async def test_get_etfs_holding_not_found():
    results = await get_etfs_holding("UNKNOWN")
    assert results == []


async def test_get_etfs_holding_sorted_by_weight():
    results = await get_etfs_holding("AAPL")
    weights = [r["weight_pct"] for r in results]
    assert weights == sorted(weights, reverse=True)


async def test_get_etf_holdings():
    holdings = await get_etf_holdings("SPY")
    assert len(holdings) == 3
    assert holdings[0]["symbol"] == "AAPL"


async def test_get_top_etf_exposure_limit():
    results = await get_top_etf_exposure("MSFT")
    assert len(results) <= 5
