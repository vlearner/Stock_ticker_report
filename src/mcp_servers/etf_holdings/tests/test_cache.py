import pytest

from mcp_etf_holdings.cache import ETFCache


@pytest.fixture
async def cache(tmp_path):
    c = ETFCache(db_path=str(tmp_path / "test.db"))
    await c.init()
    return c


async def test_set_and_get(cache):
    await cache.set("key1", [{"etf": "SPY", "weight_pct": 6.5}])
    result = await cache.get("key1")
    assert result == [{"etf": "SPY", "weight_pct": 6.5}]


async def test_miss_returns_none(cache):
    assert await cache.get("nonexistent") is None


async def test_expired_returns_none(cache):
    await cache.set("key2", [{"etf": "QQQ"}], ttl=-1)
    assert await cache.get("key2") is None


async def test_purge_expired(cache):
    await cache.set("old", [{"etf": "SPY"}], ttl=-1)
    await cache.set("fresh", [{"etf": "QQQ"}], ttl=3600)
    await cache.purge_expired()
    assert await cache.get("old") is None
    assert await cache.get("fresh") is not None
