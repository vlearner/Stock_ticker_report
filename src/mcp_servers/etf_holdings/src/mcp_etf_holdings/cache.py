import json
import time
from pathlib import Path

import aiosqlite

CACHE_TTL = 86_400  # 24 hours


class ETFCache:
    def __init__(self, db_path: str = "~/.mcp_etf_holdings/cache.db") -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialised = False

    async def init(self) -> None:
        if self._initialised:
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS etf_cache (
                    key      TEXT PRIMARY KEY,
                    value    TEXT NOT NULL,
                    expires  REAL NOT NULL
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_expires ON etf_cache (expires)"
            )
            await db.commit()
        self._initialised = True

    async def get(self, key: str) -> list | None:
        await self.init()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT value FROM etf_cache WHERE key = ? AND expires > ?",
                (key, time.time()),
            ) as cursor:
                row = await cursor.fetchone()
                return json.loads(row[0]) if row else None

    async def set(self, key: str, value: list, ttl: int = CACHE_TTL) -> None:
        await self.init()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO etf_cache (key, value, expires) VALUES (?, ?, ?)",
                (key, json.dumps(value), time.time() + ttl),
            )
            await db.commit()

    async def purge_expired(self) -> None:
        await self.init()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM etf_cache WHERE expires <= ?", (time.time(),))
            await db.commit()
