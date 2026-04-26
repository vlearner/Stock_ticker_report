# Stock Ticker Report — Project Context

## What This Project Is
A Telegram bot + FastAPI web demo for conversational stock analysis. Users send messages like "How is AAPL doing?" and receive structured financial reports.

**Entry point:** `python -m src.main` (bot + API together)
**Branch for active development:** `claude/custom-mcp-server-rWcZL`

## Tech Stack
- **Agent framework:** LangGraph 1.0.10 + LangChain 0.3
- **LLM:** Groq (llama-3.1-8b-instant for orchestrator, llama-3.3-70b-versatile for analyst)
- **Financial data:** yfinance 0.2
- **News:** Brave Search API (httpx async)
- **API:** FastAPI 0.115 + Uvicorn
- **Messaging:** python-telegram-bot 21.6
- **Persistence:** SQLite via aiosqlite (rate limiting + analyst cache)
- **Validation:** Pydantic v2

## Pipeline Flow
Telegram msg → Orchestrator (intent classification) → DataFetcher (yfinance) → NewsFetcher (Brave) → Analyst (Groq LLM) → Critic (validation loop) → Formatter → Telegram reply

## Active Plan: Custom MCP Server for ETF Holdings

### Decision Made (2026-04-26)
We decided to build a **custom open source MCP server** for ETF reverse lookup. No existing open source MCP fills this gap — all existing servers do forward lookup (ETF → holdings) but none do **reverse lookup (stock ticker → which ETFs hold it)**.

### New Repo to Create: `vlearner/mcp-etf-holdings`
A standalone, reusable Python MCP server (stdio transport) that anyone can install.

**Planned structure:**
```
mcp-etf-holdings/
  src/
    mcp_etf_holdings/
      server.py          ← MCP server entry point (stdio transport)
      tools.py           ← ETF reverse lookup logic
      etf_universe.py    ← pre-seeded list of ~100 major ETFs
      cache.py           ← SQLite cache (TTL: 24h)
  pyproject.toml
  README.md
```

**Tools to expose:**
- `get_etfs_holding(ticker)` — list of ETFs containing that stock + weight %
- `get_etf_holdings(etf_ticker)` — full holdings of a given ETF
- `get_top_etf_exposure(ticker)` — top 5 ETFs by weight for a stock

**Data source:** yfinance (already in project stack, no new API keys needed)
**Caching:** SQLite, 24h TTL

### Integration into This Project
After the MCP server is built, wire it into `src/agents/data_fetcher.py` so that ETF exposure data is fetched alongside fundamentals and included in the analyst's context.

### Pending Decision
The user needs to **manually create the `vlearner/mcp-etf-holdings` repo on GitHub** (GitHub MCP tools in this session are scoped to `vlearner/stock_ticker_report` only). Once the repo exists, Claude can build and push everything.

Alternatively, the MCP server can be scaffolded under `src/mcp_servers/etf_holdings/` in this repo first, then extracted later.

## Key Files
| File | Purpose |
|------|---------|
| `src/main.py` | Entry point — runs bot + API |
| `src/graph/pipeline.py` | LangGraph state machine (6 nodes) |
| `src/agents/data_fetcher.py` | yfinance calls — ETF MCP wires in here |
| `src/agents/analyst.py` | Groq LLM analysis |
| `src/tools/yfinance_tools.py` | Async yfinance wrappers |
| `src/schemas/agent_state.py` | LangGraph state TypedDict |
| `src/schemas/ticker_data.py` | Pydantic data models |
| `src/cache/analyst_cache.py` | SQLite daily cache for analyst output |

## GitHub Repo
`vlearner/stock_ticker_report`
