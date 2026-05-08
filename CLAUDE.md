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

### What Was Decided
Build a **custom open source MCP server** for ETF reverse lookup — no existing open source MCP does this. All existing servers do forward lookup (ETF → holdings); none do **reverse lookup (stock ticker → which ETFs hold it)**.

**Repo name chosen:** `mcp-etf-holdings`
**GitHub URL:** `https://github.com/vlearner/mcp-etf-holdings`
**License:** MIT

### Current Status (as of 2026-05-08)
The full MCP server has been **scaffolded and committed locally** at:

```
src/mcp_servers/etf_holdings/    ← lives here in this repo temporarily
```

This directory is structured as a **standalone repo** ready to push to `vlearner/mcp-etf-holdings`. It contains:

```
.github/
  workflows/ci.yml               ← GitHub Actions: pytest on Python 3.10/3.11/3.12
  workflows/publish.yml          ← Auto-publish to PyPI on git tag
  ISSUE_TEMPLATE/bug_report.md
  ISSUE_TEMPLATE/feature_request.md
  PULL_REQUEST_TEMPLATE.md
.gitignore
CHANGELOG.md
CODE_OF_CONDUCT.md
CONTRIBUTING.md
LICENSE                          ← MIT
README.md
pyproject.toml                   ← pip-installable, PyPI-ready
src/mcp_etf_holdings/
  __init__.py
  server.py                      ← FastMCP server exposing 3 tools
  tools.py                       ← reverse/forward ETF lookup via yfinance
  etf_universe.py                ← ~100 major ETFs pre-seeded
  cache.py                       ← SQLite cache, 24h TTL
tests/
  test_cache.py
  test_tools.py
```

**Three MCP tools exposed:**
- `etfs_holding(ticker)` — all ETFs holding a stock + weight %, sorted desc
- `etf_holdings(etf_ticker)` — full top-holdings list for a given ETF
- `top_etf_exposure(ticker)` — top 5 ETFs by weight for a stock

### Immediate Next Step: Push to `vlearner/mcp-etf-holdings`

The `src/mcp_servers/etf_holdings/` directory already has a git repo initialized (`git init` was run, initial commit made on `main` branch). The git proxy in this session is scoped only to `vlearner/stock_ticker_report` and cannot push to the new repo.

**To push, run these 3 commands manually or in a new session with the right credentials:**
```bash
cd /home/user/Stock_ticker_report/src/mcp_servers/etf_holdings
git remote set-url origin https://github.com/vlearner/mcp-etf-holdings
git push -u origin main
```

### After Pushing: Wire MCP into This Project
Once `vlearner/mcp-etf-holdings` is live, wire the MCP client into `src/agents/data_fetcher.py` so ETF exposure data is fetched alongside fundamentals and included in the analyst's context.

The integration point is the `run` method in `src/agents/data_fetcher.py` — after the 3 concurrent yfinance calls, add a 4th call to the MCP server for `top_etf_exposure(ticker)` and attach the result to the `TickerData` schema.

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
| `src/mcp_servers/etf_holdings/` | Standalone MCP server (push to vlearner/mcp-etf-holdings) |

## GitHub Repos
- Main project: `vlearner/stock_ticker_report`
- ETF MCP server: `vlearner/mcp-etf-holdings` (https://github.com/vlearner/mcp-etf-holdings)
