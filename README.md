# Stock Ticker Report

A conversational financial analysis bot that answers stock queries via **Telegram**. Send a message like *"How is AAPL doing?"* and the bot fetches real-time market data and news, runs multi-agent analysis, and replies with a clean, structured report.

---

## How It Works

```
Telegram message
      │
      ▼
 Orchestrator  ──── classifies intent, extracts tickers, routes
      │
      ├──▶  Data Fetcher  ──── yfinance (fundamentals, SMAs, volume)
      │
      ├──▶  News Fetcher  ──── Brave Search (headlines + sentiment)
      │
      ▼
   Analyst  ──── generates plain-language summary (Groq 70B LLM)
      │
      ▼
   Critic   ──── validates output (grounded, concise, no advice)
      │
      ▼
  Formatter ──── packages Markdown message
      │
      ▼
Telegram reply
```

**Key design principles**
- Async-first — no blocking calls; yfinance offloaded via `asyncio.to_thread`
- All tools return Pydantic models — never raw dicts
- Graceful degradation — partial data (e.g. Brave rate-limited) still produces output
- Reflection loop — Analyst → Critic cycles up to `max_critic_iterations` times
- SQLite persistence — LangGraph checkpointer, rate-limiting, user preferences

---

## Tech Stack

| Layer | Library |
|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | [Groq](https://console.groq.com) (`llama-3.1-8b-instant` / `llama-3.3-70b-versatile`) |
| Financial data | [yfinance](https://github.com/ranaroussi/yfinance) |
| News search | [Brave Search API](https://brave.com/search/api/) |
| REST API | [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org) |
| Messaging | [python-telegram-bot](https://python-telegram-bot.org) |
| Data validation | [Pydantic v2](https://docs.pydantic.dev) |
| Observability | [LangSmith](https://smith.langchain.com) |
| Persistence | SQLite via [aiosqlite](https://github.com/omnilib/aiosqlite) |

---


## API Keys

You will need accounts and API keys from the following services:

| Service | Purpose | Get your key |
|---|---|---|
| **Groq** | LLM inference (fast + free tier) | https://console.groq.com/keys |
| **Brave Search** | News & web search | https://brave.com/search/api/ |
| **Telegram** | Bot token via BotFather | https://t.me/BotFather |
| **LangSmith** *(optional)* | Tracing & observability | https://smith.langchain.com |

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd Stock_ticker_report
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example below into a `.env` file at the project root:

```dotenv
# Required
GROQ_API_KEY=your_groq_api_key
BRAVE_API_KEY=your_brave_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# Optional — LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=stock-ticker-report

# Optional — overrides for defaults
GROQ_FAST_MODEL=llama-3.1-8b-instant
GROQ_ANALYSIS_MODEL=llama-3.3-70b-versatile
SQLITE_DB_PATH=./data/state.db
YFINANCE_TIMEOUT_S=8.0
YFINANCE_RETRIES=1
RATE_LIMIT_PER_HOUR=10
MAX_CRITIC_ITERATIONS=2
PIPELINE_WARNING_THRESHOLD_S=8.0
```

### 4. Run the REST API (local testing)

The API lets you test each tool independently — no Telegram setup required.
Only the keys relevant to the routes you call need to be in `.env`.

```bash
python -m src.api.app
```

| URL | Description |
|---|---|
| `http://localhost:8000/docs` | Swagger UI — interactive browser testing |
| `http://localhost:8000/redoc` | ReDoc — clean API reference |
| `GET /api/v1/news?ticker=AAPL` | Fetch recent Brave Search news (requires `BRAVE_API_KEY`) |
| `GET /api/v1/news?ticker=TSLA&count=10` | Up to 20 results |

**Test with curl:**
```bash
curl "http://localhost:8000/api/v1/news?ticker=AAPL&count=5" | python3 -m json.tool
```

**Minimum `.env` for the news route:**
```dotenv
BRAVE_API_KEY=your_brave_api_key
```

### 5. Run the Telegram bot (full pipeline)

```bash
python -m src.main
```

---

## Project Structure

```
src/
├── config/
│   ├── __init__.py      # Full Settings (all keys) — backward-compat
│   └── brave.py         # BraveSettings — only needs BRAVE_API_KEY
├── api/
│   ├── app.py           # FastAPI app + uvicorn entry point
│   └── routes/
│       └── news.py      # GET /api/v1/news
├── tools/
│   ├── brave_tools.py   # fetch_ticker_news() + LangChain @tool wrapper
│   └── yfinance_tools.py
├── agents/              # LangGraph agent stubs
├── graph/               # Pipeline definition
├── schemas/             # Pydantic data contracts
└── main.py              # Telegram bot entry point
```

---

## Running Tests

```bash
# All tests
.venv/bin/pytest tests/ -v

# Single file
.venv/bin/pytest tests/unit/test_yfinance_tools.py -v

# Stop on first failure
.venv/bin/pytest tests/ -x
```

