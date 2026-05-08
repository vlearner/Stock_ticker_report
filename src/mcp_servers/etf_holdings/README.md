# mcp-etf-holdings

> Find which ETFs hold any stock — an open source MCP server for ETF reverse lookup.

[![CI](https://github.com/vlearner/mcp-etf-holdings/actions/workflows/ci.yml/badge.svg)](https://github.com/vlearner/mcp-etf-holdings/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mcp-etf-holdings)](https://pypi.org/project/mcp-etf-holdings/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Most ETF MCP servers go **forward**: give them an ETF and they list its holdings.  
This one goes **backward**: give it any stock ticker and it tells you which ETFs hold it — and at what weight.

---

## Tools

| Tool | Description |
|------|-------------|
| `etfs_holding(ticker)` | All ETFs in the universe that hold this stock, sorted by weight % |
| `etf_holdings(etf_ticker)` | Full top-holdings list for a given ETF |
| `top_etf_exposure(ticker)` | Top 5 ETFs by portfolio weight for a stock |

### Example responses

```jsonc
// etfs_holding("NVDA")
[
  { "etf": "SMH",  "holding_name": "NVIDIA Corp.", "weight_pct": 20.31 },
  { "etf": "SOXX", "holding_name": "NVIDIA Corp.", "weight_pct": 9.47  },
  { "etf": "QQQ",  "holding_name": "NVIDIA Corp.", "weight_pct": 8.12  },
  ...
]

// top_etf_exposure("AAPL")
[
  { "etf": "VGT",  "holding_name": "Apple Inc.", "weight_pct": 18.4 },
  { "etf": "QQQ",  "holding_name": "Apple Inc.", "weight_pct": 10.9 },
  ...
]
```

---

## Installation

```bash
pip install mcp-etf-holdings
```

No API keys required — data comes from Yahoo Finance via `yfinance`.

---

## Usage

### With Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "etf-holdings": {
      "command": "mcp-etf-holdings"
    }
  }
}
```

### With Claude Code (CLI)

```bash
claude mcp add etf-holdings -- mcp-etf-holdings
```

### Standalone (stdio)

```bash
mcp-etf-holdings
```

### As a Python library

```python
import asyncio
from mcp_etf_holdings.tools import get_etfs_holding, get_top_etf_exposure

async def main():
    print(await get_etfs_holding("NVDA"))
    print(await get_top_etf_exposure("AAPL"))

asyncio.run(main())
```

---

## ETF Universe

The server scans ~100 major ETFs across:

- **US Broad Market** — SPY, VOO, VTI, IVV, SCHB, ...
- **Growth / Value** — QQQ, IWF, VUG, SCHG, IVW, ...
- **Small / Mid Cap** — IWM, IJH, IJR, VO, VB, ...
- **Sectors** — XLK, XLF, XLE, VGT, SMH, SOXX, ...
- **International** — VEA, EFA, IEFA, EEM, VWO, MCHI, ...
- **Dividend** — VIG, SCHD, VYM, DGRO, HDV, ...
- **Thematic** — ARKK, BOTZ, ICLN, LIT, ROBO, ...

Want to add an ETF? See [CONTRIBUTING.md](CONTRIBUTING.md).

> **Note:** yfinance returns top holdings only (typically 10–25 per ETF). Stocks with very small weights may not appear.

---

## Caching

Holdings are cached in SQLite at `~/.mcp_etf_holdings/cache.db` with a 24-hour TTL. ETF compositions change infrequently, so this keeps lookups fast without hammering Yahoo Finance.

---

## Development

```bash
git clone https://github.com/vlearner/mcp-etf-holdings
cd mcp-etf-holdings
pip install -e ".[dev]"
pytest
ruff check src tests
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome — especially for expanding the ETF universe or adding new tools (e.g. sector exposure, overlap analysis).

---

## License

[MIT](LICENSE) — free to use in personal and commercial projects.
