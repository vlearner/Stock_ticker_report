# Contributing to mcp-etf-holdings

Thank you for your interest! Contributions are welcome.

## Getting started

```bash
git clone https://github.com/vlearner/mcp-etf-holdings
cd mcp-etf-holdings
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Development workflow

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`
2. Make your changes
3. Run checks: `ruff check src tests && pytest`
4. Open a pull request — fill in the PR template

## Adding ETFs to the universe

Edit `src/mcp_etf_holdings/etf_universe.py` and add the ticker to the appropriate section. Please group by category and include a comment if the ETF is niche.

## Reporting bugs

Open a GitHub issue using the **Bug report** template.

## Code style

- `ruff` for linting and import sorting
- Type hints on all public functions
- No comments unless the reason is non-obvious
