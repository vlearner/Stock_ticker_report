# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-05-08
### Added
- `etfs_holding(ticker)` — reverse ETF lookup: find all ETFs that hold a stock
- `etf_holdings(etf_ticker)` — forward lookup: list holdings of a given ETF
- `top_etf_exposure(ticker)` — top 5 ETFs by portfolio weight for a stock
- SQLite cache with 24-hour TTL
- Pre-seeded universe of ~100 major ETFs across US broad market, sectors, international, and thematic categories
- GitHub Actions CI (Python 3.10 / 3.11 / 3.12) and PyPI publish workflow
