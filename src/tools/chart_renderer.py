"""Unicode text chart renderer for Telegram.

Renders a connected price + SMA50 + SMA200 line chart as a monospace
code block — no image files needed.

Character scheme:
  ─  Price (horizontal data points, │ for vertical transitions)
  ┄  SMA50 (dashed)
  ╌  SMA200 (dot-dash)
"""

from __future__ import annotations

from src.schemas.ticker_data import MAChartData

_AXIS_V = "┤"
_AXIS_H = "─"
_CORNER = "┼"


def _paint_price(grid: list[list[str]], rows: list[int | None]) -> None:
    """Price line: ─ at data points, │ for vertical transitions."""
    prev_r: int | None = None
    for col, r in enumerate(rows):
        if r is None:
            prev_r = None
            continue
        if prev_r is not None and prev_r != r:
            lo, hi = min(prev_r, r), max(prev_r, r)
            for fill_r in range(lo, hi + 1):
                if grid[fill_r][col] == " ":
                    grid[fill_r][col] = "│"
        grid[r][col] = "─"
        prev_r = r


def _paint_sma(grid: list[list[str]], rows: list[int | None], char: str) -> None:
    """SMA line: same char for data points and vertical fills."""
    prev_r: int | None = None
    for col, r in enumerate(rows):
        if r is None:
            prev_r = None
            continue
        if prev_r is not None and prev_r != r:
            lo, hi = min(prev_r, r), max(prev_r, r)
            for fill_r in range(lo, hi + 1):
                if grid[fill_r][col] == " ":
                    grid[fill_r][col] = char
        grid[r][col] = char
        prev_r = r


def render_ma_chart(data: MAChartData, width: int = 38, height: int = 10) -> str:
    """Return a Telegram-ready string: header + monospace code block."""
    if not data.prices:
        return f"*{data.ticker}*\n_No price history available._"

    n = len(data.prices)

    # Sample evenly to fit `width` columns
    if n > width:
        indices = [round(i * (n - 1) / (width - 1)) for i in range(width)]
    else:
        indices = list(range(n))
        width = n

    s_prices = [data.prices[i] for i in indices]
    s_sma50  = [data.sma50[i]  if i < len(data.sma50)  else None for i in indices]
    s_sma200 = [data.sma200[i] if i < len(data.sma200) else None for i in indices]
    s_dates  = [data.dates[i]  if i < len(data.dates)  else ""   for i in indices]

    all_vals = [v for v in s_prices + s_sma50 + s_sma200 if v is not None]
    if not all_vals:
        return f"*{data.ticker}*\n_No data to chart._"

    y_min = min(all_vals)
    y_max = max(all_vals)
    y_range = y_max - y_min or 1.0

    def to_row(val: float | None) -> int | None:
        if val is None:
            return None
        return max(0, min(height - 1, height - 1 - round((val - y_min) / y_range * (height - 1))))

    s_price_rows = [to_row(v) for v in s_prices]
    s_sma50_rows = [to_row(v) for v in s_sma50]
    s_sma200_rows = [to_row(v) for v in s_sma200]

    grid = [[" "] * width for _ in range(height)]

    # Paint lowest priority first so higher priority overwrites
    _paint_sma(grid, s_sma200_rows, "╌")
    _paint_sma(grid, s_sma50_rows, "┄")
    _paint_price(grid, s_price_rows)

    y_label_w = max(len(f"{y_max:.1f}"), len(f"{y_min:.1f}")) + 1

    lines: list[str] = []
    for r in range(height):
        y_val = y_max - (r / (height - 1)) * y_range
        label = f"{y_val:{y_label_w}.1f}{_AXIS_V}"
        lines.append(label + "".join(grid[r]))

    lines.append(" " * y_label_w + _CORNER + _AXIS_H * width)

    # X-axis: 3 date labels at start, mid, end
    if any(s_dates):
        d_start = s_dates[0]
        d_mid   = s_dates[width // 2]
        d_end   = s_dates[-1]
        prefix  = " " * (y_label_w + 1)
        row     = d_start
        mid_target = width // 2 - len(d_mid) // 2
        gap_mid = mid_target - len(d_start)
        if gap_mid > 0:
            row += " " * gap_mid + d_mid
        filled = len(row)
        end_target = width - len(d_end)
        if end_target > filled:
            row += " " * (end_target - filled) + d_end
        lines.append(prefix + row)

    legend_parts = ["─ Price"]
    if any(v is not None for v in s_sma50):
        legend_parts.append("┄ SMA50")
    if any(v is not None for v in s_sma200):
        legend_parts.append("╌ SMA200")

    header = f"*{data.ticker}* — {len(data.prices)}d  {'  '.join(legend_parts)}"
    return f"{header}\n```\n" + "\n".join(lines) + "\n```"
