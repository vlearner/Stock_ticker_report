"""Unicode text chart renderer for Telegram.

Uses asciichartpy for the price line (╭─╮╰╯│ connected style) and
overlays SMA50/SMA200 with distinct dashed characters on top.
"""

from __future__ import annotations

import asciichartpy

from src.schemas.ticker_data import MAChartData

_SMA50_CHAR  = "┄"
_SMA200_CHAR = "╌"


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

    # --- Price line via asciichartpy ---
    chart_str = asciichartpy.plot(
        s_prices,
        {"height": height - 1, "min": y_min, "max": y_max, "format": "{:7.1f}"},
    )
    # Split into mutable character grid rows
    raw_lines = chart_str.split("\n")
    # Pad every line to a consistent width so we can safely index into it
    axis_width = next(
        (line.index("┤") + 1 for line in raw_lines if "┤" in line), 0
    )
    total_width = axis_width + width
    grid = [list(line.ljust(total_width)) for line in raw_lines]

    # --- Overlay SMA lines ---
    # Row 0 = y_max, row (height-1) = y_min  (same mapping asciichartpy uses)
    def to_row(val: float | None) -> int | None:
        if val is None:
            return None
        return max(0, min(height - 1, round((y_max - val) / y_range * (height - 1))))

    for col_i, (v50, v200) in enumerate(zip(s_sma50, s_sma200)):
        chart_col = axis_width + col_i
        for val, char in ((v50, _SMA50_CHAR), (v200, _SMA200_CHAR)):
            r = to_row(val)
            if r is not None and r < len(grid) and chart_col < len(grid[r]):
                if grid[r][chart_col] in (" ", "\x00"):
                    grid[r][chart_col] = char

    chart_lines = ["".join(row).rstrip() for row in grid]

    # --- X-axis date labels ---
    if any(s_dates):
        d_start = s_dates[0]
        d_mid   = s_dates[width // 2]
        d_end   = s_dates[-1]
        prefix  = " " * (axis_width)
        row     = d_start
        mid_target = width // 2 - len(d_mid) // 2
        gap = mid_target - len(d_start)
        if gap > 0:
            row += " " * gap + d_mid
        end_target = width - len(d_end)
        if end_target > len(row):
            row += " " * (end_target - len(row)) + d_end
        chart_lines.append(prefix + row)

    # --- Header with legend ---
    legend_parts = ["╭─ Price"]
    if any(v is not None for v in s_sma50):
        legend_parts.append(f"{_SMA50_CHAR} SMA50")
    if any(v is not None for v in s_sma200):
        legend_parts.append(f"{_SMA200_CHAR} SMA200")

    header = f"*{data.ticker}* — {len(data.prices)}d  {'  '.join(legend_parts)}"
    return f"{header}\n```\n" + "\n".join(chart_lines) + "\n```"
