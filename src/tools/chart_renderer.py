"""Unicode text chart renderer for Telegram.

Renders a price + SMA50 + SMA200 line chart as a monospace code block
that displays correctly inside Telegram without sending any image files.
"""

from __future__ import annotations

from src.schemas.ticker_data import MAChartData

_PRICE_CHAR = "▲"
_SMA50_CHAR = "━"
_SMA200_CHAR = "·"
_AXIS_V = "┤"
_AXIS_H = "─"
_CORNER = "┼"


def render_ma_chart(data: MAChartData, width: int = 38, height: int = 10) -> str:
    """Return a Telegram-ready string: header line + monospace code block."""
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

    grid = [[" "] * width for _ in range(height)]

    # Paint in ascending priority so price overwrites MAs when they overlap
    for series, char in [(s_sma200, _SMA200_CHAR), (s_sma50, _SMA50_CHAR), (s_prices, _PRICE_CHAR)]:
        for col, val in enumerate(series):
            row = to_row(val)
            if row is not None:
                grid[row][col] = char

    y_label_w = max(len(f"{y_max:.1f}"), len(f"{y_min:.1f}")) + 1

    lines: list[str] = []
    for r in range(height):
        y_val = y_max - (r / (height - 1)) * y_range
        label = f"{y_val:{y_label_w}.1f}{_AXIS_V}"
        lines.append(label + "".join(grid[r]))

    lines.append(" " * y_label_w + _CORNER + _AXIS_H * width)

    # X-axis: 3 labels at start, mid, end
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
        end_target = width - len(d_end)
        gap_end = end_target - (len(row) - len(d_start) + len(d_start))
        # simpler: pad to position
        filled = len(row)
        if end_target > filled:
            row += " " * (end_target - filled) + d_end
        lines.append(prefix + row)

    legend_parts = [f"{_PRICE_CHAR} Price"]
    if any(v is not None for v in s_sma50):
        legend_parts.append(f"{_SMA50_CHAR} SMA50")
    if any(v is not None for v in s_sma200):
        legend_parts.append(f"{_SMA200_CHAR} SMA200")

    header = f"*{data.ticker}* — {len(data.prices)}d  {'  '.join(legend_parts)}"
    return f"{header}\n```\n" + "\n".join(lines) + "\n```"
