#!/usr/bin/env python3
"""Generate PWA PNG icons using only Python stdlib.

Produces: public/icons/icon-192.png, icon-512.png, icon-180.png
Design: dark (#0f1115) background, blue (#4c8bff) stock-chart line with glow + fill.
"""
import math
import os
import struct
import zlib

BG   = (15,  17,  21)    # #0f1115
BLUE = (76, 139, 255)    # #4c8bff


def clamp(v: float) -> int:
    return max(0, min(255, int(v)))


def blend(dst: tuple, color: tuple, alpha: float) -> tuple:
    return tuple(clamp(dst[i] * (1.0 - alpha) + color[i] * alpha) for i in range(3))


def make_png(pixels: list, w: int, h: int) -> bytes:
    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    scanlines = b"".join(
        b"\x00" + bytes(v for px in pixels[y * w : (y + 1) * w] for v in px)
        for y in range(h)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines, 6))
        + chunk(b"IEND", b"")
    )


def generate(size: int) -> bytes:
    img = [BG] * (size * size)

    def put(x: float, y: float, color: tuple, alpha: float = 1.0) -> None:
        xi, yi = int(x), int(y)
        if 0 <= xi < size and 0 <= yi < size:
            i = yi * size + xi
            img[i] = blend(img[i], color, alpha)

    pad = size * 0.11
    cw  = size - 2 * pad
    ch  = size - 2 * pad

    # Normalized waypoints: (x [0,1], y [0,1]) where y=0 is top
    pts = [
        (0.00, 0.78), (0.12, 0.66), (0.22, 0.83), (0.35, 0.55),
        (0.45, 0.63), (0.55, 0.40), (0.65, 0.50), (0.75, 0.27),
        (0.87, 0.32), (1.00, 0.12),
    ]
    wpx = [(pad + p[0] * cw, pad + p[1] * ch) for p in pts]

    # ── Gradient fill under the line ─────────────────────────────────────────
    def line_y_at(x: float):
        for i in range(len(wpx) - 1):
            x0, y0 = wpx[i]; x1, y1 = wpx[i + 1]
            if x0 <= x <= x1 + 0.5:
                t = (x - x0) / max(0.01, x1 - x0)
                return y0 + t * (y1 - y0)
        return None

    floor_y = size - pad
    for x in range(size):
        ly = line_y_at(x)
        if ly is None:
            continue
        for y in range(int(ly), int(floor_y) + 1):
            t = (y - ly) / max(1, floor_y - ly)
            put(x, y, BLUE, 0.18 * (1.0 - t) ** 2)

    # ── Draw line segments ────────────────────────────────────────────────────
    def draw_seg(x0, y0, x1, y1, color, thick, alpha=1.0):
        dx = x1 - x0; dy = y1 - y0
        length = math.hypot(dx, dy)
        if length < 0.01:
            return
        nx = -dy / length; ny = dx / length
        steps = int(length * 2) + 1
        r = thick / 2.0
        for s in range(steps):
            t = s / max(1, steps - 1)
            cx = x0 + t * dx; cy = y0 + t * dy
            for d in range(int(-r - 1), int(r + 2)):
                dist = abs(d)
                if dist < r:
                    a = alpha
                elif dist < r + 1.0:
                    a = alpha * (r + 1.0 - dist)
                else:
                    continue
                put(cx + d * nx, cy + d * ny, color, a)

    thick = max(1.5, size * 0.016)

    # Glow passes (wide, low alpha)
    for i in range(len(wpx) - 1):
        x0, y0 = wpx[i]; x1, y1 = wpx[i + 1]
        draw_seg(x0, y0, x1, y1, BLUE, thick * 4.0, 0.08)
        draw_seg(x0, y0, x1, y1, BLUE, thick * 2.5, 0.18)

    # Crisp main line
    for i in range(len(wpx) - 1):
        x0, y0 = wpx[i]; x1, y1 = wpx[i + 1]
        draw_seg(x0, y0, x1, y1, BLUE, thick, 1.0)

    return make_png(img, size, size)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "public", "icons")
    os.makedirs(out_dir, exist_ok=True)
    for size, name in [(192, "icon-192.png"), (512, "icon-512.png"), (180, "icon-180.png")]:
        print(f"  Generating {name} ({size}×{size})…", end=" ", flush=True)
        data = generate(size)
        path = os.path.join(out_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        print(f"{len(data):,} bytes")
    print("Done.")


if __name__ == "__main__":
    main()
