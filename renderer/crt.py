"""Render terminal.txt as a CRT-style PNG for the profile README.

usage: python renderer/crt.py terminal.txt crt.png

terminal.txt is plain text; wrap text in <b>...</b> to draw it bright and bold.
"""
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# GitHub profile README content width at desktop sizes (>= 1280px viewport):
# 1280 container - 2*32 padding = 1216; minus 296 sidebar + 24 gap grid = 920 main;
# minus 2*1 border + 2*24 Box padding = 870.
DISPLAY_WIDTH = 870
DPR = 2                        # pixel density of the PNG (retina)
SS = 2                         # supersampling for smooth curvature edges
WIDTH = DISPLAY_WIDTH * DPR    # final PNG width: 1740

FONTS = Path(__file__).parent / "fonts"
FONT = str(FONTS / "DejaVuSansMono.ttf")
FONT_BOLD = str(FONTS / "DejaVuSansMono-Bold.ttf")
PAD_X = 0.07                   # horizontal padding, fraction of width
PAD_Y = 0.07
LINE = 1.35
BG = (8, 14, 10)
FG = (80, 255, 140)            # phosphor green
FG_BRIGHT = (190, 255, 210)

AVATAR = Path(__file__).parent / "avatar.png"
AVATAR_GRID = 24               # avatar is redrawn as AVATAR_GRID x AVATAR_GRID pixels
AVATAR_LEVELS = 4              # brightness steps
AVATAR_LINES = 5               # avatar height, in text lines
AVATAR_OFFSET = (2, -0.6)      # nudge from the top-right text corner, in (columns, lines)


def parse(src):
    """Text with <b> markup -> list of lines, each a list of (text, bold) runs."""
    lines = []
    for raw in src.rstrip("\n").split("\n"):
        runs, bold = [], False
        for part in re.split(r"(</?b>)", raw):
            if part == "<b>":
                bold = True
            elif part == "</b>":
                bold = False
            elif part:
                runs.append((part, bold))
        lines.append(runs)
    return lines


def render_text(lines, w):
    """Draw text on a w-wide canvas, sizing the font so the longest line fills it."""
    cols = max(sum(len(t) for t, _ in line) for line in lines)
    pad_x = int(w * PAD_X)
    probe = ImageFont.truetype(FONT, 100)
    size = int((w - 2 * pad_x) / cols / (probe.getlength("M") / 100))
    font = ImageFont.truetype(FONT, size)
    bold = ImageFont.truetype(FONT_BOLD, size)
    cw = font.getlength("M")
    lh = int(size * LINE)
    pad_y = int(w * PAD_Y)
    h = len(lines) * lh + 2 * pad_y
    x0 = (w - cols * cw) / 2
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    for i, runs in enumerate(lines):
        col = 0
        for text, b in runs:
            d.text((x0 + col * cw, pad_y + i * lh), text,
                   font=bold if b else font, fill=FG_BRIGHT if b else FG)
            col += len(text)

    # avatar in the top-right corner, right-aligned with the text column
    side = AVATAR_LINES * lh
    dx, dy = AVATAR_OFFSET
    draw_avatar(d, avatar_cells(AVATAR, AVATAR_GRID),
                x0 + cols * cw - side + dx * cw, pad_y + dy * lh, side)
    return img


def avatar_cells(path, n):
    """Avatar -> n x n array of brightness in [0, 1]; NaN where transparent.

    Cells that are mostly green (the sunglasses' lenses) get brightness 2,
    which draw_avatar renders in the bright phosphor colour.
    """
    im = Image.open(path).convert("RGB")
    hsv = np.asarray(im.convert("HSV")).astype(int)
    # the saturated blue/magenta gradient behind the head is background
    fg = ~((hsv[..., 0] >= 150) & (hsv[..., 0] <= 230) & (hsv[..., 1] > 200))
    lens = (hsv[..., 0] >= 50) & (hsv[..., 0] <= 100) & (hsv[..., 1] > 100)
    luma = np.asarray(im.convert("L"), np.float32) / 255
    h, w = luma.shape
    cells = np.full((n, n), np.nan)
    for r in range(n):
        for c in range(n):
            ys = slice(r * h // n, (r + 1) * h // n)
            xs = slice(c * w // n, (c + 1) * w // n)
            if lens[ys, xs].mean() > 0.25:
                cells[r, c] = 2
            elif fg[ys, xs].mean() > 0.5:
                cells[r, c] = luma[ys, xs][fg[ys, xs]].mean()
    return cells


def draw_avatar(d, cells, x, y, side):
    """Draw cells as chunky phosphor pixels, quantised to AVATAR_LEVELS steps."""
    n = len(cells)
    cell = side / n
    gap = cell * 0.12
    steps = AVATAR_LEVELS - 1
    for r in range(n):
        for c in range(n):
            v = cells[r, c]
            if np.isnan(v):
                continue
            if v == 2:
                color = FG_BRIGHT
            else:
                # black hair/beard keep a dim glow so the silhouette still reads
                level = 0.2 + 0.8 * round(v * steps) / steps
                color = tuple(int(b + (f - b) * level) for b, f in zip(BG, FG))
            d.rectangle((x + c * cell, y + r * cell,
                         x + (c + 1) * cell - gap, y + (r + 1) * cell - gap), fill=color)


def crt(img, scale):
    """Apply CRT effects; `scale` is output pixels per CSS pixel."""
    glow = img.filter(ImageFilter.GaussianBlur(3 * scale))
    a = np.asarray(img, np.float32) + 0.9 * np.asarray(glow, np.float32)
    h, w, _ = a.shape

    # chromatic aberration
    shift = max(1, scale)
    a[..., 0] = np.roll(a[..., 0], shift, axis=1)
    a[..., 2] = np.roll(a[..., 2], -shift, axis=1)

    # scanlines: one dark line every 2 CSS px
    period = 2 * scale
    rows = np.arange(h)
    a *= (0.72 + 0.28 * (rows % period < period / 2))[:, None, None]

    # vignette
    yy, xx = np.mgrid[0:h, 0:w]
    nx, ny = (xx / w) * 2 - 1, (yy / h) * 2 - 1
    a *= (1 - 0.35 * (nx**2 + ny**2))[..., None]

    # noise (fixed seed: same text -> byte-identical image)
    rng = np.random.default_rng(1024)
    a += rng.normal(0, 4, a.shape[:2])[..., None]

    # barrel distortion (screen curvature)
    k = 0.025
    r2 = nx**2 + ny**2
    sx = ((nx * (1 + k * r2)) + 1) / 2 * (w - 1)
    sy = ((ny * (1 + k * r2)) + 1) / 2 * (h - 1)
    inside = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    src = np.clip(a, 0, 255).astype(np.uint8)
    out = src[np.clip(sy.round().astype(int), 0, h - 1),
              np.clip(sx.round().astype(int), 0, w - 1)]

    # rounded screen corners
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), 30 * scale, fill=255)
    alpha = np.minimum(np.where(inside, 255, 0), np.asarray(mask)).astype(np.uint8)
    out[alpha == 0] = 0
    return Image.fromarray(np.dstack([out, alpha]), "RGBA")


def main(src_path, out_path):
    lines = parse(open(src_path, encoding="utf-8").read())
    big = crt(render_text(lines, WIDTH * SS), DPR * SS)
    img = big.resize((WIDTH, big.height * WIDTH // big.width), Image.LANCZOS)
    img.quantize(256, method=Image.Quantize.FASTOCTREE).save(out_path, optimize=True)
    print(f"{out_path}: {img.width}x{img.height} px, "
          f"display at {DISPLAY_WIDTH}x{img.height // DPR}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
