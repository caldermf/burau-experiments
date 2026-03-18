#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from score import score


def sample_tuple(level, rng):
    a = rng.randint(0, level - 1)
    b = rng.randint(0, level - 1 - a)
    c = level - 1 - a - b
    d = rng.randint(0, level)
    e = level - d
    return a, b, c, d, e


def make_plot(xs, ys, mean_x, mean_y, output, *, p, samples, max_level, seed):
    width = 1400
    height = 900
    left = 110
    right = 50
    top = 70
    bottom = 110
    plot_width = width - left - right
    plot_height = height - top - bottom

    image = Image.new("RGBA", (width, height), (250, 250, 248, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    max_score = max(ys) if ys else 0
    y_max = max(1, max_score)

    def x_to_px(x):
        if max_level <= 1:
            return left + plot_width // 2
        return left + int((x - 1) * plot_width / (max_level - 1))

    def y_to_px(y):
        return top + int(plot_height - y * plot_height / y_max)

    draw.rectangle((left, top, left + plot_width, top + plot_height), fill=(255, 255, 255, 255))

    for i in range(6):
        x_value = 1 + i * (max_level - 1) / 5 if max_level > 1 else 1
        x_px = x_to_px(x_value)
        draw.line((x_px, top, x_px, top + plot_height), fill=(230, 230, 230, 255), width=1)
        label = str(int(round(x_value)))
        draw.text((x_px - 8, top + plot_height + 12), label, fill=(60, 60, 60, 255), font=font)

    for i in range(6):
        y_value = i * y_max / 5
        y_px = y_to_px(y_value)
        draw.line((left, y_px, left + plot_width, y_px), fill=(230, 230, 230, 255), width=1)
        label = str(int(round(y_value)))
        draw.text((left - 30, y_px - 6), label, fill=(60, 60, 60, 255), font=font)

    draw.line((left, top + plot_height, left + plot_width, top + plot_height), fill=(50, 50, 50, 255), width=2)
    draw.line((left, top, left, top + plot_height), fill=(50, 50, 50, 255), width=2)

    for x, y in zip(xs, ys):
        px = x_to_px(x)
        py = y_to_px(y)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(31, 119, 180, 42))

    line_points = [(x_to_px(x), y_to_px(y)) for x, y in zip(mean_x, mean_y)]
    if len(line_points) >= 2:
        draw.line(line_points, fill=(214, 39, 40, 255), width=3)

    draw.text((left, 22), f"Projlen mod {p} vs d + e", fill=(20, 20, 20, 255), font=font)
    draw.text((left + plot_width // 2 - 20, height - 35), "d + e", fill=(20, 20, 20, 255), font=font)
    draw.text((18, top - 8), f"score(tuple, {p})", fill=(20, 20, 20, 255), font=font)

    info = (
        f"{samples} random valid tuples\n"
        f"max level = {max_level}\n"
        f"seed = {seed}\n"
        "blue = samples, red = mean"
    )
    info_box = (width - 230, 20, width - 20, 92)
    draw.rounded_rectangle(info_box, radius=8, fill=(255, 255, 255, 235), outline=(200, 200, 200, 255))
    draw.multiline_text((info_box[0] + 10, info_box[1] + 8), info, fill=(20, 20, 20, 255), font=font, spacing=3)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--max-level", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260313)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("score_vs_d_plus_e_mod3.png"),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    xs = []
    ys = []
    by_level = defaultdict(list)

    for _ in range(args.samples):
        level = rng.randint(1, args.max_level)
        tup = sample_tuple(level, rng)
        value = score(tup, args.p)
        xs.append(level)
        ys.append(value)
        by_level[level].append(value)

    mean_x = sorted(by_level)
    mean_y = [sum(by_level[level]) / len(by_level[level]) for level in mean_x]

    make_plot(
        xs,
        ys,
        mean_x,
        mean_y,
        args.output,
        p=args.p,
        samples=args.samples,
        max_level=args.max_level,
        seed=args.seed,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
