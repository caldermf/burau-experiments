#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pingpong_mlp


def rotation_matrix(yaw: float, pitch: float) -> np.ndarray:
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)

    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return rot_x @ rot_y


def project_points(points: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    rotated = points @ rotation_matrix(yaw, pitch).T
    depth = rotated[:, 2]
    scale = 1.0 / (1.0 + 0.18 * np.maximum(depth, -4.5))
    projected = rotated[:, :2] * scale[:, None]
    return projected


def panel_svg(
    projected: np.ndarray,
    labels: np.ndarray,
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    title: str,
) -> str:
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-9)
    scale = min((width - 48.0) / spans[0], (height - 58.0) / spans[1])

    def map_point(point: np.ndarray) -> tuple[float, float]:
        x = x0 + 24.0 + (point[0] - mins[0]) * scale
        y = y0 + height - 24.0 - (point[1] - mins[1]) * scale
        return x, y

    parts = [
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{width:.1f}" height="{height:.1f}" '
        'fill="#fffdf7" stroke="#d7d1c7" stroke-width="1"/>',
        f'<text x="{x0 + 18:.1f}" y="{y0 + 24:.1f}" font-size="16" '
        'font-family="Helvetica, Arial, sans-serif" fill="#222">'
        f"{title}</text>",
    ]

    for point, label in zip(projected, labels, strict=True):
        x, y = map_point(point)
        color = "#c0392b" if label == 0 else "#1f618d"
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.6" fill="{color}" fill-opacity="0.55"/>'
        )

    return "\n".join(parts)


def main() -> None:
    config = pingpong_mlp.GeneratorConfig(
        v=1.5,
        starting_vector=(1.0, 1.0, 0.0),
        power_bound=2,
        min_length=1,
        max_length=8,
        num_samples=1800,
        chunk_size=900,
        seed=11,
    )
    dataset = pingpong_mlp.generate_dataset(config)
    points = pingpong_mlp.feature_map(dataset["points"], "signed_log1p")
    labels = dataset["labels"]

    output_path = Path("artifacts/pingpong_sample_r3.svg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 1200
    height = 420
    panels = [
        (project_points(points, yaw=0.35, pitch=0.25), "View 1"),
        (project_points(points, yaw=1.2, pitch=-0.15), "View 2"),
        (project_points(points, yaw=2.15, pitch=0.45), "View 3"),
    ]

    body = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f4efe6"/>',
        '<text x="24" y="28" font-size="20" font-family="Helvetica, Arial, sans-serif" fill="#111">'
        'Ping-pong sample in R^3 (signed_log1p-scaled coordinates)</text>',
        '<text x="24" y="50" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#444">'
        'Red = A-labeled, Blue = B-labeled. Three rotated views of the same raw dataset.</text>',
        '<circle cx="930" cy="25" r="5" fill="#c0392b" fill-opacity="0.8"/>',
        '<text x="942" y="30" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#222">A</text>',
        '<circle cx="985" cy="25" r="5" fill="#1f618d" fill-opacity="0.8"/>',
        '<text x="997" y="30" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#222">B</text>',
    ]

    panel_width = 368.0
    panel_height = 330.0
    for index, (projected, title) in enumerate(panels):
        body.append(
            panel_svg(
                projected,
                labels,
                x0=24.0 + index * 384.0,
                y0=70.0,
                width=panel_width,
                height=panel_height,
                title=title,
            )
        )

    body.append("</svg>")
    output_path.write_text("\n".join(body), encoding="utf-8")
    print(output_path.resolve())


if __name__ == "__main__":
    main()
