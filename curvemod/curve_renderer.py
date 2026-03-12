from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

import setup_a3 as ct


ROOT_TO_ENDPOINTS: dict[tuple[int, int, int], tuple[int, int]] = {
    (1, 0, 0): (1, 2),
    (0, 1, 0): (2, 3),
    (0, 0, 1): (3, 4),
    (1, 1, 0): (1, 3),
    (0, 1, 1): (2, 4),
    (1, 1, 1): (1, 4),
}

DEFAULT_SAMPLING_TOLERANCE = 1e-3
DEFAULT_MAX_DEPTH = 12
DEFAULT_MAX_SEGMENT_LENGTH = 0.05
DEFAULT_ENDPOINT_TOLERANCE = 1e-5


def parse_artin_word(text: str) -> list[int]:
    cleaned = text.replace(",", " ").replace("\n", " ").strip()
    if not cleaned:
        return []

    letters: list[int] = []
    for token in cleaned.split():
        normalized = token.strip().lower()
        if not normalized:
            continue
        if normalized.startswith("s"):
            body = normalized[1:]
            if body.endswith("^-1"):
                letters.append(-int(body[:-3]))
            else:
                letters.append(int(body))
            continue
        letters.append(int(normalized))
    return letters


def _smoothstep(value: np.ndarray) -> np.ndarray:
    return value * value * (3.0 - 2.0 * value)


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    seg_norm_sq = float(np.dot(segment, segment))
    if seg_norm_sq == 0.0:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, segment) / seg_norm_sq)
    t = min(1.0, max(0.0, t))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def _to_point(point: Sequence[float]) -> np.ndarray:
    return np.asarray(point, dtype=float)


@dataclass(frozen=True)
class PunctureModel:
    punctures: np.ndarray
    outer_center: np.ndarray
    outer_radius: float
    twist_inner_radius: float
    twist_outer_radius: float

    @property
    def walls(self) -> tuple[float, float, float]:
        return (0.5, 1.5, 2.5)

    @classmethod
    def standard_a3(cls) -> "PunctureModel":
        punctures = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=float,
        )
        return cls(
            punctures=punctures,
            outer_center=np.array([1.5, 0.0], dtype=float),
            outer_radius=2.35,
            twist_inner_radius=0.6,
            twist_outer_radius=1.1,
        )

    def puncture_point(self, label: int) -> np.ndarray:
        if label < 1 or label > len(self.punctures):
            raise ValueError(f"Unsupported puncture label {label}")
        return self.punctures[label - 1].copy()

    def half_twist(self, letter: int) -> "HalfTwistMap":
        index = abs(letter)
        if index < 1 or index >= len(self.punctures):
            raise ValueError(f"Unsupported Artin letter {letter}")
        center = 0.5 * (self.puncture_point(index) + self.puncture_point(index + 1))
        direction = 1 if letter > 0 else -1
        return HalfTwistMap(
            center=center,
            inner_radius=self.twist_inner_radius,
            outer_radius=self.twist_outer_radius,
            direction=direction,
            letter=letter,
        )

    def detect_puncture_label(
        self, point: Sequence[float], tolerance: float = DEFAULT_ENDPOINT_TOLERANCE
    ) -> int:
        point_array = _to_point(point)
        distances = np.linalg.norm(self.punctures - point_array, axis=1)
        best_index = int(np.argmin(distances))
        if float(distances[best_index]) > tolerance:
            raise ValueError(
                f"Point {point_array.tolist()} does not land on a puncture; nearest distance is {distances[best_index]:.3e}"
            )
        return best_index + 1


@dataclass(frozen=True)
class HalfTwistMap:
    center: np.ndarray
    inner_radius: float
    outer_radius: float
    direction: int
    letter: int

    def apply(self, point: Sequence[float]) -> np.ndarray:
        point_array = _to_point(point)
        delta = point_array - self.center
        radius = float(np.linalg.norm(delta))
        if radius == 0.0:
            return self.center.copy()
        angle = float(np.arctan2(delta[1], delta[0]))
        if radius <= self.inner_radius:
            fraction = 1.0
        elif radius >= self.outer_radius:
            fraction = 0.0
        else:
            normalized = (self.outer_radius - radius) / (self.outer_radius - self.inner_radius)
            fraction = float(_smoothstep(np.asarray(normalized)))
        twist_angle = self.direction * np.pi * fraction
        new_angle = angle + twist_angle
        rotated = radius * np.array([np.cos(new_angle), np.sin(new_angle)], dtype=float)
        return self.center + rotated


@dataclass(frozen=True)
class ComposedMap:
    word: tuple[int, ...]
    twists: tuple[HalfTwistMap, ...]

    def apply(self, point: Sequence[float]) -> np.ndarray:
        out = _to_point(point)
        for twist in self.twists:
            out = twist.apply(out)
        return out


@dataclass(frozen=True)
class ParametricCurve:
    name: str
    start_label: int
    end_label: int
    point_at: Callable[[float], np.ndarray]

    def endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        return self.point_at(0.0), self.point_at(1.0)


@dataclass(frozen=True)
class CurvePair:
    word: tuple[int, ...]
    model: PunctureModel
    composed_map: ComposedMap
    base_curve: ParametricCurve
    moved_curve: ParametricCurve
    base_polyline: np.ndarray
    moved_polyline: np.ndarray
    moved_endpoint_pair: tuple[int, int]
    burau_endpoint_pair: tuple[int, int]
    base_endpoint_pair: tuple[int, int]
    wall_crossing_signature: tuple[tuple[int, int], ...]
    provenance: str


@dataclass(frozen=True)
class RenderStyle:
    show_base_curve: bool = True
    base_color: str = "#94a3b8"
    moved_color: str = "#b91c1c"
    base_linewidth: float = 1.0
    moved_linewidth: float = 0.8
    base_opacity: float = 0.35
    moved_opacity: float = 0.18
    puncture_radius: float = 4.5
    title: bool = True


@dataclass
class SvgFigure:
    svg_text: str

    def savefig(self, out_path: str) -> None:
        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.svg_text, encoding="utf-8")

    def clf(self) -> None:
        return None


def base_arc_curve(model: PunctureModel, base_arc: int) -> ParametricCurve:
    if base_arc < 1 or base_arc > 3:
        raise ValueError("Only base arcs P_1, P_2, and P_3 are supported")
    start = model.puncture_point(base_arc)
    end = model.puncture_point(base_arc + 1)

    def point_at(parameter: float) -> np.ndarray:
        t = float(parameter)
        return (1.0 - t) * start + t * end

    return ParametricCurve(name=f"P_{base_arc}", start_label=base_arc, end_label=base_arc + 1, point_at=point_at)


def sample_parametric_curve(
    curve: ParametricCurve,
    tolerance: float = DEFAULT_SAMPLING_TOLERANCE,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_segment_length: float = DEFAULT_MAX_SEGMENT_LENGTH,
) -> np.ndarray:
    start = curve.point_at(0.0)
    end = curve.point_at(1.0)
    points: list[np.ndarray] = [start]

    def refine(t0: float, p0: np.ndarray, t1: float, p1: np.ndarray, depth: int) -> None:
        tm = 0.5 * (t0 + t1)
        pm = curve.point_at(tm)
        segment_length = float(np.linalg.norm(p1 - p0))
        need_split = (
            depth < max_depth
            and (
                segment_length > max_segment_length
                or _point_segment_distance(pm, p0, p1) > tolerance
            )
        )
        if need_split:
            refine(t0, p0, tm, pm, depth + 1)
            refine(tm, pm, t1, p1, depth + 1)
            return
        points.append(p1)

    refine(0.0, start, 1.0, end, 0)
    return np.vstack(points)


def wall_crossing_signature(polyline: np.ndarray, model: PunctureModel) -> tuple[tuple[int, int], ...]:
    raw_signature: list[tuple[int, int]] = []
    walls = model.walls
    for index in range(len(polyline) - 1):
        start = polyline[index]
        end = polyline[index + 1]
        x0 = float(start[0])
        x1 = float(end[0])
        if x0 == x1:
            continue
        direction = 1 if x1 > x0 else -1
        lower = min(x0, x1)
        upper = max(x0, x1)
        for wall_index, wall_x in enumerate(walls, start=1):
            if not (lower < wall_x < upper):
                continue
            raw_signature.append((wall_index, direction))

    signature: list[tuple[int, int]] = []
    for crossing in raw_signature:
        if signature and signature[-1][0] == crossing[0] and signature[-1][1] == -crossing[1]:
            signature.pop()
        else:
            signature.append(crossing)
    return tuple(signature)


def curve_endpoints_from_burau(word: Sequence[int], base_arc: int = 1) -> tuple[int, int]:
    vec = ct.dim_vectors[base_arc]
    for letter in reversed(tuple(word)):
        vec = ct.oburau_fns[letter](vec)
    root = tuple(int(value) for value in ct.find_ends_vector(vec).tolist())
    if root not in ROOT_TO_ENDPOINTS:
        raise ValueError(f"Unsupported dequantized root {root} for word {tuple(word)}")
    return ROOT_TO_ENDPOINTS[root]


def curve_pair_from_word(
    word: Sequence[int],
    base_arc: int = 1,
    *,
    model: PunctureModel | None = None,
    sampling_tolerance: float = DEFAULT_SAMPLING_TOLERANCE,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_segment_length: float = DEFAULT_MAX_SEGMENT_LENGTH,
    endpoint_tolerance: float = DEFAULT_ENDPOINT_TOLERANCE,
) -> CurvePair:
    puncture_model = model or PunctureModel.standard_a3()
    word_tuple = tuple(int(letter) for letter in word)
    twists = tuple(puncture_model.half_twist(letter) for letter in reversed(word_tuple))
    composed_map = ComposedMap(word=word_tuple, twists=twists)
    base_curve = base_arc_curve(puncture_model, base_arc)

    def moved_point_at(parameter: float) -> np.ndarray:
        return composed_map.apply(base_curve.point_at(parameter))

    moved_curve = ParametricCurve(
        name=f"{word_tuple}(P_{base_arc})",
        start_label=base_curve.start_label,
        end_label=base_curve.end_label,
        point_at=moved_point_at,
    )

    base_polyline = sample_parametric_curve(
        base_curve,
        tolerance=sampling_tolerance,
        max_depth=max_depth,
        max_segment_length=max_segment_length,
    )
    moved_polyline = sample_parametric_curve(
        moved_curve,
        tolerance=sampling_tolerance,
        max_depth=max_depth,
        max_segment_length=max_segment_length,
    )

    moved_start_label = puncture_model.detect_puncture_label(moved_curve.point_at(0.0), tolerance=endpoint_tolerance)
    moved_end_label = puncture_model.detect_puncture_label(moved_curve.point_at(1.0), tolerance=endpoint_tolerance)
    moved_endpoint_pair = tuple(sorted((moved_start_label, moved_end_label)))
    burau_endpoint_pair = tuple(sorted(curve_endpoints_from_burau(word_tuple, base_arc=base_arc)))
    if moved_endpoint_pair != burau_endpoint_pair:
        raise ValueError(
            f"Geometry/Burau endpoint mismatch for word {word_tuple}: geometry {moved_endpoint_pair}, Burau {burau_endpoint_pair}"
        )

    provenance = (
        f"Curve {word_tuple}(P_{base_arc}) obtained by composing explicit half-twist homeomorphisms; "
        "the rightmost generator acts first to match the existing Burau code."
    )
    return CurvePair(
        word=word_tuple,
        model=puncture_model,
        composed_map=composed_map,
        base_curve=base_curve,
        moved_curve=moved_curve,
        base_polyline=base_polyline,
        moved_polyline=moved_polyline,
        moved_endpoint_pair=moved_endpoint_pair,
        burau_endpoint_pair=burau_endpoint_pair,
        base_endpoint_pair=(base_curve.start_label, base_curve.end_label),
        wall_crossing_signature=wall_crossing_signature(moved_polyline, puncture_model),
        provenance=provenance,
    )


def render_curve_pair(pair: CurvePair, out_path: str | None = None):
    return render_curve_pair_with_style(pair, RenderStyle(), out_path=out_path)


def render_curve_pair_with_style(pair: CurvePair, style: RenderStyle, out_path: str | None = None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        figure = _render_curve_pair_svg(pair, style)
        if out_path is not None:
            output_path = Path(out_path)
            if output_path.suffix.lower() not in {"", ".svg"}:
                raise ModuleNotFoundError("matplotlib is not installed; save to an .svg path or install matplotlib for raster output")
            figure.savefig(str(output_path))
        return figure

    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    boundary = plt.Circle(
        pair.model.outer_center,
        pair.model.outer_radius,
        fill=False,
        linestyle="--",
        linewidth=1.0,
        color="0.6",
    )
    axis.add_patch(boundary)
    if style.show_base_curve:
        axis.plot(
            pair.base_polyline[:, 0],
            pair.base_polyline[:, 1],
            color=style.base_color,
            linewidth=style.base_linewidth,
            alpha=style.base_opacity,
            label=pair.base_curve.name,
        )
    for start, end in zip(pair.moved_polyline[:-1], pair.moved_polyline[1:]):
        axis.plot(
            [float(start[0]), float(end[0])],
            [float(start[1]), float(end[1])],
            color=style.moved_color,
            linewidth=style.moved_linewidth,
            alpha=style.moved_opacity,
            solid_capstyle="round",
        )
    axis.scatter(pair.model.punctures[:, 0], pair.model.punctures[:, 1], color="black", zorder=3, s=18.0)
    for label, puncture in enumerate(pair.model.punctures, start=1):
        axis.text(float(puncture[0]), float(puncture[1] - 0.12), str(label), ha="center", va="top")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-0.5, 3.5)
    axis.set_ylim(-2.5, 2.5)
    axis.axis("off")
    if style.show_base_curve:
        axis.legend(loc="upper right")
    if style.title:
        axis.set_title(f"[w, sigma_1] via (P_1, w(P_1)); endpoints {pair.moved_endpoint_pair}")
    if out_path is not None:
        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
    return figure


def _render_curve_pair_svg(pair: CurvePair, style: RenderStyle) -> SvgFigure:
    x_min, x_max = -0.5, 3.5
    y_min, y_max = -2.5, 2.5
    width = 800.0
    height = 450.0
    padding = 35.0

    def map_point(point: Sequence[float]) -> tuple[float, float]:
        x, y = float(point[0]), float(point[1])
        px = padding + (x - x_min) * (width - 2.0 * padding) / (x_max - x_min)
        py = height - padding - (y - y_min) * (height - 2.0 * padding) / (y_max - y_min)
        return px, py

    def polyline_path(polyline: np.ndarray) -> str:
        mapped = [map_point(point) for point in polyline]
        return " ".join(f"{x:.3f},{y:.3f}" for x, y in mapped)

    center_x, center_y = map_point(pair.model.outer_center)
    radius_px = pair.model.outer_radius * (width - 2.0 * padding) / (x_max - x_min)
    punctures = []
    labels = []
    for label, puncture in enumerate(pair.model.punctures, start=1):
        px, py = map_point(puncture)
        punctures.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="{style.puncture_radius:.3f}" fill="#111111" />')
        labels.append(
            f'<text x="{px:.3f}" y="{py + 18.0:.3f}" font-size="16" text-anchor="middle" fill="#111111">{label}</text>'
        )

    title = f"[w, sigma_1] via (P_1, w(P_1)); endpoints {pair.moved_endpoint_pair}"
    legend = ""
    if style.show_base_curve:
        legend = (
            f'<text x="560" y="50" font-size="16" fill="{style.base_color}" fill-opacity="{style.base_opacity:.3f}">P_1</text>'
            f'<text x="560" y="72" font-size="16" fill="{style.moved_color}" fill-opacity="{min(1.0, style.moved_opacity * 3.0):.3f}">w(P_1)</text>'
        )

    moved_segments = []
    mapped_points = [map_point(point) for point in pair.moved_polyline]
    for start, end in zip(mapped_points[:-1], mapped_points[1:]):
        moved_segments.append(
            f'<line x1="{start[0]:.3f}" y1="{start[1]:.3f}" x2="{end[0]:.3f}" y2="{end[1]:.3f}" '
            f'stroke="{style.moved_color}" stroke-width="{style.moved_linewidth:.3f}" '
            f'stroke-opacity="{style.moved_opacity:.3f}" stroke-linecap="round" />'
        )
    base_path = ""
    if style.show_base_curve:
        base_path = (
            f'<polyline fill="none" stroke="{style.base_color}" stroke-width="{style.base_linewidth:.3f}" '
            f'stroke-opacity="{style.base_opacity:.3f}" points="{polyline_path(pair.base_polyline)}" />'
        )
    title_text = ""
    if style.title:
        title_text = f'<text x="35" y="28" font-size="18" fill="#111111">{title}</text>'
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">
<rect width="100%" height="100%" fill="white" />
<circle cx="{center_x:.3f}" cy="{center_y:.3f}" r="{radius_px:.3f}" fill="none" stroke="#999999" stroke-width="1.5" stroke-dasharray="7 5" />
{base_path}
{''.join(moved_segments)}
{''.join(punctures)}
{''.join(labels)}
{title_text}
{legend}
</svg>
"""
    return SvgFigure(svg_text=svg)
