#!/usr/bin/env python3
"""Generate the three separate panels for the principal behavioural result."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paper_figure_style import PALETTE, SvgCanvas


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts" / "paper" / "figure3"
STATISTICS = REPOSITORY_ROOT / "results" / "principal_sweep_v1_statistics.json"
EXPECTED_STATISTICS_SHA256 = (
    "55ad1f50399e54487728ade10e6605a993064369a80ec48c0674704f2f378902"
)
EXPECTED_PROTOCOL_SHA256 = (
    "1f8ececf4492f68517938ccd2fd1e05bb459e0fdd7275851536f71d782bed448"
)
EXPECTED_SCHEDULE_SHA256 = (
    "f73664bef2f49c3fb52cd3a8b3e0028ae640acc2814fce466b48fa0737d5177b"
)
BLACKOUT_STEPS = (0, 5, 10, 15, 20, 25)
STEP_MILLISECONDS = 20


@dataclass(frozen=True)
class SeriesStyle:
    label: str
    colour: str
    marker: str
    dash: str | None = None
    width: float = 3.0


STYLES = {
    "feed_forward": SeriesStyle("Feed-forward", PALETTE["orange"], "square"),
    "finite_stack_10": SeriesStyle(
        "10-step stack", PALETTE["purple"], "diamond"
    ),
    "structured_k0": SeriesStyle(
        "Structured k=0", PALETTE["green"], "circle", width=3.6
    ),
    "structured_k1": SeriesStyle(
        "k=1", PALETTE["orange"], "square", dash="10 6"
    ),
    "structured_k2": SeriesStyle(
        "k=2", PALETTE["vermillion"], "triangle", dash="12 4 3 4"
    ),
    "structured_k4": SeriesStyle(
        "k=4", PALETTE["purple"], "diamond", dash="3 5"
    ),
    "gru_n64": SeriesStyle(
        "GRU-64", PALETTE["muted"], "cross", dash="8 5"
    ),
    "teacher": SeriesStyle(
        "Teacher", PALETTE["blue"], "triangle", dash="14 5", width=3.6
    ),
}

PANEL_A = ("feed_forward", "finite_stack_10", "structured_k0", "teacher")
PANEL_B = (
    "structured_k0",
    "structured_k1",
    "structured_k2",
    "structured_k4",
    "gru_n64",
    "teacher",
)
CONTRASTS = (
    (
        "structured_k0_minus_feed_forward",
        "k=0 − feed-forward",
        "structured_k0",
    ),
    (
        "structured_k0_minus_finite_stack_10",
        "k=0 − 10-step stack",
        "structured_k0",
    ),
    ("structured_k0_minus_gru_n64", "k=0 − GRU-64", "gru_n64"),
    ("structured_k1_minus_structured_k0", "k=1 − k=0", "structured_k1"),
    ("structured_k2_minus_structured_k0", "k=2 − k=0", "structured_k2"),
    ("structured_k4_minus_structured_k0", "k=4 − k=0", "structured_k4"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_statistics() -> dict[str, Any]:
    if _sha256(STATISTICS) != EXPECTED_STATISTICS_SHA256:
        raise ValueError("principal statistics do not match the frozen analysis")
    value = json.loads(STATISTICS.read_text(encoding="utf-8"))
    if value.get("status") != "completed":
        raise ValueError("principal statistics are incomplete")
    if value.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("principal protocol hash does not match")
    if value.get("principal_test_schedule_sha256") != EXPECTED_SCHEDULE_SHA256:
        raise ValueError("principal test schedule hash does not match")
    bootstrap = value.get("bootstrap", {})
    if int(bootstrap.get("replicates", -1)) != 10_000:
        raise ValueError("Figure 3 expects the frozen 10,000-replicate analysis")
    expected_keys = {str(step) for step in BLACKOUT_STEPS}
    rates = value.get("student_save_rates", {})
    intervals = value.get("student_save_rate_95_intervals", {})
    for family_id in STYLES:
        if family_id == "teacher":
            continue
        if set(rates.get(family_id, {})) != expected_keys:
            raise ValueError(f"incomplete save-rate curve for {family_id}")
        if set(intervals.get(family_id, {})) != expected_keys:
            raise ValueError(f"incomplete interval curve for {family_id}")
    if set(value.get("teacher_save_rates", {})) != expected_keys:
        raise ValueError("incomplete teacher save-rate curve")
    differences = value.get("planned_differences_at_20_steps", {})
    if any(key not in differences for key, _, _ in CONTRASTS):
        raise ValueError("a predeclared 400 ms contrast is missing")
    return value


def _marker(
    canvas: SvgCanvas,
    x: float,
    y: float,
    *,
    shape: str,
    colour: str,
    radius: float = 6.0,
) -> None:
    if shape == "circle":
        canvas.circle(
            x,
            y,
            radius,
            fill=colour,
            stroke=PALETTE["paper"],
            stroke_width=1.4,
        )
    elif shape == "square":
        canvas.rect(
            x - radius,
            y - radius,
            2 * radius,
            2 * radius,
            fill=colour,
            stroke=PALETTE["paper"],
            stroke_width=1.4,
            rx=0.8,
        )
    elif shape == "diamond":
        canvas.polygon(
            (
                (x, y - radius - 1),
                (x + radius + 1, y),
                (x, y + radius + 1),
                (x - radius - 1, y),
            ),
            fill=colour,
            stroke=PALETTE["paper"],
            stroke_width=1.4,
        )
    elif shape == "triangle":
        canvas.polygon(
            (
                (x, y - radius - 1),
                (x + radius + 1, y + radius),
                (x - radius - 1, y + radius),
            ),
            fill=colour,
            stroke=PALETTE["paper"],
            stroke_width=1.4,
        )
    elif shape == "cross":
        canvas.line(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            stroke=colour,
            stroke_width=2.5,
        )
        canvas.line(
            x - radius,
            y + radius,
            x + radius,
            y - radius,
            stroke=colour,
            stroke_width=2.5,
        )
    else:
        raise ValueError(f"unknown marker shape {shape!r}")


def _panel_heading(canvas: SvgCanvas, label: str, title: str) -> None:
    canvas.text(18, 38, label, size=30, weight=700)
    canvas.text(70, 38, title, size=28, weight=700)
    canvas.line(18, 55, 642, 55, stroke=PALETTE["grid"], stroke_width=1.5)


def _legend(
    canvas: SvgCanvas,
    series_ids: tuple[str, ...],
    *,
    x: float,
    y: float,
    columns: int,
    column_width: float,
) -> None:
    rows = (len(series_ids) + columns - 1) // columns
    width = columns * column_width + 18
    height = rows * 29 + 14
    canvas.rect(
        x,
        y,
        width,
        height,
        fill=PALETTE["paper"],
        stroke=PALETTE["grid"],
        stroke_width=1.0,
        rx=3,
        opacity=0.96,
    )
    for index, series_id in enumerate(series_ids):
        row = index // columns
        column = index % columns
        item_x = x + 12 + column * column_width
        item_y = y + 20 + row * 29
        style = STYLES[series_id]
        canvas.line(
            item_x,
            item_y,
            item_x + 30,
            item_y,
            stroke=style.colour,
            stroke_width=style.width,
            dash=style.dash,
        )
        _marker(
            canvas,
            item_x + 15,
            item_y,
            shape=style.marker,
            colour=style.colour,
            radius=4.5,
        )
        canvas.text(
            item_x + 38,
            item_y + 6,
            style.label,
            size=18.5,
            weight=600,
        )


def _curve_panel(
    statistics: dict[str, Any],
    *,
    panel_label: str,
    title: str,
    series_ids: tuple[str, ...],
    y_min: float,
    y_ticks: tuple[float, ...],
    legend_y: float,
    subtitle: str,
) -> str:
    canvas = SvgCanvas(
        660,
        560,
        title=f"Figure 3{panel_label.strip('()')}: {title}",
        description=(
            "Principal-test save rate against blackout duration. Student curves "
            "show five-seed means and paired hierarchical bootstrap intervals."
        ),
    )
    _panel_heading(canvas, panel_label, title)
    canvas.text(
        18,
        82,
        subtitle,
        size=20,
        fill=PALETTE["muted"],
        weight=600,
    )

    x0, x1 = 88.0, 632.0
    y0, y1 = 105.0, 465.0
    y_max = 1.005

    def x_for(step: int) -> float:
        return x0 + (x1 - x0) * step / max(BLACKOUT_STEPS)

    def y_for(rate: float) -> float:
        return y1 - (y1 - y0) * (rate - y_min) / (y_max - y_min)

    canvas.rect(
        x0,
        y0,
        x1 - x0,
        y1 - y0,
        fill=PALETTE["paper"],
        stroke=PALETTE["ink"],
        stroke_width=1.2,
    )
    for tick in y_ticks:
        y = y_for(tick)
        canvas.line(
            x0,
            y,
            x1,
            y,
            stroke=PALETTE["light_grid"],
            stroke_width=1.2,
        )
        canvas.text(
            x0 - 10,
            y + 7,
            f"{100 * tick:.0f}",
            size=20,
            fill=PALETTE["muted"],
            anchor="end",
        )
    for step in BLACKOUT_STEPS:
        x = x_for(step)
        canvas.line(x, y1, x, y1 + 7, stroke=PALETTE["ink"], stroke_width=1.2)
        canvas.text(
            x,
            y1 + 28,
            f"{step * STEP_MILLISECONDS}",
            size=20,
            fill=PALETTE["muted"],
            anchor="middle",
        )
    comparison_x = x_for(20)
    canvas.line(
        comparison_x,
        y0,
        comparison_x,
        y1,
        stroke=PALETTE["muted"],
        stroke_width=1.3,
        dash="6 5",
    )
    canvas.text(
        comparison_x - 5,
        y0 - 7,
        "400 ms",
        size=18,
        fill=PALETTE["muted"],
        anchor="end",
    )
    canvas.text(
        (x0 + x1) / 2,
        538,
        "blackout duration (ms)",
        size=22,
        weight=600,
        anchor="middle",
    )
    canvas.text(
        24,
        (y0 + y1) / 2,
        "save rate (%)",
        size=22,
        weight=600,
        anchor="middle",
        rotate=-90,
    )

    rates = statistics["student_save_rates"]
    intervals = statistics["student_save_rate_95_intervals"]
    teacher_rates = statistics["teacher_save_rates"]
    for series_id in series_ids:
        if series_id == "teacher":
            continue
        style = STYLES[series_id]
        for step in BLACKOUT_STEPS:
            x = x_for(step)
            lower, upper = intervals[series_id][str(step)]
            low_y, high_y = y_for(float(lower)), y_for(float(upper))
            canvas.line(
                x,
                high_y,
                x,
                low_y,
                stroke=style.colour,
                stroke_width=1.5,
                opacity=0.62,
            )
            canvas.line(
                x - 4,
                high_y,
                x + 4,
                high_y,
                stroke=style.colour,
                stroke_width=1.5,
                opacity=0.62,
            )
            canvas.line(
                x - 4,
                low_y,
                x + 4,
                low_y,
                stroke=style.colour,
                stroke_width=1.5,
                opacity=0.62,
            )

    for series_id in series_ids:
        style = STYLES[series_id]
        source = teacher_rates if series_id == "teacher" else rates[series_id]
        points = [
            (x_for(step), y_for(float(source[str(step)])))
            for step in BLACKOUT_STEPS
        ]
        path = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in points)
        canvas.path(
            path,
            stroke=style.colour,
            stroke_width=style.width,
            dash=style.dash,
        )
        for x, y in points:
            _marker(canvas, x, y, shape=style.marker, colour=style.colour)

    _legend(
        canvas,
        series_ids,
        x=99,
        y=legend_y,
        columns=2,
        column_width=205,
    )
    return canvas.render()


def _contrast_panel(statistics: dict[str, Any]) -> str:
    canvas = SvgCanvas(
        660,
        560,
        title="Figure 3c: predeclared 400 millisecond contrasts",
        description=(
            "Forest plot of paired save-rate differences at the predeclared "
            "400 millisecond blackout with 95 percent bootstrap intervals."
        ),
    )
    _panel_heading(canvas, "(c)", "400 ms contrasts")
    canvas.text(
        18,
        82,
        "estimate [paired 95% interval]",
        size=20,
        fill=PALETTE["muted"],
        weight=600,
    )

    x0, x1 = 250.0, 638.0
    y_axis = 486.0
    domain_min, domain_max = -5.0, 70.0

    def x_for(value: float) -> float:
        return x0 + (x1 - x0) * (value - domain_min) / (
            domain_max - domain_min
        )

    for tick in (0.0, 20.0, 40.0, 60.0):
        x = x_for(tick)
        canvas.line(
            x,
            100,
            x,
            y_axis,
            stroke=PALETTE["light_grid"],
            stroke_width=1.2,
        )
        canvas.line(x, y_axis, x, y_axis + 7, stroke=PALETTE["ink"], stroke_width=1.2)
        canvas.text(
            x,
            y_axis + 28,
            f"{tick:.0f}",
            size=20,
            fill=PALETTE["muted"],
            anchor="middle",
        )
    canvas.line(x0, y_axis, x1, y_axis, stroke=PALETTE["ink"], stroke_width=1.2)
    canvas.line(
        x_for(0),
        100,
        x_for(0),
        y_axis,
        stroke=PALETTE["ink"],
        stroke_width=1.8,
    )
    canvas.text(
        (x0 + x1) / 2,
        538,
        "save-rate difference (percentage points)",
        size=21,
        weight=600,
        anchor="middle",
    )
    canvas.line(18, 295, 638, 295, stroke=PALETTE["grid"], stroke_width=1.2)

    row_positions = (128.0, 190.0, 252.0, 338.0, 400.0, 462.0)
    differences = statistics["planned_differences_at_20_steps"]
    for y, (key, label, style_id) in zip(
        row_positions, CONTRASTS, strict=True
    ):
        comparison = differences[key]
        estimate = float(comparison["estimate_points"])
        lower, upper = (
            float(value)
            for value in comparison["percentile_95_interval_points"]
        )
        style = STYLES[style_id]
        canvas.text(232, y - 6, label, size=20, weight=700, anchor="end")
        canvas.text(
            232,
            y + 17,
            f"{estimate:+.1f} [{lower:.1f}, {upper:.1f}] pp",
            size=16.5,
            fill=PALETTE["muted"],
            anchor="end",
        )
        low_x, point_x, high_x = x_for(lower), x_for(estimate), x_for(upper)
        canvas.line(
            low_x,
            y,
            high_x,
            y,
            stroke=style.colour,
            stroke_width=3.0,
        )
        canvas.line(
            low_x,
            y - 7,
            low_x,
            y + 7,
            stroke=style.colour,
            stroke_width=2.0,
        )
        canvas.line(
            high_x,
            y - 7,
            high_x,
            y + 7,
            stroke=style.colour,
            stroke_width=2.0,
        )
        _marker(
            canvas,
            point_x,
            y,
            shape=style.marker,
            colour=style.colour,
            radius=6.2,
        )
    return canvas.render()


def render_svgs(statistics: dict[str, Any]) -> dict[str, str]:
    return {
        "figure3a_memory_baselines.svg": _curve_panel(
            statistics,
            panel_label="(a)",
            title="Memory baselines",
            series_ids=PANEL_A,
            y_min=0.30,
            y_ticks=(0.4, 0.6, 0.8, 1.0),
            legend_y=354,
            subtitle="five-seed mean; paired 95% intervals",
        ),
        "figure3b_recurrent_comparison.svg": _curve_panel(
            statistics,
            panel_label="(b)",
            title="Recurrent comparison",
            series_ids=PANEL_B,
            y_min=0.90,
            y_ticks=(0.90, 0.95, 1.0),
            legend_y=352,
            subtitle="five-seed mean; expanded 90–100% scale",
        ),
        "figure3c_400ms_contrasts.svg": _contrast_panel(statistics),
    }


def _convert_svg(svg_path: Path, output_format: str) -> Path:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError(
            "rsvg-convert is required for PDF/PNG output; request SVG-only output"
        )
    output_path = svg_path.with_suffix(f".{output_format}")
    command = [converter, "--format", output_format, "--output", str(output_path)]
    if output_format == "png":
        command.extend(("--width", "1800"))
    command.append(str(svg_path))
    subprocess.run(command, check=True)
    return output_path


def generate(output_dir: Path, formats: tuple[str, ...]) -> Path:
    statistics = load_statistics()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for filename, content in render_svgs(statistics).items():
        path = output_dir / filename
        path.write_text(content, encoding="utf-8")
        generated.append(path)
        for output_format in formats:
            if output_format != "svg":
                generated.append(_convert_svg(path, output_format))

    differences = statistics["planned_differences_at_20_steps"]
    manifest = {
        "schema_version": 1,
        "figure": "figure3_principal_behavioural_result",
        "palette": "Okabe-Ito with redundant marker and line encodings",
        "source": {
            str(STATISTICS.relative_to(REPOSITORY_ROOT)): {
                "sha256": _sha256(STATISTICS),
                "protocol_sha256": statistics["protocol_sha256"],
                "principal_test_schedule_sha256": statistics[
                    "principal_test_schedule_sha256"
                ],
            }
        },
        "bootstrap": statistics["bootstrap"],
        "selected_values": {
            "blackout_steps": list(BLACKOUT_STEPS),
            "blackout_milliseconds": [
                step * STEP_MILLISECONDS for step in BLACKOUT_STEPS
            ],
            "panel_a_families": list(PANEL_A),
            "panel_b_families": list(PANEL_B),
            "contrasts_at_20_steps": {
                key: differences[key] for key, _, _ in CONTRASTS
            },
        },
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(generated)
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory for generated figure panels",
    )
    parser.add_argument(
        "--formats",
        default="svg,pdf,png",
        help="comma-separated output formats chosen from svg,pdf,png",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    formats = tuple(
        part.strip().lower()
        for part in arguments.formats.split(",")
        if part.strip()
    )
    if not formats or set(formats) - {"svg", "pdf", "png"}:
        raise SystemExit("--formats must contain one or more of svg,pdf,png")
    print(generate(arguments.output_dir.resolve(), formats))


if __name__ == "__main__":
    main()
