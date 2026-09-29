#!/usr/bin/env python3
"""Generate the two separate panels for the performance--cost frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics as summary_statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paper_figure_style import PALETTE, SvgCanvas


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts" / "paper" / "figure4"
STATISTICS = REPOSITORY_ROOT / "results" / "principal_sweep_v1_statistics.json"
OPTIMISATION = (
    REPOSITORY_ROOT / "results" / "principal_structured_optimisation_v4.json"
)
EXPECTED_STATISTICS_SHA256 = (
    "55ad1f50399e54487728ade10e6605a993064369a80ec48c0674704f2f378902"
)
EXPECTED_OPTIMISATION_SHA256 = (
    "a1a5add1c5a5e08b40bec2331e35b9fc9aea140119cd9c0c88a84bec9f3f8c85"
)
BLACKOUT_STEPS = "20"
BLACKOUT_MILLISECONDS = 400
SEEDS = (14303, 14304, 14305, 14306, 14307)


@dataclass(frozen=True)
class FamilyStyle:
    label: str
    colour: str
    marker: str


STYLES = {
    "feed_forward": FamilyStyle("Feed-forward", PALETTE["orange"], "square"),
    "finite_stack_10": FamilyStyle(
        "10-step stack", PALETTE["purple"], "diamond"
    ),
    "structured_k0": FamilyStyle("k=0", PALETTE["green"], "circle"),
    "structured_k1": FamilyStyle("k=1", PALETTE["sky"], "square"),
    "structured_k2": FamilyStyle("k=2", PALETTE["vermillion"], "triangle"),
    "structured_k4": FamilyStyle("k=4", PALETTE["muted"], "diamond"),
    "gru_n64": FamilyStyle("GRU-64", PALETTE["blue"], "cross"),
}
FAMILY_ORDER = tuple(STYLES)


@dataclass(frozen=True)
class FamilySummary:
    family_id: str
    save_rate: float
    latency_microseconds: float
    parameters: int
    seed_save_rates: tuple[float, ...]
    seed_latencies: tuple[float, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_summaries() -> tuple[dict[str, Any], dict[str, Any], dict[str, FamilySummary]]:
    if _sha256(STATISTICS) != EXPECTED_STATISTICS_SHA256:
        raise ValueError("principal statistics do not match the frozen analysis")
    if _sha256(OPTIMISATION) != EXPECTED_OPTIMISATION_SHA256:
        raise ValueError("optimisation evidence does not match the frozen V4 result")

    principal = json.loads(STATISTICS.read_text(encoding="utf-8"))
    optimisation = json.loads(OPTIMISATION.read_text(encoding="utf-8"))
    if principal.get("status") != "completed":
        raise ValueError("principal statistics are incomplete")
    if optimisation.get("decision") != "GO":
        raise ValueError("structured inference optimisation did not pass")
    if optimisation.get("latency", {}).get("new_isolated_results") != 35:
        raise ValueError("Figure 4 expects all 35 isolated latency results")

    family_latency = optimisation["latency"]["families"]
    save_rates = principal["student_save_rates"]
    seed_save_rates = principal["student_seed_save_rates"]
    seed_latency = principal["efficiency_seed_points"]
    summaries: dict[str, FamilySummary] = {}
    for family_id in FAMILY_ORDER:
        saves_by_seed = seed_save_rates[family_id][BLACKOUT_STEPS]
        latency_by_seed = {
            int(item["training_seed"]): float(item["median_microseconds"])
            for item in seed_latency[family_id]
        }
        if tuple(sorted(int(seed) for seed in saves_by_seed)) != SEEDS:
            raise ValueError(f"unexpected performance seeds for {family_id}")
        if tuple(sorted(latency_by_seed)) != SEEDS:
            raise ValueError(f"unexpected latency seeds for {family_id}")
        saves = tuple(float(saves_by_seed[str(seed)]) for seed in SEEDS)
        latencies = tuple(latency_by_seed[seed] for seed in SEEDS)
        save_rate = float(save_rates[family_id][BLACKOUT_STEPS])
        latency = float(family_latency[family_id]["v4_microseconds"])
        if not math.isclose(save_rate, sum(saves) / len(saves), abs_tol=1e-12):
            raise ValueError(f"five-seed save-rate mean mismatch for {family_id}")
        if not math.isclose(
            latency,
            summary_statistics.median(latencies),
            abs_tol=1e-12,
        ):
            raise ValueError(f"latency median mismatch for {family_id}")
        summaries[family_id] = FamilySummary(
            family_id=family_id,
            save_rate=save_rate,
            latency_microseconds=latency,
            parameters=int(family_latency[family_id]["parameters"]),
            seed_save_rates=saves,
            seed_latencies=latencies,
        )
    return principal, optimisation, summaries


def _marker(
    canvas: SvgCanvas,
    x: float,
    y: float,
    *,
    shape: str,
    colour: str,
    radius: float,
    opacity: float = 1.0,
    median: bool = False,
) -> None:
    stroke = PALETTE["paper"] if median else colour
    stroke_width = 2.0 if median else 1.5
    fill = colour if median else PALETTE["paper"]
    if shape == "circle":
        canvas.circle(
            x,
            y,
            radius,
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
            opacity=opacity,
        )
    elif shape == "square":
        canvas.rect(
            x - radius,
            y - radius,
            2 * radius,
            2 * radius,
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
            opacity=opacity,
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
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
            opacity=opacity,
        )
    elif shape == "triangle":
        canvas.polygon(
            (
                (x, y - radius - 1),
                (x + radius + 1, y + radius),
                (x - radius - 1, y + radius),
            ),
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
            opacity=opacity,
        )
    elif shape == "cross":
        canvas.line(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            stroke=colour,
            stroke_width=3.0 if median else 1.8,
            opacity=opacity,
        )
        canvas.line(
            x - radius,
            y + radius,
            x + radius,
            y - radius,
            stroke=colour,
            stroke_width=3.0 if median else 1.8,
            opacity=opacity,
        )
    else:
        raise ValueError(f"unknown marker shape {shape!r}")


def _frontier(summaries: dict[str, FamilySummary], cost: str) -> list[FamilySummary]:
    ordered = sorted(summaries.values(), key=lambda item: getattr(item, cost))
    frontier: list[FamilySummary] = []
    best_rate = -math.inf
    for item in ordered:
        if item.save_rate > best_rate:
            frontier.append(item)
            best_rate = item.save_rate
    return frontier


LATENCY_LABELS = {
    "feed_forward": (18.0, 19.0, "start"),
    "finite_stack_10": (-22.0, 76.0, "end"),
    "structured_k0": (-24.0, 48.0, "end"),
    "structured_k1": (-32.0, 116.0, "end"),
    "structured_k2": (24.0, -23.0, "start"),
    "structured_k4": (50.0, 54.0, "start"),
    "gru_n64": (-22.0, 92.0, "end"),
}

PARAMETER_LABELS = {
    "feed_forward": (18.0, 19.0, "start"),
    "finite_stack_10": (28.0, 110.0, "start"),
    "structured_k0": (-30.0, 48.0, "end"),
    "structured_k1": (-52.0, 112.0, "end"),
    "structured_k2": (35.0, -23.0, "start"),
    "structured_k4": (82.0, 55.0, "start"),
    "gru_n64": (-22.0, 92.0, "end"),
}


def _panel(
    summaries: dict[str, FamilySummary],
    *,
    panel_label: str,
    title: str,
    cost: str,
) -> str:
    canvas = SvgCanvas(
        850,
        610,
        title=f"Figure 4{panel_label.strip('()')}: {title}",
        description=(
            "Principal-test 400 millisecond save rate against student inference "
            "cost. Small symbols are training seeds and large symbols are family "
            "summaries."
        ),
    )
    canvas.text(18, 38, panel_label, size=30, weight=700)
    canvas.text(70, 38, title, size=28, weight=700)
    canvas.line(18, 55, 826, 55, stroke=PALETTE["grid"], stroke_width=1.5)
    canvas.text(
        18,
        82,
        "small symbols: five seeds · large symbols: family summary",
        size=19,
        fill=PALETTE["muted"],
        weight=600,
    )

    x0, x1 = 104.0, 814.0
    y0, y1 = 112.0, 500.0
    y_min, y_max = 0.25, 1.01
    if cost == "latency_microseconds":
        x_min, x_max = 10.0, 43.0
        x_ticks = (10.0, 20.0, 30.0, 40.0)
        x_label = "batch-one CPU latency (μs)"
        labels = LATENCY_LABELS
    elif cost == "parameters":
        x_min, x_max = 4_000.0, 31_000.0
        x_ticks = (5_000.0, 10_000.0, 15_000.0, 20_000.0, 25_000.0, 30_000.0)
        x_label = "total parameters"
        labels = PARAMETER_LABELS
    else:
        raise ValueError(f"unknown cost {cost!r}")

    def x_for(value: float) -> float:
        return x0 + (x1 - x0) * (value - x_min) / (x_max - x_min)

    def y_for(value: float) -> float:
        return y1 - (y1 - y0) * (value - y_min) / (y_max - y_min)

    canvas.rect(
        x0,
        y0,
        x1 - x0,
        y1 - y0,
        fill=PALETTE["paper"],
        stroke=PALETTE["ink"],
        stroke_width=1.2,
    )
    for tick in (0.4, 0.6, 0.8, 1.0):
        y = y_for(tick)
        canvas.line(
            x0,
            y,
            x1,
            y,
            stroke=PALETTE["light_grid"],
            stroke_width=1.3,
        )
        canvas.text(
            x0 - 11,
            y + 7,
            f"{100 * tick:.0f}",
            size=20,
            fill=PALETTE["muted"],
            anchor="end",
        )
    for tick in x_ticks:
        x = x_for(tick)
        canvas.line(x, y1, x, y1 + 7, stroke=PALETTE["ink"], stroke_width=1.2)
        label = f"{tick / 1000:.0f}k" if cost == "parameters" else f"{tick:.0f}"
        canvas.text(
            x,
            y1 + 29,
            label,
            size=20,
            fill=PALETTE["muted"],
            anchor="middle",
        )
    canvas.text(
        29,
        (y0 + y1) / 2,
        "400 ms save rate (%)",
        size=21,
        weight=600,
        anchor="middle",
        rotate=-90,
    )
    canvas.text(
        (x0 + x1) / 2,
        557,
        x_label,
        size=22,
        weight=600,
        anchor="middle",
    )

    frontier = _frontier(summaries, cost)
    canvas.polyline(
        tuple(
            (x_for(float(getattr(item, cost))), y_for(item.save_rate))
            for item in frontier
        ),
        stroke=PALETTE["grid"],
        stroke_width=2.5,
        dash="7 6",
    )

    parameter_jitter = (-120.0, -60.0, 0.0, 60.0, 120.0)
    for family_id in FAMILY_ORDER:
        item = summaries[family_id]
        style = STYLES[family_id]
        for index, save_rate in enumerate(item.seed_save_rates):
            if cost == "latency_microseconds":
                seed_cost = item.seed_latencies[index]
            else:
                seed_cost = float(item.parameters) + parameter_jitter[index]
            _marker(
                canvas,
                x_for(seed_cost),
                y_for(save_rate),
                shape=style.marker,
                colour=style.colour,
                radius=4.2,
                opacity=0.62,
            )

    for family_id in FAMILY_ORDER:
        item = summaries[family_id]
        style = STYLES[family_id]
        item_cost = float(getattr(item, cost))
        point_x, point_y = x_for(item_cost), y_for(item.save_rate)
        _marker(
            canvas,
            point_x,
            point_y,
            shape=style.marker,
            colour=style.colour,
            radius=8.3 if family_id in {"structured_k0", "gru_n64"} else 7.0,
            median=True,
        )
        dx, dy, anchor = labels[family_id]
        label_x, label_y = point_x + dx, point_y + dy
        leader_end_x = label_x - 7 if anchor == "start" else label_x + 7
        canvas.line(
            point_x,
            point_y,
            leader_end_x,
            label_y - 6,
            stroke=style.colour,
            stroke_width=1.4,
            opacity=0.82,
        )
        canvas.text(
            label_x,
            label_y,
            style.label,
            size=18,
            fill=PALETTE["ink"],
            weight=700,
            anchor=anchor,
        )
        if family_id in {"structured_k0", "gru_n64"}:
            if cost == "latency_microseconds":
                detail = f"{100 * item.save_rate:.1f}%, {item_cost:.1f} μs"
            else:
                detail = f"{100 * item.save_rate:.1f}%, {item.parameters:,}"
            canvas.text(
                label_x,
                label_y + 19,
                detail,
                size=15.5,
                fill=PALETTE["muted"],
                weight=600,
                anchor=anchor,
            )

    # Parameter-count offsets are explained in the manuscript caption.
    if cost != "parameters":
        canvas.text(
            x0 + 10,
            y1 - 14,
            "pale dashed line: observed mean frontier",
            size=15,
            fill=PALETTE["muted"],
            weight=600,
        )
    return canvas.render()


def render_svgs(summaries: dict[str, FamilySummary]) -> dict[str, str]:
    return {
        "figure4a_latency_frontier.svg": _panel(
            summaries,
            panel_label="(a)",
            title="Performance versus latency",
            cost="latency_microseconds",
        ),
        "figure4b_parameter_frontier.svg": _panel(
            summaries,
            panel_label="(b)",
            title="Performance versus model size",
            cost="parameters",
        ),
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


def _manifest(
    output_dir: Path,
    outputs: list[Path],
    summaries: dict[str, FamilySummary],
) -> dict[str, Any]:
    k0 = summaries["structured_k0"]
    gru = summaries["gru_n64"]
    return {
        "schema_version": 1,
        "figure": "4",
        "sources": {
            "results/principal_sweep_v1_statistics.json": {
                "sha256": EXPECTED_STATISTICS_SHA256,
            },
            "results/principal_structured_optimisation_v4.json": {
                "sha256": EXPECTED_OPTIMISATION_SHA256,
            },
        },
        "selected_values": {
            "blackout_milliseconds": BLACKOUT_MILLISECONDS,
            "seed_count_per_family": len(SEEDS),
            "families": {
                family_id: {
                    "save_rate": item.save_rate,
                    "latency_median_microseconds": item.latency_microseconds,
                    "parameters": item.parameters,
                }
                for family_id, item in summaries.items()
            },
            "structured_k0_vs_gru_n64": {
                "save_rate_difference_percentage_points": 100
                * (k0.save_rate - gru.save_rate),
                "parameter_reduction_fraction": 1 - k0.parameters / gru.parameters,
                "latency_reduction_fraction": 1
                - k0.latency_microseconds / gru.latency_microseconds,
            },
        },
        "files": {
            path.relative_to(output_dir).as_posix(): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in outputs
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--formats",
        default="svg,pdf,png",
        help="comma-separated output formats chosen from svg,pdf,png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = tuple(part.strip() for part in args.formats.split(",") if part.strip())
    if not formats or any(value not in {"svg", "pdf", "png"} for value in formats):
        raise ValueError("formats must be chosen from svg,pdf,png")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, _, summaries = load_summaries()
    outputs: list[Path] = []
    for filename, content in render_svgs(summaries).items():
        svg_path = args.output_dir / filename
        svg_path.write_text(content, encoding="utf-8")
        if "svg" in formats:
            outputs.append(svg_path)
        for output_format in ("pdf", "png"):
            if output_format in formats:
                outputs.append(_convert_svg(svg_path, output_format))
        if "svg" not in formats:
            svg_path.unlink()
    manifest = _manifest(args.output_dir, outputs, summaries)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
