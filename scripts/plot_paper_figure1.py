#!/usr/bin/env python3
"""Generate separate panels and the paper-width composition for Figure 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from paper_figure_style import PALETTE, SvgCanvas


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts" / "paper" / "figure1"
MEMORY_RESULT = REPOSITORY_ROOT / "results" / "stage_b_memory_validation_v3.json"
ABLATION_RESULT = REPOSITORY_ROOT / "results" / "causal_memory_ablation.json"
TASK_CONFIG = REPOSITORY_ROOT / "configs" / "env" / "direct_launch_v3.yaml"
PROTOCOL_CONFIG = (
    REPOSITORY_ROOT / "configs" / "experiments" / "principal_sweep_v1.yaml"
)


@dataclass(frozen=True)
class FigureEvidence:
    teacher_rate_20: float
    visible_rate_20: float
    teacher_advantage_20: float
    teacher_advantage_ci95: tuple[float, float]
    recurrent_rate_20: float
    reset_rate_20: float
    reset_drop_20: float
    reset_drop_ci95: tuple[float, float]
    no_blackout_recurrent_rate: float
    no_blackout_reset_rate: float
    paired_shots: int
    blackout_start_step: int
    policy_step_seconds: float
    core_blackout_steps: tuple[int, ...]
    extrapolation_blackout_steps: tuple[int, ...]

    @property
    def step_milliseconds(self) -> float:
        return 1000.0 * self.policy_step_seconds

    @property
    def blackout_start_milliseconds(self) -> float:
        return self.blackout_start_step * self.step_milliseconds


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def load_evidence() -> FigureEvidence:
    memory = _load_json(MEMORY_RESULT)
    ablation = _load_json(ABLATION_RESULT)
    task = _load_yaml(TASK_CONFIG)
    protocol = _load_yaml(PROTOCOL_CONFIG)

    memory_20 = memory["by_blackout_steps"]["20"]
    ablation_20 = ablation["by_blackout_steps"]["20"]
    ablation_0 = ablation["by_blackout_steps"]["0"]
    aliasing = task["sampling"]["observation_aliasing"]
    test = protocol["evaluation"]["test"]

    evidence = FigureEvidence(
        teacher_rate_20=float(memory_20["teacher_save_rate"]),
        visible_rate_20=float(memory_20["feed_forward_save_rate"]),
        teacher_advantage_20=float(memory_20["paired_teacher_advantage"]),
        teacher_advantage_ci95=tuple(
            float(value) for value in memory_20["paired_teacher_advantage_ci95"]
        ),
        recurrent_rate_20=float(ablation_20["recurrent_save_rate"]),
        reset_rate_20=float(ablation_20["reset_at_blackout_save_rate"]),
        reset_drop_20=float(ablation_20["paired_save_rate_drop"]),
        reset_drop_ci95=tuple(
            float(value) for value in ablation_20["paired_save_rate_drop_ci95"]
        ),
        no_blackout_recurrent_rate=float(ablation_0["recurrent_save_rate"]),
        no_blackout_reset_rate=float(ablation_0["reset_at_blackout_save_rate"]),
        paired_shots=int(memory_20["episodes"]),
        blackout_start_step=int(aliasing["blackout_start_observation_step"]),
        policy_step_seconds=float(aliasing["policy_step_seconds"]),
        core_blackout_steps=tuple(int(value) for value in test["core_blackout_steps"]),
        extrapolation_blackout_steps=tuple(
            int(value) for value in test["extrapolation_blackout_steps"]
        ),
    )
    _validate_evidence(evidence)
    return evidence


def _validate_evidence(evidence: FigureEvidence) -> None:
    rates = (
        evidence.teacher_rate_20,
        evidence.visible_rate_20,
        evidence.recurrent_rate_20,
        evidence.reset_rate_20,
        evidence.no_blackout_recurrent_rate,
        evidence.no_blackout_reset_rate,
    )
    if any(value < 0.0 or value > 1.0 for value in rates):
        raise ValueError("save rates must lie in [0, 1]")
    if evidence.paired_shots != 225:
        raise ValueError("Figure 1 expects the frozen 225-shot paired evaluation")
    if evidence.core_blackout_steps != (0, 5, 10, 15, 20):
        raise ValueError("unexpected principal core blackout schedule")
    if evidence.extrapolation_blackout_steps != (25,):
        raise ValueError("unexpected principal extrapolation blackout schedule")
    if evidence.blackout_start_step != 5 or evidence.policy_step_seconds != 0.02:
        raise ValueError("unexpected direct-launch timing contract")
    if abs(evidence.no_blackout_recurrent_rate - evidence.no_blackout_reset_rate) > 0:
        raise ValueError("the no-blackout reset control is no longer identical")


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _pp(value: float) -> str:
    return f"{100.0 * value:.1f} pp"


def _panel_heading(canvas: SvgCanvas, label: str, title: str, width: float) -> None:
    canvas.text(28, 36, label, size=22, weight=700)
    canvas.text(68, 36, title, size=20, weight=700)
    canvas.line(28, 54, width - 28, 54, stroke=PALETTE["grid"], stroke_width=1.2)


def draw_task_panel(canvas: SvgCanvas) -> None:
    """Draw the aliased direct-launch task geometry."""

    _panel_heading(canvas, "(a)", "Direct-launch defence", 720)
    canvas.text(
        68,
        64,
        "Paired shots meet when tracking disappears, then require opposite defences.",
        size=12.5,
        fill=PALETTE["muted"],
    )

    table_x, table_y, table_w, table_h = 54.0, 92.0, 612.0, 326.0
    centre_y = table_y + table_h / 2.0
    rendezvous_x = 390.0
    goal_x = table_x + 7.0
    canvas.rect(
        table_x,
        table_y,
        table_w,
        table_h,
        fill=PALETTE["paper"],
        stroke=PALETTE["ink"],
        stroke_width=2.0,
        rx=12,
    )
    canvas.line(
        table_x + table_w / 2,
        table_y,
        table_x + table_w / 2,
        table_y + table_h,
        stroke=PALETTE["grid"],
        stroke_width=1.2,
        dash="6 6",
    )
    canvas.circle(
        table_x + table_w / 2,
        centre_y,
        33,
        stroke=PALETTE["grid"],
        stroke_width=1.2,
    )

    # The tracking mask begins at the nominal rendezvous.
    canvas.rect(
        table_x + 1,
        table_y + 1,
        rendezvous_x - table_x - 1,
        table_h - 2,
        fill=PALETTE["light_grid"],
        opacity=0.92,
        rx=11,
    )
    canvas.text(
        220,
        116,
        "tracking unavailable",
        size=13,
        fill=PALETTE["muted"],
        weight=600,
        anchor="middle",
    )

    # Defending goal and posts.
    goal_half = 44.0
    canvas.line(goal_x, centre_y - goal_half, goal_x, centre_y + goal_half, stroke=PALETTE["ink"], stroke_width=5)
    canvas.circle(goal_x, centre_y - goal_half, 6, fill=PALETTE["ink"])
    canvas.circle(goal_x, centre_y + goal_half, 6, fill=PALETTE["ink"])
    canvas.text(72, centre_y + 72, "defending goal", size=12, weight=600)

    # Paired visible histories and the two hidden futures.
    upper_start = (610.0, centre_y - 17.0)
    lower_start = (610.0, centre_y + 17.0)
    rendezvous = (rendezvous_x, centre_y)
    upper_target = (goal_x + 17.0, centre_y - 34.0)
    lower_target = (goal_x + 17.0, centre_y + 34.0)
    canvas.polyline(
        (upper_start, (505, centre_y - 12), rendezvous),
        stroke=PALETTE["blue"],
        stroke_width=4.0,
        marker_end="arrow-blue",
    )
    canvas.polyline(
        (lower_start, (505, centre_y + 12), rendezvous),
        stroke=PALETTE["green"],
        stroke_width=4.0,
        marker_end="arrow-green",
    )
    canvas.polyline(
        (rendezvous, (252, centre_y - 14), upper_target),
        stroke=PALETTE["blue"],
        stroke_width=4.0,
        dash="10 7",
        marker_end="arrow-blue",
    )
    canvas.polyline(
        (rendezvous, (252, centre_y + 14), lower_target),
        stroke=PALETTE["green"],
        stroke_width=4.0,
        dash="4 6",
        marker_end="arrow-green",
    )
    for point, colour in ((upper_start, PALETTE["blue"]), (lower_start, PALETTE["green"])):
        canvas.circle(*point, 8, fill=colour, stroke=PALETTE["paper"], stroke_width=2)
    canvas.circle(*rendezvous, 9, fill=PALETTE["ink"], stroke=PALETTE["paper"], stroke_width=2)

    # Mallet at the fixed centre and two required future positions.
    mallet_x = 150.0
    canvas.circle(mallet_x, centre_y, 16, fill=PALETTE["sky"], stroke=PALETTE["blue"], stroke_width=2.5)
    canvas.circle(mallet_x, centre_y - 34, 16, fill=PALETTE["paper"], stroke=PALETTE["blue"], stroke_width=2.5, opacity=0.9)
    canvas.circle(mallet_x, centre_y + 34, 16, fill=PALETTE["paper"], stroke=PALETTE["green"], stroke_width=2.5, opacity=0.9)
    canvas.line(mallet_x + 23, centre_y - 3, mallet_x + 23, centre_y - 27, stroke=PALETTE["blue"], stroke_width=1.8, marker_end="arrow-blue")
    canvas.line(mallet_x - 23, centre_y + 3, mallet_x - 23, centre_y + 27, stroke=PALETTE["green"], stroke_width=1.8, marker_end="arrow-green")

    canvas.line(rendezvous_x, centre_y + 13, rendezvous_x, centre_y + 62, stroke=PALETTE["ink"], stroke_width=1.3)
    canvas.text(rendezvous_x, centre_y + 81, "same masked observation", size=12.5, weight=600, anchor="middle")
    canvas.text(558, 132, "visible histories", size=12.5, weight=600, anchor="middle")
    canvas.text(219, 394, "different hidden futures and mallet positions", size=12.5, weight=600, anchor="middle")

    # Compact redundant legend: colour and line pattern both distinguish paths.
    canvas.line(86, 448, 126, 448, stroke=PALETTE["blue"], stroke_width=4)
    canvas.text(136, 453, "visible puck history", size=12.5)
    canvas.line(300, 448, 340, 448, stroke=PALETTE["blue"], stroke_width=4, dash="10 7")
    canvas.text(350, 453, "hidden puck path", size=12.5)
    canvas.rect(520, 438, 25, 18, fill=PALETTE["light_grid"], stroke=PALETTE["grid"], stroke_width=0.8)
    canvas.text(555, 453, "masked interval", size=12.5)


def draw_timeline_panel(canvas: SvgCanvas, evidence: FigureEvidence) -> None:
    """Draw the observation and action timing contract."""

    _panel_heading(canvas, "(b)", "Observation and control timing", 720)
    start_ms = evidence.blackout_start_milliseconds
    max_blackout_ms = max(evidence.extrapolation_blackout_steps) * evidence.step_milliseconds
    end_ms = start_ms + max_blackout_ms
    x0, x1 = 150.0, 675.0
    timeline_width = x1 - x0

    def x_for(milliseconds: float) -> float:
        return x0 + timeline_width * milliseconds / end_ms

    # Tested blackout lengths are measured from the shared onset.
    canvas.text(28, 91, "Tested blackout duration (ms)", size=12.5, fill=PALETTE["muted"], weight=600)
    onset_x = x_for(start_ms)
    canvas.line(onset_x, 120, onset_x, 400, stroke=PALETTE["ink"], stroke_width=1.4, dash="5 5")
    all_blackouts = evidence.core_blackout_steps + evidence.extrapolation_blackout_steps
    for steps in all_blackouts:
        duration_ms = steps * evidence.step_milliseconds
        endpoint = x_for(start_ms + duration_ms)
        colour = PALETTE["vermillion"] if steps == 20 else PALETTE["muted"]
        stroke_width = 2.2 if steps == 20 else 1.1
        canvas.line(endpoint, 104, endpoint, 123, stroke=colour, stroke_width=stroke_width)
        canvas.text(endpoint, 96, f"{duration_ms:.0f}", size=11.5, fill=colour, weight=700 if steps == 20 else 400, anchor="middle")
    canvas.text(onset_x + 8, 139, "blackout onset", size=11.5, weight=700)

    rows = (
        (150.0, "Puck observation"),
        (245.0, "Defender action"),
        (340.0, "Recurrent state"),
    )
    for y, label in rows:
        canvas.text(132, y + 24, label, size=13.5, weight=600, anchor="end")

    principal_end_x = x_for(start_ms + 20 * evidence.step_milliseconds)
    extrapolation_end_x = x_for(end_ms)

    # Observation row.
    canvas.rect(x0, 150, onset_x - x0, 48, fill=PALETTE["sky"], opacity=0.42, stroke=PALETTE["blue"], stroke_width=1.2)
    canvas.rect(onset_x, 150, principal_end_x - onset_x, 48, fill=PALETTE["light_grid"], stroke=PALETTE["muted"], stroke_width=1.2)
    canvas.rect(principal_end_x, 150, extrapolation_end_x - principal_end_x, 48, fill=PALETTE["light_grid"], stroke=PALETTE["muted"], stroke_width=1.2, dash="4 4", opacity=0.7)
    canvas.text((x0 + onset_x) / 2, 180, "visible", size=12.5, weight=600, anchor="middle")
    canvas.text((onset_x + principal_end_x) / 2, 180, "puck position set to mask value", size=12.5, weight=600, anchor="middle")
    canvas.text((principal_end_x + extrapolation_end_x) / 2, 180, "500 ms", size=11.5, fill=PALETTE["muted"], anchor="middle")

    # Action row.
    canvas.rect(x0, 245, onset_x - x0, 48, fill=PALETTE["light_grid"], stroke=PALETTE["muted"], stroke_width=1.2)
    canvas.rect(onset_x, 245, extrapolation_end_x - onset_x, 48, fill=PALETTE["orange"], opacity=0.22, stroke=PALETTE["orange"], stroke_width=1.2)
    canvas.text((x0 + onset_x) / 2, 275, "fixed", size=12.5, weight=600, anchor="middle")
    canvas.text((onset_x + extrapolation_end_x) / 2, 275, "2-D mallet command", size=12.5, weight=600, anchor="middle")

    # Recurrent-state row.
    state_y = 364.0
    canvas.line(x0, state_y, extrapolation_end_x, state_y, stroke=PALETTE["blue"], stroke_width=4.0, marker_end="arrow-blue")
    for milliseconds in range(0, int(end_ms) + 1, 100):
        canvas.circle(x_for(float(milliseconds)), state_y, 5.2, fill=PALETTE["paper"], stroke=PALETTE["blue"], stroke_width=2)
    canvas.text((onset_x + extrapolation_end_x) / 2, 352, "state carried and updated through the blackout", size=12.5, weight=600, anchor="middle", fill=PALETTE["blue"])

    canvas.line(principal_end_x, 120, principal_end_x, 401, stroke=PALETTE["vermillion"], stroke_width=1.8, dash="7 5")
    canvas.text(430, 421, "400 ms principal comparison", size=12.3, weight=700, fill=PALETTE["vermillion"], anchor="middle")
    canvas.text(630, 421, "500 ms extrapolation", size=11.8, fill=PALETTE["muted"], anchor="middle")

    canvas.text(
        28,
        463,
        "During blackout, the puck-position entries are set to the fixed mask value.",
        size=12.1,
        fill=PALETTE["muted"],
    )


def draw_causal_panel(canvas: SvgCanvas, evidence: FigureEvidence) -> None:
    """Draw paired evidence that blackout performance depends on memory."""

    _panel_heading(canvas, "(c)", "Blackout performance depends on carried memory", 1180)
    canvas.text(
        68,
        64,
        "Save rate on 225 paired shots at the predeclared 400 ms comparison.",
        size=12.5,
        fill=PALETTE["muted"],
    )

    x0, x1 = 250.0, 1125.0
    y_rows = (158.0, 276.0)

    def x_for(rate: float) -> float:
        return x0 + (x1 - x0) * rate

    # Grid and axis.
    for rate in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = x_for(rate)
        canvas.line(x, 105, x, 340, stroke=PALETTE["light_grid"], stroke_width=1.2)
        canvas.text(x, 365, f"{100 * rate:.0f}", size=12, fill=PALETTE["muted"], anchor="middle")
    canvas.line(x0, 340, x1, 340, stroke=PALETTE["ink"], stroke_width=1.3)
    canvas.text((x0 + x1) / 2, 393, "save rate (%)", size=13.5, weight=600, anchor="middle")

    comparisons = (
        (
            y_rows[0],
            "Policy comparison",
            "Teacher",
            evidence.teacher_rate_20,
            "Visible-only policy",
            evidence.visible_rate_20,
            evidence.teacher_advantage_20,
            evidence.teacher_advantage_ci95,
        ),
        (
            y_rows[1],
            "Causal state reset",
            "Normal state",
            evidence.recurrent_rate_20,
            "Reset at onset",
            evidence.reset_rate_20,
            evidence.reset_drop_20,
            evidence.reset_drop_ci95,
        ),
    )
    for y, row_title, full_label, full_rate, ablated_label, ablated_rate, difference, interval in comparisons:
        canvas.text(28, y - 12, row_title, size=14, weight=700)
        canvas.text(28, y + 14, "same shots", size=11.5, fill=PALETTE["muted"])
        full_x, ablated_x = x_for(full_rate), x_for(ablated_rate)
        canvas.line(ablated_x, y, full_x, y, stroke=PALETTE["grid"], stroke_width=5)
        canvas.marker(full_x, y, shape="circle", colour=PALETTE["blue"])
        canvas.marker(ablated_x, y, shape="square", colour=PALETTE["orange"])
        canvas.text(full_x, y - 19, f"{full_label}  {_pct(full_rate)}", size=12.5, weight=700, anchor="middle", fill=PALETTE["blue"])
        canvas.text(ablated_x, y + 28, f"{ablated_label}  {_pct(ablated_rate)}", size=12.5, weight=700, anchor="middle", fill=PALETTE["orange"])
        ci = f"[{100 * interval[0]:.1f}, {100 * interval[1]:.1f}]"
        canvas.text(
            (full_x + ablated_x) / 2,
            y - 19,
            f"{_pp(difference)}  95% interval {ci}",
            size=12.3,
            weight=600,
            anchor="middle",
        )

    canvas.rect(28, 420, 1124, 44, fill=PALETTE["light_grid"], stroke=PALETTE["grid"], stroke_width=1, rx=4)
    canvas.marker(54, 442, shape="diamond", colour=PALETTE["green"])
    canvas.text(
        73,
        447,
        f"No-blackout control: normal and reset conditions are identical ({_pct(evidence.no_blackout_recurrent_rate)}).",
        size=12.8,
        weight=600,
    )
    canvas.text(
        1128,
        447,
        "paired bootstrap intervals",
        size=11.8,
        fill=PALETTE["muted"],
        anchor="end",
    )


def draw_task_panel_compact(canvas: SvgCanvas) -> None:
    """Single-row version of panel (a), with text sized for paper width."""

    canvas.text(18, 38, "(a)", size=30, weight=700)
    canvas.text(70, 38, "Direct-launch task", size=28, weight=700)
    canvas.line(18, 55, 542, 55, stroke=PALETTE["grid"], stroke_width=1.5)
    canvas.text(24, 86, "Paired histories meet at blackout onset", size=24, weight=700)

    x, y, width, height = 24.0, 106.0, 512.0, 310.0
    centre_y = y + height / 2
    onset_x = 310.0
    canvas.rect(x, y, width, height, fill=PALETTE["paper"], stroke=PALETTE["ink"], stroke_width=2.4, rx=10)
    canvas.rect(x + 1, y + 1, onset_x - x - 1, height - 2, fill=PALETTE["light_grid"], opacity=0.92, rx=9)
    canvas.text(164, 136, "tracking masked", size=23, fill=PALETTE["muted"], weight=600, anchor="middle")
    canvas.text(432, 136, "visible history", size=23, weight=600, anchor="middle")

    goal_x = x + 8
    canvas.line(goal_x, centre_y - 42, goal_x, centre_y + 42, stroke=PALETTE["ink"], stroke_width=5)
    canvas.circle(goal_x, centre_y - 42, 6, fill=PALETTE["ink"])
    canvas.circle(goal_x, centre_y + 42, 6, fill=PALETTE["ink"])
    canvas.text(44, centre_y + 71, "goal", size=23, weight=600)

    upper_start = (500.0, centre_y - 17)
    lower_start = (500.0, centre_y + 17)
    rendezvous = (onset_x, centre_y)
    canvas.polyline((upper_start, (407, centre_y - 10), rendezvous), stroke=PALETTE["blue"], stroke_width=4, marker_end="arrow-blue")
    canvas.polyline((lower_start, (407, centre_y + 10), rendezvous), stroke=PALETTE["green"], stroke_width=4, marker_end="arrow-green")
    canvas.polyline((rendezvous, (190, centre_y - 18), (goal_x + 15, centre_y - 34)), stroke=PALETTE["blue"], stroke_width=4, dash="10 7", marker_end="arrow-blue")
    canvas.polyline((rendezvous, (190, centre_y + 18), (goal_x + 15, centre_y + 34)), stroke=PALETTE["green"], stroke_width=4, dash="4 6", marker_end="arrow-green")
    canvas.circle(*upper_start, 8, fill=PALETTE["blue"], stroke=PALETTE["paper"], stroke_width=2)
    canvas.circle(*lower_start, 8, fill=PALETTE["green"], stroke=PALETTE["paper"], stroke_width=2)
    canvas.circle(*rendezvous, 9, fill=PALETTE["ink"], stroke=PALETTE["paper"], stroke_width=2)

    mallet_x = 100.0
    canvas.circle(mallet_x, centre_y, 17, fill=PALETTE["sky"], stroke=PALETTE["blue"], stroke_width=2.5)
    canvas.circle(mallet_x, centre_y - 34, 17, fill=PALETTE["paper"], stroke=PALETTE["blue"], stroke_width=2.5)
    canvas.circle(mallet_x, centre_y + 34, 17, fill=PALETTE["paper"], stroke=PALETTE["green"], stroke_width=2.5)
    canvas.text(onset_x, centre_y + 76, "same masked input", size=23, weight=700, anchor="middle")
    canvas.text(270, 447, "solid: visible", size=23, fill=PALETTE["blue"], weight=600, anchor="end")
    canvas.text(290, 447, "·", size=21, fill=PALETTE["muted"], anchor="middle")
    canvas.text(310, 447, "dashed: hidden", size=23, fill=PALETTE["muted"], weight=600)
    canvas.text(280, 482, "opposite futures require opposite defences", size=23, weight=700, anchor="middle")


def draw_timeline_panel_compact(canvas: SvgCanvas, evidence: FigureEvidence) -> None:
    """Single-row version of panel (b)."""

    canvas.text(18, 38, "(b)", size=30, weight=700)
    canvas.text(70, 38, "Blackout intervention", size=28, weight=700)
    canvas.line(18, 55, 562, 55, stroke=PALETTE["grid"], stroke_width=1.5)

    start_ms = evidence.blackout_start_milliseconds
    max_ms = start_ms + max(evidence.extrapolation_blackout_steps) * evidence.step_milliseconds
    x0, x1 = 170.0, 552.0

    def x_for(milliseconds: float) -> float:
        return x0 + (x1 - x0) * milliseconds / max_ms

    onset_x = x_for(start_ms)
    principal_x = x_for(start_ms + 20 * evidence.step_milliseconds)
    canvas.text(18, 89, "Blackout (ms)", size=23, weight=600, fill=PALETTE["muted"])
    for steps in evidence.core_blackout_steps + evidence.extrapolation_blackout_steps:
        duration_ms = steps * evidence.step_milliseconds
        endpoint = x_for(start_ms + duration_ms)
        colour = PALETTE["vermillion"] if steps == 20 else PALETTE["muted"]
        canvas.text(endpoint, 91, f"{duration_ms:.0f}", size=23, weight=700 if steps == 20 else 400, fill=colour, anchor="middle")
        canvas.line(endpoint, 99, endpoint, 116, stroke=colour, stroke_width=2 if steps == 20 else 1.2)

    canvas.line(onset_x, 112, onset_x, 410, stroke=PALETTE["ink"], stroke_width=1.5, dash="5 5")
    canvas.line(principal_x, 112, principal_x, 410, stroke=PALETTE["vermillion"], stroke_width=2.2, dash="7 5")
    canvas.text(onset_x + 8, 135, "onset", size=22, weight=700)

    rows = ((146.0, "Puck input"), (252.0, "Action"), (358.0, "Memory"))
    for y, label in rows:
        canvas.text(154, y + 31, label, size=24, weight=700, anchor="end")

    canvas.rect(x0, 146, onset_x - x0, 58, fill=PALETTE["sky"], opacity=0.42, stroke=PALETTE["blue"], stroke_width=1.3)
    canvas.rect(onset_x, 146, principal_x - onset_x, 58, fill=PALETTE["light_grid"], stroke=PALETTE["muted"], stroke_width=1.3)
    canvas.rect(principal_x, 146, x1 - principal_x, 58, fill=PALETTE["light_grid"], stroke=PALETTE["muted"], stroke_width=1.3, dash="4 4", opacity=0.7)
    canvas.text((x0 + onset_x) / 2, 181, "visible", size=23, weight=700, anchor="middle")
    canvas.text((onset_x + principal_x) / 2, 181, "masked", size=24, weight=700, anchor="middle")

    canvas.rect(x0, 252, onset_x - x0, 58, fill=PALETTE["light_grid"], stroke=PALETTE["muted"], stroke_width=1.3)
    canvas.rect(onset_x, 252, x1 - onset_x, 58, fill=PALETTE["orange"], opacity=0.22, stroke=PALETTE["orange"], stroke_width=1.3)
    canvas.text((x0 + onset_x) / 2, 287, "fixed", size=23, weight=700, anchor="middle")
    canvas.text((onset_x + x1) / 2, 287, "2-D command", size=24, weight=700, anchor="middle")

    state_y = 387.0
    canvas.line(x0, state_y, x1, state_y, stroke=PALETTE["blue"], stroke_width=4, marker_end="arrow-blue")
    for milliseconds in range(0, int(max_ms) + 1, 100):
        canvas.circle(x_for(float(milliseconds)), state_y, 5.5, fill=PALETTE["paper"], stroke=PALETTE["blue"], stroke_width=2)
    canvas.text((onset_x + x1) / 2, 370, "state carried through blackout", size=23, weight=700, fill=PALETTE["blue"], anchor="middle")
    canvas.text(330, 444, "400 ms: principal comparison", size=23, weight=700, fill=PALETTE["vermillion"], anchor="middle")
    canvas.text(330, 482, "500 ms: extrapolation", size=23, fill=PALETTE["muted"], anchor="middle")


def draw_causal_panel_compact(canvas: SvgCanvas, evidence: FigureEvidence) -> None:
    """Single-row version of panel (c)."""

    canvas.text(18, 38, "(c)", size=30, weight=700)
    canvas.text(70, 38, "Memory is causally used", size=28, weight=700)
    canvas.line(18, 55, 642, 55, stroke=PALETTE["grid"], stroke_width=1.5)
    canvas.text(18, 86, "225 paired shots · 400 ms blackout", size=23, weight=600, fill=PALETTE["muted"])

    x0, x1 = 170.0, 610.0

    def x_for(rate: float) -> float:
        return x0 + (x1 - x0) * rate

    for rate in (0.0, 0.5, 1.0):
        x = x_for(rate)
        canvas.line(x, 116, x, 400, stroke=PALETTE["light_grid"], stroke_width=1.2)
        canvas.text(x, 429, f"{100 * rate:.0f}", size=23, fill=PALETTE["muted"], anchor="middle")
    canvas.line(x0, 400, x1, 400, stroke=PALETTE["ink"], stroke_width=1.4)
    canvas.text((x0 + x1) / 2, 459, "save rate (%)", size=24, weight=700, anchor="middle")

    rows = (
        (190.0, "Teacher", evidence.teacher_rate_20, "Visible-only", evidence.visible_rate_20, evidence.teacher_advantage_20, evidence.teacher_advantage_ci95),
        (335.0, "Normal state", evidence.recurrent_rate_20, "Reset at onset", evidence.reset_rate_20, evidence.reset_drop_20, evidence.reset_drop_ci95),
    )
    row_titles = ("Task comparison", "Causal reset")
    for row_title, (y, full_label, full_rate, reduced_label, reduced_rate, difference, interval) in zip(row_titles, rows, strict=True):
        canvas.text(18, y - 17, row_title, size=24, weight=700)
        difference_word = "gap" if row_title == "Task comparison" else "drop"
        canvas.text(18, y + 12, f"{difference_word} {_pp(difference)}", size=23, weight=700)
        canvas.text(18, y + 37, f"95%: [{100 * interval[0]:.1f}, {100 * interval[1]:.1f}]", size=22, fill=PALETTE["muted"])
        full_x, reduced_x = x_for(full_rate), x_for(reduced_rate)
        canvas.line(reduced_x, y, full_x, y, stroke=PALETTE["grid"], stroke_width=5)
        canvas.marker(full_x, y, shape="circle", colour=PALETTE["blue"])
        canvas.marker(reduced_x, y, shape="square", colour=PALETTE["orange"])
        canvas.text(full_x, y - 22, f"{full_label} {_pct(full_rate)}", size=23, weight=700, fill=PALETTE["blue"], anchor="end")
        canvas.text(reduced_x, y + 32, f"{reduced_label} {_pct(reduced_rate)}", size=23, weight=700, fill=PALETTE["orange"], anchor="middle")

    canvas.marker(28, 495, shape="diamond", colour=PALETTE["green"])
    canvas.text(49, 501, f"No-blackout reset control: identical ({_pct(evidence.no_blackout_recurrent_rate)})", size=23, weight=700)


def _standalone_panel(
    width: float,
    height: float,
    *,
    title: str,
    description: str,
    draw: Any,
) -> str:
    canvas = SvgCanvas(width, height, title=title, description=description)
    draw(canvas)
    return canvas.render()


def render_svgs(evidence: FigureEvidence) -> dict[str, str]:
    panel_a = _standalone_panel(
        720,
        480,
        title="Figure 1a: direct-launch defence task",
        description=(
            "Top-down schematic of paired shots whose observed histories meet at "
            "blackout onset but whose hidden futures require opposite defences."
        ),
        draw=draw_task_panel,
    )
    panel_b = _standalone_panel(
        720,
        480,
        title="Figure 1b: observation and control timing",
        description=(
            "Timeline showing five fixed visible steps, variable puck-observation "
            "blackout, two-dimensional control and carried recurrent state."
        ),
        draw=lambda canvas: draw_timeline_panel(canvas, evidence),
    )
    panel_c = _standalone_panel(
        1180,
        490,
        title="Figure 1c: causal evidence for carried memory",
        description=(
            "Paired save-rate comparisons between the teacher and visible-only policy, "
            "and between normal recurrent state and reset-at-blackout state."
        ),
        draw=lambda canvas: draw_causal_panel(canvas, evidence),
    )

    combined = SvgCanvas(
        1800,
        530,
        title="Figure 1: task design and evidence for a memory requirement",
        description=(
            "Single-row three-panel figure combining the direct-launch geometry, "
            "blackout timing and causal paired evidence that carried state supports "
            "blackout performance."
        ),
    )
    combined.group_start("translate(0 0)")
    draw_task_panel_compact(combined)
    combined.group_end()
    combined.line(568, 18, 568, 512, stroke=PALETTE["grid"], stroke_width=1.2)
    combined.group_start("translate(580 0)")
    draw_timeline_panel_compact(combined, evidence)
    combined.group_end()
    combined.line(1156, 18, 1156, 512, stroke=PALETTE["grid"], stroke_width=1.2)
    combined.group_start("translate(1160 0)")
    draw_causal_panel_compact(combined, evidence)
    combined.group_end()

    return {
        "figure1a_task.svg": panel_a,
        "figure1b_timeline.svg": panel_b,
        "figure1c_memory_evidence.svg": panel_c,
        "figure1_combined.svg": combined.render(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _convert_svg(svg_path: Path, output_format: str) -> Path:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError(
            "rsvg-convert is required for PDF/PNG output; request SVG-only output "
            "or install librsvg"
        )
    output_path = svg_path.with_suffix(f".{output_format}")
    command = [converter, "--format", output_format, "--output", str(output_path)]
    if output_format == "png":
        target_width = "2800" if "combined" in svg_path.stem else "2200"
        command.extend(("--width", target_width))
    command.append(str(svg_path))
    subprocess.run(command, check=True)
    return output_path


def generate(output_dir: Path, formats: tuple[str, ...]) -> Path:
    evidence = load_evidence()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for filename, content in render_svgs(evidence).items():
        path = output_dir / filename
        path.write_text(content, encoding="utf-8")
        generated.append(path)
        for output_format in formats:
            if output_format != "svg":
                generated.append(_convert_svg(path, output_format))

    sources = (MEMORY_RESULT, ABLATION_RESULT, TASK_CONFIG, PROTOCOL_CONFIG)
    manifest = {
        "schema_version": 1,
        "figure": "figure1_task_and_memory_requirement",
        "palette": "Okabe-Ito with redundant marker and line encodings",
        "sources": {
            str(path.relative_to(REPOSITORY_ROOT)): {"sha256": _sha256(path)}
            for path in sources
        },
        "selected_values": {
            "paired_shots": evidence.paired_shots,
            "blackout_start_step": evidence.blackout_start_step,
            "policy_step_seconds": evidence.policy_step_seconds,
            "principal_blackout_steps": 20,
            "principal_blackout_milliseconds": 400,
            "teacher_save_rate": evidence.teacher_rate_20,
            "visible_only_save_rate": evidence.visible_rate_20,
            "paired_teacher_advantage": evidence.teacher_advantage_20,
            "paired_teacher_advantage_ci95": evidence.teacher_advantage_ci95,
            "normal_state_save_rate": evidence.recurrent_rate_20,
            "reset_state_save_rate": evidence.reset_rate_20,
            "paired_reset_drop": evidence.reset_drop_20,
            "paired_reset_drop_ci95": evidence.reset_drop_ci95,
            "no_blackout_identical_rate": evidence.no_blackout_recurrent_rate,
        },
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(generated)
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
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
    formats = tuple(part.strip().lower() for part in arguments.formats.split(",") if part.strip())
    if not formats or set(formats) - {"svg", "pdf", "png"}:
        raise SystemExit("--formats must contain one or more of svg,pdf,png")
    manifest_path = generate(arguments.output_dir.resolve(), formats)
    print(manifest_path)


if __name__ == "__main__":
    main()
