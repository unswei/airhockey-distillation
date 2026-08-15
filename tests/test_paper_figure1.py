from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "plot_paper_figure1.py"


def test_figure1_svg_panels_are_generated_from_frozen_results(tmp_path: Path) -> None:
    completed = subprocess.run(
        (
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(tmp_path),
            "--formats",
            "svg",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    manifest_path = Path(completed.stdout.strip())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["selected_values"]["paired_shots"] == 225
    assert manifest["selected_values"]["principal_blackout_milliseconds"] == 400
    assert manifest["selected_values"]["teacher_save_rate"] == 0.9688888888888889
    assert manifest["selected_values"]["visible_only_save_rate"] == 1 / 3
    assert manifest["selected_values"]["paired_reset_drop"] == 0.4577777777777778

    expected = {
        "figure1a_task.svg",
        "figure1b_timeline.svg",
        "figure1c_memory_evidence.svg",
        "figure1_combined.svg",
    }
    assert set(manifest["files"]) == expected
    for filename in expected:
        path = tmp_path / filename
        root = ET.parse(path).getroot()
        assert root.tag.endswith("svg")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][filename]["sha256"]


def test_figure1_causal_panel_contains_the_reported_comparisons(tmp_path: Path) -> None:
    subprocess.run(
        (
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(tmp_path),
            "--formats",
            "svg",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    causal = (tmp_path / "figure1c_memory_evidence.svg").read_text(encoding="utf-8")
    assert "Teacher  96.9%" in causal
    assert "Visible-only policy  33.3%" in causal
    assert "63.6 pp  95% interval [56.9, 70.2]" in causal
    assert "Normal state  96.0%" in causal
    assert "Reset at onset  50.2%" in causal
    assert "45.8 pp  95% interval [38.7, 52.9]" in causal
    assert "normal and reset conditions are identical (99.6%)" in causal


def test_figure1_alias_paths_continue_straight_through_blackout(tmp_path: Path) -> None:
    subprocess.run(
        (
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(tmp_path),
            "--formats",
            "svg",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    root = ET.parse(tmp_path / "figure1_combined.svg").getroot()
    paths_by_colour: dict[str, list[ET.Element]] = {
        "#E69F00": [],
        "#009E73": [],
    }
    for element in root.iter():
        if not element.tag.endswith("polyline"):
            continue
        colour = element.attrib.get("stroke")
        if colour in paths_by_colour:
            paths_by_colour[colour].append(element)

    for paths in paths_by_colour.values():
        assert len(paths) == 2
        visible, hidden = paths
        assert "stroke-dasharray" not in visible.attrib
        assert hidden.attrib["stroke-dasharray"] == "10 7"
        visible_points = [
            tuple(float(value) for value in point.split(","))
            for point in visible.attrib["points"].split()
        ]
        hidden_points = [
            tuple(float(value) for value in point.split(","))
            for point in hidden.attrib["points"].split()
        ]
        start, rendezvous = visible_points
        assert rendezvous == hidden_points[0]
        target = hidden_points[1]
        incoming = (rendezvous[0] - start[0], rendezvous[1] - start[1])
        outgoing = (target[0] - rendezvous[0], target[1] - rendezvous[1])
        cross_product = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        dot_product = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
        assert abs(cross_product) < 1.0
        assert dot_product > 0.0

    labels = {
        "".join(element.itertext()): float(element.attrib["x"])
        for element in root.iter()
        if element.tag.endswith("text")
        and "".join(element.itertext()) in {"dashed: hidden", "solid: visible"}
    }
    assert labels["dashed: hidden"] < labels["solid: visible"]
