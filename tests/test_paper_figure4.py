from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "plot_paper_figure4.py"
STATISTICS_SHA256 = (
    "55ad1f50399e54487728ade10e6605a993064369a80ec48c0674704f2f378902"
)
OPTIMISATION_SHA256 = (
    "a1a5add1c5a5e08b40bec2331e35b9fc9aea140119cd9c0c88a84bec9f3f8c85"
)


def _generate(tmp_path: Path) -> dict[str, object]:
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
    return json.loads(Path(completed.stdout.strip()).read_text(encoding="utf-8"))


def _svg_text(path: Path) -> set[str]:
    root = ET.parse(path).getroot()
    return {
        "".join(element.itertext())
        for element in root.iter()
        if element.tag.endswith("text")
    }


def test_figure4_is_bound_to_frozen_performance_and_efficiency_results(
    tmp_path: Path,
) -> None:
    manifest = _generate(tmp_path)
    assert manifest["sources"]["results/principal_sweep_v1_statistics.json"][
        "sha256"
    ] == STATISTICS_SHA256
    assert manifest["sources"]["results/principal_structured_optimisation_v4.json"][
        "sha256"
    ] == OPTIMISATION_SHA256
    assert manifest["selected_values"]["blackout_milliseconds"] == 400
    assert manifest["selected_values"]["seed_count_per_family"] == 5
    expected = {
        "figure4a_latency_frontier.svg",
        "figure4b_parameter_frontier.svg",
    }
    assert set(manifest["files"]) == expected
    for filename in expected:
        path = tmp_path / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][
            filename
        ]["sha256"]


def test_figure4_reports_the_predeclared_k0_gru_comparison(tmp_path: Path) -> None:
    manifest = _generate(tmp_path)
    selected = manifest["selected_values"]["families"]
    assert selected["structured_k0"] == {
        "save_rate": 0.9831111111111112,
        "latency_median_microseconds": 23.03043345,
        "parameters": 12002,
    }
    assert selected["gru_n64"] == {
        "save_rate": 0.9777777777777777,
        "latency_median_microseconds": 40.05892,
        "parameters": 28898,
    }
    comparison = manifest["selected_values"]["structured_k0_vs_gru_n64"]
    assert round(comparison["parameter_reduction_fraction"] * 100, 1) == 58.5
    assert round(comparison["latency_reduction_fraction"] * 100, 1) == 42.5
    assert round(comparison["save_rate_difference_percentage_points"], 1) == 0.5

    latency_text = _svg_text(tmp_path / "figure4a_latency_frontier.svg")
    parameter_text = _svg_text(tmp_path / "figure4b_parameter_frontier.svg")
    family_labels = {
        "Feed-forward",
        "10-step stack",
        "k=0",
        "k=1",
        "k=2",
        "k=4",
        "GRU-64",
    }
    assert family_labels <= latency_text
    assert family_labels <= parameter_text
    assert "98.3%, 23.0 μs" in latency_text
    assert "97.8%, 40.1 μs" in latency_text
    assert "98.3%, 12,002" in parameter_text
    assert "97.8%, 28,898" in parameter_text
    assert "Legend" not in latency_text | parameter_text
