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

