from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "plot_paper_figure3.py"
STATISTICS_SHA256 = (
    "55ad1f50399e54487728ade10e6605a993064369a80ec48c0674704f2f378902"
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


def test_figure3_panels_are_bound_to_frozen_principal_statistics(
    tmp_path: Path,
) -> None:
    manifest = _generate(tmp_path)
    source = manifest["source"]["results/principal_sweep_v1_statistics.json"]
    assert source["sha256"] == STATISTICS_SHA256
    assert manifest["bootstrap"]["replicates"] == 10_000
    assert manifest["selected_values"]["blackout_milliseconds"] == [
        0,
        100,
        200,
        300,
        400,
        500,
    ]
    expected = {
        "figure3a_memory_baselines.svg",
        "figure3b_recurrent_comparison.svg",
        "figure3c_400ms_contrasts.svg",
    }
    assert set(manifest["files"]) == expected
    for filename in expected:
        path = tmp_path / filename
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == manifest["files"][filename]["sha256"]


def test_figure3_panels_contain_the_predeclared_comparisons(
    tmp_path: Path,
) -> None:
    _generate(tmp_path)
    panel_a = _svg_text(tmp_path / "figure3a_memory_baselines.svg")
    assert {"Feed-forward", "10-step stack", "Structured k=0", "Teacher"} <= panel_a

    panel_b = _svg_text(tmp_path / "figure3b_recurrent_comparison.svg")
    assert {"Structured k=0", "k=1", "k=2", "k=4", "GRU-64", "Teacher"} <= panel_b

    panel_c = _svg_text(tmp_path / "figure3c_400ms_contrasts.svg")
    assert "+54.0 [42.4, 64.4] pp" in panel_c
    assert "+12.4 [6.4, 19.3] pp" in panel_c
    assert "+0.5 [-1.2, 2.5] pp" in panel_c
    assert "-0.4 [-2.4, 1.1] pp" in panel_c
    assert "+0.4 [-0.6, 1.8] pp" in panel_c
    assert "-0.2 [-1.4, 1.0] pp" in panel_c
