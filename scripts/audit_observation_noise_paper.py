#!/usr/bin/env python3
"""Check the noise table and numerical prose against the saved analysis."""

import argparse
import json
from pathlib import Path


def audit(paper: Path, result: Path):
    source = paper.read_text()
    data = json.loads(result.read_text())
    section = source.split(r"\subsection{Sensitivity to puck-position noise}", 1)[1].split(r"\section{Discussion", 1)[0]
    setup = source.split(r"\paragraph{Supplementary noise setup.}\label{sec:noise-setup}", 1)[1].split("\n% ", 1)[0]
    abstract = source.split(r"\begin{abstract}", 1)[1].split("\n% ", 1)[0]
    conclusion = source.split(r"\section{Conclusion}", 1)[1].split("\n% ", 1)[0]
    all_rates = data["groups"]["all"]["by_blackout_ms"]
    checks = 0
    for label, family in (("$k=0$", "structured_k0"), ("$k=4$", "structured_k4"), ("GRU-64", "gru_n64"), ("Teacher", "teacher")):
        row = next(line for line in section.splitlines() if line.startswith(label + " &"))
        actual = [float(x.strip().removesuffix(r"\\").strip()) for x in row.split("&")[1:]]
        expected = [float(f'{all_rates[b][n]["policies"][family]["save_rate_percent"]:.1f}')
                    for b in ("0", "400") for n in ("0", "1", "5")]
        assert actual == expected, (label, actual, expected)
        checks += len(actual)
    cell = all_rates["400"]["5"]
    for family in ("structured_k0", "gru_n64"):
        value = cell["policies"][family]["save_rate_percent"]
        assert f"{value:.1f}" + r"\%" in section
        checks += 1
    contrast = cell["paired_policy_contrasts"]["structured_k0_minus_gru_n64"]
    assert f'{contrast["estimate_points"]:.1f} percentage points' in section
    checks += 1
    support = data["groups"]["support"]["by_blackout_ms"]["400"]["5"]["paired_policy_contrasts"]["structured_k0_minus_gru_n64"]
    for item in (contrast, cell["noise_minus_clean"]["structured_k0"], cell["noise_minus_clean"]["gru_n64"], support):
        lo, hi = item["pointwise_95_interval_points"]
        assert f"[{lo:.1f},{hi:.1f}]" in section
        checks += 2
    assert f'{cell["noise_minus_clean"]["structured_k0"]["estimate_points"]:+.1f}' in section
    assert f'{cell["noise_minus_clean"]["gru_n64"]["estimate_points"]:+.1f}' in section
    assert f'{support["estimate_points"]:.1f}' in section
    checks += 3
    alias = data["groups"]["alias"]["by_blackout_ms"]["400"]["5"]["policies"]
    assert alias["structured_k0"]["save_rate_percent"] == 100
    assert f'{alias["gru_n64"]["save_rate_percent"]:.1f}' + r"\%" in section
    checks += 2
    for phrase in (r"0, 1 and 5\,mm", "225 fresh shots", "90 alias pairs and 45 support shots", r"0 and 400\,ms", "all five existing seeds", r"21\,600 episodes"):
        assert phrase in setup, phrase
        checks += 1
    rank_contrasts = (
        cell["paired_policy_contrasts"]["structured_k0_minus_structured_k4"],
        data["groups"]["support"]["by_blackout_ms"]["0"]["5"]["paired_policy_contrasts"]["structured_k0_minus_structured_k4"],
    )
    for item in rank_contrasts:
        lo, hi = item["pointwise_95_interval_points"]
        assert f'{item["estimate_points"]:.1f}$ points' in section or f'{item["estimate_points"]:.1f} points' in section
        assert f"[{lo:.1f},{hi:.1f}]" in section
        checks += 3
    for summary in (abstract, conclusion):
        for family in ("structured_k0", "gru_n64"):
            assert f'{cell["policies"][family]["save_rate_percent"]:.1f}' + r"\%" in summary
            checks += 1
        assert r"5\,mm" in summary and r"400\,ms" in summary
        checks += 2
    lo, hi = contrast["pointwise_95_interval_points"]
    assert f'{contrast["estimate_points"]:.1f} points' in abstract
    assert f"[{lo:.1f},{hi:.1f}]" in abstract
    assert "pointwise" in abstract
    checks += 3
    assert data["episode_rows"] == 21600
    return checks + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--result", type=Path, default=Path("results/observation_noise_v1.json"))
    args = parser.parse_args()
    print(json.dumps({"status": "passed", "checks": audit(args.paper, args.result)}))
