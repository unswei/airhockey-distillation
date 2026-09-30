#!/usr/bin/env python3
"""Separate, frozen-policy noise extension; original artefacts stay read-only."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from airhockey_distill.envs import BlackoutSchedule, DefenceRewardTracker, DefendShotTrackingLoss, load_defence_reward
from airhockey_distill.envs.observation_noise import NoisyPuckObservationAdapter, standard_noise
from airhockey_distill.envs.shot_distribution import load_direct_launch_distribution
from airhockey_distill.principal_sweep import load_principal_protocol, sha256_file
from airhockey_distill.students import load_principal_policy

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/observation_noise_v1.yaml"


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256_file(p)
            for directory, pattern in (("src", "*.py"), ("scripts", "*.py"), ("scripts", "*.sh"), ("configs", "*.yaml"), ("native", "*.c"))
            for p in sorted((ROOT / directory).rglob(pattern))}


def setup():
    cfg = yaml.safe_load(CONFIG.read_text())
    protocol_path = ROOT / cfg["principal_protocol"]
    if sha256_file(protocol_path) != cfg["principal_protocol_sha256"]:
        raise ValueError("original principal protocol changed")
    protocol = load_principal_protocol(protocol_path)
    shots = load_direct_launch_distribution(ROOT / cfg["distribution_config"]).generate(cfg["distribution_split"])
    return cfg, protocol, shots


def freeze(args, cfg, shots):
    """Write a single pre-run opening, including every original checkpoint hash."""
    checkpoints = {}
    for family in ["teacher", *cfg["families"]]:
        for seed in ([None] if family == "teacher" else cfg["training_seeds"]):
            name = "teacher" if seed is None else f"{family}-{seed}"
            previous = args.principal_test / ("teacher.json" if seed is None else f"students/{name}.json")
            old = json.loads(previous.read_text())
            path = Path(old["checkpoint"])
            hashed = path / "agent.pkl" if seed is None else path
            if sha256_file(hashed) != old["checkpoint_sha256"]:
                raise ValueError(f"original checkpoint changed: {name}")
            checkpoints[name] = {"path": str(path), "sha256": sha256_file(hashed)}
    opening = {
        "created_at": datetime.now(UTC).isoformat(),
        "classification": cfg["classification"], "config": cfg,
        "config_sha256": sha256_file(CONFIG), "source_sha256": source_hashes(),
        "base_code_commit": args.base_code_commit,
        "container_digest": args.container_digest,
        "shots": [s.as_dict() for s in shots],
        "shot_schedule_sha256": json_hash([s.as_dict() for s in shots]),
        "checkpoints": checkpoints,
        "noise_traces_sha256": {str(s.alias_family_id or s.shot.shot_id): hashlib.sha256(
            standard_noise(str(s.alias_family_id or s.shot.shot_id), cfg["noise_seed"], cfg["timeout_steps"] + 1).astype("<f8").tobytes()
        ).hexdigest() for s in shots},
        "episodes_per_policy": len(shots) * len(cfg["blackout_steps"]) * len(cfg["noise_std_mm"]),
        "analysis_fixed_before_outcomes": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "opening.json").open("x") as stream:
        json.dump(opening, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"opening_sha256": sha256_file(args.output / "opening.json"),
                      "policies": len(checkpoints), "episodes_per_policy": opening["episodes_per_policy"]}), flush=True)


def evaluate(policy, teacher, shots, cfg, protocol):
    shadow = protocol["shadow_labelling"]
    env = DefendShotTrackingLoss(
        reward_tracker=DefenceRewardTracker(load_defence_reward(shadow["reward_config"])),
        action_lock_steps=int(shadow["action_lock_steps"]), timeout_steps=cfg["timeout_steps"],
    )
    rows = []
    geometry = env.table_geometry
    ranges = np.array([geometry.length, geometry.width]) - 2 * geometry.puck_radius
    # Assert the physical-to-normalised conversion against the pinned upstream.
    np.testing.assert_allclose(ranges, np.diff(env.backend._agent.puck_table_minmax, axis=1).ravel(), rtol=0, atol=1e-12)
    try:
        for shot in shots:
            unit = str(shot.alias_family_id or shot.shot.shot_id)
            normals = standard_noise(unit, cfg["noise_seed"], env.timeout_steps + 1)
            for blackout in cfg["blackout_steps"]:
                for sigma in cfg["noise_std_mm"]:
                    adapter = NoisyPuckObservationAdapter(normals, sigma, ranges)
                    env.observation_adapter = adapter
                    env.blackout = BlackoutSchedule(int(shadow["blackout_start_observation_step"]), blackout)
                    observation, _ = env.reset(shot=shot.shot)
                    carry = policy.init_policy(1) if teacher else policy.initial_carry()
                    reward, score, steps, outcome = 0.0, 0.0, 0, "rollout_limit"
                    while steps < env.timeout_steps:
                        if teacher:
                            inputs = {"image": np.asarray([observation], np.float32),
                                      "reward": np.asarray([reward], np.float32),
                                      "is_first": np.asarray([steps == 0]),
                                      "is_last": np.asarray([False]), "is_terminal": np.asarray([False])}
                            carry, actions, _ = policy.policy(carry, inputs, mode="eval")
                            action = np.clip(np.asarray(actions["action"], np.float32)[0], -1, 1)
                        else:
                            action, carry = policy.act(observation, carry)
                        adapter.observation_step = steps + 1
                        observation, reward, terminated, truncated, info = env.step(action)
                        score += float(reward)
                        steps += 1
                        if terminated or truncated:
                            outcome = str(info["outcome"])
                            break
                    rows.append({"shot_id": shot.shot.shot_id, "alias_family_id": shot.alias_family_id,
                                 "launch_region": shot.launch_region, "target_region": shot.target_region,
                                 "blackout_steps": blackout, "noise_std_mm": sigma,
                                 "outcome": outcome, "score": score, "steps": steps})
                    if len(rows) % 100 == 0:
                        print(json.dumps({"completed_episodes": len(rows)}), flush=True)
    finally:
        env.close()
    return rows, ranges.tolist()


def run(args, cfg, protocol, shots):
    opening_path = args.output / "opening.json"
    opening = json.loads(opening_path.read_text())
    if opening["source_sha256"] != source_hashes() or opening["config_sha256"] != sha256_file(CONFIG):
        raise ValueError("evaluation source differs from frozen opening")
    if opening["shot_schedule_sha256"] != json_hash([s.as_dict() for s in shots]):
        raise ValueError("shot schedule changed")
    teacher = args.family == "teacher"
    name = "teacher" if teacher else f"{args.family}-{args.seed}"
    checkpoint = opening["checkpoints"][name]
    path = Path(checkpoint["path"])
    if sha256_file(path / "agent.pkl" if teacher else path) != checkpoint["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    result_path = args.output / (f"smoke-{name}.json" if args.smoke else f"{name}.json")
    if result_path.exists():
        raise FileExistsError(result_path)
    started = perf_counter()
    if teacher:
        from scripts.collect_principal_shadow_dataset import _load_teacher, _load_mapping
        policy, _, _ = _load_teacher(protocol, _load_mapping(Path(protocol["shadow_labelling"]["teacher_config"])), path, args.output)
    else:
        policy = load_principal_policy(args.family, path)
        meta = policy.metadata
        if (meta["training_stage"] != "final" or meta["training_seed"] != args.seed or
                meta["protocol_sha256"] != cfg["principal_protocol_sha256"]):
            raise ValueError("checkpoint metadata mismatch")
    if args.smoke:
        # Engineering check uses previously opened validation shots, not new shots.
        shots = load_direct_launch_distribution("configs/env/direct_launch_v3.yaml").generate("validation")[:2]
    rows, ranges = evaluate(policy, teacher, shots, cfg, protocol)
    if args.smoke:
        from scripts.evaluate_principal_student import evaluate_policy_on_schedule
        from scripts.evaluate_principal_test_once import evaluate_teacher_on_schedule
        schedule = tuple((s, b) for s in shots for b in cfg["blackout_steps"])
        old, _ = (evaluate_teacher_on_schedule if teacher else evaluate_policy_on_schedule)(policy, schedule, protocol)
        clean = [r for r in rows if r["noise_std_mm"] == 0]
        for expected, actual in zip(old, clean, strict=True):
            for key in ("shot_id", "blackout_steps", "outcome", "score", "steps"):
                if expected[key] != actual[key]:
                    raise ValueError(f"zero-noise regression: {key}")
    result = {"status": "completed", "classification": "engineering_smoke" if args.smoke else cfg["classification"],
              "family_id": args.family, "training_seed": None if teacher else args.seed,
              "opening_sha256": sha256_file(opening_path), "checkpoint_sha256": checkpoint["sha256"],
              "episodes": rows, "puck_coordinate_range_metres": ranges,
              "zero_noise_regression_passed": True if args.smoke else None,
              "elapsed_seconds": perf_counter() - started, "completed_at": datetime.now(UTC).isoformat()}
    with result_path.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"result": result_path.name, "episodes": len(rows), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--principal-test", type=Path)
    parser.add_argument("--base-code-commit")
    parser.add_argument("--container-digest")
    parser.add_argument("--family", choices=["teacher", "structured_k0", "structured_k4", "gru_n64"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg, protocol, shots = setup()
    if args.prepare:
        if not all((args.principal_test, args.base_code_commit, args.container_digest)):
            parser.error("prepare requires principal-test, base-code-commit and container-digest")
        freeze(args, cfg, shots)
    else:
        if args.family is None or (args.family != "teacher" and args.seed not in cfg["training_seeds"]):
            parser.error("evaluation requires a declared family and seed")
        run(args, cfg, protocol, shots)


if __name__ == "__main__":
    main()
