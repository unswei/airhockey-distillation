#!/usr/bin/env python3
"""Print the pinned upstream MuJoCo interface as machine-readable JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from air_hockey_challenge.framework import AirHockeyChallengeWrapper

from drl_air_hockey.dreamer.__main__ import _make_env


def _shape(value) -> list[int]:
    return [int(item) for item in value.shape]


def main() -> None:
    make_env = _make_env(
        env_id="tournament",
        sim="mujoco",
        agent="spacer",
        interpolation_order=-1,
        render=False,
        logdir=Path("/tmp/airhockey-interface-audit"),
        self_play=False,
        self_play_save_model_every_n_episodes=None,
        self_play_max_opponent_models=None,
        self_play_opponent_models_path=None,
    )
    env = make_env()
    try:
        original_reset = AirHockeyChallengeWrapper._drl_original_reset
        raw_reset_observation = original_reset(env)
        reset_observation = env.reset()
        processed_action = np.zeros(env.action_space.shape, dtype=np.float32)
        next_observation, reward, done, info = env.step(processed_action)

        env_info = env.env_info
        report = {
            "environment": {
                "name": env.env_name,
                "class": type(env.base_env).__name__,
                "dt_seconds": float(env_info["dt"]),
                "control_frequency_hz": int(
                    env_info["robot"]["control_frequency"]
                ),
                "table": {
                    key: float(value) for key, value in env_info["table"].items()
                },
                "n_agents": int(env_info["n_agents"]),
            },
            "raw_challenge_interface": {
                "per_agent_observation_space_shape": _shape(
                    env.base_env.info.observation_space
                ),
                "combined_reset_observation_shape": list(
                    raw_reset_observation.shape
                ),
                "low_level_joint_action_space_shape": _shape(
                    env.base_env.info.action_space
                ),
                "per_agent_action_shapes": [
                    list(map(int, shape)) for shape in env.base_env.action_shape
                ],
                "puck_position_ids": list(map(int, env_info["puck_pos_ids"])),
                "puck_velocity_ids": list(map(int, env_info["puck_vel_ids"])),
                "joint_position_ids": list(map(int, env_info["joint_pos_ids"])),
                "joint_velocity_ids": list(map(int, env_info["joint_vel_ids"])),
                "opponent_end_effector_ids": list(
                    map(int, env_info["opponent_ee_ids"])
                ),
            },
            "drl_policy_interface": {
                "observation_shape": _shape(env.observation_space),
                "action_shape": _shape(env.action_space),
                "reset_observation_shape": list(reset_observation.shape),
                "step_observation_shape": list(next_observation.shape),
                "action_semantics": [
                    "target_x",
                    "target_y",
                    "stiffness_x",
                    "stiffness_y",
                    "damping_x",
                    "damping_y",
                ],
                "observation_semantics": [
                    "joint_position[7]",
                    "joint_velocity[7]",
                    "end_effector_xy[2]",
                    "puck_position_xy[2]",
                    "puck_velocity_xy[2]",
                ],
                "end_effector_workspace_xy": env._agent_1.ee_table_minmax.tolist(),
                "puck_workspace_xy": env._agent_1.puck_table_minmax.tolist(),
                "stiffness_xy_range": list(env._agent_1.osc_stiffness_xy_range),
                "damping_xy_range": list(env._agent_1.osc_damping_xy_range),
            },
            "one_step_smoke": {
                "reward": float(reward),
                "done": bool(done),
                "info_keys": sorted(info),
            },
        }
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    main()
