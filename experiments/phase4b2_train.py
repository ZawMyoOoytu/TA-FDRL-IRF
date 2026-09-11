from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agents.sac_agent import SACAgent
from environment.irf_env import IRFConfig, IRFEnvironment


# =============================================================================
# TA-FDRL-IRF
# Phase-4B.2
#
# Polarization-Aware Observation SAC Training
#
# B2-B:
#   Polarization PHY ON
#   Polarization observation OFF
#   State = 100D
#
# B2-C:
#   Polarization PHY ON
#   Polarization observation ON
#   State = 140D
#
# Common:
#   Action = 40D
#   RIS optimization = OFF
#   Adaptive trust = ON
# =============================================================================


CONTROLLED_SEEDS = (
    42,
    123,
    2026,
    4096,
    7777,
    9999,
)


@dataclass(frozen=True)
class ExperimentConfig:
    condition: str
    seed: int

    train_episodes: int = 300
    steps_per_episode: int = 200

    eval_interval: int = 10
    eval_episodes: int = 3
    final_eval_episodes: int = 6

    hidden_dim: int = 256

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    gamma: float = 0.99
    tau: float = 0.005

    buffer_size: int = 100_000
    batch_size: int = 256

    num_users: int = 20
    num_ris_elements: int = 64

    bandwidth_hz: float = 100e6
    carrier_frequency_hz: float = 28e9

    polarization_enabled: bool = True
    polarization_state_enabled: bool = False
    cross_polarization_factor: float = 0.15

    polarization_v_strength: float = 1.0
    polarization_h_strength: float = 1.0

    optimize_ris: bool = False
    fixed_trust: bool = False

    trust_memory: float = 0.90
    trust_learning_rate: float = 0.10
    trust_target_rate_bps: float = 1e7

    trust_service_weight: float = 0.35
    trust_interference_weight: float = 0.20
    trust_queue_weight: float = 0.25
    trust_instability_weight: float = 0.20

    trust_neutral_point: float = 0.50
    trust_floor: float = 0.0

    se_weight: float = 0.45
    ee_weight: float = 0.20
    trust_weight: float = 0.15
    trust_delta_weight: float = 0.10
    interference_penalty: float = 0.05
    power_penalty: float = 0.03
    queue_penalty: float = 0.07

    se_target: float = 0.20
    ee_target: float = 1e6


# =============================================================================
# Reproducibility
# =============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# =============================================================================
# Condition configuration
# =============================================================================

def make_config(condition: str, seed: int, args) -> ExperimentConfig:
    condition = condition.upper()

    if condition not in {"B2-B", "B2-C"}:
        raise ValueError(
            f"Unsupported condition: {condition}. "
            "Expected B2-B or B2-C."
        )

    polarization_observation = condition == "B2-C"

    return ExperimentConfig(
        condition=condition,
        seed=seed,

        train_episodes=args.episodes,
        steps_per_episode=args.steps,

        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        final_eval_episodes=args.final_eval_episodes,

        hidden_dim=args.hidden_dim,

        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,

        gamma=args.gamma,
        tau=args.tau,

        buffer_size=args.buffer_size,
        batch_size=args.batch_size,

        polarization_state_enabled=polarization_observation,
    )


# =============================================================================
# Environment
# =============================================================================

def create_environment(config: ExperimentConfig, seed: int) -> IRFEnvironment:

    cfg = IRFConfig(
        num_users=config.num_users,
        num_ris_elements=config.num_ris_elements,

        bandwidth_hz=config.bandwidth_hz,
        carrier_frequency_hz=config.carrier_frequency_hz,

        polarization_enabled=config.polarization_enabled,
        polarization_state_enabled=config.polarization_state_enabled,
        cross_polarization_factor=config.cross_polarization_factor,

        polarization_v_strength=config.polarization_v_strength,
        polarization_h_strength=config.polarization_h_strength,

        max_power_w=1.0,
        circuit_power_w=0.1,

        noise_figure_db=7.0,
        noise_density_dbm_hz=-174.0,

        max_steps=config.steps_per_episode,

        optimize_ris=config.optimize_ris,
        fixed_trust=config.fixed_trust,

        trust_memory=config.trust_memory,
        trust_learning_rate=config.trust_learning_rate,
        trust_target_rate_bps=config.trust_target_rate_bps,

        trust_service_weight=config.trust_service_weight,
        trust_interference_weight=config.trust_interference_weight,
        trust_queue_weight=config.trust_queue_weight,
        trust_instability_weight=config.trust_instability_weight,

        trust_neutral_point=config.trust_neutral_point,
        trust_floor=config.trust_floor,

        se_weight=config.se_weight,
        ee_weight=config.ee_weight,
        trust_weight=config.trust_weight,
        trust_delta_weight=config.trust_delta_weight,
        interference_penalty=config.interference_penalty,
        power_penalty=config.power_penalty,
        queue_penalty=config.queue_penalty,

        se_target=config.se_target,
        ee_target=config.ee_target,

        seed=seed,
    )

    return IRFEnvironment(cfg)


# =============================================================================
# Environment API compatibility
# =============================================================================

def reset_environment(
    env: IRFEnvironment,
    seed: int | None = None,
) -> np.ndarray:

    result = env.reset(seed=seed) if seed is not None else env.reset()

    if isinstance(result, tuple):
        state = result[0]
    else:
        state = result

    return np.asarray(state, dtype=np.float32)


def step_environment(
    env: IRFEnvironment,
    action: np.ndarray,
):
    result = env.step(action)

    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)

    elif len(result) == 4:
        next_state, reward, done, info = result
        done = bool(done)

    else:
        raise RuntimeError(
            f"Unsupported env.step() return length: {len(result)}"
        )

    next_state = np.asarray(
        next_state,
        dtype=np.float32,
    )

    reward = float(reward)

    if not np.isfinite(reward):
        raise RuntimeError(
            f"Non-finite reward detected: {reward}"
        )

    info = dict(info) if isinstance(info, dict) else {}

    return next_state, reward, done, info


# =============================================================================
# Validation
# =============================================================================

def validate_environment(
    env: IRFEnvironment,
    config: ExperimentConfig,
) -> dict[str, Any]:

    expected_state_dim = (
        config.num_users * 7
        if config.polarization_state_enabled
        else config.num_users * 5
    )

    expected_action_dim = config.num_users * 2

    state = reset_environment(
        env,
        seed=config.seed,
    )

    actual_state_dim = int(state.size)
    actual_action_dim = int(env.action_dim)

    if actual_state_dim != expected_state_dim:
        raise RuntimeError(
            "STATE DIMENSION MISMATCH\n"
            f"Expected: {expected_state_dim}\n"
            f"Actual:   {actual_state_dim}"
        )

    if actual_action_dim != expected_action_dim:
        raise RuntimeError(
            "ACTION DIMENSION MISMATCH\n"
            f"Expected: {expected_action_dim}\n"
            f"Actual:   {actual_action_dim}"
        )

    if not np.all(np.isfinite(state)):
        raise RuntimeError(
            "Initial environment state contains NaN/Inf."
        )

    action = np.zeros(
        actual_action_dim,
        dtype=np.float32,
    )

    _, reward, _, info = step_environment(
        env,
        action,
    )

    required = {
        "spectral_efficiency",
        "energy_efficiency",
        "mean_trust",
        "mean_behavior_score",
        "mean_queue",
        "mean_power",
        "total_power",
        "interference_ratio",
        "mean_pqi",
        "mean_xpi",
    }

    missing = sorted(
        required - set(info.keys())
    )

    if missing:
        raise RuntimeError(
            "Required telemetry missing:\n"
            + "\n".join(
                f"  - {item}"
                for item in missing
            )
        )

    pqi = float(info["mean_pqi"])
    xpi = float(info["mean_xpi"])

    if config.polarization_enabled:

        if not (
            0.0 <= pqi <= 1.0
            and 0.0 <= xpi <= 1.0
        ):
            raise RuntimeError(
                f"Invalid polarization telemetry: "
                f"PQI={pqi}, XPI={xpi}"
            )

        if abs((pqi + xpi) - 1.0) > 1e-3:
            raise RuntimeError(
                "Polarization diagnostic invariant failed: "
                f"PQI + XPI = {pqi + xpi:.8f}"
            )

    return {
        "state_dim": actual_state_dim,
        "action_dim": actual_action_dim,
        "initial_pqi": pqi,
        "initial_xpi": xpi,
        "diagnostic_reward": reward,
    }


# =============================================================================
# Telemetry
# =============================================================================

TELEMETRY_KEYS = [
    "spectral_efficiency",
    "energy_efficiency",
    "mean_trust",
    "mean_behavior_score",
    "mean_queue",
    "mean_power",
    "total_power",
    "interference_ratio",
    "mean_pqi",
    "mean_xpi",
]


def metric(info: dict[str, Any], key: str) -> float:

    if key not in info:
        raise RuntimeError(
            f"Missing telemetry key during training: {key}"
        )

    value = float(info[key])

    if not np.isfinite(value):
        raise RuntimeError(
            f"Non-finite telemetry detected: "
            f"{key}={value}"
        )

    return value


# =============================================================================
# Episode aggregation
# =============================================================================

def aggregate_episode(
    rewards: list[float],
    telemetry: list[dict[str, float]],
) -> dict[str, Any]:

    if not rewards or not telemetry:
        raise RuntimeError(
            "Empty episode result."
        )

    result = {
        "steps": len(rewards),
        "total_reward": float(np.sum(rewards)),
        "per_step_reward": float(np.mean(rewards)),
    }

    for key in TELEMETRY_KEYS:
        values = np.asarray(
            [row[key] for row in telemetry],
            dtype=np.float64,
        )

        result[key] = float(
            np.mean(values)
        )

    result["initial_trust"] = float(
        telemetry[0]["mean_trust"]
    )

    result["final_trust"] = float(
        telemetry[-1]["mean_trust"]
    )

    result["trust_change"] = (
        result["final_trust"]
        - result["initial_trust"]
    )

    result["initial_pqi"] = float(
        telemetry[0]["mean_pqi"]
    )

    result["final_pqi"] = float(
        telemetry[-1]["mean_pqi"]
    )

    result["initial_xpi"] = float(
        telemetry[0]["mean_xpi"]
    )

    result["final_xpi"] = float(
        telemetry[-1]["mean_xpi"]
    )

    return result


# =============================================================================
# Episode execution
# =============================================================================

def run_episode(
    env: IRFEnvironment,
    agent: SACAgent,
    seed: int,
    steps: int,
    training: bool,
) -> tuple[dict[str, Any], int]:

    state = reset_environment(
        env,
        seed=seed,
    )

    if not np.all(np.isfinite(state)):
        raise RuntimeError(
            "Initial state contains NaN/Inf."
        )

    rewards = []
    telemetry = []
    updates = 0

    for _ in range(steps):

        action = agent.select_action(
            state,
            evaluate=not training,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        if action.shape != (env.action_dim,):
            raise RuntimeError(
                "Invalid SAC action shape: "
                f"{action.shape}; "
                f"expected {(env.action_dim,)}"
            )

        if not np.all(np.isfinite(action)):
            raise RuntimeError(
                "SAC produced NaN/Inf action."
            )

        next_state, reward, done, info = step_environment(
            env,
            action,
        )

        if not np.all(np.isfinite(next_state)):
            raise RuntimeError(
                "Environment produced NaN/Inf next_state."
            )

        row = {
            key: metric(info, key)
            for key in TELEMETRY_KEYS
        }

        # Trust must remain physically meaningful.
        if not (
            0.0 <= row["mean_trust"] <= 1.0
        ):
            raise RuntimeError(
                "Trust outside [0,1]: "
                f"{row['mean_trust']}"
            )

        # Polarization invariant.
        if (
            abs(
                row["mean_pqi"]
                + row["mean_xpi"]
                - 1.0
            )
            > 1e-3
        ):
            raise RuntimeError(
                "PQI + XPI invariant violated during training: "
                f"{row['mean_pqi'] + row['mean_xpi']}"
            )

        if training:

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            update_result = agent.update()

            if update_result is not None:
                updates += 1

        rewards.append(reward)
        telemetry.append(row)

        state = next_state

        if done:
            break

    result = aggregate_episode(
        rewards,
        telemetry,
    )

    result["updates"] = updates

    return result, updates


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_agent(
    config: ExperimentConfig,
    agent: SACAgent,
    episodes: int,
    seed_offset: int,
) -> dict[str, Any]:

    results = []

    for i in range(episodes):

        eval_seed = (
            seed_offset
            + i
        )

        env = create_environment(
            config,
            eval_seed,
        )

        result, _ = run_episode(
            env=env,
            agent=agent,
            seed=eval_seed,
            steps=config.steps_per_episode,
            training=False,
        )

        result["seed"] = eval_seed

        results.append(result)

    return aggregate_results(
        results
    )


def aggregate_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:

    if not results:
        raise RuntimeError(
            "No evaluation results."
        )

    numeric_keys = [
        "total_reward",
        "per_step_reward",
        "spectral_efficiency",
        "energy_efficiency",
        "mean_trust",
        "mean_behavior_score",
        "mean_queue",
        "mean_power",
        "total_power",
        "interference_ratio",
        "mean_pqi",
        "mean_xpi",
        "initial_trust",
        "final_trust",
        "trust_change",
        "initial_pqi",
        "final_pqi",
        "initial_xpi",
        "final_xpi",
    ]

    output = {
        "episodes": len(results),
        "per_episode": results,
    }

    for key in numeric_keys:

        values = [
            float(row[key])
            for row in results
        ]

        output[key] = float(
            np.mean(values)
        )

        output[f"{key}_std"] = float(
            np.std(
                values,
                ddof=1,
            )
            if len(values) > 1
            else 0.0
        )

    return output


# =============================================================================
# Sanity diagnostics
# =============================================================================

def diagnostic_flags(
    result: dict[str, Any],
    condition: str,
) -> list[str]:

    warnings = []

    trust = result["final_trust"]
    pqi = result["mean_pqi"]
    xpi = result["mean_xpi"]

    if trust <= 1e-8:
        warnings.append(
            "TRUST_DEGENERATE: final trust approximately zero."
        )

    if (
        pqi <= 1e-8
        and xpi <= 1e-8
    ):
        warnings.append(
            "POLARIZATION_TELEMETRY_DEGENERATE: "
            "both PQI and XPI approximately zero."
        )

    if condition == "B2-C":

        if abs(
            (pqi + xpi) - 1.0
        ) > 1e-3:
            warnings.append(
                "POLARIZATION_INVARIANT_WARNING: "
                "mean PQI + mean XPI != 1."
            )

    return warnings


# =============================================================================
# CSV
# =============================================================================

def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# =============================================================================
# Main training
# =============================================================================

def train(config: ExperimentConfig, output_root: Path, device: str | None):

    set_global_seed(
        config.seed
    )

    run_dir = (
        output_root
        / config.condition
        / f"seed_{config.seed}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint = (
        run_dir
        / "best_sac.pt"
    )

    final_checkpoint = (
        run_dir
        / "final_sac.pt"
    )

    history_csv = (
        run_dir
        / "training_history.csv"
    )

    summary_json = (
        run_dir
        / "training_summary.json"
    )

    config_json = (
        run_dir
        / "config.json"
    )

    print()
    print("=" * 80)
    print("TA-FDRL-IRF | PHASE-4B.2 SAC TRAINING")
    print("=" * 80)
    print()
    print(f"Condition                : {config.condition}")
    print(f"Seed                     : {config.seed}")
    print(f"State dimension          : "
          f"{140 if config.polarization_state_enabled else 100}")
    print("Action dimension         : 40")
    print("Polarization PHY         : ON")
    print(
        "Polarization observation : "
        f"{'ON' if config.polarization_state_enabled else 'OFF'}"
    )
    print("Cross-polarization       : 0.15")
    print("RIS optimization         : OFF")
    print("Adaptive trust           : ON")
    print(f"Training episodes        : {config.train_episodes}")
    print(f"Steps / episode          : {config.steps_per_episode}")
    print()

    # -------------------------------------------------------------------------
    # Environment validation
    # -------------------------------------------------------------------------

    validation_env = create_environment(
        config,
        config.seed,
    )

    validation = validate_environment(
        validation_env,
        config,
    )

    print("Environment validation   : PASSED")
    print(
        f"Validated state          : "
        f"{validation['state_dim']}"
    )
    print(
        f"Validated action         : "
        f"{validation['action_dim']}"
    )
    print(
        f"Initial diagnostic PQI   : "
        f"{validation['initial_pqi']:.6f}"
    )
    print(
        f"Initial diagnostic XPI   : "
        f"{validation['initial_xpi']:.6f}"
    )
    print()

    # -------------------------------------------------------------------------
    # Fresh SAC
    # -------------------------------------------------------------------------

    agent = SACAgent(
        state_dim=validation["state_dim"],
        action_dim=validation["action_dim"],

        hidden_dim=config.hidden_dim,

        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        alpha_lr=config.alpha_lr,

        gamma=config.gamma,
        tau=config.tau,

        buffer_size=config.buffer_size,
        batch_size=config.batch_size,

        device=device,
    )

    print(
        f"SAC device               : {agent.device}"
    )
    print(
        "Checkpoint source        : FRESH"
    )
    print()

    # -------------------------------------------------------------------------
    # Random baseline
    # -------------------------------------------------------------------------

    random_env = create_environment(
        config,
        config.seed + 10_000,
    )

    # Random policy through temporary random actions.
    state = reset_environment(
        random_env,
        seed=config.seed + 10_000,
    )

    random_rewards = []
    random_telemetry = []

    for _ in range(
        config.steps_per_episode
    ):

        action = np.random.uniform(
            -1.0,
            1.0,
            size=random_env.action_dim,
        ).astype(np.float32)

        next_state, reward, done, info = step_environment(
            random_env,
            action,
        )

        random_rewards.append(
            reward
        )

        random_telemetry.append(
            {
                key: metric(info, key)
                for key in TELEMETRY_KEYS
            }
        )

        state = next_state

        if done:
            break

    random_baseline = aggregate_episode(
        random_rewards,
        random_telemetry,
    )

    print("=" * 80)
    print("RANDOM BASELINE")
    print("=" * 80)
    print(
        f"Reward                   : "
        f"{random_baseline['total_reward']:.6f}"
    )
    print(
        f"Per-step reward          : "
        f"{random_baseline['per_step_reward']:.6f}"
    )
    print(
        f"SE                       : "
        f"{random_baseline['spectral_efficiency']:.6f}"
    )
    print(
        f"EE                       : "
        f"{random_baseline['energy_efficiency']:.6e}"
    )
    print(
        f"Final trust              : "
        f"{random_baseline['final_trust']:.6f}"
    )
    print(
        f"PQI                      : "
        f"{random_baseline['mean_pqi']:.6f}"
    )
    print(
        f"XPI                      : "
        f"{random_baseline['mean_xpi']:.6f}"
    )
    print()

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    history = []
    best_eval_reward = -math.inf
    best_episode = -1
    total_updates = 0

    start_time = time.time()

    for episode in range(
        1,
        config.train_episodes + 1,
    ):

        episode_seed = (
            config.seed
            + episode
        )

        env = create_environment(
            config,
            episode_seed,
        )

        result, updates = run_episode(
            env=env,
            agent=agent,
            seed=episode_seed,
            steps=config.steps_per_episode,
            training=True,
        )

        total_updates += updates

        result["episode"] = episode
        result["seed"] = episode_seed

        history.append(
            result
        )

        # -------------------------------------------------------------
        # Evaluation
        # -------------------------------------------------------------

        eval_reward = None

        if (
            episode == 1
            or episode % config.eval_interval == 0
        ):

            evaluation = evaluate_agent(
                config=config,
                agent=agent,
                episodes=config.eval_episodes,
                seed_offset=100_000 + config.seed * 10,
            )

            eval_reward = evaluation[
                "per_step_reward"
            ]

            if eval_reward > best_eval_reward:

                best_eval_reward = (
                    eval_reward
                )

                best_episode = episode

                agent.save(
                    str(best_checkpoint)
                )

                result["best_checkpoint"] = True

            else:
                result["best_checkpoint"] = False

            result["eval_per_step_reward"] = (
                eval_reward
            )

        else:

            result["best_checkpoint"] = False
            result["eval_per_step_reward"] = None

        # -------------------------------------------------------------
        # Console
        # -------------------------------------------------------------

        avg10 = float(
            np.mean(
                [
                    row["total_reward"]
                    for row in history[-10:]
                ]
            )
        )

        eval_text = (
            f"{eval_reward:.6f}"
            if eval_reward is not None
            else "---"
        )

        print(
            f"Episode {episode:03d} | "
            f"Reward {result['total_reward']:9.4f} | "
            f"Avg10 {avg10:9.4f} | "
            f"SE {result['spectral_efficiency']:.5f} | "
            f"EE {result['energy_efficiency']:.3e} | "
            f"Trust {result['final_trust']:.6f} | "
            f"PQI {result['mean_pqi']:.6f} | "
            f"XPI {result['mean_xpi']:.6f} | "
            f"Eval {eval_text}"
            + (
                " ★ BEST"
                if result["best_checkpoint"]
                else ""
            )
        )

    elapsed = (
        time.time()
        - start_time
    )

    # -------------------------------------------------------------------------
    # Final checkpoint
    # -------------------------------------------------------------------------

    agent.save(
        str(final_checkpoint)
    )

    if not best_checkpoint.exists():

        agent.save(
            str(best_checkpoint)
        )

        best_episode = config.train_episodes

        best_eval_reward = float(
            history[-1]["per_step_reward"]
        )

    # -------------------------------------------------------------------------
    # Restore best checkpoint
    # -------------------------------------------------------------------------

    agent.load(
        str(best_checkpoint)
    )

    # -------------------------------------------------------------------------
    # Final evaluation
    # -------------------------------------------------------------------------

    final_evaluation = evaluate_agent(
        config=config,
        agent=agent,
        episodes=config.final_eval_episodes,
        seed_offset=200_000 + config.seed * 10,
    )

    warnings = diagnostic_flags(
        final_evaluation,
        config.condition,
    )

    # -------------------------------------------------------------------------
    # Save history
    # -------------------------------------------------------------------------

    write_csv(
        history_csv,
        history,
    )

    # -------------------------------------------------------------------------
    # Save config
    # -------------------------------------------------------------------------

    with config_json.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            asdict(config),
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    training_rewards = np.asarray(
        [
            row["total_reward"]
            for row in history
        ],
        dtype=np.float64,
    )

    summary = {
        "experiment": "TA-FDRL-IRF Phase-4B.2",
        "condition": config.condition,
        "seed": config.seed,

        "state_dim": validation["state_dim"],
        "action_dim": validation["action_dim"],

        "training_episodes": config.train_episodes,
        "steps_per_episode": config.steps_per_episode,

        "random_baseline": random_baseline,

        "training": {
            "first_10_average_reward": float(
                np.mean(
                    training_rewards[:10]
                )
            ),
            "last_10_average_reward": float(
                np.mean(
                    training_rewards[-10:]
                )
            ),
            "best_training_reward": float(
                np.max(training_rewards)
            ),
            "total_updates": total_updates,
            "elapsed_seconds": elapsed,
        },

        "best_checkpoint": {
            "path": str(best_checkpoint),
            "episode": best_episode,
            "evaluation_per_step_reward": (
                best_eval_reward
            ),
        },

        "final_evaluation": final_evaluation,

        "diagnostic_warnings": warnings,

        "scientific_scope": {
            "polarization_phy": True,
            "polarization_observation": (
                config.polarization_state_enabled
            ),
            "ris_optimization": False,
            "action_space": (
                "20 bandwidth + 20 power = 40D"
            ),
            "interpretation": (
                "Measures whether polarization-aware "
                "observations improve SAC performance "
                "under the same PHY and action space."
            ),
        },

        "training_history": history,
    }

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL BEST-CHECKPOINT EVALUATION")
    print("=" * 80)

    print(
        f"Condition                : "
        f"{config.condition}"
    )

    print(
        f"Seed                     : "
        f"{config.seed}"
    )

    print(
        f"Best episode             : "
        f"{best_episode}"
    )

    print(
        f"Best eval reward         : "
        f"{best_eval_reward:.6f}"
    )

    print(
        f"Final reward             : "
        f"{final_evaluation['total_reward']:.6f}"
    )

    print(
        f"Final step reward        : "
        f"{final_evaluation['per_step_reward']:.6f}"
    )

    print(
        f"Final SE                 : "
        f"{final_evaluation['spectral_efficiency']:.6f}"
    )

    print(
        f"Final EE                 : "
        f"{final_evaluation['energy_efficiency']:.6e}"
    )

    print(
        f"Final trust              : "
        f"{final_evaluation['final_trust']:.6f}"
    )

    print(
        f"Mean PQI                 : "
        f"{final_evaluation['mean_pqi']:.6f}"
    )

    print(
        f"Mean XPI                 : "
        f"{final_evaluation['mean_xpi']:.6f}"
    )

    print(
        f"Elapsed                  : "
        f"{elapsed:.2f} sec"
    )

    if warnings:

        print()
        print("DIAGNOSTIC WARNINGS")

        for warning in warnings:
            print(
                f"  [WARNING] {warning}"
            )

    else:

        print()
        print(
            "Diagnostic status        : PASSED"
        )

    print()
    print(
        f"Results directory        : "
        f"{run_dir}"
    )

    print(
        f"Best checkpoint          : "
        f"{best_checkpoint}"
    )

    print(
        f"Training CSV             : "
        f"{history_csv}"
    )

    print(
        f"Summary JSON             : "
        f"{summary_json}"
    )

    print("=" * 80)

    return summary


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "TA-FDRL-IRF Phase-4B.2 "
            "Polarization-Aware Observation SAC Training"
        )
    )

    parser.add_argument(
        "--condition",
        choices=["B2-B", "B2-C"],
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--final-eval-episodes",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--actor-lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--critic-lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--alpha-lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="results/phase4b2",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    config = make_config(
        args.condition,
        args.seed,
        args,
    )

    train(
        config=config,
        output_root=Path(args.output_root),
        device=args.device,
    )


if __name__ == "__main__":
    main()