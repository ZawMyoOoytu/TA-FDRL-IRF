from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

# =========================================================
# REPOSITORY ROOT
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import numpy as np
import torch

from agents.sac_agent import SACAgent
from environment.irf_env import (
    IRFConfig,
    IRFEnvironment,
)


# =========================================================
# TA-FDRL-IRF
#
# Phase-4B.1 Controlled SAC Benchmark
#
# Purpose:
#   Evaluate the existing 100-D SAC policy under the
#   Phase-4B.1 polarization-aware PHY environment.
#
# Experimental conditions:
#
#   - Existing SAC checkpoint
#   - NO SAC training
#   - State dimension = 100
#   - Action dimension = 40
#   - Polarization PHY = ON
#   - Polarization state features = OFF
#   - RIS optimization = OFF
#   - Adaptive trust = ON
#   - 6 controlled seeds
#
# Scientific interpretation:
#
#   PQI/XPI are diagnostics only in Phase-4B.1.
#   They are NOT exposed to the SAC policy state.
#
#   Therefore this phase evaluates integration of the
#   existing SAC policy with the new polarization-aware
#   PHY model. It does NOT establish explicit
#   polarization-aware learning.
#
#   Phase-4B.2 will expose PQI/XPI to the state:
#
#       20 users x 7 features = 140 dimensions
#
#   and therefore requires fresh SAC training.
# =========================================================


# =========================================================
# PATHS
# =========================================================

RESULTS_DIR = (
    ROOT_DIR
    / "results"
    / "phase4b1_controlled_sac_benchmark"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CHECKPOINT_PATH = (
    ROOT_DIR
    / "results"
    / "best_sac_irf_phase3_adaptive_trust"
    / "sac_phase3_adaptive_trust.pt"
)


TXT_PATH = (
    RESULTS_DIR
    / "phase4b1_controlled_sac_benchmark.txt"
)

JSON_PATH = (
    RESULTS_DIR
    / "phase4b1_controlled_sac_benchmark.json"
)

CSV_PATH = (
    RESULTS_DIR
    / "phase4b1_controlled_sac_benchmark.csv"
)


# =========================================================
# EXPERIMENT SETTINGS
# =========================================================

CONTROLLED_SEEDS = [
    101,
    202,
    303,
    404,
    505,
    606,
]

STEPS_PER_EPISODE = 200

# SAC architecture
HIDDEN_DIM = 256

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4

GAMMA = 0.99
TAU = 0.005

BUFFER_SIZE = 100_000
BATCH_SIZE = 256

# Numerical reproducibility tolerance
REPRO_TOLERANCE = 1e-12


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_global_seed(seed: int) -> None:
    """
    Set Python, NumPy and PyTorch random seeds.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# ENVIRONMENT
# =========================================================

def create_environment(
    seed: int,
) -> IRFEnvironment:
    """
    Create the exact Phase-4B.1 environment configuration.
    """

    config = IRFConfig(

        # -------------------------------------------------
        # Network
        # -------------------------------------------------

        num_users=20,
        num_ris_elements=64,

        bandwidth_hz=100e6,
        carrier_frequency_hz=28e9,

        # -------------------------------------------------
        # Polarization PHY
        # -------------------------------------------------

        polarization_enabled=True,

        # Important:
        # PQI/XPI are diagnostics only.
        polarization_state_enabled=False,

        cross_polarization_factor=0.15,

        polarization_v_strength=1.0,
        polarization_h_strength=1.0,

        # -------------------------------------------------
        # Power
        # -------------------------------------------------

        max_power_w=1.0,
        circuit_power_w=0.1,

        # -------------------------------------------------
        # Noise
        # -------------------------------------------------

        noise_figure_db=7.0,
        noise_density_dbm_hz=-174.0,

        # -------------------------------------------------
        # Episode
        # -------------------------------------------------

        max_steps=STEPS_PER_EPISODE,

        # -------------------------------------------------
        # Phase-4B.1 control
        # -------------------------------------------------

        optimize_ris=False,
        fixed_trust=False,

        # -------------------------------------------------
        # Adaptive trust
        # -------------------------------------------------

        trust_memory=0.90,
        trust_learning_rate=0.10,
        trust_target_rate_bps=1e7,

        # -------------------------------------------------
        # Trust weights
        # -------------------------------------------------

        trust_service_weight=0.35,
        trust_interference_weight=0.20,
        trust_queue_weight=0.25,
        trust_instability_weight=0.20,

        # -------------------------------------------------
        # Trust reference
        # -------------------------------------------------

        trust_neutral_point=0.50,
        trust_floor=0.0,

        # -------------------------------------------------
        # Reward
        # -------------------------------------------------

        se_weight=0.45,
        ee_weight=0.20,
        trust_weight=0.15,
        trust_delta_weight=0.10,

        interference_penalty=0.05,
        power_penalty=0.03,
        queue_penalty=0.07,

        # -------------------------------------------------
        # Reward normalization
        # -------------------------------------------------

        se_target=0.20,
        ee_target=1e6,

        # -------------------------------------------------
        # Seed
        # -------------------------------------------------

        seed=seed,
    )

    return IRFEnvironment(config)


# =========================================================
# ENVIRONMENT VALIDATION
# =========================================================

def validate_phase4b1_environment(
    env: IRFEnvironment,
) -> None:
    """
    Strictly validate the Phase-4B.1 experimental conditions.
    """

    if env.state_dim != 100:
        raise RuntimeError(
            "Phase-4B.1 requires state_dim=100, "
            f"but received {env.state_dim}."
        )

    if env.action_dim != 40:
        raise RuntimeError(
            "Phase-4B.1 requires action_dim=40, "
            f"but received {env.action_dim}."
        )

    if env.num_users != 20:
        raise RuntimeError(
            "Phase-4B.1 requires 20 users, "
            f"but received {env.num_users}."
        )

    if env.num_ris != 64:
        raise RuntimeError(
            "Phase-4B.1 requires 64 RIS elements, "
            f"but received {env.num_ris}."
        )

    if not env.cfg.polarization_enabled:
        raise RuntimeError(
            "Polarization PHY must be enabled."
        )

    if env.cfg.polarization_state_enabled:
        raise RuntimeError(
            "Polarization state features must be "
            "disabled in Phase-4B.1."
        )

    if env.cfg.optimize_ris:
        raise RuntimeError(
            "RIS optimization must be disabled "
            "in Phase-4B.1."
        )

    if env.cfg.fixed_trust:
        raise RuntimeError(
            "Adaptive trust must be enabled "
            "in Phase-4B.1."
        )

    if env.cfg.cross_polarization_factor != 0.15:
        raise RuntimeError(
            "Phase-4B.1 requires "
            "cross_polarization_factor=0.15."
        )


# =========================================================
# SAC AGENT
# =========================================================

def create_agent(
    env: IRFEnvironment,
) -> SACAgent:
    """
    Create SAC with the same architecture used by the
    existing Phase-3 checkpoint.
    """

    agent = SACAgent(

        state_dim=env.state_dim,

        action_dim=env.action_dim,

        hidden_dim=HIDDEN_DIM,

        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        alpha_lr=ALPHA_LR,

        gamma=GAMMA,
        tau=TAU,

        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
    )

    return agent


# =========================================================
# METRIC CONTAINER
# =========================================================

def empty_metrics() -> dict:
    """
    Initialize all benchmark metrics.
    """

    return {
        # Reward
        "episode_reward": 0.0,

        # PHY / performance
        "spectral_efficiency": 0.0,
        "energy_efficiency": 0.0,
        "total_rate": 0.0,
        "mean_sinr": 0.0,

        # Trust
        "initial_trust": 0.0,
        "final_trust": 0.0,
        "trust_change": 0.0,
        "mean_trust": 0.0,
        "mean_trust_delta": 0.0,

        # Behavior
        "mean_behavior_score": 0.0,

        # Queue / power / interference
        "mean_queue": 0.0,
        "mean_power": 0.0,
        "interference_ratio": 0.0,

        # Polarization
        "mean_pqi": 0.0,
        "min_pqi": 1.0,
        "max_pqi": 0.0,

        "mean_xpi": 0.0,
        "min_xpi": 1.0,
        "max_xpi": 0.0,

        # Reward statistics
        "step_reward_mean": 0.0,
        "step_reward_std": 0.0,

        # Episode length
        "steps": 0,
    }


# =========================================================
# METRIC UPDATE
# =========================================================

def accumulate_info(
    metrics: dict,
    info: dict,
) -> tuple[float, float]:
    """
    Accumulate one environment step.

    Returns:
        pqi, xpi
    """

    metrics["spectral_efficiency"] += float(
        info["spectral_efficiency"]
    )

    metrics["energy_efficiency"] += float(
        info["energy_efficiency"]
    )

    metrics["total_rate"] += float(
        info["rate"]
    )

    metrics["mean_sinr"] += float(
        info["sinr"]
    )

    metrics["mean_trust"] += float(
        info["trust"]
    )

    metrics["mean_trust_delta"] += float(
        info["trust_delta"]
    )

    metrics["mean_behavior_score"] += float(
        info["behavior_score"]
    )

    metrics["mean_queue"] += float(
        info["queue"]
    )

    metrics["mean_power"] += float(
        info["power"]
    )

    metrics["interference_ratio"] += float(
        info["interference"]
    )

    pqi = float(
        info["mean_polarization_quality"]
    )

    xpi = float(
        info["mean_cross_polarization_ratio"]
    )

    return pqi, xpi


# =========================================================
# AGGREGATE METRICS
# =========================================================

def finalize_metrics(
    metrics: dict,
    env: IRFEnvironment,
    initial_trust: float,
    step_rewards: list[float],
    pqi_values: list[float],
    xpi_values: list[float],
    min_pqi_values: list[float],
    max_pqi_values: list[float],
    min_xpi_values: list[float],
    max_xpi_values: list[float],
) -> dict:
    """
    Convert accumulated step metrics into episode metrics.
    """

    steps = len(step_rewards)

    metrics["steps"] = steps

    if steps > 0:

        averaged_metrics = [
            "spectral_efficiency",
            "energy_efficiency",
            "total_rate",
            "mean_sinr",
            "mean_trust",
            "mean_trust_delta",
            "mean_behavior_score",
            "mean_queue",
            "mean_power",
            "interference_ratio",
        ]

        for key in averaged_metrics:
            metrics[key] /= steps

        metrics["mean_pqi"] = float(
            np.mean(pqi_values)
        )

        metrics["mean_xpi"] = float(
            np.mean(xpi_values)
        )

        # Global extrema across the episode's per-step
        # diagnostic extrema.
        metrics["min_pqi"] = float(
            np.min(min_pqi_values)
        )

        metrics["max_pqi"] = float(
            np.max(max_pqi_values)
        )

        metrics["min_xpi"] = float(
            np.min(min_xpi_values)
        )

        metrics["max_xpi"] = float(
            np.max(max_xpi_values)
        )

        metrics["step_reward_mean"] = float(
            np.mean(step_rewards)
        )

        metrics["step_reward_std"] = float(
            np.std(step_rewards)
        )

    else:

        metrics["mean_pqi"] = float("nan")
        metrics["min_pqi"] = float("nan")
        metrics["max_pqi"] = float("nan")

        metrics["mean_xpi"] = float("nan")
        metrics["min_xpi"] = float("nan")
        metrics["max_xpi"] = float("nan")

        metrics["step_reward_mean"] = float("nan")
        metrics["step_reward_std"] = float("nan")

    # Final trust is read from the environment's committed state.
    final_trust = float(
        np.mean(env.trust)
    )

    metrics["initial_trust"] = float(
        initial_trust
    )

    metrics["final_trust"] = final_trust

    metrics["trust_change"] = (
        final_trust
        - initial_trust
    )

    return metrics


# =========================================================
# SAC EVALUATION
# =========================================================

def evaluate_sac(
    seed: int,
    checkpoint_path: Path,
) -> dict:
    """
    Evaluate the existing deterministic SAC policy.

    IMPORTANT:
        No training occurs here.
    """

    set_global_seed(seed)

    env = create_environment(seed)

    validate_phase4b1_environment(env)

    agent = create_agent(env)

    agent.load(
        str(checkpoint_path)
    )

    # -----------------------------------------------------
    # Reset
    # -----------------------------------------------------

    state = env.reset(
        seed=seed
    )

    initial_trust = float(
        np.mean(env.trust)
    )

    metrics = empty_metrics()

    pqi_values = []
    xpi_values = []

    min_pqi_values = []
    max_pqi_values = []

    min_xpi_values = []
    max_xpi_values = []

    step_rewards = []

    # -----------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------

    for _ in range(STEPS_PER_EPISODE):

        action = agent.select_action(
            state,
            evaluate=True,
        )

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        reward = float(reward)

        metrics["episode_reward"] += reward

        step_rewards.append(reward)

        pqi, xpi = accumulate_info(
            metrics,
            info,
        )

        pqi_values.append(pqi)
        xpi_values.append(xpi)

        min_pqi_values.append(
            float(
                info["min_polarization_quality"]
            )
        )

        max_pqi_values.append(
            float(
                info["max_polarization_quality"]
            )
        )

        min_xpi_values.append(
            float(
                info["min_cross_polarization_ratio"]
            )
        )

        max_xpi_values.append(
            float(
                info["max_cross_polarization_ratio"]
            )
        )

        state = next_state

        if terminated or truncated:
            break

    return finalize_metrics(
        metrics=metrics,
        env=env,
        initial_trust=initial_trust,
        step_rewards=step_rewards,
        pqi_values=pqi_values,
        xpi_values=xpi_values,
        min_pqi_values=min_pqi_values,
        max_pqi_values=max_pqi_values,
        min_xpi_values=min_xpi_values,
        max_xpi_values=max_xpi_values,
    )


# =========================================================
# RANDOM BASELINE
# =========================================================

def evaluate_random(
    seed: int,
) -> dict:
    """
    Evaluate a controlled random-action baseline.

    NOTE:
        The random action sequence is NOT identical to the
        SAC action sequence.

        The environment initialization is controlled by the
        same seed, while random actions use an independent
        deterministic RNG stream (seed + 3000).
    """

    set_global_seed(seed)

    env = create_environment(seed)

    validate_phase4b1_environment(env)

    env.reset(
        seed=seed
    )

    initial_trust = float(
        np.mean(env.trust)
    )

    metrics = empty_metrics()

    pqi_values = []
    xpi_values = []

    min_pqi_values = []
    max_pqi_values = []

    min_xpi_values = []
    max_xpi_values = []

    step_rewards = []

    rng = np.random.default_rng(
        seed + 3000
    )

    # -----------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------

    for _ in range(STEPS_PER_EPISODE):

        action = rng.uniform(
            -1.0,
            1.0,
            size=env.action_dim,
        )

        (
            _,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        reward = float(reward)

        metrics["episode_reward"] += reward

        step_rewards.append(reward)

        pqi, xpi = accumulate_info(
            metrics,
            info,
        )

        pqi_values.append(pqi)
        xpi_values.append(xpi)

        min_pqi_values.append(
            float(
                info["min_polarization_quality"]
            )
        )

        max_pqi_values.append(
            float(
                info["max_polarization_quality"]
            )
        )

        min_xpi_values.append(
            float(
                info["min_cross_polarization_ratio"]
            )
        )

        max_xpi_values.append(
            float(
                info["max_cross_polarization_ratio"]
            )
        )

        if terminated or truncated:
            break

    return finalize_metrics(
        metrics=metrics,
        env=env,
        initial_trust=initial_trust,
        step_rewards=step_rewards,
        pqi_values=pqi_values,
        xpi_values=xpi_values,
        min_pqi_values=min_pqi_values,
        max_pqi_values=max_pqi_values,
        min_xpi_values=min_xpi_values,
        max_xpi_values=max_xpi_values,
    )


# =========================================================
# REPRODUCIBILITY CHECK
# =========================================================

def reproducibility_check(
    seed: int,
    checkpoint_path: Path,
) -> dict:
    """
    Run the same deterministic SAC evaluation twice and
    compare the resulting metrics.
    """

    first = evaluate_sac(
        seed,
        checkpoint_path,
    )

    second = evaluate_sac(
        seed,
        checkpoint_path,
    )

    keys = [
        "episode_reward",
        "spectral_efficiency",
        "energy_efficiency",
        "total_rate",
        "mean_sinr",
        "initial_trust",
        "final_trust",
        "trust_change",
        "mean_trust",
        "mean_trust_delta",
        "mean_behavior_score",
        "mean_queue",
        "mean_power",
        "interference_ratio",
        "mean_pqi",
        "min_pqi",
        "max_pqi",
        "mean_xpi",
        "min_xpi",
        "max_xpi",
        "step_reward_mean",
        "step_reward_std",
    ]

    errors = {}

    max_error = 0.0

    for key in keys:

        error = abs(
            float(first[key])
            - float(second[key])
        )

        errors[key] = error

        max_error = max(
            max_error,
            error,
        )

    return {
        "seed": seed,
        "tolerance": REPRO_TOLERANCE,
        "max_absolute_error": max_error,
        "passed": bool(
            max_error <= REPRO_TOLERANCE
        ),
        "errors": errors,
    }


# =========================================================
# SUMMARY
# =========================================================

def summarize(
    records: list[dict],
) -> dict:
    """
    Calculate mean/std/min/max over controlled seeds.
    """

    metric_names = [
        "episode_reward",
        "spectral_efficiency",
        "energy_efficiency",
        "total_rate",
        "mean_sinr",
        "initial_trust",
        "final_trust",
        "trust_change",
        "mean_trust",
        "mean_trust_delta",
        "mean_behavior_score",
        "mean_queue",
        "mean_power",
        "interference_ratio",
        "mean_pqi",
        "min_pqi",
        "max_pqi",
        "mean_xpi",
        "min_xpi",
        "max_xpi",
        "step_reward_mean",
        "step_reward_std",
    ]

    summary = {}

    for metric in metric_names:

        values = np.asarray(
            [
                float(record[metric])
                for record in records
            ],
            dtype=np.float64,
        )

        summary[metric] = {
            "mean": float(
                np.mean(values)
            ),
            "std": float(
                np.std(
                    values,
                    ddof=1,
                )
            ) if len(values) > 1 else 0.0,
            "min": float(
                np.min(values)
            ),
            "max": float(
                np.max(values)
            ),
        }

    return summary


# =========================================================
# PERCENTAGE IMPROVEMENT
# =========================================================

def percentage_improvement(
    sac_value: float,
    baseline_value: float,
) -> float:
    """
    Percentage improvement relative to the baseline.

    Formula:
        (SAC - baseline) / |baseline| * 100
    """

    denominator = abs(
        baseline_value
    )

    if denominator < 1e-12:
        return 0.0

    return (
        (
            sac_value
            - baseline_value
        )
        / denominator
        * 100.0
    )


# =========================================================
# CSV
# =========================================================

def save_csv(
    sac_records: list[dict],
    random_records: list[dict],
) -> None:
    """
    Save per-seed SAC and random metrics.
    """

    rows = []

    for record in sac_records:

        row = {
            "policy": "SAC",
            "seed": record["seed"],
        }

        row.update(
            record["metrics"]
        )

        rows.append(row)

    for record in random_records:

        row = {
            "policy": "RANDOM",
            "seed": record["seed"],
        }

        row.update(
            record["metrics"]
        )

        rows.append(row)

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


# =========================================================
# TEXT REPORT
# =========================================================

def write_report(
    sac_records: list[dict],
    random_records: list[dict],
    sac_summary: dict,
    random_summary: dict,
    reproducibility: list[dict],
) -> None:
    """
    Write the research-grade TXT report.
    """

    sac_reward = sac_summary[
        "episode_reward"
    ]["mean"]

    random_reward = random_summary[
        "episode_reward"
    ]["mean"]

    reward_improvement = (
        percentage_improvement(
            sac_reward,
            random_reward,
        )
    )

    sac_se = sac_summary[
        "spectral_efficiency"
    ]["mean"]

    random_se = random_summary[
        "spectral_efficiency"
    ]["mean"]

    se_improvement = (
        percentage_improvement(
            sac_se,
            random_se,
        )
    )

    sac_ee = sac_summary[
        "energy_efficiency"
    ]["mean"]

    random_ee = random_summary[
        "energy_efficiency"
    ]["mean"]

    ee_improvement = (
        percentage_improvement(
            sac_ee,
            random_ee,
        )
    )

    all_repro_pass = all(
        item["passed"]
        for item in reproducibility
    )

    sac_pqi = sac_summary[
        "mean_pqi"
    ]["mean"]

    sac_xpi = sac_summary[
        "mean_xpi"
    ]["mean"]

    random_pqi = random_summary[
        "mean_pqi"
    ]["mean"]

    random_xpi = random_summary[
        "mean_xpi"
    ]["mean"]

    sac_polarization_sum_error = abs(
        (sac_pqi + sac_xpi) - 1.0
    )

    random_polarization_sum_error = abs(
        (random_pqi + random_xpi) - 1.0
    )

    with TXT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        write = handle.write

        # =================================================
        # HEADER
        # =================================================

        write("=" * 78 + "\n")
        write(
            "TA-FDRL-IRF | PHASE-4B.1 "
            "CONTROLLED SAC BENCHMARK\n"
        )
        write("=" * 78 + "\n\n")

        # =================================================
        # CONFIGURATION
        # =================================================

        write("EXPERIMENT CONFIGURATION\n")
        write("-" * 78 + "\n")

        write("Users:                       20\n")
        write("RIS elements:               64\n")
        write("Bandwidth:                  100 MHz\n")
        write("Carrier frequency:          28 GHz\n")

        write(
            "Polarization PHY:           ENABLED\n"
        )

        write(
            "Polarization state:         DISABLED\n"
        )

        write(
            "Cross-polarization factor:  0.15\n"
        )

        write(
            "RIS optimization:           DISABLED\n"
        )

        write(
            "Adaptive trust:             ENABLED\n"
        )

        write(
            "State dimension:             100\n"
        )

        write(
            "Action dimension:            40\n"
        )

        write(
            "Episode steps:              200\n"
        )

        write(
            "Controlled seeds:           6\n"
        )

        write(
            "SAC training:               NOT PERFORMED\n"
        )

        write(
            "SAC evaluation:             DETERMINISTIC\n"
        )

        write("\n")

        # =================================================
        # CHECKPOINT
        # =================================================

        write("SAC CHECKPOINT\n")
        write("-" * 78 + "\n")

        write(
            f"{CHECKPOINT_PATH}\n"
        )

        write("\n")

        # =================================================
        # SEEDS
        # =================================================

        write("CONTROLLED SEEDS\n")
        write("-" * 78 + "\n")

        write(
            ", ".join(
                str(seed)
                for seed in CONTROLLED_SEEDS
            )
            + "\n\n"
        )

        # =================================================
        # SAC PER SEED
        # =================================================

        write("SAC PER-SEED RESULTS\n")
        write("-" * 78 + "\n")

        for record in sac_records:

            seed = record["seed"]
            metrics = record["metrics"]

            write(
                f"Seed {seed}: "
                f"Reward={metrics['episode_reward']:.9f}, "
                f"SE={metrics['spectral_efficiency']:.9e}, "
                f"EE={metrics['energy_efficiency']:.9e}, "
                f"Trust={metrics['final_trust']:.9f}, "
                f"TrustChange={metrics['trust_change']:.9f}, "
                f"PQI={metrics['mean_pqi']:.9f}, "
                f"XPI={metrics['mean_xpi']:.9f}\n"
            )

        write("\n")

        # =================================================
        # RANDOM PER SEED
        # =================================================

        write(
            "RANDOM BASELINE PER-SEED RESULTS\n"
        )

        write("-" * 78 + "\n")

        for record in random_records:

            seed = record["seed"]
            metrics = record["metrics"]

            write(
                f"Seed {seed}: "
                f"Reward={metrics['episode_reward']:.9f}, "
                f"SE={metrics['spectral_efficiency']:.9e}, "
                f"EE={metrics['energy_efficiency']:.9e}, "
                f"Trust={metrics['final_trust']:.9f}, "
                f"TrustChange={metrics['trust_change']:.9f}, "
                f"PQI={metrics['mean_pqi']:.9f}, "
                f"XPI={metrics['mean_xpi']:.9f}\n"
            )

        write("\n")

        # =================================================
        # SAC SUMMARY
        # =================================================

        write("SAC SUMMARY\n")
        write("-" * 78 + "\n")

        write(
            f"Reward mean:                "
            f"{sac_reward:.9f}\n"
        )

        write(
            f"Reward std:                 "
            f"{sac_summary['episode_reward']['std']:.9f}\n"
        )

        write(
            f"SE mean:                    "
            f"{sac_se:.9e}\n"
        )

        write(
            f"EE mean:                    "
            f"{sac_ee:.9e}\n"
        )

        write(
            f"Final trust mean:           "
            f"{sac_summary['final_trust']['mean']:.9f}\n"
        )

        write(
            f"Trust change mean:          "
            f"{sac_summary['trust_change']['mean']:.9f}\n"
        )

        write(
            f"PQI mean:                   "
            f"{sac_pqi:.9f}\n"
        )

        write(
            f"XPI mean:                   "
            f"{sac_xpi:.9f}\n"
        )

        write("\n")

        # =================================================
        # RANDOM SUMMARY
        # =================================================

        write(
            "RANDOM BASELINE SUMMARY\n"
        )

        write("-" * 78 + "\n")

        write(
            f"Reward mean:                "
            f"{random_reward:.9f}\n"
        )

        write(
            f"Reward std:                 "
            f"{random_summary['episode_reward']['std']:.9f}\n"
        )

        write(
            f"SE mean:                    "
            f"{random_se:.9e}\n"
        )

        write(
            f"EE mean:                    "
            f"{random_ee:.9e}\n"
        )

        write(
            f"Final trust mean:           "
            f"{random_summary['final_trust']['mean']:.9f}\n"
        )

        write(
            f"Trust change mean:          "
            f"{random_summary['trust_change']['mean']:.9f}\n"
        )

        write(
            f"PQI mean:                   "
            f"{random_pqi:.9f}\n"
        )

        write(
            f"XPI mean:                   "
            f"{random_xpi:.9f}\n"
        )

        write("\n")

        # =================================================
        # COMPARISON
        # =================================================

        write("SAC VS RANDOM COMPARISON\n")
        write("-" * 78 + "\n")

        write(
            f"Reward improvement:         "
            f"{reward_improvement:.4f}%\n"
        )

        write(
            f"SE improvement:             "
            f"{se_improvement:.4f}%\n"
        )

        write(
            f"EE improvement:             "
            f"{ee_improvement:.4f}%\n"
        )

        write("\n")

        # =================================================
        # TRUST
        # =================================================

        write("TRUST DYNAMICS\n")
        write("-" * 78 + "\n")

        write(
            f"SAC initial trust mean:     "
            f"{sac_summary['initial_trust']['mean']:.9f}\n"
        )

        write(
            f"SAC final trust mean:       "
            f"{sac_summary['final_trust']['mean']:.9f}\n"
        )

        write(
            f"SAC trust change mean:      "
            f"{sac_summary['trust_change']['mean']:.9f}\n"
        )

        write(
            f"Random initial trust mean:  "
            f"{random_summary['initial_trust']['mean']:.9f}\n"
        )

        write(
            f"Random final trust mean:    "
            f"{random_summary['final_trust']['mean']:.9f}\n"
        )

        write(
            f"Random trust change mean:   "
            f"{random_summary['trust_change']['mean']:.9f}\n"
        )

        write("\n")

        # =================================================
        # POLARIZATION
        # =================================================

        write("POLARIZATION DIAGNOSTICS\n")
        write("-" * 78 + "\n")

        write(
            f"SAC mean PQI:               "
            f"{sac_pqi:.12f}\n"
        )

        write(
            f"SAC mean XPI:               "
            f"{sac_xpi:.12f}\n"
        )

        write(
            f"SAC PQI + XPI:              "
            f"{sac_pqi + sac_xpi:.12f}\n"
        )

        write(
            f"SAC |PQI + XPI - 1|:        "
            f"{sac_polarization_sum_error:.12e}\n"
        )

        write("\n")

        write(
            f"Random mean PQI:            "
            f"{random_pqi:.12f}\n"
        )

        write(
            f"Random mean XPI:            "
            f"{random_xpi:.12f}\n"
        )

        write(
            f"Random PQI + XPI:           "
            f"{random_pqi + random_xpi:.12f}\n"
        )

        write(
            f"Random |PQI + XPI - 1|:     "
            f"{random_polarization_sum_error:.12e}\n"
        )

        write("\n")

        # =================================================
        # REPRODUCIBILITY
        # =================================================

        write("REPRODUCIBILITY\n")
        write("-" * 78 + "\n")

        for item in reproducibility:

            write(
                f"Seed {item['seed']}: "
                f"max_error="
                f"{item['max_absolute_error']:.12e} "
                f"PASS={item['passed']}\n"
            )

        write(
            f"\nOverall reproducibility: "
            f"{'PASS' if all_repro_pass else 'FAIL'}\n"
        )

        write("\n")

        # =================================================
        # SCIENTIFIC INTERPRETATION
        # =================================================

        write("SCIENTIFIC INTERPRETATION\n")
        write("-" * 78 + "\n\n")

        write(
            "Phase-4B.1 evaluates the integration of an existing "
            "100-dimensional SAC policy with a polarization-aware "
            "PHY environment.\n\n"
        )

        write(
            "The polarization-aware channel model is enabled, "
            "while polarization state features remain disabled. "
            "Therefore, the SAC policy continues to operate on "
            "the original 100-dimensional state representation.\n\n"
        )

        write(
            "PQI and XPI are collected as PHY-level diagnostics. "
            "They are not provided to the SAC policy as state "
            "features in this phase.\n\n"
        )

        write(
            "Consequently, Phase-4B.1 does not establish that SAC "
            "has learned explicit polarization-aware control. "
            "Instead, it evaluates whether the existing policy "
            "can operate consistently under the updated "
            "polarization-aware physical-layer model.\n\n"
        )

        write(
            "The SAC-versus-random comparison provides a controlled "
            "system-level performance comparison using the same "
            "environment seeds. The random baseline uses a "
            "separate deterministic action RNG stream and does "
            "not use the SAC action sequence.\n\n"
        )

        write(
            "The polarization diagnostics provide additional "
            "physical-layer observability. PQI represents the "
            "fraction of received polarization power associated "
            "with the desired V-polarized component, while XPI "
            "represents the cross-polarized H component under the "
            "current diagnostic definition.\n\n"
        )

        write(
            "For Phase-4B.2, PQI and XPI can be exposed directly "
            "to the learning state. With 20 users and seven "
            "features per user [SINR, Interference, Queue, Power, "
            "Trust, PQI, XPI], the state dimension becomes 140. "
            "That change is architecturally incompatible with "
            "the current 100-dimensional checkpoint and therefore "
            "requires fresh SAC training.\n\n"
        )

        write(
            "SAC training was intentionally not performed in "
            "Phase-4B.1 so that the experiment remains a controlled "
            "integration benchmark rather than a retraining study.\n\n"
        )

        # =================================================
        # FINAL
        # =================================================

        write("=" * 78 + "\n")
        write(
            "PHASE-4B.1 CONTROLLED SAC BENCHMARK COMPLETE\n"
        )
        write("=" * 78 + "\n")


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    print()
    print("=" * 78)
    print(
        "TA-FDRL-IRF | Phase-4B.1 "
        "Controlled SAC Benchmark"
    )
    print("=" * 78)
    print()

    # =====================================================
    # CHECKPOINT
    # =====================================================

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "SAC checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print(
        f"Results:   {RESULTS_DIR}"
    )

    print()

    # =====================================================
    # ENVIRONMENT CONFIGURATION CHECK
    # =====================================================

    test_env = create_environment(
        CONTROLLED_SEEDS[0]
    )

    validate_phase4b1_environment(
        test_env
    )

    print(
        "Environment configuration"
    )

    print("-" * 50)

    print(
        f"Users:                       "
        f"{test_env.num_users}"
    )

    print(
        f"RIS elements:               "
        f"{test_env.num_ris}"
    )

    print(
        f"State dimension:             "
        f"{test_env.state_dim}"
    )

    print(
        f"Action dimension:            "
        f"{test_env.action_dim}"
    )

    print(
        f"Polarization PHY:            "
        f"{test_env.cfg.polarization_enabled}"
    )

    print(
        f"Polarization state features: "
        f"{test_env.cfg.polarization_state_enabled}"
    )

    print(
        f"Cross-polarization factor:   "
        f"{test_env.cfg.cross_polarization_factor}"
    )

    print(
        f"RIS optimization:            "
        f"{test_env.cfg.optimize_ris}"
    )

    print(
        f"Adaptive trust:              "
        f"{not test_env.cfg.fixed_trust}"
    )

    print()

    print(
        "Phase-4B.1 configuration: PASS"
    )

    print()

    # =====================================================
    # BENCHMARK
    # =====================================================

    sac_records = []
    random_records = []

    print(
        "Running controlled SAC / Random benchmark..."
    )

    print()

    for seed in CONTROLLED_SEEDS:

        # -------------------------------------------------
        # SAC
        # -------------------------------------------------

        print(
            f"[Seed {seed}] SAC evaluation..."
        )

        sac_metrics = evaluate_sac(
            seed,
            CHECKPOINT_PATH,
        )

        sac_records.append(
            {
                "seed": seed,
                "metrics": sac_metrics,
            }
        )

        print(
            f"  SAC reward = "
            f"{sac_metrics['episode_reward']:.9f}"
        )

        print(
            f"  SAC SE     = "
            f"{sac_metrics['spectral_efficiency']:.9e}"
        )

        print(
            f"  SAC EE     = "
            f"{sac_metrics['energy_efficiency']:.9e}"
        )

        print(
            f"  SAC trust  = "
            f"{sac_metrics['final_trust']:.9f}"
        )

        print(
            f"  SAC PQI    = "
            f"{sac_metrics['mean_pqi']:.9f}"
        )

        print(
            f"  SAC XPI    = "
            f"{sac_metrics['mean_xpi']:.9f}"
        )

        # -------------------------------------------------
        # RANDOM
        # -------------------------------------------------

        print(
            f"[Seed {seed}] Random evaluation..."
        )

        random_metrics = evaluate_random(
            seed
        )

        random_records.append(
            {
                "seed": seed,
                "metrics": random_metrics,
            }
        )

        print(
            f"  Random reward = "
            f"{random_metrics['episode_reward']:.9f}"
        )

        print(
            f"  Random SE     = "
            f"{random_metrics['spectral_efficiency']:.9e}"
        )

        print(
            f"  Random EE     = "
            f"{random_metrics['energy_efficiency']:.9e}"
        )

        print(
            f"  Random trust  = "
            f"{random_metrics['final_trust']:.9f}"
        )

        print(
            f"  Random PQI    = "
            f"{random_metrics['mean_pqi']:.9f}"
        )

        print(
            f"  Random XPI    = "
            f"{random_metrics['mean_xpi']:.9f}"
        )

        print()

    # =====================================================
    # SUMMARY
    # =====================================================

    sac_summary = summarize(
        [
            record["metrics"]
            for record in sac_records
        ]
    )

    random_summary = summarize(
        [
            record["metrics"]
            for record in random_records
        ]
    )

    # =====================================================
    # REPRODUCIBILITY
    # =====================================================

    print(
        "Running SAC reproducibility checks..."
    )

    reproducibility = []

    for seed in CONTROLLED_SEEDS:

        result = reproducibility_check(
            seed,
            CHECKPOINT_PATH,
        )

        reproducibility.append(
            result
        )

        print(
            f"  Seed {seed}: "
            f"{'PASS' if result['passed'] else 'FAIL'} "
            f"(max error = "
            f"{result['max_absolute_error']:.3e})"
        )

    print()

    # =====================================================
    # JSON
    # =====================================================

    json_output = {

        "experiment": {

            "name":
                "Phase-4B.1 Controlled SAC Benchmark",

            "phase":
                "4B.1",

            "training_performed":
                False,

            "evaluation_policy":
                "deterministic",

            "state_dim":
                100,

            "action_dim":
                40,

            "num_users":
                20,

            "num_ris_elements":
                64,

            "bandwidth_hz":
                100e6,

            "carrier_frequency_hz":
                28e9,

            "polarization_enabled":
                True,

            "polarization_state_enabled":
                False,

            "cross_polarization_factor":
                0.15,

            "optimize_ris":
                False,

            "fixed_trust":
                False,

            "steps_per_episode":
                STEPS_PER_EPISODE,

            "controlled_seeds":
                CONTROLLED_SEEDS,

            "random_action_seed_offset":
                3000,

            "reproducibility_tolerance":
                REPRO_TOLERANCE,
        },

        "checkpoint":
            str(CHECKPOINT_PATH),

        "sac": {

            "architecture": {
                "state_dim": 100,
                "action_dim": 40,
                "hidden_dim": HIDDEN_DIM,
                "actor_lr": ACTOR_LR,
                "critic_lr": CRITIC_LR,
                "alpha_lr": ALPHA_LR,
                "gamma": GAMMA,
                "tau": TAU,
                "buffer_size": BUFFER_SIZE,
                "batch_size": BATCH_SIZE,
            },

            "per_seed":
                sac_records,

            "summary":
                sac_summary,
        },

        "random_baseline": {

            "per_seed":
                random_records,

            "summary":
                random_summary,
        },

        "reproducibility":
            reproducibility,
    }

    with JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            json_output,
            handle,
            indent=2,
        )

    # =====================================================
    # CSV
    # =====================================================

    save_csv(
        sac_records,
        random_records,
    )

    # =====================================================
    # TXT
    # =====================================================

    write_report(
        sac_records,
        random_records,
        sac_summary,
        random_summary,
        reproducibility,
    )

    # =====================================================
    # CONSOLE SUMMARY
    # =====================================================

    sac_reward = sac_summary[
        "episode_reward"
    ]["mean"]

    random_reward = random_summary[
        "episode_reward"
    ]["mean"]

    reward_improvement = (
        percentage_improvement(
            sac_reward,
            random_reward,
        )
    )

    sac_se = sac_summary[
        "spectral_efficiency"
    ]["mean"]

    random_se = random_summary[
        "spectral_efficiency"
    ]["mean"]

    se_improvement = (
        percentage_improvement(
            sac_se,
            random_se,
        )
    )

    sac_ee = sac_summary[
        "energy_efficiency"
    ]["mean"]

    random_ee = random_summary[
        "energy_efficiency"
    ]["mean"]

    ee_improvement = (
        percentage_improvement(
            sac_ee,
            random_ee,
        )
    )

    all_repro_pass = all(
        item["passed"]
        for item in reproducibility
    )

    print("=" * 78)

    print(
        "PHASE-4B.1 CONTROLLED SAC BENCHMARK COMPLETE"
    )

    print("=" * 78)

    print()

    print(
        f"SAC mean reward:       "
        f"{sac_reward:.9f}"
    )

    print(
        f"Random mean reward:    "
        f"{random_reward:.9f}"
    )

    print(
        f"Reward improvement:    "
        f"{reward_improvement:.4f}%"
    )

    print()

    print(
        f"SAC mean SE:           "
        f"{sac_se:.9e}"
    )

    print(
        f"Random mean SE:        "
        f"{random_se:.9e}"
    )

    print(
        f"SE improvement:        "
        f"{se_improvement:.4f}%"
    )

    print()

    print(
        f"SAC mean EE:           "
        f"{sac_ee:.9e}"
    )

    print(
        f"Random mean EE:        "
        f"{random_ee:.9e}"
    )

    print(
        f"EE improvement:        "
        f"{ee_improvement:.4f}%"
    )

    print()

    print(
        f"SAC final trust:       "
        f"{sac_summary['final_trust']['mean']:.9f}"
    )

    print(
        f"SAC trust change:      "
        f"{sac_summary['trust_change']['mean']:.9f}"
    )

    print()

    print(
        f"SAC mean PQI:          "
        f"{sac_summary['mean_pqi']['mean']:.9f}"
    )

    print(
        f"SAC mean XPI:          "
        f"{sac_summary['mean_xpi']['mean']:.9f}"
    )

    print(
        f"SAC PQI + XPI:         "
        f"{(
            sac_summary['mean_pqi']['mean']
            + sac_summary['mean_xpi']['mean']
        ):.9f}"
    )

    print()

    print(
        "Reproducibility:       "
        f"{'PASS' if all_repro_pass else 'FAIL'}"
    )

    print()

    print(
        f"TXT:  {TXT_PATH}"
    )

    print(
        f"JSON: {JSON_PATH}"
    )

    print(
        f"CSV:  {CSV_PATH}"
    )

    print()

    print("=" * 78)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()