from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

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
# Scientific objective:
#   Determine whether a freshly trained SAC controller can exploit
#   polarization-aware observations (PQI/XPI) when the PHY is polarization
#   aware, while keeping the action space unchanged.
#
# IMPORTANT:
#   This phase is NOT polarization control.
#   RIS optimization remains disabled and action_dim remains 40.
#
# B2-A:
#   Scalar / polarization-unaware observation baseline
#   state_dim = 100
#
# B2-B:
#   Polarization PHY enabled
#   polarization-unaware observation
#   state_dim = 100
#
# B2-C:
#   Polarization PHY enabled
#   polarization-aware observation
#   state_dim = 140
#
# This script trains ONLY B2-C.
# =============================================================================


# =============================================================================
# Global experiment configuration
# =============================================================================

SEED = 42

NUM_TRAIN_EPISODES = 300
STEPS_PER_EPISODE = 200

EVAL_INTERVAL = 10
EVAL_EPISODES = 5
FINAL_EVAL_EPISODES = 10

CONTROLLED_SEEDS = (
    42,
    123,
    2026,
    4096,
    7777,
    9999,
)

NUM_USERS = 20
NUM_RIS_ELEMENTS = 64

BANDWIDTH_HZ = 100e6
CARRIER_FREQUENCY_HZ = 28e9

POLARIZATION_ENABLED = True
POLARIZATION_STATE_ENABLED = True
CROSS_POLARIZATION_FACTOR = 0.15

OPTIMIZE_RIS = False
FIXED_TRUST = False

TRUST_MEMORY = 0.90
TRUST_LEARNING_RATE = 0.10
TRUST_TARGET_RATE_BPS = 1e7

MAX_POWER_W = 1.0
CIRCUIT_POWER_W = 0.1

NOISE_FIGURE_DB = 7.0
NOISE_DENSITY_DBM_HZ = -174.0

# Reward configuration
TRUST_SERVICE_WEIGHT = 0.35
TRUST_INTERFERENCE_WEIGHT = 0.20
TRUST_QUEUE_WEIGHT = 0.25
TRUST_INSTABILITY_WEIGHT = 0.20

TRUST_NEUTRAL_POINT = 0.50
TRUST_FLOOR = 0.0

SE_WEIGHT = 0.45
EE_WEIGHT = 0.20
TRUST_WEIGHT = 0.15
TRUST_DELTA_WEIGHT = 0.10
INTERFERENCE_PENALTY = 0.05
POWER_PENALTY = 0.03
QUEUE_PENALTY = 0.07

SE_TARGET = 0.20
EE_TARGET = 1e6


# =============================================================================
# Paths
# =============================================================================

RESULTS_DIR = os.path.join(
    "results",
    "best_sac_irf_phase4b2_polarization_observation",
)

BEST_MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "sac_phase4b2_polarization_observation.pt",
)

SUMMARY_PATH = os.path.join(
    "results",
    "phase4b2_polarization_observation_summary.txt",
)

JSON_PATH = os.path.join(
    "results",
    "phase4b2_polarization_observation.json",
)

CSV_PATH = os.path.join(
    "results",
    "phase4b2_polarization_observation_episodes.csv",
)

TRAIN_REWARDS_NPY = os.path.join(
    "results",
    "phase4b2_training_rewards.npy",
)

EVAL_REWARDS_NPY = os.path.join(
    "results",
    "phase4b2_evaluation_rewards.npy",
)

EVAL_SE_NPY = os.path.join(
    "results",
    "phase4b2_evaluation_spectral_efficiency.npy",
)

EVAL_EE_NPY = os.path.join(
    "results",
    "phase4b2_evaluation_energy_efficiency.npy",
)


# =============================================================================
# Required telemetry
# =============================================================================

REQUIRED_INFO_KEYS = {
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


# =============================================================================
# Reproducibility
# =============================================================================

def set_global_seed(seed: int) -> None:
    """
    Set Python, NumPy and PyTorch random seeds.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior where supported.
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# =============================================================================
# Utility helpers
# =============================================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convert a value to finite float.

    This function is intentionally conservative:
    non-finite values are replaced by default.
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)

    if not np.isfinite(value):
        return float(default)

    return value


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def ensure_results_dirs() -> None:
    os.makedirs("results", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# Environment construction
# =============================================================================

def create_environment(seed: int = SEED) -> IRFEnvironment:
    """
    Create the exact Phase-4B.2 environment configuration.
    """

    config = IRFConfig(
        num_users=NUM_USERS,
        num_ris_elements=NUM_RIS_ELEMENTS,
        bandwidth_hz=BANDWIDTH_HZ,
        carrier_frequency_hz=CARRIER_FREQUENCY_HZ,

        polarization_enabled=POLARIZATION_ENABLED,
        polarization_state_enabled=POLARIZATION_STATE_ENABLED,
        cross_polarization_factor=CROSS_POLARIZATION_FACTOR,

        max_power_w=MAX_POWER_W,
        circuit_power_w=CIRCUIT_POWER_W,

        noise_figure_db=NOISE_FIGURE_DB,
        noise_density_dbm_hz=NOISE_DENSITY_DBM_HZ,

        max_steps=STEPS_PER_EPISODE,

        optimize_ris=OPTIMIZE_RIS,
        fixed_trust=FIXED_TRUST,

        trust_memory=TRUST_MEMORY,
        trust_learning_rate=TRUST_LEARNING_RATE,
        trust_target_rate_bps=TRUST_TARGET_RATE_BPS,

        trust_service_weight=TRUST_SERVICE_WEIGHT,
        trust_interference_weight=TRUST_INTERFERENCE_WEIGHT,
        trust_queue_weight=TRUST_QUEUE_WEIGHT,
        trust_instability_weight=TRUST_INSTABILITY_WEIGHT,

        trust_neutral_point=TRUST_NEUTRAL_POINT,
        trust_floor=TRUST_FLOOR,

        se_weight=SE_WEIGHT,
        ee_weight=EE_WEIGHT,
        trust_weight=TRUST_WEIGHT,
        trust_delta_weight=TRUST_DELTA_WEIGHT,
        interference_penalty=INTERFERENCE_PENALTY,
        power_penalty=POWER_PENALTY,
        queue_penalty=QUEUE_PENALTY,

        se_target=SE_TARGET,
        ee_target=EE_TARGET,

        seed=seed,
    )

    return IRFEnvironment(config)


# =============================================================================
# Reset compatibility
# =============================================================================

def environment_reset(
    env: IRFEnvironment,
    seed: int | None = None,
) -> np.ndarray:
    """
    Support both:
        state = env.reset(...)
    and:
        state, info = env.reset(...)
    """

    if seed is None:
        result = env.reset()
    else:
        result = env.reset(seed=seed)

    if isinstance(result, tuple):
        state = result[0]
    else:
        state = result

    state = np.asarray(state, dtype=np.float32)

    return state


# =============================================================================
# Step compatibility
# =============================================================================

def environment_step(
    env: IRFEnvironment,
    action: np.ndarray,
) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    """
    Support Gymnasium 5-tuple and legacy 4-tuple step APIs.

    Gymnasium:
        observation, reward, terminated, truncated, info

    Legacy:
        observation, reward, done, info
    """

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

    next_state = np.asarray(next_state, dtype=np.float32)
    reward = safe_float(reward)
    info = dict(info) if isinstance(info, dict) else {}

    return next_state, reward, done, info


# =============================================================================
# Environment validation
# =============================================================================

def validate_environment(
    env: IRFEnvironment,
) -> Dict[str, Any]:
    """
    Validate the expected Phase-4B.2 state/action dimensions and telemetry.
    """

    state = environment_reset(env, seed=SEED)

    state_dim = int(state.shape[0])
    action_dim = int(env.action_dim)

    expected_state_dim = NUM_USERS * 7
    expected_action_dim = NUM_USERS * 2

    if POLARIZATION_STATE_ENABLED:
        if state_dim != expected_state_dim:
            raise RuntimeError(
                "Phase-4B.2 state dimension mismatch: "
                f"expected {expected_state_dim}, got {state_dim}"
            )
    else:
        expected_state_dim = NUM_USERS * 5

        if state_dim != expected_state_dim:
            raise RuntimeError(
                "State dimension mismatch: "
                f"expected {expected_state_dim}, got {state_dim}"
            )

    if action_dim != expected_action_dim:
        raise RuntimeError(
            "Action dimension mismatch: "
            f"expected {expected_action_dim}, got {action_dim}"
        )

    # Validate that one real environment step executes.
    action = np.zeros(action_dim, dtype=np.float32)

    _, _, _, info = environment_step(env, action)

    missing = sorted(
        key
        for key in REQUIRED_INFO_KEYS
        if key not in info
    )

    if missing:
        raise RuntimeError(
            "Required Phase-4B.2 telemetry is missing from env.step() info:\n"
            + "\n".join(f"  - {key}" for key in missing)
        )

    # Polarization-specific sanity check.
    pqi = safe_float(info["mean_pqi"])
    xpi = safe_float(info["mean_xpi"])

    if POLARIZATION_ENABLED:
        if not (0.0 <= pqi <= 1.0):
            raise RuntimeError(
                f"Invalid mean_pqi={pqi}. Expected [0,1]."
            )

        if not (0.0 <= xpi <= 1.0):
            raise RuntimeError(
                f"Invalid mean_xpi={xpi}. Expected [0,1]."
            )

    return {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "info_keys": sorted(info.keys()),
        "mean_pqi": pqi,
        "mean_xpi": xpi,
    }


# =============================================================================
# Telemetry extraction
# =============================================================================

def extract_metrics(
    info: Dict[str, Any],
    env: IRFEnvironment | None = None,
) -> Dict[str, float]:
    """
    Extract research telemetry.

    IMPORTANT:
        Required telemetry must exist in info.

    We do NOT silently convert missing research metrics to zero.

    Optional environment fallback is only used for values that are genuinely
    exposed as environment scalar attributes.
    """

    missing = sorted(
        key
        for key in REQUIRED_INFO_KEYS
        if key not in info
    )

    if missing:
        raise RuntimeError(
            "Required telemetry missing during experiment:\n"
            + "\n".join(f"  - {key}" for key in missing)
        )

    metrics = {
        "spectral_efficiency": safe_float(
            info["spectral_efficiency"]
        ),
        "energy_efficiency": safe_float(
            info["energy_efficiency"]
        ),
        "total_rate": safe_float(
            info.get("total_rate", 0.0)
        ),
        "mean_sinr": safe_float(
            info.get("mean_sinr", 0.0)
        ),
        "mean_trust": safe_float(
            info["mean_trust"]
        ),
        "min_trust": safe_float(
            info.get("min_trust", 0.0)
        ),
        "max_trust": safe_float(
            info.get("max_trust", 0.0)
        ),
        "trust_delta": safe_float(
            info.get("trust_delta", 0.0)
        ),
        "mean_behavior_score": safe_float(
            info["mean_behavior_score"]
        ),
        "mean_power": safe_float(
            info["mean_power"]
        ),
        "total_power": safe_float(
            info["total_power"]
        ),
        "mean_queue": safe_float(
            info["mean_queue"]
        ),
        "interference_ratio": safe_float(
            info["interference_ratio"]
        ),
        "mean_pqi": safe_float(
            info["mean_pqi"]
        ),
        "mean_xpi": safe_float(
            info["mean_xpi"]
        ),
        "min_pqi": safe_float(
            info.get("min_pqi", 0.0)
        ),
        "max_pqi": safe_float(
            info.get("max_pqi", 0.0)
        ),
        "min_xpi": safe_float(
            info.get("min_xpi", 0.0)
        ),
        "max_xpi": safe_float(
            info.get("max_xpi", 0.0)
        ),
        "reward": safe_float(
            info.get("reward", 0.0)
        ),
    }

    return metrics


# =============================================================================
# Telemetry validation / diagnostic
# =============================================================================

def print_telemetry_validation(
    info: Dict[str, Any],
) -> None:

    print()
    print("=" * 78)
    print("TELEMETRY VALIDATION")
    print("=" * 78)

    print()
    print("Available info keys:")
    for key in sorted(info.keys()):
        print(f"  {key}")

    print()
    print("Required Phase-4B.2 telemetry:")
    for key in sorted(REQUIRED_INFO_KEYS):
        print(
            f"  {key:25s}: "
            f"{info.get(key, '<MISSING>')}"
        )

    print()
    print("Polarization diagnostics:")
    print(
        f"  mean_pqi                  : "
        f"{safe_float(info.get('mean_pqi', np.nan), np.nan):.6f}"
    )
    print(
        f"  mean_xpi                  : "
        f"{safe_float(info.get('mean_xpi', np.nan), np.nan):.6f}"
    )

    print("=" * 78)
    print()


# =============================================================================
# Episode runner
# =============================================================================

def run_episode(
    env: IRFEnvironment,
    agent: SACAgent | None = None,
    seed: int | None = None,
    evaluate: bool = False,
    steps: int = STEPS_PER_EPISODE,
    random_policy: bool = False,
) -> Dict[str, Any]:
    """
    Run one complete environment episode.

    If agent is None or random_policy=True:
        uniform random action in [-1,1]

    Otherwise:
        SAC policy is used.

    Returns episode-level aggregate metrics.
    """

    state = environment_reset(env, seed=seed)

    total_reward = 0.0

    metric_history: List[Dict[str, float]] = []

    done = False

    for _ in range(steps):

        if random_policy:
            action = env_action_random(env)
        else:
            if agent is None:
                raise ValueError(
                    "agent is required when random_policy=False"
                )

            action = agent.select_action(
                state,
                evaluate=evaluate,
            )

        next_state, reward, done, info = environment_step(
            env,
            action,
        )

        metrics = extract_metrics(info, env)

        total_reward += reward

        metric_history.append(metrics)

        state = next_state

        if done:
            break

    if not metric_history:
        raise RuntimeError(
            "Episode produced no telemetry records."
        )

    aggregate = aggregate_episode_metrics(
        metric_history,
        total_reward,
    )

    return aggregate


# =============================================================================
# Random action
# =============================================================================

def env_action_random(
    env: IRFEnvironment,
) -> np.ndarray:
    """
    Random action in SAC's normalized action range.
    """

    return np.random.uniform(
        low=-1.0,
        high=1.0,
        size=env.action_dim,
    ).astype(np.float32)


# =============================================================================
# Aggregate episode metrics
# =============================================================================

def aggregate_episode_metrics(
    metric_history: List[Dict[str, float]],
    total_reward: float,
) -> Dict[str, Any]:

    if not metric_history:
        raise ValueError("metric_history cannot be empty")

    def mean(key: str) -> float:
        return float(
            np.mean(
                [
                    safe_float(item.get(key, 0.0))
                    for item in metric_history
                ]
            )
        )

    last = metric_history[-1]
    first = metric_history[0]

    return {
        "total_reward": float(total_reward),
        "per_step_reward": float(
            total_reward / len(metric_history)
        ),

        "spectral_efficiency": mean(
            "spectral_efficiency"
        ),
        "energy_efficiency": mean(
            "energy_efficiency"
        ),
        "total_rate": mean(
            "total_rate"
        ),
        "mean_sinr": mean(
            "mean_sinr"
        ),

        "mean_trust": mean(
            "mean_trust"
        ),
        "initial_trust": safe_float(
            first.get("mean_trust", 0.0)
        ),
        "final_trust": safe_float(
            last.get("mean_trust", 0.0)
        ),
        "trust_change": (
            safe_float(last.get("mean_trust", 0.0))
            - safe_float(first.get("mean_trust", 0.0))
        ),

        "mean_behavior_score": mean(
            "mean_behavior_score"
        ),

        "mean_queue": mean(
            "mean_queue"
        ),
        "final_queue": safe_float(
            last.get("mean_queue", 0.0)
        ),

        "mean_power": mean(
            "mean_power"
        ),
        "total_power": mean(
            "total_power"
        ),

        "interference_ratio": mean(
            "interference_ratio"
        ),

        "mean_pqi": mean(
            "mean_pqi"
        ),
        "mean_xpi": mean(
            "mean_xpi"
        ),

        "min_pqi": float(
            np.min(
                [
                    safe_float(item.get("min_pqi", 0.0))
                    for item in metric_history
                ]
            )
        ),
        "max_pqi": float(
            np.max(
                [
                    safe_float(item.get("max_pqi", 0.0))
                    for item in metric_history
                ]
            )
        ),

        "min_xpi": float(
            np.min(
                [
                    safe_float(item.get("min_xpi", 0.0))
                    for item in metric_history
                ]
            )
        ),
        "max_xpi": float(
            np.max(
                [
                    safe_float(item.get("max_xpi", 0.0))
                    for item in metric_history
                ]
            )
        ),

        "steps": len(metric_history),
    }


# =============================================================================
# Random-policy baseline
# =============================================================================

def evaluate_random_policy(
    episodes: int = 1,
) -> Dict[str, Any]:

    env = create_environment(seed=SEED)

    all_results = []

    for episode in range(episodes):

        result = run_episode(
            env,
            agent=None,
            seed=10000 + episode,
            evaluate=False,
            steps=STEPS_PER_EPISODE,
            random_policy=True,
        )

        all_results.append(result)

    return aggregate_results(
        all_results
    )


# =============================================================================
# SAC evaluation
# =============================================================================

def evaluate_sac_policy(
    agent: SACAgent,
    episodes: int = FINAL_EVAL_EPISODES,
) -> Dict[str, Any]:

    all_results = []

    for episode in range(episodes):

        env = create_environment(
            seed=20000 + episode
        )

        result = run_episode(
            env,
            agent=agent,
            seed=20000 + episode,
            evaluate=True,
            steps=STEPS_PER_EPISODE,
            random_policy=False,
        )

        all_results.append(result)

    return aggregate_results(
        all_results
    )


# =============================================================================
# Aggregate multiple episodes
# =============================================================================

def aggregate_results(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:

    if not results:
        raise ValueError(
            "No episode results were supplied."
        )

    numeric_keys = [
        "total_reward",
        "per_step_reward",
        "spectral_efficiency",
        "energy_efficiency",
        "total_rate",
        "mean_sinr",
        "mean_trust",
        "initial_trust",
        "final_trust",
        "trust_change",
        "mean_behavior_score",
        "mean_queue",
        "final_queue",
        "mean_power",
        "total_power",
        "interference_ratio",
        "mean_pqi",
        "mean_xpi",
        "min_pqi",
        "max_pqi",
        "min_xpi",
        "max_xpi",
    ]

    aggregate = {
        "episodes": len(results)
    }

    for key in numeric_keys:
        aggregate[key] = float(
            np.mean(
                [
                    safe_float(result[key])
                    for result in results
                ]
            )
        )

    aggregate["episode_results"] = results

    return aggregate


# =============================================================================
# Controlled six-seed evaluation
# =============================================================================

def controlled_six_seed_evaluation(
    agent: SACAgent,
) -> Dict[str, Any]:

    results = []

    print()
    print("=" * 78)
    print("CONTROLLED SIX-SEED PHASE-4B.2 EVALUATION")
    print("=" * 78)

    for seed in CONTROLLED_SEEDS:

        env = create_environment(
            seed=seed
        )

        validation = validate_environment(
            env
        )

        result = run_episode(
            env,
            agent=agent,
            seed=seed,
            evaluate=True,
            steps=STEPS_PER_EPISODE,
            random_policy=False,
        )

        result["seed"] = seed
        result["state_dim"] = validation["state_dim"]
        result["action_dim"] = validation["action_dim"]

        results.append(result)

        print(
            f"Seed {seed:5d} | "
            f"Reward {result['total_reward']:.6f} | "
            f"Step {result['per_step_reward']:.6f} | "
            f"SE {result['spectral_efficiency']:.6f} | "
            f"EE {result['energy_efficiency']:.4e} | "
            f"Trust {result['final_trust']:.6f} | "
            f"PQI {result['mean_pqi']:.6f} | "
            f"XPI {result['mean_xpi']:.6f}"
        )

    aggregate = aggregate_results(
        results
    )

    print()
    print("-" * 78)
    print("CONTROLLED AGGREGATE")
    print("-" * 78)

    print(
        f"Mean reward          : "
        f"{aggregate['total_reward']:.6f}"
    )

    print(
        f"Mean step reward     : "
        f"{aggregate['per_step_reward']:.6f}"
    )

    print(
        f"Mean SE              : "
        f"{aggregate['spectral_efficiency']:.6f}"
    )

    print(
        f"Mean EE              : "
        f"{aggregate['energy_efficiency']:.6e}"
    )

    print(
        f"Mean final trust     : "
        f"{aggregate['final_trust']:.6f}"
    )

    print(
        f"Mean PQI             : "
        f"{aggregate['mean_pqi']:.6f}"
    )

    print(
        f"Mean XPI             : "
        f"{aggregate['mean_xpi']:.6f}"
    )

    print("=" * 78)

    return {
        "seeds": list(CONTROLLED_SEEDS),
        "aggregate": aggregate,
        "per_seed": results,
    }


# =============================================================================
# Reproducibility test
# =============================================================================

def reproducibility_check(
    agent: SACAgent,
) -> Dict[str, float]:

    env = create_environment(
        seed=SEED
    )

    state_a = environment_reset(
        env,
        seed=SEED,
    )

    action_a = agent.select_action(
        state_a,
        evaluate=True,
    )

    state_b = environment_reset(
        env,
        seed=SEED,
    )

    action_b = agent.select_action(
        state_b,
        evaluate=True,
    )

    state_error = float(
        np.max(
            np.abs(
                state_a - state_b
            )
        )
    )

    action_error = float(
        np.max(
            np.abs(
                action_a - action_b
            )
        )
    )

    passed = (
        state_error <= 1e-7
        and action_error <= 1e-7
    )

    print()
    print("=" * 78)
    print("REPRODUCIBILITY CHECK")
    print("=" * 78)

    print(
        f"State max error       : "
        f"{state_error:.12e}"
    )

    print(
        f"Action max error      : "
        f"{action_error:.12e}"
    )

    print(
        f"Status                : "
        f"{'PASSED' if passed else 'FAILED'}"
    )

    print("=" * 78)

    if not passed:
        raise RuntimeError(
            "Phase-4B.2 reproducibility check failed."
        )

    return {
        "state_max_error": state_error,
        "action_max_error": action_error,
        "passed": True,
    }


# =============================================================================
# Training
# =============================================================================

def train_phase4b2() -> Dict[str, Any]:

    ensure_results_dirs()
    set_global_seed(SEED)

    print()
    print("=" * 78)
    print(
        "TA-FDRL-IRF | "
        "Phase-4B.2 Polarization-Aware Observation SAC"
    )
    print("=" * 78)

    print()

    print("Configuration")
    print("-" * 78)

    print(
        f"Users                    : "
        f"{NUM_USERS}"
    )

    print(
        f"RIS elements             : "
        f"{NUM_RIS_ELEMENTS}"
    )

    print(
        f"Carrier frequency        : "
        f"{CARRIER_FREQUENCY_HZ / 1e9:.2f} GHz"
    )

    print(
        f"Bandwidth                : "
        f"{BANDWIDTH_HZ / 1e6:.2f} MHz"
    )

    print(
        f"Polarization PHY         : "
        f"{POLARIZATION_ENABLED}"
    )

    print(
        f"Polarization observation : "
        f"{POLARIZATION_STATE_ENABLED}"
    )

    print(
        f"Cross-polarization       : "
        f"{CROSS_POLARIZATION_FACTOR}"
    )

    print(
        f"RIS optimization         : "
        f"{OPTIMIZE_RIS}"
    )

    print(
        f"Adaptive trust           : "
        f"{not FIXED_TRUST}"
    )

    print(
        f"Training episodes        : "
        f"{NUM_TRAIN_EPISODES}"
    )

    print(
        f"Steps per episode        : "
        f"{STEPS_PER_EPISODE}"
    )

    print(
        f"Controlled seeds         : "
        f"{CONTROLLED_SEEDS}"
    )

    print()

    # -------------------------------------------------------------------------
    # Environment validation
    # -------------------------------------------------------------------------

    validation_env = create_environment(
        seed=SEED
    )

    validation = validate_environment(
        validation_env
    )

    print(
        "Phase-4B.2 environment validation: PASSED"
    )

    print()
    print(
        f"State dimension          : "
        f"{validation['state_dim']}"
    )

    print(
        f"Action dimension         : "
        f"{validation['action_dim']}"
    )

    print(
        f"Initial diagnostic PQI   : "
        f"{validation['mean_pqi']:.6f}"
    )

    print(
        f"Initial diagnostic XPI   : "
        f"{validation['mean_xpi']:.6f}"
    )

    print()

    # -------------------------------------------------------------------------
    # Telemetry validation
    # -------------------------------------------------------------------------

    telemetry_env = create_environment(
        seed=SEED
    )

    environment_reset(
        telemetry_env,
        seed=SEED,
    )

    diagnostic_action = np.zeros(
        telemetry_env.action_dim,
        dtype=np.float32,
    )

    _, _, _, diagnostic_info = environment_step(
        telemetry_env,
        diagnostic_action,
    )

    print_telemetry_validation(
        diagnostic_info
    )

    # -------------------------------------------------------------------------
    # Fresh SAC
    # -------------------------------------------------------------------------

    print(
        "Initializing fresh Phase-4B.2 SAC agent..."
    )

    print()

    agent = SACAgent(
        state_dim=validation["state_dim"],
        action_dim=validation["action_dim"],
    )

    print(
        f"Device                   : "
        f"{agent.device}"
    )

    print(
        "Checkpoint source        : "
        "FRESH INITIALIZATION"
    )

    print(
        "Legacy 100D checkpoint   : "
        "NOT LOADED"
    )

    print()

    # -------------------------------------------------------------------------
    # Random baseline
    # -------------------------------------------------------------------------

    print(
        "Evaluating random-policy baseline..."
    )

    random_baseline = evaluate_random_policy(
        episodes=1
    )

    print()
    print("=" * 78)
    print("RANDOM BASELINE")
    print("=" * 78)

    print(
        f"Episode reward : "
        f"{random_baseline['total_reward']:.6f}"
    )

    print(
        f"Per-step reward: "
        f"{random_baseline['per_step_reward']:.6f}"
    )

    print(
        f"SE             : "
        f"{random_baseline['spectral_efficiency']:.6f}"
    )

    print(
        f"EE             : "
        f"{random_baseline['energy_efficiency']:.4e}"
    )

    print(
        f"Initial trust  : "
        f"{random_baseline['initial_trust']:.6f}"
    )

    print(
        f"Final trust    : "
        f"{random_baseline['final_trust']:.6f}"
    )

    print(
        f"Trust change   : "
        f"{random_baseline['trust_change']:.6f}"
    )

    print(
        f"Behavior score : "
        f"{random_baseline['mean_behavior_score']:.6f}"
    )

    print(
        f"Queue          : "
        f"{random_baseline['mean_queue']:.6f}"
    )

    print(
        f"Power          : "
        f"{random_baseline['mean_power']:.6f}"
    )

    print(
        f"Interference   : "
        f"{random_baseline['interference_ratio']:.6f}"
    )

    print(
        f"Mean PQI       : "
        f"{random_baseline['mean_pqi']:.6f}"
    )

    print(
        f"Mean XPI       : "
        f"{random_baseline['mean_xpi']:.6f}"
    )

    print("=" * 78)
    print()

    # -------------------------------------------------------------------------
    # Training state
    # -------------------------------------------------------------------------

    training_rewards: List[float] = []
    training_metrics: List[Dict[str, Any]] = []
    evaluation_history: List[Dict[str, Any]] = []

    best_eval_reward = -np.inf
    best_eval_episode = -1

    total_updates = 0

    start_time = time.time()

    # -------------------------------------------------------------------------
    # Main training loop
    # -------------------------------------------------------------------------

    for episode in range(
        1,
        NUM_TRAIN_EPISODES + 1,
    ):

        episode_seed = SEED + episode

        env = create_environment(
            seed=episode_seed
        )

        state = environment_reset(
            env,
            seed=episode_seed,
        )

        total_reward = 0.0
        episode_metrics: List[Dict[str, float]] = []

        updates_this_episode = 0

        for step in range(
            STEPS_PER_EPISODE
        ):

            action = agent.select_action(
                state,
                evaluate=False,
            )

            next_state, reward, done, info = environment_step(
                env,
                action,
            )

            metrics = extract_metrics(
                info,
                env,
            )

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            update_result = agent.update()

            if update_result is not None:
                updates_this_episode += 1
                total_updates += 1

            total_reward += reward
            episode_metrics.append(metrics)

            state = next_state

            if done:
                break

        episode_result = aggregate_episode_metrics(
            episode_metrics,
            total_reward,
        )

        episode_result["episode"] = episode
        episode_result["seed"] = episode_seed
        episode_result["updates"] = updates_this_episode

        training_rewards.append(
            episode_result["total_reward"]
        )

        training_metrics.append(
            episode_result
        )

        # ---------------------------------------------------------------------
        # Periodic deterministic evaluation
        # ---------------------------------------------------------------------

        eval_value = None
        best_marker = ""

        if episode == 1 or episode % EVAL_INTERVAL == 0:

            eval_result = evaluate_sac_policy(
                agent,
                episodes=EVAL_EPISODES,
            )

            eval_record = {
                "episode": episode,
                "per_step_reward": eval_result[
                    "per_step_reward"
                ],
                "total_reward": eval_result[
                    "total_reward"
                ],
                "spectral_efficiency": eval_result[
                    "spectral_efficiency"
                ],
                "energy_efficiency": eval_result[
                    "energy_efficiency"
                ],
                "mean_trust": eval_result[
                    "mean_trust"
                ],
                "final_trust": eval_result[
                    "final_trust"
                ],
                "mean_pqi": eval_result[
                    "mean_pqi"
                ],
                "mean_xpi": eval_result[
                    "mean_xpi"
                ],
            }

            evaluation_history.append(
                eval_record
            )

            eval_value = eval_result[
                "per_step_reward"
            ]

            if eval_value > best_eval_reward:

                best_eval_reward = eval_value
                best_eval_episode = episode

                agent.save(
                    BEST_MODEL_PATH
                )

                best_marker = " ★ BEST"

        avg10 = float(
            np.mean(
                training_rewards[-10:]
            )
        )

        print(
            f"Episode {episode:03d} | "
            f"Reward {episode_result['total_reward']:.4f} | "
            f"Avg10 {avg10:.4f} | "
            f"SE {episode_result['spectral_efficiency']:.4f} | "
            f"EE {episode_result['energy_efficiency']:.4e} | "
            f"Trust {episode_result['final_trust']:.6f} | "
            f"dTrust {episode_result['trust_change']:.6f} | "
            f"PQI {episode_result['mean_pqi']:.6f} | "
            f"XPI {episode_result['mean_xpi']:.6f} | "
            f"Eval "
            f"{eval_value:.6f}"
            if eval_value is not None
            else
            f"Episode {episode:03d} | "
            f"Reward {episode_result['total_reward']:.4f} | "
            f"Avg10 {avg10:.4f} | "
            f"SE {episode_result['spectral_efficiency']:.4f} | "
            f"EE {episode_result['energy_efficiency']:.4e} | "
            f"Trust {episode_result['final_trust']:.6f} | "
            f"dTrust {episode_result['trust_change']:.6f} | "
            f"PQI {episode_result['mean_pqi']:.6f} | "
            f"XPI {episode_result['mean_xpi']:.6f} | "
            f"Eval ---",
            end="",
        )

        print(
            f" Updates {updates_this_episode}"
            f"{best_marker}"
        )

    # -------------------------------------------------------------------------
    # Training duration
    # -------------------------------------------------------------------------

    elapsed_seconds = time.time() - start_time

    # -------------------------------------------------------------------------
    # Safety: make sure at least one checkpoint exists
    # -------------------------------------------------------------------------

    if not os.path.exists(BEST_MODEL_PATH):

        print()
        print(
            "WARNING: no periodic best checkpoint was created."
        )

        print(
            "Saving final training state as fallback checkpoint."
        )

        agent.save(
            BEST_MODEL_PATH
        )

        best_eval_episode = NUM_TRAIN_EPISODES

    # -------------------------------------------------------------------------
    # CRITICAL:
    # Load BEST checkpoint before final evaluation.
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("BEST CHECKPOINT RESTORATION")
    print("=" * 78)

    print(
        f"Best checkpoint path : "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"Best selection value : "
        f"{best_eval_reward:.6f}"
    )

    print(
        f"Best episode         : "
        f"{best_eval_episode}"
    )

    print()

    agent.load(
        BEST_MODEL_PATH
    )

    print(
        "Best checkpoint loaded successfully."
    )

    print("=" * 78)
    print()

    # -------------------------------------------------------------------------
    # Final evaluation using BEST checkpoint
    # -------------------------------------------------------------------------

    print(
        "Final SAC evaluation using BEST checkpoint..."
    )

    final_eval = evaluate_sac_policy(
        agent,
        episodes=FINAL_EVAL_EPISODES,
    )

    print()
    print("=" * 78)
    print("FINAL SAC EVALUATION — BEST CHECKPOINT")
    print("=" * 78)

    print(
        f"Episode reward : "
        f"{final_eval['total_reward']:.6f}"
    )

    print(
        f"Per-step reward: "
        f"{final_eval['per_step_reward']:.6f}"
    )

    print(
        f"SE             : "
        f"{final_eval['spectral_efficiency']:.6f}"
    )

    print(
        f"EE             : "
        f"{final_eval['energy_efficiency']:.6e}"
    )

    print(
        f"Initial trust  : "
        f"{final_eval['initial_trust']:.6f}"
    )

    print(
        f"Final trust    : "
        f"{final_eval['final_trust']:.6f}"
    )

    print(
        f"Trust change   : "
        f"{final_eval['trust_change']:.6f}"
    )

    print(
        f"Behavior score : "
        f"{final_eval['mean_behavior_score']:.6f}"
    )

    print(
        f"Queue          : "
        f"{final_eval['mean_queue']:.6f}"
    )

    print(
        f"Power          : "
        f"{final_eval['mean_power']:.6f}"
    )

    print(
        f"Interference   : "
        f"{final_eval['interference_ratio']:.6f}"
    )

    print(
        f"Mean PQI       : "
        f"{final_eval['mean_pqi']:.6f}"
    )

    print(
        f"Mean XPI       : "
        f"{final_eval['mean_xpi']:.6f}"
    )

    print("=" * 78)
    print()

    # -------------------------------------------------------------------------
    # Controlled six-seed evaluation using BEST checkpoint
    # -------------------------------------------------------------------------

    controlled = controlled_six_seed_evaluation(
        agent
    )

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------

    reproducibility = reproducibility_check(
        agent
    )

    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------

    first10_average = float(
        np.mean(
            training_rewards[:10]
        )
    )

    last10_average = float(
        np.mean(
            training_rewards[-10:]
        )
    )

    if evaluation_history:
        best10_average = float(
            np.mean(
                [
                    item["per_step_reward"]
                    for item in evaluation_history[-10:]
                ]
            )
        )
    else:
        best10_average = float(
            final_eval["per_step_reward"]
        )

    random_step_reward = (
        random_baseline["per_step_reward"]
    )

    final_step_reward = (
        final_eval["per_step_reward"]
    )

    if random_step_reward != 0.0:
        sac_improvement_vs_random = (
            (
                final_step_reward
                - random_step_reward
            )
            / abs(random_step_reward)
            * 100.0
        )
    else:
        sac_improvement_vs_random = np.nan

    # -------------------------------------------------------------------------
    # Scientific interpretation
    # -------------------------------------------------------------------------

    scientific_interpretation = {
        "experiment": (
            "Phase-4B.2 Polarization-Aware Observation SAC"
        ),
        "purpose": (
            "Evaluate fresh SAC training with polarization-aware "
            "observations under a polarization-aware PHY."
        ),
        "action_space": (
            "40D: 20 bandwidth allocations + 20 power allocations."
        ),
        "observation_space": (
            "140D: 20 users × "
            "[SINR, interference, queue, power, trust, PQI, XPI]."
        ),
        "polarization_control": False,
        "ris_optimization": OPTIMIZE_RIS,
        "interpretation_limit": (
            "This experiment evaluates polarization-aware observation, "
            "not direct polarization control or joint RIS/polarization "
            "optimization."
        ),
        "controlled_evaluation": (
            "Six independent seeds using the restored best SAC checkpoint."
        ),
    }

    # -------------------------------------------------------------------------
    # JSON result
    # -------------------------------------------------------------------------

    result_json = {
        "experiment": "TA-FDRL-IRF Phase-4B.2",
        "title": (
            "Polarization-Aware Observation SAC Training"
        ),

        "configuration": {
            "seed": SEED,
            "training_episodes": NUM_TRAIN_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "eval_interval": EVAL_INTERVAL,
            "eval_episodes": EVAL_EPISODES,
            "final_eval_episodes": FINAL_EVAL_EPISODES,

            "controlled_seeds": list(
                CONTROLLED_SEEDS
            ),

            "num_users": NUM_USERS,
            "num_ris_elements": NUM_RIS_ELEMENTS,
            "bandwidth_hz": BANDWIDTH_HZ,
            "carrier_frequency_hz": CARRIER_FREQUENCY_HZ,

            "polarization_enabled": (
                POLARIZATION_ENABLED
            ),
            "polarization_state_enabled": (
                POLARIZATION_STATE_ENABLED
            ),
            "cross_polarization_factor": (
                CROSS_POLARIZATION_FACTOR
            ),

            "optimize_ris": OPTIMIZE_RIS,
            "fixed_trust": FIXED_TRUST,

            "trust_memory": TRUST_MEMORY,
            "trust_learning_rate": TRUST_LEARNING_RATE,
            "trust_target_rate_bps": TRUST_TARGET_RATE_BPS,

            "state_dim": validation["state_dim"],
            "action_dim": validation["action_dim"],
        },

        "training": {
            "first_10_average_reward": (
                first10_average
            ),
            "last_10_average_reward": (
                last10_average
            ),
            "best_10_evaluation_average": (
                best10_average
            ),
            "total_updates": total_updates,
            "elapsed_seconds": elapsed_seconds,
        },

        "random_baseline": random_baseline,

        "final_best_checkpoint_evaluation": final_eval,

        "controlled_six_seed": controlled,

        "reproducibility": reproducibility,

        "best_model_selection": {
            "metric": "evaluation_per_step_reward",
            "higher_is_better": True,
            "best_value": (
                best_eval_reward
            ),
            "episode": (
                best_eval_episode
            ),
            "checkpoint": (
                BEST_MODEL_PATH
            ),
        },

        "comparison": {
            "random_per_step_reward": (
                random_step_reward
            ),
            "best_sac_per_step_reward": (
                final_step_reward
            ),
            "sac_improvement_vs_random_percent": (
                sac_improvement_vs_random
            ),
        },

        "scientific_interpretation": (
            scientific_interpretation
        ),

        "training_episode_history": training_metrics,

        "evaluation_history": evaluation_history,
    }

    # -------------------------------------------------------------------------
    # Write JSON
    # -------------------------------------------------------------------------

    with open(
        JSON_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result_json,
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Write CSV
    # -------------------------------------------------------------------------

    csv_fields = [
        "episode",
        "seed",
        "total_reward",
        "per_step_reward",
        "spectral_efficiency",
        "energy_efficiency",
        "total_rate",
        "mean_sinr",
        "mean_trust",
        "initial_trust",
        "final_trust",
        "trust_change",
        "mean_behavior_score",
        "mean_queue",
        "final_queue",
        "mean_power",
        "total_power",
        "interference_ratio",
        "mean_pqi",
        "mean_xpi",
        "min_pqi",
        "max_pqi",
        "min_xpi",
        "max_xpi",
        "steps",
        "updates",
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for row in training_metrics:

            writer.writerow(
                {
                    key: row.get(
                        key,
                        "",
                    )
                    for key in csv_fields
                }
            )

    # -------------------------------------------------------------------------
    # Save NumPy arrays
    # -------------------------------------------------------------------------

    np.save(
        TRAIN_REWARDS_NPY,
        np.asarray(
            training_rewards,
            dtype=np.float64,
        ),
    )

    np.save(
        EVAL_REWARDS_NPY,
        np.asarray(
            [
                item["per_step_reward"]
                for item in evaluation_history
            ],
            dtype=np.float64,
        ),
    )

    np.save(
        EVAL_SE_NPY,
        np.asarray(
            [
                item["spectral_efficiency"]
                for item in evaluation_history
            ],
            dtype=np.float64,
        ),
    )

    np.save(
        EVAL_EE_NPY,
        np.asarray(
            [
                item["energy_efficiency"]
                for item in evaluation_history
            ],
            dtype=np.float64,
        ),
    )

    # -------------------------------------------------------------------------
    # Human-readable summary
    # -------------------------------------------------------------------------

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        f.write("=" * 78 + "\n")
        f.write(
            "TA-FDRL-IRF | "
            "PHASE-4B.2 POLARIZATION-AWARE OBSERVATION SAC\n"
        )
        f.write("=" * 78 + "\n\n")

        f.write(
            "EXPERIMENT PURPOSE\n"
        )
        f.write(
            "Evaluate whether a freshly trained SAC controller can exploit "
            "polarization-aware observations under a polarization-aware PHY.\n"
        )
        f.write(
            "This phase is an observation-awareness experiment, not direct "
            "polarization control.\n\n"
        )

        f.write(
            "CONFIGURATION\n"
        )
        f.write("-" * 78 + "\n")

        f.write(
            f"Users                    : {NUM_USERS}\n"
        )

        f.write(
            f"RIS elements             : "
            f"{NUM_RIS_ELEMENTS}\n"
        )

        f.write(
            f"Bandwidth                : "
            f"{BANDWIDTH_HZ:.0f} Hz\n"
        )

        f.write(
            f"Carrier frequency        : "
            f"{CARRIER_FREQUENCY_HZ:.0f} Hz\n"
        )

        f.write(
            f"Polarization PHY         : "
            f"{POLARIZATION_ENABLED}\n"
        )

        f.write(
            f"Polarization observation : "
            f"{POLARIZATION_STATE_ENABLED}\n"
        )

        f.write(
            f"Cross-polarization       : "
            f"{CROSS_POLARIZATION_FACTOR}\n"
        )

        f.write(
            f"RIS optimization         : "
            f"{OPTIMIZE_RIS}\n"
        )

        f.write(
            f"State dimension          : "
            f"{validation['state_dim']}\n"
        )

        f.write(
            f"Action dimension         : "
            f"{validation['action_dim']}\n"
        )

        f.write(
            f"Training episodes        : "
            f"{NUM_TRAIN_EPISODES}\n"
        )

        f.write(
            f"Steps per episode        : "
            f"{STEPS_PER_EPISODE}\n\n"
        )

        f.write(
            "BEST MODEL\n"
        )
        f.write("-" * 78 + "\n")

        f.write(
            f"Selection metric         : "
            f"evaluation_per_step_reward\n"
        )

        f.write(
            f"Best episode             : "
            f"{best_eval_episode}\n"
        )

        f.write(
            f"Best evaluation value    : "
            f"{best_eval_reward:.8f}\n"
        )

        f.write(
            f"Checkpoint               : "
            f"{BEST_MODEL_PATH}\n\n"
        )

        f.write(
            "TRAINING SUMMARY\n"
        )
        f.write("-" * 78 + "\n")

        f.write(
            f"First-10 avg reward      : "
            f"{first10_average:.8f}\n"
        )

        f.write(
            f"Last-10 avg reward       : "
            f"{last10_average:.8f}\n"
        )

        f.write(
            f"Best-10 eval avg         : "
            f"{best10_average:.8f}\n"
        )

        f.write(
            f"Total updates            : "
            f"{total_updates}\n"
        )

        f.write(
            f"Elapsed seconds          : "
            f"{elapsed_seconds:.4f}\n\n"
        )

        f.write(
            "RANDOM BASELINE\n"
        )
        f.write("-" * 78 + "\n")

        f.write(
            f"Reward                   : "
            f"{random_baseline['total_reward']:.8f}\n"
        )

        f.write(
            f"Per-step reward          : "
            f"{random_baseline['per_step_reward']:.8f}\n"
        )

        f.write(
            f"SE                       : "
            f"{random_baseline['spectral_efficiency']:.8f}\n"
        )

        f.write(
            f"EE                       : "
            f"{random_baseline['energy_efficiency']:.8e}\n"
        )

        f.write(
            f"Final trust              : "
            f"{random_baseline['final_trust']:.8f}\n"
        )

        f.write(
            f"Mean PQI                 : "
            f"{random_baseline['mean_pqi']:.8f}\n"
        )

        f.write(
            f"Mean XPI                 : "
            f"{random_baseline['mean_xpi']:.8f}\n\n"
        )

        f.write(
            "FINAL BEST-CHECKPOINT SAC\n"
        )
        f.write("-" * 78 + "\n")

        f.write(
            f"Reward                   : "
            f"{final_eval['total_reward']:.8f}\n"
        )

        f.write(
            f"Per-step reward          : "
            f"{final_eval['per_step_reward']:.8f}\n"
        )

        f.write(
            f"SE                       : "
            f"{final_eval['spectral_efficiency']:.8f}\n"
        )

        f.write(
            f"EE                       : "
            f"{final_eval['energy_efficiency']:.8e}\n"
        )

        f.write(
            f"Initial trust            : "
            f"{final_eval['initial_trust']:.8f}\n"
        )

        f.write(
            f"Final trust              : "
            f"{final_eval['final_trust']:.8f}\n"
        )

        f.write(
            f"Trust change             : "
            f"{final_eval['trust_change']:.8f}\n"
        )

        f.write(
            f"Mean behavior score      : "
            f"{final_eval['mean_behavior_score']:.8f}\n"
        )

        f.write(
            f"Mean queue               : "
            f"{final_eval['mean_queue']:.8f}\n"
        )

        f.write(
            f"Mean power               : "
            f"{final_eval['mean_power']:.8f}\n"
        )

        f.write(
            f"Interference ratio       : "
            f"{final_eval['interference_ratio']:.8f}\n"
        )

        f.write(
            f"Mean PQI                 : "
            f"{final_eval['mean_pqi']:.8f}\n"
        )

        f.write(
            f"Mean XPI                 : "
            f"{final_eval['mean_xpi']:.8f}\n\n"
        )

        f.write(
            "CONTROLLED SIX-SEED EVALUATION\n"
        )
        f.write("-" * 78 + "\n")

        for row in controlled["per_seed"]:

            f.write(
                f"Seed {row['seed']:5d} | "
                f"Reward {row['total_reward']:.8f} | "
                f"SE {row['spectral_efficiency']:.8f} | "
                f"EE {row['energy_efficiency']:.8e} | "
                f"Trust {row['final_trust']:.8f} | "
                f"PQI {row['mean_pqi']:.8f} | "
                f"XPI {row['mean_xpi']:.8f}\n"
            )

        f.write("\n")

        f.write(
            "CONTROLLED AGGREGATE\n"
        )
        f.write("-" * 78 + "\n")

        f.write(
            f"Mean reward              : "
            f"{controlled['aggregate']['total_reward']:.8f}\n"
        )

        f.write(
            f"Mean step reward         : "
            f"{controlled['aggregate']['per_step_reward']:.8f}\n"
        )

        f.write(
            f"Mean SE                  : "
            f"{controlled['aggregate']['spectral_efficiency']:.8f}\n"
        )

        f.write(
            f"Mean EE                  : "
            f"{controlled['aggregate']['energy_efficiency']:.8e}\n"
        )

        f.write(
            f"Mean final trust         : "
            f"{controlled['aggregate']['final_trust']:.8f}\n"
        )

        f.write(
            f"Mean PQI                 : "
            f"{controlled['aggregate']['mean_pqi']:.8f}\n"
        )

        f.write(
            f"Mean XPI                 : "
            f"{controlled['aggregate']['mean_xpi']:.8f}\n\n"
        )

        f.write(
            "REPRODUCIBILITY\n"
        )
        f.write("-" * 78 + "\n")

        f.write(
            f"State max error          : "
            f"{reproducibility['state_max_error']:.12e}\n"
        )

        f.write(
            f"Action max error         : "
            f"{reproducibility['action_max_error']:.12e}\n"
        )

        f.write(
            "Status                   : PASSED\n\n"
        )

        f.write(
            "COMPARISON WITH RANDOM\n"
        )
        f.write("-" * 78 + "\n")

        f.write(
            f"Random step reward       : "
            f"{random_step_reward:.8f}\n"
        )

        f.write(
            f"Best SAC step reward     : "
            f"{final_step_reward:.8f}\n"
        )

        f.write(
            f"SAC improvement          : "
            f"{sac_improvement_vs_random:.4f}%\n\n"
        )

        f.write(
            "SCIENTIFIC INTERPRETATION\n"
        )
        f.write("-" * 78 + "\n")

        f.write(
            "Phase-4B.2 introduces PQI/XPI into the observation state while "
            "keeping the action space unchanged at 40 dimensions.\n\n"
        )

        f.write(
            "The 140D state represents 20 users × "
            "[SINR, interference, queue, power, trust, PQI, XPI].\n\n"
        )

        f.write(
            "RIS optimization remains disabled. Therefore this experiment "
            "does not demonstrate polarization control, optimal RIS phase "
            "optimization, joint RIS/polarization optimization, or hardware "
            "polarization control.\n\n"
        )

        f.write(
            "The scientifically decisive comparison is between appropriately "
            "trained controllers under matched experimental conditions. "
            "Controlled fixed-action B2-B/B2-C comparisons alone cannot show "
            "a benefit from additional observations because the externally "
            "supplied actions remain identical.\n\n"
        )

        f.write(
            "END OF PHASE-4B.2 SAC TRAINING REPORT\n"
        )

    # -------------------------------------------------------------------------
    # Final console summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("TRAINING COMPLETE")
    print("=" * 78)

    print(
        f"First-10 avg reward        : "
        f"{first10_average:.6f}"
    )

    print(
        f"Last-10 avg reward         : "
        f"{last10_average:.6f}"
    )

    print(
        f"Best-10 eval average       : "
        f"{best10_average:.6f}"
    )

    print(
        f"Final BEST SAC reward      : "
        f"{final_eval['total_reward']:.6f}"
    )

    print(
        f"Final BEST SAC step       : "
        f"{final_eval['per_step_reward']:.6f}"
    )

    print(
        f"Final BEST SAC SE         : "
        f"{final_eval['spectral_efficiency']:.6f}"
    )

    print(
        f"Final BEST SAC EE         : "
        f"{final_eval['energy_efficiency']:.6e}"
    )

    print(
        f"Final BEST SAC trust      : "
        f"{final_eval['final_trust']:.6f}"
    )

    print(
        f"Controlled mean reward    : "
        f"{controlled['aggregate']['total_reward']:.6f}"
    )

    print(
        f"Controlled mean SE        : "
        f"{controlled['aggregate']['spectral_efficiency']:.6f}"
    )

    print(
        f"Controlled mean EE        : "
        f"{controlled['aggregate']['energy_efficiency']:.6e}"
    )

    print(
        f"Controlled mean PQI       : "
        f"{controlled['aggregate']['mean_pqi']:.6f}"
    )

    print(
        f"Controlled mean XPI       : "
        f"{controlled['aggregate']['mean_xpi']:.6f}"
    )

    print(
        f"SAC vs random             : "
        f"{sac_improvement_vs_random:.4f}%"
    )

    print()
    print(
        f"Best checkpoint:"
    )
    print(
        BEST_MODEL_PATH
    )

    print()
    print(
        f"Summary:"
    )
    print(
        SUMMARY_PATH
    )

    print()
    print(
        f"JSON:"
    )
    print(
        JSON_PATH
    )

    print()
    print(
        f"CSV:"
    )
    print(
        CSV_PATH
    )

    print()
    print("=" * 78)
    print(
        "PHASE-4B.2 SAC TRAINING COMPLETED"
    )
    print(
        "BEST CHECKPOINT RESTORED BEFORE FINAL EVALUATION"
    )
    print(
        "CONTROLLED SIX-SEED EVALUATION COMPLETED"
    )
    print(
        "REPRODUCIBILITY VALIDATED"
    )
    print("=" * 78)

    return result_json


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    train_phase4b2()