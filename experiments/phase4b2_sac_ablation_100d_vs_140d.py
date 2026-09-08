"""
TA-FDRL-IRF
Phase-4B.2 — 100D vs 140D SAC Ablation

Title:
    Polarization-Aware Observation SAC Training Ablation:
    100D vs 140D

Conditions:
    B2-B: 100D observation
          [SINR, Interference, Queue, Power, Trust]

    B2-C: 140D observation
          [SINR, Interference, Queue, Power, Trust, PQI, XPI]

Controlled physics:
    - Polarization PHY: ON
    - Cross-polarization factor: 0.15
    - Adaptive trust: ON
    - RIS optimization: OFF
    - Action dimension: 40
    - Users: 20
    - RIS elements: 64
    - Episode steps: 200

Research protocol:
    - Fresh SAC model for every condition
    - Same controlled training seeds
    - Paired comparison
    - Mean difference
    - Standard deviation
    - 95% CI
    - Paired t-test
    - Paired Cohen's dz

Usage:
    python experiments/phase4b2_sac_ablation_100d_vs_140d.py \
        --episodes 300 \
        --steps 200

Smoke test:
    python experiments/phase4b2_sac_ablation_100d_vs_140d.py \
        --episodes 2 \
        --steps 20

Optional:
    --device cpu
    --device cuda
    --device auto
    --output-dir results/phase4b2_ablation
"""

from __future__ import annotations

# ============================================================
# Standard library
# ============================================================

import argparse
import csv
import importlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Third-party
# ============================================================

import numpy as np
import torch


# ============================================================
# Optional SciPy
# ============================================================

try:
    from scipy import stats as scipy_stats

    SCIPY_AVAILABLE = True
except Exception:
    scipy_stats = None
    SCIPY_AVAILABLE = False


# ============================================================
# Configuration
# ============================================================

DEFAULT_SEEDS = (
    42,
    123,
    2026,
    4096,
    7777,
    9999,
)

DEFAULT_EPISODES = 300
DEFAULT_STEPS = 200

NUM_USERS = 20
NUM_RIS_ELEMENTS = 64

ACTION_DIM = 40

BANDWIDTH_HZ = 100e6
CARRIER_FREQUENCY_HZ = 28e9

POLARIZATION_ENABLED = True
CROSS_POLARIZATION_FACTOR = 0.15

POLARIZATION_V_STRENGTH = 1.0
POLARIZATION_H_STRENGTH = 1.0

OPTIMIZE_RIS = False
FIXED_TRUST = False

MAX_POWER_W = 1.0
CIRCUIT_POWER_W = 0.1

NOISE_FIGURE_DB = 7.0
NOISE_DENSITY_DBM_HZ = -174.0

TRUST_MEMORY = 0.90
TRUST_LEARNING_RATE = 0.10
TRUST_TARGET_RATE_BPS = 1e7

TRUST_SERVICE_WEIGHT = 0.35
TRUST_INTERFERENCE_WEIGHT = 0.20
TRUST_QUEUE_WEIGHT = 0.25
TRUST_INSTABILITY_WEIGHT = 0.20

TRUST_NEUTRAL_POINT = 0.50

REWARD_SE_WEIGHT = 0.45
REWARD_EE_WEIGHT = 0.20
REWARD_TRUST_WEIGHT = 0.15
REWARD_TRUST_DELTA_WEIGHT = 0.10
REWARD_INTERFERENCE_PENALTY = 0.05
REWARD_POWER_PENALTY = 0.03
REWARD_QUEUE_PENALTY = 0.07

SE_TARGET = 0.20
EE_TARGET = 1e6


# ============================================================
# Metrics
# ============================================================

METRIC_KEYS = [
    "reward",
    "per_step_reward",
    "spectral_efficiency",
    "energy_efficiency",
    "mean_trust",
    "final_trust",
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


# ============================================================
# Data structures
# ============================================================

@dataclass
class ConditionResult:
    condition: str
    state_dim: int
    action_dim: int
    training_seed: int
    episodes: int
    steps_per_episode: int

    training_time_sec: float

    final_eval_reward: float
    final_eval_per_step_reward: float
    final_eval_se: float
    final_eval_ee: float
    final_eval_trust: float

    controlled_reward: float
    controlled_per_step_reward: float
    controlled_se: float
    controlled_ee: float
    controlled_final_trust: float
    controlled_pqi: float
    controlled_xpi: float

    controlled_reward_std: float
    controlled_per_step_reward_std: float

    best_eval_reward: float
    best_eval_per_step_reward: float

    first_10_reward: float
    last_10_reward: float

    checkpoint: str


# ============================================================
# Utility
# ============================================================

def log(message: str = "") -> None:
    print(message, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass

    return float(default)


def mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    return float(np.mean(np.asarray(values, dtype=np.float64)))


def std_or_zero(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0

    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


# ============================================================
# Reproducibility
# ============================================================

def set_global_seed(seed: int) -> None:
    """
    Research-grade deterministic seed configuration.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ============================================================
# Device
# ============================================================

def resolve_device(requested: str) -> str:
    requested = requested.lower().strip()

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            log("[WARN] CUDA requested but CUDA is unavailable.")
            log("[WARN] Falling back to CPU.")
            return "cpu"

        return "cuda"

    return "cpu"


# ============================================================
# Dynamic module discovery
# ============================================================

def find_file(filename: str) -> Optional[Path]:
    """
    Search project root recursively for a Python file.
    """

    direct = PROJECT_ROOT / filename

    if direct.exists():
        return direct

    matches = list(PROJECT_ROOT.rglob(filename))

    if matches:
        return matches[0]

    return None


def import_module_from_file(
    module_name: str,
    file_path: Path,
):
    """
    Import a module directly from a file.

    Used only as a fallback when normal package imports are unavailable.
    """

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(file_path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot create import specification for {file_path}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


# ============================================================
# Import SACAgent
# ============================================================

def load_sac_agent_class():
    """
    Import SACAgent.

    Expected project location:
        agents/sac_agent.py

    The function first uses the normal package import, then falls
    back to direct file import.
    """

    errors = []

    candidates = [
        "agents.sac_agent",
        "src.agents.sac_agent",
        "models.sac_agent",
        "sac_agent",
    ]

    for module_name in candidates:

        try:
            module = importlib.import_module(module_name)

            if hasattr(module, "SACAgent"):
                return module.SACAgent

        except Exception as exc:
            errors.append(
                f"{module_name}: {type(exc).__name__}: {exc}"
            )

    file_path = find_file("sac_agent.py")

    if file_path is not None:

        try:
            module = import_module_from_file(
                "_ta_fdrl_irf_sac_agent",
                file_path,
            )

            if hasattr(module, "SACAgent"):
                return module.SACAgent

        except Exception as exc:
            errors.append(
                f"{file_path}: {type(exc).__name__}: {exc}"
            )

    message = [
        "",
        "=" * 78,
        "SACAgent IMPORT FAILED",
        "=" * 78,
        "Project root:",
        f"  {PROJECT_ROOT}",
        "",
        "Errors:",
    ]

    message.extend(
        f"  - {error}"
        for error in errors
    )

    message.extend(
        [
            "",
            "Expected file:",
            "  agents/sac_agent.py",
            "",
            "=" * 78,
        ]
    )

    raise ImportError("\n".join(message))


# ============================================================
# Import environment/config
# ============================================================

def load_environment_classes():
    """
    Try common import layouts first.

    Expected project concepts:
        IRFConfig
        IRFEnvironment

    If they are not found, produce a diagnostic error instead
    of silently continuing.
    """

    import_errors = []

    module_candidates = [
        "environment.irf_environment",
        "envs.irf_environment",
        "environments.irf_environment",
        "src.environment.irf_environment",
        "src.envs.irf_environment",
        "src.environments.irf_environment",
        "irf_environment",
    ]

    config_candidates = [
        "config.irf_config",
        "configs.irf_config",
        "src.config.irf_config",
        "src.configs.irf_config",
        "irf_config",
    ]

    environment_class = None
    config_class = None

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    for module_name in module_candidates:

        try:
            module = importlib.import_module(module_name)

            if hasattr(module, "IRFEnvironment"):
                environment_class = module.IRFEnvironment
                break

        except Exception as exc:
            import_errors.append(
                f"{module_name}: {type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    for module_name in config_candidates:

        try:
            module = importlib.import_module(module_name)

            if hasattr(module, "IRFConfig"):
                config_class = module.IRFConfig
                break

        except Exception as exc:
            import_errors.append(
                f"{module_name}: {type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # Direct file fallback
    # --------------------------------------------------------

    if environment_class is None:

        environment_file = find_file(
            "irf_environment.py"
        )

        if environment_file is not None:

            try:
                module = import_module_from_file(
                    "_ta_fdrl_irf_environment",
                    environment_file,
                )

                if hasattr(module, "IRFEnvironment"):
                    environment_class = module.IRFEnvironment

            except Exception as exc:
                import_errors.append(
                    f"{environment_file}: "
                    f"{type(exc).__name__}: {exc}"
                )

    if config_class is None:

        config_file = find_file("irf_config.py")

        if config_file is not None:

            try:
                module = import_module_from_file(
                    "_ta_fdrl_irf_config",
                    config_file,
                )

                if hasattr(module, "IRFConfig"):
                    config_class = module.IRFConfig

            except Exception as exc:
                import_errors.append(
                    f"{config_file}: "
                    f"{type(exc).__name__}: {exc}"
                )

    # --------------------------------------------------------
    # Diagnostic
    # --------------------------------------------------------

    if environment_class is None or config_class is None:

        lines = [
            "",
            "=" * 78,
            "IRF ENVIRONMENT IMPORT FAILED",
            "=" * 78,
            f"Project root: {PROJECT_ROOT}",
            "",
            "Detected files:",
        ]

        for name in (
            "sac_agent.py",
            "irf_environment.py",
            "irf_config.py",
        ):
            found = find_file(name)
            lines.append(
                f"  {name}: {found if found else 'NOT FOUND'}"
            )

        lines.extend(
            [
                "",
                "Import attempts:",
            ]
        )

        lines.extend(
            f"  - {item}"
            for item in import_errors
        )

        lines.append("=" * 78)

        raise ImportError("\n".join(lines))

    return config_class, environment_class


# ============================================================
# Classes are loaded after project path setup
# ============================================================

SACAgent = load_sac_agent_class()
IRFConfig, IRFEnvironment = load_environment_classes()


# ============================================================
# Configuration factory
# ============================================================

def build_config(
    state_dim: int,
    seed: int,
    steps: int,
):
    """
    Construct IRFConfig.

    The implementation tries the full Phase-4A/4B parameter set
    first. If the local IRFConfig has a slightly different signature,
    a filtered keyword fallback is used.
    """

    requested = {
        "num_users": NUM_USERS,
        "num_ris_elements": NUM_RIS_ELEMENTS,

        "bandwidth_hz": BANDWIDTH_HZ,
        "carrier_frequency_hz": CARRIER_FREQUENCY_HZ,

        "polarization_enabled": POLARIZATION_ENABLED,

        "cross_polarization_factor":
            CROSS_POLARIZATION_FACTOR,

        "polarization_v_strength":
            POLARIZATION_V_STRENGTH,

        "polarization_h_strength":
            POLARIZATION_H_STRENGTH,

        "max_power_w": MAX_POWER_W,
        "circuit_power_w": CIRCUIT_POWER_W,

        "noise_figure_db": NOISE_FIGURE_DB,
        "noise_density_dbm_hz":
            NOISE_DENSITY_DBM_HZ,

        "max_steps": steps,

        "optimize_ris": OPTIMIZE_RIS,
        "fixed_trust": FIXED_TRUST,

        "trust_memory": TRUST_MEMORY,
        "trust_learning_rate":
            TRUST_LEARNING_RATE,
        "trust_target_rate_bps":
            TRUST_TARGET_RATE_BPS,

        "trust_service_weight":
            TRUST_SERVICE_WEIGHT,
        "trust_interference_weight":
            TRUST_INTERFERENCE_WEIGHT,
        "trust_queue_weight":
            TRUST_QUEUE_WEIGHT,
        "trust_instability_weight":
            TRUST_INSTABILITY_WEIGHT,

        "trust_neutral_point":
            TRUST_NEUTRAL_POINT,

        "se_weight": REWARD_SE_WEIGHT,
        "ee_weight": REWARD_EE_WEIGHT,
        "trust_weight": REWARD_TRUST_WEIGHT,
        "trust_delta_weight":
            REWARD_TRUST_DELTA_WEIGHT,

        "interference_penalty":
            REWARD_INTERFERENCE_PENALTY,

        "power_penalty":
            REWARD_POWER_PENALTY,

        "queue_penalty":
            REWARD_QUEUE_PENALTY,

        "se_target": SE_TARGET,
        "ee_target": EE_TARGET,

        # Different project versions may support one of these.
        "seed": seed,

        # Phase-4B.2 observation switch.
        "polarization_state_enabled":
            state_dim == 140,

        "polarization_observation_enabled":
            state_dim == 140,
    }

    # --------------------------------------------------------
    # Try full constructor
    # --------------------------------------------------------

    try:
        return IRFConfig(**requested)

    except TypeError:
        pass

    # --------------------------------------------------------
    # Filter kwargs based on constructor signature
    # --------------------------------------------------------

    try:
        import inspect

        signature = inspect.signature(IRFConfig)

        parameters = signature.parameters

        filtered = {
            key: value
            for key, value in requested.items()
            if key in parameters
        }

        return IRFConfig(**filtered)

    except Exception as exc:

        raise RuntimeError(
            "Could not construct IRFConfig.\n"
            f"IRFConfig={IRFConfig}\n"
            f"Original requested keys={list(requested.keys())}\n"
            f"Error={exc}"
        ) from exc


# ============================================================
# Environment factory
# ============================================================

def create_environment(
    state_dim: int,
    seed: int,
    steps: int,
):
    config = build_config(
        state_dim=state_dim,
        seed=seed,
        steps=steps,
    )

    # --------------------------------------------------------
    # Try common constructor forms
    # --------------------------------------------------------

    constructors = [
        lambda: IRFEnvironment(config),
        lambda: IRFEnvironment(config=config),
        lambda: IRFEnvironment(
            config=config,
            seed=seed,
        ),
    ]

    errors = []

    for constructor in constructors:

        try:
            env = constructor()

            return env

        except Exception as exc:
            errors.append(
                f"{type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        "Could not construct IRFEnvironment.\n"
        + "\n".join(errors)
    )


# ============================================================
# Environment reset compatibility
# ============================================================

def environment_reset(
    env: Any,
    seed: Optional[int] = None,
):
    if seed is not None:

        try:
            result = env.reset(seed=seed)

        except TypeError:

            try:
                result = env.reset(
                    random_seed=seed
                )

            except TypeError:
                result = env.reset()

    else:
        result = env.reset()

    if isinstance(result, tuple):

        return result[0]

    return result


# ============================================================
# Environment step compatibility
# ============================================================

def environment_step(
    env: Any,
    action: np.ndarray,
):
    result = env.step(action)

    if len(result) == 5:

        next_state, reward, terminated, truncated, info = result

        done = bool(
            terminated or truncated
        )

        return (
            next_state,
            float(reward),
            done,
            info,
        )

    if len(result) == 4:

        next_state, reward, done, info = result

        return (
            next_state,
            float(reward),
            bool(done),
            info,
        )

    raise RuntimeError(
        "Unsupported environment.step() output."
    )


# ============================================================
# SAC constructor
# ============================================================

def create_agent(
    state_dim: int,
    action_dim: int,
    device: str,
):
    """
    Construct a fresh SACAgent.

    No old checkpoint is loaded.
    """

    attempts = [
        {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "device": device,
        },
        {
            "state_dim": state_dim,
            "action_dim": action_dim,
        },
    ]

    errors = []

    for kwargs in attempts:

        try:
            agent = SACAgent(**kwargs)

            return agent

        except Exception as exc:
            errors.append(
                f"{kwargs}: {type(exc).__name__}: {exc}"
            )

    # Positional fallback.
    try:
        return SACAgent(
            state_dim,
            action_dim,
        )

    except Exception as exc:

        errors.append(
            f"positional: {type(exc).__name__}: {exc}"
        )

    raise RuntimeError(
        "Could not construct SACAgent.\n"
        + "\n".join(errors)
    )


# ============================================================
# SAC action compatibility
# ============================================================

def select_action(
    agent: Any,
    state: np.ndarray,
    deterministic: bool = False,
):
    methods = [
        "select_action",
        "get_action",
        "act",
    ]

    last_error = None

    for method_name in methods:

        if not hasattr(agent, method_name):
            continue

        method = getattr(agent, method_name)

        attempts = [
            lambda: method(
                state,
                deterministic=deterministic,
            ),
            lambda: method(
                state,
                evaluate=deterministic,
            ),
            lambda: method(state),
        ]

        for attempt in attempts:

            try:
                action = attempt()

                action = np.asarray(
                    action,
                    dtype=np.float32,
                )

                action = action.reshape(-1)

                if action.size != ACTION_DIM:

                    raise ValueError(
                        f"SAC returned action dimension "
                        f"{action.size}; expected {ACTION_DIM}."
                    )

                return np.clip(
                    action,
                    -1.0,
                    1.0,
                )

            except Exception as exc:
                last_error = exc

    raise RuntimeError(
        "Could not obtain action from SACAgent. "
        f"Last error: {last_error}"
    )


# ============================================================
# Replay memory compatibility
# ============================================================

def remember(
    agent: Any,
    state: np.ndarray,
    action: np.ndarray,
    reward: float,
    next_state: np.ndarray,
    done: bool,
):
    methods = [
        "remember",
        "store_transition",
        "replay_buffer_add",
    ]

    for method_name in methods:

        if not hasattr(agent, method_name):
            continue

        method = getattr(agent, method_name)

        attempts = [
            lambda: method(
                state,
                action,
                reward,
                next_state,
                done,
            ),
            lambda: method(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            ),
        ]

        for attempt in attempts:

            try:
                attempt()
                return

            except Exception:
                continue

    raise RuntimeError(
        "SACAgent does not expose a compatible replay-memory method."
    )


# ============================================================
# SAC update compatibility
# ============================================================

def update_agent(agent: Any):
    methods = [
        "update",
        "train_step",
        "learn",
    ]

    for method_name in methods:

        if not hasattr(agent, method_name):
            continue

        method = getattr(agent, method_name)

        try:
            result = method()

            return result

        except TypeError:
            continue

    raise RuntimeError(
        "SACAgent does not expose update()/train_step()/learn()."
    )


# ============================================================
# Checkpoint saving
# ============================================================

def save_agent(
    agent: Any,
    path: Path,
):
    ensure_dir(path.parent)

    # --------------------------------------------------------
    # Native save
    # --------------------------------------------------------

    for method_name in (
        "save",
        "save_model",
        "save_checkpoint",
    ):

        if not hasattr(agent, method_name):
            continue

        method = getattr(agent, method_name)

        try:
            method(str(path))

            return

        except TypeError:

            try:
                method(path)

                return

            except Exception:
                pass

        except Exception:
            pass

    # --------------------------------------------------------
    # State dict fallback
    # --------------------------------------------------------

    if hasattr(agent, "state_dict"):

        state = agent.state_dict()

        torch.save(
            state,
            str(path),
        )

        return

    # --------------------------------------------------------
    # Attribute fallback
    # --------------------------------------------------------

    state = {}

    for attribute in (
        "actor",
        "critic",
        "critic1",
        "critic2",
        "actor_optimizer",
        "critic_optimizer",
        "alpha",
        "log_alpha",
    ):

        if hasattr(agent, attribute):

            obj = getattr(agent, attribute)

            if hasattr(obj, "state_dict"):
                state[attribute] = obj.state_dict()

            else:
                state[attribute] = obj

    if state:

        torch.save(
            state,
            str(path),
        )

        return

    raise RuntimeError(
        "SACAgent does not provide a supported checkpoint method."
    )


# ============================================================
# Telemetry extraction
# ============================================================

def get_info_value(
    info: Any,
    *keys: str,
    default: float = 0.0,
):
    if not isinstance(info, dict):
        return default

    for key in keys:

        if key in info:
            return finite_float(
                info[key],
                default,
            )

    # Nested telemetry.
    for parent_key in (
        "telemetry",
        "metrics",
        "info",
    ):

        nested = info.get(parent_key)

        if isinstance(nested, dict):

            for key in keys:

                if key in nested:
                    return finite_float(
                        nested[key],
                        default,
                    )

    return default


def extract_metrics(
    info: Any,
    reward: float,
) -> Dict[str, float]:

    metrics = {
        "reward": float(reward),

        "spectral_efficiency":
            get_info_value(
                info,
                "spectral_efficiency",
                "se",
            ),

        "energy_efficiency":
            get_info_value(
                info,
                "energy_efficiency",
                "ee",
            ),

        "mean_trust":
            get_info_value(
                info,
                "mean_trust",
                "trust",
            ),

        "mean_behavior_score":
            get_info_value(
                info,
                "mean_behavior_score",
                "behavior_score",
            ),

        "mean_queue":
            get_info_value(
                info,
                "mean_queue",
                "queue",
            ),

        "mean_power":
            get_info_value(
                info,
                "mean_power",
                "power",
            ),

        "total_power":
            get_info_value(
                info,
                "total_power",
            ),

        "interference_ratio":
            get_info_value(
                info,
                "interference_ratio",
                "interference",
            ),

        "mean_pqi":
            get_info_value(
                info,
                "mean_pqi",
                "pqi",
            ),

        "mean_xpi":
            get_info_value(
                info,
                "mean_xpi",
                "xpi",
            ),
    }

    return metrics


# ============================================================
# Episode aggregation
# ============================================================

def aggregate_episode(
    step_metrics: List[Dict[str, float]],
    final_info: Any,
) -> Dict[str, float]:

    if not step_metrics:
        return {
            key: 0.0
            for key in METRIC_KEYS
        }

    result = {}

    # --------------------------------------------------------
    # Step averages
    # --------------------------------------------------------

    for key in (
        "reward",
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
    ):

        values = [
            finite_float(
                item.get(key, 0.0)
            )
            for item in step_metrics
        ]

        result[key] = mean_or_zero(values)

    # --------------------------------------------------------
    # Reward
    # --------------------------------------------------------

    result["per_step_reward"] = result["reward"]

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    result["final_trust"] = get_info_value(
        final_info,
        "mean_trust",
        "final_trust",
        "trust",
        default=result["mean_trust"],
    )

    result["final_queue"] = get_info_value(
        final_info,
        "mean_queue",
        "final_queue",
        "queue",
        default=result["mean_queue"],
    )

    result["min_pqi"] = min(
        item.get(
            "mean_pqi",
            0.0,
        )
        for item in step_metrics
    )

    result["max_pqi"] = max(
        item.get(
            "mean_pqi",
            0.0,
        )
        for item in step_metrics
    )

    result["min_xpi"] = min(
        item.get(
            "mean_xpi",
            0.0,
        )
        for item in step_metrics
    )

    result["max_xpi"] = max(
        item.get(
            "mean_xpi",
            0.0,
        )
        for item in step_metrics
    )

    return result


# ============================================================
# Run episode
# ============================================================

def run_episode(
    agent: Optional[Any],
    state_dim: int,
    env_seed: int,
    steps: int,
    deterministic: bool,
) -> Dict[str, float]:

    env = create_environment(
        state_dim=state_dim,
        seed=env_seed,
        steps=steps,
    )

    state = environment_reset(
        env,
        seed=env_seed,
    )

    state = np.asarray(
        state,
        dtype=np.float32,
    ).reshape(-1)

    if state.size != state_dim:

        raise RuntimeError(
            f"State dimension mismatch. "
            f"Expected {state_dim}, got {state.size}."
        )

    records = []

    final_info = {}

    for _ in range(steps):

        if agent is None:

            action = np.random.uniform(
                -1.0,
                1.0,
                size=ACTION_DIM,
            ).astype(np.float32)

        else:

            action = select_action(
                agent,
                state,
                deterministic=deterministic,
            )

        (
            next_state,
            reward,
            done,
            info,
        ) = environment_step(
            env,
            action,
        )

        next_state = np.asarray(
            next_state,
            dtype=np.float32,
        ).reshape(-1)

        if next_state.size != state_dim:

            raise RuntimeError(
                f"Next-state dimension mismatch. "
                f"Expected {state_dim}, "
                f"got {next_state.size}."
            )

        records.append(
            extract_metrics(
                info,
                reward,
            )
        )

        final_info = info

        if agent is not None:

            remember(
                agent,
                state,
                action,
                reward,
                next_state,
                done,
            )

            # SAC implementations differ in when update() becomes
            # effective. Calling it here preserves the existing
            # training workflow.
            try:
                update_agent(agent)
            except Exception:
                # Some SAC implementations intentionally refuse
                # early updates before replay warm-up.
                pass

        state = next_state

        if done:
            break

    result = aggregate_episode(
        records,
        final_info,
    )

    result["steps"] = len(records)

    return result


# ============================================================
# Validate environment
# ============================================================

def validate_environment(
    state_dim: int,
    steps: int,
    seed: int,
):
    env = create_environment(
        state_dim=state_dim,
        seed=seed,
        steps=steps,
    )

    state = environment_reset(
        env,
        seed=seed,
    )

    state = np.asarray(
        state,
        dtype=np.float32,
    ).reshape(-1)

    if state.size != state_dim:

        raise RuntimeError(
            f"Environment state dimension mismatch: "
            f"expected {state_dim}, got {state.size}"
        )

    detected_action_dim = getattr(
        env,
        "action_dim",
        ACTION_DIM,
    )

    detected_action_dim = int(
        detected_action_dim
    )

    if detected_action_dim != ACTION_DIM:

        raise RuntimeError(
            f"Environment action dimension mismatch: "
            f"expected {ACTION_DIM}, "
            f"got {detected_action_dim}"
        )

    action = np.zeros(
        ACTION_DIM,
        dtype=np.float32,
    )

    (
        next_state,
        reward,
        done,
        info,
    ) = environment_step(
        env,
        action,
    )

    next_state = np.asarray(
        next_state,
        dtype=np.float32,
    ).reshape(-1)

    if next_state.size != state_dim:

        raise RuntimeError(
            "Environment next-state dimension mismatch: "
            f"expected {state_dim}, "
            f"got {next_state.size}"
        )

    telemetry = extract_metrics(
        info,
        reward,
    )

    log(
        f"[VALIDATION] state={state_dim} "
        f"action={ACTION_DIM} "
        f"polarization={POLARIZATION_ENABLED} "
        f"PQI={telemetry['mean_pqi']:.6f} "
        f"XPI={telemetry['mean_xpi']:.6f}"
    )

    return {
        "state_dim": state_dim,
        "action_dim": detected_action_dim,
    }


# ============================================================
# Training
# ============================================================

def train_condition(
    condition: str,
    state_dim: int,
    seed: int,
    episodes: int,
    steps: int,
    device: str,
    output_dir: Path,
) -> Tuple[ConditionResult, Any, List[Dict[str, Any]]]:

    log("")
    log("=" * 78)
    log(f"TRAINING CONDITION: {condition}")
    log("=" * 78)
    log(
        f"Observation dimension : {state_dim}"
    )
    log(
        f"Action dimension      : {ACTION_DIM}"
    )
    log(
        f"Training seed         : {seed}"
    )
    log(
        f"Episodes              : {episodes}"
    )
    log(
        f"Steps / episode      : {steps}"
    )
    log(
        f"Device                : {device}"
    )
    log(
        f"Polarization PHY      : ON"
    )
    log(
        f"Cross-pol factor      : {CROSS_POLARIZATION_FACTOR}"
    )
    log(
        f"RIS optimization      : OFF"
    )
    log(
        f"Adaptive trust        : ON"
    )
    log("")

    set_global_seed(seed)

    validation = validate_environment(
        state_dim=state_dim,
        steps=steps,
        seed=seed,
    )

    agent = create_agent(
        state_dim=state_dim,
        action_dim=ACTION_DIM,
        device=device,
    )

    checkpoint_dir = (
        output_dir
        / condition
        / "checkpoints"
    )

    ensure_dir(checkpoint_dir)

    checkpoint_path = (
        checkpoint_dir
        / f"sac_phase4b2_{condition.lower()}_best.pt"
    )

    episode_rows: List[Dict[str, Any]] = []

    best_eval_per_step = -float("inf")
    best_eval_reward = -float("inf")

    best_eval_metrics = None

    training_rewards = []

    start_time = time.time()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for episode in range(1, episodes + 1):

        env_seed = seed + episode

        result = run_episode(
            agent=agent,
            state_dim=state_dim,
            env_seed=env_seed,
            steps=steps,
            deterministic=False,
        )

        training_rewards.append(
            result["reward"]
        )

        row = {
            "condition": condition,
            "seed": seed,
            "episode": episode,
            "phase": "train",
            **result,
        }

        episode_rows.append(row)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            episode == 1
            or episode % 10 == 0
            or episode == episodes
        ):

            recent = training_rewards[
                -min(10, len(training_rewards)):
            ]

            recent_mean = mean_or_zero(
                recent
            )

            log(
                f"[{condition}] "
                f"Episode {episode:4d}/{episodes} | "
                f"reward={result['reward']:.6f} | "
                f"SE={result['spectral_efficiency']:.6f} | "
                f"EE={result['energy_efficiency']:.3e} | "
                f"trust={result['final_trust']:.6f} | "
                f"last10={recent_mean:.6f}"
            )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        if (
            episode == 1
            or episode % 10 == 0
            or episode == episodes
        ):

            eval_results = []

            for eval_index in range(3):

                eval_seed = (
                    100000
                    + seed * 10
                    + episode
                    + eval_index
                )

                eval_result = run_episode(
                    agent=agent,
                    state_dim=state_dim,
                    env_seed=eval_seed,
                    steps=steps,
                    deterministic=True,
                )

                eval_results.append(
                    eval_result
                )

            eval_reward = mean_or_zero(
                [
                    item["reward"]
                    for item in eval_results
                ]
            )

            eval_per_step = mean_or_zero(
                [
                    item["per_step_reward"]
                    for item in eval_results
                ]
            )

            eval_se = mean_or_zero(
                [
                    item["spectral_efficiency"]
                    for item in eval_results
                ]
            )

            eval_ee = mean_or_zero(
                [
                    item["energy_efficiency"]
                    for item in eval_results
                ]
            )

            eval_trust = mean_or_zero(
                [
                    item["final_trust"]
                    for item in eval_results
                ]
            )

            eval_row = {
                "condition": condition,
                "seed": seed,
                "episode": episode,
                "phase": "eval",
                "reward": eval_reward,
                "per_step_reward": eval_per_step,
                "spectral_efficiency": eval_se,
                "energy_efficiency": eval_ee,
                "final_trust": eval_trust,
            }

            episode_rows.append(
                eval_row
            )

            log(
                f"[{condition}] "
                f"  EVAL "
                f"reward={eval_reward:.6f} | "
                f"step={eval_per_step:.6f} | "
                f"SE={eval_se:.6f} | "
                f"EE={eval_ee:.3e} | "
                f"trust={eval_trust:.6f}"
            )

            # ------------------------------------------------
            # Save best
            # ------------------------------------------------

            if eval_per_step > best_eval_per_step:

                best_eval_per_step = (
                    eval_per_step
                )

                best_eval_reward = (
                    eval_reward
                )

                best_eval_metrics = {
                    "reward": eval_reward,
                    "per_step_reward":
                        eval_per_step,
                    "spectral_efficiency":
                        eval_se,
                    "energy_efficiency":
                        eval_ee,
                    "final_trust":
                        eval_trust,
                }

                save_agent(
                    agent,
                    checkpoint_path,
                )

                log(
                    f"[{condition}] "
                    f"  NEW BEST CHECKPOINT"
                )

    training_time = time.time() - start_time

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    final_eval_results = []

    for index in range(10):

        eval_seed = (
            200000
            + seed * 100
            + index
        )

        final_eval_results.append(
            run_episode(
                agent=agent,
                state_dim=state_dim,
                env_seed=eval_seed,
                steps=steps,
                deterministic=True,
            )
        )

    final_eval_reward = mean_or_zero(
        [
            item["reward"]
            for item in final_eval_results
        ]
    )

    final_eval_per_step = mean_or_zero(
        [
            item["per_step_reward"]
            for item in final_eval_results
        ]
    )

    final_eval_se = mean_or_zero(
        [
            item["spectral_efficiency"]
            for item in final_eval_results
        ]
    )

    final_eval_ee = mean_or_zero(
        [
            item["energy_efficiency"]
            for item in final_eval_results
        ]
    )

    final_eval_trust = mean_or_zero(
        [
            item["final_trust"]
            for item in final_eval_results
        ]
    )

    # --------------------------------------------------------
    # Controlled six-seed evaluation
    # --------------------------------------------------------

    controlled = []

    log("")
    log(
        f"[{condition}] CONTROLLED SIX-SEED EVALUATION"
    )

    for controlled_seed in DEFAULT_SEEDS:

        result = run_episode(
            agent=agent,
            state_dim=state_dim,
            env_seed=controlled_seed,
            steps=steps,
            deterministic=True,
        )

        result["seed"] = controlled_seed

        controlled.append(
            result
        )

        log(
            f"[{condition}] "
            f"seed={controlled_seed:5d} | "
            f"reward={result['reward']:.6f} | "
            f"step={result['per_step_reward']:.6f} | "
            f"SE={result['spectral_efficiency']:.6f} | "
            f"EE={result['energy_efficiency']:.3e} | "
            f"trust={result['final_trust']:.6f} | "
            f"PQI={result['mean_pqi']:.6f} | "
            f"XPI={result['mean_xpi']:.6f}"
        )

    controlled_reward_values = [
        item["reward"]
        for item in controlled
    ]

    controlled_step_values = [
        item["per_step_reward"]
        for item in controlled
    ]

    controlled_result = {
        "reward":
            mean_or_zero(
                controlled_reward_values
            ),

        "per_step_reward":
            mean_or_zero(
                controlled_step_values
            ),

        "spectral_efficiency":
            mean_or_zero(
                [
                    item["spectral_efficiency"]
                    for item in controlled
                ]
            ),

        "energy_efficiency":
            mean_or_zero(
                [
                    item["energy_efficiency"]
                    for item in controlled
                ]
            ),

        "final_trust":
            mean_or_zero(
                [
                    item["final_trust"]
                    for item in controlled
                ]
            ),

        "mean_pqi":
            mean_or_zero(
                [
                    item["mean_pqi"]
                    for item in controlled
                ]
            ),

        "mean_xpi":
            mean_or_zero(
                [
                    item["mean_xpi"]
                    for item in controlled
                ]
            ),
    }

    # --------------------------------------------------------
    # First/last/best training statistics
    # --------------------------------------------------------

    first_n = min(
        10,
        len(training_rewards),
    )

    last_n = min(
        10,
        len(training_rewards),
    )

    first_10_reward = mean_or_zero(
        training_rewards[:first_n]
    )

    last_10_reward = mean_or_zero(
        training_rewards[-last_n:]
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    result = ConditionResult(
        condition=condition,
        state_dim=state_dim,
        action_dim=ACTION_DIM,
        training_seed=seed,
        episodes=episodes,
        steps_per_episode=steps,

        training_time_sec=training_time,

        final_eval_reward=final_eval_reward,
        final_eval_per_step_reward=final_eval_per_step,
        final_eval_se=final_eval_se,
        final_eval_ee=final_eval_ee,
        final_eval_trust=final_eval_trust,

        controlled_reward=
            controlled_result["reward"],

        controlled_per_step_reward=
            controlled_result["per_step_reward"],

        controlled_se=
            controlled_result["spectral_efficiency"],

        controlled_ee=
            controlled_result["energy_efficiency"],

        controlled_final_trust=
            controlled_result["final_trust"],

        controlled_pqi=
            controlled_result["mean_pqi"],

        controlled_xpi=
            controlled_result["mean_xpi"],

        controlled_reward_std=
            std_or_zero(
                controlled_reward_values
            ),

        controlled_per_step_reward_std=
            std_or_zero(
                controlled_step_values
            ),

        best_eval_reward=
            best_eval_reward,

        best_eval_per_step_reward=
            best_eval_per_step,

        first_10_reward=
            first_10_reward,

        last_10_reward=
            last_10_reward,

        checkpoint=str(
            checkpoint_path
        ),
    )

    # --------------------------------------------------------
    # Condition CSV
    # --------------------------------------------------------

    condition_csv = (
        output_dir
        / condition
        / f"{condition}_episodes.csv"
    )

    ensure_dir(
        condition_csv.parent
    )

    if episode_rows:

        all_fields = sorted(
            {
                key
                for row in episode_rows
                for key in row.keys()
            }
        )

        with condition_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=all_fields,
            )

            writer.writeheader()

            writer.writerows(
                episode_rows
            )

    # --------------------------------------------------------
    # Controlled CSV
    # --------------------------------------------------------

    controlled_csv = (
        output_dir
        / condition
        / f"{condition}_controlled_seeds.csv"
    )

    with controlled_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        fields = [
            "condition",
            "seed",
            *METRIC_KEYS,
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        for item in controlled:

            row = {
                "condition": condition,
                **item,
            }

            writer.writerow(row)

    log("")
    log(
        f"[{condition}] TRAINING COMPLETE"
    )
    log(
        f"  Time             : "
        f"{training_time:.2f} sec"
    )
    log(
        f"  Final eval reward: "
        f"{final_eval_reward:.6f}"
    )
    log(
        f"  Controlled reward: "
        f"{controlled_result['reward']:.6f}"
    )
    log(
        f"  Controlled SE    : "
        f"{controlled_result['spectral_efficiency']:.6f}"
    )
    log(
        f"  Controlled EE    : "
        f"{controlled_result['energy_efficiency']:.3e}"
    )
    log(
        f"  Controlled trust : "
        f"{controlled_result['final_trust']:.6f}"
    )
    log(
        f"  Controlled PQI   : "
        f"{controlled_result['mean_pqi']:.6f}"
    )
    log(
        f"  Controlled XPI   : "
        f"{controlled_result['mean_xpi']:.6f}"
    )
    log(
        f"  Checkpoint       : "
        f"{checkpoint_path}"
    )

    return (
        result,
        agent,
        controlled,
    )


# ============================================================
# Statistical analysis
# ============================================================

def paired_statistics(
    values_100d: Sequence[float],
    values_140d: Sequence[float],
) -> Dict[str, float]:

    a = np.asarray(
        values_100d,
        dtype=np.float64,
    )

    b = np.asarray(
        values_140d,
        dtype=np.float64,
    )

    differences = b - a

    n = len(differences)

    mean_difference = (
        float(np.mean(differences))
        if n
        else 0.0
    )

    std_difference = (
        float(np.std(
            differences,
            ddof=1,
        ))
        if n > 1
        else 0.0
    )

    if n > 1 and std_difference > 0:

        standard_error = (
            std_difference
            / math.sqrt(n)
        )

    else:
        standard_error = 0.0

    # --------------------------------------------------------
    # 95% CI
    # --------------------------------------------------------

    if n > 1:

        if SCIPY_AVAILABLE:

            t_critical = float(
                scipy_stats.t.ppf(
                    0.975,
                    df=n - 1,
                )
            )

        else:

            # Approximation fallback.
            t_critical = 2.571

        margin = (
            t_critical
            * standard_error
        )

    else:

        margin = 0.0

    ci_low = (
        mean_difference - margin
    )

    ci_high = (
        mean_difference + margin
    )

    # --------------------------------------------------------
    # Paired t-test
    # --------------------------------------------------------

    if (
        n > 1
        and std_difference > 0
    ):

        if SCIPY_AVAILABLE:

            t_stat, p_value = (
                scipy_stats.ttest_rel(
                    b,
                    a,
                )
            )

            t_stat = float(t_stat)
            p_value = float(p_value)

        else:

            t_stat = (
                mean_difference
                / standard_error
            )

            p_value = float("nan")

    else:

        t_stat = 0.0
        p_value = float("nan")

    # --------------------------------------------------------
    # Cohen's dz
    # --------------------------------------------------------

    if std_difference > 0:

        cohens_dz = (
            mean_difference
            / std_difference
        )

    else:

        cohens_dz = float("nan")

    return {
        "n": n,
        "mean_100d": float(np.mean(a)),
        "mean_140d": float(np.mean(b)),
        "mean_difference_140d_minus_100d":
            mean_difference,
        "std_difference":
            std_difference,
        "standard_error":
            standard_error,
        "ci95_low":
            ci_low,
        "ci95_high":
            ci_high,
        "t_statistic":
            t_stat,
        "p_value":
            p_value,
        "cohens_dz":
            cohens_dz,
    }


# ============================================================
# Statistical comparison
# ============================================================

def compare_conditions(
    controlled_100d: Sequence[Dict[str, Any]],
    controlled_140d: Sequence[Dict[str, Any]],
):

    map_100d = {
        int(item["seed"]): item
        for item in controlled_100d
    }

    map_140d = {
        int(item["seed"]): item
        for item in controlled_140d
    }

    paired_seeds = sorted(
        set(map_100d)
        & set(map_140d)
    )

    if not paired_seeds:

        raise RuntimeError(
            "No common controlled seeds found."
        )

    rewards_100d = [
        map_100d[seed]["reward"]
        for seed in paired_seeds
    ]

    rewards_140d = [
        map_140d[seed]["reward"]
        for seed in paired_seeds
    ]

    step_100d = [
        map_100d[seed]["per_step_reward"]
        for seed in paired_seeds
    ]

    step_140d = [
        map_140d[seed]["per_step_reward"]
        for seed in paired_seeds
    ]

    se_100d = [
        map_100d[seed]["spectral_efficiency"]
        for seed in paired_seeds
    ]

    se_140d = [
        map_140d[seed]["spectral_efficiency"]
        for seed in paired_seeds
    ]

    ee_100d = [
        map_100d[seed]["energy_efficiency"]
        for seed in paired_seeds
    ]

    ee_140d = [
        map_140d[seed]["energy_efficiency"]
        for seed in paired_seeds
    ]

    trust_100d = [
        map_100d[seed]["final_trust"]
        for seed in paired_seeds
    ]

    trust_140d = [
        map_140d[seed]["final_trust"]
        for seed in paired_seeds
    ]

    pqi_100d = [
        map_100d[seed]["mean_pqi"]
        for seed in paired_seeds
    ]

    pqi_140d = [
        map_140d[seed]["mean_pqi"]
        for seed in paired_seeds
    ]

    xpi_100d = [
        map_100d[seed]["mean_xpi"]
        for seed in paired_seeds
    ]

    xpi_140d = [
        map_140d[seed]["mean_xpi"]
        for seed in paired_seeds
    ]

    statistics = {
        "reward": paired_statistics(
            rewards_100d,
            rewards_140d,
        ),

        "per_step_reward": paired_statistics(
            step_100d,
            step_140d,
        ),

        "spectral_efficiency": paired_statistics(
            se_100d,
            se_140d,
        ),

        "energy_efficiency": paired_statistics(
            ee_100d,
            ee_140d,
        ),

        "final_trust": paired_statistics(
            trust_100d,
            trust_140d,
        ),

        "mean_pqi": paired_statistics(
            pqi_100d,
            pqi_140d,
        ),

        "mean_xpi": paired_statistics(
            xpi_100d,
            xpi_140d,
        ),
    }

    return paired_seeds, statistics


# ============================================================
# Interpretation
# ============================================================

def interpret_result(
    reward_stats: Dict[str, float],
) -> str:

    delta = reward_stats[
        "mean_difference_140d_minus_100d"
    ]

    p_value = reward_stats[
        "p_value"
    ]

    if delta > 0:

        if (
            math.isfinite(p_value)
            and p_value < 0.05
        ):
            return (
                "140D outperforms 100D on mean paired reward "
                "with statistical significance at alpha=0.05. "
                "The additional PQI/XPI observations appear "
                "to provide exploitable information to SAC "
                "under the controlled Phase-4B.2 protocol."
            )

        return (
            "140D has higher mean paired reward than 100D, "
            "but the difference is not statistically significant "
            "at alpha=0.05. The additional PQI/XPI observations "
            "may be useful, but the six-seed evidence is "
            "insufficient for a strong claim."
        )

    if delta < 0:

        if (
            math.isfinite(p_value)
            and p_value < 0.05
        ):
            return (
                "140D performs significantly worse than 100D. "
                "The additional observation dimensions may impose "
                "learning difficulty, redundancy, or optimization "
                "cost under the fixed action/RIS configuration."
            )

        return (
            "140D has lower mean paired reward than 100D, "
            "but the difference is not statistically significant. "
            "The additional observation dimensions do not show "
            "a demonstrated benefit under this protocol."
        )

    return (
        "140D and 100D have effectively identical paired reward. "
        "The additional PQI/XPI observations are represented but "
        "do not produce a measurable reward advantage under the "
        "controlled action and RIS configuration."
    )


# ============================================================
# Save final comparison CSV
# ============================================================

def save_comparison_csv(
    path: Path,
    controlled_100d: Sequence[Dict[str, Any]],
    controlled_140d: Sequence[Dict[str, Any]],
):
    map_100d = {
        int(item["seed"]): item
        for item in controlled_100d
    }

    map_140d = {
        int(item["seed"]): item
        for item in controlled_140d
    }

    seeds = sorted(
        set(map_100d)
        & set(map_140d)
    )

    ensure_dir(path.parent)

    fields = [
        "seed",

        "reward_100d",
        "reward_140d",
        "reward_delta",

        "per_step_reward_100d",
        "per_step_reward_140d",
        "per_step_reward_delta",

        "se_100d",
        "se_140d",
        "se_delta",

        "ee_100d",
        "ee_140d",
        "ee_delta",

        "final_trust_100d",
        "final_trust_140d",
        "final_trust_delta",

        "pqi_100d",
        "pqi_140d",
        "pqi_delta",

        "xpi_100d",
        "xpi_140d",
        "xpi_delta",
    ]

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

        for seed in seeds:

            a = map_100d[seed]
            b = map_140d[seed]

            writer.writerow(
                {
                    "seed": seed,

                    "reward_100d":
                        a["reward"],
                    "reward_140d":
                        b["reward"],
                    "reward_delta":
                        b["reward"]
                        - a["reward"],

                    "per_step_reward_100d":
                        a["per_step_reward"],
                    "per_step_reward_140d":
                        b["per_step_reward"],
                    "per_step_reward_delta":
                        b["per_step_reward"]
                        - a["per_step_reward"],

                    "se_100d":
                        a["spectral_efficiency"],
                    "se_140d":
                        b["spectral_efficiency"],
                    "se_delta":
                        b["spectral_efficiency"]
                        - a["spectral_efficiency"],

                    "ee_100d":
                        a["energy_efficiency"],
                    "ee_140d":
                        b["energy_efficiency"],
                    "ee_delta":
                        b["energy_efficiency"]
                        - a["energy_efficiency"],

                    "final_trust_100d":
                        a["final_trust"],
                    "final_trust_140d":
                        b["final_trust"],
                    "final_trust_delta":
                        b["final_trust"]
                        - a["final_trust"],

                    "pqi_100d":
                        a["mean_pqi"],
                    "pqi_140d":
                        b["mean_pqi"],
                    "pqi_delta":
                        b["mean_pqi"]
                        - a["mean_pqi"],

                    "xpi_100d":
                        a["mean_xpi"],
                    "xpi_140d":
                        b["mean_xpi"],
                    "xpi_delta":
                        b["mean_xpi"]
                        - a["mean_xpi"],
                }
            )


# ============================================================
# Save JSON
# ============================================================

def save_json(
    path: Path,
    payload: Dict[str, Any],
):
    ensure_dir(path.parent)

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


# ============================================================
# Save summary TXT
# ============================================================

def save_summary(
    path: Path,
    payload: Dict[str, Any],
):
    ensure_dir(path.parent)

    comparison = payload[
        "statistics"
    ]

    interpretation = payload[
        "interpretation"
    ]

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "=" * 78 + "\n"
        )

        handle.write(
            "TA-FDRL-IRF\n"
        )

        handle.write(
            "PHASE-4B.2 SAC ABLATION\n"
        )

        handle.write(
            "100D VS 140D POLARIZATION-AWARE OBSERVATION\n"
        )

        handle.write(
            "=" * 78 + "\n\n"
        )

        handle.write(
            "EXPERIMENT CONFIGURATION\n"
        )

        handle.write(
            "-" * 78 + "\n"
        )

        config = payload[
            "configuration"
        ]

        for key, value in config.items():

            handle.write(
                f"{key}: {value}\n"
            )

        handle.write("\n")

        handle.write(
            "PAIRED SEEDS\n"
        )

        handle.write(
            "-" * 78 + "\n"
        )

        handle.write(
            ", ".join(
                str(x)
                for x in payload[
                    "paired_seeds"
                ]
            )
        )

        handle.write("\n\n")

        handle.write(
            "CONDITION RESULTS\n"
        )

        handle.write(
            "-" * 78 + "\n"
        )

        for condition_name, result in (
            payload["conditions"].items()
        ):

            handle.write(
                f"\n[{condition_name}]\n"
            )

            for key, value in result.items():

                handle.write(
                    f"{key}: {value}\n"
                )

        handle.write("\n")

        handle.write(
            "PAIRED STATISTICS\n"
        )

        handle.write(
            "-" * 78 + "\n"
        )

        for metric_name, stats in (
            comparison.items()
        ):

            handle.write(
                f"\n{metric_name}\n"
            )

            for key, value in stats.items():

                handle.write(
                    f"  {key}: {value}\n"
                )

        handle.write("\n")

        handle.write(
            "SCIENTIFIC INTERPRETATION\n"
        )

        handle.write(
            "-" * 78 + "\n"
        )

        handle.write(
            interpretation
        )

        handle.write("\n\n")

        handle.write(
            "LIMITATION\n"
        )

        handle.write(
            "-" * 78 + "\n"
        )

        handle.write(
            "RIS optimization is disabled. Therefore this experiment "
            "does not demonstrate optimal RIS phase control, joint "
            "RIS/polarization optimization, or hardware-level "
            "polarization control. It tests whether additional "
            "polarization-aware observations (PQI/XPI) improve SAC "
            "learning under a fixed action/RIS configuration.\n"
        )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "TA-FDRL-IRF Phase-4B.2 "
            "100D vs 140D SAC ablation."
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=(
            f"Training episodes "
            f"(default: {DEFAULT_EPISODES})"
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=(
            f"Steps per episode "
            f"(default: {DEFAULT_STEPS})"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        help="Training device.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "phase4b2_sac_ablation_100d_vs_140d"
        ),
        help="Output directory.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Training seed used for the two conditions "
            "in this run. Default: 42"
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if args.episodes < 1:
        raise ValueError(
            "--episodes must be >= 1"
        )

    if args.steps < 1:
        raise ValueError(
            "--steps must be >= 1"
        )

    device = resolve_device(
        args.device
    )

    output_dir = Path(
        args.output_dir
    ).resolve()

    ensure_dir(output_dir)

    log("")
    log("=" * 78)
    log(
        "TA-FDRL-IRF PHASE-4B.2"
    )
    log(
        "100D VS 140D SAC ABLATION"
    )
    log("=" * 78)

    log(
        f"Project root : {PROJECT_ROOT}"
    )

    log(
        f"Output dir   : {output_dir}"
    )

    log(
        f"PyTorch      : {torch.__version__}"
    )

    log(
        f"Device       : {device}"
    )

    log(
        f"Python       : {sys.version.split()[0]}"
    )

    log("")

    # ========================================================
    # B2-B — 100D
    # ========================================================

    result_100d, agent_100d, controlled_100d = (
        train_condition(
            condition="B2-B_100D",
            state_dim=100,
            seed=args.seed,
            episodes=args.episodes,
            steps=args.steps,
            device=device,
            output_dir=output_dir,
        )
    )

    # ========================================================
    # B2-C — 140D
    # ========================================================

    result_140d, agent_140d, controlled_140d = (
        train_condition(
            condition="B2-C_140D",
            state_dim=140,
            seed=args.seed,
            episodes=args.episodes,
            steps=args.steps,
            device=device,
            output_dir=output_dir,
        )
    )

    # ========================================================
    # Paired statistics
    # ========================================================

    paired_seeds, statistics = compare_conditions(
        controlled_100d,
        controlled_140d,
    )

    interpretation = interpret_result(
        statistics["reward"]
    )

    # ========================================================
    # Save comparison CSV
    # ========================================================

    comparison_csv = (
        output_dir
        / "phase4b2_100d_vs_140d_paired_comparison.csv"
    )

    save_comparison_csv(
        comparison_csv,
        controlled_100d,
        controlled_140d,
    )

    # ========================================================
    # Payload
    # ========================================================

    payload = {
        "experiment": (
            "TA-FDRL-IRF Phase-4B.2 "
            "100D vs 140D SAC Ablation"
        ),

        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "configuration": {
            "episodes": args.episodes,
            "steps_per_episode": args.steps,
            "training_seed": args.seed,

            "users": NUM_USERS,
            "ris_elements":
                NUM_RIS_ELEMENTS,

            "action_dim": ACTION_DIM,

            "state_dim_100d": 100,
            "state_dim_140d": 140,

            "polarization_enabled":
                POLARIZATION_ENABLED,

            "cross_polarization_factor":
                CROSS_POLARIZATION_FACTOR,

            "polarization_v_strength":
                POLARIZATION_V_STRENGTH,

            "polarization_h_strength":
                POLARIZATION_H_STRENGTH,

            "optimize_ris":
                OPTIMIZE_RIS,

            "fixed_trust":
                FIXED_TRUST,

            "adaptive_trust":
                True,

            "device": device,

            "scipy_available":
                SCIPY_AVAILABLE,
        },

        "controlled_seed_protocol":
            list(DEFAULT_SEEDS),

        "paired_seeds":
            paired_seeds,

        "conditions": {
            "B2-B_100D":
                asdict(result_100d),

            "B2-C_140D":
                asdict(result_140d),
        },

        "statistics":
            statistics,

        "interpretation":
            interpretation,

        "files": {
            "comparison_csv":
                str(comparison_csv),

            "100d_checkpoint":
                result_100d.checkpoint,

            "140d_checkpoint":
                result_140d.checkpoint,
        },
    }

    # ========================================================
    # JSON
    # ========================================================

    json_path = (
        output_dir
        / "phase4b2_100d_vs_140d.json"
    )

    save_json(
        json_path,
        payload,
    )

    # ========================================================
    # Summary
    # ========================================================

    summary_path = (
        output_dir
        / "phase4b2_100d_vs_140d_summary.txt"
    )

    save_summary(
        summary_path,
        payload,
    )

    # ========================================================
    # Final console report
    # ========================================================

    reward_stats = statistics[
        "reward"
    ]

    step_stats = statistics[
        "per_step_reward"
    ]

    se_stats = statistics[
        "spectral_efficiency"
    ]

    ee_stats = statistics[
        "energy_efficiency"
    ]

    trust_stats = statistics[
        "final_trust"
    ]

    pqi_stats = statistics[
        "mean_pqi"
    ]

    xpi_stats = statistics[
        "mean_xpi"
    ]

    log("")
    log("=" * 78)
    log(
        "FINAL PHASE-4B.2 ABLATION RESULT"
    )
    log("=" * 78)

    log("")
    log(
        "CONTROLLED MEAN RESULTS"
    )
    log(
        "-" * 78
    )

    log(
        f"{'Metric':<24}"
        f"{'100D':>16}"
        f"{'140D':>16}"
        f"{'Delta':>16}"
    )

    log(
        f"{'Reward':<24}"
        f"{reward_stats['mean_100d']:>16.6f}"
        f"{reward_stats['mean_140d']:>16.6f}"
        f"{reward_stats['mean_difference_140d_minus_100d']:>16.6f}"
    )

    log(
        f"{'Per-step reward':<24}"
        f"{step_stats['mean_100d']:>16.6f}"
        f"{step_stats['mean_140d']:>16.6f}"
        f"{step_stats['mean_difference_140d_minus_100d']:>16.6f}"
    )

    log(
        f"{'Spectral efficiency':<24}"
        f"{se_stats['mean_100d']:>16.6f}"
        f"{se_stats['mean_140d']:>16.6f}"
        f"{se_stats['mean_difference_140d_minus_100d']:>16.6f}"
    )

    log(
        f"{'Energy efficiency':<24}"
        f"{ee_stats['mean_100d']:>16.3e}"
        f"{ee_stats['mean_140d']:>16.3e}"
        f"{ee_stats['mean_difference_140d_minus_100d']:>16.3e}"
    )

    log(
        f"{'Final trust':<24}"
        f"{trust_stats['mean_100d']:>16.6f}"
        f"{trust_stats['mean_140d']:>16.6f}"
        f"{trust_stats['mean_difference_140d_minus_100d']:>16.6f}"
    )

    log(
        f"{'Mean PQI':<24}"
        f"{pqi_stats['mean_100d']:>16.6f}"
        f"{pqi_stats['mean_140d']:>16.6f}"
        f"{pqi_stats['mean_difference_140d_minus_100d']:>16.6f}"
    )

    log(
        f"{'Mean XPI':<24}"
        f"{xpi_stats['mean_100d']:>16.6f}"
        f"{xpi_stats['mean_140d']:>16.6f}"
        f"{xpi_stats['mean_difference_140d_minus_100d']:>16.6f}"
    )

    log("")
    log(
        "PAIRED REWARD STATISTICS"
    )

    log(
        "-" * 78
    )

    log(
        f"N                : "
        f"{reward_stats['n']}"
    )

    log(
        f"Mean delta       : "
        f"{reward_stats['mean_difference_140d_minus_100d']:.6f}"
    )

    log(
        f"Std delta        : "
        f"{reward_stats['std_difference']:.6f}"
    )

    log(
        f"95% CI           : "
        f"[{reward_stats['ci95_low']:.6f}, "
        f"{reward_stats['ci95_high']:.6f}]"
    )

    log(
        f"t-statistic      : "
        f"{reward_stats['t_statistic']:.6f}"
    )

    log(
        f"p-value          : "
        f"{reward_stats['p_value']:.6g}"
    )

    log(
        f"Cohen's dz       : "
        f"{reward_stats['cohens_dz']:.6f}"
    )

    log("")
    log(
        "SCIENTIFIC INTERPRETATION"
    )

    log(
        "-" * 78
    )

    log(
        interpretation
    )

    log("")
    log(
        "OUTPUT FILES"
    )

    log(
        "-" * 78
    )

    log(
        f"JSON       : {json_path}"
    )

    log(
        f"Summary    : {summary_path}"
    )

    log(
        f"Comparison : {comparison_csv}"
    )

    log(
        f"100D CSV   : "
        f"{output_dir / 'B2-B_100D' / 'B2-B_100D_episodes.csv'}"
    )

    log(
        f"140D CSV   : "
        f"{output_dir / 'B2-C_140D' / 'B2-C_140D_episodes.csv'}"
    )

    log("")
    log("=" * 78)
    log(
        "EXPERIMENT COMPLETE"
    )
    log("=" * 78)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        log("")
        log(
            "[STOPPED] Experiment interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        log("")
        log("=" * 78)
        log(
            "EXPERIMENT FAILED"
        )
        log("=" * 78)
        log(
            f"{type(exc).__name__}: {exc}"
        )
        log("")
        log(
            "The experiment was stopped before producing "
            "a potentially invalid scientific result."
        )
        log("=" * 78)

        raise