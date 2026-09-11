"""
TA-FDRL-IRF
Dashboard Runtime Manager
---------------------------------

Research-grade runtime controller for:

    Phase-4A
    Phase-4B.1
    Phase-4B.2

Features
--------
- Real IRFEnvironment telemetry
- SAC agent integration
- Optional GovernanceEngine
- Episode lifecycle management
- Auto Episode execution
- Exact bounded multi-episode execution
- Episode history
- Episode reward tracking
- Continuous runtime
- Pause / Stop support
- Polarization telemetry
- No synthetic telemetry
- Strict state/action dimension validation
"""

from __future__ import annotations

import importlib
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Optional SAC imports
# ============================================================

SAC_IMPORT_CANDIDATES = [
    ("agents.sac_agent", "SACAgent"),
    ("agents.sac", "SACAgent"),
    ("agent.sac_agent", "SACAgent"),
    ("agent.sac", "SACAgent"),
    ("models.sac_agent", "SACAgent"),
    ("models.sac", "SACAgent"),
    ("sac_agent", "SACAgent"),
]


# ============================================================
# Optional Governance imports
# ============================================================

GOVERNANCE_IMPORT_CANDIDATES = [
    ("engines.governance_engine", "GovernanceEngine"),
    ("governance.governance_engine", "GovernanceEngine"),
    ("governance_engine", "GovernanceEngine"),
    ("governance.engine", "GovernanceEngine"),
]


# ============================================================
# Safe helpers
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        if value is None:
            return default

        if isinstance(
            value,
            (list, tuple, np.ndarray),
        ):

            arr = np.asarray(
                value,
                dtype=float,
            )

            if arr.size == 0:
                return default

            value = np.nanmean(arr)

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:

    try:
        return int(value)
    except Exception:
        return default


def safe_bool(
    value: Any,
    default: bool = False,
) -> bool:

    try:
        return bool(value)
    except Exception:
        return default


def to_numpy(
    value: Any,
) -> Optional[np.ndarray]:

    if value is None:
        return None

    try:
        return np.asarray(value)
    except Exception:
        return None


def json_safe(
    value: Any,
) -> Any:
    """
    Convert numpy/scalar objects into
    Streamlit/JSON-safe objects.
    """

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):

        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [
            json_safe(v)
            for v in value
        ]

    return value


# ============================================================
# Dynamic import
# ============================================================

def load_class(
    candidates: List[Tuple[str, str]],
) -> Tuple[
    Optional[type],
    Optional[str],
]:

    for module_name, class_name in candidates:

        try:

            module = importlib.import_module(
                module_name
            )

            cls = getattr(
                module,
                class_name,
            )

            return (
                cls,
                f"{module_name}.{class_name}",
            )

        except Exception:
            continue

    return None, None


# ============================================================
# Runtime mode configuration
# ============================================================

@dataclass
class RuntimeModeConfig:

    name: str

    num_users: int = 20
    num_ris_elements: int = 64

    state_dim: int = 100
    action_dim: int = 40

    polarization_enabled: bool = True
    polarization_state_enabled: bool = False

    adaptive_trust: bool = True
    fixed_trust: bool = False

    optimize_ris: bool = False

    max_steps: int = 200

    default_seed: int = 42

    bandwidth_hz: float = 100e6
    carrier_frequency_hz: float = 28e9

    checkpoint: Optional[str] = None


# ============================================================
# Supported research modes
# ============================================================

MODE_REGISTRY: Dict[
    str,
    RuntimeModeConfig,
] = {

    # --------------------------------------------------------
    # Phase-4A
    # --------------------------------------------------------

    "Phase-4A": RuntimeModeConfig(
        name="Phase-4A",
        num_users=20,
        num_ris_elements=64,
        state_dim=100,
        action_dim=40,
        polarization_enabled=True,
        polarization_state_enabled=False,
        adaptive_trust=True,
        fixed_trust=False,
        optimize_ris=False,
        max_steps=200,
        default_seed=42,
        bandwidth_hz=100e6,
        carrier_frequency_hz=28e9,
        checkpoint=None,
    ),

    # --------------------------------------------------------
    # Phase-4B.1
    # --------------------------------------------------------

    "Phase-4B.1": RuntimeModeConfig(
        name="Phase-4B.1",
        num_users=20,
        num_ris_elements=64,
        state_dim=100,
        action_dim=40,
        polarization_enabled=True,
        polarization_state_enabled=False,
        adaptive_trust=True,
        fixed_trust=False,
        optimize_ris=False,
        max_steps=200,
        default_seed=42,
        bandwidth_hz=100e6,
        carrier_frequency_hz=28e9,
        checkpoint=None,
    ),

    # --------------------------------------------------------
    # Phase-4B.2
    # --------------------------------------------------------

    "Phase-4B.2": RuntimeModeConfig(
        name="Phase-4B.2",
        num_users=20,
        num_ris_elements=64,
        state_dim=140,
        action_dim=40,
        polarization_enabled=True,
        polarization_state_enabled=True,
        adaptive_trust=True,
        fixed_trust=False,
        optimize_ris=False,
        max_steps=200,
        default_seed=42,
        bandwidth_hz=100e6,
        carrier_frequency_hz=28e9,
        checkpoint=(
            "results/"
            "best_sac_irf_phase4b2_polarization_observation/"
            "sac_phase4b2_polarization_observation.pt"
        ),
    ),
}


# ============================================================
# Runtime Manager
# ============================================================

class RuntimeManager:
    """
    Controls the actual IRF environment and SAC agent.

    Episode lifecycle:

        initialize
            ↓
        Episode 1
            ↓
        Step 1 ... Step N
            ↓
        finalize Episode 1
            ↓
        reset environment
            ↓
        Episode 2
            ↓
        ...

    No synthetic telemetry is generated here.
    """

    def __init__(
        self,
        mode: str = "Phase-4B.2",
        seed: Optional[int] = None,
    ):

        if mode not in MODE_REGISTRY:

            raise ValueError(
                f"Unsupported runtime mode: {mode}. "
                f"Available: {list(MODE_REGISTRY.keys())}"
            )

        self.mode_config = MODE_REGISTRY[mode]
        self.mode = mode

        self.seed = (
            self.mode_config.default_seed
            if seed is None
            else int(seed)
        )

        # ----------------------------------------------------
        # Runtime counters
        # ----------------------------------------------------

        self.current_episode = 1
        self.completed_episodes = 0

        self.step_count = 0
        self.total_steps = 0

        self.episode_reward = 0.0
        self.last_reward = 0.0

        self.episode_rewards: List[
            float
        ] = []

        self.episode_history: List[
            Dict[str, Any]
        ] = []

        self.step_history: List[
            Dict[str, Any]
        ] = []

        self.episode_summary: Optional[
            Dict[str, Any]
        ] = None

        # ----------------------------------------------------
        # Runtime flags
        # ----------------------------------------------------

        self.running = False
        self.paused = False
        self.continuous = False

        self.terminated = False
        self.truncated = False

        # ----------------------------------------------------
        # Runtime objects
        # ----------------------------------------------------

        self.env = None
        self.agent = None
        self.governance = None

        self.sac_available = False
        self.governance_available = False

        self.sac_source = None
        self.governance_source = None

        self.checkpoint_path = (
            self.mode_config.checkpoint
        )

        self.checkpoint_loaded = False

        # ----------------------------------------------------
        # Current state
        # ----------------------------------------------------

        self.state = None

        self.last_action = None

        self.last_info: Dict[
            str,
            Any,
        ] = {}

        # ----------------------------------------------------
        # Build runtime
        # ----------------------------------------------------

        self._build_runtime()

        # ----------------------------------------------------
        # Initial environment reset
        # ----------------------------------------------------

        self.reset_episode(
            seed=self.seed
        )

    # ========================================================
    # Build environment
    # ========================================================

    def _build_environment(self):

        candidates = [
            (
                "environment.irf_env",
                "IRFEnvironment",
            ),
            (
                "environment",
                "IRFEnvironment",
            ),
            (
                "irf_env",
                "IRFEnvironment",
            ),
        ]

        IRFEnvironment, source = load_class(
            candidates
        )

        if IRFEnvironment is None:

            raise ImportError(
                "IRFEnvironment could not be imported. "
                "Expected one of: "
                "environment.irf_env.IRFEnvironment, "
                "environment.IRFEnvironment, "
                "irf_env.IRFEnvironment"
            )

        try:

            config_candidates = [
                (
                    "environment.irf_env",
                    "IRFConfig",
                ),
                (
                    "environment",
                    "IRFConfig",
                ),
                (
                    "irf_env",
                    "IRFConfig",
                ),
            ]

            IRFConfig, _ = load_class(
                config_candidates
            )

            if IRFConfig is not None:

                config = IRFConfig()

                self._set_config_value(
                    config,
                    "num_users",
                    self.mode_config.num_users,
                )

                self._set_config_value(
                    config,
                    "num_ris_elements",
                    self.mode_config.num_ris_elements,
                )

                self._set_config_value(
                    config,
                    "bandwidth_hz",
                    self.mode_config.bandwidth_hz,
                )

                self._set_config_value(
                    config,
                    "carrier_frequency_hz",
                    self.mode_config.carrier_frequency_hz,
                )

                self._set_config_value(
                    config,
                    "polarization_enabled",
                    self.mode_config.polarization_enabled,
                )

                self._set_config_value(
                    config,
                    "polarization_state_enabled",
                    self.mode_config.polarization_state_enabled,
                )

                self._set_config_value(
                    config,
                    "optimize_ris",
                    self.mode_config.optimize_ris,
                )

                self._set_config_value(
                    config,
                    "fixed_trust",
                    self.mode_config.fixed_trust,
                )

                self._set_config_value(
                    config,
                    "max_steps",
                    self.mode_config.max_steps,
                )

                if hasattr(
                    config,
                    "adaptive_trust",
                ):

                    self._set_config_value(
                        config,
                        "adaptive_trust",
                        self.mode_config.adaptive_trust,
                    )

                if hasattr(
                    config,
                    "adaptive_trust_enabled",
                ):

                    self._set_config_value(
                        config,
                        "adaptive_trust_enabled",
                        self.mode_config.adaptive_trust,
                    )

                try:

                    env = IRFEnvironment(
                        config=config
                    )

                except TypeError:

                    try:

                        env = IRFEnvironment(
                            config
                        )

                    except TypeError:

                        env = IRFEnvironment()

            else:

                env = IRFEnvironment()

            # ------------------------------------------------
            # Explicit environment flags
            # ------------------------------------------------

            self._set_config_value(
                env,
                "polarization_enabled",
                self.mode_config.polarization_enabled,
            )

            self._set_config_value(
                env,
                "polarization_state_enabled",
                self.mode_config.polarization_state_enabled,
            )

            self._set_config_value(
                env,
                "optimize_ris",
                self.mode_config.optimize_ris,
            )

            self._set_config_value(
                env,
                "fixed_trust",
                self.mode_config.fixed_trust,
            )

            return env

        except Exception as exc:

            raise RuntimeError(
                f"Failed to construct IRFEnvironment: {exc}"
            ) from exc

    # ========================================================
    # Generic attribute setter
    # ========================================================

    @staticmethod
    def _set_config_value(
        obj: Any,
        name: str,
        value: Any,
    ) -> None:

        try:

            if hasattr(
                obj,
                name,
            ):

                setattr(
                    obj,
                    name,
                    value,
                )

        except Exception:
            pass

    # ========================================================
    # Build SAC
    # ========================================================

    def _build_agent(self):

        SACAgent, source = load_class(
            SAC_IMPORT_CANDIDATES
        )

        if SACAgent is None:

            self.sac_available = False
            self.sac_source = None
            self.checkpoint_loaded = False

            return None

        self.sac_source = source

        cfg = self.mode_config

        # ----------------------------------------------------
        # Constructor attempts
        # ----------------------------------------------------

        constructor_attempts = [

            {
                "state_dim": cfg.state_dim,
                "action_dim": cfg.action_dim,
                "hidden_dim": 256,
                "actor_lr": 3e-4,
                "critic_lr": 3e-4,
                "alpha_lr": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "buffer_size": 100_000,
                "batch_size": 256,
                "device": None,
            },

            {
                "state_dim": cfg.state_dim,
                "action_dim": cfg.action_dim,
            },
        ]

        agent = None
        last_error = None

        for kwargs in constructor_attempts:

            try:

                agent = SACAgent(
                    **kwargs
                )

                break

            except Exception as exc:

                last_error = exc

        if agent is None:

            self.sac_available = False

            self.sac_source = (
                f"{source} "
                f"(constructor failed: {last_error})"
            )

            self.checkpoint_loaded = False

            return None

        self.sac_available = True
        self.checkpoint_loaded = False

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        checkpoint = self.checkpoint_path

        if checkpoint is None:
            return agent

        checkpoint = os.path.abspath(
            os.path.normpath(
                checkpoint
            )
        )

        self.checkpoint_path = checkpoint

        if not os.path.isfile(
            checkpoint
        ):

            self.checkpoint_loaded = False

            self.sac_source = (
                f"{source} "
                "(checkpoint not found)"
            )

            return agent

        # ----------------------------------------------------
        # Preferred load()
        # ----------------------------------------------------

        if hasattr(
            agent,
            "load",
        ):

            try:

                # The project's SACAgent.load()
                # returns None on success.
                agent.load(
                    checkpoint
                )

                self.checkpoint_loaded = True

                return agent

            except Exception as exc:

                self.checkpoint_loaded = False

                self.sac_source = (
                    f"{source} "
                    f"(checkpoint load failed: {exc})"
                )

        # ----------------------------------------------------
        # Alternative load_checkpoint()
        # ----------------------------------------------------

        if hasattr(
            agent,
            "load_checkpoint",
        ):

            try:

                result = agent.load_checkpoint(
                    checkpoint
                )

                self.checkpoint_loaded = (
                    result is not False
                )

                if self.checkpoint_loaded:
                    return agent

            except Exception as exc:

                self.checkpoint_loaded = False

                self.sac_source = (
                    f"{source} "
                    f"(checkpoint load failed: {exc})"
                )

        return agent

    # ========================================================
    # Build Governance
    # ========================================================

    def _build_governance(self):

        GovernanceEngine, source = load_class(
            GOVERNANCE_IMPORT_CANDIDATES
        )

        if GovernanceEngine is None:

            self.governance_available = False
            self.governance_source = None

            return None

        self.governance_source = source

        attempts = [
            {},
            {
                "mode": self.mode
            },
        ]

        for kwargs in attempts:

            try:

                governance = GovernanceEngine(
                    **kwargs
                )

                self.governance_available = True

                return governance

            except Exception:
                continue

        self.governance_available = False

        return None

    # ========================================================
    # Build entire runtime
    # ========================================================

    def _build_runtime(self):

        self.env = self._build_environment()

        self.agent = self._build_agent()

        self.governance = self._build_governance()

    # ========================================================
    # Reset episode
    # ========================================================

    def reset_episode(
        self,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:

        if seed is not None:

            self.seed = int(
                seed
            )

        if self.env is None:

            raise RuntimeError(
                "Environment is not initialized."
            )

        # ----------------------------------------------------
        # Environment reset
        # ----------------------------------------------------

        reset_result = self.env.reset(
            seed=self.seed
        )

        if isinstance(
            reset_result,
            tuple,
        ):

            self.state = (
                reset_result[0]
            )

            if len(reset_result) > 1:

                self.last_info = (
                    reset_result[1]
                    if isinstance(
                        reset_result[1],
                        dict,
                    )
                    else {}
                )

            else:

                self.last_info = {}

        else:

            self.state = reset_result
            self.last_info = {}

        # ----------------------------------------------------
        # Validate initial state
        # ----------------------------------------------------

        state_array = np.asarray(
            self.state,
            dtype=np.float32,
        ).reshape(-1)

        expected_state_dim = (
            self.mode_config.state_dim
        )

        if state_array.size != expected_state_dim:

            raise ValueError(
                "Environment state dimension mismatch "
                f"for {self.mode}: "
                f"expected {expected_state_dim}, "
                f"got {state_array.size}"
            )

        self.state = state_array

        # ----------------------------------------------------
        # Reset episode counters
        # ----------------------------------------------------

        self.step_count = 0

        self.episode_reward = 0.0

        self.last_reward = 0.0

        self.terminated = False

        self.truncated = False

        self.last_action = np.zeros(
            self.mode_config.action_dim,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Runtime state
        # ----------------------------------------------------

        self.running = False
        self.paused = False

        self.episode_summary = None

        return self.snapshot()

    # ========================================================
    # SAC action
    # ========================================================

    def _select_action(self) -> np.ndarray:

        action_dim = (
            self.mode_config.action_dim
        )

        expected_state_dim = (
            self.mode_config.state_dim
        )

        # ----------------------------------------------------
        # Agent must exist
        # ----------------------------------------------------

        if self.agent is None:

            raise RuntimeError(
                "SAC agent is unavailable. "
                "Cannot execute AI-agent action."
            )

        if not hasattr(
            self.agent,
            "select_action",
        ):

            raise RuntimeError(
                "SAC agent does not implement "
                "select_action()."
            )

        # ----------------------------------------------------
        # Validate state
        # ----------------------------------------------------

        state = np.asarray(
            self.state,
            dtype=np.float32,
        ).reshape(-1)

        if state.size != expected_state_dim:

            raise ValueError(
                "State dimension mismatch: "
                f"expected {expected_state_dim}, "
                f"got {state.size}"
            )

        # ----------------------------------------------------
        # SAC inference
        # ----------------------------------------------------

        try:

            action = self.agent.select_action(
                state,
                evaluate=True,
            )

        except TypeError:

            try:

                action = self.agent.select_action(
                    state
                )

            except Exception as exc:

                raise RuntimeError(
                    "SAC agent failed to produce "
                    f"an action: {exc}"
                ) from exc

        except Exception as exc:

            raise RuntimeError(
                "SAC agent failed to produce "
                f"an action: {exc}"
            ) from exc

        # ----------------------------------------------------
        # Validate action
        # ----------------------------------------------------

        if action is None:

            raise RuntimeError(
                "SAC agent returned None action."
            )

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if action.size != action_dim:

            raise ValueError(
                "SAC action dimension mismatch: "
                f"expected {action_dim}, "
                f"got {action.size}"
            )

        # ----------------------------------------------------
        # Numerical safety
        # ----------------------------------------------------

        if not np.isfinite(
            action
        ).all():

            raise FloatingPointError(
                "SAC agent produced NaN or "
                "infinite action values."
            )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        return action

    # ========================================================
    # Governance
    # ========================================================

    def _evaluate_governance(
        self,
        proposed_action: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        Dict[str, Any],
    ]:

        # ----------------------------------------------------
        # Governance unavailable
        # ----------------------------------------------------

        if self.governance is None:

            return proposed_action, {
                "status": "BYPASS",
                "available": False,
                "reason": (
                    "GovernanceEngine unavailable"
                ),
            }

        # ----------------------------------------------------
        # Common APIs
        # ----------------------------------------------------

        result = None

        method_names = [
            "evaluate",
            "govern",
            "process",
            "check_action",
            "validate_action",
        ]

        for method_name in method_names:

            if not hasattr(
                self.governance,
                method_name,
            ):
                continue

            method = getattr(
                self.governance,
                method_name,
            )

            attempts = [

                (
                    self.state,
                    proposed_action,
                ),

                (
                    proposed_action,
                ),

                (
                    {
                        "state": self.state,
                        "action": proposed_action,
                    },
                ),
            ]

            for args in attempts:

                try:

                    result = method(
                        *args
                    )

                    break

                except Exception:
                    continue

            if result is not None:
                break

        # ----------------------------------------------------
        # No compatible API
        # ----------------------------------------------------

        if result is None:

            return proposed_action, {
                "status": "BYPASS",
                "available": True,
                "reason": (
                    "No compatible governance API"
                ),
            }

        # ----------------------------------------------------
        # Parse result
        # ----------------------------------------------------

        executed_action = (
            proposed_action.copy()
        )

        governance_info: Dict[
            str,
            Any,
        ] = {
            "status": "ALLOW",
            "available": True,
        }

        # ----------------------------------------------------
        # Dict result
        # ----------------------------------------------------

        if isinstance(
            result,
            dict,
        ):

            governance_info.update(
                json_safe(result)
            )

            status = str(
                result.get(
                    "status",
                    result.get(
                        "decision",
                        "ALLOW",
                    ),
                )
            ).upper()

            governance_info[
                "status"
            ] = status

            candidate_action = (
                result.get("action")
                or result.get(
                    "executed_action"
                )
                or result.get(
                    "constrained_action"
                )
            )

            if candidate_action is not None:

                try:

                    arr = np.asarray(
                        candidate_action,
                        dtype=np.float32,
                    ).reshape(-1)

                    if (
                        arr.size
                        == self.mode_config.action_dim
                    ):

                        executed_action = arr

                except Exception:
                    pass

        # ----------------------------------------------------
        # Tuple result
        # ----------------------------------------------------

        elif isinstance(
            result,
            tuple,
        ):

            if len(result) >= 1:

                first = result[0]

                try:

                    arr = np.asarray(
                        first,
                        dtype=np.float32,
                    ).reshape(-1)

                    if (
                        arr.size
                        == self.mode_config.action_dim
                    ):

                        executed_action = arr

                except Exception:
                    pass

            if (
                len(result) >= 2
                and isinstance(
                    result[1],
                    dict,
                )
            ):

                governance_info.update(
                    json_safe(
                        result[1]
                    )
                )

        # ----------------------------------------------------
        # String result
        # ----------------------------------------------------

        elif isinstance(
            result,
            str,
        ):

            governance_info[
                "status"
            ] = result.upper()

        # ----------------------------------------------------
        # Numerical safety
        # ----------------------------------------------------

        executed_action = np.asarray(
            executed_action,
            dtype=np.float32,
        ).reshape(-1)

        if executed_action.size != (
            self.mode_config.action_dim
        ):

            raise ValueError(
                "Governance returned invalid "
                "action dimension: "
                f"expected "
                f"{self.mode_config.action_dim}, "
                f"got {executed_action.size}"
            )

        executed_action = np.nan_to_num(
            executed_action,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        executed_action = np.clip(
            executed_action,
            -1.0,
            1.0,
        )

        return (
            executed_action,
            json_safe(
                governance_info
            ),
        )

    # ========================================================
    # Build actual telemetry
    # ========================================================

    def _build_environment_telemetry(
        self,
        info: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:

        info = (
            info
            if isinstance(
                info,
                dict,
            )
            else {}
        )

        telemetry: Dict[
            str,
            Any,
        ] = {}

        # ----------------------------------------------------
        # Start from environment info
        # ----------------------------------------------------

        telemetry.update(
            json_safe(info)
        )

        # ----------------------------------------------------
        # Actual environment scalar fields
        # ----------------------------------------------------

        actual_fields = [

            "spectral_efficiency",
            "energy_efficiency",
            "total_rate",

            "mean_sinr",
            "mean_trust",
            "mean_behavior_score",

            "mean_queue",
            "mean_power",

            "radiated_power",
            "total_power",

            "interference_ratio",

            "mean_pqi",
            "mean_xpi",

            "arrival_mean",
            "arrival_sum",

            "state_interference_mean",
            "state_power_mean",

            "cross_polarization_factor",
            "mean_polarization_quality",
            "mean_cross_polarization_ratio",

            "min_pqi",
            "max_pqi",
            "min_xpi",
            "max_xpi",

            "polarization_enabled",
            "polarization_state_enabled",

            "optimize_ris",
            "fixed_trust",

            "current_step",
        ]

        for field in actual_fields:

            if hasattr(
                self.env,
                field,
            ):

                try:

                    telemetry[field] = json_safe(
                        getattr(
                            self.env,
                            field,
                        )
                    )

                except Exception:
                    pass

        # ----------------------------------------------------
        # Config values
        # ----------------------------------------------------

        telemetry.setdefault(
            "users",
            self.mode_config.num_users,
        )

        telemetry.setdefault(
            "num_users",
            self.mode_config.num_users,
        )

        telemetry.setdefault(
            "ris_elements",
            self.mode_config.num_ris_elements,
        )

        telemetry.setdefault(
            "num_ris_elements",
            self.mode_config.num_ris_elements,
        )

        telemetry.setdefault(
            "state_dim",
            self.mode_config.state_dim,
        )

        telemetry.setdefault(
            "action_dim",
            self.mode_config.action_dim,
        )

        telemetry.setdefault(
            "bandwidth_hz",
            self.mode_config.bandwidth_hz,
        )

        telemetry.setdefault(
            "carrier_frequency_hz",
            self.mode_config.carrier_frequency_hz,
        )

        # ----------------------------------------------------
        # Actual per-user arrays
        # ----------------------------------------------------

        user_fields = [

            "rate",
            "sinr",
            "trust",
            "behavior_score",
            "trust_delta",

            "queue",
            "power",
            "interference",

            "pqi",
            "xpi",

            "polarization_quality",
            "cross_polarization_ratio",
        ]

        for field in user_fields:

            if hasattr(
                self.env,
                field,
            ):

                try:

                    value = getattr(
                        self.env,
                        field,
                    )

                    arr = to_numpy(
                        value
                    )

                    if arr is not None:

                        telemetry[field] = (
                            json_safe(arr)
                        )

                except Exception:
                    pass

        # ----------------------------------------------------
        # Alternative names
        # ----------------------------------------------------

        aliases = {

            "rate": [
                "rates",
                "user_rates",
                "data_rates",
            ],

            "sinr": [
                "sinrs",
                "user_sinr",
            ],

            "trust": [
                "trust_scores",
                "user_trust",
            ],

            "queue": [
                "queues",
                "queue_lengths",
            ],

            "power": [
                "powers",
                "user_power",
            ],

            "interference": [
                "interferences",
                "user_interference",
            ],
        }

        for canonical, candidates in (
            aliases.items()
        ):

            if canonical in telemetry:
                continue

            for candidate in candidates:

                if hasattr(
                    self.env,
                    candidate,
                ):

                    try:

                        value = getattr(
                            self.env,
                            candidate,
                        )

                        arr = to_numpy(
                            value
                        )

                        if arr is not None:

                            telemetry[
                                canonical
                            ] = json_safe(arr)

                            break

                    except Exception:
                        continue

        # ----------------------------------------------------
        # Runtime metadata
        # ----------------------------------------------------

        telemetry[
            "episode"
        ] = self.current_episode

        telemetry[
            "current_episode"
        ] = self.current_episode

        telemetry[
            "step"
        ] = self.step_count

        telemetry[
            "current_step"
        ] = self.step_count

        telemetry[
            "max_steps"
        ] = self.mode_config.max_steps

        telemetry[
            "episode_reward"
        ] = self.episode_reward

        telemetry[
            "last_reward"
        ] = self.last_reward

        telemetry[
            "completed_episodes"
        ] = self.completed_episodes

        telemetry[
            "total_steps"
        ] = self.total_steps

        telemetry[
            "running"
        ] = self.running

        telemetry[
            "paused"
        ] = self.paused

        telemetry[
            "continuous"
        ] = self.continuous

        telemetry[
            "mode"
        ] = self.mode

        telemetry[
            "sac_available"
        ] = self.sac_available

        telemetry[
            "checkpoint_loaded"
        ] = self.checkpoint_loaded

        telemetry[
            "governance_available"
        ] = self.governance_available

        return json_safe(
            telemetry
        )

    # ========================================================
    # Step
    # ========================================================

    def step(
        self,
        action: Optional[
            np.ndarray
        ] = None,
        sleep_seconds: float = 0.0,
        auto_start: bool = True,
    ) -> Dict[str, Any]:

        if self.env is None:

            raise RuntimeError(
                "Environment is not initialized."
            )

        if self.state is None:

            self.reset_episode(
                seed=self.seed
            )

        # ----------------------------------------------------
        # If episode was already completed
        # ----------------------------------------------------

        if self.is_episode_done():

            self._finalize_episode()

            if (
                self.continuous
                and auto_start
            ):

                self._start_next_episode()

            else:

                self.running = False

                return {
                    "telemetry":
                        self._build_environment_telemetry(
                            self.last_info
                        ),

                    "governance": {
                        "status":
                            "EPISODE_DONE",

                        "available":
                            self.governance_available,
                    },

                    "episode_done": True,

                    "snapshot":
                        self.snapshot(),
                }

        # ----------------------------------------------------
        # Action
        # ----------------------------------------------------

        if action is None:

            proposed_action = (
                self._select_action()
            )

        else:

            proposed_action = np.asarray(
                action,
                dtype=np.float32,
            ).reshape(-1)

        # ----------------------------------------------------
        # Action validation
        # ----------------------------------------------------

        if (
            proposed_action.size
            != self.mode_config.action_dim
        ):

            raise ValueError(
                "Action dimension mismatch: "
                f"expected "
                f"{self.mode_config.action_dim}, "
                f"got {proposed_action.size}"
            )

        if not np.isfinite(
            proposed_action
        ).all():

            raise FloatingPointError(
                "Proposed action contains "
                "NaN or infinite values."
            )

        proposed_action = np.clip(
            proposed_action,
            -1.0,
            1.0,
        )

        # ----------------------------------------------------
        # Governance
        # ----------------------------------------------------

        (
            executed_action,
            governance_info,
        ) = self._evaluate_governance(
            proposed_action
        )

        # ----------------------------------------------------
        # Environment step
        # ----------------------------------------------------

        result = self.env.step(
            executed_action
        )

        if not isinstance(
            result,
            tuple,
        ):

            raise RuntimeError(
                "IRFEnvironment.step() "
                "must return a tuple."
            )

        # ----------------------------------------------------
        # Gymnasium format
        # ----------------------------------------------------

        if len(result) == 5:

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = result

        # ----------------------------------------------------
        # Legacy Gym format
        # ----------------------------------------------------

        elif len(result) == 4:

            (
                next_state,
                reward,
                done,
                info,
            ) = result

            terminated = bool(
                done
            )

            truncated = False

        else:

            raise RuntimeError(
                "Unsupported environment.step() "
                "return format."
            )

        # ----------------------------------------------------
        # Validate next state
        # ----------------------------------------------------

        next_state_array = np.asarray(
            next_state,
            dtype=np.float32,
        ).reshape(-1)

        expected_state_dim = (
            self.mode_config.state_dim
        )

        if (
            next_state_array.size
            != expected_state_dim
        ):

            raise ValueError(
                "Environment next-state dimension "
                "mismatch: "
                f"expected {expected_state_dim}, "
                f"got {next_state_array.size}"
            )

        if not np.isfinite(
            next_state_array
        ).all():

            raise FloatingPointError(
                "Environment returned NaN or "
                "infinite state values."
            )

        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------

        self.state = (
            next_state_array
        )

        self.last_reward = safe_float(
            reward
        )

        self.episode_reward += (
            self.last_reward
        )

        self.step_count += 1

        self.total_steps += 1

        self.terminated = bool(
            terminated
        )

        self.truncated = bool(
            truncated
        )

        self.last_action = np.asarray(
            executed_action,
            dtype=np.float32,
        ).reshape(-1)

        self.last_info = (
            info
            if isinstance(
                info,
                dict,
            )
            else {}
        )

        # ----------------------------------------------------
        # Actual telemetry
        # ----------------------------------------------------

        telemetry = (
            self._build_environment_telemetry(
                self.last_info
            )
        )

        # ----------------------------------------------------
        # Step history
        # ----------------------------------------------------

        history_row = {

            "episode":
                self.current_episode,

            "step":
                self.step_count,

            "reward":
                self.last_reward,

            "episode_reward":
                self.episode_reward,

            "spectral_efficiency":
                safe_float(
                    telemetry.get(
                        "spectral_efficiency"
                    )
                ),

            "energy_efficiency":
                safe_float(
                    telemetry.get(
                        "energy_efficiency"
                    )
                ),

            "mean_trust":
                safe_float(
                    telemetry.get(
                        "mean_trust"
                    )
                ),

            "mean_sinr":
                safe_float(
                    telemetry.get(
                        "mean_sinr"
                    )
                ),

            "mean_queue":
                safe_float(
                    telemetry.get(
                        "mean_queue"
                    )
                ),

            "mean_power":
                safe_float(
                    telemetry.get(
                        "mean_power"
                    )
                ),
        }

        self.step_history.append(
            json_safe(
                history_row
            )
        )

        # ----------------------------------------------------
        # Keep history bounded
        # ----------------------------------------------------

        if len(
            self.step_history
        ) > 20_000:

            self.step_history = (
                self.step_history[
                    -20_000:
                ]
            )

        # ----------------------------------------------------
        # Episode completion
        # ----------------------------------------------------

        episode_done = (
            self.terminated
            or self.truncated
            or (
                self.step_count
                >= self.mode_config.max_steps
            )
        )

        # ----------------------------------------------------
        # Finalize
        # ----------------------------------------------------

        if episode_done:

            if (
                self.step_count
                >= self.mode_config.max_steps
            ):

                self.terminated = True

            self._finalize_episode(
                telemetry=telemetry
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # auto_start=False is used by run_episodes()
            # so it can decide exactly when the next episode
            # starts.
            # ------------------------------------------------

            if (
                self.continuous
                and auto_start
            ):

                self._start_next_episode()

            else:

                self.running = False

        else:

            self.running = True

        # ----------------------------------------------------
        # Optional delay
        # ----------------------------------------------------

        if sleep_seconds > 0:

            time.sleep(
                float(
                    sleep_seconds
                )
            )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return {

            "telemetry":
                telemetry,

            "governance":
                json_safe(
                    governance_info
                ),

            "proposed_action":
                json_safe(
                    proposed_action
                ),

            "executed_action":
                json_safe(
                    executed_action
                ),

            "episode_done":
                episode_done,

            "snapshot":
                self.snapshot(),
        }

    # ========================================================
    # Episode finalization
    # ========================================================

    def _finalize_episode(
        self,
        telemetry: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:

        # ----------------------------------------------------
        # Prevent duplicate finalization
        # ----------------------------------------------------

        if self.episode_summary is not None:

            return self.episode_summary

        if telemetry is None:

            telemetry = (
                self._build_environment_telemetry(
                    self.last_info
                )
            )

        steps = max(
            0,
            self.step_count,
        )

        mean_reward = (
            self.episode_reward
            / steps
            if steps > 0
            else 0.0
        )

        summary = {

            "episode":
                self.current_episode,

            "steps":
                steps,

            "episode_reward":
                safe_float(
                    self.episode_reward
                ),

            "mean_reward":
                safe_float(
                    mean_reward
                ),

            "final_trust":
                safe_float(
                    telemetry.get(
                        "mean_trust"
                    )
                ),

            "final_spectral_efficiency":
                safe_float(
                    telemetry.get(
                        "spectral_efficiency"
                    )
                ),

            "final_energy_efficiency":
                safe_float(
                    telemetry.get(
                        "energy_efficiency"
                    )
                ),

            "final_total_rate":
                safe_float(
                    telemetry.get(
                        "total_rate"
                    )
                ),

            "final_mean_sinr":
                safe_float(
                    telemetry.get(
                        "mean_sinr"
                    )
                ),

            "final_mean_queue":
                safe_float(
                    telemetry.get(
                        "mean_queue"
                    )
                ),

            "final_mean_power":
                safe_float(
                    telemetry.get(
                        "mean_power"
                    )
                ),

            "final_interference_ratio":
                safe_float(
                    telemetry.get(
                        "interference_ratio"
                    )
                ),

            "final_mean_pqi":
                safe_float(
                    telemetry.get(
                        "mean_pqi"
                    )
                ),

            "final_mean_xpi":
                safe_float(
                    telemetry.get(
                        "mean_xpi"
                    )
                ),

            "terminated":
                bool(
                    self.terminated
                ),

            "truncated":
                bool(
                    self.truncated
                ),

            "seed":
                self.seed,

            "mode":
                self.mode,
        }

        summary = json_safe(
            summary
        )

        self.episode_summary = (
            summary
        )

        self.episode_history.append(
            summary
        )

        self.episode_rewards.append(
            safe_float(
                self.episode_reward
            )
        )

        self.completed_episodes += 1

        return summary

    # ========================================================
    # Start next episode
    # ========================================================

    def _start_next_episode(self):

        # ----------------------------------------------------
        # Next episode number
        # ----------------------------------------------------

        self.current_episode += 1

        # ----------------------------------------------------
        # Deterministic per-episode seed
        #
        # Episode 1 = base seed
        # Episode 2 = base seed + 1
        # Episode 3 = base seed + 2
        # ----------------------------------------------------

        next_seed = (
            self.seed + 1
        )

        self.seed = next_seed

        # ----------------------------------------------------
        # Reset environment
        # ----------------------------------------------------

        self.reset_episode(
            seed=next_seed
        )

        self.running = True
        self.paused = False

    # ========================================================
    # Start runtime
    # ========================================================

    def start_runtime(
        self,
        continuous: Optional[
            bool
        ] = None,
    ) -> Dict[str, Any]:

        if continuous is not None:

            self.continuous = bool(
                continuous
            )

        # ----------------------------------------------------
        # If current episode is complete,
        # move to next episode.
        # ----------------------------------------------------

        if self.is_episode_done():

            if self.episode_summary is None:

                self._finalize_episode()

            if self.continuous:

                self._start_next_episode()

            else:

                self.reset_episode(
                    seed=self.seed
                )

        self.running = True
        self.paused = False

        return self.snapshot()

    # ========================================================
    # Pause
    # ========================================================

    def pause_runtime(self):

        self.paused = True
        self.running = False

        return self.snapshot()

    # ========================================================
    # Stop
    # ========================================================

    def stop(self):

        self.running = False
        self.paused = False

        return self.snapshot()

    # ========================================================
    # Set continuous
    # ========================================================

    def set_continuous(
        self,
        enabled: bool,
    ):

        self.continuous = bool(
            enabled
        )

        return self.snapshot()

    # ========================================================
    # Run bounded number of steps
    # ========================================================

    def run_steps(
        self,
        num_steps: int = 1,
        speed: float = 0.0,
        continuous: Optional[
            bool
        ] = None,
    ) -> Dict[str, Any]:

        num_steps = max(
            1,
            int(num_steps),
        )

        if continuous is not None:

            self.continuous = bool(
                continuous
            )

        self.start_runtime(
            continuous=self.continuous
        )

        delay = max(
            0.0,
            float(speed),
        )

        last_result = None

        for _ in range(
            num_steps
        ):

            if not self.running:

                break

            last_result = self.step(
                sleep_seconds=delay
            )

            # ------------------------------------------------
            # Manual mode:
            # stop immediately after episode completion.
            # ------------------------------------------------

            if (
                last_result.get(
                    "episode_done",
                    False,
                )
                and not self.continuous
            ):

                break

        if last_result is not None:

            return last_result

        return {

            "telemetry":
                self._build_environment_telemetry(
                    self.last_info
                ),

            "snapshot":
                self.snapshot(),
        }

    # ========================================================
    # Run exactly one episode
    # ========================================================

    def run_episode(
        self,
        seed: Optional[
            int
        ] = None,
    ) -> Dict[str, Any]:

        # ----------------------------------------------------
        # Optional explicit seed
        # ----------------------------------------------------

        if seed is not None:

            self.seed = int(
                seed
            )

        # ----------------------------------------------------
        # If current episode is already complete,
        # start a fresh episode.
        # ----------------------------------------------------

        if self.is_episode_done():

            if self.episode_summary is None:

                self._finalize_episode()

            # Move episode number forward.
            self.current_episode += 1

            if seed is None:

                next_seed = (
                    self.seed + 1
                )

            else:

                next_seed = self.seed

            self.seed = next_seed

            self.reset_episode(
                seed=next_seed
            )

        # ----------------------------------------------------
        # Exactly one episode
        # ----------------------------------------------------

        self.continuous = False
        self.running = True
        self.paused = False

        # ----------------------------------------------------
        # Execute until completed
        # ----------------------------------------------------

        while self.running:

            self.step(
                auto_start=False
            )

            if (
                self.episode_summary
                is not None
            ):

                break

        self.running = False
        self.paused = False

        return self.snapshot()

    # ========================================================
    # Run multiple episodes
    # ========================================================

    def run_episodes(
        self,
        num_episodes: int = 1,
        speed: float = 0.0,
    ) -> Dict[str, Any]:

        """
        Execute exactly `num_episodes` completed episodes.

        Important design:

        run_episodes() owns the episode transition.

        step(auto_start=False) finalizes the current episode
        but does NOT immediately create the next episode.

        This prevents:

            Episode N complete
                ↓
            accidental Episode N+1 start
                ↓
            target counter mismatch

        Therefore:

            requested N
                =
            exactly N completed episodes
        """

        num_episodes = max(
            1,
            int(num_episodes),
        )

        delay = max(
            0.0,
            float(speed),
        )

        # ----------------------------------------------------
        # Save starting completion count
        # ----------------------------------------------------

        start_completed = (
            self.completed_episodes
        )

        target_completed = (
            start_completed
            + num_episodes
        )

        # ----------------------------------------------------
        # Auto execution mode
        # ----------------------------------------------------

        self.continuous = True
        self.running = True
        self.paused = False

        # ----------------------------------------------------
        # If current episode is already done,
        # move to a fresh episode.
        # ----------------------------------------------------

        if self.is_episode_done():

            if self.episode_summary is None:

                self._finalize_episode()

            self._start_next_episode()

        # ----------------------------------------------------
        # Execute EXACT requested number
        # ----------------------------------------------------

        last_result = None

        while (
            self.running
            and self.completed_episodes
            < target_completed
        ):

            # ------------------------------------------------
            # Respect pause/stop
            # ------------------------------------------------

            if self.paused:

                self.running = False
                break

            # ------------------------------------------------
            # Execute ONE environment step.
            #
            # auto_start=False is critical.
            # ------------------------------------------------

            last_result = self.step(
                sleep_seconds=delay,
                auto_start=False,
            )

            # ------------------------------------------------
            # Episode completed
            # ------------------------------------------------

            if last_result.get(
                "episode_done",
                False,
            ):

                # --------------------------------------------
                # Requested target reached.
                #
                # DO NOT start another episode.
                # --------------------------------------------

                if (
                    self.completed_episodes
                    >= target_completed
                ):

                    self.running = False

                    break

                # --------------------------------------------
                # More episodes requested.
                # --------------------------------------------

                self._start_next_episode()

        # ----------------------------------------------------
        # End auto run
        # ----------------------------------------------------

        self.running = False
        self.paused = False
        self.continuous = False

        # ----------------------------------------------------
        # Episodes completed during this call
        # ----------------------------------------------------

        completed_this_run = (
            self.completed_episodes
            - start_completed
        )

        # ----------------------------------------------------
        # Episode history generated during this run
        # ----------------------------------------------------

        run_history = (
            self.episode_history[
                start_completed:
            ]
        )

        # ----------------------------------------------------
        # Final snapshot
        # ----------------------------------------------------

        final_snapshot = (
            self.snapshot()
        )

        final_telemetry = (
            self._build_environment_telemetry(
                self.last_info
            )
        )

        return {

            "telemetry":
                final_telemetry,

            "episode_summary":
                json_safe(
                    self.episode_summary
                ),

            "episode_history":
                json_safe(
                    run_history
                ),

            "completed_episodes":
                self.completed_episodes,

            "requested_episodes":
                num_episodes,

            "episodes_completed_this_run":
                completed_this_run,

            "running":
                self.running,

            "continuous":
                self.continuous,

            "snapshot":
                final_snapshot,

            "last_result":
                json_safe(
                    last_result
                )
                if last_result is not None
                else None,
        }

    # ========================================================
    # Episode status
    # ========================================================

    def is_episode_done(
        self,
    ) -> bool:

        return bool(
            self.terminated
            or self.truncated
            or (
                self.step_count
                >= self.mode_config.max_steps
            )
        )

    # ========================================================
    # Getters
    # ========================================================

    def get_state(self):

        return self.state

    def get_action(self):

        return self.last_action

    def get_telemetry(self):

        return (
            self._build_environment_telemetry(
                self.last_info
            )
        )

    def get_episode_history(self):

        return json_safe(
            self.episode_history
        )

    def get_step_history(self):

        return json_safe(
            self.step_history
        )

    def get_episode_summary(self):

        return json_safe(
            self.episode_summary
        )

    def get_episode_reward(self):

        return safe_float(
            self.episode_reward
        )

    def get_completed_episodes(self):

        return int(
            self.completed_episodes
        )

    # ========================================================
    # Snapshot
    # ========================================================

    def snapshot(
        self,
    ) -> Dict[str, Any]:

        telemetry = (
            self._build_environment_telemetry(
                self.last_info
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Keep all keys expected by dashboard/app.py.
        # ----------------------------------------------------

        return {

            "mode":
                self.mode,

            "episode":
                self.current_episode,

            "current_episode":
                self.current_episode,

            "step":
                self.step_count,

            "max_steps":
                self.mode_config.max_steps,

            "episode_reward":
                safe_float(
                    self.episode_reward
                ),

            "last_reward":
                safe_float(
                    self.last_reward
                ),

            "completed_episodes":
                self.completed_episodes,

            "total_steps":
                self.total_steps,

            "episode_done":
                self.is_episode_done(),

            "running":
                self.running,

            "paused":
                self.paused,

            "continuous":
                self.continuous,

            "terminated":
                self.terminated,

            "truncated":
                self.truncated,

            "state_dim":
                self.mode_config.state_dim,

            "action_dim":
                self.mode_config.action_dim,

            "users":
                self.mode_config.num_users,

            "ris_elements":
                self.mode_config.num_ris_elements,

            "polarization_enabled":
                self.mode_config.polarization_enabled,

            "polarization_state_enabled":
                self.mode_config.polarization_state_enabled,

            "adaptive_trust":
                self.mode_config.adaptive_trust,

            "fixed_trust":
                self.mode_config.fixed_trust,

            "optimize_ris":
                self.mode_config.optimize_ris,

            "bandwidth_hz":
                self.mode_config.bandwidth_hz,

            "carrier_frequency_hz":
                self.mode_config.carrier_frequency_hz,

            # ------------------------------------------------
            # SAC
            # ------------------------------------------------

            "sac_available":
                bool(
                    self.sac_available
                ),

            "sac_source":
                self.sac_source,

            # ------------------------------------------------
            # Governance
            # ------------------------------------------------

            "governance_available":
                bool(
                    self.governance_available
                ),

            "governance_source":
                self.governance_source,

            # ------------------------------------------------
            # Checkpoint
            # ------------------------------------------------

            "checkpoint_path":
                self.checkpoint_path,

            "checkpoint_loaded":
                bool(
                    self.checkpoint_loaded
                ),

            # ------------------------------------------------
            # Episode
            # ------------------------------------------------

            "episode_summary":
                json_safe(
                    self.episode_summary
                ),

            # ------------------------------------------------
            # Telemetry
            # ------------------------------------------------

            "telemetry":
                telemetry,
        }

    # ========================================================
    # Configuration snapshot
    # ========================================================

    def get_configuration(self):

        return json_safe(
            asdict(
                self.mode_config
            )
        )