
"""
TA-FDRL-IRF
H5-C: GOVERNANCE CLOSED-LOOP BENCHMARK
======================================

Purpose
-------
Validate the interaction between the REAL governance runtime, the REAL
TrustEngine, and the REAL IRF execution environment under a controlled,
sequential closed-loop workload.

Closed-loop path
----------------

    Candidate Action
          |
          v
    REAL GovernanceEngine.evaluate()
          |
     +----+---------+
     |              |
   ALLOW        CONSTRAIN / BLOCK
     |              |
     |         +----+----------------+
     |         |                     |
     |      CONSTRAIN              BLOCK
     |         |                     |
     |    SafeEnvelope          Safe fallback
     |         |                     |
     +---------+---------------------+
                       |
                       v
                Executed Action
                       |
                       v
              REAL IRFEnvironment
                       |
                       v
                Execution Outcome
                       |
                       v
       REAL TrustEngine.post_execution_feedback()
                       |
                       v
              Updated Trust State
                       |
                       v
                  Next step


Research scope
--------------
H5-C validates a TRUE sequential closed loop:

    Candidate Action
        ->
    Governance Decision
        ->
    Actual Executed Action
        ->
    Real IRF Environment
        ->
    Real Execution Outcome
        ->
    Real Trust Feedback
        ->
    Updated Trust State
        ->
    Next Governance Decision

This benchmark does NOT:

    - modify GovernanceEngine
    - modify RiskEngine
    - modify PolicyEngine
    - modify TrustEngine
    - modify SafeEnvelope
    - train SAC
    - fabricate governance decisions
    - fabricate environment rewards
    - directly mutate TrustEngine.trust_score
    - inject artificial RiskAssessment objects
    - claim physical-world 6G validation


Execution policy
----------------
ALLOW
    -> execute REAL GovernanceEngine decision.action

CONSTRAIN
    -> execute REAL GovernanceEngine decision.action
       where the action is expected to be SafeEnvelope-modified

BLOCK
    -> candidate action MUST NOT execute
    -> zero/safe fallback executes to preserve environment continuity
    -> fallback execution is NOT candidate execution

Trust feedback
--------------
Only the ACTUAL action reaching the environment is passed to:

    TrustEngine.post_execution_feedback()

Therefore:

    ALLOW:
        executed_action = governance decision.action

    CONSTRAIN:
        executed_action = SafeEnvelope-modified governance action

    BLOCK:
        executed_action = safe fallback action

No direct trust_score mutation is performed.


Sequential closed-loop rule
---------------------------
The scenario's initial trust is externally controlled only for the first
governance decision.

After each successful REAL TrustEngine.post_execution_feedback():

    current_trust = REAL TrustEngine.trust_score

The next timestep uses this updated trust value.

This prevents the benchmark from repeatedly overwriting the TrustEngine's
learned governance trust during a sequential episode.


Reproducibility
---------------
Each scenario receives:

    - fresh REAL IRFEnvironment
    - fresh REAL TrustEngine
    - fresh REAL GovernanceEngine

Within each scenario:

    - GovernanceEngine remains persistent
    - TrustEngine remains persistent
    - feedback is applied sequentially
    - updated trust influences later decisions


Environment import policy
-------------------------
The benchmark refuses to create a fake/stub environment.

It first tries known repository module layouts.

If those fail, it scans the project source tree for a Python module
containing both:

    class IRFEnvironment
    class IRFConfig

If found, the real implementation is loaded directly.

If no real implementation can be found, the benchmark FAILS.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# PROJECT ROOT
# ============================================================================

CURRENT_FILE = Path(__file__).resolve()
EXPERIMENTS_DIR = CURRENT_FILE.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# REAL GOVERNANCE IMPORTS
# ============================================================================

from trust.governance import GovernanceEngine

from trust.trust_engine import (
    TrustAssessment,
    TrustEngine,
    TrustEngineConfig,
)


# ============================================================================
# IRF ENVIRONMENT DISCOVERY
# ============================================================================

IRFEnvironment = None
IRFConfig = None

_ENV_MODULE_NAME: Optional[str] = None
_ENV_FILE_PATH: Optional[str] = None
_ENV_IMPORT_ERRORS: List[str] = []


KNOWN_ENV_MODULES = (
    "environment.irf_environment",
    "environment.irf_env",
    "environment",
    "env.irf_environment",
    "env.irf_env",
    "env",
    "irf.environment",
    "irf.irf_environment",
    "irf_environment",
    "environments.irf_environment",
    "environments.environment",
    "src.environment.irf_environment",
    "src.environment.irf_env",
    "src.environment",
)


def _load_environment_from_module(
    module_name: str,
) -> bool:

    global IRFEnvironment
    global IRFConfig
    global _ENV_MODULE_NAME

    try:

        module = importlib.import_module(
            module_name
        )

        environment_class = getattr(
            module,
            "IRFEnvironment",
            None,
        )

        config_class = getattr(
            module,
            "IRFConfig",
            None,
        )

        if environment_class is None:
            return False

        if config_class is None:
            return False

        IRFEnvironment = environment_class
        IRFConfig = config_class
        _ENV_MODULE_NAME = module_name

        return True

    except Exception as exc:

        _ENV_IMPORT_ERRORS.append(
            f"{module_name}: "
            f"{type(exc).__name__}: {exc}"
        )

        return False


def _file_defines_required_classes(
    path: Path,
) -> bool:

    try:

        source = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        tree = ast.parse(
            source,
            filename=str(path),
        )

    except Exception:

        return False

    found_environment = False
    found_config = False

    for node in ast.walk(tree):

        if isinstance(node, ast.ClassDef):

            if node.name == "IRFEnvironment":
                found_environment = True

            elif node.name == "IRFConfig":
                found_config = True

    return (
        found_environment
        and found_config
    )


def _load_environment_from_file(
    path: Path,
) -> bool:

    global IRFEnvironment
    global IRFConfig
    global _ENV_MODULE_NAME
    global _ENV_FILE_PATH

    try:

        module_name = (
            "_ta_fdrl_irf_real_environment_"
            + str(
                abs(
                    hash(
                        str(path)
                    )
                )
            )
        )

        spec = importlib.util.spec_from_file_location(
            module_name,
            str(path),
        )

        if spec is None:
            return False

        if spec.loader is None:
            return False

        module = importlib.util.module_from_spec(
            spec
        )

        sys.modules[module_name] = module

        spec.loader.exec_module(module)

        environment_class = getattr(
            module,
            "IRFEnvironment",
            None,
        )

        config_class = getattr(
            module,
            "IRFConfig",
            None,
        )

        if environment_class is None:
            return False

        if config_class is None:
            return False

        IRFEnvironment = environment_class
        IRFConfig = config_class

        _ENV_MODULE_NAME = module_name
        _ENV_FILE_PATH = str(path)

        return True

    except Exception as exc:

        _ENV_IMPORT_ERRORS.append(
            f"{path}: "
            f"{type(exc).__name__}: {exc}"
        )

        return False


def discover_real_environment() -> bool:

    # ------------------------------------------------------------------------
    # Known imports
    # ------------------------------------------------------------------------

    for module_name in KNOWN_ENV_MODULES:

        if _load_environment_from_module(
            module_name
        ):
            return True

    # ------------------------------------------------------------------------
    # Source scan
    # ------------------------------------------------------------------------

    excluded_directories = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "site-packages",
        "dist",
        "build",
        "experiments",
        "results",
    }

    python_files: List[Path] = []

    try:

        for path in PROJECT_ROOT.rglob("*.py"):

            if path == CURRENT_FILE:
                continue

            if any(
                part in excluded_directories
                for part in path.parts
            ):
                continue

            python_files.append(path)

    except Exception as exc:

        _ENV_IMPORT_ERRORS.append(
            "Source scan failure: "
            f"{type(exc).__name__}: {exc}"
        )

        return False

    python_files.sort(
        key=lambda p: (
            0
            if (
                "irf" in p.name.lower()
                or "environment" in p.name.lower()
                or "env" in p.name.lower()
            )
            else 1,
            len(p.parts),
            str(p),
        )
    )

    for path in python_files:

        if not _file_defines_required_classes(
            path
        ):
            continue

        if _load_environment_from_file(
            path
        ):
            return True

    return False


ENVIRONMENT_AVAILABLE = (
    discover_real_environment()
)


# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================

EXPECTED_STATE_DIM = 100
EXPECTED_ACTION_DIM = 40

NUM_USERS = 20
MAX_STEPS = 20

MINIMUM_TRUST = 0.60

ACTION_EQUAL_ATOL = 1e-7


SCENARIOS = (
    "SAFE_ALLOW",
    "MEDIUM_RISK_CONSTRAIN",
    "POLICY_BLOCK",
    "LOW_TRUST_BLOCK",
    "HIGH_RISK_BLOCK",
    "MIXED_SEQUENCE",
)


SCENARIO_SEEDS = {
    "SAFE_ALLOW": 101,
    "MEDIUM_RISK_CONSTRAIN": 102,
    "POLICY_BLOCK": 103,
    "LOW_TRUST_BLOCK": 104,
    "HIGH_RISK_BLOCK": 105,
    "MIXED_SEQUENCE": 106,
}


# ============================================================================
# RESULT DATACLASSES
# ============================================================================

@dataclass
class StepResult:

    seed: int
    step: int

    controlled_trust: float
    environment_trust: float
    governance_trust: float

    candidate_action_magnitude: float
    executed_action_magnitude: float

    candidate_action: np.ndarray
    executed_action: np.ndarray
    decision_action: np.ndarray
    fallback_action: np.ndarray

    queue_pressure: float
    interference_pressure: float

    governance_status: str
    governance_reason: str

    policy_allowed: bool
    action_modified: bool

    risk_score: float
    risk_level: str

    candidate_executed: bool
    fallback_executed: bool

    reward: float
    done: bool

    trust_before_feedback: float
    trust_after_feedback: float

    environment_step_success: bool
    feedback_success: bool

    history_trust_length_after: int
    history_action_length_after: int
    history_risk_length_after: int
    history_policy_length_after: int
    history_outcome_length_after: int

    error: Optional[str] = None


@dataclass
class EpisodeResult:

    scenario: str
    seed: int

    steps_requested: int
    steps_completed: int

    allow_count: int = 0
    constrain_count: int = 0
    block_count: int = 0

    candidate_execution_count: int = 0
    fallback_execution_count: int = 0

    feedback_call_count: int = 0
    feedback_success_count: int = 0
    feedback_failure_count: int = 0

    trust_initial: float = 0.0
    trust_final: float = 0.0

    total_reward: float = 0.0
    mean_reward: float = 0.0

    mean_risk: float = 0.0
    max_risk: float = 0.0

    action_modification_count: int = 0

    environment_steps: int = 0
    environment_failures: int = 0

    trust_history_length: int = 0
    action_history_length: int = 0
    risk_history_length: int = 0
    policy_history_length: int = 0
    outcome_history_length: int = 0

    passed: bool = False

    errors: List[str] = field(
        default_factory=list
    )

    steps: List[StepResult] = field(
        default_factory=list
    )


# ============================================================================
# UTILITIES
# ============================================================================

def banner(
    title: str,
) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        result = float(value)

        if np.isfinite(result):
            return result

    except Exception:
        pass

    return default


def clip01(
    value: Any,
) -> float:

    return float(
        np.clip(
            safe_float(value),
            0.0,
            1.0,
        )
    )


def action_magnitude(
    action: np.ndarray,
) -> float:

    array = np.asarray(
        action,
        dtype=np.float64,
    )

    if array.size == 0:
        return 0.0

    return float(
        np.mean(
            np.abs(array)
        )
    )


def actions_equal(
    left: np.ndarray,
    right: np.ndarray,
    atol: float = ACTION_EQUAL_ATOL,
) -> bool:

    try:

        left_array = np.asarray(
            left,
            dtype=np.float64,
        ).reshape(-1)

        right_array = np.asarray(
            right,
            dtype=np.float64,
        ).reshape(-1)

        if left_array.shape != right_array.shape:
            return False

        return bool(
            np.allclose(
                left_array,
                right_array,
                atol=atol,
                rtol=0.0,
            )
        )

    except Exception:

        return False


def make_action(
    magnitude: float,
    action_dim: int = EXPECTED_ACTION_DIM,
) -> np.ndarray:

    return np.full(
        action_dim,
        float(magnitude),
        dtype=np.float64,
    )


def make_safe_fallback(
    action_dim: int = EXPECTED_ACTION_DIM,
) -> np.ndarray:

    return np.zeros(
        action_dim,
        dtype=np.float64,
    )


def make_telemetry(
    queue_pressure: float,
    interference_pressure: float,
) -> Dict[str, np.ndarray]:

    return {
        "queue": np.full(
            NUM_USERS,
            clip01(queue_pressure),
            dtype=np.float64,
        ),
        "interference": np.full(
            NUM_USERS,
            clip01(interference_pressure),
            dtype=np.float64,
        ),
    }


def extract_numeric_array(
    value: Any,
    default_size: int,
) -> np.ndarray:

    try:

        array = np.asarray(
            value,
            dtype=np.float64,
        )

        if array.ndim == 0:

            return np.full(
                default_size,
                float(array),
                dtype=np.float64,
            )

        return array.reshape(-1)

    except Exception:

        return np.zeros(
            default_size,
            dtype=np.float64,
        )


# ============================================================================
# REAL ENVIRONMENT
# ============================================================================

def create_irf_environment(
    seed: int,
):

    if IRFEnvironment is None:

        raise RuntimeError(
            "Real IRFEnvironment is unavailable."
        )

    config = None

    if IRFConfig is not None:

        config_attempts = (
            {
                "num_users": NUM_USERS,
                "max_steps": MAX_STEPS,
                "polarization_enabled": True,
                "optimize_ris": False,
                "fixed_trust": False,
            },
            {
                "num_users": NUM_USERS,
                "max_steps": MAX_STEPS,
            },
            {},
        )

        for kwargs in config_attempts:

            try:

                config = IRFConfig(
                    **kwargs
                )

                break

            except Exception:

                continue

    constructor_attempts = []

    if config is not None:

        constructor_attempts.extend(
            [
                (
                    (config,),
                    {"seed": seed},
                ),
                (
                    (config,),
                    {},
                ),
                (
                    (),
                    {
                        "config": config,
                        "seed": seed,
                    },
                ),
                (
                    (),
                    {
                        "config": config,
                    },
                ),
            ]
        )

    constructor_attempts.extend(
        [
            (
                (),
                {"seed": seed},
            ),
            (
                (),
                {},
            ),
        ]
    )

    last_error = None

    for args, kwargs in constructor_attempts:

        try:

            env = IRFEnvironment(
                *args,
                **kwargs,
            )

            try:

                if hasattr(
                    env,
                    "seed",
                ):

                    env.seed(
                        seed
                    )

            except Exception:

                pass

            return env

        except Exception as exc:

            last_error = exc

    raise RuntimeError(
        "Unable to construct the REAL IRFEnvironment. "
        f"Last error: {last_error}"
    )


# ============================================================================
# REAL GOVERNANCE
# ============================================================================

def create_governance(
    initial_trust: float,
) -> GovernanceEngine:

    trust_engine = TrustEngine(
        TrustEngineConfig(
            initial_trust=float(
                initial_trust
            ),
        )
    )

    governance = GovernanceEngine(
        trust_engine=trust_engine,
    )

    return governance


# ============================================================================
# ENVIRONMENT RESET
# ============================================================================

def reset_environment(
    env: Any,
    seed: int,
) -> Any:

    try:

        result = env.reset(
            seed=seed
        )

    except TypeError:

        result = env.reset()

    if isinstance(
        result,
        tuple,
    ):

        if len(result) >= 1:
            return result[0]

    return result


# ============================================================================
# ENVIRONMENT STEP
# ============================================================================

def environment_step(
    env: Any,
    action: np.ndarray,
) -> Tuple[
    Any,
    float,
    bool,
    Dict[str, Any],
]:

    result = env.step(
        action
    )

    if not isinstance(
        result,
        tuple,
    ):

        raise RuntimeError(
            "IRFEnvironment.step() returned "
            "a non-tuple result."
        )

    if len(result) == 3:

        next_state, reward, done = result

        return (
            next_state,
            safe_float(reward),
            bool(done),
            {},
        )

    if len(result) == 4:

        (
            next_state,
            reward,
            done,
            info,
        ) = result

        if not isinstance(
            info,
            dict,
        ):
            info = {}

        return (
            next_state,
            safe_float(reward),
            bool(done),
            info,
        )

    if len(result) == 5:

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = result

        if not isinstance(
            info,
            dict,
        ):
            info = {}

        return (
            next_state,
            safe_float(reward),
            bool(
                terminated
                or truncated
            ),
            info,
        )

    raise RuntimeError(
        "Unsupported IRFEnvironment.step() "
        f"return format: {len(result)} values."
    )


# ============================================================================
# TELEMETRY
# ============================================================================

def telemetry_from_environment(
    env: Any,
    state: Any,
    info: Optional[Dict[str, Any]],
) -> Dict[str, np.ndarray]:

    if info is None:
        info = {}

    queue_candidates = (
        "queue",
        "queues",
        "queue_pressure",
        "queue_state",
    )

    interference_candidates = (
        "interference",
        "interferences",
        "interference_power",
        "interference_pressure",
    )

    queue_value = None
    interference_value = None

    for key in queue_candidates:

        if key in info:

            queue_value = info[key]
            break

    for key in interference_candidates:

        if key in info:

            interference_value = info[key]
            break

    if (
        queue_value is not None
        and interference_value is not None
    ):

        return {
            "queue": extract_numeric_array(
                queue_value,
                NUM_USERS,
            ),
            "interference": extract_numeric_array(
                interference_value,
                NUM_USERS,
            ),
        }

    if queue_value is None:

        for name in queue_candidates:

            try:

                if not hasattr(
                    env,
                    name,
                ):
                    continue

                value = getattr(
                    env,
                    name,
                )

                if callable(value):

                    value = value()

                if value is not None:

                    queue_value = value
                    break

            except Exception:

                continue

    if interference_value is None:

        for name in interference_candidates:

            try:

                if not hasattr(
                    env,
                    name,
                ):
                    continue

                value = getattr(
                    env,
                    name,
                )

                if callable(value):

                    value = value()

                if value is not None:

                    interference_value = value
                    break

            except Exception:

                continue

    if (
        queue_value is not None
        and interference_value is not None
    ):

        return {
            "queue": extract_numeric_array(
                queue_value,
                NUM_USERS,
            ),
            "interference": extract_numeric_array(
                interference_value,
                NUM_USERS,
            ),
        }

    try:

        state_array = np.asarray(
            state,
            dtype=np.float64,
        ).reshape(-1)

        if state_array.size == EXPECTED_STATE_DIM:

            features = state_array.reshape(
                NUM_USERS,
                5,
            )

            return {
                "queue": features[:, 2],
                "interference": features[:, 1],
            }

    except Exception:

        pass

    return {
        "queue": np.zeros(
            NUM_USERS,
            dtype=np.float64,
        ),
        "interference": np.zeros(
            NUM_USERS,
            dtype=np.float64,
        ),
    }


# ============================================================================
# TRUST EXTRACTION
# ============================================================================

def get_environment_trust(
    env: Any,
    default: float,
) -> float:

    names = (
        "trust",
        "trust_score",
        "mean_trust",
        "current_trust",
    )

    for name in names:

        try:

            if not hasattr(
                env,
                name,
            ):
                continue

            value = getattr(
                env,
                name,
            )

            if callable(value):

                value = value()

            return clip01(value)

        except Exception:

            continue

    return clip01(
        default
    )


def get_trust_engine(
    governance: GovernanceEngine,
) -> TrustEngine:

    trust_engine = getattr(
        governance,
        "trust_engine",
        None,
    )

    if trust_engine is None:

        raise RuntimeError(
            "GovernanceEngine does not expose "
            "a TrustEngine."
        )

    if not isinstance(
        trust_engine,
        TrustEngine,
    ):

        raise RuntimeError(
            "GovernanceEngine.trust_engine is "
            f"not a REAL TrustEngine: "
            f"{type(trust_engine)}"
        )

    return trust_engine


def get_governance_trust(
    decision: Any,
    governance: GovernanceEngine,
    default: float,
) -> float:

    try:

        assessment = getattr(
            decision,
            "trust",
            None,
        )

        if assessment is not None:

            value = getattr(
                assessment,
                "trust_score",
                None,
            )

            if value is not None:

                return clip01(
                    value
                )

    except Exception:

        pass

    try:

        trust_engine = get_trust_engine(
            governance
        )

        return clip01(
            trust_engine.trust_score
        )

    except Exception:

        return clip01(
            default
        )


def get_current_trust(
    governance: GovernanceEngine,
    fallback: float,
) -> float:

    try:

        trust_engine = get_trust_engine(
            governance
        )

        return clip01(
            trust_engine.trust_score
        )

    except Exception:

        return clip01(
            fallback
        )


# ============================================================================
# TRUST HISTORY
# ============================================================================

def history_length(
    obj: Any,
    names: Tuple[str, ...],
) -> int:

    for name in names:

        try:

            value = getattr(
                obj,
                name,
            )

            if value is not None:

                return len(value)

        except Exception:

            continue

    return 0


def get_history_lengths(
    governance: GovernanceEngine,
) -> Dict[str, int]:

    trust_engine = get_trust_engine(
        governance
    )

    return {
        "trust_history_length": history_length(
            trust_engine,
            (
                "trust_history",
                "history",
            ),
        ),
        "action_history_length": history_length(
            trust_engine,
            (
                "action_history",
            ),
        ),
        "risk_history_length": history_length(
            trust_engine,
            (
                "risk_history",
            ),
        ),
        "policy_history_length": history_length(
            trust_engine,
            (
                "policy_history",
            ),
        ),
        "outcome_history_length": history_length(
            trust_engine,
            (
                "outcome_history",
            ),
        ),
    }


# ============================================================================
# REAL POST-EXECUTION TRUST FEEDBACK
# ============================================================================

def post_execution_feedback(
    governance: GovernanceEngine,
    *,
    reward: float,
    environment_trust: float,
    executed_action: np.ndarray,
    policy_allowed: bool,
    risk_score: float,
) -> Tuple[
    bool,
    Optional[TrustAssessment],
    Optional[str],
]:

    trust_engine = get_trust_engine(
        governance
    )

    method = getattr(
        trust_engine,
        "post_execution_feedback",
        None,
    )

    if method is None:

        return (
            False,
            None,
            (
                "REAL TrustEngine."
                "post_execution_feedback() "
                "is unavailable."
            ),
        )

    try:

        assessment = method(
            reward=float(
                reward
            ),
            environment_trust=float(
                environment_trust
            ),
            action=np.asarray(
                executed_action,
                dtype=np.float32,
            ),
            policy_allowed=bool(
                policy_allowed
            ),
            risk_score=float(
                risk_score
            ),
        )

        if not isinstance(
            assessment,
            TrustAssessment,
        ):

            return (
                False,
                None,
                (
                    "REAL TrustEngine."
                    "post_execution_feedback() "
                    "returned unexpected type: "
                    f"{type(assessment)}"
                ),
            )

        return (
            True,
            assessment,
            None,
        )

    except Exception as exc:

        return (
            False,
            None,
            (
                "REAL TrustEngine."
                "post_execution_feedback() "
                f"failed: {type(exc).__name__}: {exc}"
            ),
        )


# ============================================================================
# GOVERNANCE ORACLE
# ============================================================================

def expected_governance_reason(
    policy_allowed: bool,
    risk_level: str,
    governance_trust: float,
) -> str:

    if not policy_allowed:
        return "POLICY_VIOLATION"

    if risk_level == "HIGH":
        return "HIGH_RISK"

    if governance_trust < MINIMUM_TRUST:
        return "TRUST_BELOW_MINIMUM"

    if risk_level == "MEDIUM":
        return "MEDIUM_RISK_SAFE_ENVELOPE"

    return "GOVERNANCE_APPROVED"


def expected_governance_status(
    reason: str,
) -> str:

    if reason in {
        "POLICY_VIOLATION",
        "HIGH_RISK",
        "TRUST_BELOW_MINIMUM",
    }:

        return "BLOCK"

    if reason == "MEDIUM_RISK_SAFE_ENVELOPE":

        return "CONSTRAIN"

    return "ALLOW"


# ============================================================================
# GOVERNANCE EVALUATION
# ============================================================================

def evaluate_governance(
    governance: GovernanceEngine,
    action: np.ndarray,
    trust_score: float,
    telemetry: Dict[str, np.ndarray],
):

    return governance.evaluate(
        action=action,
        trust_score=float(
            trust_score
        ),
        telemetry=telemetry,
    )


# ============================================================================
# SCENARIO PARAMETERS
# ============================================================================

def scenario_parameters(
    scenario: str,
    step: int,
) -> Tuple[
    float,
    float,
    float,
    float,
]:

    if scenario == "SAFE_ALLOW":

        return (
            1.0,
            0.50,
            0.0,
            0.0,
        )

    if scenario == "MEDIUM_RISK_CONSTRAIN":

        return (
            0.60,
            0.90,
            0.0,
            1.0,
        )

    if scenario == "POLICY_BLOCK":

        return (
            1.0,
            1.00,
            0.0,
            0.0,
        )

    if scenario == "LOW_TRUST_BLOCK":

        return (
            0.50,
            0.0,
            0.0,
            0.0,
        )

    if scenario == "HIGH_RISK_BLOCK":

        return (
            0.0,
            0.90,
            1.0,
            1.0,
        )

    if scenario == "MIXED_SEQUENCE":

        sequence = (
            (
                1.0,
                0.50,
                0.0,
                0.0,
            ),
            (
                0.60,
                0.90,
                0.0,
                1.0,
            ),
            (
                1.0,
                1.00,
                0.0,
                0.0,
            ),
            (
                0.50,
                0.0,
                0.0,
                0.0,
            ),
            (
                0.0,
                0.90,
                1.0,
                1.0,
            ),
        )

        return sequence[
            step % len(sequence)
        ]

    raise ValueError(
        f"Unknown scenario: {scenario}"
    )


# ============================================================================
# ACTION EXECUTION VALIDATION
# ============================================================================

def validate_executed_action(
    status: str,
    candidate_action: np.ndarray,
    executed_action: np.ndarray,
    decision_action: np.ndarray,
    fallback_action: np.ndarray,
    candidate_executed: bool,
    fallback_executed: bool,
    action_modified: bool,
) -> Tuple[
    bool,
    Optional[str],
]:

    # ------------------------------------------------------------------------
    # ALLOW
    # ------------------------------------------------------------------------

    if status == "ALLOW":

        if not candidate_executed:

            return (
                False,
                "ALLOW candidate was not executed.",
            )

        if fallback_executed:

            return (
                False,
                "ALLOW incorrectly used fallback.",
            )

        if not actions_equal(
            executed_action,
            decision_action,
        ):

            return (
                False,
                "ALLOW did not execute governance.action.",
            )

        # The benchmark does NOT require candidate_action to be numerically
        # identical to decision_action. The authoritative execution invariant
        # is that the action reaching the environment is the REAL governance
        # decision action.

        return (
            True,
            None,
        )

    # ------------------------------------------------------------------------
    # CONSTRAIN
    # ------------------------------------------------------------------------

    if status == "CONSTRAIN":

        if not candidate_executed:

            return (
                False,
                "CONSTRAIN candidate was not executed.",
            )

        if fallback_executed:

            return (
                False,
                "CONSTRAIN incorrectly used fallback.",
            )

        if not action_modified:

            return (
                False,
                "CONSTRAIN decision is not marked modified.",
            )

        if not actions_equal(
            executed_action,
            decision_action,
        ):

            return (
                False,
                "CONSTRAIN did not execute governance.action.",
            )

        if actions_equal(
            executed_action,
            candidate_action,
        ):

            return (
                False,
                "CONSTRAIN action was not modified.",
            )

        return (
            True,
            None,
        )

    # ------------------------------------------------------------------------
    # BLOCK
    # ------------------------------------------------------------------------

    if status == "BLOCK":

        if candidate_executed:

            return (
                False,
                "BLOCK candidate action was executed.",
            )

        if not fallback_executed:

            return (
                False,
                "BLOCK did not execute safe fallback.",
            )

        if not actions_equal(
            executed_action,
            fallback_action,
        ):

            return (
                False,
                "BLOCK did not execute the designated fallback action.",
            )

        # IMPORTANT:
        #
        # fallback_action is allowed to be numerically identical to
        # candidate_action. This is possible when the candidate itself is
        # already zero-valued.
        #
        # BLOCK correctness is determined by the execution gate:
        #
        #     candidate_executed == False
        #     fallback_executed == True

        return (
            True,
            None,
        )

    return (
        False,
        f"Unknown governance status: {status}",
    )


# ============================================================================
# SINGLE CLOSED-LOOP EPISODE
# ============================================================================

def run_episode(
    scenario: str,
    seed: int,
    steps: int = MAX_STEPS,
    verbose: bool = False,
) -> EpisodeResult:

    env = create_irf_environment(
        seed
    )

    initial_trust, _, _, _ = (
        scenario_parameters(
            scenario,
            0,
        )
    )

    governance = create_governance(
        initial_trust=initial_trust
    )

    state = reset_environment(
        env,
        seed,
    )

    result = EpisodeResult(
        scenario=scenario,
        seed=seed,
        steps_requested=steps,
        steps_completed=0,
        trust_initial=initial_trust,
    )

    rewards: List[float] = []
    risks: List[float] = []

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # The first step receives the scenario's initial trust.
    #
    # After feedback, current_trust is read from the REAL TrustEngine.
    #
    # That value becomes the next timestep's governance input.
    # ------------------------------------------------------------------------

    current_trust = clip01(
        initial_trust
    )

    # =========================================================================
    # TIMESTEPS
    # =========================================================================

    for step in range(steps):

        (
            scenario_trust,
            magnitude,
            queue,
            interference,
        ) = scenario_parameters(
            scenario,
            step,
        )

        # ---------------------------------------------------------------------
        # Sequential trust control
        # ---------------------------------------------------------------------
        #
        # First timestep:
        #     use controlled scenario initial trust.
        #
        # Subsequent timesteps:
        #     use REAL TrustEngine trust.
        #
        # MIXED_SEQUENCE therefore remains a true closed loop.
        # The scenario sequence controls action/telemetry conditions, but does
        # not overwrite the learned trust state after the first timestep.
        # ---------------------------------------------------------------------

        if step == 0:

            controlled_trust = clip01(
                scenario_trust
            )

        else:

            controlled_trust = clip01(
                current_trust
            )

        # ---------------------------------------------------------------------
        # Candidate action
        # ---------------------------------------------------------------------

        candidate_action = make_action(
            magnitude,
            EXPECTED_ACTION_DIM,
        )

        # ---------------------------------------------------------------------
        # Candidate telemetry
        # ---------------------------------------------------------------------

        telemetry = make_telemetry(
            queue,
            interference,
        )

        # ---------------------------------------------------------------------
        # REAL GOVERNANCE
        # ---------------------------------------------------------------------

        try:

            decision = evaluate_governance(
                governance=governance,
                action=candidate_action,
                trust_score=controlled_trust,
                telemetry=telemetry,
            )

        except Exception as exc:

            result.errors.append(
                f"Step {step}: "
                f"governance evaluation failed: {exc}"
            )

            break

        status = str(
            getattr(
                decision,
                "status",
                "UNKNOWN",
            )
        )

        reason = str(
            getattr(
                decision,
                "reason",
                "UNKNOWN",
            )
        )

        policy = getattr(
            decision,
            "policy",
            None,
        )

        risk = getattr(
            decision,
            "risk",
            None,
        )

        policy_allowed = bool(
            getattr(
                policy,
                "allowed",
                False,
            )
        )

        risk_score = clip01(
            getattr(
                risk,
                "risk_score",
                0.0,
            )
        )

        risk_level = str(
            getattr(
                risk,
                "risk_level",
                "UNKNOWN",
            )
        )

        governance_trust = (
            get_governance_trust(
                decision,
                governance,
                controlled_trust,
            )
        )

        environment_trust = (
            get_environment_trust(
                env,
                controlled_trust,
            )
        )

        modified = bool(
            getattr(
                decision,
                "modified",
                False,
            )
        )

        decision_action = np.asarray(
            getattr(
                decision,
                "action",
                candidate_action,
            ),
            dtype=np.float64,
        ).reshape(-1)

        # ---------------------------------------------------------------------
        # Dimension validation
        # ---------------------------------------------------------------------

        if candidate_action.size != EXPECTED_ACTION_DIM:

            result.errors.append(
                f"Step {step}: candidate action dimension "
                f"{candidate_action.size} != "
                f"{EXPECTED_ACTION_DIM}"
            )

        if decision_action.size != EXPECTED_ACTION_DIM:

            result.errors.append(
                f"Step {step}: decision action dimension "
                f"{decision_action.size} != "
                f"{EXPECTED_ACTION_DIM}"
            )

        # ---------------------------------------------------------------------
        # Governance oracle
        #
        # IMPORTANT:
        #
        # This is ONLY a validation oracle.
        #
        # The actual governance decision still comes from the REAL
        # GovernanceEngine.
        # ---------------------------------------------------------------------

        expected_reason = (
            expected_governance_reason(
                policy_allowed,
                risk_level,
                governance_trust,
            )
        )

        expected_status = (
            expected_governance_status(
                expected_reason
            )
        )

        if status != expected_status:

            result.errors.append(
                f"Step {step}: status mismatch; "
                f"actual={status}, "
                f"expected={expected_status}"
            )

        if reason != expected_reason:

            result.errors.append(
                f"Step {step}: reason mismatch; "
                f"actual={reason}, "
                f"expected={expected_reason}"
            )

        # ---------------------------------------------------------------------
        # EXECUTION GATE
        # ---------------------------------------------------------------------

        fallback_action = (
            make_safe_fallback(
                EXPECTED_ACTION_DIM
            )
        )

        if status == "ALLOW":

            executed_action = (
                decision_action.copy()
            )

            candidate_executed = True
            fallback_executed = False

            result.allow_count += 1

        elif status == "CONSTRAIN":

            executed_action = (
                decision_action.copy()
            )

            candidate_executed = True
            fallback_executed = False

            result.constrain_count += 1

        elif status == "BLOCK":

            executed_action = (
                fallback_action.copy()
            )

            candidate_executed = False
            fallback_executed = True

            result.block_count += 1

        else:

            result.errors.append(
                f"Step {step}: invalid governance "
                f"status={status}"
            )

            break

        # ---------------------------------------------------------------------
        # Execution accounting
        # ---------------------------------------------------------------------

        if candidate_executed:

            result.candidate_execution_count += 1

        if fallback_executed:

            result.fallback_execution_count += 1

        if modified:

            result.action_modification_count += 1

        # ---------------------------------------------------------------------
        # Execution gate validation
        # ---------------------------------------------------------------------

        gate_ok, gate_error = (
            validate_executed_action(
                status=status,
                candidate_action=candidate_action,
                executed_action=executed_action,
                decision_action=decision_action,
                fallback_action=fallback_action,
                candidate_executed=candidate_executed,
                fallback_executed=fallback_executed,
                action_modified=modified,
            )
        )

        if not gate_ok:

            result.errors.append(
                f"Step {step}: {gate_error}"
            )

        # ---------------------------------------------------------------------
        # REAL IRF ENVIRONMENT
        # ---------------------------------------------------------------------

        environment_success = False
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        try:

            (
                next_state,
                reward,
                done,
                info,
            ) = environment_step(
                env,
                executed_action,
            )

            environment_success = True

            result.environment_steps += 1

            rewards.append(
                safe_float(
                    reward
                )
            )

        except Exception as exc:

            result.environment_failures += 1

            result.errors.append(
                f"Step {step}: environment step failed: "
                f"{exc}"
            )

            next_state = state

        # ---------------------------------------------------------------------
        # REAL outcome telemetry
        # ---------------------------------------------------------------------

        outcome_telemetry = (
            telemetry_from_environment(
                env,
                next_state,
                info,
            )
        )

        outcome = {
            "executed": bool(
                candidate_executed
            ),
            "fallback_executed": bool(
                fallback_executed
            ),
            "governance_status": status,
            "governance_reason": reason,
            "reward": safe_float(
                reward
            ),
            "done": bool(
                done
            ),
            "environment_step_success": bool(
                environment_success
            ),
            "telemetry": outcome_telemetry,
        }

        # Keep the outcome object as explicit evidence of what happened.
        # The REAL TrustEngine receives the actual executed action and the
        # real execution-derived reward/risk/policy values.
        _ = outcome

        # ---------------------------------------------------------------------
        # Trust before feedback
        # ---------------------------------------------------------------------

        trust_before_feedback = (
            get_current_trust(
                governance,
                governance_trust,
            )
        )

        trust_after_feedback = (
            trust_before_feedback
        )

        feedback_success = False

        # ---------------------------------------------------------------------
        # REAL TRUST FEEDBACK
        # ---------------------------------------------------------------------

        if environment_success:

            result.feedback_call_count += 1

            (
                feedback_ok,
                trust_assessment,
                feedback_error,
            ) = post_execution_feedback(
                governance=governance,
                reward=reward,
                environment_trust=environment_trust,
                executed_action=executed_action,
                policy_allowed=policy_allowed,
                risk_score=risk_score,
            )

            if feedback_ok:

                feedback_success = True

                result.feedback_success_count += 1

                trust_after_feedback = clip01(
                    trust_assessment.trust_score
                )

                # ----------------------------------------------------------------
                # CRITICAL SEQUENTIAL STATE UPDATE
                # ----------------------------------------------------------------
                #
                # The next timestep receives the REAL TrustEngine state.
                #
                # No direct trust_score mutation is performed.
                # ----------------------------------------------------------------

                current_trust = (
                    get_current_trust(
                        governance,
                        trust_after_feedback,
                    )
                )

                if not np.isclose(
                    current_trust,
                    trust_after_feedback,
                    atol=1e-9,
                    rtol=0.0,
                ):

                    result.errors.append(
                        f"Step {step}: TrustAssessment trust "
                        "does not match REAL TrustEngine state."
                    )

            else:

                result.feedback_failure_count += 1

                result.errors.append(
                    f"Step {step}: "
                    f"trust feedback failed: "
                    f"{feedback_error}"
                )

        # ---------------------------------------------------------------------
        # Verify REAL TrustEngine state
        # ---------------------------------------------------------------------

        try:

            trust_engine = get_trust_engine(
                governance
            )

            actual_trust = clip01(
                trust_engine.trust_score
            )

            if feedback_success:

                if not np.isclose(
                    actual_trust,
                    trust_after_feedback,
                    atol=1e-9,
                    rtol=0.0,
                ):

                    result.errors.append(
                        f"Step {step}: TrustAssessment "
                        "does not match TrustEngine.trust_score"
                    )

        except Exception as exc:

            result.errors.append(
                f"Step {step}: TrustEngine state "
                f"verification failed: {exc}"
            )

        # ---------------------------------------------------------------------
        # Trust histories
        # ---------------------------------------------------------------------

        history = get_history_lengths(
            governance
        )

        # ---------------------------------------------------------------------
        # Step result
        # ---------------------------------------------------------------------

        step_result = StepResult(
            seed=seed,
            step=step,

            controlled_trust=controlled_trust,
            environment_trust=environment_trust,
            governance_trust=governance_trust,

            candidate_action_magnitude=(
                action_magnitude(
                    candidate_action
                )
            ),

            executed_action_magnitude=(
                action_magnitude(
                    executed_action
                )
            ),

            candidate_action=(
                candidate_action.copy()
            ),

            executed_action=(
                executed_action.copy()
            ),

            decision_action=(
                decision_action.copy()
            ),

            fallback_action=(
                fallback_action.copy()
            ),

            queue_pressure=queue,
            interference_pressure=interference,

            governance_status=status,
            governance_reason=reason,

            policy_allowed=policy_allowed,
            action_modified=modified,

            risk_score=risk_score,
            risk_level=risk_level,

            candidate_executed=candidate_executed,
            fallback_executed=fallback_executed,

            reward=safe_float(
                reward
            ),
            done=bool(
                done
            ),

            trust_before_feedback=(
                trust_before_feedback
            ),

            trust_after_feedback=(
                trust_after_feedback
            ),

            environment_step_success=(
                environment_success
            ),

            feedback_success=(
                feedback_success
            ),

            history_trust_length_after=(
                history[
                    "trust_history_length"
                ]
            ),

            history_action_length_after=(
                history[
                    "action_history_length"
                ]
            ),

            history_risk_length_after=(
                history[
                    "risk_history_length"
                ]
            ),

            history_policy_length_after=(
                history[
                    "policy_history_length"
                ]
            ),

            history_outcome_length_after=(
                history[
                    "outcome_history_length"
                ]
            ),
        )

        result.steps.append(
            step_result
        )

        risks.append(
            risk_score
        )

        result.steps_completed += 1

        state = next_state

        if verbose:

            print(
                f"  step={step:02d} "
                f"trust={governance_trust:.3f} "
                f"post={trust_after_feedback:.3f} "
                f"risk={risk_score:.3f} "
                f"{risk_level:<6} "
                f"{status:<9} "
                f"feedback="
                f"{'YES' if feedback_success else 'NO'} "
                f"{reason}"
            )

        if done:

            break

    # =========================================================================
    # METRICS
    # =========================================================================

    result.total_reward = (
        float(
            np.sum(
                rewards
            )
        )
        if rewards
        else 0.0
    )

    result.mean_reward = (
        float(
            np.mean(
                rewards
            )
        )
        if rewards
        else 0.0
    )

    result.mean_risk = (
        float(
            np.mean(
                risks
            )
        )
        if risks
        else 0.0
    )

    result.max_risk = (
        float(
            np.max(
                risks
            )
        )
        if risks
        else 0.0
    )

    # -------------------------------------------------------------------------
    # Final trust comes from the REAL TrustEngine whenever possible.
    # -------------------------------------------------------------------------

    try:

        result.trust_final = get_current_trust(
            governance,
            result.trust_initial,
        )

    except Exception:

        if result.steps:

            result.trust_final = (
                result.steps[-1]
                .trust_after_feedback
            )

        else:

            result.trust_final = (
                result.trust_initial
            )

    # =========================================================================
    # FINAL TRUST ENGINE HISTORY
    # =========================================================================

    history = get_history_lengths(
        governance
    )

    result.trust_history_length = (
        history[
            "trust_history_length"
        ]
    )

    result.action_history_length = (
        history[
            "action_history_length"
        ]
    )

    result.risk_history_length = (
        history[
            "risk_history_length"
        ]
    )

    result.policy_history_length = (
        history[
            "policy_history_length"
        ]
    )

    result.outcome_history_length = (
        history[
            "outcome_history_length"
        ]
    )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    result.passed = validate_episode(
        result
    )

    return result


# ============================================================================
# EPISODE VALIDATION
# ============================================================================

def validate_episode(
    result: EpisodeResult,
) -> bool:

    checks: List[bool] = []

    # ------------------------------------------------------------------------
    # Basic execution
    # ------------------------------------------------------------------------

    checks.append(
        result.steps_completed > 0
    )

    checks.append(
        result.environment_failures == 0
    )

    checks.append(
        result.environment_steps
        == result.steps_completed
    )

    # ------------------------------------------------------------------------
    # Valid governance statuses
    # ------------------------------------------------------------------------

    checks.append(
        all(
            step.governance_status
            in {
                "ALLOW",
                "CONSTRAIN",
                "BLOCK",
            }
            for step in result.steps
        )
    )

    # ------------------------------------------------------------------------
    # Every successful environment execution must have feedback
    # ------------------------------------------------------------------------

    successful_environment_steps = sum(
        bool(
            step.environment_step_success
        )
        for step in result.steps
    )

    successful_feedback_steps = sum(
        bool(
            step.feedback_success
        )
        for step in result.steps
    )

    checks.append(
        result.feedback_call_count
        == successful_environment_steps
    )

    checks.append(
        result.feedback_success_count
        == successful_feedback_steps
    )

    checks.append(
        result.feedback_failure_count == 0
    )

    checks.append(
        successful_feedback_steps
        == successful_environment_steps
    )

    # ------------------------------------------------------------------------
    # Execution accounting
    # ------------------------------------------------------------------------

    checks.append(
        (
            result.candidate_execution_count
            + result.fallback_execution_count
        )
        == result.environment_steps
    )

    checks.append(
        result.candidate_execution_count
        == (
            result.allow_count
            + result.constrain_count
        )
    )

    checks.append(
        result.fallback_execution_count
        == result.block_count
    )

    # ------------------------------------------------------------------------
    # BLOCK
    # ------------------------------------------------------------------------

    blocked_steps = [
        step
        for step in result.steps
        if step.governance_status == "BLOCK"
    ]

    checks.append(
        all(
            not step.candidate_executed
            for step in blocked_steps
        )
    )

    checks.append(
        all(
            step.fallback_executed
            for step in blocked_steps
        )
    )

    checks.append(
        all(
            actions_equal(
                step.executed_action,
                step.fallback_action,
            )
            for step in blocked_steps
        )
    )

    # ------------------------------------------------------------------------
    # ALLOW
    # ------------------------------------------------------------------------

    allow_steps = [
        step
        for step in result.steps
        if step.governance_status == "ALLOW"
    ]

    checks.append(
        all(
            step.candidate_executed
            for step in allow_steps
        )
    )

    checks.append(
        all(
            not step.fallback_executed
            for step in allow_steps
        )
    )

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # The authoritative ALLOW execution invariant is:
    #
    #     executed_action == decision_action
    #
    # We do NOT require:
    #
    #     candidate_action == executed_action
    #
    # because the real GovernanceEngine owns the authoritative governed
    # action representation.
    # ------------------------------------------------------------------------

    checks.append(
        all(
            actions_equal(
                step.executed_action,
                step.decision_action,
            )
            for step in allow_steps
        )
    )

    # ------------------------------------------------------------------------
    # CONSTRAIN
    # ------------------------------------------------------------------------

    constrain_steps = [
        step
        for step in result.steps
        if step.governance_status == "CONSTRAIN"
    ]

    checks.append(
        all(
            step.candidate_executed
            for step in constrain_steps
        )
    )

    checks.append(
        all(
            not step.fallback_executed
            for step in constrain_steps
        )
    )

    checks.append(
        all(
            step.action_modified
            for step in constrain_steps
        )
    )

    checks.append(
        all(
            not actions_equal(
                step.candidate_action,
                step.executed_action,
            )
            for step in constrain_steps
        )
    )

    checks.append(
        all(
            actions_equal(
                step.executed_action,
                step.decision_action,
            )
            for step in constrain_steps
        )
    )

    # ------------------------------------------------------------------------
    # Trust history
    #
    # REAL TrustEngine semantics:
    #
    # initial trust_history = [initial_trust]
    #
    # Every successful post_execution_feedback() adds one trust history item.
    #
    # Therefore:
    #
    #     trust_history = feedback_count + 1
    #
    # and execution histories:
    #
    #     action_history  = feedback_count
    #     risk_history    = feedback_count
    #     policy_history  = feedback_count
    #     outcome_history = feedback_count
    # ------------------------------------------------------------------------

    checks.append(
        result.trust_history_length
        == successful_feedback_steps + 1
    )

    checks.append(
        result.action_history_length
        == successful_feedback_steps
    )

    checks.append(
        result.risk_history_length
        == successful_feedback_steps
    )

    checks.append(
        result.policy_history_length
        == successful_feedback_steps
    )

    checks.append(
        result.outcome_history_length
        == successful_feedback_steps
    )

    # ------------------------------------------------------------------------
    # Step history monotonicity
    # ------------------------------------------------------------------------

    previous_trust_history = 0
    previous_action_history = 0
    previous_risk_history = 0
    previous_policy_history = 0
    previous_outcome_history = 0

    for step in result.steps:

        checks.append(
            step.history_trust_length_after
            >= previous_trust_history
        )

        checks.append(
            step.history_action_length_after
            >= previous_action_history
        )

        checks.append(
            step.history_risk_length_after
            >= previous_risk_history
        )

        checks.append(
            step.history_policy_length_after
            >= previous_policy_history
        )

        checks.append(
            step.history_outcome_length_after
            >= previous_outcome_history
        )

        previous_trust_history = (
            step.history_trust_length_after
        )

        previous_action_history = (
            step.history_action_length_after
        )

        previous_risk_history = (
            step.history_risk_length_after
        )

        previous_policy_history = (
            step.history_policy_length_after
        )

        previous_outcome_history = (
            step.history_outcome_length_after
        )

    # ------------------------------------------------------------------------
    # Feedback trust consistency
    # ------------------------------------------------------------------------

    checks.append(
        all(
            (
                step.feedback_success
                and np.isfinite(
                    step.trust_after_feedback
                )
            )
            for step in result.steps
            if step.environment_step_success
        )
    )

    # ------------------------------------------------------------------------
    # Sequential trust propagation
    #
    # If there are consecutive successful steps, the next step's controlled
    # trust must equal the previous REAL TrustEngine post-feedback trust.
    #
    # This validates the actual closed-loop feedback path rather than merely
    # validating history lengths.
    # ------------------------------------------------------------------------

    for previous, current in zip(
        result.steps,
        result.steps[1:],
    ):

        checks.append(
            np.isclose(
                current.controlled_trust,
                previous.trust_after_feedback,
                atol=1e-9,
                rtol=0.0,
            )
        )

    # ------------------------------------------------------------------------
    # No benchmark errors
    # ------------------------------------------------------------------------

    checks.append(
        len(result.errors) == 0
    )

    return all(
        checks
    )


# ============================================================================
# PRINT EPISODE
# ============================================================================

def print_episode_result(
    result: EpisodeResult,
) -> None:

    print()

    print(
        f"SCENARIO: {result.scenario}"
    )

    print(
        f"  Seed                    : "
        f"{result.seed}"
    )

    print(
        f"  Steps completed         : "
        f"{result.steps_completed}/"
        f"{result.steps_requested}"
    )

    print(
        f"  ALLOW                   : "
        f"{result.allow_count}"
    )

    print(
        f"  CONSTRAIN               : "
        f"{result.constrain_count}"
    )

    print(
        f"  BLOCK                   : "
        f"{result.block_count}"
    )

    print(
        f"  Candidate executions    : "
        f"{result.candidate_execution_count}"
    )

    print(
        f"  Fallback executions     : "
        f"{result.fallback_execution_count}"
    )

    print(
        f"  Action modifications    : "
        f"{result.action_modification_count}"
    )

    print(
        f"  Feedback calls          : "
        f"{result.feedback_call_count}"
    )

    print(
        f"  Feedback successes      : "
        f"{result.feedback_success_count}"
    )

    print(
        f"  Feedback failures       : "
        f"{result.feedback_failure_count}"
    )

    print(
        f"  Initial trust           : "
        f"{result.trust_initial:.6f}"
    )

    print(
        f"  Final trust             : "
        f"{result.trust_final:.6f}"
    )

    print(
        f"  Total reward            : "
        f"{result.total_reward:.6f}"
    )

    print(
        f"  Mean reward             : "
        f"{result.mean_reward:.6f}"
    )

    print(
        f"  Mean risk               : "
        f"{result.mean_risk:.6f}"
    )

    print(
        f"  Maximum risk            : "
        f"{result.max_risk:.6f}"
    )

    print(
        f"  Environment steps       : "
        f"{result.environment_steps}"
    )

    print(
        f"  Environment failures    : "
        f"{result.environment_failures}"
    )

    print(
        f"  Trust history length    : "
        f"{result.trust_history_length}"
    )

    print(
        f"  Action history length   : "
        f"{result.action_history_length}"
    )

    print(
        f"  Risk history length     : "
        f"{result.risk_history_length}"
    )

    print(
        f"  Policy history length   : "
        f"{result.policy_history_length}"
    )

    print(
        f"  Outcome history length  : "
        f"{result.outcome_history_length}"
    )

    print(
        f"  VALIDATION              : "
        f"{'PASS' if result.passed else 'FAIL'}"
    )

    if result.errors:

        print()

        print(
            "  Errors:"
        )

        for error in result.errors:

            print(
                f"    - {error}"
            )


# ============================================================================
# SCENARIO SUITE
# ============================================================================

def run_scenario_suite(
    steps: int = MAX_STEPS,
) -> List[EpisodeResult]:

    results: List[EpisodeResult] = []

    for scenario in SCENARIOS:

        seed = SCENARIO_SEEDS[
            scenario
        ]

        print(
            f"Running scenario: {scenario}"
        )

        result = run_episode(
            scenario=scenario,
            seed=seed,
            steps=steps,
            verbose=False,
        )

        results.append(
            result
        )

        print_episode_result(
            result
        )

    return results


# ============================================================================
# CLOSED-LOOP VALIDATION
# ============================================================================

def validate_closed_loop(
    results: List[EpisodeResult],
) -> bool:

    banner(
        "CLOSED-LOOP INTEGRATION VALIDATION"
    )

    print(
        f"Scenarios checked        : "
        f"{len(results)}"
    )

    passed_episodes = sum(
        bool(
            result.passed
        )
        for result in results
    )

    print(
        f"Scenario validations     : "
        f"{passed_episodes}/"
        f"{len(results)}"
    )

    total_allow = sum(
        result.allow_count
        for result in results
    )

    total_constrain = sum(
        result.constrain_count
        for result in results
    )

    total_block = sum(
        result.block_count
        for result in results
    )

    print()

    print(
        f"ALLOW observations       : "
        f"{total_allow}"
    )

    print(
        f"CONSTRAIN observations   : "
        f"{total_constrain}"
    )

    print(
        f"BLOCK observations       : "
        f"{total_block}"
    )

    # ------------------------------------------------------------------------
    # Governance regions
    # ------------------------------------------------------------------------

    allow_pass = (
        total_allow > 0
    )

    constrain_pass = (
        total_constrain > 0
    )

    block_pass = (
        total_block > 0
    )

    print(
        f"[{'PASS' if allow_pass else 'FAIL'}] "
        "ALLOW closed-loop region observed"
    )

    print(
        f"[{'PASS' if constrain_pass else 'FAIL'}] "
        "CONSTRAIN closed-loop region observed"
    )

    print(
        f"[{'PASS' if block_pass else 'FAIL'}] "
        "BLOCK closed-loop region observed"
    )

    # ------------------------------------------------------------------------
    # BLOCK isolation
    # ------------------------------------------------------------------------

    blocked_candidate_executions = sum(
        1
        for result in results
        for step in result.steps
        if (
            step.governance_status == "BLOCK"
            and step.candidate_executed
        )
    )

    execution_isolation_pass = (
        blocked_candidate_executions == 0
    )

    print(
        f"[{'PASS' if execution_isolation_pass else 'FAIL'}] "
        "BLOCK prevents candidate execution"
    )

    # ------------------------------------------------------------------------
    # Fallback accounting
    # ------------------------------------------------------------------------

    fallback_accounting_pass = all(
        result.fallback_execution_count
        == result.block_count
        for result in results
    )

    print(
        f"[{'PASS' if fallback_accounting_pass else 'FAIL'}] "
        "BLOCK fallback accounting is consistent"
    )

    # ------------------------------------------------------------------------
    # Fallback identity
    # ------------------------------------------------------------------------
    #
    # This validates that the actual environment action equals the benchmark's
    # designated fallback.
    #
    # It intentionally does NOT require fallback != candidate.
    # ------------------------------------------------------------------------

    fallback_action_pass = all(
        all(
            actions_equal(
                step.executed_action,
                step.fallback_action,
            )
            for step in result.steps
            if step.governance_status == "BLOCK"
        )
        for result in results
    )

    print(
        f"[{'PASS' if fallback_action_pass else 'FAIL'}] "
        "BLOCK executes designated safe fallback"
    )

    # ------------------------------------------------------------------------
    # CONSTRAIN
    # ------------------------------------------------------------------------

    constrain_steps = [
        step
        for result in results
        for step in result.steps
        if step.governance_status == "CONSTRAIN"
    ]

    constrain_enforcement_pass = all(
        (
            step.candidate_executed
            and not step.fallback_executed
            and step.action_modified
            and not actions_equal(
                step.candidate_action,
                step.executed_action,
            )
            and actions_equal(
                step.executed_action,
                step.decision_action,
            )
        )
        for step in constrain_steps
    )

    print(
        f"[{'PASS' if constrain_enforcement_pass else 'FAIL'}] "
        "CONSTRAIN executes SafeEnvelope-modified action"
    )

    # ------------------------------------------------------------------------
    # ALLOW
    # ------------------------------------------------------------------------

    allow_steps = [
        step
        for result in results
        for step in result.steps
        if step.governance_status == "ALLOW"
    ]

    allow_gate_pass = all(
        (
            step.candidate_executed
            and not step.fallback_executed
            and actions_equal(
                step.executed_action,
                step.decision_action,
            )
        )
        for step in allow_steps
    )

    print(
        f"[{'PASS' if allow_gate_pass else 'FAIL'}] "
        "ALLOW executes governance decision action"
    )

    # ------------------------------------------------------------------------
    # ALLOW candidate-to-decision observation
    # ------------------------------------------------------------------------
    #
    # This is informational/research evidence.
    #
    # It does not make the execution gate fail simply because the real
    # GovernanceEngine returns an equivalent governed representation that is
    # numerically different from the original candidate.
    # ------------------------------------------------------------------------

    allow_candidate_equivalent = all(
        actions_equal(
            step.candidate_action,
            step.decision_action,
        )
        for step in allow_steps
    )

    print(
        f"[{'PASS' if allow_candidate_equivalent else 'INFO'}] "
        "ALLOW candidate-to-decision numerical equivalence"
    )

    # ------------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------------

    environment_pass = all(
        result.environment_steps > 0
        and result.environment_failures == 0
        for result in results
    )

    print(
        f"[{'PASS' if environment_pass else 'FAIL'}] "
        "Real IRF environment executed"
    )

    # ------------------------------------------------------------------------
    # Trust feedback
    # ------------------------------------------------------------------------

    trust_feedback_pass = all(
        (
            result.feedback_call_count
            == result.environment_steps
            and result.feedback_success_count
            == result.environment_steps
            and result.feedback_failure_count
            == 0
        )
        for result in results
    )

    print(
        f"[{'PASS' if trust_feedback_pass else 'FAIL'}] "
        "Post-execution TrustEngine feedback integrated"
    )

    # ------------------------------------------------------------------------
    # Trust history continuity
    # ------------------------------------------------------------------------

    trust_history_pass = all(
        result.trust_history_length
        == result.feedback_success_count + 1
        for result in results
    )

    print(
        f"[{'PASS' if trust_history_pass else 'FAIL'}] "
        "Trust history records sequential feedback state"
    )

    # ------------------------------------------------------------------------
    # Full execution-history accounting
    # ------------------------------------------------------------------------

    history_accounting_pass = all(
        (
            result.action_history_length
            == result.feedback_success_count
            and result.risk_history_length
            == result.feedback_success_count
            and result.policy_history_length
            == result.feedback_success_count
            and result.outcome_history_length
            == result.feedback_success_count
        )
        for result in results
    )

    print(
        f"[{'PASS' if history_accounting_pass else 'FAIL'}] "
        "TrustEngine execution histories are complete"
    )

    # ------------------------------------------------------------------------
    # Sequential state
    # ------------------------------------------------------------------------

    sequential_state_pass = True

    for result in results:

        for previous, current in zip(
            result.steps,
            result.steps[1:],
        ):

            if (
                current.history_trust_length_after
                < previous.history_trust_length_after
            ):

                sequential_state_pass = False

            if (
                current.history_action_length_after
                < previous.history_action_length_after
            ):

                sequential_state_pass = False

            if (
                current.history_risk_length_after
                < previous.history_risk_length_after
            ):

                sequential_state_pass = False

            if (
                current.history_policy_length_after
                < previous.history_policy_length_after
            ):

                sequential_state_pass = False

            if (
                current.history_outcome_length_after
                < previous.history_outcome_length_after
            ):

                sequential_state_pass = False

            # ---------------------------------------------------------------
            # Actual trust propagation
            # ---------------------------------------------------------------

            if not np.isclose(
                current.controlled_trust,
                previous.trust_after_feedback,
                atol=1e-9,
                rtol=0.0,
            ):

                sequential_state_pass = False

    print(
        f"[{'PASS' if sequential_state_pass else 'FAIL'}] "
        "Sequential governance state preserved"
    )

    # ------------------------------------------------------------------------
    # Scenario validation
    # ------------------------------------------------------------------------

    scenario_validation_pass = (
        len(results) == len(SCENARIOS)
        and passed_episodes == len(results)
    )

    print(
        f"[{'PASS' if scenario_validation_pass else 'FAIL'}] "
        "All scenario-level validations passed"
    )

    # ------------------------------------------------------------------------
    # Overall
    # ------------------------------------------------------------------------

    overall = all(
        [
            scenario_validation_pass,
            allow_pass,
            constrain_pass,
            block_pass,
            execution_isolation_pass,
            fallback_accounting_pass,
            fallback_action_pass,
            constrain_enforcement_pass,
            allow_gate_pass,
            environment_pass,
            trust_feedback_pass,
            trust_history_pass,
            history_accounting_pass,
            sequential_state_pass,
        ]
    )

    print()

    print(
        f"RESULT                  : "
        f"{'PASS' if overall else 'FAIL'}"
    )

    return overall


# ============================================================================
# SUMMARY
# ============================================================================

def print_closed_loop_summary(
    results: List[EpisodeResult],
) -> None:

    banner(
        "CLOSED-LOOP GOVERNANCE SUMMARY"
    )

    total_steps = sum(
        result.steps_completed
        for result in results
    )

    total_allow = sum(
        result.allow_count
        for result in results
    )

    total_constrain = sum(
        result.constrain_count
        for result in results
    )

    total_block = sum(
        result.block_count
        for result in results
    )

    total_candidate = sum(
        result.candidate_execution_count
        for result in results
    )

    total_fallback = sum(
        result.fallback_execution_count
        for result in results
    )

    total_environment = sum(
        result.environment_steps
        for result in results
    )

    total_environment_failures = sum(
        result.environment_failures
        for result in results
    )

    total_feedback_calls = sum(
        result.feedback_call_count
        for result in results
    )

    total_feedback_success = sum(
        result.feedback_success_count
        for result in results
    )

    total_feedback_failure = sum(
        result.feedback_failure_count
        for result in results
    )

    total_reward = sum(
        result.total_reward
        for result in results
    )

    print(
        f"Total scenarios           : "
        f"{len(results)}"
    )

    print(
        f"Total closed-loop steps   : "
        f"{total_steps}"
    )

    print(
        f"ALLOW                     : "
        f"{total_allow}"
    )

    print(
        f"CONSTRAIN                 : "
        f"{total_constrain}"
    )

    print(
        f"BLOCK                     : "
        f"{total_block}"
    )

    print(
        f"Candidate executions      : "
        f"{total_candidate}"
    )

    print(
        f"Fallback executions       : "
        f"{total_fallback}"
    )

    print(
        f"Environment executions    : "
        f"{total_environment}"
    )

    print(
        f"Environment failures      : "
        f"{total_environment_failures}"
    )

    print(
        f"Trust feedback calls      : "
        f"{total_feedback_calls}"
    )

    print(
        f"Trust feedback successes  : "
        f"{total_feedback_success}"
    )

    print(
        f"Trust feedback failures   : "
        f"{total_feedback_failure}"
    )

    print(
        f"Total reward              : "
        f"{total_reward:.6f}"
    )

    if total_steps > 0:

        print(
            f"ALLOW rate                : "
            f"{100.0 * total_allow / total_steps:.2f}%"
        )

        print(
            f"CONSTRAIN rate            : "
            f"{100.0 * total_constrain / total_steps:.2f}%"
        )

        print(
            f"BLOCK rate                : "
            f"{100.0 * total_block / total_steps:.2f}%"
        )

        print(
            f"Candidate execution rate  : "
            f"{100.0 * total_candidate / total_steps:.2f}%"
        )

        print(
            f"Fallback execution rate   : "
            f"{100.0 * total_fallback / total_steps:.2f}%"
        )

        print(
            f"Feedback success rate     : "
            f"{100.0 * total_feedback_success / total_steps:.2f}%"
        )

    print()

    print(
        "SCENARIO RESULTS"
    )

    print(
        "-" * 78
    )

    print(
        f"{'Scenario':28s} "
        f"{'ALLOW':>8s} "
        f"{'CONSTR':>8s} "
        f"{'BLOCK':>8s} "
        f"{'FB':>6s} "
        f"{'PASS':>8s}"
    )

    print(
        "-" * 78
    )

    for result in results:

        print(
            f"{result.scenario:28s} "
            f"{result.allow_count:8d} "
            f"{result.constrain_count:8d} "
            f"{result.block_count:8d} "
            f"{result.feedback_success_count:6d} "
            f"{'YES' if result.passed else 'NO':>8s}"
        )


# ============================================================================
# ENVIRONMENT DIAGNOSTIC
# ============================================================================

def print_environment_diagnostic() -> None:

    banner(
        "REAL ENVIRONMENT DISCOVERY"
    )

    print(
        f"Project root          : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Experiments directory : "
        f"{EXPERIMENTS_DIR}"
    )

    if IRFEnvironment is not None:

        print(
            "IRFEnvironment        : FOUND"
        )

        print(
            "IRFConfig             : "
            f"{'FOUND' if IRFConfig is not None else 'MISSING'}"
        )

        print(
            "Module                : "
            f"{_ENV_MODULE_NAME}"
        )

        if _ENV_FILE_PATH:

            print(
                "Source file           : "
                f"{_ENV_FILE_PATH}"
            )

        print(
            "Environment class     : "
            f"{IRFEnvironment}"
        )

        return

    print(
        "IRFEnvironment        : NOT FOUND"
    )

    if _ENV_IMPORT_ERRORS:

        print()

        print(
            "Import/discovery diagnostics:"
        )

        for error in _ENV_IMPORT_ERRORS[-20:]:

            print(
                f"  - {error}"
            )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    start_time = time.perf_counter()

    banner(
        "TA-FDRL-IRF H5-C GOVERNANCE CLOSED-LOOP BENCHMARK"
    )

    print(
        "REAL COMPONENTS:"
    )

    print(
        "  TrustEngine        : REAL"
    )

    print(
        "  PolicyEngine       : REAL"
    )

    print(
        "  RiskEngine         : REAL"
    )

    print(
        "  SafeEnvelope       : REAL"
    )

    print(
        "  GovernanceEngine   : REAL"
    )

    print(
        "  IRFEnvironment     : REAL"
    )

    print()

    print(
        "No artificial RiskAssessment injection is used."
    )

    print(
        "No SAC training is performed."
    )

    print(
        "No post-hoc governance status is fabricated."
    )

    print()

    print(
        "CONTROL METHOD:"
    )

    print(
        "  Fresh REAL IRFEnvironment per scenario."
    )

    print(
        "  Fresh REAL GovernanceEngine per scenario."
    )

    print(
        "  Fresh REAL TrustEngine per scenario."
    )

    print(
        "  Real GovernanceEngine.evaluate()."
    )

    print(
        "  Real IRFEnvironment.step()."
    )

    print(
        "  Real TrustEngine.post_execution_feedback()."
    )

    print(
        "  BLOCK candidate actions never execute."
    )

    print(
        "  BLOCK uses zero/safe fallback."
    )

    print(
        "  CONSTRAIN executes SafeEnvelope-modified action."
    )

    print(
        "  ALLOW executes governance decision action."
    )

    print(
        "  Trust feedback uses ACTUAL executed action."
    )

    print(
        "  Trust state is preserved across timesteps."
    )

    print(
        "  Next-step trust comes from REAL TrustEngine state."
    )

    print()

    print(
        f"EXPECTED STATE DIMENSION : "
        f"{EXPECTED_STATE_DIM}"
    )

    print(
        f"EXPECTED ACTION DIMENSION: "
        f"{EXPECTED_ACTION_DIM}"
    )

    print(
        f"NUM USERS                : "
        f"{NUM_USERS}"
    )

    print(
        f"STEPS / SCENARIO         : "
        f"{MAX_STEPS}"
    )

    # ------------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------------

    print_environment_diagnostic()

    if not ENVIRONMENT_AVAILABLE:

        banner(
            "ENVIRONMENT IMPORT FAILURE"
        )

        print(
            "The REAL IRFEnvironment could not be "
            "imported/discovered."
        )

        print(
            "Benchmark refuses to use a fake environment."
        )

        print()

        print(
            "RESULT: H5-C GOVERNANCE CLOSED-LOOP "
            "BENCHMARK FAILED"
        )

        return 1

    # ------------------------------------------------------------------------
    # Scenarios
    # ------------------------------------------------------------------------

    results = run_scenario_suite(
        steps=MAX_STEPS
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print_closed_loop_summary(
        results
    )

    # ------------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------------

    overall_pass = validate_closed_loop(
        results
    )

    # ------------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # ------------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------------

    banner(
        "FINAL H5-C VALIDATION"
    )

    scenario_pass = (
        len(results) == len(SCENARIOS)
        and all(
            result.passed
            for result in results
        )
    )

    environment_pass = all(
        result.environment_failures == 0
        and result.environment_steps > 0
        for result in results
    )

    allow_pass = any(
        result.allow_count > 0
        for result in results
    )

    constrain_pass = any(
        result.constrain_count > 0
        for result in results
    )

    block_pass = any(
        result.block_count > 0
        for result in results
    )

    execution_isolation_pass = all(
        not any(
            (
                step.governance_status == "BLOCK"
                and step.candidate_executed
            )
            for step in result.steps
        )
        for result in results
    )

    fallback_accounting_pass = all(
        result.fallback_execution_count
        == result.block_count
        for result in results
    )

    fallback_action_pass = all(
        all(
            actions_equal(
                step.executed_action,
                step.fallback_action,
            )
            for step in result.steps
            if step.governance_status == "BLOCK"
        )
        for result in results
    )

    constrain_pass_strict = all(
        (
            step.candidate_executed
            and not step.fallback_executed
            and step.action_modified
            and not actions_equal(
                step.candidate_action,
                step.executed_action,
            )
            and actions_equal(
                step.executed_action,
                step.decision_action,
            )
        )
        for result in results
        for step in result.steps
        if step.governance_status == "CONSTRAIN"
    )

    allow_execution_pass = all(
        (
            step.candidate_executed
            and not step.fallback_executed
            and actions_equal(
                step.executed_action,
                step.decision_action,
            )
        )
        for result in results
        for step in result.steps
        if step.governance_status == "ALLOW"
    )

    trust_feedback_pass = all(
        (
            result.feedback_call_count
            == result.environment_steps
            and result.feedback_success_count
            == result.environment_steps
            and result.feedback_failure_count == 0
        )
        for result in results
    )

    trust_history_pass = all(
        result.trust_history_length
        == result.feedback_success_count + 1
        for result in results
    )

    history_accounting_pass = all(
        (
            result.action_history_length
            == result.feedback_success_count
            and result.risk_history_length
            == result.feedback_success_count
            and result.policy_history_length
            == result.feedback_success_count
            and result.outcome_history_length
            == result.feedback_success_count
        )
        for result in results
    )

    sequential_state_pass = all(
        all(
            (
                current.history_trust_length_after
                >= previous.history_trust_length_after
                and current.history_action_length_after
                >= previous.history_action_length_after
                and current.history_risk_length_after
                >= previous.history_risk_length_after
                and current.history_policy_length_after
                >= previous.history_policy_length_after
                and current.history_outcome_length_after
                >= previous.history_outcome_length_after
                and np.isclose(
                    current.controlled_trust,
                    previous.trust_after_feedback,
                    atol=1e-9,
                    rtol=0.0,
                )
            )
            for previous, current in zip(
                result.steps,
                result.steps[1:],
            )
        )
        for result in results
    )

    print(
        f"Real GovernanceEngine       : "
        f"{'PASS' if scenario_pass else 'FAIL'}"
    )

    print(
        f"Real IRF execution          : "
        f"{'PASS' if environment_pass else 'FAIL'}"
    )

    print(
        f"ALLOW gate                  : "
        f"{'PASS' if allow_execution_pass and allow_pass else 'FAIL'}"
    )

    print(
        f"CONSTRAIN gate              : "
        f"{'PASS' if constrain_pass_strict and constrain_pass else 'FAIL'}"
    )

    print(
        f"BLOCK gate                  : "
        f"{'PASS' if block_pass else 'FAIL'}"
    )

    print(
        f"Execution isolation         : "
        f"{'PASS' if execution_isolation_pass else 'FAIL'}"
    )

    print(
        f"Fallback execution          : "
        f"{'PASS' if fallback_action_pass and fallback_accounting_pass else 'FAIL'}"
    )

    print(
        f"Trust feedback integration  : "
        f"{'PASS' if trust_feedback_pass else 'FAIL'}"
    )

    print(
        f"Trust history continuity    : "
        f"{'PASS' if trust_history_pass else 'FAIL'}"
    )

    print(
        f"Trust execution histories   : "
        f"{'PASS' if history_accounting_pass else 'FAIL'}"
    )

    print(
        f"Sequential closed-loop state: "
        f"{'PASS' if sequential_state_pass else 'FAIL'}"
    )

    print(
        f"Runtime elapsed             : "
        f"{elapsed:.3f} seconds"
    )

    print()

    # ------------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------------

    if (
        overall_pass
        and scenario_pass
        and environment_pass
        and allow_pass
        and constrain_pass
        and block_pass
        and execution_isolation_pass
        and fallback_accounting_pass
        and fallback_action_pass
        and allow_execution_pass
        and constrain_pass_strict
        and trust_feedback_pass
        and trust_history_pass
        and history_accounting_pass
        and sequential_state_pass
    ):

        print(
            "RESULT: H5-C GOVERNANCE CLOSED-LOOP "
            "BENCHMARK PASSED"
        )

        print(
            "RESULT: REAL GOVERNANCE-TO-IRF "
            "EXECUTION PATH VALIDATED"
        )

        print(
            "RESULT: ALLOW / CONSTRAIN / BLOCK "
            "EXECUTION GATE VALIDATED"
        )

        print(
            "RESULT: BLOCKED CANDIDATE ACTION "
            "ISOLATION VALIDATED"
        )

        print(
            "RESULT: SAFE-ENVELOPE CONSTRAINED "
            "EXECUTION VALIDATED"
        )

        print(
            "RESULT: REAL POST-EXECUTION "
            "TRUST FEEDBACK VALIDATED"
        )

        print(
            "RESULT: TRUST HISTORY CONTINUITY "
            "VALIDATED"
        )

        print(
            "RESULT: TRUST EXECUTION HISTORIES "
            "VALIDATED"
        )

        print(
            "RESULT: SEQUENTIAL CLOSED-LOOP "
            "STATE VALIDATED"
        )

        return 0

    print(
        "RESULT: H5-C GOVERNANCE CLOSED-LOOP "
        "BENCHMARK FAILED"
    )

    return 1


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )

