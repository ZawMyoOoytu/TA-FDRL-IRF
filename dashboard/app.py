from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

try:
    from environment.irf_env import IRFConfig, IRFEnvironment
    from agents.sac_agent import SACAgent
    from trust.governance import GovernanceEngine

    # H5-C validation oracle
    from experiments.governance_closed_loop_benchmark import (
        SCENARIOS,
        SCENARIO_SEEDS,
        run_episode,
    )

except Exception as exc:
    st.error("TA-FDRL-IRF project imports failed.")
    st.exception(exc)
    st.stop()


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="TA-FDRL-IRF Research Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# RESEARCH CONFIGURATION
# ============================================================

PROJECT_NAME = "TA-FDRL-IRF"

PHASE = "Phase-4B.1"

NUM_USERS = 20
NUM_RIS_ELEMENTS = 64

STATE_DIM = 100
ACTION_DIM = 40

MAX_STEPS = 200

BANDWIDTH_HZ = 100e6
CARRIER_FREQUENCY_HZ = 28e9

POLARIZATION_ENABLED = True
POLARIZATION_STATE_ENABLED = False
CROSS_POLARIZATION_FACTOR = 0.15

RIS_OPTIMIZATION = False
FIXED_TRUST = False
ADAPTIVE_TRUST = True

H5C_DEFAULT_STEPS = 20


# ============================================================
# CHECKPOINT
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "best_sac_irf_phase3_adaptive_trust"
    / "sac_phase3_adaptive_trust.pt"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1rem;
        opacity: 0.75;
        margin-bottom: 1rem;
    }

    .live-box {
        padding: 0.8rem 1rem;
        border-radius: 0.6rem;
        border: 1px solid rgba(128,128,128,0.35);
        margin-bottom: 1rem;
    }

    .status-live {
        font-size: 1.1rem;
        font-weight: 700;
    }

    .small-muted {
        font-size: 0.82rem;
        opacity: 0.65;
    }

    .pass-box {
        padding: 0.8rem 1rem;
        border-radius: 0.6rem;
        border: 1px solid rgba(0,180,0,0.35);
        margin-bottom: 1rem;
    }

    .validation-box {
        padding: 1rem;
        border-radius: 0.7rem;
        border: 1px solid rgba(128,128,128,0.35);
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convert scalar-like input to a finite float."""

    try:
        if value is None:
            return float(default)

        arr = np.asarray(value)

        if arr.size == 0:
            return float(default)

        result = float(arr.reshape(-1)[0])

        if not np.isfinite(result):
            return float(default)

        return result

    except Exception:
        return float(default)


def safe_mean(
    value: Any,
    default: float = 0.0,
) -> float:
    """Finite mean for scalar or array-like input."""

    try:
        arr = np.asarray(
            value,
            dtype=float,
        )

        if arr.size == 0:
            return float(default)

        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            return float(default)

        return float(np.mean(arr))

    except Exception:
        return float(default)


def safe_bool(
    value: Any,
    default: bool = False,
) -> bool:
    """Safely convert a value to bool."""

    try:
        return bool(value)
    except Exception:
        return bool(default)


def get_field(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:
    """
    Generic field reader supporting:

        dataclass/object attributes

    and:

        dictionary keys
    """

    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


# ============================================================
# DATAFRAME SAFETY
# ============================================================

def arrow_safe_dataframe(
    dataframe: pd.DataFrame,
    stringify_object_columns: bool = False,
) -> pd.DataFrame:
    """
    Make DataFrame safe for Streamlit/PyArrow display.
    """

    df = dataframe.copy()

    for column in df.columns:

        series = df[column]

        if series.dtype == "object":

            if stringify_object_columns:

                df[column] = series.map(
                    lambda value:
                    ""
                    if value is None
                    else str(value)
                )

            else:

                numeric = pd.to_numeric(
                    series,
                    errors="coerce",
                )

                if numeric.notna().all():

                    df[column] = numeric

                else:

                    df[column] = series.map(
                        lambda value:
                        ""
                        if value is None
                        else str(value)
                    )

        elif pd.api.types.is_numeric_dtype(series):

            df[column] = pd.to_numeric(
                series,
                errors="coerce",
            )

    return df


def safe_display_dataframe(
    dataframe: pd.DataFrame,
    *,
    hide_index: bool = False,
    stringify_object_columns: bool = False,
) -> None:

    safe_df = arrow_safe_dataframe(
        dataframe,
        stringify_object_columns=stringify_object_columns,
    )

    st.dataframe(
        safe_df,
        width="stretch",
        hide_index=hide_index,
    )


# ============================================================
# ENVIRONMENT RESET COMPATIBILITY
# ============================================================

def extract_state(
    reset_result: Any,
) -> np.ndarray:
    """
    Support both:

        reset() -> state

    and:

        reset() -> (state, info)
    """

    if isinstance(
        reset_result,
        tuple,
    ):

        if len(reset_result) == 0:

            raise ValueError(
                "Environment reset returned "
                "an empty tuple."
            )

        state = reset_result[0]

    else:

        state = reset_result

    state = np.asarray(
        state,
        dtype=np.float32,
    ).reshape(-1)

    validate_state(
        state
    )

    return state


# ============================================================
# STATE / ACTION VALIDATION
# ============================================================

def validate_state(
    state: np.ndarray,
) -> None:

    if state.size != STATE_DIM:

        raise ValueError(
            "State dimension mismatch: "
            f"expected {STATE_DIM}, "
            f"got {state.size}"
        )

    if not np.all(
        np.isfinite(state)
    ):

        raise ValueError(
            "Environment returned "
            "non-finite state values."
        )


def normalize_action(
    action: Any,
) -> np.ndarray:

    action = np.asarray(
        action,
        dtype=np.float32,
    ).reshape(-1)

    if action.size != ACTION_DIM:

        raise ValueError(
            "Action dimension mismatch: "
            f"expected {ACTION_DIM}, "
            f"got {action.size}"
        )

    if not np.all(
        np.isfinite(action)
    ):

        raise ValueError(
            "Action contains non-finite values."
        )

    return np.clip(
        action,
        -1.0,
        1.0,
    )


def zero_action() -> np.ndarray:

    return np.zeros(
        ACTION_DIM,
        dtype=np.float32,
    )


# ============================================================
# ENVIRONMENT FACTORY
#
# IMPORTANT:
# Do NOT cache the mutable environment.
# ============================================================

def create_environment(
    seed: int = 42,
):

    config = IRFConfig(

        # ----------------------------------------------------
        # Network / PHY
        # ----------------------------------------------------

        num_users=NUM_USERS,
        num_ris_elements=NUM_RIS_ELEMENTS,

        bandwidth_hz=BANDWIDTH_HZ,
        carrier_frequency_hz=CARRIER_FREQUENCY_HZ,

        # ----------------------------------------------------
        # Polarization-aware PHY
        # ----------------------------------------------------

        polarization_enabled=POLARIZATION_ENABLED,

        cross_polarization_factor=(
            CROSS_POLARIZATION_FACTOR
        ),

        polarization_v_strength=1.0,
        polarization_h_strength=1.0,

        # ----------------------------------------------------
        # Power / noise
        # ----------------------------------------------------

        max_power_w=1.0,
        circuit_power_w=0.1,

        noise_figure_db=7.0,
        noise_density_dbm_hz=-174.0,

        # ----------------------------------------------------
        # Runtime
        # ----------------------------------------------------

        max_steps=MAX_STEPS,

        optimize_ris=RIS_OPTIMIZATION,
        fixed_trust=FIXED_TRUST,

        # ----------------------------------------------------
        # Adaptive Trust
        # ----------------------------------------------------

        trust_memory=0.90,
        trust_learning_rate=0.10,
        trust_target_rate_bps=1e7,

        trust_service_weight=0.35,
        trust_interference_weight=0.20,
        trust_queue_weight=0.25,
        trust_instability_weight=0.20,

        trust_neutral_point=0.50,
        trust_floor=0.0,

        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------

        se_weight=0.45,
        ee_weight=0.20,
        trust_weight=0.15,
        trust_delta_weight=0.10,

        interference_penalty=0.05,
        power_penalty=0.03,
        queue_penalty=0.07,

        se_target=0.20,
        ee_target=1e6,

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        seed=int(seed),
    )

    env = IRFEnvironment(
        config
    )

    return env


def reset_environment(
    seed: int,
):
    """Create a completely fresh environment."""

    env = create_environment(
        seed
    )

    state = extract_state(
        env.reset(
            seed=int(seed)
        )
    )

    return env, state


# ============================================================
# SAC AGENT
#
# Agent may safely be cached because it is not the runtime
# environment state.
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def create_agent():

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "SAC checkpoint not found:\n"
            f"{CHECKPOINT_PATH}\n\n"
            "Make sure the checkpoint exists "
            "in the deployment."
        )

    agent = SACAgent(

        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,

        hidden_dim=256,

        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,

        gamma=0.99,
        tau=0.005,

        buffer_size=100000,
        batch_size=256,
    )

    agent.load(
        str(CHECKPOINT_PATH)
    )

    return agent


# ============================================================
# GOVERNANCE FACTORY
#
# IMPORTANT:
# GovernanceEngine owns mutable TrustEngine state.
# Therefore it must NOT be cached.
# ============================================================

def create_governance():

    return GovernanceEngine()


# ============================================================
# GOVERNANCE HELPERS
# ============================================================

def governance_status(
    decision: Any,
) -> str:

    status = get_field(
        decision,
        "status",
        None,
    )

    if status is None:

        status = get_field(
            decision,
            "decision",
            None,
        )

    if status is None:

        status = get_field(
            decision,
            "governance",
            "UNKNOWN",
        )

    return str(
        status
    ).upper()


def governance_action(
    decision: Any,
    proposed_action: np.ndarray,
) -> np.ndarray:

    action = get_field(
        decision,
        "action",
        None,
    )

    if action is None:

        action = proposed_action

    return normalize_action(
        action
    )


def governance_policy_allowed(
    decision: Any,
    fallback: bool = False,
) -> bool:
    """
    decision.policy is an assessment object.

    Read:

        decision.policy.allowed
    """

    policy = get_field(
        decision,
        "policy",
        None,
    )

    return safe_bool(
        get_field(
            policy,
            "allowed",
            fallback,
        ),
        fallback,
    )


def governance_risk_score(
    decision: Any,
    fallback: float = 0.0,
) -> float:
    """
    decision.risk is an assessment object.

    Read:

        decision.risk.risk_score
    """

    risk = get_field(
        decision,
        "risk",
        None,
    )

    return safe_float(
        get_field(
            risk,
            "risk_score",
            fallback,
        ),
        fallback,
    )


def governance_trust_score(
    decision: Any,
    fallback: float = 0.0,
) -> float:

    trust = get_field(
        decision,
        "trust",
        None,
    )

    return safe_float(
        get_field(
            trust,
            "trust_score",
            fallback,
        ),
        fallback,
    )


# ============================================================
# TELEMETRY FROM STATE
# ============================================================

def telemetry_from_state(
    state: np.ndarray,
) -> dict[str, float]:

    validate_state(
        state
    )

    users = state.reshape(
        NUM_USERS,
        5,
    )

    # [SINR, interference, queue, power, trust]

    sinr = users[:, 0]
    interference = users[:, 1]
    queue = users[:, 2]
    power = users[:, 3]
    trust = users[:, 4]

    return {

        "mean_sinr":
            safe_mean(sinr),

        "mean_interference":
            safe_mean(interference),

        "mean_queue":
            safe_mean(queue),

        "mean_power":
            safe_mean(power),

        "mean_trust":
            safe_mean(trust),

        "queue":
            float(
                np.clip(
                    safe_mean(queue),
                    0.0,
                    1.0,
                )
            ),

        "interference":
            float(
                np.clip(
                    safe_mean(interference),
                    0.0,
                    1.0,
                )
            ),
    }


# ============================================================
# ENVIRONMENT TELEMETRY
# ============================================================

def get_environment_telemetry(
    env: Any,
    info: Any = None,
) -> dict[str, float]:

    if not isinstance(
        info,
        dict,
    ):

        info = {}

    telemetry = {

        "mean_sinr":
            safe_mean(
                getattr(
                    env,
                    "sinr",
                    0.0,
                )
            ),

        "mean_interference":
            safe_mean(
                getattr(
                    env,
                    "interference",
                    0.0,
                )
            ),

        "mean_queue":
            safe_mean(
                getattr(
                    env,
                    "queue",
                    0.0,
                )
            ),

        "mean_power":
            safe_mean(
                getattr(
                    env,
                    "power",
                    0.0,
                )
            ),

        "mean_trust":
            safe_mean(
                getattr(
                    env,
                    "trust",
                    0.0,
                )
            ),
    }

    for key in (
        "spectral_efficiency",
        "energy_efficiency",
        "trust",
        "risk",
        "reward",
        "mean_sinr",
        "mean_interference",
        "mean_queue",
        "mean_power",
        "mean_trust",
        "pqi",
        "xpi",
        "mean_pqi",
        "mean_xpi",
    ):

        if key in info:

            telemetry[key] = safe_float(
                info[key],
                telemetry.get(
                    key,
                    0.0,
                ),
            )

    telemetry["queue"] = float(
        np.clip(
            telemetry["mean_queue"],
            0.0,
            1.0,
        )
    )

    telemetry["interference"] = float(
        np.clip(
            telemetry["mean_interference"],
            0.0,
            1.0,
        )
    )

    return telemetry


# ============================================================
# TRUST STATE
# ============================================================

def get_governance_trust_state(
    governance: Any,
) -> dict[str, Any]:

    try:

        state = governance.get_trust_state()

        if isinstance(
            state,
            dict,
        ):

            return state

        return {
            "trust_score": safe_float(
                get_field(
                    state,
                    "trust_score",
                    0.0,
                )
            )
        }

    except Exception:

        return {}


def history_lengths(
    trust_state: dict[str, Any],
) -> dict[str, int]:
    """Extract TrustEngine execution-history lengths."""

    def length_of(
        name: str,
    ) -> int:

        value = trust_state.get(
            name,
            [],
        )

        if isinstance(
            value,
            (list, tuple),
        ):

            return len(value)

        return 0

    return {

        "trust_history":
            length_of("trust_history"),

        "action_history":
            length_of("action_history"),

        "risk_history":
            length_of("risk_history"),

        "policy_history":
            length_of("policy_history"),

        "outcome_history":
            length_of("outcome_history"),
    }


def validate_trust_history_invariant(
    trust_state: dict[str, Any],
) -> tuple[bool, str]:
    """
    TrustEngine invariant after N executions:

        trust_history == N + 1
        action_history == N
        risk_history == N
        policy_history == N
        outcome_history == N
    """

    lengths = history_lengths(
        trust_state
    )

    n = lengths[
        "action_history"
    ]

    checks = {

        "trust_history":
            lengths["trust_history"]
            == n + 1,

        "action_history":
            lengths["action_history"]
            == n,

        "risk_history":
            lengths["risk_history"]
            == n,

        "policy_history":
            lengths["policy_history"]
            == n,

        "outcome_history":
            lengths["outcome_history"]
            == n,
    }

    passed = all(
        checks.values()
    )

    if passed:

        return (
            True,
            (
                "PASS: Trust history invariant "
                f"N+1/N = "
                f"{lengths['trust_history']}/"
                f"{n}"
            ),
        )

    return (
        False,
        (
            "FAIL: Trust history invariant "
            f"trust={lengths['trust_history']} "
            f"actions={lengths['action_history']} "
            f"risk={lengths['risk_history']} "
            f"policy={lengths['policy_history']} "
            f"outcome={lengths['outcome_history']}"
        ),
    )


# ============================================================
# RUN ONE REAL CLOSED-LOOP STEP
# ============================================================

def run_step(
    env: Any,
    agent: Any,
    governance: Any,
    state: np.ndarray,
):
    """
    Real Phase-4B.1 closed loop.

    SAC
      ↓
    Governance.evaluate()
      ↓
    ALLOW / CONSTRAIN / BLOCK
      ↓
    Executed / Fallback Action
      ↓
    REAL IRFEnvironment.step()
      ↓
    REAL telemetry
      ↓
    Governance.update_trust()
      ↓
    next state
    """

    validate_state(
        state
    )

    # ========================================================
    # 1. SAC
    # ========================================================

    proposed_action = agent.select_action(
        state,
        evaluate=True,
    )

    proposed_action = normalize_action(
        proposed_action
    )

    # ========================================================
    # 2. PRE-EXECUTION TELEMETRY
    # ========================================================

    state_telemetry = telemetry_from_state(
        state
    )

    current_environment_trust = safe_float(
        getattr(
            env,
            "trust",
            state_telemetry["mean_trust"],
        ),
        state_telemetry["mean_trust"],
    )

    # ========================================================
    # 3. GOVERNANCE
    # ========================================================

    decision = governance.evaluate(
        proposed_action,
        trust_score=current_environment_trust,
        telemetry=state_telemetry,
    )

    status = governance_status(
        decision
    )

    pre_risk_score = governance_risk_score(
        decision,
        fallback=state_telemetry.get(
            "risk",
            0.0,
        ),
    )

    pre_governance_trust = governance_trust_score(
        decision,
        fallback=current_environment_trust,
    )

    policy_allowed = governance_policy_allowed(
        decision,
        fallback=(
            status == "ALLOW"
            or status == "CONSTRAIN"
        ),
    )

    # ========================================================
    # 4. GOVERNANCE ENFORCEMENT
    # ========================================================

    if status == "ALLOW":

        executed_action = governance_action(
            decision,
            proposed_action,
        )

        candidate_executed = True
        fallback_executed = False
        execution_allowed = True

        # ALLOW is not an intervention.
        modified = False

    elif status == "CONSTRAIN":

        executed_action = governance_action(
            decision,
            proposed_action,
        )

        candidate_executed = True
        fallback_executed = False
        execution_allowed = True

        modified = safe_bool(
            get_field(
                decision,
                "modified",
                True,
            ),
            True,
        )

    else:

        # Candidate autonomous action is NEVER executed.
        executed_action = zero_action()

        candidate_executed = False
        fallback_executed = True
        execution_allowed = False

        # BLOCK fallback is not "modified".
        modified = False

    # ========================================================
    # 5. ACTION DIFFERENCE
    # ========================================================

    action_delta = np.abs(
        proposed_action
        - executed_action
    )

    max_action_delta = safe_float(
        np.max(action_delta)
    )

    mean_action_delta = safe_float(
        np.mean(action_delta)
    )

    # ========================================================
    # 6. REAL IRF EXECUTION
    # ========================================================

    result = env.step(
        executed_action
    )

    if not isinstance(
        result,
        tuple,
    ):

        raise ValueError(
            "IRFEnvironment.step() must return a tuple."
        )

    if len(result) != 5:

        raise ValueError(
            "IRFEnvironment.step() must return "
            "(next_state, reward, terminated, truncated, info). "
            f"Received {len(result)} values."
        )

    (
        next_state,
        reward,
        terminated,
        truncated,
        info,
    ) = result

    next_state = np.asarray(
        next_state,
        dtype=np.float32,
    ).reshape(-1)

    validate_state(
        next_state
    )

    # ========================================================
    # 7. POST-EXECUTION TELEMETRY
    # ========================================================

    actual_telemetry = get_environment_telemetry(
        env,
        info,
    )

    # ========================================================
    # 8. PERFORMANCE METRICS
    # ========================================================

    reward_value = safe_float(
        reward
    )

    spectral_efficiency = safe_float(
        info.get(
            "spectral_efficiency",
            actual_telemetry.get(
                "spectral_efficiency",
                0.0,
            ),
        )
        if isinstance(info, dict)
        else 0.0
    )

    energy_efficiency = safe_float(
        info.get(
            "energy_efficiency",
            actual_telemetry.get(
                "energy_efficiency",
                0.0,
            ),
        )
        if isinstance(info, dict)
        else 0.0
    )

    environment_trust_after_execution = safe_float(
        info.get(
            "trust",
            getattr(
                env,
                "trust",
                actual_telemetry[
                    "mean_trust"
                ],
            ),
        )
        if isinstance(info, dict)
        else getattr(
            env,
            "trust",
            actual_telemetry[
                "mean_trust"
            ],
        )
    )

    # ========================================================
    # 9. REAL POST-EXECUTION TRUST FEEDBACK
    # ========================================================

    trust_feedback_called = False
    trust_feedback_success = False

    post_governance_trust = (
        pre_governance_trust
    )

    trust_feedback_error = ""

    try:

        trust_feedback_called = True

        feedback_result = governance.update_trust(
            reward=reward_value,
            environment_trust=(
                environment_trust_after_execution
            ),
            action=executed_action,
            decision=decision,
            telemetry=actual_telemetry,
        )

        trust_feedback_success = True

        post_governance_trust = safe_float(
            get_field(
                feedback_result,
                "trust_score",
                post_governance_trust,
            ),
            post_governance_trust,
        )

        if isinstance(
            feedback_result,
            dict,
        ):

            post_governance_trust = safe_float(
                feedback_result.get(
                    "trust_score",
                    feedback_result.get(
                        "governance_trust",
                        post_governance_trust,
                    ),
                ),
                post_governance_trust,
            )

        # Read authoritative TrustEngine state.
        trust_state = get_governance_trust_state(
            governance
        )

        if trust_state:

            post_governance_trust = safe_float(
                trust_state.get(
                    "trust_score",
                    post_governance_trust,
                ),
                post_governance_trust,
            )

    except Exception as exc:

        trust_feedback_error = str(
            exc
        )

    # ========================================================
    # 10. TRUST STATE / HISTORY
    # ========================================================

    trust_state = get_governance_trust_state(
        governance
    )

    lengths = history_lengths(
        trust_state
    )

    trust_history_length = lengths[
        "trust_history"
    ]

    action_history_length = lengths[
        "action_history"
    ]

    risk_history_length = lengths[
        "risk_history"
    ]

    policy_history_length = lengths[
        "policy_history"
    ]

    outcome_history_length = lengths[
        "outcome_history"
    ]

    history_invariant_passed, history_invariant_message = (
        validate_trust_history_invariant(
            trust_state
        )
    )

    # ========================================================
    # 11. POLARIZATION DIAGNOSTICS
    # ========================================================

    pqi = safe_float(
        info.get(
            "pqi",
            info.get(
                "mean_pqi",
                0.0,
            ),
        )
        if isinstance(info, dict)
        else 0.0
    )

    xpi = safe_float(
        info.get(
            "xpi",
            info.get(
                "mean_xpi",
                0.0,
            ),
        )
        if isinstance(info, dict)
        else 0.0
    )

    # ========================================================
    # 12. RETURN RECORD
    # ========================================================

    record = {

        "step": 0,

        # ----------------------------------------------------
        # Performance
        # ----------------------------------------------------

        "reward":
            reward_value,

        "spectral_efficiency":
            spectral_efficiency,

        "energy_efficiency":
            energy_efficiency,

        # ----------------------------------------------------
        # Trust
        # ----------------------------------------------------

        "environment_trust":
            environment_trust_after_execution,

        "pre_governance_trust":
            pre_governance_trust,

        "trust":
            post_governance_trust,

        "trust_feedback_called":
            trust_feedback_called,

        "trust_feedback_success":
            trust_feedback_success,

        "trust_feedback_error":
            trust_feedback_error,

        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        "risk":
            pre_risk_score,

        # ----------------------------------------------------
        # Governance
        # ----------------------------------------------------

        "governance":
            status,

        "reason":
            str(
                get_field(
                    decision,
                    "reason",
                    "",
                )
                or ""
            ),

        "policy_allowed":
            policy_allowed,

        "execution_allowed":
            execution_allowed,

        "modified":
            modified,

        # ----------------------------------------------------
        # Candidate / fallback
        # ----------------------------------------------------

        "candidate_executed":
            candidate_executed,

        "fallback_executed":
            fallback_executed,

        "block_fallback":
            fallback_executed,

        # ----------------------------------------------------
        # Action enforcement
        # ----------------------------------------------------

        "max_action_delta":
            max_action_delta,

        "mean_action_delta":
            mean_action_delta,

        # ----------------------------------------------------
        # Network
        # ----------------------------------------------------

        "mean_sinr":
            actual_telemetry[
                "mean_sinr"
            ],

        "mean_interference":
            actual_telemetry[
                "mean_interference"
            ],

        "mean_queue":
            actual_telemetry[
                "mean_queue"
            ],

        "mean_power":
            actual_telemetry[
                "mean_power"
            ],

        # ----------------------------------------------------
        # Polarization
        # ----------------------------------------------------

        "pqi":
            pqi,

        "xpi":
            xpi,

        # ----------------------------------------------------
        # Trust history
        # ----------------------------------------------------

        "trust_history_length":
            trust_history_length,

        "action_history_length":
            action_history_length,

        "risk_history_length":
            risk_history_length,

        "policy_history_length":
            policy_history_length,

        "outcome_history_length":
            outcome_history_length,

        "history_invariant_passed":
            history_invariant_passed,

        "history_invariant_message":
            history_invariant_message,

        # ----------------------------------------------------
        # Termination
        # ----------------------------------------------------

        "terminated":
            safe_bool(
                terminated
            ),

        "truncated":
            safe_bool(
                truncated
            ),
    }

    return (
        next_state,
        record,
    )


# ============================================================
# H5-C RESULT HELPERS
# ============================================================

def h5_field(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:
    """
    Generic H5-C result field reader.

    Supports dictionaries and dataclasses/objects.
    """

    if obj is None:
        return default

    if isinstance(
        obj,
        dict,
    ):

        return obj.get(
            name,
            default,
        )

    return getattr(
        obj,
        name,
        default,
    )


def scenario_name(
    scenario: Any,
) -> str:

    if isinstance(
        scenario,
        str,
    ):

        return scenario

    return str(
        h5_field(
            scenario,
            "name",
            scenario,
        )
    )


# ============================================================
# H5-C SCENARIO RUNNER
#
# IMPORTANT:
#
# Validated benchmark API:
#
#     run_episode(
#         scenario,
#         seed=...,
#         steps=...,
#         verbose=False,
#     )
#
# NOT:
#
#     max_steps=...
# ============================================================

def run_h5c_scenario(
    scenario: Any,
    seed: int,
    steps: int = H5C_DEFAULT_STEPS,
):
    """
    H5-C benchmark is the validation oracle.

    The dashboard does not duplicate H5-C scenario logic.
    It directly reuses the validated benchmark implementation.
    """

    return run_episode(
        scenario,
        seed=int(seed),
        steps=int(steps),
        verbose=False,
    )


def normalize_h5c_steps(
    result: Any,
) -> list[Any]:

    steps = h5_field(
        result,
        "steps",
        [],
    )

    if steps is None:
        return []

    if isinstance(
        steps,
        list,
    ):

        return steps

    try:

        return list(
            steps
        )

    except Exception:

        return []


# ============================================================
# H5-C STEP FIELD HELPERS
#
# Current benchmark StepResult:
#
# governance_status
# governance_reason
# risk_score
# trust_after_feedback
# feedback_success
# environment_step_success
#
# Legacy aliases are retained for compatibility.
# ============================================================

def h5c_step_status(
    item: Any,
) -> str:

    value = h5_field(
        item,
        "governance_status",
        h5_field(
            item,
            "status",
            h5_field(
                item,
                "governance",
                "",
            ),
        ),
    )

    return str(
        value or ""
    ).upper()


def h5c_step_reason(
    item: Any,
) -> str:

    value = h5_field(
        item,
        "governance_reason",
        h5_field(
            item,
            "reason",
            "",
        ),
    )

    return str(
        value or ""
    )


def h5c_step_risk(
    item: Any,
) -> float:

    return safe_float(
        h5_field(
            item,
            "risk_score",
            h5_field(
                item,
                "risk",
                0.0,
            ),
        )
    )


def h5c_step_trust(
    item: Any,
) -> float:

    return safe_float(
        h5_field(
            item,
            "trust_after_feedback",
            h5_field(
                item,
                "trust",
                h5_field(
                    item,
                    "post_trust",
                    0.0,
                ),
            ),
        )
    )


def h5c_step_feedback_success(
    item: Any,
) -> bool:

    return safe_bool(
        h5_field(
            item,
            "feedback_success",
            h5_field(
                item,
                "trust_feedback_success",
                False,
            ),
        )
    )


def h5c_step_environment_success(
    item: Any,
) -> bool:

    return safe_bool(
        h5_field(
            item,
            "environment_step_success",
            True,
        )
    )


def h5c_step_candidate_executed(
    item: Any,
) -> bool:

    return safe_bool(
        h5_field(
            item,
            "candidate_executed",
            False,
        )
    )


def h5c_step_fallback_executed(
    item: Any,
) -> bool:

    return safe_bool(
        h5_field(
            item,
            "fallback_executed",
            False,
        )
    )


# ============================================================
# H5-C SUMMARY
# ============================================================

def h5c_summary_from_result(
    result: Any,
) -> dict[str, Any]:
    """
    Build a dashboard-safe H5-C summary.

    Prefer authoritative EpisodeResult aggregate fields.
    Derive from StepResult only when an aggregate field is absent.
    """

    steps = normalize_h5c_steps(
        result
    )

    # --------------------------------------------------------
    # Statuses from trace
    # --------------------------------------------------------

    statuses = []

    for item in steps:

        status = h5c_step_status(
            item
        )

        if status:
            statuses.append(
                status
            )

    derived_status_counts = {
        "ALLOW":
            statuses.count("ALLOW"),

        "CONSTRAIN":
            statuses.count("CONSTRAIN"),

        "BLOCK":
            statuses.count("BLOCK"),
    }

    # --------------------------------------------------------
    # Pass status
    # --------------------------------------------------------

    explicit_pass = h5_field(
        result,
        "passed",
        None,
    )

    if explicit_pass is None:

        explicit_pass = h5_field(
            result,
            "pass",
            None,
        )

    # --------------------------------------------------------
    # Authoritative aggregate counts
    # --------------------------------------------------------

    allow_count = h5_field(
        result,
        "allow_count",
        None,
    )

    constrain_count = h5_field(
        result,
        "constrain_count",
        None,
    )

    block_count = h5_field(
        result,
        "block_count",
        None,
    )

    if allow_count is None:
        allow_count = derived_status_counts["ALLOW"]

    if constrain_count is None:
        constrain_count = derived_status_counts["CONSTRAIN"]

    if block_count is None:
        block_count = derived_status_counts["BLOCK"]

    # --------------------------------------------------------
    # Core metrics
    # --------------------------------------------------------

    total_reward = safe_float(
        h5_field(
            result,
            "total_reward",
            h5_field(
                result,
                "reward",
                0.0,
            ),
        )
    )

    initial_trust = safe_float(
        h5_field(
            result,
            "initial_trust",
            0.0,
        )
    )

    final_trust = safe_float(
        h5_field(
            result,
            "final_trust",
            0.0,
        )
    )

    mean_risk = safe_float(
        h5_field(
            result,
            "mean_risk",
            0.0,
        )
    )

    max_risk = safe_float(
        h5_field(
            result,
            "max_risk",
            0.0,
        )
    )

    # --------------------------------------------------------
    # Derive reward if aggregate unavailable
    # --------------------------------------------------------

    if total_reward == 0.0 and steps:

        total_reward = float(
            np.sum(
                [
                    safe_float(
                        h5_field(
                            item,
                            "reward",
                            0.0,
                        )
                    )
                    for item in steps
                ]
            )
        )

    # --------------------------------------------------------
    # Derive risk if aggregate unavailable
    # --------------------------------------------------------

    risks = [
        h5c_step_risk(item)
        for item in steps
    ]

    if mean_risk == 0.0 and risks:

        mean_risk = float(
            np.mean(risks)
        )

    if max_risk == 0.0 and risks:

        max_risk = float(
            np.max(risks)
        )

    # --------------------------------------------------------
    # Additional authoritative metrics
    # --------------------------------------------------------

    candidate_default = sum(
        h5c_step_candidate_executed(item)
        for item in steps
    )

    fallback_default = sum(
        h5c_step_fallback_executed(item)
        for item in steps
    )

    feedback_default = sum(
        h5c_step_feedback_success(item)
        for item in steps
    )

    environment_failure_default = sum(
        not h5c_step_environment_success(item)
        for item in steps
    )

    candidate_execution_count = int(
        safe_float(
            h5_field(
                result,
                "candidate_execution_count",
                candidate_default,
            )
        )
    )

    fallback_execution_count = int(
        safe_float(
            h5_field(
                result,
                "fallback_execution_count",
                fallback_default,
            )
        )
    )

    trust_feedback_successes = int(
        safe_float(
            h5_field(
                result,
                "trust_feedback_successes",
                feedback_default,
            )
        )
    )

    environment_failures = int(
        safe_float(
            h5_field(
                result,
                "environment_failures",
                environment_failure_default,
            )
        )
    )

    feedback_calls = int(
        safe_float(
            h5_field(
                result,
                "feedback_calls",
                len(steps),
            )
        )
    )

    return {

        "passed":
            explicit_pass,

        "steps":
            len(steps),

        "allow":
            int(
                safe_float(
                    allow_count
                )
            ),

        "constrain":
            int(
                safe_float(
                    constrain_count
                )
            ),

        "block":
            int(
                safe_float(
                    block_count
                )
            ),

        "reward":
            total_reward,

        "initial_trust":
            initial_trust,

        "final_trust":
            final_trust,

        "mean_risk":
            mean_risk,

        "max_risk":
            max_risk,

        "candidate_execution_count":
            candidate_execution_count,

        "fallback_execution_count":
            fallback_execution_count,

        "trust_feedback_calls":
            feedback_calls,

        "trust_feedback_successes":
            trust_feedback_successes,

        "environment_failures":
            environment_failures,
    }


# ============================================================
# H5-C TRACE DATAFRAME
# ============================================================

def h5c_steps_dataframe(
    result: Any,
) -> pd.DataFrame:

    steps = normalize_h5c_steps(
        result
    )

    rows = []

    for index, item in enumerate(
        steps,
        start=1,
    ):

        step_value = h5_field(
            item,
            "step",
            index,
        )

        rows.append(
            {
                "Step":
                    int(
                        safe_float(
                            step_value,
                            index,
                        )
                    ),

                "Governance":
                    h5c_step_status(
                        item
                    ),

                "Reason":
                    h5c_step_reason(
                        item
                    ),

                "Risk":
                    h5c_step_risk(
                        item
                    ),

                "Candidate Executed":
                    h5c_step_candidate_executed(
                        item
                    ),

                "Fallback Executed":
                    h5c_step_fallback_executed(
                        item
                    ),

                "Reward":
                    safe_float(
                        h5_field(
                            item,
                            "reward",
                            0.0,
                        )
                    ),

                "Trust Before Feedback":
                    safe_float(
                        h5_field(
                            item,
                            "trust_before_feedback",
                            h5_field(
                                item,
                                "controlled_trust",
                                0.0,
                            ),
                        )
                    ),

                "Trust After Feedback":
                    h5c_step_trust(
                        item
                    ),

                "Trust Feedback":
                    h5c_step_feedback_success(
                        item
                    ),

                "Environment":
                    h5c_step_environment_success(
                        item
                    ),

                "Modified":
                    safe_bool(
                        h5_field(
                            item,
                            "action_modified",
                            h5_field(
                                item,
                                "modified",
                                False,
                            ),
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_SESSION = {

    "env":
        None,

    "agent":
        None,

    "governance":
        None,

    "state":
        None,

    "history":
        [],

    "running":
        False,

    "runtime_ready":
        False,

    "active_seed":
        None,

    "runtime_mode":
        "Live SAC Runtime",

    "h5c_result":
        None,

    "h5c_results":
        [],

}


for key, value in DEFAULT_SESSION.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📡 TA-FDRL-IRF</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Trust-Aware Adaptive Federated Deep Reinforcement Learning
    for Intelligent Radio Fabric in 6G Networks
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"{PHASE} • "
    "Trust-aware closed-loop research dashboard"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Runtime Control"
)

runtime_mode = st.sidebar.radio(
    "Runtime Mode",
    [
        "Live SAC Runtime",
        "H5-C Validation Scenario",
    ],
    index=(
        1
        if st.session_state.runtime_mode
        == "H5-C Validation Scenario"
        else 0
    ),
)

st.session_state.runtime_mode = runtime_mode


# ------------------------------------------------------------
# Live controls
# ------------------------------------------------------------

seed_value = st.sidebar.number_input(
    "Environment Seed",
    min_value=0,
    max_value=999999,
    value=int(
        st.session_state.active_seed
        if st.session_state.active_seed is not None
        else 42
    ),
    step=1,
)

start_reset = False
one_step = False
full_episode = False
clear_history = False

auto_refresh = False
refresh_interval = 1.0


if runtime_mode == "Live SAC Runtime":

    st.sidebar.markdown(
        "---"
    )

    start_reset = st.sidebar.button(
        "▶️ Start / Reset",
        width="stretch",
    )

    one_step = st.sidebar.button(
        "⏭️ Run One Step",
        width="stretch",
    )

    full_episode = st.sidebar.button(
        "▶️ Run Full Episode",
        width="stretch",
    )

    clear_history = st.sidebar.button(
        "🧹 Clear History",
        width="stretch",
    )

    auto_refresh = st.sidebar.checkbox(
        "Auto Refresh",
        value=False,
    )

    refresh_interval = st.sidebar.slider(
        "Refresh Interval (sec)",
        min_value=0.2,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )

else:

    st.sidebar.markdown(
        "---"
    )

    scenario_names = [
        scenario_name(
            scenario
        )
        for scenario in SCENARIOS
    ]

    selected_scenario_name = st.sidebar.selectbox(
        "H5-C Scenario",
        scenario_names,
    )

    selected_scenario = next(
        scenario
        for scenario in SCENARIOS
        if scenario_name(scenario)
        == selected_scenario_name
    )

    default_h5_seed = int(
        SCENARIO_SEEDS.get(
            selected_scenario_name,
            42,
        )
    )

    h5c_seed = st.sidebar.number_input(
        "H5-C Seed",
        min_value=0,
        max_value=999999,
        value=default_h5_seed,
        step=1,
    )

    h5c_steps = st.sidebar.number_input(
        "H5-C Steps",
        min_value=1,
        max_value=MAX_STEPS,
        value=H5C_DEFAULT_STEPS,
        step=1,
    )

    run_h5c = st.sidebar.button(
        "🧪 Run Selected Scenario",
        width="stretch",
    )

    run_all_h5c = st.sidebar.button(
        "🧪 Run All 6 Scenarios",
        width="stretch",
    )

    if st.sidebar.button(
        "🧹 Clear H5-C Results",
        width="stretch",
    ):

        st.session_state.h5c_result = None
        st.session_state.h5c_results = []

        st.rerun()


# ============================================================
# SIDEBAR SYSTEM STATUS
# ============================================================

st.sidebar.markdown(
    "---"
)

st.sidebar.subheader(
    "System Configuration"
)

st.sidebar.write(
    f"**Phase:** {PHASE}"
)

st.sidebar.write(
    f"**State:** {STATE_DIM}"
)

st.sidebar.write(
    f"**Action:** {ACTION_DIM}"
)

st.sidebar.write(
    f"**Users:** {NUM_USERS}"
)

st.sidebar.write(
    f"**IRF Elements:** {NUM_RIS_ELEMENTS}"
)

st.sidebar.write(
    f"**Carrier:** "
    f"{CARRIER_FREQUENCY_HZ / 1e9:.0f} GHz"
)

st.sidebar.write(
    f"**Bandwidth:** "
    f"{BANDWIDTH_HZ / 1e6:.0f} MHz"
)

st.sidebar.write(
    f"**Polarization:** "
    f"{'ON' if POLARIZATION_ENABLED else 'OFF'}"
)

st.sidebar.write(
    f"**Polarization State:** "
    f"{'ON' if POLARIZATION_STATE_ENABLED else 'OFF'}"
)

st.sidebar.write(
    f"**Cross-Polarization:** "
    f"{CROSS_POLARIZATION_FACTOR:.2f}"
)


# ============================================================
# RUNTIME INITIALIZATION
# ============================================================

if runtime_mode == "Live SAC Runtime":

    if st.session_state.agent is None:

        try:

            with st.spinner(
                "Loading SAC checkpoint..."
            ):

                st.session_state.agent = (
                    create_agent()
                )

        except Exception as exc:

            st.session_state.runtime_ready = False

            st.error(
                "❌ SAC agent initialization failed."
            )

            st.exception(
                exc
            )

            st.stop()

    if st.session_state.env is None:

        try:

            (
                st.session_state.env,
                st.session_state.state,
            ) = reset_environment(
                int(seed_value)
            )

            st.session_state.governance = (
                create_governance()
            )

            st.session_state.active_seed = (
                int(seed_value)
            )

            st.session_state.runtime_ready = True

        except Exception as exc:

            st.session_state.runtime_ready = False

            st.error(
                "❌ Runtime initialization failed."
            )

            st.exception(
                exc
            )

            st.stop()


# ============================================================
# START / RESET
# ============================================================

if (
    runtime_mode == "Live SAC Runtime"
    and start_reset
):

    try:

        (
            st.session_state.env,
            st.session_state.state,
        ) = reset_environment(
            int(seed_value)
        )

        # Fresh GovernanceEngine = fresh TrustEngine
        st.session_state.governance = (
            create_governance()
        )

        st.session_state.history = []

        st.session_state.running = True

        st.session_state.runtime_ready = True

        st.session_state.active_seed = (
            int(seed_value)
        )

        st.rerun()

    except Exception as exc:

        st.error(
            "Runtime reset failed."
        )

        st.exception(
            exc
        )


# ============================================================
# CLEAR HISTORY
# ============================================================

if (
    runtime_mode == "Live SAC Runtime"
    and clear_history
):

    st.session_state.history = []

    st.rerun()


# ============================================================
# H5-C RUN SINGLE SCENARIO
# ============================================================

if (
    runtime_mode
    == "H5-C Validation Scenario"
    and run_h5c
):

    try:

        with st.spinner(
            f"Running H5-C scenario: "
            f"{selected_scenario_name}"
        ):

            result = run_h5c_scenario(
                selected_scenario,
                int(h5c_seed),
                int(h5c_steps),
            )

        st.session_state.h5c_result = result

    except Exception as exc:

        st.session_state.h5c_result = None

        st.error(
            "H5-C scenario execution failed."
        )

        st.exception(
            exc
        )


# ============================================================
# H5-C RUN ALL SCENARIOS
# ============================================================

if (
    runtime_mode
    == "H5-C Validation Scenario"
    and run_all_h5c
):

    results = []

    progress = st.progress(
        0.0
    )

    try:

        scenario_count = len(
            SCENARIOS
        )

        for index, scenario in enumerate(
            SCENARIOS
        ):

            name = scenario_name(
                scenario
            )

            seed = int(
                SCENARIO_SEEDS.get(
                    name,
                    42,
                )
            )

            result = run_h5c_scenario(
                scenario,
                seed,
                int(h5c_steps),
            )

            results.append(
                (
                    name,
                    seed,
                    result,
                )
            )

            progress.progress(
                (index + 1)
                / max(
                    scenario_count,
                    1,
                )
            )

        st.session_state.h5c_results = (
            results
        )

        st.success(
            "H5-C all-scenario validation completed."
        )

    except Exception as exc:

        st.error(
            "H5-C all-scenario validation failed."
        )

        st.exception(
            exc
        )


# ============================================================
# H5-C VALIDATION DISPLAY
# ============================================================

if runtime_mode == "H5-C Validation Scenario":

    st.markdown(
        """
        <div class="validation-box">
            <b>🧪 H5-C VALIDATION MODE</b><br>
            Real GovernanceEngine → Real IRFEnvironment
            → Real TrustEngine post-execution feedback.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "H5-C is the controlled governance closed-loop "
        "validation oracle. The dashboard reuses the benchmark "
        "implementation rather than duplicating its scenario logic."
    )

    # ========================================================
    # SINGLE RESULT
    # ========================================================

    if st.session_state.h5c_result is not None:

        result = (
            st.session_state.h5c_result
        )

        summary = h5c_summary_from_result(
            result
        )

        st.header(
            "H5-C Scenario Result"
        )

        passed = summary["passed"]

        if passed is True:

            st.success(
                "✅ H5-C SCENARIO PASSED"
            )

        elif passed is False:

            st.error(
                "❌ H5-C SCENARIO FAILED"
            )

        else:

            coverage_ok = (
                summary["steps"] > 0
                and (
                    summary["allow"]
                    + summary["constrain"]
                    + summary["block"]
                )
                == summary["steps"]
            )

            if coverage_ok:

                st.success(
                    "✅ H5-C TRACE EXECUTED"
                )

            else:

                st.warning(
                    "H5-C result returned without "
                    "an explicit pass flag."
                )

        h1, h2, h3, h4, h5, h6 = st.columns(6)

        h1.metric(
            "Steps",
            summary["steps"],
        )

        h2.metric(
            "ALLOW",
            summary["allow"],
        )

        h3.metric(
            "CONSTRAIN",
            summary["constrain"],
        )

        h4.metric(
            "BLOCK",
            summary["block"],
        )

        h5.metric(
            "Total Reward",
            f"{summary['reward']:.4f}",
        )

        h6.metric(
            "Mean Risk",
            f"{summary['mean_risk']:.4f}",
        )

        t1, t2, t3, t4 = st.columns(4)

        t1.metric(
            "Initial Trust",
            f"{summary['initial_trust']:.6f}",
        )

        t2.metric(
            "Final Trust",
            f"{summary['final_trust']:.6f}",
        )

        t3.metric(
            "Max Risk",
            f"{summary['max_risk']:.6f}",
        )

        t4.metric(
            "Feedback Success",
            f"{summary['trust_feedback_successes']}/"
            f"{summary['trust_feedback_calls']}",
        )

        e1, e2, e3 = st.columns(3)

        e1.metric(
            "Candidate Executions",
            summary[
                "candidate_execution_count"
            ],
        )

        e2.metric(
            "Fallback Executions",
            summary[
                "fallback_execution_count"
            ],
        )

        e3.metric(
            "Environment Failures",
            summary[
                "environment_failures"
            ],
        )

        # ----------------------------------------------------
        # Raw result object
        # ----------------------------------------------------

        with st.expander(
            "H5-C Result Object"
        ):

            result_data = {}

            if hasattr(
                result,
                "__dict__",
            ):

                result_data = dict(
                    result.__dict__
                )

            elif isinstance(
                result,
                dict,
            ):

                result_data = result

            else:

                result_data = {
                    "result":
                        str(result)
                }

            scalar_rows = []

            for key, value in result_data.items():

                if isinstance(
                    value,
                    (
                        list,
                        tuple,
                        dict,
                    ),
                ):
                    continue

                scalar_rows.append(
                    {
                        "Field":
                            key,
                        "Value":
                            value,
                    }
                )

            safe_display_dataframe(
                pd.DataFrame(
                    scalar_rows
                ),
                hide_index=True,
                stringify_object_columns=True,
            )

        # ----------------------------------------------------
        # Step trace
        # ----------------------------------------------------

        steps_df = h5c_steps_dataframe(
            result
        )

        if not steps_df.empty:

            st.header(
                "H5-C Step Trace"
            )

            safe_display_dataframe(
                steps_df,
                hide_index=True,
                stringify_object_columns=True,
            )

            st.subheader(
                "Governance Status Timeline"
            )

            status_chart = (
                steps_df[
                    [
                        "Step",
                        "Risk",
                    ]
                ]
                .set_index("Step")
            )

            st.line_chart(
                arrow_safe_dataframe(
                    status_chart
                ),
                width="stretch",
            )

    # ========================================================
    # ALL RESULTS
    # ========================================================

    if st.session_state.h5c_results:

        st.header(
            "H5-C Six-Scenario Aggregate"
        )

        aggregate_rows = []

        for (
            name,
            seed,
            result,
        ) in st.session_state.h5c_results:

            summary = h5c_summary_from_result(
                result
            )

            aggregate_rows.append(
                {
                    "Scenario":
                        name,

                    "Seed":
                        seed,

                    "Passed":
                        summary["passed"],

                    "Steps":
                        summary["steps"],

                    "ALLOW":
                        summary["allow"],

                    "CONSTRAIN":
                        summary["constrain"],

                    "BLOCK":
                        summary["block"],

                    "Candidate":
                        summary[
                            "candidate_execution_count"
                        ],

                    "Fallback":
                        summary[
                            "fallback_execution_count"
                        ],

                    "Feedback":
                        summary[
                            "trust_feedback_successes"
                        ],

                    "Total Reward":
                        summary["reward"],

                    "Initial Trust":
                        summary["initial_trust"],

                    "Final Trust":
                        summary["final_trust"],

                    "Mean Risk":
                        summary["mean_risk"],

                    "Max Risk":
                        summary["max_risk"],

                    "Env Failures":
                        summary[
                            "environment_failures"
                        ],
                }
            )

        aggregate_df = pd.DataFrame(
            aggregate_rows
        )

        safe_display_dataframe(
            aggregate_df,
            hide_index=True,
            stringify_object_columns=True,
        )

        # ----------------------------------------------------
        # Aggregate totals
        # ----------------------------------------------------

        total_steps = int(
            aggregate_df["Steps"].sum()
        )

        total_allow = int(
            aggregate_df["ALLOW"].sum()
        )

        total_constrain = int(
            aggregate_df["CONSTRAIN"].sum()
        )

        total_block = int(
            aggregate_df["BLOCK"].sum()
        )

        total_candidate = int(
            aggregate_df["Candidate"].sum()
        )

        total_fallback = int(
            aggregate_df["Fallback"].sum()
        )

        total_feedback = int(
            aggregate_df["Feedback"].sum()
        )

        total_environment_failures = int(
            aggregate_df["Env Failures"].sum()
        )

        total_reward = safe_float(
            aggregate_df[
                "Total Reward"
            ].sum()
        )

        a1, a2, a3, a4, a5, a6 = st.columns(6)

        a1.metric(
            "Total Steps",
            total_steps,
        )

        a2.metric(
            "ALLOW",
            total_allow,
        )

        a3.metric(
            "CONSTRAIN",
            total_constrain,
        )

        a4.metric(
            "BLOCK",
            total_block,
        )

        a5.metric(
            "Candidate",
            total_candidate,
        )

        a6.metric(
            "Fallback",
            total_fallback,
        )

        b1, b2, b3 = st.columns(3)

        b1.metric(
            "Aggregate Reward",
            f"{total_reward:.6f}",
        )

        b2.metric(
            "Trust Feedback",
            f"{total_feedback}/{total_steps}",
        )

        b3.metric(
            "Environment Failures",
            total_environment_failures,
        )

        # ----------------------------------------------------
        # Rates
        # ----------------------------------------------------

        if total_steps > 0:

            allow_rate = (
                100.0
                * total_allow
                / total_steps
            )

            constrain_rate = (
                100.0
                * total_constrain
                / total_steps
            )

            block_rate = (
                100.0
                * total_block
                / total_steps
            )

            candidate_rate = (
                100.0
                * total_candidate
                / total_steps
            )

            fallback_rate = (
                100.0
                * total_fallback
                / total_steps
            )

            feedback_rate = (
                100.0
                * total_feedback
                / total_steps
            )

        else:

            allow_rate = 0.0
            constrain_rate = 0.0
            block_rate = 0.0
            candidate_rate = 0.0
            fallback_rate = 0.0
            feedback_rate = 0.0

        rate_df = pd.DataFrame(
            {
                "Metric": [
                    "ALLOW",
                    "CONSTRAIN",
                    "BLOCK",
                    "Candidate Execution",
                    "Fallback Execution",
                    "Feedback Success",
                ],
                "Rate (%)": [
                    allow_rate,
                    constrain_rate,
                    block_rate,
                    candidate_rate,
                    fallback_rate,
                    feedback_rate,
                ],
            }
        )

        st.subheader(
            "H5-C Aggregate Rates"
        )

        safe_display_dataframe(
            rate_df,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Status distribution
        # ----------------------------------------------------

        distribution = pd.DataFrame(
            {
                "Status":
                    [
                        "ALLOW",
                        "CONSTRAIN",
                        "BLOCK",
                    ],

                "Count":
                    [
                        total_allow,
                        total_constrain,
                        total_block,
                    ],
            }
        )

        st.bar_chart(
            distribution.set_index(
                "Status"
            ),
            width="stretch",
        )

        # ----------------------------------------------------
        # Aggregate pass status
        # ----------------------------------------------------

        passed_values = aggregate_df[
            "Passed"
        ].tolist()

        all_passed = (
            len(passed_values) == len(SCENARIOS)
            and all(
                value is True
                for value in passed_values
            )
        )

        if all_passed:

            st.success(
                "✅ ALL 6 H5-C SCENARIOS PASSED"
            )

        else:

            st.error(
                "❌ One or more H5-C scenarios failed."
            )

    # ========================================================
    # EXPECTED ARCHITECTURE
    # ========================================================

    st.header(
        "H5-C Validation Architecture"
    )

    st.code(
        """
Candidate Action
       |
       v
REAL GovernanceEngine.evaluate()
       |
       +-----------------------------+
       |             |               |
     ALLOW       CONSTRAIN         BLOCK
       |             |               |
       |        SafeEnvelope          |
       |             |               |
       +-------------+---------------+
                     |
                     v
             Executed Action
                     |
                     v
          REAL IRFEnvironment.step()
                     |
                     v
             Actual Telemetry
                     |
                     v
      REAL GovernanceEngine.update_trust()
                     |
                     v
        TrustEngine.post_execution_feedback()
                     |
                     v
             Updated Trust State
                     |
                     v
                  Next Step
        """,
        language="text",
    )

    st.info(
        "H5-C validates the execution gate itself: "
        "ALLOW executes the governed action, CONSTRAIN executes "
        "the safe-envelope action, and BLOCK prevents candidate "
        "execution while using a fallback only for environment "
        "continuity."
    )

    st.divider()

    st.caption(
        "H5-C Validation Mode complete."
    )

    st.stop()


# ============================================================
# LIVE RUNTIME STATUS
# ============================================================

if st.session_state.runtime_ready:

    st.markdown(
        """
        <div class="live-box">
            <span class="status-live">
                🟢 LIVE CLOSED-LOOP RUNTIME READY
            </span><br>
            SAC → Governance → IRF → Telemetry
            → TrustEngine Feedback
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CURRENT HISTORY
# ============================================================

history = st.session_state.history


# ============================================================
# SYSTEM METRICS
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)

if history:

    latest = history[-1]

    col1.metric(
        "Reward",
        f"{latest['reward']:.4f}",
    )

    col2.metric(
        "Trust",
        f"{latest['trust']:.4f}",
    )

    col3.metric(
        "Risk",
        f"{latest['risk']:.4f}",
    )

    col4.metric(
        "Spectral Efficiency",
        f"{latest['spectral_efficiency']:.4f}",
    )

    col5.metric(
        "Energy Efficiency",
        f"{latest['energy_efficiency']:.3e}",
    )

else:

    col1.metric("Reward", "—")
    col2.metric("Trust", "—")
    col3.metric("Risk", "—")
    col4.metric("Spectral Efficiency", "—")
    col5.metric("Energy Efficiency", "—")


# ============================================================
# RUN ONE STEP
# ============================================================

if one_step:

    try:

        next_state, record = run_step(
            st.session_state.env,
            st.session_state.agent,
            st.session_state.governance,
            st.session_state.state,
        )

        record["step"] = (
            len(
                st.session_state.history
            )
            + 1
        )

        st.session_state.state = (
            next_state
        )

        st.session_state.history.append(
            record
        )

        st.session_state.running = True

        if (
            record["terminated"]
            or record["truncated"]
        ):

            st.session_state.running = False

        st.rerun()

    except Exception as exc:

        st.error(
            "One-step execution failed."
        )

        st.exception(
            exc
        )


# ============================================================
# FULL EPISODE
# ============================================================

if full_episode:

    try:

        (
            st.session_state.env,
            state,
        ) = reset_environment(
            int(seed_value)
        )

        # Fresh governance state
        st.session_state.governance = (
            create_governance()
        )

        st.session_state.history = []

        progress = st.progress(
            0.0
        )

        for step in range(
            MAX_STEPS
        ):

            next_state, record = run_step(
                st.session_state.env,
                st.session_state.agent,
                st.session_state.governance,
                state,
            )

            record["step"] = (
                step + 1
            )

            st.session_state.history.append(
                record
            )

            state = next_state

            progress.progress(
                (step + 1)
                / MAX_STEPS
            )

            if (
                record["terminated"]
                or record["truncated"]
            ):

                break

        st.session_state.state = (
            state
        )

        st.session_state.running = False

        st.session_state.active_seed = (
            int(seed_value)
        )

        st.session_state.runtime_ready = True

        st.rerun()

    except Exception as exc:

        st.error(
            "Full episode execution failed."
        )

        st.exception(
            exc
        )


# ============================================================
# CURRENT GOVERNANCE DECISION
# ============================================================

st.header(
    "Current Governance Decision"
)

if history:

    latest = history[-1]

    decision_status = (
        latest["governance"]
    )

    if decision_status == "ALLOW":

        st.success(
            f"ALLOW — "
            f"{latest['reason'] or 'Policy accepted'}"
        )

    elif decision_status == "CONSTRAIN":

        st.warning(
            f"CONSTRAIN — "
            f"{latest['reason'] or 'Safe envelope applied'}"
        )

    elif decision_status in (
        "BLOCK",
        "REVIEW",
        "REJECT",
    ):

        st.error(
            f"{decision_status} — "
            f"{latest['reason'] or 'Direct execution prevented'}"
        )

    else:

        st.info(
            decision_status
        )

    d1, d2, d3, d4 = st.columns(4)

    risk_level = (
        "HIGH"
        if latest["risk"] >= 0.8
        else "MEDIUM"
        if latest["risk"] >= 0.5
        else "LOW"
    )

    d1.metric(
        "Risk Level",
        risk_level,
    )

    d2.metric(
        "Risk Score",
        f"{latest['risk']:.4f}",
    )

    d3.metric(
        "Governance Trust",
        f"{latest['trust']:.4f}",
    )

    d4.metric(
        "Action Modified",
        "YES"
        if latest["modified"]
        else "NO",
    )

    st.write(
        f"**Policy Allowed:** "
        f"{latest['policy_allowed']}"
    )

    st.write(
        f"**Execution Allowed:** "
        f"{latest['execution_allowed']}"
    )

    st.write(
        f"**Candidate Executed:** "
        f"{latest['candidate_executed']}"
    )

    st.write(
        f"**Fallback Executed:** "
        f"{latest['fallback_executed']}"
    )

    st.write(
        f"**Trust Feedback Called:** "
        f"{latest['trust_feedback_called']}"
    )

    st.write(
        f"**Trust Feedback Success:** "
        f"{latest['trust_feedback_success']}"
    )

    if latest["trust_feedback_error"]:

        st.warning(
            "Trust feedback error: "
            f"{latest['trust_feedback_error']}"
        )

else:

    st.info(
        "No governance decision yet. "
        "Run one step or a full episode."
    )


# ============================================================
# TRUST ENGINE STATE
# ============================================================

st.header(
    "TrustEngine State"
)

if history:

    trust_state = get_governance_trust_state(
        st.session_state.governance
    )

    lengths = history_lengths(
        trust_state
    )

    ts1, ts2, ts3, ts4, ts5 = st.columns(5)

    ts1.metric(
        "Current Trust",
        f"{safe_float(trust_state.get('trust_score', 0.0)):.6f}",
    )

    ts2.metric(
        "Trust History",
        lengths["trust_history"],
    )

    ts3.metric(
        "Action History",
        lengths["action_history"],
    )

    ts4.metric(
        "Risk History",
        lengths["risk_history"],
    )

    ts5.metric(
        "Policy History",
        lengths["policy_history"],
    )

    expected_executions = len(
        history
    )

    actual_action_history = (
        lengths["action_history"]
    )

    actual_risk_history = (
        lengths["risk_history"]
    )

    actual_policy_history = (
        lengths["policy_history"]
    )

    actual_outcome_history = (
        lengths["outcome_history"]
    )

    actual_trust_history = (
        lengths["trust_history"]
    )

    invariant_pass = (
        actual_action_history
        == expected_executions
        and actual_risk_history
        == expected_executions
        and actual_policy_history
        == expected_executions
        and actual_outcome_history
        == expected_executions
        and actual_trust_history
        == expected_executions + 1
    )

    if invariant_pass:

        st.success(
            "✅ Trust history invariant PASSED: "
            "after N executions → "
            "action/risk/policy/outcome histories = N, "
            "trust history = N+1."
        )

    else:

        st.error(
            "❌ Trust history invariant FAILED."
        )

        st.write(
            {
                "executions":
                    expected_executions,

                "action_history":
                    actual_action_history,

                "risk_history":
                    actual_risk_history,

                "policy_history":
                    actual_policy_history,

                "outcome_history":
                    actual_outcome_history,

                "trust_history":
                    actual_trust_history,
            }
        )

else:

    st.info(
        "TrustEngine state will appear after execution."
    )


# ============================================================
# LIVE TELEMETRY
# ============================================================

st.header(
    "Live Telemetry"
)

if history:

    df = pd.DataFrame(
        history
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Performance",
            "Trust & Risk",
            "Network",
            "Polarization",
            "Closed Loop",
        ]
    )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    with tab1:

        performance_df = df[
            [
                "step",
                "reward",
                "spectral_efficiency",
            ]
        ].set_index(
            "step"
        )

        st.line_chart(
            arrow_safe_dataframe(
                performance_df
            ),
            width="stretch",
        )

        ee_df = df[
            [
                "step",
                "energy_efficiency",
            ]
        ].set_index(
            "step"
        )

        st.line_chart(
            arrow_safe_dataframe(
                ee_df
            ),
            width="stretch",
        )

    # ========================================================
    # TRUST / RISK
    # ========================================================

    with tab2:

        trust_risk_df = df[
            [
                "step",
                "environment_trust",
                "pre_governance_trust",
                "trust",
                "risk",
            ]
        ].set_index(
            "step"
        )

        st.line_chart(
            arrow_safe_dataframe(
                trust_risk_df
            ),
            width="stretch",
        )

    # ========================================================
    # NETWORK
    # ========================================================

    with tab3:

        network_df = df[
            [
                "step",
                "mean_sinr",
                "mean_interference",
                "mean_queue",
                "mean_power",
            ]
        ].set_index(
            "step"
        )

        st.line_chart(
            arrow_safe_dataframe(
                network_df
            ),
            width="stretch",
        )

    # ========================================================
    # POLARIZATION
    # ========================================================

    with tab4:

        polarization_df = df[
            [
                "step",
                "pqi",
                "xpi",
            ]
        ].set_index(
            "step"
        )

        if (
            polarization_df["pqi"].abs().sum()
            > 0
            or polarization_df["xpi"].abs().sum()
            > 0
        ):

            st.line_chart(
                arrow_safe_dataframe(
                    polarization_df
                ),
                width="stretch",
            )

        else:

            st.info(
                "PQI/XPI diagnostics are not exposed "
                "through the current environment info."
            )

        st.caption(
            "Phase-4B.1: polarization diagnostics remain "
            "observational. PQI/XPI are not included in "
            "the SAC state because polarization_state_enabled "
            "is disabled."
        )

    # ========================================================
    # CLOSED LOOP
    # ========================================================

    with tab5:

        closed_loop_df = df[
            [
                "step",
                "candidate_executed",
                "fallback_executed",
                "trust_feedback_called",
                "trust_feedback_success",
            ]
        ].set_index(
            "step"
        ).astype(
            float
        )

        st.line_chart(
            arrow_safe_dataframe(
                closed_loop_df
            ),
            width="stretch",
        )

        st.caption(
            "1.0 = True / successful execution path."
        )

else:

    st.info(
        "Live telemetry will appear after execution."
    )


# ============================================================
# GOVERNANCE STATISTICS
# ============================================================

st.header(
    "Governance Statistics"
)

if history:

    df = pd.DataFrame(
        history
    )

    total_decisions = len(
        df
    )

    allow_count = int(
        (
            df["governance"]
            == "ALLOW"
        ).sum()
    )

    constrain_count = int(
        (
            df["governance"]
            == "CONSTRAIN"
        ).sum()
    )

    block_count = int(
        (
            df["governance"]
            == "BLOCK"
        ).sum()
    )

    review_count = int(
        (
            df["governance"]
            == "REVIEW"
        ).sum()
    )

    reject_count = int(
        (
            df["governance"]
            == "REJECT"
        ).sum()
    )

    g1, g2, g3, g4, g5 = st.columns(5)

    g1.metric(
        "Total",
        total_decisions,
    )

    g2.metric(
        "ALLOW",
        allow_count,
    )

    g3.metric(
        "CONSTRAIN",
        constrain_count,
    )

    g4.metric(
        "BLOCK",
        block_count,
    )

    g5.metric(
        "REVIEW / REJECT",
        review_count + reject_count,
    )

    distribution_df = pd.DataFrame(
        {
            "Governance":
                [
                    "ALLOW",
                    "CONSTRAIN",
                    "BLOCK",
                    "REVIEW",
                    "REJECT",
                ],

            "Count":
                [
                    allow_count,
                    constrain_count,
                    block_count,
                    review_count,
                    reject_count,
                ],
        }
    )

    st.bar_chart(
        distribution_df.set_index(
            "Governance"
        ),
        width="stretch",
    )

else:

    st.info(
        "No governance decisions recorded."
    )


# ============================================================
# ACTION ENFORCEMENT
# ============================================================

st.header(
    "Action Enforcement"
)

if history:

    df = pd.DataFrame(
        history
    )

    modified_count = int(
        df["modified"].sum()
    )

    fallback_count = int(
        df["fallback_executed"].sum()
    )

    candidate_count = int(
        df["candidate_executed"].sum()
    )

    max_delta = safe_float(
        df["max_action_delta"].max()
    )

    mean_delta = safe_float(
        df["mean_action_delta"].mean()
    )

    e1, e2, e3, e4, e5 = st.columns(5)

    e1.metric(
        "Modified Decisions",
        modified_count,
    )

    e2.metric(
        "Candidate Executions",
        candidate_count,
    )

    e3.metric(
        "Fallback Executions",
        fallback_count,
    )

    e4.metric(
        "Maximum Action Δ",
        f"{max_delta:.4f}",
    )

    e5.metric(
        "Mean Action Δ",
        f"{mean_delta:.4f}",
    )

    enforcement_df = df[
        [
            "step",
            "max_action_delta",
            "mean_action_delta",
        ]
    ].set_index(
        "step"
    )

    st.line_chart(
        arrow_safe_dataframe(
            enforcement_df
        ),
        width="stretch",
    )

    st.caption(
        "BLOCK fallback is not counted as an action "
        "modification. Candidate execution and fallback "
        "execution are tracked independently."
    )

else:

    st.info(
        "No action enforcement data yet."
    )


# ============================================================
# POLICY / GOVERNANCE INTERVENTIONS
# ============================================================

st.header(
    "Policy / Governance Interventions"
)

if history:

    df = pd.DataFrame(
        history
    )

    interventions = df[
        ~df["governance"].isin(
            ["ALLOW"]
        )
    ]

    if len(
        interventions
    ) == 0:

        st.success(
            "No non-ALLOW governance interventions "
            "were recorded."
        )

    else:

        st.warning(
            f"{len(interventions)} "
            "non-ALLOW governance decision(s) recorded."
        )

        display_columns = [
            "step",
            "governance",
            "reason",
            "risk",
            "trust",
            "modified",
            "candidate_executed",
            "fallback_executed",
        ]

        safe_display_dataframe(
            interventions[
                display_columns
            ],
            hide_index=True,
            stringify_object_columns=True,
        )

else:

    st.info(
        "Governance intervention analysis will appear "
        "after execution."
    )


# ============================================================
# RECENT DECISIONS
# ============================================================

st.header(
    "Recent Decisions"
)

if history:

    df = pd.DataFrame(
        history
    )

    recent_columns = [
        "step",
        "reward",
        "environment_trust",
        "trust",
        "risk",
        "governance",
        "reason",
        "modified",
        "candidate_executed",
        "fallback_executed",
        "trust_feedback_success",
    ]

    recent_df = df[
        recent_columns
    ].tail(
        20
    )

    safe_display_dataframe(
        recent_df,
        hide_index=True,
        stringify_object_columns=True,
    )

else:

    st.info(
        "No decisions recorded yet."
    )


# ============================================================
# CLOSED-LOOP VALIDATION
# ============================================================

st.header(
    "Closed-Loop Validation"
)

if history:

    df = pd.DataFrame(
        history
    )

    total_steps = len(
        df
    )

    feedback_called = int(
        df[
            "trust_feedback_called"
        ].sum()
    )

    feedback_success = int(
        df[
            "trust_feedback_success"
        ].sum()
    )

    candidate_executions = int(
        df[
            "candidate_executed"
        ].sum()
    )

    fallback_executions = int(
        df[
            "fallback_executed"
        ].sum()
    )

    invariant_rows = (
        df[
            "trust_history_length"
        ]
        == (
            np.arange(
                1,
                total_steps + 1,
            )
            + 1
        )
    )

    feedback_pass = (
        feedback_called
        == total_steps
        and feedback_success
        == total_steps
    )

    if (
        feedback_pass
        and invariant_rows.all()
    ):

        st.success(
            "✅ REAL CLOSED-LOOP VALIDATION PASSED — "
            "all executed steps received successful "
            "post-execution TrustEngine feedback and "
            "trust history advanced as N+1."
        )

    else:

        st.error(
            "❌ REAL CLOSED-LOOP VALIDATION FAILED."
        )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Steps",
        total_steps,
    )

    c2.metric(
        "Feedback Success",
        f"{feedback_success}/{total_steps}",
    )

    c3.metric(
        "Candidate Executions",
        candidate_executions,
    )

    c4.metric(
        "Fallback Executions",
        fallback_executions,
    )

else:

    st.info(
        "Run the live runtime to validate the "
        "Governance → IRF → TrustEngine loop."
    )


# ============================================================
# CSV EXPORT
# ============================================================

st.header(
    "Research Data"
)

if history:

    df = pd.DataFrame(
        history
    )

    csv_data = df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        label="⬇️ Export Live Telemetry CSV",
        data=csv_data,
        file_name=(
            "ta_fdrl_irf_phase4b1_live_telemetry.csv"
        ),
        mime="text/csv",
        width="stretch",
    )


# ============================================================
# PHASE-4B.1 RESEARCH INTERPRETATION
# ============================================================

st.header(
    "Phase-4B.1 Research Interpretation"
)

r1, r2, r3 = st.columns(3)

with r1:

    st.metric(
        "State Dimension",
        STATE_DIM,
    )

    st.caption(
        "20 users × 5 state features"
    )

with r2:

    st.metric(
        "Action Dimension",
        ACTION_DIM,
    )

    st.caption(
        "SAC-compatible action space"
    )

with r3:

    st.metric(
        "Cross-Polarization",
        f"{CROSS_POLARIZATION_FACTOR:.2f}",
    )

    st.caption(
        "Polarization-aware PHY"
    )


st.info(
    "Phase-4B.1 keeps the validated 100-dimensional "
    "SAC state unchanged while enabling polarization-aware "
    "PHY diagnostics."
)

st.warning(
    "PQI/XPI remain diagnostic variables in Phase-4B.1. "
    "They are not part of the SAC observation state. "
    "Therefore this dashboard does not claim that the "
    "current SAC policy learned polarization-aware control. "
    "Phase-4B.2 requires a 140-dimensional observation state "
    "and fresh SAC training."
)


# ============================================================
# RUNTIME ARCHITECTURE
# ============================================================

st.header(
    "Runtime Architecture"
)

st.code(
    """
                         TA-FDRL-IRF
                              |
                              v
                        SAC Controller
                        State = 100
                        Action = 40
                              |
                              v
                       Proposed Action
                              |
                              v
                    GovernanceEngine.evaluate()
                              |
                 +------------+------------+
                 |            |            |
               ALLOW       CONSTRAIN     BLOCK
                 |            |            |
                 |       Safe Envelope     |
                 |            |            |
                 +------------+------------+
                              |
                              v
                      Executed Action
                              |
                              v
                    IRFEnvironment.step()
                              |
                              v
                  Polarization-aware PHY
                              |
                    +---------+---------+
                    |         |         |
                   SE        EE       Trust
                    |         |         |
                    +---------+---------+
                              |
                              v
                       Actual Telemetry
                              |
                              v
                  GovernanceEngine.update_trust()
                              |
                              v
              TrustEngine.post_execution_feedback()
                              |
                              v
                     Updated Trust State
                              |
                              +------> Dashboard
                              |
                              +------> Next SAC State
    """,
    language="text",
)


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.header(
    "System Information"
)

system_data = {

    "Project":
        PROJECT_NAME,

    "Phase":
        PHASE,

    "Controller":
        "Soft Actor-Critic",

    "State":
        f"{STATE_DIM} dimensions",

    "Action":
        f"{ACTION_DIM} dimensions",

    "Users":
        NUM_USERS,

    "IRF Elements":
        NUM_RIS_ELEMENTS,

    "Carrier":
        f"{CARRIER_FREQUENCY_HZ / 1e9:.0f} GHz",

    "Bandwidth":
        f"{BANDWIDTH_HZ / 1e6:.0f} MHz",

    "Polarization PHY":
        "Enabled",

    "Polarization State":
        "Disabled",

    "Cross-Polarization Factor":
        CROSS_POLARIZATION_FACTOR,

    "RIS Optimization":
        "Disabled",

    "Adaptive Trust":
        "Enabled",

    "Governance":
        "Real GovernanceEngine",

    "Trust":
        "Real TrustEngine",

    "Environment":
        "Real IRFEnvironment",
}

system_df = pd.DataFrame(
    list(
        system_data.items()
    ),
    columns=[
        "Parameter",
        "Value",
    ],
)

system_df["Parameter"] = (
    system_df["Parameter"]
    .astype(str)
)

system_df["Value"] = (
    system_df["Value"]
    .astype(str)
)

st.dataframe(
    system_df,
    width="stretch",
    hide_index=True,
)


# ============================================================
# CHECKPOINT INFORMATION
# ============================================================

st.header(
    "SAC Checkpoint"
)

try:

    checkpoint_relative = (
        CHECKPOINT_PATH.relative_to(
            PROJECT_ROOT
        )
    )

except ValueError:

    checkpoint_relative = (
        CHECKPOINT_PATH
    )

st.code(
    str(
        checkpoint_relative
    ),
    language="text",
)

if CHECKPOINT_PATH.exists():

    checkpoint_size_mb = (
        CHECKPOINT_PATH.stat().st_size
        / (1024 * 1024)
    )

    st.success(
        "SAC checkpoint available."
    )

    st.caption(
        f"Checkpoint size: "
        f"{checkpoint_size_mb:.2f} MB"
    )

else:

    st.error(
        "SAC checkpoint missing."
    )


# ============================================================
# LIVE AUTO REFRESH
# ============================================================

if (
    runtime_mode == "Live SAC Runtime"
    and auto_refresh
    and st.session_state.running
):

    time.sleep(
        float(
            refresh_interval
        )
    )

    if (
        len(
            st.session_state.history
        )
        < MAX_STEPS
    ):

        try:

            next_state, record = run_step(
                st.session_state.env,
                st.session_state.agent,
                st.session_state.governance,
                st.session_state.state,
            )

            record["step"] = (
                len(
                    st.session_state.history
                )
                + 1
            )

            st.session_state.state = (
                next_state
            )

            st.session_state.history.append(
                record
            )

            if (
                record["terminated"]
                or record["truncated"]
            ):

                st.session_state.running = False

        except Exception as exc:

            st.session_state.running = False

            st.error(
                "Auto-refresh execution failed."
            )

            st.exception(
                exc
            )

        st.rerun()

    else:

        st.session_state.running = False


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TA-FDRL-IRF • Phase-4B.1 • "
    "Trust-Aware Governance • "
    "Polarization-Aware PHY • "
    "Real Governance → IRF → TrustEngine Closed Loop"
)