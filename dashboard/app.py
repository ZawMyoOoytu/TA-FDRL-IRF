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
except Exception as exc:
    st.error("TA-FDRL-IRF project imports failed.")
    st.exception(exc)
    st.stop()


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="TA-FDRL-IRF Live Research Dashboard",
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SAFE NUMERIC HELPERS
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert scalar-like input to a finite float."""

    try:
        arr = np.asarray(value)

        if arr.size == 0:
            return float(default)

        result = float(arr.reshape(-1)[0])

        if not np.isfinite(result):
            return float(default)

        return result

    except Exception:
        return float(default)


def safe_mean(value: Any, default: float = 0.0) -> float:
    """Finite mean for scalar or array-like input."""

    try:
        arr = np.asarray(value, dtype=float)

        if arr.size == 0:
            return float(default)

        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            return float(default)

        return float(np.mean(arr))

    except Exception:
        return float(default)


def safe_bool(value: Any, default: bool = False) -> bool:
    """Safely convert a value to bool."""

    try:
        return bool(value)
    except Exception:
        return bool(default)


# ============================================================
# DATAFRAME SAFETY
# ============================================================

def arrow_safe_dataframe(
    dataframe: pd.DataFrame,
    stringify_object_columns: bool = False,
) -> pd.DataFrame:
    """
    Make DataFrame safe for Streamlit/PyArrow display.

    Mixed object columns can otherwise produce ArrowTypeError.
    """

    df = dataframe.copy()

    for column in df.columns:

        series = df[column]

        if series.dtype == "object":

            if stringify_object_columns:

                df[column] = series.map(
                    lambda value: ""
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
                        lambda value: ""
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
# RESET RESULT COMPATIBILITY
# ============================================================

def extract_state(reset_result: Any) -> np.ndarray:
    """
    Support:

        reset() -> state

    and:

        reset() -> (state, info)
    """

    if isinstance(reset_result, tuple):

        if len(reset_result) == 0:
            raise ValueError("Environment reset returned an empty tuple.")

        state = reset_result[0]

    else:

        state = reset_result

    state = np.asarray(
        state,
        dtype=np.float32,
    ).reshape(-1)

    validate_state(state)

    return state


# ============================================================
# STATE / ACTION VALIDATION
# ============================================================

def validate_state(state: np.ndarray) -> None:

    if state.size != STATE_DIM:

        raise ValueError(
            "State dimension mismatch: "
            f"expected {STATE_DIM}, "
            f"got {state.size}"
        )

    if not np.all(np.isfinite(state)):

        raise ValueError(
            "Environment returned non-finite state values."
        )


def normalize_action(action: Any) -> np.ndarray:

    action = np.asarray(
        action,
        dtype=np.float32,
    ).reshape(-1)

    if action.size != ACTION_DIM:

        raise ValueError(
            "SAC action dimension mismatch: "
            f"expected {ACTION_DIM}, "
            f"got {action.size}"
        )

    if not np.all(np.isfinite(action)):

        raise ValueError(
            "SAC produced non-finite action values."
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
# ENVIRONMENT
# ============================================================

@st.cache_resource(show_spinner=False)
def create_environment(seed: int = 42):

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
        # Trust
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

    env = IRFEnvironment(config)

    return env


# ============================================================
# SAC AGENT
# ============================================================

@st.cache_resource(show_spinner=False)
def create_agent():

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "SAC checkpoint not found:\n"
            f"{CHECKPOINT_PATH}\n\n"
            "Make sure the checkpoint is committed to GitHub "
            "and included in the Render deployment."
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
# GOVERNANCE
# ============================================================

@st.cache_resource(show_spinner=False)
def create_governance():

    return GovernanceEngine()


# ============================================================
# GOVERNANCE HELPERS
# ============================================================

def get_decision_field(
    decision: Any,
    name: str,
    default: Any = None,
) -> Any:
    """
    Read either object attributes or dictionary keys.
    """

    if decision is None:
        return default

    if isinstance(decision, dict):

        return decision.get(
            name,
            default,
        )

    return getattr(
        decision,
        name,
        default,
    )


def governance_status(decision: Any) -> str:

    status = get_decision_field(
        decision,
        "status",
        None,
    )

    if status is None:

        status = get_decision_field(
            decision,
            "decision",
            None,
        )

    if status is None:

        status = get_decision_field(
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
    """
    Extract governed action if the governance engine provides one.

    Otherwise preserve the SAC proposal.
    """

    action = get_decision_field(
        decision,
        "action",
        None,
    )

    if action is None:

        action = proposed_action

    return normalize_action(
        action
    )


# ============================================================
# TELEMETRY FROM STATE
# ============================================================

def telemetry_from_state(
    state: np.ndarray,
) -> dict[str, float]:

    validate_state(state)

    users = state.reshape(
        NUM_USERS,
        5,
    )

    # --------------------------------------------------------
    # Phase-4B.1 state:
    #
    # [SINR, interference, queue, power, trust]
    # --------------------------------------------------------

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

    if not isinstance(info, dict):
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

    # --------------------------------------------------------
    # Prefer explicit environment info
    # --------------------------------------------------------

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
# RUN ONE CLOSED-LOOP STEP
# ============================================================

def run_step(
    env: Any,
    agent: Any,
    governance: Any,
    state: np.ndarray,
):
    """
    Closed-loop execution:

        SAC
          ↓
        Governance
          ↓
        IRF
          ↓
        Telemetry
    """

    validate_state(state)

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
    # 2. Current telemetry
    # ========================================================

    state_telemetry = telemetry_from_state(
        state
    )

    current_trust = safe_float(
        getattr(
            env,
            "trust",
            state_telemetry["mean_trust"],
        ),
        state_telemetry["mean_trust"],
    )

    # ========================================================
    # 3. Governance
    # ========================================================

    decision = governance.evaluate(
        proposed_action,
        trust_score=current_trust,
        telemetry=state_telemetry,
    )

    status = governance_status(
        decision
    )

    # ========================================================
    # 4. Enforcement
    # ========================================================

    # --------------------------------------------------------
    # ALLOW
    # --------------------------------------------------------

    if status == "ALLOW":

        executed_action = governance_action(
            decision,
            proposed_action,
        )

        block_fallback = False
        execution_allowed = True

    # --------------------------------------------------------
    # CONSTRAIN
    #
    # Existing TA-FDRL-IRF runtime uses this state as a
    # governed safe-envelope intervention.
    # --------------------------------------------------------

    elif status == "CONSTRAIN":

        executed_action = governance_action(
            decision,
            proposed_action,
        )

        block_fallback = False
        execution_allowed = True

    # --------------------------------------------------------
    # BLOCK / REVIEW / REJECT
    #
    # These states must not directly execute the proposed
    # autonomous action.
    # --------------------------------------------------------

    else:

        executed_action = zero_action()

        block_fallback = True
        execution_allowed = False

    # ========================================================
    # 5. IRF execution
    # ========================================================

    result = env.step(
        executed_action
    )

    if not isinstance(result, tuple):

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
    # 6. Environment telemetry
    # ========================================================

    actual_telemetry = (
        get_environment_telemetry(
            env,
            info,
        )
    )

    # ========================================================
    # 7. Metrics
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

    trust_value = safe_float(
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
    # 8. Governance statistics
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

    modified_default = (
        max_action_delta > 1e-8
    )

    modified = safe_bool(
        get_decision_field(
            decision,
            "modified",
            modified_default,
        ),
        modified_default,
    )

    policy_allowed = safe_bool(
        get_decision_field(
            decision,
            "policy",
            execution_allowed,
        ),
        execution_allowed,
    )

    risk_score = safe_float(
        get_decision_field(
            decision,
            "risk",
            info.get(
                "risk",
                0.0,
            )
            if isinstance(info, dict)
            else 0.0,
        )
    )

    reason = str(
        get_decision_field(
            decision,
            "reason",
            "",
        )
        or ""
    )

    # ========================================================
    # 9. Polarization diagnostics
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

    return (
        next_state,
        {
            "step": 0,

            "reward":
                reward_value,

            "trust":
                trust_value,

            "risk":
                risk_score,

            "spectral_efficiency":
                spectral_efficiency,

            "energy_efficiency":
                energy_efficiency,

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

            "governance":
                status,

            "reason":
                reason,

            "policy_allowed":
                policy_allowed,

            "modified":
                modified,

            "block_fallback":
                block_fallback,

            "execution_allowed":
                execution_allowed,

            "max_action_delta":
                max_action_delta,

            "mean_action_delta":
                mean_action_delta,

            "pqi":
                pqi,

            "xpi":
                xpi,

            "terminated":
                safe_bool(
                    terminated
                ),

            "truncated":
                safe_bool(
                    truncated
                ),
        },
    )


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_SESSION = {
    "env": None,
    "agent": None,
    "governance": None,
    "state": None,
    "history": [],
    "running": False,
    "runtime_ready": False,
    "active_seed": None,
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
    f"""
    <div class="subtitle">
    Trust-Aware Adaptive Federated Deep Reinforcement Learning
    for Intelligent Radio Fabric in 6G Networks
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"{PHASE} • "
    "Live closed-loop research simulation • "
    "SAC → Governance → IRF → Telemetry"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Runtime Control"
    )

    seed_value = st.number_input(
        "Environment seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
    )

    auto_refresh = st.checkbox(
        "Live auto refresh",
        value=False,
    )

    refresh_interval = st.slider(
        "Refresh interval (seconds)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
    )

    st.divider()

    start_reset = st.button(
        "▶️ Start / Reset",
        width="stretch",
    )

    one_step = st.button(
        "⏭️ Run One Step",
        width="stretch",
    )

    full_episode = st.button(
        "🚀 Run Full Episode",
        width="stretch",
    )

    clear_history = st.button(
        "🧹 Clear History",
        width="stretch",
    )

    st.divider()

    st.caption(
        f"Phase: {PHASE}"
    )

    st.caption(
        f"State: {STATE_DIM}"
    )

    st.caption(
        f"Action: {ACTION_DIM}"
    )


# ============================================================
# LOAD RUNTIME
# ============================================================

try:

    if st.session_state.agent is None:

        st.session_state.agent = (
            create_agent()
        )

    if st.session_state.governance is None:

        st.session_state.governance = (
            create_governance()
        )

    # --------------------------------------------------------
    # Environment is recreated when the selected seed changes.
    # --------------------------------------------------------

    if (
        st.session_state.env is None
        or st.session_state.active_seed
        != int(seed_value)
    ):

        st.session_state.env = (
            create_environment(
                int(seed_value)
            )
        )

        reset_result = (
            st.session_state.env.reset(
                seed=int(seed_value)
            )
        )

        st.session_state.state = (
            extract_state(
                reset_result
            )
        )

        st.session_state.history = []

        st.session_state.active_seed = (
            int(seed_value)
        )

    elif st.session_state.state is None:

        reset_result = (
            st.session_state.env.reset(
                seed=int(seed_value)
            )
        )

        st.session_state.state = (
            extract_state(
                reset_result
            )
        )

    st.session_state.runtime_ready = True

except Exception as exc:

    st.session_state.runtime_ready = False

    st.error(
        "❌ Runtime initialization failed."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# RESET
# ============================================================

if start_reset:

    st.session_state.env = (
        create_environment(
            int(seed_value)
        )
    )

    reset_result = (
        st.session_state.env.reset(
            seed=int(seed_value)
        )
    )

    st.session_state.state = (
        extract_state(
            reset_result
        )
    )

    st.session_state.history = []

    st.session_state.running = True

    st.session_state.active_seed = (
        int(seed_value)
    )

    st.rerun()


# ============================================================
# CLEAR HISTORY
# ============================================================

if clear_history:

    st.session_state.history = []

    st.rerun()


# ============================================================
# RUNTIME STATUS
# ============================================================

if st.session_state.runtime_ready:

    st.markdown(
        """
        <div class="live-box">
            <span class="status-live">
                🟢 LIVE RUNTIME READY
            </span><br>
            SAC → GOVERNANCE → IRF → TELEMETRY
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

    col1.metric(
        "Reward",
        "—",
    )

    col2.metric(
        "Trust",
        "—",
    )

    col3.metric(
        "Risk",
        "—",
    )

    col4.metric(
        "Spectral Efficiency",
        "—",
    )

    col5.metric(
        "Energy Efficiency",
        "—",
    )


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

        st.exception(exc)


# ============================================================
# FULL EPISODE
# ============================================================

if full_episode:

    try:

        st.session_state.env = (
            create_environment(
                int(seed_value)
            )
        )

        reset_result = (
            st.session_state.env.reset(
                seed=int(seed_value)
            )
        )

        state = extract_state(
            reset_result
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

        st.rerun()

    except Exception as exc:

        st.error(
            "Full episode execution failed."
        )

        st.exception(exc)


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
        "Trust",
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
        f"**BLOCK/Non-execution Fallback:** "
        f"{latest['block_fallback']}"
    )

else:

    st.info(
        "No governance decision yet. "
        "Run one step or a full episode."
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

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Performance",
            "Trust & Risk",
            "Network",
            "Polarization",
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
                "through the current environment info "
                "during this runtime session."
            )

        st.caption(
            "Phase-4B.1: polarization diagnostics are "
            "observational. PQI/XPI are not included in "
            "the SAC state because polarization_state_enabled "
            "is disabled."
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
        df["block_fallback"].sum()
    )

    max_delta = safe_float(
        df["max_action_delta"].max()
    )

    mean_delta = safe_float(
        df["mean_action_delta"].mean()
    )

    e1, e2, e3, e4 = st.columns(4)

    e1.metric(
        "Modified Decisions",
        modified_count,
    )

    e2.metric(
        "Non-execution Fallbacks",
        fallback_count,
    )

    e3.metric(
        "Maximum Action Δ",
        f"{max_delta:.4f}",
    )

    e4.metric(
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

else:

    st.info(
        "No action enforcement data yet."
    )


# ============================================================
# POLICY VIOLATIONS
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

    if len(interventions) == 0:

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
            "block_fallback",
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
        "trust",
        "risk",
        "governance",
        "reason",
        "modified",
        "block_fallback",
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
            "ta_fdrl_irf_live_telemetry.csv"
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
    "PQI/XPI are diagnostic variables in Phase-4B.1. "
    "They are not part of the SAC observation state. "
    "Therefore this dashboard must not claim that the "
    "SAC policy explicitly learned polarization-aware "
    "control. Phase-4B.2 requires a 140-dimensional state "
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
                Governance Engine
                         |
             +-----------+-----------+
             |           |           |
           ALLOW      CONSTRAIN   BLOCK/REVIEW
             |           |           |
             |      Safe Action      |
             |           |           |
             +-----------+-----------+
                         |
                         v
                  Executed Action
                         |
                         v
                  IRF Environment
                         |
                         v
              Polarization-aware PHY
                         |
              +----------+----------+
              |          |          |
             SE         EE        Trust
              |          |          |
              +----------+----------+
                         |
                         v
                     Telemetry
                         |
                         +------> Governance
                         |
                         +------> Dashboard
                         |
                         +------> SAC
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

    "Environment":
        "Live Simulation",
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

checkpoint_relative = (
    CHECKPOINT_PATH.relative_to(
        PROJECT_ROOT
    )
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
    auto_refresh
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

            st.exception(exc)

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
    "Research Demonstration"
)