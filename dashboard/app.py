
from pathlib import Path
import sys
import time

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

from environment.irf_env import IRFConfig, IRFEnvironment
from agents.sac_agent import SACAgent
from trust.governance import GovernanceEngine


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
# CONSTANTS
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "best_sac_irf_phase3_adaptive_trust"
    / "sac_phase3_adaptive_trust.pt"
)

NUM_USERS = 20
NUM_RIS_ELEMENTS = 64
STATE_DIM = 100
ACTION_DIM = 40
MAX_STEPS = 200


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
# DATAFRAME / ARROW SAFETY
# ============================================================

def arrow_safe_dataframe(
    dataframe,
    stringify_object_columns=False,
):
    """
    Return a Streamlit/PyArrow-compatible copy of a DataFrame.

    This protects the dashboard from mixed Python/numpy object
    types that can cause errors such as:

        pyarrow.lib.ArrowTypeError:
        Expected bytes, got a 'int' object

    Numeric columns remain numeric by default.

    Object columns can optionally be converted to strings for
    display-only tables.
    """

    df = dataframe.copy()

    for column in df.columns:

        series = df[column]

        # Convert numpy scalar/object values into ordinary Python
        # scalar values where possible.
        if series.dtype == "object":

            if stringify_object_columns:
                df[column] = series.map(
                    lambda value: ""
                    if value is None
                    else str(value)
                )

            else:
                # Try to preserve numeric columns if possible.
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

        elif pd.api.types.is_bool_dtype(series):
            df[column] = series.astype(bool)

        elif pd.api.types.is_numeric_dtype(series):
            # Normalize numpy numeric scalars to a regular numeric
            # pandas representation.
            df[column] = pd.to_numeric(
                series,
                errors="coerce",
            )

    return df


def safe_display_dataframe(
    dataframe,
    *,
    width="stretch",
    hide_index=False,
    stringify_object_columns=False,
):
    """
    Centralized safe wrapper around st.dataframe().
    """

    safe_df = arrow_safe_dataframe(
        dataframe,
        stringify_object_columns=stringify_object_columns,
    )

    st.dataframe(
        safe_df,
        width=width,
        hide_index=hide_index,
    )


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):
    """
    Convert scalar-like values safely to finite float.
    """

    try:

        arr = np.asarray(value)

        if arr.size == 0:
            return float(default)

        result = float(
            arr.reshape(-1)[0]
        )

        if not np.isfinite(result):
            return float(default)

        return result

    except Exception:

        return float(default)


def safe_mean(value, default=0.0):
    """
    Convert scalar/array-like values to finite mean.
    """

    try:

        arr = np.asarray(
            value,
            dtype=float,
        )

        if arr.size == 0:
            return float(default)

        arr = arr[
            np.isfinite(arr)
        ]

        if arr.size == 0:
            return float(default)

        return float(
            np.mean(arr)
        )

    except Exception:

        return float(default)


def extract_state(reset_result):
    """
    Supports environments where reset() returns either:

        state

    or:

        (state, info)
    """

    if isinstance(
        reset_result,
        tuple,
    ):

        return np.asarray(
            reset_result[0],
            dtype=np.float32,
        )

    return np.asarray(
        reset_result,
        dtype=np.float32,
    )


def normalize_action(action):
    """
    Ensure SAC action is a finite 1-D vector of ACTION_DIM.
    """

    action = np.asarray(
        action,
        dtype=np.float32,
    ).reshape(-1)

    if action.size != ACTION_DIM:

        raise ValueError(
            f"SAC action dimension mismatch: "
            f"expected {ACTION_DIM}, "
            f"got {action.size}"
        )

    if not np.all(
        np.isfinite(action)
    ):

        raise ValueError(
            "SAC produced non-finite action values."
        )

    return np.clip(
        action,
        -1.0,
        1.0,
    )


def zero_action():
    """
    Safe fallback for governance BLOCK.
    """

    return np.zeros(
        ACTION_DIM,
        dtype=np.float32,
    )


# ============================================================
# ENVIRONMENT
# ============================================================

@st.cache_resource
def create_environment(seed=42):

    config = IRFConfig(

        # ----------------------------------------------------
        # Network / PHY
        # ----------------------------------------------------

        num_users=NUM_USERS,
        num_ris_elements=NUM_RIS_ELEMENTS,
        bandwidth_hz=100e6,
        carrier_frequency_hz=28e9,

        # ----------------------------------------------------
        # Polarization-aware PHY
        # ----------------------------------------------------

        polarization_enabled=True,
        cross_polarization_factor=0.15,
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
        optimize_ris=False,
        fixed_trust=False,

        # ----------------------------------------------------
        # Trust model
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

    return IRFEnvironment(config)


# ============================================================
# SAC AGENT
# ============================================================

@st.cache_resource
def create_agent():

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"SAC checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
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

@st.cache_resource
def create_governance():

    return GovernanceEngine()


# ============================================================
# TELEMETRY FROM STATE
# ============================================================

def telemetry_from_state(state):

    state = np.asarray(
        state,
        dtype=np.float32,
    ).reshape(-1)

    if state.size != STATE_DIM:

        raise ValueError(
            f"State dimension mismatch: "
            f"expected {STATE_DIM}, "
            f"got {state.size}"
        )

    users = state.reshape(
        NUM_USERS,
        5,
    )

    # State layout:
    #
    # [SINR, interference, queue, power, trust]

    sinr = users[:, 0]
    interference = users[:, 1]
    queue = users[:, 2]
    power = users[:, 3]
    trust = users[:, 4]

    return {

        "mean_sinr": safe_mean(
            sinr
        ),

        "mean_interference": safe_mean(
            interference
        ),

        "mean_queue": safe_mean(
            queue
        ),

        "mean_power": safe_mean(
            power
        ),

        "mean_trust": safe_mean(
            trust
        ),

        # Normalized telemetry for governance risk.
        "queue": np.clip(
            safe_mean(queue),
            0.0,
            1.0,
        ),

        "interference": np.clip(
            safe_mean(interference),
            0.0,
            1.0,
        ),
    }


# ============================================================
# TELEMETRY FROM ENVIRONMENT
# ============================================================

def get_environment_telemetry(
    env,
    info=None,
):

    info = info or {}

    telemetry = {

        "mean_sinr": safe_mean(
            getattr(
                env,
                "sinr",
                0.0,
            )
        ),

        "mean_interference": safe_mean(
            getattr(
                env,
                "interference",
                0.0,
            )
        ),

        "mean_queue": safe_mean(
            getattr(
                env,
                "queue",
                0.0,
            )
        ),

        "mean_power": safe_mean(
            getattr(
                env,
                "power",
                0.0,
            )
        ),

        "mean_trust": safe_mean(
            getattr(
                env,
                "trust",
                0.0,
            )
        ),
    }

    # --------------------------------------------------------
    # Prefer explicit info values when available
    # --------------------------------------------------------

    if isinstance(
        info,
        dict,
    ):

        for key in (
            "spectral_efficiency",
            "energy_efficiency",
            "trust",
            "risk",
            "reward",
        ):

            if key in info:

                telemetry[key] = safe_float(
                    info[key],
                    telemetry.get(
                        key,
                        0.0,
                    ),
                )

    telemetry["queue"] = np.clip(
        telemetry["mean_queue"],
        0.0,
        1.0,
    )

    telemetry["interference"] = np.clip(
        telemetry["mean_interference"],
        0.0,
        1.0,
    )

    return telemetry


# ============================================================
# RUN ONE CLOSED-LOOP STEP
# ============================================================

def run_step(
    env,
    agent,
    governance,
    state,
):

    # --------------------------------------------------------
    # 1. SAC proposes action
    # --------------------------------------------------------

    proposed_action = agent.select_action(
        state,
        evaluate=True,
    )

    proposed_action = normalize_action(
        proposed_action
    )

    # --------------------------------------------------------
    # 2. Current state telemetry
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 3. Governance decision
    # --------------------------------------------------------

    decision = governance.evaluate(
        proposed_action,
        trust_score=current_trust,
        telemetry=state_telemetry,
    )

    # --------------------------------------------------------
    # 4. Enforcement
    # --------------------------------------------------------

    status = str(
        getattr(
            decision,
            "status",
            "UNKNOWN",
        )
    ).upper()

    if status == "BLOCK":

        executed_action = zero_action()

        block_fallback = True

    else:

        executed_action = normalize_action(
            getattr(
                decision,
                "action",
                proposed_action,
            )
        )

        block_fallback = False

    # --------------------------------------------------------
    # 5. Execute action in IRF
    # --------------------------------------------------------

    result = env.step(
        executed_action
    )

    if len(result) != 5:

        raise ValueError(
            "IRFEnvironment.step() must return "
            "(next_state, reward, terminated, truncated, info)."
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

    # --------------------------------------------------------
    # 6. Actual environment telemetry
    # --------------------------------------------------------

    actual_telemetry = get_environment_telemetry(
        env,
        info,
    )

    # --------------------------------------------------------
    # 7. Extract metrics
    # --------------------------------------------------------

    reward_value = safe_float(
        reward
    )

    spectral_efficiency = safe_float(
        info.get(
            "spectral_efficiency",
            0.0,
        )
        if isinstance(
            info,
            dict,
        )
        else 0.0
    )

    energy_efficiency = safe_float(
        info.get(
            "energy_efficiency",
            0.0,
        )
        if isinstance(
            info,
            dict,
        )
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
        if isinstance(
            info,
            dict,
        )
        else getattr(
            env,
            "trust",
            actual_telemetry[
                "mean_trust"
            ],
        )
    )

    # --------------------------------------------------------
    # 8. Action enforcement statistics
    # --------------------------------------------------------

    action_delta = np.abs(
        proposed_action
        - executed_action
    )

    max_action_delta = float(
        np.max(action_delta)
    )

    mean_action_delta = float(
        np.mean(action_delta)
    )

    modified = bool(
        getattr(
            decision,
            "modified",
            max_action_delta > 1e-8,
        )
    )

    policy_allowed = bool(
        getattr(
            decision,
            "policy",
            True,
        )
    )

    risk_score = safe_float(
        getattr(
            decision,
            "risk",
            info.get(
                "risk",
                0.0,
            )
            if isinstance(
                info,
                dict,
            )
            else 0.0,
        )
    )

    reason = str(
        getattr(
            decision,
            "reason",
            "",
        )
    )

    return (

        next_state,

        {
            "step": 0,

            "reward": reward_value,

            "trust": trust_value,

            "risk": risk_score,

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

            "governance": status,

            "reason": reason,

            "policy_allowed":
                policy_allowed,

            "modified":
                modified,

            "block_fallback":
                block_fallback,

            "max_action_delta":
                max_action_delta,

            "mean_action_delta":
                mean_action_delta,

            "terminated":
                bool(terminated),

            "truncated":
                bool(truncated),
        },
    )


# ============================================================
# SESSION STATE
# ============================================================

if "env" not in st.session_state:
    st.session_state.env = None

if "agent" not in st.session_state:
    st.session_state.agent = None

if "governance" not in st.session_state:
    st.session_state.governance = None

if "state" not in st.session_state:
    st.session_state.state = None

if "history" not in st.session_state:
    st.session_state.history = []

if "running" not in st.session_state:
    st.session_state.running = False

if "runtime_ready" not in st.session_state:
    st.session_state.runtime_ready = False


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
    "Live closed-loop research simulation: "
    "SAC → Governance → IRF → Telemetry → SAC"
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

    st.subheader(
        "Runtime"
    )

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

    if st.session_state.env is None:

        st.session_state.env = (
            create_environment(
                int(seed_value)
            )
        )

    if st.session_state.state is None:

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
        "Runtime initialization failed."
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
# SYSTEM METRICS
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)

history = st.session_state.history

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
# LIVE CONTROL EXECUTION
# ============================================================

if one_step:

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

    st.rerun()


# ============================================================
# FULL EPISODE
# ============================================================

if full_episode:

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

    st.rerun()


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

    elif decision_status == "BLOCK":

        st.error(
            f"BLOCK — "
            f"{latest['reason'] or 'Governance policy blocked action'}"
        )

    else:

        st.info(
            decision_status
        )

    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "Risk Level",
        (
            "HIGH"
            if latest["risk"] >= 0.8
            else "MEDIUM"
            if latest["risk"] >= 0.5
            else "LOW"
        ),
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
        f"**BLOCK Fallback:** "
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

    tab1, tab2, tab3 = st.tabs(
        [
            "Performance",
            "Trust & Risk",
            "Network",
        ]
    )

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

        performance_df = (
            arrow_safe_dataframe(
                performance_df
            )
        )

        st.line_chart(
            performance_df,
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

        ee_df = (
            arrow_safe_dataframe(
                ee_df
            )
        )

        st.line_chart(
            ee_df,
            width="stretch",
        )

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

        trust_risk_df = (
            arrow_safe_dataframe(
                trust_risk_df
            )
        )

        st.line_chart(
            trust_risk_df,
            width="stretch",
        )

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

        network_df = (
            arrow_safe_dataframe(
                network_df
            )
        )

        st.line_chart(
            network_df,
            width="stretch",
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

    g1, g2, g3, g4 = st.columns(4)

    g1.metric(
        "Total Decisions",
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

    max_delta = float(
        df["max_action_delta"].max()
    )

    e1, e2, e3 = st.columns(3)

    e1.metric(
        "Modified Decisions",
        modified_count,
    )

    e2.metric(
        "BLOCK Fallbacks",
        fallback_count,
    )

    e3.metric(
        "Maximum Action Δ",
        f"{max_delta:.4f}",
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

    enforcement_df = (
        arrow_safe_dataframe(
            enforcement_df
        )
    )

    st.line_chart(
        enforcement_df,
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
    "Policy Violations"
)

if history:

    df = pd.DataFrame(
        history
    )

    violations = df[
        df["policy_allowed"] == False
    ]

    if len(violations) == 0:

        st.success(
            "No policy violations detected."
        )

    else:

        st.warning(
            f"{len(violations)} "
            f"policy violation decision(s) detected."
        )

        violations_display = violations[
            [
                "step",
                "governance",
                "reason",
                "risk",
                "trust",
            ]
        ]

        safe_display_dataframe(
            violations_display,
            width="stretch",
            hide_index=False,
            stringify_object_columns=True,
        )

else:

    st.info(
        "Policy violation analysis will appear "
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
        width="stretch",
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
# H5 EVIDENCE
# ============================================================

st.header(
    "H5 Governance Evidence"
)

h5a, h5b = st.columns(2)

with h5a:

    st.subheader(
        "H5-A"
    )

    st.success(
        "Completed"
    )

    st.write(
        "Normal-operation governed SAC benchmark"
    )

with h5b:

    st.subheader(
        "H5-B"
    )

    st.success(
        "100%"
    )

    st.write(
        "Governance branch accuracy"
    )

c1, c2 = st.columns(2)

with c1:

    st.metric(
        "Controlled Decisions",
        "5,000",
    )

with c2:

    st.metric(
        "Operating Regions",
        "5",
    )

st.info(
    "H5 validates governance enforcement and "
    "controlled operating-region behavior. "
    "It does not by itself prove that governance "
    "improves physical-layer performance."
)

st.warning(
    "The observed SAC trust distribution in the "
    "normal-operation benchmark should be interpreted "
    "together with the current hard trust policy threshold. "
    "Any mismatch between displayed trust and governance "
    "decision trust must be resolved before using the "
    "runtime output as a scientific claim."
)


# ============================================================
# RUNTIME ARCHITECTURE
# ============================================================

st.header(
    "Runtime Architecture"
)

st.code(
    """
                    TrustOSAI
                        |
                        v
                 Trust / Risk Policy
                        |
                        v
                 SAC Controller
                        |
                        v
                 Proposed Action
                        |
                        v
                Governance Engine
                        |
            +-----------+-----------+
            |           |           |
          ALLOW      CONSTRAIN     BLOCK
            |           |           |
            |      Safe Envelope    |
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
                        v
                    Telemetry
                        |
        +---------------+---------------+
        |               |               |
       SE              EE             Trust
        |               |               |
        +---------------+---------------+
                        |
                        v
                 Governance Loop
                        |
                        +------> SAC
                        |
                        +------> Dashboard
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
        "TA-FDRL-IRF",

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
        "28 GHz",

    "Bandwidth":
        "100 MHz",

    "Polarization":
        "Enabled",

    "RIS Optimization":
        "Disabled",

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

# ------------------------------------------------------------
# IMPORTANT:
# The Value column intentionally contains mixed Python types
# such as integers and strings. Convert the display table to
# homogeneous string columns before Streamlit/PyArrow sees it.
# ------------------------------------------------------------

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
# CHECKPOINT
# ============================================================

st.header(
    "SAC Checkpoint"
)

st.code(
    str(
        CHECKPOINT_PATH.relative_to(
            PROJECT_ROOT
        )
    ),
    language="text",
)

if CHECKPOINT_PATH.exists():

    st.success(
        "Checkpoint available"
    )

else:

    st.error(
        "Checkpoint missing"
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

        st.rerun()

    else:

        st.session_state.running = False


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TA-FDRL-IRF • TrustOSAI Governance Control Plane "
    "• Research Demonstration • Live Simulation"
)

