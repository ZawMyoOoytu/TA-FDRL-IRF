"""
TA-FDRL-IRF
Live Research Dashboard
---------------------------------

Runtime dashboard with:

- SAC Agent
- Real IRF telemetry
- Auto Episode execution
- Episode history
- Runtime controls
- Polarization telemetry
- Trust telemetry
- Per-user telemetry
- Governance status
"""

from __future__ import annotations

import time
from typing import Any, List

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.runtime_manager import (
    MODE_REGISTRY,
    RuntimeManager,
)


# ============================================================
# Page
# ============================================================

st.set_page_config(
    page_title="TA-FDRL-IRF Research Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Helpers
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


def get_array(
    data: dict,
    key: str,
) -> np.ndarray:

    value = data.get(key)

    if value is None:
        return np.asarray([])

    try:

        return np.asarray(
            value,
            dtype=float,
        ).reshape(-1)

    except Exception:

        return np.asarray([])


def make_user_dataframe(
    telemetry: dict,
) -> pd.DataFrame:

    candidate_keys = [
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

    arrays = {}

    for key in candidate_keys:

        arr = get_array(
            telemetry,
            key,
        )

        if arr.size:
            arrays[key] = arr

    if not arrays:
        return pd.DataFrame()

    max_len = max(
        len(v)
        for v in arrays.values()
    )

    data = {}

    for key, arr in arrays.items():

        padded = np.full(
            max_len,
            np.nan,
            dtype=float,
        )

        padded[:len(arr)] = arr

        data[key] = padded

    df = pd.DataFrame(data)

    df.insert(
        0,
        "user",
        np.arange(
            1,
            len(df) + 1,
        ),
    )

    return df


def format_large(
    value: Any,
) -> str:

    x = safe_float(value)

    if abs(x) >= 1e9:
        return f"{x / 1e9:.3f}B"

    if abs(x) >= 1e6:
        return f"{x / 1e6:.3f}M"

    if abs(x) >= 1e3:
        return f"{x / 1e3:.3f}K"

    return f"{x:.4f}"


def runtime_speed_delay(
    label: str,
) -> float:

    mapping = {
        "0x — Maximum": 0.0,
        "0.25x — Slow": 0.20,
        "0.5x": 0.10,
        "1x — Normal": 0.05,
        "2x": 0.02,
        "4x — Fast": 0.0,
    }

    return mapping.get(
        label,
        0.0,
    )


# ============================================================
# Session state
# ============================================================

if "runtime" not in st.session_state:

    st.session_state.runtime = None


if "last_result" not in st.session_state:

    st.session_state.last_result = None


if "runtime_seed" not in st.session_state:

    st.session_state.runtime_seed = 42


# ------------------------------------------------------------
# Persistent Episode Summary
# ------------------------------------------------------------

if "episode_summary" not in st.session_state:

    st.session_state.episode_summary = None


# ------------------------------------------------------------
# Persistent Episode History
# ------------------------------------------------------------

if "episode_history" not in st.session_state:

    st.session_state.episode_history = []


runtime: RuntimeManager | None = (
    st.session_state.runtime
)


# ============================================================
# Header
# ============================================================

st.title(
    "📡 TA-FDRL-IRF — Live Research Dashboard"
)

st.caption(
    "Trust-Aware Adaptive Federated Deep Reinforcement "
    "Learning for Intelligent Radio Fabric in 6G Networks"
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("⚙️ Runtime Control")

    mode = st.selectbox(
        "Research Phase",
        options=list(
            MODE_REGISTRY.keys()
        ),
        index=list(
            MODE_REGISTRY.keys()
        ).index(
            "Phase-4B.2"
        )
        if "Phase-4B.2"
        in MODE_REGISTRY
        else 0,
    )

    cfg = MODE_REGISTRY[mode]

    st.markdown("---")

    st.subheader(
        "Research Configuration"
    )

    st.write(
        f"**State:** `{cfg.state_dim}D`"
    )

    st.write(
        f"**Action:** `{cfg.action_dim}D`"
    )

    st.write(
        f"**Users:** `{cfg.num_users}`"
    )

    st.write(
        f"**RIS Elements:** `{cfg.num_ris_elements}`"
    )

    st.write(
        f"**Max Steps:** `{cfg.max_steps}`"
    )

    st.write(
        f"**Bandwidth:** "
        f"`{cfg.bandwidth_hz / 1e6:.0f} MHz`"
    )

    st.write(
        f"**Carrier:** "
        f"`{cfg.carrier_frequency_hz / 1e9:.0f} GHz`"
    )

    st.write(
        f"**Polarization PHY:** "
        f"`{'ON' if cfg.polarization_enabled else 'OFF'}`"
    )

    st.write(
        f"**Polarization Observation:** "
        f"`{'ON' if cfg.polarization_state_enabled else 'OFF'}`"
    )

    st.write(
        f"**Adaptive Trust:** "
        f"`{'ON' if cfg.adaptive_trust else 'OFF'}`"
    )

    st.write(
        f"**RIS Optimization:** "
        f"`{'ON' if cfg.optimize_ris else 'OFF'}`"
    )

    st.markdown("---")

    st.subheader("Seed")

    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=999999,
        value=int(
            st.session_state.runtime_seed
        ),
        step=1,
    )

    st.session_state.runtime_seed = int(seed)


    # ========================================================
    # Initialize
    # ========================================================

    if st.button(
        "🔄 Initialize / Reload Runtime",
        use_container_width=True,
    ):

        try:

            st.session_state.runtime = (
                RuntimeManager(
                    mode=mode,
                    seed=int(seed),
                )
            )

            st.session_state.last_result = None

            st.session_state.episode_summary = None

            st.session_state.episode_history = []

            st.rerun()

        except Exception as exc:

            st.error(
                f"Runtime initialization failed:\n\n{exc}"
            )


    runtime = st.session_state.runtime


    # ========================================================
    # Runtime controls
    # ========================================================

    if runtime is not None:

        st.markdown("---")

        st.subheader(
            "AI Agent Runtime"
        )

        snap = runtime.snapshot()

        sac_status = (
            "AVAILABLE"
            if snap["sac_available"]
            else "UNAVAILABLE"
        )

        gov_status = (
            "AVAILABLE"
            if snap["governance_available"]
            else "BYPASS"
        )

        checkpoint_status = (
            "LOADED"
            if snap["checkpoint_loaded"]
            else "NOT LOADED"
        )

        st.write(
            f"**Agent:** SAC — `{sac_status}`"
        )

        st.write(
            f"**Governance:** `{gov_status}`"
        )

        st.write(
            f"**Checkpoint:** `{checkpoint_status}`"
        )


        # ====================================================
        # Auto Episode
        # ====================================================

        auto_episode = st.toggle(
            "🤖 Auto Episode",
            value=True,
            help=(
                "When enabled, completed episodes are "
                "automatically reset and the next episode starts."
            ),
        )

        runtime.set_continuous(
            auto_episode
        )


        # ====================================================
        # Speed
        # ====================================================

        speed_label = st.selectbox(
            "Runtime Speed",
            [
                "0x — Maximum",
                "0.25x — Slow",
                "0.5x",
                "1x — Normal",
                "2x",
                "4x — Fast",
            ],
            index=3,
        )

        speed_delay = runtime_speed_delay(
            speed_label
        )


        # ====================================================
        # Run steps
        # ====================================================

        run_steps_count = st.number_input(
            "Steps per Run",
            min_value=1,
            max_value=5000,
            value=20,
            step=1,
        )

        if st.button(
            "▶ RUN",
            type="primary",
            use_container_width=True,
        ):

            try:

                result = runtime.run_steps(
                    num_steps=int(
                        run_steps_count
                    ),
                    speed=speed_delay,
                    continuous=auto_episode,
                )

                st.session_state.last_result = (
                    result
                )

                # ------------------------------------------------
                # Capture episode information if a completed
                # episode was produced during RUN.
                # ------------------------------------------------

                if isinstance(
                    result,
                    dict,
                ):

                    summary = result.get(
                        "episode_summary"
                    )

                    if isinstance(
                        summary,
                        dict,
                    ):

                        st.session_state.episode_summary = (
                            summary
                        )

                    history = result.get(
                        "episode_history"
                    )

                    if isinstance(
                        history,
                        list,
                    ):

                        st.session_state.episode_history = (
                            list(history)
                        )

                st.rerun()

            except Exception as exc:

                runtime.stop()

                st.error(
                    f"Runtime step failed:\n\n{exc}"
                )


        # ====================================================
        # Run one full episode
        # ====================================================

        if st.button(
            "▶ RUN FULL EPISODE",
            use_container_width=True,
        ):

            try:

                # Full single episode should always stop
                # after one completed episode.

                result = runtime.run_episode(
                    seed=None
                )

                st.session_state.last_result = (
                    result
                )

                # ------------------------------------------------
                # Persist summary
                # ------------------------------------------------

                if isinstance(
                    result,
                    dict,
                ):

                    summary = result.get(
                        "episode_summary"
                    )

                    if isinstance(
                        summary,
                        dict,
                    ):

                        st.session_state.episode_summary = (
                            summary
                        )

                    history = result.get(
                        "episode_history"
                    )

                    if isinstance(
                        history,
                        list,
                    ):

                        st.session_state.episode_history = (
                            list(history)
                        )

                # ------------------------------------------------
                # Runtime fallback
                # ------------------------------------------------

                history = (
                    runtime.get_episode_history()
                )

                if history:

                    st.session_state.episode_history = (
                        list(history)
                    )

                    if (
                        st.session_state.episode_summary
                        is None
                    ):

                        st.session_state.episode_summary = (
                            history[-1]
                        )

                st.rerun()

            except Exception as exc:

                runtime.stop()

                st.error(
                    f"Episode failed:\n\n{exc}"
                )


        # ====================================================
        # Run N episodes
        # ====================================================

        episodes_to_run = st.number_input(
            "Auto Episodes",
            min_value=1,
            max_value=1000,
            value=5,
            step=1,
        )

        if st.button(
            "🤖 RUN AUTO EPISODES",
            use_container_width=True,
        ):

            try:

                # ------------------------------------------------
                # Execute requested number of episodes.
                # RuntimeManager handles episode lifecycle.
                # ------------------------------------------------

                result = runtime.run_episodes(
                    num_episodes=int(
                        episodes_to_run
                    ),
                    speed=speed_delay,
                )


                # =================================================
                # Store complete execution result
                # =================================================

                st.session_state.last_result = (
                    result
                )


                # =================================================
                # Store result data
                # =================================================

                if isinstance(
                    result,
                    dict,
                ):

                    # ------------------------------------------------
                    # Latest completed episode summary
                    # ------------------------------------------------

                    summary = result.get(
                        "episode_summary"
                    )

                    if isinstance(
                        summary,
                        dict,
                    ):

                        st.session_state.episode_summary = (
                            summary
                        )


                    # ------------------------------------------------
                    # Complete episode history
                    # ------------------------------------------------

                    history = result.get(
                        "episode_history"
                    )

                    if isinstance(
                        history,
                        list,
                    ):

                        st.session_state.episode_history = (
                            list(history)
                        )


                # =================================================
                # Runtime fallback
                # =================================================

                runtime_history = (
                    runtime.get_episode_history()
                )

                if runtime_history:

                    st.session_state.episode_history = (
                        list(runtime_history)
                    )


                # =================================================
                # Final summary fallback
                # =================================================

                if (
                    st.session_state.get(
                        "episode_summary"
                    )
                    is None
                ):

                    history = (
                        st.session_state.get(
                            "episode_history",
                            [],
                        )
                    )

                    if history:

                        st.session_state.episode_summary = (
                            history[-1]
                        )


                # =================================================
                # Debug-safe result persistence
                # =================================================

                if (
                    st.session_state.get(
                        "episode_summary"
                    )
                    is None
                ):

                    if isinstance(
                        result,
                        dict,
                    ):

                        candidate = result.get(
                            "last_result"
                        )

                        if isinstance(
                            candidate,
                            dict,
                        ):

                            candidate_summary = (
                                candidate.get(
                                    "episode_summary"
                                )
                            )

                            if isinstance(
                                candidate_summary,
                                dict,
                            ):

                                st.session_state.episode_summary = (
                                    candidate_summary
                                )


                # ------------------------------------------------
                # Force Streamlit rerun so the UI renders
                # the completed episode result.
                # ------------------------------------------------

                st.rerun()

            except Exception as exc:

                runtime.stop()

                st.error(
                    f"Auto Episode execution failed:\n\n{exc}"
                )


        # ====================================================
        # Pause / Stop / Reset
        # ====================================================

        col_a, col_b = st.columns(2)

        with col_a:

            if st.button(
                "⏸ PAUSE",
                use_container_width=True,
            ):

                runtime.pause_runtime()

                st.rerun()


        with col_b:

            if st.button(
                "■ STOP",
                use_container_width=True,
            ):

                runtime.stop()

                st.rerun()


        # ====================================================
        # Reset Episode
        # ====================================================

        if st.button(
            "↻ RESET EPISODE",
            use_container_width=True,
        ):

            runtime.reset_episode(
                seed=int(seed)
            )

            st.session_state.last_result = None

            st.session_state.episode_summary = None

            # Keep episode history so completed episodes
            # remain visible after reset.
            #
            # RuntimeManager is the primary source.
            runtime_history = (
                runtime.get_episode_history()
            )

            if runtime_history:

                st.session_state.episode_history = (
                    list(runtime_history)
                )

            st.rerun()


# ============================================================
# No runtime
# ============================================================

if runtime is None:

    st.info(
        "Initialize the runtime from the sidebar."
    )

    st.markdown(
        """
### Recommended Phase

**Phase-4B.2**

- State: 140D
- Action: 40D
- Users: 20
- RIS Elements: 64
- Polarization PHY: ON
- Polarization Observation: ON
- Adaptive Trust: ON
- RIS Optimization: OFF
- Episode length: 200 steps
        """
    )

    st.stop()


# ============================================================
# Current snapshot
# ============================================================

snapshot = runtime.snapshot()

telemetry = snapshot.get(
    "telemetry",
    {},
)


# ============================================================
# Episode Summary Binding
# ============================================================

# ------------------------------------------------------------
# Primary Source
# ------------------------------------------------------------
# RuntimeManager snapshot.
# ------------------------------------------------------------

episode_summary = snapshot.get(
    "episode_summary"
)


# ------------------------------------------------------------
# Secondary Source
# ------------------------------------------------------------
# Persistent Streamlit session-state summary.
# ------------------------------------------------------------

if not isinstance(
    episode_summary,
    dict,
):

    episode_summary = (
        st.session_state.get(
            "episode_summary"
        )
    )


# ------------------------------------------------------------
# Tertiary Source
# ------------------------------------------------------------
# Latest complete execution result.
# ------------------------------------------------------------

if not isinstance(
    episode_summary,
    dict,
):

    last_result = (
        st.session_state.get(
            "last_result"
        )
    )

    if isinstance(
        last_result,
        dict,
    ):

        candidate = last_result.get(
            "episode_summary"
        )

        if isinstance(
            candidate,
            dict,
        ):

            episode_summary = (
                candidate
            )

            st.session_state.episode_summary = (
                candidate
            )


# ------------------------------------------------------------
# Fourth Source
# ------------------------------------------------------------
# Runtime episode history.
# ------------------------------------------------------------

if not isinstance(
    episode_summary,
    dict,
):

    history = runtime.get_episode_history()

    if history:

        latest_history = history[-1]

        if isinstance(
            latest_history,
            dict,
        ):

            episode_summary = (
                latest_history
            )

            st.session_state.episode_summary = (
                latest_history
            )


# ------------------------------------------------------------
# Fifth Source
# ------------------------------------------------------------
# Persistent Streamlit episode history.
# ------------------------------------------------------------

if not isinstance(
    episode_summary,
    dict,
):

    session_history = (
        st.session_state.get(
            "episode_history",
            [],
        )
    )

    if isinstance(
        session_history,
        list,
    ) and session_history:

        latest_history = (
            session_history[-1]
        )

        if isinstance(
            latest_history,
            dict,
        ):

            episode_summary = (
                latest_history
            )


# ------------------------------------------------------------
# Final Normalization
# ------------------------------------------------------------

if not isinstance(
    episode_summary,
    dict,
):

    episode_summary = None


# ============================================================
# Runtime status banner
# ============================================================

st.markdown("---")

st.subheader(
    "🎛️ RUNTIME / EPISODE STATUS"
)

status_col1, status_col2, status_col3, status_col4 = (
    st.columns(4)
)


with status_col1:

    if snapshot["running"]:

        st.success(
            "● RUNNING"
        )

    elif snapshot["paused"]:

        st.warning(
            "Ⅱ PAUSED"
        )

    else:

        st.info(
            "○ IDLE"
        )


with status_col2:

    st.metric(
        "Current Episode",
        snapshot["current_episode"],
    )


with status_col3:

    st.metric(
        "Step",
        f"{snapshot['step']} / {snapshot['max_steps']}",
    )


with status_col4:

    st.metric(
        "Completed Episodes",
        snapshot["completed_episodes"],
    )


# ============================================================
# Episode progress
# ============================================================

progress = (
    snapshot["step"]
    / max(
        1,
        snapshot["max_steps"],
    )
)

st.progress(
    min(
        1.0,
        max(
            0.0,
            progress,
        ),
    )
)


runtime_col1, runtime_col2, runtime_col3, runtime_col4 = (
    st.columns(4)
)


with runtime_col1:

    st.metric(
        "Episode Reward",
        f"{snapshot['episode_reward']:.4f}",
    )


with runtime_col2:

    mean_reward = (
        snapshot["episode_reward"]
        / snapshot["step"]
        if snapshot["step"] > 0
        else 0.0
    )

    st.metric(
        "Mean Reward",
        f"{mean_reward:.4f}",
    )


with runtime_col3:

    st.metric(
        "Total Steps",
        snapshot["total_steps"],
    )


with runtime_col4:

    mode_text = (
        "AUTO"
        if snapshot["continuous"]
        else "MANUAL"
    )

    st.metric(
        "Episode Mode",
        mode_text,
    )


# ============================================================
# Live telemetry
# ============================================================

st.markdown("---")

st.subheader(
    "📡 LIVE NETWORK TELEMETRY"
)


m1, m2, m3, m4, m5 = st.columns(5)


with m1:

    st.metric(
        "Spectral Efficiency",
        f"{safe_float(telemetry.get('spectral_efficiency')):.6f}",
    )


with m2:

    st.metric(
        "Energy Efficiency",
        format_large(
            telemetry.get(
                "energy_efficiency"
            )
        ),
    )


with m3:

    st.metric(
        "Mean Trust",
        f"{safe_float(telemetry.get('mean_trust')):.6f}",
    )


with m4:

    st.metric(
        "Mean SINR",
        f"{safe_float(telemetry.get('mean_sinr')):.6f}",
    )


with m5:

    st.metric(
        "Total Rate",
        format_large(
            telemetry.get(
                "total_rate"
            )
        ),
    )


m6, m7, m8, m9, m10 = st.columns(5)


with m6:

    st.metric(
        "Mean Queue",
        f"{safe_float(telemetry.get('mean_queue')):.6f}",
    )


with m7:

    st.metric(
        "Mean Power",
        f"{safe_float(telemetry.get('mean_power')):.6f}",
    )


with m8:

    st.metric(
        "Interference Ratio",
        f"{safe_float(telemetry.get('interference_ratio')):.6f}",
    )


with m9:

    st.metric(
        "Mean PQI",
        f"{safe_float(telemetry.get('mean_pqi')):.6f}",
    )


with m10:

    st.metric(
        "Mean XPI",
        f"{safe_float(telemetry.get('mean_xpi')):.6f}",
    )


# ============================================================
# Episode performance
# ============================================================

st.markdown("---")

st.subheader(
    "🏁 EPISODE PERFORMANCE"
)


if isinstance(
    episode_summary,
    dict,
):

    ep1, ep2, ep3, ep4, ep5 = st.columns(5)


    with ep1:

        st.metric(
            "Episode",
            episode_summary.get(
                "episode",
                "-",
            ),
        )


    with ep2:

        st.metric(
            "Episode Reward",
            f"{safe_float(episode_summary.get('episode_reward')):.4f}",
        )


    with ep3:

        st.metric(
            "Mean Reward",
            f"{safe_float(episode_summary.get('mean_reward')):.4f}",
        )


    with ep4:

        st.metric(
            "Final Trust",
            f"{safe_float(episode_summary.get('final_trust')):.6f}",
        )


    with ep5:

        st.metric(
            "Final SE",
            f"{safe_float(episode_summary.get('final_spectral_efficiency')):.6f}",
        )


    st.write(
        f"**Final EE:** "
        f"{format_large(episode_summary.get('final_energy_efficiency'))}"
    )


else:

    st.info(
        "No completed episode yet. "
        "Run the AI Agent to generate Episode data."
    )


# ============================================================
# Tabs
# ============================================================

tabs = st.tabs(
    [
        "Overview",
        "20 Users",
        "Performance",
        "Trust",
        "Polarization",
        "Governance",
        "State / Action",
        "Episode History",
        "Step History",
        "Raw Telemetry",
        "Configuration",
    ]
)


# ============================================================
# Overview
# ============================================================

with tabs[0]:

    st.subheader(
        "Overview"
    )

    a, b, c = st.columns(3)


    with a:

        st.write(
            f"**Mode:** `{snapshot['mode']}`"
        )

        st.write(
            f"**State Dimension:** `{snapshot['state_dim']}`"
        )

        st.write(
            f"**Action Dimension:** `{snapshot['action_dim']}`"
        )


    with b:

        st.write(
            f"**Users:** `{snapshot['users']}`"
        )

        st.write(
            f"**RIS Elements:** `{snapshot['ris_elements']}`"
        )

        st.write(
            f"**Max Steps:** `{snapshot['max_steps']}`"
        )


    with c:

        st.write(
            f"**SAC:** "
            f"`{'AVAILABLE' if snapshot['sac_available'] else 'UNAVAILABLE'}`"
        )

        st.write(
            f"**Checkpoint:** "
            f"`{'LOADED' if snapshot['checkpoint_loaded'] else 'NOT LOADED'}`"
        )

        st.write(
            f"**Governance:** "
            f"`{'AVAILABLE' if snapshot['governance_available'] else 'BYPASS'}`"
        )


# ============================================================
# 20 Users
# ============================================================

with tabs[1]:

    st.subheader(
        "👥 Per-User Live Telemetry"
    )

    user_df = make_user_dataframe(
        telemetry
    )

    if user_df.empty:

        st.warning(
            "Per-user telemetry is not available "
            "from the current environment."
        )

    else:

        st.dataframe(
            user_df,
            use_container_width=True,
            height=520,
        )


# ============================================================
# Performance
# ============================================================

with tabs[2]:

    st.subheader(
        "📈 Network Performance"
    )

    performance_data = {

        "Metric": [
            "Spectral Efficiency",
            "Energy Efficiency",
            "Total Rate",
            "Mean SINR",
            "Mean Queue",
            "Mean Power",
            "Radiated Power",
            "Total Power",
            "Interference Ratio",
        ],

        "Actual Value": [

            safe_float(
                telemetry.get(
                    "spectral_efficiency"
                )
            ),

            safe_float(
                telemetry.get(
                    "energy_efficiency"
                )
            ),

            safe_float(
                telemetry.get(
                    "total_rate"
                )
            ),

            safe_float(
                telemetry.get(
                    "mean_sinr"
                )
            ),

            safe_float(
                telemetry.get(
                    "mean_queue"
                )
            ),

            safe_float(
                telemetry.get(
                    "mean_power"
                )
            ),

            safe_float(
                telemetry.get(
                    "radiated_power"
                )
            ),

            safe_float(
                telemetry.get(
                    "total_power"
                )
            ),

            safe_float(
                telemetry.get(
                    "interference_ratio"
                )
            ),
        ],
    }

    st.dataframe(
        pd.DataFrame(
            performance_data
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# Trust
# ============================================================

with tabs[3]:

    st.subheader(
        "🛡️ Trust Telemetry"
    )

    trust_delta_array = get_array(
        telemetry,
        "trust_delta",
    )

    trust_df = pd.DataFrame(
        {
            "Metric": [
                "Mean Trust",
                "Mean Behavior Score",
                "Mean Queue",
                "Trust Delta Mean",
            ],
            "Value": [
                safe_float(
                    telemetry.get(
                        "mean_trust"
                    )
                ),
                safe_float(
                    telemetry.get(
                        "mean_behavior_score"
                    )
                ),
                safe_float(
                    telemetry.get(
                        "mean_queue"
                    )
                ),
                float(
                    np.nanmean(
                        trust_delta_array
                    )
                )
                if trust_delta_array.size
                else 0.0,
            ],
        }
    )

    st.dataframe(
        trust_df,
        use_container_width=True,
        hide_index=True,
    )

    trust_array = get_array(
        telemetry,
        "trust",
    )

    if trust_array.size:

        chart_df = pd.DataFrame(
            {
                "User": np.arange(
                    1,
                    len(trust_array) + 1,
                ),
                "Trust": trust_array,
            }
        ).set_index("User")

        st.line_chart(
            chart_df
        )


# ============================================================
# Polarization
# ============================================================

with tabs[4]:

    st.subheader(
        "📡 Polarization-Aware PHY"
    )

    p1, p2, p3, p4 = st.columns(4)


    with p1:

        st.metric(
            "Polarization PHY",
            "ON"
            if snapshot[
                "polarization_enabled"
            ]
            else "OFF",
        )


    with p2:

        st.metric(
            "Polarization Observation",
            "ON"
            if snapshot[
                "polarization_state_enabled"
            ]
            else "OFF",
        )


    with p3:

        st.metric(
            "Mean PQI",
            f"{safe_float(telemetry.get('mean_pqi')):.6f}",
        )


    with p4:

        st.metric(
            "Mean XPI",
            f"{safe_float(telemetry.get('mean_xpi')):.6f}",
        )


    polarization_df = pd.DataFrame(
        {
            "Metric": [
                "Cross Polarization Factor",
                "Mean PQI",
                "Mean XPI",
                "Mean Polarization Quality",
                "Mean Cross Polarization Ratio",
                "Min PQI",
                "Max PQI",
                "Min XPI",
                "Max XPI",
            ],

            "Value": [

                safe_float(
                    telemetry.get(
                        "cross_polarization_factor"
                    )
                ),

                safe_float(
                    telemetry.get(
                        "mean_pqi"
                    )
                ),

                safe_float(
                    telemetry.get(
                        "mean_xpi"
                    )
                ),

                safe_float(
                    telemetry.get(
                        "mean_polarization_quality"
                    )
                ),

                safe_float(
                    telemetry.get(
                        "mean_cross_polarization_ratio"
                    )
                ),

                safe_float(
                    telemetry.get(
                        "min_pqi"
                    )
                ),

                safe_float(
                    telemetry.get(
                        "max_pqi"
                    )
                ),

                safe_float(
                    telemetry.get(
                        "min_xpi"
                    )
                ),

                safe_float(
                    telemetry.get(
                        "max_xpi"
                    )
                ),
            ],
        }
    )

    st.dataframe(
        polarization_df,
        use_container_width=True,
        hide_index=True,
    )


    pqi = get_array(
        telemetry,
        "pqi",
    )

    xpi = get_array(
        telemetry,
        "xpi",
    )


    if pqi.size:

        st.write(
            "Per-user PQI"
        )

        pqi_df = pd.DataFrame(
            {
                "User": np.arange(
                    1,
                    len(pqi) + 1,
                ),
                "PQI": pqi,
            }
        ).set_index("User")

        st.line_chart(
            pqi_df
        )


    if xpi.size:

        st.write(
            "Per-user Cross-Polarization Ratio"
        )

        xpi_df = pd.DataFrame(
            {
                "User": np.arange(
                    1,
                    len(xpi) + 1,
                ),
                "XPI": xpi,
            }
        ).set_index("User")

        st.line_chart(
            xpi_df
        )


# ============================================================
# Governance
# ============================================================

with tabs[5]:

    st.subheader(
        "⚖️ Governance"
    )

    if snapshot[
        "governance_available"
    ]:

        st.success(
            "GovernanceEngine AVAILABLE"
        )

        st.write(
            f"Source: `{snapshot['governance_source']}`"
        )

    else:

        st.warning(
            "GovernanceEngine BYPASS — "
            "runtime continues without governance enforcement."
        )

    last_result = (
        st.session_state.last_result
    )

    if last_result:

        governance = last_result.get(
            "governance",
            {},
        )

        st.write(
            f"**Last Decision:** "
            f"`{governance.get('status', 'UNKNOWN')}`"
        )

        st.json(
            governance
        )


# ============================================================
# State / Action
# ============================================================

with tabs[6]:

    st.subheader(
        "🧠 SAC State / Action"
    )

    state = runtime.get_state()

    action = runtime.get_action()

    s1, s2 = st.columns(2)


    with s1:

        st.write(
            "### State"
        )

        if state is not None:

            state_array = np.asarray(
                state
            ).reshape(-1)

            st.write(
                f"Dimension: `{len(state_array)}`"
            )

            st.dataframe(
                pd.DataFrame(
                    {
                        "Index": np.arange(
                            len(state_array)
                        ),
                        "Value": state_array,
                    }
                ),
                use_container_width=True,
                height=420,
            )


    with s2:

        st.write(
            "### Action"
        )

        if action is not None:

            action_array = np.asarray(
                action
            ).reshape(-1)

            st.write(
                f"Dimension: `{len(action_array)}`"
            )

            st.dataframe(
                pd.DataFrame(
                    {
                        "Index": np.arange(
                            len(action_array)
                        ),
                        "Value": action_array,
                    }
                ),
                use_container_width=True,
                height=420,
            )


# ============================================================
# Episode History
# ============================================================

with tabs[7]:

    st.subheader(
        "🏁 Episode History"
    )


    # --------------------------------------------------------
    # Primary source: RuntimeManager
    # --------------------------------------------------------

    episode_history = (
        runtime.get_episode_history()
    )


    # --------------------------------------------------------
    # Fallback: Streamlit session state
    # --------------------------------------------------------

    if not episode_history:

        episode_history = (
            st.session_state.get(
                "episode_history",
                [],
            )
        )


    if not episode_history:

        st.info(
            "No completed episodes yet."
        )

    else:

        ep_df = pd.DataFrame(
            episode_history
        )

        st.dataframe(
            ep_df,
            use_container_width=True,
            height=450,
        )


        if "episode_reward" in ep_df.columns:

            chart = ep_df[
                [
                    "episode",
                    "episode_reward",
                ]
            ].copy()

            chart = chart.set_index(
                "episode"
            )

            st.write(
                "Episode Reward"
            )

            st.line_chart(
                chart
            )


        if "final_trust" in ep_df.columns:

            trust_chart = ep_df[
                [
                    "episode",
                    "final_trust",
                ]
            ].copy()

            trust_chart = trust_chart.set_index(
                "episode"
            )

            st.write(
                "Final Trust per Episode"
            )

            st.line_chart(
                trust_chart
            )


        if (
            "final_spectral_efficiency"
            in ep_df.columns
        ):

            se_chart = ep_df[
                [
                    "episode",
                    "final_spectral_efficiency",
                ]
            ].copy()

            se_chart = se_chart.set_index(
                "episode"
            )

            st.write(
                "Final Spectral Efficiency"
            )

            st.line_chart(
                se_chart
            )


# ============================================================
# Step History
# ============================================================

with tabs[8]:

    st.subheader(
        "📊 Step History"
    )

    step_history = (
        runtime.get_step_history()
    )

    if not step_history:

        st.info(
            "No step history yet."
        )

    else:

        step_df = pd.DataFrame(
            step_history
        )

        st.dataframe(
            step_df.tail(1000),
            use_container_width=True,
            height=500,
        )


        if "reward" in step_df.columns:

            reward_chart = step_df[
                [
                    "episode",
                    "step",
                    "reward",
                ]
            ].copy()

            reward_chart["index"] = np.arange(
                len(reward_chart)
            )

            reward_chart = reward_chart.set_index(
                "index"
            )

            st.write(
                "Step Reward"
            )

            st.line_chart(
                reward_chart[
                    ["reward"]
                ]
            )


# ============================================================
# Raw Telemetry
# ============================================================

with tabs[9]:

    st.subheader(
        "🔬 Raw Environment Telemetry"
    )

    st.caption(
        "These values are read from the actual "
        "IRFEnvironment/runtime info."
    )

    st.json(
        telemetry
    )


# ============================================================
# Configuration
# ============================================================

with tabs[10]:

    st.subheader(
        "🔧 Runtime Configuration"
    )

    st.json(
        runtime.get_configuration()
    )


# ============================================================
# Footer
# ============================================================

st.markdown("---")

footer1, footer2, footer3 = st.columns(3)


with footer1:

    st.write(
        f"**Phase:** {snapshot['mode']}"
    )


with footer2:

    st.write(
        f"**Episode:** "
        f"{snapshot['current_episode']} "
        f"| **Step:** "
        f"{snapshot['step']}/{snapshot['max_steps']}"
    )


with footer3:

    st.write(
        f"**Completed:** "
        f"{snapshot['completed_episodes']} "
        f"| **Total Steps:** "
        f"{snapshot['total_steps']}"
    )