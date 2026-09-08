# =========================================================
# test_environment.py
#
# TA-FDRL-IRF
#
# Phase-4B.1:
# Polarization-Aware PHY Environment Validation
#
# Purpose:
#   Validate the environment before SAC retraining.
#
# Phase-4B.1 design:
#   State dimension  = 100
#   Action dimension = 40
#   Polarization PHY = ENABLED
#   PQI/XPI          = diagnostics only
#   Polarization state features = DISABLED
#   RIS optimization = DISABLED
#   Adaptive trust   = ENABLED
# =========================================================

import numpy as np

from environment.irf_env import (
    IRFConfig,
    IRFEnvironment,
)


# =========================================================
# TEST CONSTANTS
# =========================================================

SEED = 42
NUM_STEPS = 10

EXPECTED_STATE_DIM = 100
EXPECTED_ACTION_DIM = 40

NUM_USERS = 20
NUM_RIS_ELEMENTS = 64

PQI_XPI_TOLERANCE = 1e-5
STATE_SYNC_TOLERANCE = 1e-5
REPRODUCIBILITY_TOLERANCE = 1e-8


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def check_finite(name, value):
    """
    Check that a scalar or array contains no NaN / Inf.
    """

    array = np.asarray(value)

    assert np.all(np.isfinite(array)), (
        f"{name} contains NaN or Inf."
    )


def extract_reset_state(reset_result):
    """
    Support both:

        state

    and:

        (state, info)
    """

    if isinstance(reset_result, tuple):
        return np.asarray(reset_result[0])

    return np.asarray(reset_result)


def expected_interference_state(interference):
    """
    Convert the environment's raw interference vector
    into the normalized representation used by _get_state().

    Environment representation:

        raw interference
              |
              v
        divide by max(interference)
              |
              v
        clip to [0, 1]
              |
              v
        state[NUM_USERS:2*NUM_USERS]

    This is intentionally kept identical to the
    normalization policy used by IRFEnvironment._get_state().
    """

    raw_interference = np.asarray(
        interference,
        dtype=np.float64,
    )

    max_interference = max(
        float(np.max(raw_interference)),
        1e-12,
    )

    normalized_interference = np.clip(
        raw_interference / max_interference,
        0.0,
        1.0,
    )

    return normalized_interference.astype(
        np.float32
    )


def validate_interference_state_sync(
    state,
    env,
    step_label,
):
    """
    Validate synchronization between:

        env.interference

    and:

        state[NUM_USERS:2*NUM_USERS]

    Important:
        env.interference is RAW.
        State interference is NORMALIZED.

    Therefore the test compares the state representation
    against the expected normalized representation rather
    than comparing raw and normalized values directly.
    """

    state = np.asarray(
        state
    )

    raw_interference = np.asarray(
        env.interference,
        dtype=np.float64,
    )

    state_interference = state[
        NUM_USERS:
        2 * NUM_USERS
    ]

    expected_state = expected_interference_state(
        raw_interference
    )

    error = np.max(
        np.abs(
            state_interference.astype(np.float64)
            - expected_state.astype(np.float64)
        )
    )

    print(
        f"{step_label} interference sync error: "
        f"{error:.12e}"
    )

    check_finite(
        f"{step_label} state interference",
        state_interference,
    )

    check_finite(
        f"{step_label} raw interference",
        raw_interference,
    )

    check_finite(
        f"{step_label} expected interference state",
        expected_state,
    )

    assert (
        error < STATE_SYNC_TOLERANCE
    ), (
        f"{step_label}: interference state is stale "
        f"or normalization is inconsistent. "
        f"error={error:.12e}"
    )

    return error


# =========================================================
# ENVIRONMENT FACTORY
# =========================================================

def create_environment(seed=SEED):
    """
    Create a controlled Phase-4B.1 IRF environment.
    """

    config = IRFConfig(
        # -------------------------------------------------
        # System
        # -------------------------------------------------

        num_users=NUM_USERS,
        num_ris_elements=NUM_RIS_ELEMENTS,

        bandwidth_hz=100e6,
        carrier_frequency_hz=28e9,

        # -------------------------------------------------
        # Polarization PHY
        # -------------------------------------------------

        polarization_enabled=True,

        cross_polarization_factor=0.15,

        polarization_v_strength=1.0,
        polarization_h_strength=1.0,

        # -------------------------------------------------
        # Phase-4B.1 state policy
        #
        # PQI/XPI are diagnostics only.
        # They are NOT part of SAC state yet.
        # -------------------------------------------------

        polarization_state_enabled=False,

        # -------------------------------------------------
        # RIS
        # -------------------------------------------------

        optimize_ris=False,

        # -------------------------------------------------
        # Trust
        # -------------------------------------------------

        fixed_trust=False,

        trust_memory=0.90,
        trust_learning_rate=0.10,
        trust_target_rate_bps=1e7,

        # -------------------------------------------------
        # PHY power / noise
        # -------------------------------------------------

        max_power_w=1.0,
        circuit_power_w=0.1,

        noise_figure_db=7.0,
        noise_density_dbm_hz=-174.0,

        # -------------------------------------------------
        # Episode
        # -------------------------------------------------

        max_steps=200,

        # -------------------------------------------------
        # Reproducibility
        # -------------------------------------------------

        seed=seed,
    )

    env = IRFEnvironment(
        config
    )

    return env


# =========================================================
# MAIN TEST
# =========================================================

def main():

    print("=" * 78)

    print(
        "TA-FDRL-IRF | "
        "Phase-4B.1 Polarization-Aware PHY Environment Validation"
    )

    print("=" * 78)

    # =====================================================
    # 1. CONFIGURATION
    # =====================================================

    print()
    print("[1] Configuration")
    print("-" * 78)

    config = IRFConfig(
        num_users=NUM_USERS,
        num_ris_elements=NUM_RIS_ELEMENTS,

        bandwidth_hz=100e6,
        carrier_frequency_hz=28e9,

        polarization_enabled=True,

        cross_polarization_factor=0.15,

        polarization_v_strength=1.0,
        polarization_h_strength=1.0,

        # Phase-4B.1:
        # PQI/XPI are diagnostics only.
        polarization_state_enabled=False,

        optimize_ris=False,
        fixed_trust=False,

        trust_memory=0.90,
        trust_learning_rate=0.10,
        trust_target_rate_bps=1e7,

        max_power_w=1.0,
        circuit_power_w=0.1,

        noise_figure_db=7.0,
        noise_density_dbm_hz=-174.0,

        max_steps=200,

        seed=SEED,
    )

    print(
        f"Users: {config.num_users}"
    )

    print(
        f"RIS elements: {config.num_ris_elements}"
    )

    print(
        f"Bandwidth: "
        f"{config.bandwidth_hz / 1e6:.2f} MHz"
    )

    print(
        f"Carrier frequency: "
        f"{config.carrier_frequency_hz / 1e9:.2f} GHz"
    )

    print(
        f"RIS optimization: "
        f"{config.optimize_ris}"
    )

    print(
        f"Fixed trust: "
        f"{config.fixed_trust}"
    )

    print(
        f"Polarization enabled: "
        f"{config.polarization_enabled}"
    )

    print(
        f"Polarization state enabled: "
        f"{config.polarization_state_enabled}"
    )

    print(
        f"Cross-polarization factor: "
        f"{config.cross_polarization_factor}"
    )

    assert config.polarization_enabled is True

    assert config.polarization_state_enabled is False

    assert config.optimize_ris is False

    assert config.fixed_trust is False

    assert np.isclose(
        config.cross_polarization_factor,
        0.15,
    )

    print("PASS")

    # =====================================================
    # 2. ENVIRONMENT CREATION
    # =====================================================

    print()
    print("[2] Environment creation")
    print("-" * 78)

    env = IRFEnvironment(
        config
    )

    assert env is not None

    print(
        "Environment created successfully."
    )

    print("PASS")

    # =====================================================
    # 3. BASIC DIMENSIONS
    # =====================================================

    print()
    print("[3] Basic dimensions")
    print("-" * 78)

    print(
        f"State dimension: "
        f"{env.state_dim}"
    )

    print(
        f"Action dimension: "
        f"{env.action_dim}"
    )

    assert env.state_dim == EXPECTED_STATE_DIM, (
        f"Expected state_dim={EXPECTED_STATE_DIM}, "
        f"got {env.state_dim}"
    )

    assert env.action_dim == EXPECTED_ACTION_DIM, (
        f"Expected action_dim={EXPECTED_ACTION_DIM}, "
        f"got {env.action_dim}"
    )

    print("PASS")

    # =====================================================
    # 4. RESET
    # =====================================================

    print()
    print("[4] Environment reset")
    print("-" * 78)

    reset_result = env.reset(
        seed=SEED
    )

    state = extract_reset_state(
        reset_result
    )

    print(
        f"Initial state shape: "
        f"{state.shape}"
    )

    assert state.shape == (
        EXPECTED_STATE_DIM,
    ), (
        f"Expected state shape "
        f"({EXPECTED_STATE_DIM},), "
        f"got {state.shape}"
    )

    check_finite(
        "Initial state",
        state,
    )

    print("PASS")

    # =====================================================
    # 5. INITIAL TRUST
    # =====================================================

    print()
    print("[5] Initial trust validation")
    print("-" * 78)

    initial_mean_trust = float(
        np.mean(env.trust)
    )

    initial_min_trust = float(
        np.min(env.trust)
    )

    initial_max_trust = float(
        np.max(env.trust)
    )

    print(
        f"Initial mean trust: "
        f"{initial_mean_trust:.6f}"
    )

    print(
        f"Initial min trust: "
        f"{initial_min_trust:.6f}"
    )

    print(
        f"Initial max trust: "
        f"{initial_max_trust:.6f}"
    )

    check_finite(
        "Initial trust",
        env.trust,
    )

    assert np.all(
        env.trust >= 0.0
    )

    assert np.all(
        env.trust <= 1.0
    )

    print("PASS")

    # =====================================================
    # 6. POLARIZATION CHANNELS
    # =====================================================

    print()
    print("[6] Polarization channel validation")
    print("-" * 78)

    assert (
        env.direct_polarization_channel
        is not None
    )

    assert (
        env.bs_ris_polarization_channel
        is not None
    )

    assert (
        env.ris_user_polarization_channel
        is not None
    )

    direct_shape = (
        env.direct_polarization_channel.shape
    )

    bs_ris_shape = (
        env.bs_ris_polarization_channel.shape
    )

    ris_user_shape = (
        env.ris_user_polarization_channel.shape
    )

    print(
        f"Direct:   {direct_shape}"
    )

    print(
        f"BS-RIS:   {bs_ris_shape}"
    )

    print(
        f"RIS-user: {ris_user_shape}"
    )

    assert direct_shape == (
        NUM_USERS,
        2,
        2,
    )

    assert bs_ris_shape == (
        NUM_RIS_ELEMENTS,
        2,
        2,
    )

    assert ris_user_shape == (
        NUM_USERS,
        NUM_RIS_ELEMENTS,
        2,
        2,
    )

    check_finite(
        "Direct polarization channel",
        env.direct_polarization_channel,
    )

    check_finite(
        "BS-RIS polarization channel",
        env.bs_ris_polarization_channel,
    )

    check_finite(
        "RIS-user polarization channel",
        env.ris_user_polarization_channel,
    )

    print("PASS")

    # =====================================================
    # 7. EFFECTIVE POLARIZATION CHANNEL
    # =====================================================

    print()
    print("[7] Effective polarization channel")
    print("-" * 78)

    ris_phase = np.zeros(
        NUM_RIS_ELEMENTS,
        dtype=np.float64,
    )

    h_pol = (
        env._effective_polarization_channel(
            ris_phase
        )
    )

    print(
        f"Effective channel shape: "
        f"{h_pol.shape}"
    )

    assert h_pol.shape == (
        NUM_USERS,
        2,
        2,
    )

    check_finite(
        "Effective polarization channel",
        h_pol,
    )

    print("PASS")

    # =====================================================
    # 8. POLARIZATION CHANNEL GAIN
    # =====================================================

    print()
    print("[8] Polarization channel gain")
    print("-" * 78)

    channel_gain = (
        env._polarization_channel_gain(
            ris_phase
        )
    )

    print(
        f"Channel gain shape: "
        f"{channel_gain.shape}"
    )

    print(
        f"Mean channel gain: "
        f"{np.mean(channel_gain):.6e}"
    )

    print(
        f"Min channel gain: "
        f"{np.min(channel_gain):.6e}"
    )

    print(
        f"Max channel gain: "
        f"{np.max(channel_gain):.6e}"
    )

    assert channel_gain.shape == (
        NUM_USERS,
    )

    check_finite(
        "Polarization channel gain",
        channel_gain,
    )

    assert np.all(
        channel_gain >= 0.0
    )

    print("PASS")

    # =====================================================
    # 9. PQI / XPI
    # =====================================================

    print()
    print("[9] Polarization diagnostics: PQI / XPI")
    print("-" * 78)

    pqi, xpi = (
        env._polarization_diagnostics(
            ris_phase
        )
    )

    print(
        f"PQI shape: "
        f"{pqi.shape}"
    )

    print(
        f"XPI shape: "
        f"{xpi.shape}"
    )

    print(
        f"Mean PQI: "
        f"{np.mean(pqi):.6f}"
    )

    print(
        f"Mean XPI: "
        f"{np.mean(xpi):.6f}"
    )

    print(
        f"Min PQI: "
        f"{np.min(pqi):.6f}"
    )

    print(
        f"Max PQI: "
        f"{np.max(pqi):.6f}"
    )

    print(
        f"Min XPI: "
        f"{np.min(xpi):.6f}"
    )

    print(
        f"Max XPI: "
        f"{np.max(xpi):.6f}"
    )

    assert pqi.shape == (
        NUM_USERS,
    )

    assert xpi.shape == (
        NUM_USERS,
    )

    check_finite(
        "PQI",
        pqi,
    )

    check_finite(
        "XPI",
        xpi,
    )

    assert np.all(
        pqi >= 0.0
    )

    assert np.all(
        pqi <= 1.0
    )

    assert np.all(
        xpi >= 0.0
    )

    assert np.all(
        xpi <= 1.0
    )

    # PQI + XPI should represent
    # the complete two-polarization received power.
    polarization_sum_error = np.max(
        np.abs(
            pqi
            + xpi
            - 1.0
        )
    )

    print(
        f"Max |PQI + XPI - 1|: "
        f"{polarization_sum_error:.12e}"
    )

    assert (
        polarization_sum_error
        < PQI_XPI_TOLERANCE
    ), (
        "PQI + XPI is not approximately 1."
    )

    # Save diagnostics into environment.
    env.polarization_quality = pqi.copy()

    env.cross_polarization_ratio = xpi.copy()

    print("PASS")

    # =====================================================
    # 10. INTERFERENCE STATE SYNCHRONIZATION
    # =====================================================

    print()
    print("[10] Interference state synchronization")
    print("-" * 78)

    # Important:
    #
    # env.interference
    #     = raw internal interference
    #
    # state[20:40]
    #     = normalized interference representation
    #
    # Therefore we validate the transformation rather
    # than comparing raw and normalized values directly.

    interference_error = (
        validate_interference_state_sync(
            state=state,
            env=env,
            step_label="Initial",
        )
    )

    print(
        f"State/internal interference error: "
        f"{interference_error:.12e}"
    )

    print("PASS")

    # =====================================================
    # 11. ENVIRONMENT STEP TEST
    # =====================================================

    print()
    print("[11] Running environment steps")
    print("-" * 78)

    total_reward = 0.0

    initial_trust = float(
        np.mean(env.trust)
    )

    rng = np.random.default_rng(
        SEED + 5000
    )

    next_state = state

    for step in range(NUM_STEPS):

        # -------------------------------------------------
        # SAC-compatible action
        # -------------------------------------------------

        action = rng.uniform(
            -1.0,
            1.0,
            size=env.action_dim,
        ).astype(
            np.float32
        )

        result = env.step(
            action
        )

        # -------------------------------------------------
        # Gymnasium-style API
        # -------------------------------------------------

        assert len(result) == 5, (
            "Expected env.step() to return "
            "(state, reward, terminated, truncated, info)."
        )

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = result

        next_state = np.asarray(
            next_state
        )

        reward = float(
            reward
        )

        total_reward += reward

        # -------------------------------------------------
        # Required existing metrics
        # -------------------------------------------------

        spectral_efficiency = float(
            info["spectral_efficiency"]
        )

        energy_efficiency = float(
            info["energy_efficiency"]
        )

        trust = float(
            info["trust"]
        )

        trust_delta = float(
            info["trust_delta"]
        )

        queue = float(
            info["queue"]
        )

        # -------------------------------------------------
        # Polarization metrics
        # -------------------------------------------------

        mean_pqi = float(
            info["mean_polarization_quality"]
        )

        mean_xpi = float(
            info["mean_cross_polarization_ratio"]
        )

        # -------------------------------------------------
        # Output
        # -------------------------------------------------

        print(
            f"Step {step + 1:02d} | "
            f"Reward={reward: .4f} | "
            f"SE={spectral_efficiency:.4f} | "
            f"EE={energy_efficiency:.4e} | "
            f"Trust={trust:.4f} | "
            f"dTrust={trust_delta:.6f} | "
            f"Queue={queue:.4f} | "
            f"PQI={mean_pqi:.6f} | "
            f"XPI={mean_xpi:.6f}"
        )

        # -------------------------------------------------
        # State validation
        # -------------------------------------------------

        assert next_state.shape == (
            EXPECTED_STATE_DIM,
        )

        check_finite(
            f"Step {step + 1} state",
            next_state,
        )

        # -------------------------------------------------
        # Reward validation
        # -------------------------------------------------

        assert np.isfinite(
            reward
        )

        # -------------------------------------------------
        # Existing metric validation
        # -------------------------------------------------

        assert np.isfinite(
            spectral_efficiency
        )

        assert np.isfinite(
            energy_efficiency
        )

        assert np.isfinite(
            trust
        )

        assert np.isfinite(
            trust_delta
        )

        assert np.isfinite(
            queue
        )

        assert (
            0.0
            <= trust
            <= 1.0
        )

        # -------------------------------------------------
        # Polarization validation
        # -------------------------------------------------

        assert np.isfinite(
            mean_pqi
        )

        assert np.isfinite(
            mean_xpi
        )

        assert (
            0.0
            <= mean_pqi
            <= 1.0
        )

        assert (
            0.0
            <= mean_xpi
            <= 1.0
        )

        # -------------------------------------------------
        # PQI + XPI consistency
        # -------------------------------------------------

        assert abs(
            mean_pqi
            + mean_xpi
            - 1.0
        ) < PQI_XPI_TOLERANCE, (
            f"Step {step + 1}: "
            "PQI + XPI consistency failed."
        )

        # -------------------------------------------------
        # Interference synchronization
        # -------------------------------------------------

        validate_interference_state_sync(
            state=next_state,
            env=env,
            step_label=f"Step {step + 1}",
        )

        # -------------------------------------------------
        # Internal diagnostics validation
        # -------------------------------------------------

        check_finite(
            f"Step {step + 1} env.polarization_quality",
            env.polarization_quality,
        )

        check_finite(
            f"Step {step + 1} env.cross_polarization_ratio",
            env.cross_polarization_ratio,
        )

        # -------------------------------------------------
        # Internal state validation
        # -------------------------------------------------

        check_finite(
            f"Step {step + 1} env.interference",
            env.interference,
        )

        check_finite(
            f"Step {step + 1} env.power",
            env.power,
        )

        check_finite(
            f"Step {step + 1} env.trust",
            env.trust,
        )

        assert np.all(
            env.trust >= 0.0
        )

        assert np.all(
            env.trust <= 1.0
        )

        if terminated or truncated:
            print(
                f"Episode ended at step "
                f"{step + 1}."
            )
            break

    # =====================================================
    # 12. FINAL ENVIRONMENT STATE
    # =====================================================

    print()
    print("[12] Final environment state")
    print("-" * 78)

    final_trust = float(
        np.mean(env.trust)
    )

    trust_change = (
        final_trust
        - initial_trust
    )

    final_pqi = float(
        np.mean(
            env.polarization_quality
        )
    )

    final_xpi = float(
        np.mean(
            env.cross_polarization_ratio
        )
    )

    print(
        f"Final state shape: "
        f"{next_state.shape}"
    )

    print(
        f"Initial mean trust: "
        f"{initial_trust:.6f}"
    )

    print(
        f"Final mean trust: "
        f"{final_trust:.6f}"
    )

    print(
        f"Total trust change: "
        f"{trust_change:.6f}"
    )

    print(
        f"Final minimum trust: "
        f"{np.min(env.trust):.6f}"
    )

    print(
        f"Final maximum trust: "
        f"{np.max(env.trust):.6f}"
    )

    print(
        f"Final mean PQI: "
        f"{final_pqi:.6f}"
    )

    print(
        f"Final mean XPI: "
        f"{final_xpi:.6f}"
    )

    check_finite(
        "Final trust",
        env.trust,
    )

    check_finite(
        "Final PQI",
        env.polarization_quality,
    )

    check_finite(
        "Final XPI",
        env.cross_polarization_ratio,
    )

    check_finite(
        "Final interference",
        env.interference,
    )

    check_finite(
        "Final power",
        env.power,
    )

    assert (
        0.0
        <= final_pqi
        <= 1.0
    )

    assert (
        0.0
        <= final_xpi
        <= 1.0
    )

    # Final synchronization check.
    validate_interference_state_sync(
        state=next_state,
        env=env,
        step_label="Final",
    )

    print("PASS")

    # =====================================================
    # 13. SAME-SEED REPRODUCIBILITY
    # =====================================================

    print()
    print("[13] Same-seed reproducibility")
    print("-" * 78)

    env_a = create_environment(
        SEED
    )

    env_b = create_environment(
        SEED
    )

    state_a = extract_reset_state(
        env_a.reset(
            seed=SEED
        )
    )

    state_b = extract_reset_state(
        env_b.reset(
            seed=SEED
        )
    )

    reset_error = np.max(
        np.abs(
            state_a
            - state_b
        )
    )

    print(
        f"Reset state error: "
        f"{reset_error:.12e}"
    )

    assert (
        reset_error
        < REPRODUCIBILITY_TOLERANCE
    ), (
        "Same-seed reset is not reproducible."
    )

    # -----------------------------------------------------
    # Verify initial internal state reproducibility
    # -----------------------------------------------------

    interference_reset_error = np.max(
        np.abs(
            np.asarray(env_a.interference)
            - np.asarray(env_b.interference)
        )
    )

    trust_reset_error = np.max(
        np.abs(
            np.asarray(env_a.trust)
            - np.asarray(env_b.trust)
        )
    )

    print(
        f"Initial interference error: "
        f"{interference_reset_error:.12e}"
    )

    print(
        f"Initial trust error: "
        f"{trust_reset_error:.12e}"
    )

    assert (
        interference_reset_error
        < REPRODUCIBILITY_TOLERANCE
    ), (
        "Same-seed initial interference "
        "is not reproducible."
    )

    assert (
        trust_reset_error
        < REPRODUCIBILITY_TOLERANCE
    ), (
        "Same-seed initial trust "
        "is not reproducible."
    )

    # -----------------------------------------------------
    # Controlled actions
    # -----------------------------------------------------

    action_rng_a = np.random.default_rng(
        SEED + 7000
    )

    action_rng_b = np.random.default_rng(
        SEED + 7000
    )

    for reproducibility_step in range(5):

        action_a = action_rng_a.uniform(
            -1.0,
            1.0,
            size=EXPECTED_ACTION_DIM,
        ).astype(
            np.float32
        )

        action_b = action_rng_b.uniform(
            -1.0,
            1.0,
            size=EXPECTED_ACTION_DIM,
        ).astype(
            np.float32
        )

        action_error = np.max(
            np.abs(
                action_a
                - action_b
            )
        )

        assert (
            action_error
            < REPRODUCIBILITY_TOLERANCE
        ), (
            f"Controlled action generation failed "
            f"at step {reproducibility_step + 1}."
        )

        result_a = env_a.step(
            action_a
        )

        result_b = env_b.step(
            action_b
        )

        state_a_next = np.asarray(
            result_a[0]
        )

        state_b_next = np.asarray(
            result_b[0]
        )

        reward_a = float(
            result_a[1]
        )

        reward_b = float(
            result_b[1]
        )

        state_error = np.max(
            np.abs(
                state_a_next
                - state_b_next
            )
        )

        reward_error = abs(
            reward_a
            - reward_b
        )

        interference_error = np.max(
            np.abs(
                np.asarray(env_a.interference)
                - np.asarray(env_b.interference)
            )
        )

        trust_error = np.max(
            np.abs(
                np.asarray(env_a.trust)
                - np.asarray(env_b.trust)
            )
        )

        pqi_error = np.max(
            np.abs(
                np.asarray(
                    env_a.polarization_quality
                )
                -
                np.asarray(
                    env_b.polarization_quality
                )
            )
        )

        xpi_error = np.max(
            np.abs(
                np.asarray(
                    env_a.cross_polarization_ratio
                )
                -
                np.asarray(
                    env_b.cross_polarization_ratio
                )
            )
        )

        print(
            f"Step {reproducibility_step + 1}: "
            f"state_error={state_error:.12e}, "
            f"reward_error={reward_error:.12e}, "
            f"interference_error={interference_error:.12e}, "
            f"trust_error={trust_error:.12e}, "
            f"PQI_error={pqi_error:.12e}, "
            f"XPI_error={xpi_error:.12e}"
        )

        assert (
            state_error
            < REPRODUCIBILITY_TOLERANCE
        ), (
            f"State reproducibility failed "
            f"at step {reproducibility_step + 1}."
        )

        assert (
            reward_error
            < REPRODUCIBILITY_TOLERANCE
        ), (
            f"Reward reproducibility failed "
            f"at step {reproducibility_step + 1}."
        )

        assert (
            interference_error
            < REPRODUCIBILITY_TOLERANCE
        ), (
            f"Interference reproducibility failed "
            f"at step {reproducibility_step + 1}."
        )

        assert (
            trust_error
            < REPRODUCIBILITY_TOLERANCE
        ), (
            f"Trust reproducibility failed "
            f"at step {reproducibility_step + 1}."
        )

        assert (
            pqi_error
            < REPRODUCIBILITY_TOLERANCE
        ), (
            f"PQI reproducibility failed "
            f"at step {reproducibility_step + 1}."
        )

        assert (
            xpi_error
            < REPRODUCIBILITY_TOLERANCE
        ), (
            f"XPI reproducibility failed "
            f"at step {reproducibility_step + 1}."
        )

        # Verify both environments maintain their
        # state/internal interference synchronization.
        validate_interference_state_sync(
            state=state_a_next,
            env=env_a,
            step_label=(
                f"Reproducibility A "
                f"step {reproducibility_step + 1}"
            ),
        )

        validate_interference_state_sync(
            state=state_b_next,
            env=env_b,
            step_label=(
                f"Reproducibility B "
                f"step {reproducibility_step + 1}"
            ),
        )

    print("PASS")

    # =====================================================
    # FINAL REPORT
    # =====================================================

    print()
    print("=" * 78)

    print(
        "PHASE-4B.1 ENVIRONMENT VALIDATION PASS"
    )

    print("=" * 78)

    print()

    print(
        f"State dimension:             "
        f"{env.state_dim}"
    )

    print(
        f"Action dimension:            "
        f"{env.action_dim}"
    )

    print(
        f"Users:                       "
        f"{env.num_users}"
    )

    print(
        f"RIS elements:                "
        f"{config.num_ris_elements}"
    )

    print(
        f"Polarization PHY:            "
        f"{'ENABLED' if config.polarization_enabled else 'DISABLED'}"
    )

    print(
        f"Polarization state features: "
        f"{'ENABLED' if config.polarization_state_enabled else 'DISABLED'}"
    )

    print(
        f"Cross-polarization factor:   "
        f"{config.cross_polarization_factor}"
    )

    print(
        f"RIS optimization:            "
        f"{'ENABLED' if config.optimize_ris else 'DISABLED'}"
    )

    print(
        f"Adaptive trust:              "
        f"{'ENABLED' if not config.fixed_trust else 'DISABLED'}"
    )

    print()

    print(
        f"Total reward:                "
        f"{total_reward:.6f}"
    )

    print(
        f"Final mean trust:            "
        f"{final_trust:.6f}"
    )

    print(
        f"Final mean PQI:              "
        f"{final_pqi:.6f}"
    )

    print(
        f"Final mean XPI:              "
        f"{final_xpi:.6f}"
    )

    print()

    print(
        "Interference state sync:     PASS"
    )

    print(
        "PQI range validation:        PASS"
    )

    print(
        "XPI range validation:        PASS"
    )

    print(
        "PQI + XPI consistency:       PASS"
    )

    print(
        "NaN / Inf validation:        PASS"
    )

    print(
        "Internal state validation:   PASS"
    )

    print(
        "Same-seed reproducibility:   PASS"
    )

    print()

    print(
        "SAC architecture:            UNCHANGED"
    )

    print(
        "SAC training:                NOT PERFORMED"
    )

    print()

    print(
        "Phase-4B.1 environment "
        "validation complete."
    )

    print(
        "Next stage: controlled SAC "
        "benchmark / integration."
    )

    print()
    print("=" * 78)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
