from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from environment.irf_env import (
    IRFConfig,
    IRFEnvironment,
)


# =========================================================
# PHASE-4B.2
# POLARIZATION OBSERVATION BENCHMARK
# =========================================================
#
# Controlled comparison:
#
# B2-A  BASELINE
#   Polarization PHY       = OFF
#   Polarization state     = OFF
#   State dimension        = 100
#
# B2-B  PHY ONLY
#   Polarization PHY       = ON
#   Polarization state     = OFF
#   State dimension        = 100
#
# B2-C  PHY + OBSERVATION
#   Polarization PHY       = ON
#   Polarization state     = ON
#   State dimension        = 140
#
# Action dimension remains 40 in all configurations.
#
# No SAC training is performed here.
# This is a controlled environment/observation benchmark.
#
# =========================================================


# =========================================================
# CONFIGURATION
# =========================================================

SEEDS = (
    42,
    123,
    2026,
    4096,
    7777,
    9999,
)

STEPS = 200

NUM_USERS = 20
NUM_RIS_ELEMENTS = 64

BANDWIDTH_HZ = 100e6
CARRIER_FREQUENCY_HZ = 28e9

CROSS_POLARIZATION_FACTOR = 0.15

EXPECTED_BASE_STATE_DIM = NUM_USERS * 5
EXPECTED_POLARIZATION_STATE_DIM = NUM_USERS * 7
EXPECTED_ACTION_DIM = NUM_USERS * 2


# =========================================================
# RESULT
# =========================================================

@dataclass
class BenchmarkResult:

    name: str
    seed: int

    state_dim: int
    action_dim: int

    total_reward: float
    mean_reward: float

    spectral_efficiency: float
    energy_efficiency: float

    mean_trust: float
    final_trust: float

    mean_pqi: float
    mean_xpi: float

    final_queue: float

    steps_completed: int


# =========================================================
# CONFIG FACTORY
# =========================================================

def make_config(
    *,
    seed: int,
    polarization_enabled: bool,
    polarization_state_enabled: bool,
) -> IRFConfig:
    """
    Create a controlled IRF configuration.

    Important:
        carrier_frequency_hz uses the canonical
        CARRIER_FREQUENCY_HZ constant.
    """

    return IRFConfig(
        # -------------------------------------------------
        # Network / PHY dimensions
        # -------------------------------------------------

        num_users=NUM_USERS,
        num_ris_elements=NUM_RIS_ELEMENTS,

        bandwidth_hz=BANDWIDTH_HZ,
        carrier_frequency_hz=CARRIER_FREQUENCY_HZ,

        # -------------------------------------------------
        # Polarization
        # -------------------------------------------------

        polarization_enabled=bool(
            polarization_enabled
        ),

        polarization_state_enabled=bool(
            polarization_state_enabled
        ),

        cross_polarization_factor=(
            CROSS_POLARIZATION_FACTOR
        ),

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

        max_steps=STEPS,

        # -------------------------------------------------
        # RIS / Trust
        # -------------------------------------------------

        optimize_ris=False,
        fixed_trust=False,

        trust_memory=0.90,
        trust_learning_rate=0.10,
        trust_target_rate_bps=1e7,

        # -------------------------------------------------
        # Reproducibility
        # -------------------------------------------------

        seed=int(seed),
    )


# =========================================================
# CONTROLLED ACTION GENERATOR
# =========================================================

def generate_actions(
    seed: int,
    steps: int,
    action_dim: int,
) -> np.ndarray:
    """
    Generate deterministic external actions.

    The same seed generates the same action sequence.

    This allows B2-A, B2-B and B2-C to receive
    the same action sequence for a paired comparison.
    """

    rng = np.random.default_rng(
        int(seed) + 5000
    )

    actions = rng.uniform(
        -1.0,
        1.0,
        size=(steps, action_dim),
    ).astype(np.float32)

    return actions


# =========================================================
# INFO HELPERS
# =========================================================

def get_info_float(
    info: Dict,
    key: str,
    default: float = 0.0,
) -> float:
    """
    Safely extract a numeric value from environment info.
    """

    value = info.get(
        key,
        default,
    )

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return float(default)


# =========================================================
# DIMENSION VALIDATION
# =========================================================

def validate_dimensions(
    env: IRFEnvironment,
    *,
    polarization_enabled: bool,
    polarization_state_enabled: bool,
) -> None:
    """
    Validate state/action dimensions against the intended
    Phase-4B.2 experimental design.
    """

    # -----------------------------------------------------
    # Action dimension
    # -----------------------------------------------------

    expected_action_dim = EXPECTED_ACTION_DIM

    if env.action_dim != expected_action_dim:
        raise AssertionError(
            "Unexpected action dimension: "
            f"expected={expected_action_dim}, "
            f"actual={env.action_dim}"
        )

    # -----------------------------------------------------
    # State dimension
    # -----------------------------------------------------

    if polarization_state_enabled:

        expected_state_dim = (
            EXPECTED_POLARIZATION_STATE_DIM
        )

    else:

        expected_state_dim = (
            EXPECTED_BASE_STATE_DIM
        )

    if env.state_dim != expected_state_dim:
        raise AssertionError(
            "Unexpected state dimension: "
            f"expected={expected_state_dim}, "
            f"actual={env.state_dim}, "
            f"polarization_enabled="
            f"{polarization_enabled}, "
            f"polarization_state_enabled="
            f"{polarization_state_enabled}"
        )


# =========================================================
# SINGLE RUN
# =========================================================

def run_configuration(
    *,
    name: str,
    seed: int,
    polarization_enabled: bool,
    polarization_state_enabled: bool,
) -> BenchmarkResult:
    """
    Run one controlled benchmark configuration.
    """

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------

    config = make_config(
        seed=seed,
        polarization_enabled=(
            polarization_enabled
        ),
        polarization_state_enabled=(
            polarization_state_enabled
        ),
    )

    # -----------------------------------------------------
    # Environment
    # -----------------------------------------------------

    env = IRFEnvironment(config)

    # -----------------------------------------------------
    # Dimension validation
    # -----------------------------------------------------

    validate_dimensions(
        env,
        polarization_enabled=(
            polarization_enabled
        ),
        polarization_state_enabled=(
            polarization_state_enabled
        ),
    )

    # -----------------------------------------------------
    # Reset
    # -----------------------------------------------------

    state = env.reset(
        seed=int(seed)
    )

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    if state.shape != (
        env.state_dim,
    ):
        raise AssertionError(
            "Initial state shape mismatch: "
            f"expected={(env.state_dim,)}, "
            f"actual={state.shape}"
        )

    # -----------------------------------------------------
    # Controlled external action sequence
    # -----------------------------------------------------

    actions = generate_actions(
        seed=seed,
        steps=STEPS,
        action_dim=env.action_dim,
    )

    # -----------------------------------------------------
    # Metric histories
    # -----------------------------------------------------

    rewards: List[float] = []

    se_values: List[float] = []
    ee_values: List[float] = []

    trust_values: List[float] = []

    pqi_values: List[float] = []
    xpi_values: List[float] = []

    # -----------------------------------------------------
    # Episode execution
    # -----------------------------------------------------

    steps_completed = 0

    for step_index, action in enumerate(
        actions,
        start=1,
    ):

        # -------------------------------------------------
        # Action validation
        # -------------------------------------------------

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        if action.shape != (
            env.action_dim,
        ):
            raise AssertionError(
                "Action shape mismatch at "
                f"step={step_index}: "
                f"expected={(env.action_dim,)}, "
                f"actual={action.shape}"
            )

        # -------------------------------------------------
        # Environment step
        # -------------------------------------------------

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        next_state = np.asarray(
            next_state,
            dtype=np.float32,
        )

        # -------------------------------------------------
        # State validation
        # -------------------------------------------------

        if next_state.shape != (
            env.state_dim,
        ):
            raise AssertionError(
                "Next-state shape mismatch at "
                f"step={step_index}: "
                f"expected={(env.state_dim,)}, "
                f"actual={next_state.shape}"
            )

        # -------------------------------------------------
        # Reward
        # -------------------------------------------------

        rewards.append(
            float(reward)
        )

        # -------------------------------------------------
        # Spectral Efficiency
        # -------------------------------------------------

        se_values.append(
            get_info_float(
                info,
                "spectral_efficiency",
            )
        )

        # -------------------------------------------------
        # Energy Efficiency
        # -------------------------------------------------

        ee_values.append(
            get_info_float(
                info,
                "energy_efficiency",
            )
        )

        # -------------------------------------------------
        # Trust
        # -------------------------------------------------

        trust_values.append(
            get_info_float(
                info,
                "trust",
            )
        )

        # -------------------------------------------------
        # Polarization Quality Index
        # -------------------------------------------------

        pqi_values.append(
            get_info_float(
                info,
                "mean_polarization_quality",
            )
        )

        # -------------------------------------------------
        # Cross Polarization Ratio
        # -------------------------------------------------

        xpi_values.append(
            get_info_float(
                info,
                "mean_cross_polarization_ratio",
            )
        )

        # -------------------------------------------------
        # Advance state
        # -------------------------------------------------

        state = next_state

        steps_completed += 1

        # -------------------------------------------------
        # Episode termination
        # -------------------------------------------------

        if terminated or truncated:
            break

    # =====================================================
    # Post-run validation
    # =====================================================

    if steps_completed != STEPS:
        raise AssertionError(
            "Episode terminated before requested "
            f"steps: requested={STEPS}, "
            f"completed={steps_completed}"
        )

    if not rewards:
        raise AssertionError(
            "No reward values were recorded."
        )

    if not se_values:
        raise AssertionError(
            "No spectral-efficiency values were recorded."
        )

    if not ee_values:
        raise AssertionError(
            "No energy-efficiency values were recorded."
        )

    if not trust_values:
        raise AssertionError(
            "No trust values were recorded."
        )

    # =====================================================
    # Result
    # =====================================================

    return BenchmarkResult(
        name=name,
        seed=int(seed),

        state_dim=int(
            env.state_dim
        ),

        action_dim=int(
            env.action_dim
        ),

        total_reward=float(
            np.sum(rewards)
        ),

        mean_reward=float(
            np.mean(rewards)
        ),

        spectral_efficiency=float(
            np.mean(se_values)
        ),

        energy_efficiency=float(
            np.mean(ee_values)
        ),

        mean_trust=float(
            np.mean(trust_values)
        ),

        final_trust=float(
            trust_values[-1]
        ),

        mean_pqi=float(
            np.mean(pqi_values)
        ),

        mean_xpi=float(
            np.mean(xpi_values)
        ),

        final_queue=float(
            env.mean_queue
        ),

        steps_completed=int(
            steps_completed
        ),
    )


# =========================================================
# ALL CONFIGURATIONS
# =========================================================

def run_benchmark() -> List[BenchmarkResult]:
    """
    Run the complete controlled Phase-4B.2 benchmark.

    For every seed:

        B2-A  Polarization OFF
        B2-B  Polarization ON, observation OFF
        B2-C  Polarization ON, observation ON
    """

    results: List[BenchmarkResult] = []

    for seed in SEEDS:

        # =================================================
        # B2-A
        # BASELINE
        # =================================================

        results.append(
            run_configuration(
                name="B2-A_BASELINE",
                seed=seed,

                polarization_enabled=False,

                polarization_state_enabled=False,
            )
        )

        # =================================================
        # B2-B
        # PHY ONLY
        # =================================================

        results.append(
            run_configuration(
                name="B2-B_PHY_ONLY",
                seed=seed,

                polarization_enabled=True,

                polarization_state_enabled=False,
            )
        )

        # =================================================
        # B2-C
        # PHY + OBSERVATION
        # =================================================

        results.append(
            run_configuration(
                name="B2-C_PHY_PLUS_OBSERVATION",
                seed=seed,

                polarization_enabled=True,

                polarization_state_enabled=True,
            )
        )

    return results


# =========================================================
# AGGREGATE REPORT
# =========================================================

def print_report(
    results: List[BenchmarkResult],
) -> None:
    """
    Print detailed benchmark results.
    """

    print()
    print("=" * 100)
    print(
        "PHASE-4B.2 POLARIZATION OBSERVATION BENCHMARK"
    )
    print("=" * 100)

    print()

    print(
        "Controlled configurations:"
    )

    print(
        "  B2-A = Polarization OFF, "
        "Observation OFF, State=100"
    )

    print(
        "  B2-B = Polarization ON, "
        "Observation OFF, State=100"
    )

    print(
        "  B2-C = Polarization ON, "
        "Observation ON, State=140"
    )

    print(
        "  Action dimension = 40 for all configurations"
    )

    print()

    print("-" * 100)

    for result in results:

        print(
            f"{result.name:30s} "
            f"seed={result.seed:<5d} "
            f"state={result.state_dim:<3d} "
            f"action={result.action_dim:<3d} "
            f"reward={result.total_reward:10.6f} "
            f"SE={result.spectral_efficiency:.6f} "
            f"EE={result.energy_efficiency:.3f} "
            f"trust={result.final_trust:.6f} "
            f"PQI={result.mean_pqi:.6f} "
            f"XPI={result.mean_xpi:.6f}"
        )

    print("-" * 100)


# =========================================================
# SUMMARY BY CONFIGURATION
# =========================================================

def summarize_configuration(
    results: List[BenchmarkResult],
    name: str,
) -> Dict[str, float]:
    """
    Aggregate results for one configuration.
    """

    selected = [
        result
        for result in results
        if result.name == name
    ]

    if not selected:
        raise ValueError(
            f"No results found for configuration: {name}"
        )

    return {
        "total_reward_mean": float(
            np.mean(
                [
                    r.total_reward
                    for r in selected
                ]
            )
        ),

        "mean_reward_mean": float(
            np.mean(
                [
                    r.mean_reward
                    for r in selected
                ]
            )
        ),

        "spectral_efficiency_mean": float(
            np.mean(
                [
                    r.spectral_efficiency
                    for r in selected
                ]
            )
        ),

        "energy_efficiency_mean": float(
            np.mean(
                [
                    r.energy_efficiency
                    for r in selected
                ]
            )
        ),

        "mean_trust_mean": float(
            np.mean(
                [
                    r.mean_trust
                    for r in selected
                ]
            )
        ),

        "final_trust_mean": float(
            np.mean(
                [
                    r.final_trust
                    for r in selected
                ]
            )
        ),

        "mean_pqi": float(
            np.mean(
                [
                    r.mean_pqi
                    for r in selected
                ]
            )
        ),

        "mean_xpi": float(
            np.mean(
                [
                    r.mean_xpi
                    for r in selected
                ]
            )
        ),

        "final_queue_mean": float(
            np.mean(
                [
                    r.final_queue
                    for r in selected
                ]
            )
        ),
    }


# =========================================================
# PERCENTAGE CHANGE
# =========================================================

def percentage_change(
    baseline: float,
    comparison: float,
) -> float:
    """
    Compute percentage change from baseline to comparison.
    """

    if abs(baseline) < 1e-12:
        return float("nan")

    return float(
        (
            (comparison - baseline)
            / baseline
        )
        * 100.0
    )


# =========================================================
# SCIENTIFIC SUMMARY
# =========================================================

def print_scientific_summary(
    results: List[BenchmarkResult],
) -> None:
    """
    Print controlled aggregate comparisons.

    Comparisons:

        B2-B vs B2-A
        B2-C vs B2-B
        B2-C vs B2-A
    """

    baseline = summarize_configuration(
        results,
        "B2-A_BASELINE",
    )

    phy_only = summarize_configuration(
        results,
        "B2-B_PHY_ONLY",
    )

    phy_observation = summarize_configuration(
        results,
        "B2-C_PHY_PLUS_OBSERVATION",
    )

    print()
    print("=" * 100)
    print(
        "PHASE-4B.2 SCIENTIFIC SUMMARY"
    )
    print("=" * 100)

    print()

    print(
        "Metric                         "
        "B2-A             "
        "B2-B             "
        "B2-C"
    )

    print("-" * 100)

    print(
        f"Total Reward Mean             "
        f"{baseline['total_reward_mean']:14.6f} "
        f"{phy_only['total_reward_mean']:14.6f} "
        f"{phy_observation['total_reward_mean']:14.6f}"
    )

    print(
        f"Mean Reward                   "
        f"{baseline['mean_reward_mean']:14.6f} "
        f"{phy_only['mean_reward_mean']:14.6f} "
        f"{phy_observation['mean_reward_mean']:14.6f}"
    )

    print(
        f"Spectral Efficiency           "
        f"{baseline['spectral_efficiency_mean']:14.6f} "
        f"{phy_only['spectral_efficiency_mean']:14.6f} "
        f"{phy_observation['spectral_efficiency_mean']:14.6f}"
    )

    print(
        f"Energy Efficiency             "
        f"{baseline['energy_efficiency_mean']:14.3f} "
        f"{phy_only['energy_efficiency_mean']:14.3f} "
        f"{phy_observation['energy_efficiency_mean']:14.3f}"
    )

    print(
        f"Mean Trust                    "
        f"{baseline['mean_trust_mean']:14.6f} "
        f"{phy_only['mean_trust_mean']:14.6f} "
        f"{phy_observation['mean_trust_mean']:14.6f}"
    )

    print(
        f"Final Trust                   "
        f"{baseline['final_trust_mean']:14.6f} "
        f"{phy_only['final_trust_mean']:14.6f} "
        f"{phy_observation['final_trust_mean']:14.6f}"
    )

    print(
        f"Mean PQI                      "
        f"{baseline['mean_pqi']:14.6f} "
        f"{phy_only['mean_pqi']:14.6f} "
        f"{phy_observation['mean_pqi']:14.6f}"
    )

    print(
        f"Mean XPI                      "
        f"{baseline['mean_xpi']:14.6f} "
        f"{phy_only['mean_xpi']:14.6f} "
        f"{phy_observation['mean_xpi']:14.6f}"
    )

    print(
        f"Final Queue                   "
        f"{baseline['final_queue_mean']:14.6f} "
        f"{phy_only['final_queue_mean']:14.6f} "
        f"{phy_observation['final_queue_mean']:14.6f}"
    )

    print()

    print(
        "Percentage change:"
    )

    print()

    # -----------------------------------------------------
    # PHY effect
    # -----------------------------------------------------

    reward_phy_change = percentage_change(
        baseline["total_reward_mean"],
        phy_only["total_reward_mean"],
    )

    se_phy_change = percentage_change(
        baseline["spectral_efficiency_mean"],
        phy_only["spectral_efficiency_mean"],
    )

    ee_phy_change = percentage_change(
        baseline["energy_efficiency_mean"],
        phy_only["energy_efficiency_mean"],
    )

    # -----------------------------------------------------
    # Observation effect
    # -----------------------------------------------------

    reward_observation_change = percentage_change(
        phy_only["total_reward_mean"],
        phy_observation["total_reward_mean"],
    )

    se_observation_change = percentage_change(
        phy_only["spectral_efficiency_mean"],
        phy_observation["spectral_efficiency_mean"],
    )

    ee_observation_change = percentage_change(
        phy_only["energy_efficiency_mean"],
        phy_observation["energy_efficiency_mean"],
    )

    # -----------------------------------------------------
    # Full B2-A -> B2-C
    # -----------------------------------------------------

    reward_total_change = percentage_change(
        baseline["total_reward_mean"],
        phy_observation["total_reward_mean"],
    )

    se_total_change = percentage_change(
        baseline["spectral_efficiency_mean"],
        phy_observation["spectral_efficiency_mean"],
    )

    ee_total_change = percentage_change(
        baseline["energy_efficiency_mean"],
        phy_observation["energy_efficiency_mean"],
    )

    print(
        f"PHY effect: "
        f"B2-B vs B2-A"
    )

    print(
        f"  Reward change = "
        f"{reward_phy_change:+.4f}%"
    )

    print(
        f"  SE change     = "
        f"{se_phy_change:+.4f}%"
    )

    print(
        f"  EE change     = "
        f"{ee_phy_change:+.4f}%"
    )

    print()

    print(
        f"Observation effect: "
        f"B2-C vs B2-B"
    )

    print(
        f"  Reward change = "
        f"{reward_observation_change:+.4f}%"
    )

    print(
        f"  SE change     = "
        f"{se_observation_change:+.4f}%"
    )

    print(
        f"  EE change     = "
        f"{ee_observation_change:+.4f}%"
    )

    print()

    print(
        f"Full effect: "
        f"B2-C vs B2-A"
    )

    print(
        f"  Reward change = "
        f"{reward_total_change:+.4f}%"
    )

    print(
        f"  SE change     = "
        f"{se_total_change:+.4f}%"
    )

    print(
        f"  EE change     = "
        f"{ee_total_change:+.4f}%"
    )

    print()

    print(
        "Interpretation:"
    )

    print(
        "  B2-A isolates the scalar-PHY baseline."
    )

    print(
        "  B2-B isolates the effect of enabling "
        "polarization-aware PHY without changing "
        "the observation space."
    )

    print(
        "  B2-C enables polarization-aware PHY plus "
        "PQI/XPI observations in the state."
    )

    print(
        "  B2-C does NOT demonstrate polarization "
        "control or optimal polarization optimization."
    )

    print(
        "  B2-C is an observation-awareness experiment."
    )

    print("=" * 100)


# =========================================================
# FINAL VALIDATION
# =========================================================

def validate_results(
    results: List[BenchmarkResult],
) -> None:
    """
    Validate the complete benchmark result set.
    """

    expected_result_count = (
        len(SEEDS) * 3
    )

    # -----------------------------------------------------
    # Result count
    # -----------------------------------------------------

    assert len(results) == (
        expected_result_count
    ), (
        "Unexpected number of benchmark results: "
        f"expected={expected_result_count}, "
        f"actual={len(results)}"
    )

    # -----------------------------------------------------
    # Per-result validation
    # -----------------------------------------------------

    for result in results:

        # ---------------------------------------------
        # Steps
        # ---------------------------------------------

        assert result.steps_completed == (
            STEPS
        ), (
            f"{result.name} seed={result.seed}: "
            "incomplete episode"
        )

        # ---------------------------------------------
        # Action dimension
        # ---------------------------------------------

        assert result.action_dim == (
            EXPECTED_ACTION_DIM
        ), (
            f"{result.name} seed={result.seed}: "
            f"expected action_dim="
            f"{EXPECTED_ACTION_DIM}, "
            f"actual={result.action_dim}"
        )

        # ---------------------------------------------
        # State dimensions
        # ---------------------------------------------

        if result.name == (
            "B2-A_BASELINE"
        ):

            assert result.state_dim == (
                EXPECTED_BASE_STATE_DIM
            ), (
                f"B2-A state dimension invalid: "
                f"{result.state_dim}"
            )

        elif result.name == (
            "B2-B_PHY_ONLY"
        ):

            assert result.state_dim == (
                EXPECTED_BASE_STATE_DIM
            ), (
                f"B2-B state dimension invalid: "
                f"{result.state_dim}"
            )

        elif result.name == (
            "B2-C_PHY_PLUS_OBSERVATION"
        ):

            assert result.state_dim == (
                EXPECTED_POLARIZATION_STATE_DIM
            ), (
                f"B2-C state dimension invalid: "
                f"{result.state_dim}"
            )

        else:

            raise AssertionError(
                "Unknown benchmark configuration: "
                f"{result.name}"
            )

        # ---------------------------------------------
        # Finite metrics
        # ---------------------------------------------

        numeric_values = (
            result.total_reward,
            result.mean_reward,
            result.spectral_efficiency,
            result.energy_efficiency,
            result.mean_trust,
            result.final_trust,
            result.mean_pqi,
            result.mean_xpi,
            result.final_queue,
        )

        for value in numeric_values:

            assert np.isfinite(
                value
            ), (
                f"Non-finite metric in "
                f"{result.name}, "
                f"seed={result.seed}: "
                f"{value}"
            )

    # -----------------------------------------------------
    # Seed coverage
    # -----------------------------------------------------

    for seed in SEEDS:

        seed_results = [
            result
            for result in results
            if result.seed == seed
        ]

        assert len(seed_results) == 3, (
            f"Seed {seed} does not contain "
            "all three configurations."
        )

        names = {
            result.name
            for result in seed_results
        }

        expected_names = {
            "B2-A_BASELINE",
            "B2-B_PHY_ONLY",
            "B2-C_PHY_PLUS_OBSERVATION",
        }

        assert names == expected_names, (
            f"Seed {seed} configuration mismatch: "
            f"{names}"
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print()

    print(
        "Starting Phase-4B.2 controlled "
        "polarization observation benchmark..."
    )

    print()

    print(
        f"Seeds              : {SEEDS}"
    )

    print(
        f"Steps / seed       : {STEPS}"
    )

    print(
        f"Users              : {NUM_USERS}"
    )

    print(
        f"IRF elements       : {NUM_RIS_ELEMENTS}"
    )

    print(
        f"Bandwidth          : {BANDWIDTH_HZ / 1e6:.0f} MHz"
    )

    print(
        f"Carrier frequency  : "
        f"{CARRIER_FREQUENCY_HZ / 1e9:.0f} GHz"
    )

    print(
        f"Cross-pol factor   : "
        f"{CROSS_POLARIZATION_FACTOR}"
    )

    print(
        f"Expected states    : "
        f"B2-A/B={EXPECTED_BASE_STATE_DIM}, "
        f"B2-C={EXPECTED_POLARIZATION_STATE_DIM}"
    )

    print(
        f"Expected actions   : "
        f"{EXPECTED_ACTION_DIM}"
    )

    print()

    # =====================================================
    # RUN
    # =====================================================

    results = run_benchmark()

    # =====================================================
    # VALIDATE
    # =====================================================

    validate_results(
        results
    )

    # =====================================================
    # REPORT
    # =====================================================

    print_report(
        results
    )

    print_scientific_summary(
        results
    )

    # =====================================================
    # FINAL PASS
    # =====================================================

    print()

    print("=" * 100)

    print(
        "PHASE-4B.2 CONTROLLED "
        "BENCHMARK PASSED"
    )

    print(
        "All configurations completed "
        "with expected state/action dimensions."
    )

    print(
        "B2-A: Polarization OFF / State=100"
    )

    print(
        "B2-B: Polarization ON / State=100"
    )

    print(
        "B2-C: Polarization ON + PQI/XPI "
        "Observation / State=140"
    )

    print(
        "Action dimension remains 40 "
        "across all configurations."
    )

    print(
        "No SAC training was performed."
    )

    print("=" * 100)

    print()