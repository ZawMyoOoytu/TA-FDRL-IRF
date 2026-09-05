
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ============================================================
# PROJECT ROOT / IMPORT PATH
# ============================================================
#
# This script lives in:
#
#   TA-FDRL-IRF/
#       experiments/
#           phase4a_multiseed_benchmark.py
#
# The environment package lives in:
#
#   TA-FDRL-IRF/
#       environment/
#           irf_env.py
#
# When running:
#
#   python experiments\phase4a_multiseed_benchmark.py
#
# Python may not automatically include the project root in
# sys.path. Therefore we explicitly add it here.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from environment.irf_env import IRFConfig, IRFEnvironment


# ============================================================
# PHASE-4A RESEARCH CONFIGURATION
# ============================================================

SEEDS = [
    42,
    123,
    2026,
    4096,
    7777,
    9999,
]

NUM_STEPS = 200

EXPECTED_STATE_DIM = 100
EXPECTED_ACTION_DIM = 40


# ============================================================
# RESULT CONTAINER
# ============================================================

@dataclass
class BenchmarkResult:
    seed: int
    polarization_enabled: bool

    total_reward: float
    mean_reward: float

    mean_spectral_efficiency: float
    mean_energy_efficiency: float

    initial_trust: float
    mean_trust: float
    final_trust: float
    trust_change: float

    mean_queue: float
    final_queue: float

    steps_completed: int


# ============================================================
# COMMON ACTION GENERATION
# ============================================================

def generate_action_sequence(
    seed: int,
    num_steps: int,
    action_dim: int,
) -> np.ndarray:
    """
    Generate one deterministic action sequence.

    IMPORTANT:
    The same action sequence is reused for Polarization OFF
    and Polarization ON for the same seed.

    This creates a controlled paired comparison.
    """

    rng = np.random.default_rng(seed)

    actions = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(num_steps, action_dim),
    )

    return actions


# ============================================================
# SINGLE CONTROLLED RUN
# ============================================================

def run_single_condition(
    seed: int,
    polarization_enabled: bool,
    actions: np.ndarray,
) -> BenchmarkResult:
    """
    Run one Phase-4A condition.

    The only controlled difference between OFF and ON is
    polarization-aware PHY modeling.

    The environment itself is seeded identically.
    The external action sequence is also identical.
    """

    cfg = IRFConfig(
        polarization_enabled=polarization_enabled,
        optimize_ris=False,
        fixed_trust=False,
        seed=seed,
        max_steps=NUM_STEPS,
    )

    env = IRFEnvironment(cfg)

    # --------------------------------------------------------
    # Reset environment
    # --------------------------------------------------------

    state = env.reset(seed=seed)

    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    if state.shape != (EXPECTED_STATE_DIM,):
        raise RuntimeError(
            "Unexpected state dimension: "
            f"{state.shape}; expected "
            f"({EXPECTED_STATE_DIM},)"
        )

    if actions.shape[1] != EXPECTED_ACTION_DIM:
        raise RuntimeError(
            "Unexpected action dimension: "
            f"{actions.shape[1]}; expected "
            f"{EXPECTED_ACTION_DIM}"
        )

    # --------------------------------------------------------
    # Initial trust
    # --------------------------------------------------------

    initial_trust = float(
        np.mean(env.trust)
    )

    # --------------------------------------------------------
    # Metric collectors
    # --------------------------------------------------------

    rewards: list[float] = []
    spectral_efficiencies: list[float] = []
    energy_efficiencies: list[float] = []
    trusts: list[float] = []
    queues: list[float] = []

    # --------------------------------------------------------
    # Environment rollout
    # --------------------------------------------------------

    for step in range(NUM_STEPS):

        action = actions[step]

        (
            state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        rewards.append(
            float(reward)
        )

        spectral_efficiencies.append(
            float(env.spectral_efficiency)
        )

        energy_efficiencies.append(
            float(env.energy_efficiency)
        )

        trusts.append(
            float(np.mean(env.trust))
        )

        queues.append(
            float(np.mean(env.queue))
        )

        if terminated or truncated:
            break

    # --------------------------------------------------------
    # Empty rollout protection
    # --------------------------------------------------------

    if not rewards:
        raise RuntimeError(
            f"Seed {seed} produced zero environment steps."
        )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    final_trust = float(
        trusts[-1]
    )

    final_queue = float(
        queues[-1]
    )

    return BenchmarkResult(
        seed=seed,
        polarization_enabled=polarization_enabled,

        total_reward=float(
            np.sum(rewards)
        ),

        mean_reward=float(
            np.mean(rewards)
        ),

        mean_spectral_efficiency=float(
            np.mean(spectral_efficiencies)
        ),

        mean_energy_efficiency=float(
            np.mean(energy_efficiencies)
        ),

        initial_trust=initial_trust,

        mean_trust=float(
            np.mean(trusts)
        ),

        final_trust=final_trust,

        trust_change=float(
            final_trust - initial_trust
        ),

        mean_queue=float(
            np.mean(queues)
        ),

        final_queue=final_queue,

        steps_completed=len(rewards),
    )


# ============================================================
# STATISTICS
# ============================================================

def mean_std_ci(
    values: np.ndarray,
) -> tuple[float, float, float]:
    """
    Return:

        mean
        sample standard deviation
        approximate 95% CI half-width

    The CI uses the normal approximation:

        mean +/- 1.96 * std / sqrt(n)

    With six seeds this should be treated as descriptive
    rather than as a definitive inferential test.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.size == 0:
        raise ValueError(
            "Cannot calculate statistics for empty data."
        )

    n = len(values)

    mean = float(
        np.mean(values)
    )

    if n <= 1:
        return (
            mean,
            0.0,
            0.0,
        )

    std = float(
        np.std(
            values,
            ddof=1,
        )
    )

    margin = (
        1.96
        * std
        / np.sqrt(n)
    )

    return (
        mean,
        std,
        float(margin),
    )


# ============================================================
# RESULT COLLECTION
# ============================================================

def collect_metric(
    results: list[BenchmarkResult],
    polarization_enabled: bool,
    attribute: str,
) -> np.ndarray:
    """
    Collect one metric for either Polarization OFF or ON.
    """

    selected = [
        result
        for result in results
        if result.polarization_enabled
        == polarization_enabled
    ]

    if not selected:
        raise ValueError(
            "No results found for "
            f"polarization_enabled={polarization_enabled}"
        )

    return np.asarray(
        [
            getattr(result, attribute)
            for result in selected
        ],
        dtype=float,
    )


# ============================================================
# AGGREGATE SUMMARY
# ============================================================

def summarize_results(
    results: list[BenchmarkResult],
    polarization_enabled: bool,
) -> dict[str, dict[str, float]]:
    """
    Calculate mean, standard deviation and 95% CI
    for each benchmark metric.
    """

    metrics = [
        "total_reward",
        "mean_reward",
        "mean_spectral_efficiency",
        "mean_energy_efficiency",
        "mean_trust",
        "trust_change",
        "mean_queue",
        "final_trust",
        "final_queue",
    ]

    summary: dict[str, dict[str, float]] = {}

    for metric in metrics:

        values = collect_metric(
            results,
            polarization_enabled,
            metric,
        )

        mean, std, ci95 = mean_std_ci(
            values
        )

        summary[metric] = {
            "mean": mean,
            "std": std,
            "ci95": ci95,
        }

    return summary


# ============================================================
# PERCENTAGE CHANGE
# ============================================================

def percentage_change(
    baseline: float,
    new_value: float,
) -> float:
    """
    Calculate percentage change from baseline to new value.
    """

    if abs(baseline) < 1e-12:
        return 0.0

    return (
        (new_value - baseline)
        / abs(baseline)
        * 100.0
    )


# ============================================================
# PAIRED DIFFERENCES
# ============================================================

def paired_metric_differences(
    results: list[BenchmarkResult],
    attribute: str,
) -> np.ndarray:
    """
    Calculate paired ON - OFF differences for each seed.

    Pairing is performed by seed.
    """

    off_by_seed = {
        result.seed: result
        for result in results
        if not result.polarization_enabled
    }

    on_by_seed = {
        result.seed: result
        for result in results
        if result.polarization_enabled
    }

    common_seeds = sorted(
        set(off_by_seed)
        & set(on_by_seed)
    )

    if not common_seeds:
        raise ValueError(
            "No common seeds found for paired comparison."
        )

    differences = []

    for seed in common_seeds:

        off_value = float(
            getattr(
                off_by_seed[seed],
                attribute,
            )
        )

        on_value = float(
            getattr(
                on_by_seed[seed],
                attribute,
            )
        )

        differences.append(
            on_value - off_value
        )

    return np.asarray(
        differences,
        dtype=float,
    )


# ============================================================
# PAIRED SUMMARY
# ============================================================

def paired_summary(
    results: list[BenchmarkResult],
    attribute: str,
) -> dict[str, float]:
    """
    Calculate statistics over paired ON - OFF differences.
    """

    differences = paired_metric_differences(
        results,
        attribute,
    )

    mean, std, ci95 = mean_std_ci(
        differences
    )

    return {
        "mean_difference": mean,
        "std_difference": std,
        "ci95": ci95,
    }


# ============================================================
# PRINT AGGREGATE METRIC
# ============================================================

def print_metric_summary(
    label: str,
    key: str,
    off_summary: dict[str, dict[str, float]],
    on_summary: dict[str, dict[str, float]],
) -> None:
    """
    Print OFF/ON aggregate statistics and percentage change.
    """

    off = off_summary[key]
    on = on_summary[key]

    change = percentage_change(
        off["mean"],
        on["mean"],
    )

    print()
    print(label)

    print(
        f"  OFF : "
        f"{off['mean']:.10f} "
        f"+/- {off['std']:.10f} "
        f"(95% CI +/- {off['ci95']:.10f})"
    )

    print(
        f"  ON  : "
        f"{on['mean']:.10f} "
        f"+/- {on['std']:.10f} "
        f"(95% CI +/- {on['ci95']:.10f})"
    )

    print(
        f"  Change: {change:+.4f}%"
    )


# ============================================================
# INITIAL TRUST CONSISTENCY
# ============================================================

def verify_initial_trust_consistency(
    results: list[BenchmarkResult],
) -> float:
    """
    Verify that Polarization OFF and ON use identical
    initial trust values for every seed.
    """

    off_initial = collect_metric(
        results,
        False,
        "initial_trust",
    )

    on_initial = collect_metric(
        results,
        True,
        "initial_trust",
    )

    if len(off_initial) != len(on_initial):
        raise RuntimeError(
            "OFF and ON seed counts differ."
        )

    difference = float(
        np.max(
            np.abs(
                off_initial - on_initial
            )
        )
    )

    return difference


# ============================================================
# COMMON-RANDOMNESS VERIFICATION
# ============================================================

def print_rng_verification(
    results: list[BenchmarkResult],
) -> None:

    initial_difference = (
        verify_initial_trust_consistency(
            results
        )
    )

    print()
    print("=" * 70)
    print("COMMON-RANDOMNESS VERIFICATION")
    print("=" * 70)

    print(
        "Maximum initial-trust difference "
        "(OFF vs ON): "
        f"{initial_difference:.12e}"
    )

    if initial_difference < 1e-12:

        print(
            "Initial trust consistency: PASS"
        )

    else:

        print(
            "Initial trust consistency: WARNING"
        )


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main() -> None:

    print("=" * 70)
    print(
        "TA-FDRL-IRF | "
        "PHASE-4A MULTI-SEED POLARIZATION BENCHMARK"
    )
    print("=" * 70)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        f"Number of seeds: {len(SEEDS)}"
    )

    print(
        f"Steps per run: {NUM_STEPS}"
    )

    print(
        f"State dimension target: "
        f"{EXPECTED_STATE_DIM}"
    )

    print(
        f"Action dimension: "
        f"{EXPECTED_ACTION_DIM}"
    )

    print(
        "Adaptive trust: ENABLED"
    )

    print(
        "RIS optimization: DISABLED"
    )

    print(
        "SAC training: NOT PERFORMED"
    )

    print()

    all_results: list[BenchmarkResult] = []

    # ========================================================
    # RUN ALL SEEDS
    # ========================================================

    for seed in SEEDS:

        print("-" * 70)
        print(
            f"SEED {seed}"
        )
        print("-" * 70)

        # ----------------------------------------------------
        # Common action sequence
        # ----------------------------------------------------

        actions = generate_action_sequence(
            seed=seed,
            num_steps=NUM_STEPS,
            action_dim=EXPECTED_ACTION_DIM,
        )

        print(
            f"Actions generated: "
            f"{len(actions)}"
        )

        # ----------------------------------------------------
        # Polarization OFF
        # ----------------------------------------------------

        print(
            "Running Polarization OFF..."
        )

        off_result = run_single_condition(
            seed=seed,
            polarization_enabled=False,
            actions=actions,
        )

        all_results.append(
            off_result
        )

        print(
            f"  Reward = "
            f"{off_result.total_reward:.6f} | "
            f"SE = "
            f"{off_result.mean_spectral_efficiency:.8f} | "
            f"EE = "
            f"{off_result.mean_energy_efficiency:.2f} | "
            f"Trust = "
            f"{off_result.final_trust:.6f} | "
            f"Queue = "
            f"{off_result.final_queue:.6f}"
        )

        # ----------------------------------------------------
        # Polarization ON
        # ----------------------------------------------------

        print(
            "Running Polarization ON..."
        )

        on_result = run_single_condition(
            seed=seed,
            polarization_enabled=True,
            actions=actions,
        )

        all_results.append(
            on_result
        )

        print(
            f"  Reward = "
            f"{on_result.total_reward:.6f} | "
            f"SE = "
            f"{on_result.mean_spectral_efficiency:.8f} | "
            f"EE = "
            f"{on_result.mean_energy_efficiency:.2f} | "
            f"Trust = "
            f"{on_result.final_trust:.6f} | "
            f"Queue = "
            f"{on_result.final_queue:.6f}"
        )

        # ----------------------------------------------------
        # Paired comparison
        # ----------------------------------------------------

        reward_delta = percentage_change(
            off_result.total_reward,
            on_result.total_reward,
        )

        se_delta = percentage_change(
            off_result.mean_spectral_efficiency,
            on_result.mean_spectral_efficiency,
        )

        ee_delta = percentage_change(
            off_result.mean_energy_efficiency,
            on_result.mean_energy_efficiency,
        )

        print(
            f"  ON vs OFF: "
            f"Reward {reward_delta:+.4f}% | "
            f"SE {se_delta:+.4f}% | "
            f"EE {ee_delta:+.4f}%"
        )

    # ========================================================
    # AGGREGATE SUMMARIES
    # ========================================================

    off_summary = summarize_results(
        all_results,
        polarization_enabled=False,
    )

    on_summary = summarize_results(
        all_results,
        polarization_enabled=True,
    )

    # ========================================================
    # STATISTICAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("MULTI-SEED STATISTICAL SUMMARY")
    print("=" * 70)

    print_metric_summary(
        "Total Reward",
        "total_reward",
        off_summary,
        on_summary,
    )

    print_metric_summary(
        "Mean Reward / Step",
        "mean_reward",
        off_summary,
        on_summary,
    )

    print_metric_summary(
        "Spectral Efficiency",
        "mean_spectral_efficiency",
        off_summary,
        on_summary,
    )

    print_metric_summary(
        "Energy Efficiency",
        "mean_energy_efficiency",
        off_summary,
        on_summary,
    )

    print_metric_summary(
        "Mean Trust",
        "mean_trust",
        off_summary,
        on_summary,
    )

    print_metric_summary(
        "Trust Change",
        "trust_change",
        off_summary,
        on_summary,
    )

    print_metric_summary(
        "Mean Queue",
        "mean_queue",
        off_summary,
        on_summary,
    )

    print_metric_summary(
        "Final Trust",
        "final_trust",
        off_summary,
        on_summary,
    )

    print_metric_summary(
        "Final Queue",
        "final_queue",
        off_summary,
        on_summary,
    )

    # ========================================================
    # PAIRED DIFFERENCE ANALYSIS
    # ========================================================

    print()
    print("=" * 70)
    print("PAIRED ON - OFF DIFFERENCES")
    print("=" * 70)

    paired_metrics = [
        (
            "Total Reward",
            "total_reward",
        ),
        (
            "Mean Reward",
            "mean_reward",
        ),
        (
            "Spectral Efficiency",
            "mean_spectral_efficiency",
        ),
        (
            "Energy Efficiency",
            "mean_energy_efficiency",
        ),
        (
            "Mean Trust",
            "mean_trust",
        ),
        (
            "Trust Change",
            "trust_change",
        ),
        (
            "Mean Queue",
            "mean_queue",
        ),
    ]

    paired_results = {}

    for label, key in paired_metrics:

        result = paired_summary(
            all_results,
            key,
        )

        paired_results[key] = result

        print()
        print(label)

        print(
            f"  Mean ON-OFF difference: "
            f"{result['mean_difference']:.10f}"
        )

        print(
            f"  Std of difference: "
            f"{result['std_difference']:.10f}"
        )

        print(
            f"  95% CI +/-: "
            f"{result['ci95']:.10f}"
        )

    # ========================================================
    # RNG VERIFICATION
    # ========================================================

    print_rng_verification(
        all_results
    )

    initial_difference = (
        verify_initial_trust_consistency(
            all_results
        )
    )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    results_dir = (
        PROJECT_ROOT
        / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        results_dir
        / "phase4a_multiseed_benchmark.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        f.write(
            "TA-FDRL-IRF | "
            "PHASE-4A MULTI-SEED "
            "POLARIZATION BENCHMARK\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"Project root: {PROJECT_ROOT}\n"
        )

        f.write(
            f"Seeds: {SEEDS}\n"
        )

        f.write(
            f"Number of seeds: {len(SEEDS)}\n"
        )

        f.write(
            f"Steps per run: {NUM_STEPS}\n"
        )

        f.write(
            f"State dimension: "
            f"{EXPECTED_STATE_DIM}\n"
        )

        f.write(
            f"Action dimension: "
            f"{EXPECTED_ACTION_DIM}\n"
        )

        f.write(
            "Adaptive trust: ENABLED\n"
        )

        f.write(
            "RIS optimization: DISABLED\n"
        )

        f.write(
            "SAC training: NOT PERFORMED\n\n"
        )

        # ----------------------------------------------------
        # Methodology
        # ----------------------------------------------------

        f.write(
            "=" * 70
            + "\n"
        )

        f.write(
            "BENCHMARK METHODOLOGY\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "Polarization OFF and ON are evaluated using the "
            "same seed-specific external action sequence.\n"
        )

        f.write(
            "Environment channel randomness is separated from "
            "dynamics randomness.\n"
        )

        f.write(
            "The same seed is used for the corresponding "
            "OFF and ON environment resets.\n"
        )

        f.write(
            "Adaptive trust remains enabled.\n"
        )

        f.write(
            "RIS optimization remains disabled.\n"
        )

        f.write(
            "State and action dimensions remain unchanged "
            "from Phase-3 baseline.\n\n"
        )

        # ----------------------------------------------------
        # Per-seed results
        # ----------------------------------------------------

        f.write(
            "=" * 70
            + "\n"
        )

        f.write(
            "PER-SEED RESULTS\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        for seed in SEEDS:

            off = next(
                result
                for result in all_results
                if result.seed == seed
                and not result.polarization_enabled
            )

            on = next(
                result
                for result in all_results
                if result.seed == seed
                and result.polarization_enabled
            )

            f.write(
                f"Seed {seed}\n"
            )

            f.write(
                "-" * 50
                + "\n"
            )

            f.write(
                f"Steps OFF: "
                f"{off.steps_completed}\n"
            )

            f.write(
                f"Steps ON : "
                f"{on.steps_completed}\n\n"
            )

            f.write(
                f"OFF total reward: "
                f"{off.total_reward:.10f}\n"
            )

            f.write(
                f"ON total reward : "
                f"{on.total_reward:.10f}\n"
            )

            f.write(
                f"Reward change: "
                f"{percentage_change(off.total_reward, on.total_reward):+.6f}%\n\n"
            )

            f.write(
                f"OFF mean reward: "
                f"{off.mean_reward:.10f}\n"
            )

            f.write(
                f"ON mean reward : "
                f"{on.mean_reward:.10f}\n\n"
            )

            f.write(
                f"OFF spectral efficiency: "
                f"{off.mean_spectral_efficiency:.10f}\n"
            )

            f.write(
                f"ON spectral efficiency : "
                f"{on.mean_spectral_efficiency:.10f}\n"
            )

            f.write(
                f"SE change: "
                f"{percentage_change(off.mean_spectral_efficiency, on.mean_spectral_efficiency):+.6f}%\n\n"
            )

            f.write(
                f"OFF energy efficiency: "
                f"{off.mean_energy_efficiency:.10f}\n"
            )

            f.write(
                f"ON energy efficiency : "
                f"{on.mean_energy_efficiency:.10f}\n"
            )

            f.write(
                f"EE change: "
                f"{percentage_change(off.mean_energy_efficiency, on.mean_energy_efficiency):+.6f}%\n\n"
            )

            f.write(
                f"OFF initial trust: "
                f"{off.initial_trust:.10f}\n"
            )

            f.write(
                f"ON initial trust : "
                f"{on.initial_trust:.10f}\n"
            )

            f.write(
                f"Initial trust difference: "
                f"{on.initial_trust - off.initial_trust:.12e}\n\n"
            )

            f.write(
                f"OFF mean trust: "
                f"{off.mean_trust:.10f}\n"
            )

            f.write(
                f"ON mean trust : "
                f"{on.mean_trust:.10f}\n"
            )

            f.write(
                f"OFF final trust: "
                f"{off.final_trust:.10f}\n"
            )

            f.write(
                f"ON final trust : "
                f"{on.final_trust:.10f}\n"
            )

            f.write(
                f"Trust-change difference: "
                f"{on.trust_change - off.trust_change:.10f}\n\n"
            )

            f.write(
                f"OFF mean queue: "
                f"{off.mean_queue:.10f}\n"
            )

            f.write(
                f"ON mean queue : "
                f"{on.mean_queue:.10f}\n"
            )

            f.write(
                f"OFF final queue: "
                f"{off.final_queue:.10f}\n"
            )

            f.write(
                f"ON final queue : "
                f"{on.final_queue:.10f}\n\n"
            )

        # ----------------------------------------------------
        # Statistical summary
        # ----------------------------------------------------

        f.write(
            "=" * 70
            + "\n"
        )

        f.write(
            "STATISTICAL SUMMARY\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        for label, key in [
            (
                "Total Reward",
                "total_reward",
            ),
            (
                "Mean Reward",
                "mean_reward",
            ),
            (
                "Spectral Efficiency",
                "mean_spectral_efficiency",
            ),
            (
                "Energy Efficiency",
                "mean_energy_efficiency",
            ),
            (
                "Mean Trust",
                "mean_trust",
            ),
            (
                "Trust Change",
                "trust_change",
            ),
            (
                "Mean Queue",
                "mean_queue",
            ),
            (
                "Final Trust",
                "final_trust",
            ),
            (
                "Final Queue",
                "final_queue",
            ),
        ]:

            off = off_summary[key]
            on = on_summary[key]

            change = percentage_change(
                off["mean"],
                on["mean"],
            )

            f.write(
                f"{label}\n"
            )

            f.write(
                f"  OFF mean = "
                f"{off['mean']:.10f}\n"
            )

            f.write(
                f"  OFF std = "
                f"{off['std']:.10f}\n"
            )

            f.write(
                f"  OFF 95% CI +/- = "
                f"{off['ci95']:.10f}\n"
            )

            f.write(
                f"  ON mean = "
                f"{on['mean']:.10f}\n"
            )

            f.write(
                f"  ON std = "
                f"{on['std']:.10f}\n"
            )

            f.write(
                f"  ON 95% CI +/- = "
                f"{on['ci95']:.10f}\n"
            )

            f.write(
                f"  OFF -> ON change = "
                f"{change:+.6f}%\n\n"
            )

        # ----------------------------------------------------
        # Paired differences
        # ----------------------------------------------------

        f.write(
            "=" * 70
            + "\n"
        )

        f.write(
            "PAIRED ON - OFF DIFFERENCES\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        for label, key in paired_metrics:

            result = paired_results[key]

            f.write(
                f"{label}\n"
            )

            f.write(
                f"  Mean ON-OFF difference = "
                f"{result['mean_difference']:.10f}\n"
            )

            f.write(
                f"  Std difference = "
                f"{result['std_difference']:.10f}\n"
            )

            f.write(
                f"  95% CI +/- = "
                f"{result['ci95']:.10f}\n\n"
            )

        # ----------------------------------------------------
        # RNG control
        # ----------------------------------------------------

        f.write(
            "=" * 70
            + "\n"
        )

        f.write(
            "COMMON-RANDOMNESS VERIFICATION\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "Same seed-specific action sequence: YES\n"
        )

        f.write(
            "Channel RNG separated from dynamics RNG: YES\n"
        )

        f.write(
            "Same initial trust OFF/ON target: YES\n"
        )

        f.write(
            "Maximum initial-trust difference: "
            f"{initial_difference:.12e}\n"
        )

        if initial_difference < 1e-12:

            f.write(
                "Initial trust consistency: PASS\n"
            )

        else:

            f.write(
                "Initial trust consistency: WARNING\n"
            )

        # ----------------------------------------------------
        # Final status
        # ----------------------------------------------------

        f.write(
            "\n"
            + "=" * 70
            + "\n"
        )

        f.write(
            "FINAL STATUS\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "Environment validation: PASSED\n"
        )

        f.write(
            "Controlled action sequence: PASSED\n"
        )

        f.write(
            "Multi-seed benchmark: COMPLETED\n"
        )

        f.write(
            "SAC training: NOT PERFORMED\n"
        )

    # ========================================================
    # FINAL CONSOLE STATUS
    # ========================================================

    print()
    print("=" * 70)
    print(
        "PHASE-4A MULTI-SEED BENCHMARK COMPLETE"
    )
    print("=" * 70)

    print()

    print(
        "Report saved to:"
    )

    print(
        report_path
    )

    print()

    print(
        "Environment validation: PASSED"
    )

    print(
        "Controlled action sequence: PASSED"
    )

    print(
        "Common dynamics RNG verification: "
        + (
            "PASSED"
            if initial_difference < 1e-12
            else "WARNING"
        )
    )

    print(
        "Multi-seed controlled PHY benchmark: COMPLETED"
    )

    print(
        "SAC training: NOT PERFORMED"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

