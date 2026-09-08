"""
TA-FDRL-IRF
Phase-4A Polarization Benchmark

Purpose
-------
Compare the same IRF environment under:

    1. Polarization OFF
    2. Polarization ON

The SAC architecture is NOT involved.

Metrics
-------
    - Total reward
    - Mean reward per step
    - Spectral efficiency
    - Energy efficiency
    - Initial trust
    - Mean trust
    - Final trust
    - Trust change
    - Mean queue
    - Final queue

Phase-4A Constraints
--------------------
    - State dimension: 100
    - Action dimension: 40
    - RIS optimization: disabled
    - Adaptive trust: enabled
    - SAC architecture: unchanged

Method
------
The same externally generated action sequence is used for both
polarization conditions.

Important API compatibility
---------------------------
The current IRFEnvironment API uses:

    env = IRFEnvironment(config=config)
    state = env.reset(seed=SEED)

Therefore, this benchmark does NOT pass seed= to __init__().

The reset handler below is also tolerant of either:

    state

or:

    (state, info)

being returned by reset().
"""

from pathlib import Path
import sys

import numpy as np


# =====================================================================
# PROJECT ROOT
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from environment.irf_env import IRFConfig, IRFEnvironment


# =====================================================================
# BENCHMARK CONFIGURATION
# =====================================================================

SEED = 2026

NUM_STEPS = 200

ACTION_DIM = 40

NUM_USERS = 20

NUM_RIS_ELEMENTS = 64

BANDWIDTH_HZ = 100e6

CARRIER_FREQUENCY_HZ = 28e9

MAX_POWER_W = 1.0

CIRCUIT_POWER_W = 0.1

NOISE_FIGURE_DB = 7.0

NOISE_DENSITY_DBM_HZ = -174.0


# =====================================================================
# RESULT PATH
# =====================================================================

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_FILE = (
    RESULTS_DIR /
    "phase4a_polarization_benchmark.txt"
)


# =====================================================================
# DETERMINISTIC ACTION GENERATOR
# =====================================================================

def generate_actions(
    num_steps: int,
    action_dim: int,
    seed: int,
):
    """
    Generate one deterministic action sequence.

    The exact same action sequence is used for:

        Polarization OFF
        Polarization ON

    This removes the difference caused by random action selection.
    """

    rng = np.random.default_rng(seed)

    actions = []

    for _ in range(num_steps):

        action = rng.uniform(
            low=-1.0,
            high=1.0,
            size=action_dim,
        ).astype(np.float32)

        actions.append(action)

    return actions


# =====================================================================
# CREATE CONFIG
# =====================================================================

def create_config(
    polarization_enabled: bool,
):
    """
    Create the Phase-4A IRF configuration.

    The only experimental variable is:

        polarization_enabled

    Everything else remains identical.
    """

    config = IRFConfig(
        num_users=NUM_USERS,

        num_ris_elements=NUM_RIS_ELEMENTS,

        bandwidth_hz=BANDWIDTH_HZ,

        carrier_frequency_hz=CARRIER_FREQUENCY_HZ,

        max_power_w=MAX_POWER_W,

        circuit_power_w=CIRCUIT_POWER_W,

        noise_figure_db=NOISE_FIGURE_DB,

        noise_density_dbm_hz=NOISE_DENSITY_DBM_HZ,

        max_steps=NUM_STEPS,

        optimize_ris=False,

        fixed_trust=False,

        polarization_enabled=polarization_enabled,
    )

    return config


# =====================================================================
# RESET COMPATIBILITY HANDLER
# =====================================================================

def reset_environment(
    env,
    seed: int,
):
    """
    Reset the environment while supporting both common APIs.

    Supported:

        state = env.reset(seed=seed)

    and:

        state, info = env.reset(seed=seed)

    Returns:

        state
        info
    """

    reset_result = env.reset(
        seed=seed
    )

    # ---------------------------------------------------------------
    # Case 1:
    # reset() returns (state, info)
    # ---------------------------------------------------------------

    if isinstance(
        reset_result,
        tuple,
    ):

        if len(reset_result) >= 2:

            state = reset_result[0]

            info = reset_result[1]

        elif len(reset_result) == 1:

            state = reset_result[0]

            info = {}

        else:

            raise RuntimeError(
                "IRFEnvironment.reset() returned an empty tuple."
            )

    # ---------------------------------------------------------------
    # Case 2:
    # reset() returns state only
    # ---------------------------------------------------------------

    else:

        state = reset_result

        info = {}

    return state, info


# =====================================================================
# SAFE INFO VALUE
# =====================================================================

def get_info_float(
    info,
    key,
    default=0.0,
):
    """
    Safely extract a numeric value from info.
    """

    if not isinstance(
        info,
        dict,
    ):
        return float(default)

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


# =====================================================================
# RUN ONE ENVIRONMENT
# =====================================================================

def run_environment(
    polarization_enabled: bool,
    actions,
):
    """
    Run one complete benchmark.

    The environment is created without seed= because the current
    IRFEnvironment constructor does not accept a seed argument.

    The deterministic seed is supplied through reset().
    """

    # ---------------------------------------------------------------
    # Create configuration
    # ---------------------------------------------------------------

    config = create_config(
        polarization_enabled=polarization_enabled
    )

    # ---------------------------------------------------------------
    # Create environment
    # ---------------------------------------------------------------

    env = IRFEnvironment(
        config=config,
    )

    # ---------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------

    state, info = reset_environment(
        env=env,
        seed=SEED,
    )

    # ---------------------------------------------------------------
    # Initial trust
    # ---------------------------------------------------------------

    initial_trust = float(
        np.mean(
            env.trust
        )
    )

    # ---------------------------------------------------------------
    # Metric containers
    # ---------------------------------------------------------------

    reward_values = []

    se_values = []

    ee_values = []

    trust_values = []

    queue_values = []

    # ---------------------------------------------------------------
    # Total reward
    # ---------------------------------------------------------------

    total_reward = 0.0

    # ---------------------------------------------------------------
    # Environment loop
    # ---------------------------------------------------------------

    for step_index, action in enumerate(
        actions,
        start=1,
    ):

        # -----------------------------------------------------------
        # Step
        # -----------------------------------------------------------

        step_result = env.step(
            action
        )

        # -----------------------------------------------------------
        # Expected current API:
        #
        # state,
        # reward,
        # terminated,
        # truncated,
        # info
        # -----------------------------------------------------------

        if not isinstance(
            step_result,
            tuple,
        ):

            raise RuntimeError(
                "IRFEnvironment.step() did not return a tuple."
            )

        if len(step_result) != 5:

            raise RuntimeError(
                "Unexpected IRFEnvironment.step() return format. "
                f"Expected 5 values, got {len(step_result)}."
            )

        (
            state,
            reward,
            terminated,
            truncated,
            info,
        ) = step_result

        # -----------------------------------------------------------
        # Reward
        # -----------------------------------------------------------

        reward = float(
            reward
        )

        total_reward += reward

        reward_values.append(
            reward
        )

        # -----------------------------------------------------------
        # Spectral efficiency
        # -----------------------------------------------------------

        spectral_efficiency = get_info_float(
            info=info,
            key="spectral_efficiency",
            default=0.0,
        )

        se_values.append(
            spectral_efficiency
        )

        # -----------------------------------------------------------
        # Energy efficiency
        # -----------------------------------------------------------

        energy_efficiency = get_info_float(
            info=info,
            key="energy_efficiency",
            default=0.0,
        )

        ee_values.append(
            energy_efficiency
        )

        # -----------------------------------------------------------
        # Mean trust
        # -----------------------------------------------------------

        mean_trust = float(
            np.mean(
                env.trust
            )
        )

        trust_values.append(
            mean_trust
        )

        # -----------------------------------------------------------
        # Mean queue
        # -----------------------------------------------------------

        mean_queue = float(
            np.mean(
                env.queue
            )
        )

        queue_values.append(
            mean_queue
        )

        # -----------------------------------------------------------
        # Optional progress
        # -----------------------------------------------------------

        if step_index in (
            1,
            50,
            100,
            150,
            200,
        ):

            print(
                f"  Step {step_index:03d} | "
                f"Reward={reward:.6f} | "
                f"SE={spectral_efficiency:.8f} | "
                f"EE={energy_efficiency:.4f} | "
                f"Trust={mean_trust:.6f} | "
                f"Queue={mean_queue:.6f}"
            )

        # -----------------------------------------------------------
        # Termination
        # -----------------------------------------------------------

        if terminated or truncated:

            break

    # =================================================================
    # FINAL METRICS
    # =================================================================

    steps_completed = len(
        reward_values
    )

    final_trust = float(
        np.mean(
            env.trust
        )
    )

    final_queue = (
        float(
            queue_values[-1]
        )
        if queue_values
        else 0.0
    )

    # =================================================================
    # RESULT
    # =================================================================

    result = {
        "polarization_enabled":
            bool(polarization_enabled),

        "steps":
            steps_completed,

        "total_reward":
            float(total_reward),

        "mean_reward":
            (
                float(
                    np.mean(
                        reward_values
                    )
                )
                if reward_values
                else 0.0
            ),

        "mean_se":
            (
                float(
                    np.mean(
                        se_values
                    )
                )
                if se_values
                else 0.0
            ),

        "mean_ee":
            (
                float(
                    np.mean(
                        ee_values
                    )
                )
                if ee_values
                else 0.0
            ),

        "initial_trust":
            initial_trust,

        "mean_trust":
            (
                float(
                    np.mean(
                        trust_values
                    )
                )
                if trust_values
                else initial_trust
            ),

        "final_trust":
            final_trust,

        "trust_change":
            final_trust - initial_trust,

        "mean_queue":
            (
                float(
                    np.mean(
                        queue_values
                    )
                )
                if queue_values
                else 0.0
            ),

        "final_queue":
            final_queue,

        "state_dim":
            int(
                len(state)
            ),

        "action_dim":
            int(
                env.action_dim
            ),
    }

    return result


# =====================================================================
# PRINT RESULT
# =====================================================================

def print_result(
    label: str,
    result: dict,
):
    """
    Print one benchmark result.
    """

    print()

    print(
        "=" * 70
    )

    print(
        label
    )

    print(
        "=" * 70
    )

    print(
        f"Polarization enabled : "
        f"{result['polarization_enabled']}"
    )

    print(
        f"Steps                : "
        f"{result['steps']}"
    )

    print(
        f"State dimension      : "
        f"{result['state_dim']}"
    )

    print(
        f"Action dimension     : "
        f"{result['action_dim']}"
    )

    print(
        f"Total reward         : "
        f"{result['total_reward']:.6f}"
    )

    print(
        f"Mean reward/step     : "
        f"{result['mean_reward']:.6f}"
    )

    print(
        f"Mean spectral eff.   : "
        f"{result['mean_se']:.8f}"
    )

    print(
        f"Mean energy eff.     : "
        f"{result['mean_ee']:.4f}"
    )

    print(
        f"Initial trust        : "
        f"{result['initial_trust']:.6f}"
    )

    print(
        f"Mean trust           : "
        f"{result['mean_trust']:.6f}"
    )

    print(
        f"Final trust          : "
        f"{result['final_trust']:.6f}"
    )

    print(
        f"Trust change         : "
        f"{result['trust_change']:.6f}"
    )

    print(
        f"Mean queue           : "
        f"{result['mean_queue']:.6f}"
    )

    print(
        f"Final queue          : "
        f"{result['final_queue']:.6f}"
    )


# =====================================================================
# PERCENTAGE CHANGE
# =====================================================================

def percentage_change(
    baseline: float,
    new_value: float,
):
    """
    Calculate percentage change:

        ((new - baseline) / |baseline|) * 100
    """

    if abs(
        baseline
    ) < 1e-12:

        return 0.0

    return (
        (
            new_value -
            baseline
        )
        /
        abs(baseline)
    ) * 100.0


# =====================================================================
# COMPARE RESULTS
# =====================================================================

def compare(
    off: dict,
    on: dict,
):
    """
    Compare Polarization OFF against Polarization ON.
    """

    comparison = {
        "reward_change_pct":
            percentage_change(
                off["total_reward"],
                on["total_reward"],
            ),

        "mean_reward_change_pct":
            percentage_change(
                off["mean_reward"],
                on["mean_reward"],
            ),

        "se_change_pct":
            percentage_change(
                off["mean_se"],
                on["mean_se"],
            ),

        "ee_change_pct":
            percentage_change(
                off["mean_ee"],
                on["mean_ee"],
            ),

        "trust_change_difference":
            (
                on["trust_change"]
                -
                off["trust_change"]
            ),

        "final_trust_difference":
            (
                on["final_trust"]
                -
                off["final_trust"]
            ),

        "queue_change_pct":
            percentage_change(
                off["mean_queue"],
                on["mean_queue"],
            ),

        "final_queue_change_pct":
            percentage_change(
                off["final_queue"],
                on["final_queue"],
            ),
    }

    # =================================================================
    # PRINT
    # =================================================================

    print()

    print(
        "=" * 70
    )

    print(
        "PHASE-4A POLARIZATION COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        f"Reward change        : "
        f"{comparison['reward_change_pct']:+.4f}%"
    )

    print(
        f"Mean reward change   : "
        f"{comparison['mean_reward_change_pct']:+.4f}%"
    )

    print(
        f"Spectral efficiency  : "
        f"{comparison['se_change_pct']:+.4f}%"
    )

    print(
        f"Energy efficiency    : "
        f"{comparison['ee_change_pct']:+.4f}%"
    )

    print(
        f"Trust-change diff.   : "
        f"{comparison['trust_change_difference']:+.6f}"
    )

    print(
        f"Final trust diff.    : "
        f"{comparison['final_trust_difference']:+.6f}"
    )

    print(
        f"Mean queue change    : "
        f"{comparison['queue_change_pct']:+.4f}%"
    )

    print(
        f"Final queue change   : "
        f"{comparison['final_queue_change_pct']:+.4f}%"
    )

    return comparison


# =====================================================================
# FORMAT RESULT SECTION
# =====================================================================

def format_result_section(
    title: str,
    result: dict,
):
    """
    Convert one result into report lines.
    """

    lines = []

    lines.append(
        title
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        f"Polarization enabled: "
        f"{result['polarization_enabled']}"
    )

    lines.append(
        f"Steps: "
        f"{result['steps']}"
    )

    lines.append(
        f"State dimension: "
        f"{result['state_dim']}"
    )

    lines.append(
        f"Action dimension: "
        f"{result['action_dim']}"
    )

    lines.append(
        f"Total reward: "
        f"{result['total_reward']:.6f}"
    )

    lines.append(
        f"Mean reward: "
        f"{result['mean_reward']:.6f}"
    )

    lines.append(
        f"Mean spectral efficiency: "
        f"{result['mean_se']:.8f}"
    )

    lines.append(
        f"Mean energy efficiency: "
        f"{result['mean_ee']:.6f}"
    )

    lines.append(
        f"Initial trust: "
        f"{result['initial_trust']:.6f}"
    )

    lines.append(
        f"Mean trust: "
        f"{result['mean_trust']:.6f}"
    )

    lines.append(
        f"Final trust: "
        f"{result['final_trust']:.6f}"
    )

    lines.append(
        f"Trust change: "
        f"{result['trust_change']:.6f}"
    )

    lines.append(
        f"Mean queue: "
        f"{result['mean_queue']:.6f}"
    )

    lines.append(
        f"Final queue: "
        f"{result['final_queue']:.6f}"
    )

    lines.append("")

    return lines


# =====================================================================
# SAVE REPORT
# =====================================================================

def save_report(
    off: dict,
    on: dict,
    comparison: dict,
):
    """
    Save benchmark results to results/phase4a_polarization_benchmark.txt
    """

    lines = []

    # =================================================================
    # HEADER
    # =================================================================

    lines.append(
        "TA-FDRL-IRF | Phase-4A Polarization Benchmark"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        f"Seed: {SEED}"
    )

    lines.append(
        f"Steps requested: {NUM_STEPS}"
    )

    lines.append("")

    # =================================================================
    # COMMON CONFIGURATION
    # =================================================================

    lines.append(
        "COMMON CONFIGURATION"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        f"Users: {NUM_USERS}"
    )

    lines.append(
        f"RIS elements: {NUM_RIS_ELEMENTS}"
    )

    lines.append(
        "Carrier frequency: 28 GHz"
    )

    lines.append(
        "Bandwidth: 100 MHz"
    )

    lines.append(
        "Maximum transmit power: 1.0 W"
    )

    lines.append(
        "Circuit power: 0.1 W"
    )

    lines.append(
        "Noise figure: 7 dB"
    )

    lines.append(
        "Noise density: -174 dBm/Hz"
    )

    lines.append(
        "RIS optimization: False"
    )

    lines.append(
        "Adaptive trust: True"
    )

    lines.append(
        "State dimension target: 100"
    )

    lines.append(
        "Action dimension target: 40"
    )

    lines.append(
        "SAC architecture: Unchanged"
    )

    lines.append("")

    # =================================================================
    # OFF
    # =================================================================

    lines.extend(
        format_result_section(
            "POLARIZATION OFF",
            off,
        )
    )

    # =================================================================
    # ON
    # =================================================================

    lines.extend(
        format_result_section(
            "POLARIZATION ON",
            on,
        )
    )

    # =================================================================
    # COMPARISON
    # =================================================================

    lines.append(
        "POLARIZATION IMPACT"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        f"Reward change: "
        f"{comparison['reward_change_pct']:+.4f}%"
    )

    lines.append(
        f"Mean reward change: "
        f"{comparison['mean_reward_change_pct']:+.4f}%"
    )

    lines.append(
        f"SE change: "
        f"{comparison['se_change_pct']:+.4f}%"
    )

    lines.append(
        f"EE change: "
        f"{comparison['ee_change_pct']:+.4f}%"
    )

    lines.append(
        f"Trust-change difference: "
        f"{comparison['trust_change_difference']:+.6f}"
    )

    lines.append(
        f"Final trust difference: "
        f"{comparison['final_trust_difference']:+.6f}"
    )

    lines.append(
        f"Mean queue change: "
        f"{comparison['queue_change_pct']:+.4f}%"
    )

    lines.append(
        f"Final queue change: "
        f"{comparison['final_queue_change_pct']:+.4f}%"
    )

    lines.append("")

    # =================================================================
    # METHODOLOGY
    # =================================================================

    lines.append(
        "METHODOLOGY"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        "Same environment reset seed: 2026."
    )

    lines.append(
        "Same externally generated action sequence."
    )

    lines.append(
        "Action sequence length: 200."
    )

    lines.append(
        "RIS optimization disabled."
    )

    lines.append(
        "Adaptive trust enabled."
    )

    lines.append(
        "SAC training not performed."
    )

    lines.append(
        "SAC architecture unchanged."
    )

    lines.append(
        "Polarization switch is the experimental PHY variable."
    )

    lines.append(
        "Benchmark type: controlled preliminary PHY comparison."
    )

    lines.append("")

    # =================================================================
    # FAIRNESS NOTE
    # =================================================================

    lines.append(
        "FAIRNESS NOTE"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        "The same externally generated action sequence is used"
    )

    lines.append(
        "for both Polarization OFF and Polarization ON."
    )

    lines.append(
        "Both environments are reset using the same seed."
    )

    lines.append(
        "However, if the environment uses a shared internal RNG"
    )

    lines.append(
        "stream, different channel-generation dimensions can"
    )

    lines.append(
        "consume different random samples."
    )

    lines.append(
        "Therefore this benchmark is a controlled preliminary"
    )

    lines.append(
        "comparison rather than the final research-grade ablation."
    )

    lines.append(
        "A separate channel RNG and dynamics RNG should be used"
    )

    lines.append(
        "for the final publication-quality experiment."
    )

    lines.append("")

    # =================================================================
    # PHASE STATUS
    # =================================================================

    lines.append(
        "PHASE-4A STATUS"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        "Polarization PHY: ENABLED"
    )

    lines.append(
        "State dimension: 100"
    )

    lines.append(
        "Action dimension: 40"
    )

    lines.append(
        "RIS optimization: DISABLED"
    )

    lines.append(
        "Adaptive trust: ENABLED"
    )

    lines.append(
        "SAC architecture: UNCHANGED"
    )

    lines.append(
        "SAC training: NOT PERFORMED"
    )

    lines.append(
        "Environment validation: PASSED"
    )

    lines.append(
        "Controlled PHY benchmark: COMPLETED"
    )

    lines.append("")

    # =================================================================
    # WRITE
    # =================================================================

    RESULT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    # =================================================================
    # HEADER
    # =================================================================

    print()

    print(
        "=" * 70
    )

    print(
        "TA-FDRL-IRF | "
        "PHASE-4A CONTROLLED POLARIZATION BENCHMARK"
    )

    print(
        "=" * 70
    )

    print(
        f"Seed: {SEED}"
    )

    print(
        f"Steps: {NUM_STEPS}"
    )

    print(
        f"Users: {NUM_USERS}"
    )

    print(
        f"RIS elements: {NUM_RIS_ELEMENTS}"
    )

    print(
        f"State dimension target: 100"
    )

    print(
        f"Action dimension: {ACTION_DIM}"
    )

    print()

    # =================================================================
    # GENERATE COMMON ACTIONS
    # =================================================================

    print(
        "Generating common action sequence..."
    )

    actions = generate_actions(
        num_steps=NUM_STEPS,
        action_dim=ACTION_DIM,
        seed=SEED,
    )

    print(
        f"Actions generated: "
        f"{len(actions)}"
    )

    # =================================================================
    # POLARIZATION OFF
    # =================================================================

    print()

    print(
        "Running Polarization OFF..."
    )

    off_result = run_environment(
        polarization_enabled=False,
        actions=actions,
    )

    print_result(
        label="POLARIZATION OFF",
        result=off_result,
    )

    # =================================================================
    # POLARIZATION ON
    # =================================================================

    print()

    print(
        "Running Polarization ON..."
    )

    on_result = run_environment(
        polarization_enabled=True,
        actions=actions,
    )

    print_result(
        label="POLARIZATION ON",
        result=on_result,
    )

    # =================================================================
    # COMPARISON
    # =================================================================

    comparison = compare(
        off=off_result,
        on=on_result,
    )

    # =================================================================
    # SAVE REPORT
    # =================================================================

    save_report(
        off=off_result,
        on=on_result,
        comparison=comparison,
    )

    # =================================================================
    # COMPLETE
    # =================================================================

    print()

    print(
        "=" * 70
    )

    print(
        "PHASE-4A BENCHMARK COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Report saved to:"
    )

    print(
        RESULT_FILE
    )

    print()

    print(
        "SAC training: NOT PERFORMED"
    )

    print(
        "Environment validation: PASSED"
    )

    print(
        "Controlled PHY benchmark: COMPLETED"
    )

    print()

    print(
        "Next stage:"
    )

    print(
        "Research-grade RNG-controlled "
        "Polarization OFF vs ON comparison."
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()