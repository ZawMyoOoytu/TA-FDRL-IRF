
import numpy as np

from environment.irf_env import (
    IRFConfig,
    IRFEnvironment,
)


def main():

    # =====================================================
    # Phase-3 configuration
    # =====================================================

    config = IRFConfig(

        num_users=20,

        num_ris_elements=64,

        bandwidth_hz=100e6,

        carrier_frequency_hz=28e9,

        max_power_w=1.0,

        circuit_power_w=0.1,

        max_steps=10,

        # -------------------------------------------------
        # Phase-3
        # -------------------------------------------------

        # RIS remains fixed for this experiment.
        optimize_ris=False,

        # IMPORTANT:
        # False = adaptive trust
        fixed_trust=False,

        seed=42,
    )

    # =====================================================
    # Environment
    # =====================================================

    env = IRFEnvironment(config)

    state = env.reset()

    # Save initial trust for comparison.
    initial_trust = env.trust.copy()

    # =====================================================
    # Header
    # =====================================================

    print(
        "State dimension:",
        env.state_dim,
    )

    print(
        "Action dimension:",
        env.action_dim,
    )

    print(
        "Initial state shape:",
        state.shape,
    )

    print(
        "RIS optimization:",
        config.optimize_ris,
    )

    print(
        "Fixed trust:",
        config.fixed_trust,
    )

    print(
        "Initial mean trust:",
        f"{np.mean(initial_trust):.4f}",
    )

    print(
        "Initial min trust:",
        f"{np.min(initial_trust):.4f}",
    )

    print(
        "Initial max trust:",
        f"{np.max(initial_trust):.4f}",
    )

    print("-" * 70)

    # =====================================================
    # Test loop
    # =====================================================

    total_reward = 0.0

    next_state = state.copy()

    for step in range(
        config.max_steps
    ):

        # -------------------------------------------------
        # Random action
        # -------------------------------------------------

        action = np.random.uniform(
            -1.0,
            1.0,
            size=env.action_dim,
        )

        (
            next_state,
            reward,
            done,
            info,
        ) = env.step(
            action
        )

        total_reward += reward

        # -------------------------------------------------
        # Trust statistics
        # -------------------------------------------------

        mean_trust = info["mean_trust"]

        min_trust = info["min_trust"]

        max_trust = info["max_trust"]

        # -------------------------------------------------
        # Trust change from initial state
        # -------------------------------------------------

        trust_change = (
            mean_trust
            - float(np.mean(initial_trust))
        )

        # -------------------------------------------------
        # Print
        # -------------------------------------------------

        print(
            f"Step {step + 1:02d} | "
            f"Reward={reward: .4f} | "
            f"SE={info['spectral_efficiency']:.4f} | "
            f"EE={info['energy_efficiency']:.4e} | "
            f"Trust={mean_trust:.4f} | "
            f"TrustRange=[{min_trust:.4f}, {max_trust:.4f}] | "
            f"dTrust={trust_change:+.4f} | "
            f"Queue={info['mean_queue']:.4f}"
        )

        if done:
            break

    # =====================================================
    # Summary
    # =====================================================

    final_trust = float(
        info["mean_trust"]
    )

    initial_mean_trust = float(
        np.mean(initial_trust)
    )

    total_trust_change = (
        final_trust
        - initial_mean_trust
    )

    print("-" * 70)

    print(
        "Total reward:",
        f"{total_reward:.6f}",
    )

    print(
        "Final state shape:",
        next_state.shape,
    )

    print(
        "Initial mean trust:",
        f"{initial_mean_trust:.6f}",
    )

    print(
        "Final mean trust:",
        f"{final_trust:.6f}",
    )

    print(
        "Total trust change:",
        f"{total_trust_change:+.6f}",
    )

    print(
        "Final minimum trust:",
        f"{info['min_trust']:.6f}",
    )

    print(
        "Final maximum trust:",
        f"{info['max_trust']:.6f}",
    )


if __name__ == "__main__":
    main()

