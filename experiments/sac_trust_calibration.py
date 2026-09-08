
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.sac_agent import SACAgent
from environment.irf_env import IRFEnvironment


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

STATE_DIM = 100
ACTION_DIM = 40

NUM_USERS = 20
FEATURES_PER_USER = 5
TRUST_FEATURE_INDEX = 4

NUM_EPISODES = 5
MAX_STEPS = 200

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "best_sac_irf_phase2.pt"
)


# ---------------------------------------------------------------------
# Trust extraction
# ---------------------------------------------------------------------

def extract_trust(state):
    """
    Extract per-user trust values from the 100-dimensional IRF state.

    State layout per user:
        [SINR, interference, queue, power, trust]

    Therefore:
        20 users x 5 features = 100 dimensions
    """

    state = np.asarray(state, dtype=np.float32)

    expected_shape = (
        NUM_USERS * FEATURES_PER_USER,
    )

    if state.shape != expected_shape:
        raise ValueError(
            f"Expected state shape {expected_shape}, "
            f"got {state.shape}"
        )

    features = state.reshape(
        NUM_USERS,
        FEATURES_PER_USER,
    )

    trust = features[:, TRUST_FEATURE_INDEX]

    return trust


# ---------------------------------------------------------------------
# Main calibration
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("TA-FDRL-IRF SAC TRUST CALIBRATION")
    print("=" * 70)

    print("\nMODEL")
    print("-" * 70)
    print(f"Model path: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"SAC model not found: {MODEL_PATH}"
        )

    # -------------------------------------------------------------
    # Initialize SAC agent
    # -------------------------------------------------------------

    print("\nInitializing SAC agent...")

    agent = SACAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
    )

    agent.load(str(MODEL_PATH))

    print("SAC model loaded successfully.")

    # -------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------

    all_trust_values = []

    episode_mean_trust = []
    episode_min_trust = []
    episode_max_trust = []

    episode_rewards = []

    # -------------------------------------------------------------
    # Episodes
    # -------------------------------------------------------------

    for episode in range(NUM_EPISODES):

        print("\n" + "-" * 70)
        print(
            f"EPISODE {episode + 1}/{NUM_EPISODES}"
        )
        print("-" * 70)

        env = IRFEnvironment()

        reset_result = env.reset()

        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        episode_trust = []

        total_reward = 0.0

        for step in range(MAX_STEPS):

            # -----------------------------------------------------
            # Record trust BEFORE action
            # -----------------------------------------------------

            trust = extract_trust(state)

            episode_trust.extend(
                trust.tolist()
            )

            all_trust_values.extend(
                trust.tolist()
            )

            # -----------------------------------------------------
            # SAC action
            # -----------------------------------------------------

            action = agent.select_action(
                state,
                evaluate=True,
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            )

            if action.shape != (ACTION_DIM,):
                raise ValueError(
                    f"Expected SAC action shape "
                    f"{(ACTION_DIM,)}, got {action.shape}"
                )

            # -----------------------------------------------------
            # Environment step
            # -----------------------------------------------------

            step_result = env.step(action)

            if len(step_result) == 5:

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = step_result

                done = bool(
                    terminated or truncated
                )

            elif len(step_result) == 4:

                (
                    next_state,
                    reward,
                    done,
                    info,
                ) = step_result

                done = bool(done)

            else:

                raise RuntimeError(
                    "Unexpected environment.step() "
                    f"return length: {len(step_result)}"
                )

            total_reward += float(reward)

            state = np.asarray(
                next_state,
                dtype=np.float32,
            )

            if done:
                break

        # ---------------------------------------------------------
        # Episode statistics
        # ---------------------------------------------------------

        episode_trust_array = np.asarray(
            episode_trust,
            dtype=np.float32,
        )

        mean_trust = float(
            np.mean(episode_trust_array)
        )

        min_trust = float(
            np.min(episode_trust_array)
        )

        max_trust = float(
            np.max(episode_trust_array)
        )

        episode_mean_trust.append(
            mean_trust
        )

        episode_min_trust.append(
            min_trust
        )

        episode_max_trust.append(
            max_trust
        )

        episode_rewards.append(
            total_reward
        )

        print(
            f"Mean trust:       {mean_trust:.6f}"
        )

        print(
            f"Minimum trust:    {min_trust:.6f}"
        )

        print(
            f"Maximum trust:    {max_trust:.6f}"
        )

        print(
            f"Total reward:     {total_reward:.6f}"
        )

    # -----------------------------------------------------------------
    # Global distribution
    # -----------------------------------------------------------------

    trust = np.asarray(
        all_trust_values,
        dtype=np.float32,
    )

    print("\n")
    print("=" * 70)
    print("SAC TRUST DISTRIBUTION")
    print("=" * 70)

    print(
        f"Samples:             {trust.size}"
    )

    print(
        f"Minimum:             {np.min(trust):.6f}"
    )

    print(
        f"Maximum:             {np.max(trust):.6f}"
    )

    print(
        f"Mean:                {np.mean(trust):.6f}"
    )

    print(
        f"Median:              {np.median(trust):.6f}"
    )

    print(
        f"Std:                 {np.std(trust):.6f}"
    )

    print(
        f"P10:                 "
        f"{np.percentile(trust, 10):.6f}"
    )

    print(
        f"P25:                 "
        f"{np.percentile(trust, 25):.6f}"
    )

    print(
        f"P50:                 "
        f"{np.percentile(trust, 50):.6f}"
    )

    print(
        f"P75:                 "
        f"{np.percentile(trust, 75):.6f}"
    )

    print(
        f"P90:                 "
        f"{np.percentile(trust, 90):.6f}"
    )

    # -----------------------------------------------------------------
    # Threshold analysis
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SAC TRUST GOVERNANCE THRESHOLD ANALYSIS")
    print("=" * 70)

    thresholds = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    for threshold in thresholds:

        fraction_below = float(
            np.mean(trust < threshold)
        )

        fraction_above = (
            1.0 - fraction_below
        )

        print(
            f"Threshold {threshold:.2f} | "
            f"Below: {fraction_below * 100:6.2f}% | "
            f"Above: {fraction_above * 100:6.2f}%"
        )

    # -----------------------------------------------------------------
    # Episode summary
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EPISODE SUMMARY")
    print("=" * 70)

    for index in range(NUM_EPISODES):

        print(
            f"Episode {index + 1}: "
            f"mean={episode_mean_trust[index]:.6f}, "
            f"min={episode_min_trust[index]:.6f}, "
            f"max={episode_max_trust[index]:.6f}, "
            f"reward={episode_rewards[index]:.6f}"
        )

    # -----------------------------------------------------------------
    # Calibration comparison
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CALIBRATION COMPARISON")
    print("=" * 70)

    print(
        "Environment baseline mean trust: "
        "0.404580"
    )

    print(
        "SAC trajectory mean trust:        "
        f"{np.mean(trust):.6f}"
    )

    difference = (
        float(np.mean(trust))
        - 0.404580
    )

    print(
        "Difference:                       "
        f"{difference:+.6f}"
    )

    # -----------------------------------------------------------------
    # Final
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SAC TRUST CALIBRATION COMPLETE")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do NOT change minimum_trust yet."
    )

    print(
        "Use this distribution to determine "
        "the governance operating threshold."
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()

