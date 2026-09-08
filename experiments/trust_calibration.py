
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from environment.irf_env import IRFEnvironment


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

STATE_DIM = 100
NUM_USERS = 20
TRUST_FEATURE_INDEX = 4

NUM_STEPS = 200
NUM_EPISODES = 5


# ------------------------------------------------------------------
# Extract trust from state
# ------------------------------------------------------------------

def extract_trust(state):

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    if state.shape != (STATE_DIM,):
        raise ValueError(
            f"Expected state shape {(STATE_DIM,)}, "
            f"got {state.shape}"
        )

    features = state.reshape(NUM_USERS, 5)

    trust = features[:, TRUST_FEATURE_INDEX]

    return trust


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    print("=" * 70)
    print("TA-FDRL-IRF TRUST CALIBRATION")
    print("=" * 70)

    all_trust_values = []

    episode_means = []

    # --------------------------------------------------------------
    # Collect environment trust
    # --------------------------------------------------------------

    for episode in range(NUM_EPISODES):

        print("\n" + "-" * 70)
        print(f"EPISODE {episode + 1}/{NUM_EPISODES}")
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

        for step in range(NUM_STEPS):

            trust = extract_trust(state)

            mean_trust = float(
                np.mean(trust)
            )

            episode_trust.extend(
                trust.tolist()
            )

            all_trust_values.extend(
                trust.tolist()
            )

            # ------------------------------------------------------
            # Neutral action
            # ------------------------------------------------------

            action = np.zeros(
                40,
                dtype=np.float32,
            )

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

            state = np.asarray(
                next_state,
                dtype=np.float32,
            )

            if done:
                break

        episode_mean = float(
            np.mean(episode_trust)
        )

        episode_means.append(
            episode_mean
        )

        print(
            f"Episode mean trust: "
            f"{episode_mean:.6f}"
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    trust = np.asarray(
        all_trust_values,
        dtype=np.float32,
    )

    print("\n")
    print("=" * 70)
    print("TRUST DISTRIBUTION")
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

    print("\n")
    print("=" * 70)
    print("EPISODE TRUST MEANS")
    print("=" * 70)

    for index, value in enumerate(
        episode_means,
        start=1,
    ):

        print(
            f"Episode {index}:       "
            f"{value:.6f}"
        )

    # ------------------------------------------------------------------
    # Governance threshold analysis
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("GOVERNANCE THRESHOLD ANALYSIS")
    print("=" * 70)

    thresholds = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
    ]

    for threshold in thresholds:

        fraction_below = float(
            np.mean(
                trust < threshold
            )
        )

        fraction_above = 1.0 - fraction_below

        print(
            f"Threshold {threshold:.2f} | "
            f"Below: {fraction_below * 100:6.2f}% | "
            f"Above: {fraction_above * 100:6.2f}%"
        )

    # ------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TRUST CALIBRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

