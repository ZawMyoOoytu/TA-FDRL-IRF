
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
from trust.risk import RiskEngine


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

STATE_DIM = 100
ACTION_DIM = 40

NUM_USERS = 20
FEATURES_PER_USER = 5

SINR_INDEX = 0
INTERFERENCE_INDEX = 1
QUEUE_INDEX = 2
POWER_INDEX = 3
TRUST_INDEX = 4

NUM_EPISODES = 5
MAX_STEPS = 200

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "best_sac_irf_phase2.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "trust_risk_analysis.txt"
)


# ---------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------

def extract_telemetry(state):
    """
    Decode the 100-dimensional IRF state.

    Per-user feature order:

        [SINR, interference, queue, power, trust]

    20 users x 5 features = 100 dimensions.
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

    return {
        "sinr": features[:, SINR_INDEX].copy(),
        "interference": features[:, INTERFERENCE_INDEX].copy(),
        "queue": features[:, QUEUE_INDEX].copy(),
        "power": features[:, POWER_INDEX].copy(),
        "trust": features[:, TRUST_INDEX].copy(),
    }


# ---------------------------------------------------------------------
# Correlation helper
# ---------------------------------------------------------------------

def safe_correlation(x, y):
    """
    Pearson correlation with protection against
    constant / invalid arrays.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if x.size < 2:
        return np.nan

    if np.std(x) == 0.0:
        return np.nan

    if np.std(y) == 0.0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("TA-FDRL-IRF TRUST-RISK ANALYSIS")
    print("=" * 70)

    # -------------------------------------------------------------
    # Validate model
    # -------------------------------------------------------------

    print("\nMODEL")
    print("-" * 70)
    print(f"Model path: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"SAC model not found: {MODEL_PATH}"
        )

    # -------------------------------------------------------------
    # SAC
    # -------------------------------------------------------------

    print("\nInitializing SAC agent...")

    agent = SACAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
    )

    agent.load(str(MODEL_PATH))

    print("SAC model loaded successfully.")

    # -------------------------------------------------------------
    # Risk engine
    # -------------------------------------------------------------

    risk_engine = RiskEngine()

    print("Risk engine initialized.")

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    records = []

    episode_summaries = []

    # -------------------------------------------------------------
    # Run episodes
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

        episode_rewards = []
        episode_trust = []
        episode_risk = []
        episode_anomaly = []

        for step in range(MAX_STEPS):

            # -----------------------------------------------------
            # Telemetry BEFORE action
            # -----------------------------------------------------

            telemetry = extract_telemetry(state)

            trust_values = telemetry["trust"]

            mean_trust = float(
                np.mean(trust_values)
            )

            min_trust = float(
                np.min(trust_values)
            )

            p10_trust = float(
                np.percentile(
                    trust_values,
                    10,
                )
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
                    f"Expected action shape "
                    f"{(ACTION_DIM,)}, "
                    f"got {action.shape}"
                )

            # -----------------------------------------------------
            # Risk assessment
            # -----------------------------------------------------

            risk = risk_engine.evaluate(
                action,
                trust_score=mean_trust,
                telemetry=telemetry,
            )

            # -----------------------------------------------------
            # Environment execution
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

            reward = float(reward)

            # -----------------------------------------------------
            # Record timestep
            # -----------------------------------------------------

            records.append(
                {
                    "episode": episode + 1,
                    "step": step + 1,
                    "mean_trust": mean_trust,
                    "min_trust": min_trust,
                    "p10_trust": p10_trust,
                    "risk_score": float(
                        risk.risk_score
                    ),
                    "action_anomaly": float(
                        risk.action_anomaly
                    ),
                    "reward": reward,
                    "mean_sinr": float(
                        np.mean(
                            telemetry["sinr"]
                        )
                    ),
                    "mean_interference": float(
                        np.mean(
                            np.abs(
                                telemetry[
                                    "interference"
                                ]
                            )
                        )
                    ),
                    "mean_queue": float(
                        np.mean(
                            telemetry["queue"]
                        )
                    ),
                    "mean_power": float(
                        np.mean(
                            telemetry["power"]
                        )
                    ),
                    "risk_level": risk.risk_level,
                }
            )

            episode_rewards.append(
                reward
            )

            episode_trust.append(
                mean_trust
            )

            episode_risk.append(
                float(risk.risk_score)
            )

            episode_anomaly.append(
                float(risk.action_anomaly)
            )

            # -----------------------------------------------------
            # Next state
            # -----------------------------------------------------

            state = np.asarray(
                next_state,
                dtype=np.float32,
            )

            if done:
                break

        # ---------------------------------------------------------
        # Episode summary
        # ---------------------------------------------------------

        episode_summaries.append(
            {
                "episode": episode + 1,
                "mean_trust": float(
                    np.mean(episode_trust)
                ),
                "mean_risk": float(
                    np.mean(episode_risk)
                ),
                "mean_anomaly": float(
                    np.mean(episode_anomaly)
                ),
                "total_reward": float(
                    np.sum(episode_rewards)
                ),
            }
        )

        print(
            f"Mean trust:       "
            f"{np.mean(episode_trust):.6f}"
        )

        print(
            f"Mean risk:        "
            f"{np.mean(episode_risk):.6f}"
        )

        print(
            f"Mean anomaly:     "
            f"{np.mean(episode_anomaly):.6f}"
        )

        print(
            f"Total reward:     "
            f"{np.sum(episode_rewards):.6f}"
        )

    # -----------------------------------------------------------------
    # Convert records
    # -----------------------------------------------------------------

    mean_trust = np.asarray(
        [r["mean_trust"] for r in records]
    )

    min_trust = np.asarray(
        [r["min_trust"] for r in records]
    )

    p10_trust = np.asarray(
        [r["p10_trust"] for r in records]
    )

    risk_score = np.asarray(
        [r["risk_score"] for r in records]
    )

    action_anomaly = np.asarray(
        [r["action_anomaly"] for r in records]
    )

    reward = np.asarray(
        [r["reward"] for r in records]
    )

    mean_sinr = np.asarray(
        [r["mean_sinr"] for r in records]
    )

    mean_interference = np.asarray(
        [r["mean_interference"]
         for r in records]
    )

    mean_queue = np.asarray(
        [r["mean_queue"] for r in records]
    )

    mean_power = np.asarray(
        [r["mean_power"] for r in records]
    )

    # -----------------------------------------------------------------
    # Correlations
    # -----------------------------------------------------------------

    trust_risk_corr = safe_correlation(
        mean_trust,
        risk_score,
    )

    trust_reward_corr = safe_correlation(
        mean_trust,
        reward,
    )

    trust_anomaly_corr = safe_correlation(
        mean_trust,
        action_anomaly,
    )

    trust_queue_corr = safe_correlation(
        mean_trust,
        mean_queue,
    )

    trust_interference_corr = safe_correlation(
        mean_trust,
        mean_interference,
    )

    risk_reward_corr = safe_correlation(
        risk_score,
        reward,
    )

    # -----------------------------------------------------------------
    # Output helper
    # -----------------------------------------------------------------

    output_lines = []

    def out(text=""):
        print(text)
        output_lines.append(text)

    output_lines = []

    # Re-print concise final report into output_lines.

    out("\n")
    out("=" * 70)
    out("TRUST-RISK STATISTICAL ANALYSIS")
    out("=" * 70)

    out(f"Samples:                 {len(records)}")

    out("\nTRUST")
    out("-" * 70)

    out(
        f"Mean trust:              "
        f"{np.mean(mean_trust):.6f}"
    )

    out(
        f"Median trust:            "
        f"{np.median(mean_trust):.6f}"
    )

    out(
        f"Minimum mean-trust:      "
        f"{np.min(mean_trust):.6f}"
    )

    out(
        f"Maximum mean-trust:      "
        f"{np.max(mean_trust):.6f}"
    )

    out(
        f"Mean P10 trust:          "
        f"{np.mean(p10_trust):.6f}"
    )

    out(
        f"Mean minimum trust:      "
        f"{np.mean(min_trust):.6f}"
    )

    out("\nRISK")
    out("-" * 70)

    out(
        f"Mean risk score:         "
        f"{np.mean(risk_score):.6f}"
    )

    out(
        f"Minimum risk score:      "
        f"{np.min(risk_score):.6f}"
    )

    out(
        f"Maximum risk score:      "
        f"{np.max(risk_score):.6f}"
    )

    out(
        f"Mean action anomaly:     "
        f"{np.mean(action_anomaly):.6f}"
    )

    out("\nREWARD")
    out("-" * 70)

    out(
        f"Mean reward/step:        "
        f"{np.mean(reward):.6f}"
    )

    out(
        f"Total reward:            "
        f"{np.sum(reward):.6f}"
    )

    # -----------------------------------------------------------------
    # Correlation table
    # -----------------------------------------------------------------

    out("\n")
    out("=" * 70)
    out("CORRELATION ANALYSIS")
    out("=" * 70)

    out(
        f"Trust vs Risk:           "
        f"{trust_risk_corr:+.6f}"
    )

    out(
        f"Trust vs Reward:         "
        f"{trust_reward_corr:+.6f}"
    )

    out(
        f"Trust vs Action Anomaly: "
        f"{trust_anomaly_corr:+.6f}"
    )

    out(
        f"Trust vs Queue:          "
        f"{trust_queue_corr:+.6f}"
    )

    out(
        f"Trust vs Interference:   "
        f"{trust_interference_corr:+.6f}"
    )

    out(
        f"Risk vs Reward:          "
        f"{risk_reward_corr:+.6f}"
    )

    # -----------------------------------------------------------------
    # Trust regions
    # -----------------------------------------------------------------

    out("\n")
    out("=" * 70)
    out("TRUST REGION ANALYSIS")
    out("=" * 70)

    regions = [
        ("VERY_LOW", 0.00, 0.30),
        ("LOW", 0.30, 0.40),
        ("MEDIUM", 0.40, 0.50),
        ("GOOD", 0.50, 0.70),
        ("HIGH", 0.70, 1.01),
    ]

    for name, lower, upper in regions:

        mask = (
            (mean_trust >= lower)
            & (mean_trust < upper)
        )

        count = int(
            np.sum(mask)
        )

        if count == 0:

            out(
                f"{name:10s} | "
                f"Samples: {count:5d} | "
                f"No observations"
            )

            continue

        out(
            f"{name:10s} | "
            f"Samples: {count:5d} | "
            f"Trust: {np.mean(mean_trust[mask]):.4f} | "
            f"Risk: {np.mean(risk_score[mask]):.4f} | "
            f"Anomaly: {np.mean(action_anomaly[mask]):.4f} | "
            f"Reward: {np.mean(reward[mask]):.4f}"
        )

    # -----------------------------------------------------------------
    # Risk level distribution
    # -----------------------------------------------------------------

    risk_levels = [
        r["risk_level"]
        for r in records
    ]

    low_count = risk_levels.count("LOW")
    medium_count = risk_levels.count("MEDIUM")
    high_count = risk_levels.count("HIGH")

    out("\n")
    out("=" * 70)
    out("RISK LEVEL DISTRIBUTION")
    out("=" * 70)

    out(
        f"LOW:                    "
        f"{low_count:5d} "
        f"({100 * low_count / len(records):.2f}%)"
    )

    out(
        f"MEDIUM:                 "
        f"{medium_count:5d} "
        f"({100 * medium_count / len(records):.2f}%)"
    )

    out(
        f"HIGH:                   "
        f"{high_count:5d} "
        f"({100 * high_count / len(records):.2f}%)"
    )

    # -----------------------------------------------------------------
    # Threshold analysis
    # -----------------------------------------------------------------

    out("\n")
    out("=" * 70)
    out("TRUST THRESHOLD EXPOSURE")
    out("=" * 70)

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

        below = float(
            np.mean(
                mean_trust < threshold
            )
        )

        out(
            f"Threshold {threshold:.2f} | "
            f"Below: {below * 100:6.2f}% | "
            f"Above: {(1.0 - below) * 100:6.2f}%"
        )

    # -----------------------------------------------------------------
    # Episode summary
    # -----------------------------------------------------------------

    out("\n")
    out("=" * 70)
    out("EPISODE SUMMARY")
    out("=" * 70)

    for summary in episode_summaries:

        out(
            f"Episode {summary['episode']}: "
            f"trust={summary['mean_trust']:.6f}, "
            f"risk={summary['mean_risk']:.6f}, "
            f"anomaly={summary['mean_anomaly']:.6f}, "
            f"reward={summary['total_reward']:.6f}"
        )

    # -----------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------

    out("\n")
    out("=" * 70)
    out("SCIENTIFIC INTERPRETATION")
    out("=" * 70)

    out(
        "The experiment measures the empirical relationship "
        "between environment-level trust, SAC actions, "
        "risk assessment, and reward."
    )

    out(
        "The governance trust threshold is NOT changed "
        "by this experiment."
    )

    out(
        "Threshold selection should be based on observed "
        "trust-risk behavior rather than convenience."
    )

    out(
        "The existing SAC scientific benchmark remains "
        "unchanged."
    )

    # -----------------------------------------------------------------
    # Save report
    # -----------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        "\n".join(output_lines),
        encoding="utf-8",
    )

    print("\n")
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(
        f"Report saved to:\n"
        f"{OUTPUT_PATH}"
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()

