
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------
# governed_sac_demo.py is located in:
#
#     TA-FDRL-IRF/
#         experiments/
#             governed_sac_demo.py
#
# Add TA-FDRL-IRF root to sys.path so that:
#     agents/
#     environment/
#     trust/
# can be imported correctly.
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------

from agents.sac_agent import SACAgent
from environment.irf_env import IRFEnvironment
from trust.governance import GovernanceEngine


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

STATE_DIM = 100
ACTION_DIM = 40

MODEL_PATH = PROJECT_ROOT / "results" / "best_sac_irf_phase2.pt"

MAX_STEPS = 20


# ------------------------------------------------------------------
# Telemetry extraction
# ------------------------------------------------------------------

def extract_telemetry(state):
    """
    Extract telemetry from the 100-dimensional IRF state.

    State layout:

        [SINR, interference, queue, power, trust] x 20 users

    Therefore:

        20 users x 5 features = 100 state dimensions
    """

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    if state.shape != (STATE_DIM,):
        raise ValueError(
            f"Expected state shape {(STATE_DIM,)}, "
            f"got {state.shape}"
        )

    features = state.reshape(20, 5)

    return {
        "sinr": features[:, 0],
        "interference": features[:, 1],
        "queue": features[:, 2],
        "power": features[:, 3],
        "trust": features[:, 4],
    }


# ------------------------------------------------------------------
# Main demo
# ------------------------------------------------------------------

def main():

    print("=" * 70)
    print("TA-FDRL-IRF GOVERNED SAC DEMO")
    print("=" * 70)

    # --------------------------------------------------------------
    # Environment
    # --------------------------------------------------------------

    print("\n[1/5] Initializing IRF environment...")

    env = IRFEnvironment()

    print("Environment initialized.")

    # --------------------------------------------------------------
    # SAC Agent
    # --------------------------------------------------------------

    print("\n[2/5] Initializing SAC agent...")

    agent = SACAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
    )

    # --------------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------------

    print("\n[3/5] Loading trained SAC model...")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"SAC model not found:\n{MODEL_PATH}"
        )

    agent.load(str(MODEL_PATH))

    print(f"SAC model loaded:")
    print(f"  {MODEL_PATH}")

    # --------------------------------------------------------------
    # Governance
    # --------------------------------------------------------------

    print("\n[4/5] Initializing governance layer...")

    governance = GovernanceEngine()

    print("GovernanceEngine initialized.")

    # --------------------------------------------------------------
    # Environment reset
    # --------------------------------------------------------------

    print("\n[5/5] Resetting environment...")

    reset_result = env.reset()

    if isinstance(reset_result, tuple):

        state = reset_result[0]

    else:

        state = reset_result

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    print(f"Initial state dimension: {state.shape}")

    if state.shape != (STATE_DIM,):

        raise ValueError(
            f"Expected initial state shape {(STATE_DIM,)}, "
            f"got {state.shape}"
        )

    # --------------------------------------------------------------
    # Evaluation statistics
    # --------------------------------------------------------------

    total_reward = 0.0

    allow_count = 0
    constrain_count = 0
    block_count = 0

    rewards = []

    # --------------------------------------------------------------
    # Governed SAC evaluation loop
    # --------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STARTING GOVERNED SAC EVALUATION")
    print("=" * 70)

    for step in range(MAX_STEPS):

        print("\n" + "-" * 70)
        print(f"STEP {step + 1}/{MAX_STEPS}")
        print("-" * 70)

        # ==========================================================
        # 1. SAC proposes an action
        # ==========================================================

        proposed_action = agent.select_action(
            state,
            evaluate=True,
        )

        proposed_action = np.asarray(
            proposed_action,
            dtype=np.float32,
        )

        print(
            f"Proposed action dimension: "
            f"{proposed_action.shape}"
        )

        # ==========================================================
        # 2. Extract current telemetry
        # ==========================================================

        telemetry = extract_telemetry(state)

        # ==========================================================
        # 3. Calculate aggregate trust
        # ==========================================================

        trust_values = np.asarray(
            telemetry["trust"],
            dtype=np.float32,
        )

        trust_score = float(
            np.clip(
                np.mean(trust_values),
                0.0,
                1.0,
            )
        )

        # ==========================================================
        # 4. Governance evaluation
        # ==========================================================

        decision = governance.evaluate(
            action=proposed_action,
            trust_score=trust_score,
            telemetry=telemetry,
        )

        # ==========================================================
        # 5. Display governance decision
        # ==========================================================

        print(f"Trust score:        {trust_score:.4f}")
        print(
            f"Risk score:         "
            f"{decision.risk.risk_score:.4f}"
        )
        print(
            f"Risk level:         "
            f"{decision.risk.risk_level}"
        )
        print(
            f"Policy allowed:     "
            f"{decision.policy.allowed}"
        )
        print(
            f"Governance status:  "
            f"{decision.status}"
        )
        print(
            f"Governance reason:  "
            f"{decision.reason}"
        )
        print(
            f"Action modified:    "
            f"{decision.modified}"
        )

        # ==========================================================
        # Risk reasons
        # ==========================================================

        if decision.risk.reasons:

            print(
                "Risk reasons:       "
                + ", ".join(
                    decision.risk.reasons
                )
            )

        # ==========================================================
        # Policy violations
        # ==========================================================

        if decision.policy.violations:

            print(
                "Policy violations:  "
                + ", ".join(
                    decision.policy.violations
                )
            )

        # ==========================================================
        # 6. Select approved action
        # ==========================================================

        if decision.status == "ALLOW":

            allow_count += 1

            approved_action = decision.action

            print(
                "Action execution:   ALLOWED"
            )

        elif decision.status == "CONSTRAIN":

            constrain_count += 1

            approved_action = decision.action

            print(
                "Action execution:   CONSTRAINED"
            )

        elif decision.status == "BLOCK":

            block_count += 1

            print(
                "Action execution:   BLOCKED"
            )

            # ------------------------------------------------------
            # Safe fallback
            # ------------------------------------------------------
            # A blocked action is replaced by a neutral action.
            #
            # IMPORTANT:
            # This is a governance-demo fallback and is NOT part
            # of the original SAC scientific benchmark.
            # ------------------------------------------------------

            approved_action = np.zeros(
                ACTION_DIM,
                dtype=np.float32,
            )

        else:

            raise RuntimeError(
                f"Unknown governance status: "
                f"{decision.status}"
            )

        # ==========================================================
        # 7. Validate approved action
        # ==========================================================

        approved_action = np.asarray(
            approved_action,
            dtype=np.float32,
        )

        if approved_action.shape != (ACTION_DIM,):

            raise ValueError(
                "Governance returned an invalid action shape: "
                f"{approved_action.shape}"
            )

        if not np.all(
            np.isfinite(approved_action)
        ):

            raise ValueError(
                "Governance returned a non-finite action."
            )

        # ==========================================================
        # 8. Execute approved action in IRF environment
        # ==========================================================

        step_result = env.step(
            approved_action
        )

        # ==========================================================
        # Support Gym / Gymnasium style APIs
        # ==========================================================

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

        # ==========================================================
        # 9. Normalize next state
        # ==========================================================

        next_state = np.asarray(
            next_state,
            dtype=np.float32,
        )

        if next_state.shape != (STATE_DIM,):

            raise ValueError(
                f"Expected next state shape "
                f"{(STATE_DIM,)}, "
                f"got {next_state.shape}"
            )

        # ==========================================================
        # 10. Reward
        # ==========================================================

        reward = float(reward)

        rewards.append(reward)

        total_reward += reward

        print(
            f"Reward:             {reward:.6f}"
        )

        print(
            f"Cumulative reward:  "
            f"{total_reward:.6f}"
        )

        # ==========================================================
        # 11. Move to next state
        # ==========================================================

        state = next_state

        # ==========================================================
        # Episode termination
        # ==========================================================

        if done:

            print(
                "\nEnvironment episode finished."
            )

            break

    # ------------------------------------------------------------------
    # Final statistics
    # ------------------------------------------------------------------

    executed_steps = len(rewards)

    mean_reward = (
        float(np.mean(rewards))
        if rewards
        else 0.0
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("GOVERNED SAC DEMO COMPLETE")
    print("=" * 70)

    print()
    print("MODEL")
    print("-" * 70)
    print(f"SAC model:          {MODEL_PATH}")

    print()
    print("EVALUATION")
    print("-" * 70)
    print(f"State dimension:    {STATE_DIM}")
    print(f"Action dimension:   {ACTION_DIM}")
    print(f"Steps executed:     {executed_steps}")

    print()
    print("GOVERNANCE")
    print("-" * 70)
    print(f"ALLOW decisions:    {allow_count}")
    print(f"CONSTRAIN decisions: {constrain_count}")
    print(f"BLOCK decisions:     {block_count}")

    print()
    print("PERFORMANCE")
    print("-" * 70)
    print(f"Total reward:       {total_reward:.6f}")
    print(f"Mean reward:        {mean_reward:.6f}")

    print()
    print("=" * 70)
    print("SAC -> GOVERNANCE -> IRF CLOSED LOOP EXECUTED")
    print("=" * 70)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()

