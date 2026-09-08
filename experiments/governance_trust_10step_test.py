from __future__ import annotations

import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Project root import path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# TA-FDRL-IRF imports
# ---------------------------------------------------------------------------

from dashboard.live_runner import (
    create_live_session,
    run_step,
    summarize_records,
)


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 78)
    print("TA-FDRL-IRF GOVERNANCE TRUST 10-STEP VALIDATION")
    print("=" * 78)

    # -----------------------------------------------------------------------
    # Create fresh research session
    # -----------------------------------------------------------------------

    env, agent, governance, state = create_live_session(
        seed=42
    )

    records = []

    # -----------------------------------------------------------------------
    # Step-by-step output header
    # -----------------------------------------------------------------------

    print()

    print(
        "STEP | STATUS     | REWARD     | RISK       | "
        "GOV TRUST  | ENV TRUST  | OUTCOME | HISTORY"
    )

    print("-" * 95)

    # -----------------------------------------------------------------------
    # Execute 10 closed-loop governance steps
    # -----------------------------------------------------------------------

    for _ in range(10):

        state, record = run_step(
            env=env,
            agent=agent,
            governance=governance,
            state=state,
        )

        records.append(record)

        print(
            f"{record['step']:04d} | "
            f"{record['governance']:10s} | "
            f"{record['reward']:10.6f} | "
            f"{record['risk_score']:10.6f} | "
            f"{record['governance_trust']:10.6f} | "
            f"{record['environment_trust']:10.6f} | "
            f"{record['outcome_score']:7.3f} | "
            f"{record['outcome_history_length']:7d}"
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    summary = summarize_records(records)

    for key, value in summary.items():
        print(f"{key}: {value}")

    # -----------------------------------------------------------------------
    # Final TrustEngine state
    # -----------------------------------------------------------------------

    print()

    print("=" * 78)
    print("FINAL GOVERNANCE TRUST STATE")
    print("=" * 78)

    final_state = governance.get_trust_state()

    for key, value in final_state.items():
        print(f"{key}: {value}")

    # -----------------------------------------------------------------------
    # Validation checks
    # -----------------------------------------------------------------------

    print()

    print("=" * 78)
    print("VALIDATION CHECKS")
    print("=" * 78)

    checks = {
        "10 steps executed":
            len(records) == 10,

        "Outcome history reached 10":
            final_state.get(
                "outcome_history_length",
                0,
            ) == 10,

        "Governance trust available":
            "trust_score" in final_state,

        "Environment trust available":
            "environment_trust" in final_state,

        "Outcome score available":
            "outcome_score" in final_state,

        "Risk history available":
            final_state.get(
                "risk_history_length",
                0,
            ) >= 10,

        "Policy history available":
            final_state.get(
                "policy_history_length",
                0,
            ) >= 10,
    }

    # -----------------------------------------------------------------------
    # Print validation results
    # -----------------------------------------------------------------------

    all_passed = True

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(
            f"[{status}] {name}"
        )

        if not passed:
            all_passed = False

    # -----------------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------------

    print()

    print("=" * 78)

    if all_passed:

        print(
            "RESULT: 10-STEP GOVERNANCE TRUST "
            "CLOSED-LOOP VALIDATION PASSED"
        )

    else:

        print(
            "RESULT: VALIDATION FAILED"
        )

    print("=" * 78)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()