
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


from trust.governance import GovernanceEngine


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ACTION_DIM = 40
NUM_USERS = 20


# ---------------------------------------------------------------------
# Controlled input helpers
# ---------------------------------------------------------------------

def uniform_action(value: float) -> np.ndarray:
    return np.full(
        ACTION_DIM,
        value,
        dtype=np.float32,
    )


def policy_violation_action() -> np.ndarray:
    """
    Power action > 0.90.
    This should be blocked by PolicyEngine.
    """

    action = uniform_action(0.0)

    action[20] = 0.95

    return action


def medium_risk_action() -> np.ndarray:
    """
    Action = 0.89.

    This remains within the current power policy limit
    of 0.90, while producing substantial action anomaly.

    Combined with:
        trust = 0.60
        queue = 1.0
        interference = 1.0

    the current RiskEngine should produce MEDIUM risk.
    """

    return uniform_action(0.89)


def high_risk_action() -> np.ndarray:
    """
    High anomaly but still policy-compliant action.

    Used with very low trust and severe telemetry stress
    to reach the HIGH risk branch.
    """

    return uniform_action(0.89)


def zero_action() -> np.ndarray:
    return uniform_action(0.0)


def telemetry(
    queue: float,
    interference: float,
) -> dict:

    return {
        "queue": np.full(
            NUM_USERS,
            queue,
            dtype=np.float32,
        ),
        "interference": np.full(
            NUM_USERS,
            interference,
            dtype=np.float32,
        ),
    }


# ---------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------

def build_scenarios():

    return [

        {
            "name": "NORMAL_ALLOW",

            "trust": 0.90,

            "action": zero_action(),

            "telemetry": telemetry(
                queue=0.10,
                interference=0.10,
            ),

            "expected_status": "ALLOW",

            "expected_reason": "GOVERNANCE_APPROVED",
        },

        {
            "name": "LOW_TRUST_BLOCK",

            "trust": 0.30,

            "action": zero_action(),

            "telemetry": telemetry(
                queue=0.10,
                interference=0.10,
            ),

            "expected_status": "BLOCK",

            "expected_reason": "TRUST_BELOW_MINIMUM",
        },

        {
            "name": "POLICY_VIOLATION_BLOCK",

            "trust": 0.90,

            "action": policy_violation_action(),

            "telemetry": telemetry(
                queue=0.10,
                interference=0.10,
            ),

            "expected_status": "BLOCK",

            "expected_reason": "POLICY_VIOLATION",
        },

        {
            "name": "MEDIUM_RISK_CONSTRAIN",

            "trust": 0.60,

            "action": medium_risk_action(),

            "telemetry": telemetry(
                queue=1.00,
                interference=1.00,
            ),

            "expected_status": "CONSTRAIN",

            "expected_reason": "MEDIUM_RISK_SAFE_ENVELOPE",
        },

        {
            "name": "HIGH_RISK_BLOCK",

            "trust": 0.15,

            "action": high_risk_action(),

            "telemetry": telemetry(
                queue=1.00,
                interference=1.00,
            ),

            "expected_status": "BLOCK",

            "expected_reason": "HIGH_RISK",
        },

        {
            "name": "COMBINED_STRESS_BLOCK",

            "trust": 0.30,

            "action": policy_violation_action(),

            "telemetry": telemetry(
                queue=1.00,
                interference=1.00,
            ),

            "expected_status": "BLOCK",

            "expected_reason": "POLICY_VIOLATION",
        },
    ]


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------

def print_vector_summary(
    name: str,
    action: np.ndarray,
):

    print(
        f"{name}: "
        f"min={np.min(action):.4f}, "
        f"max={np.max(action):.4f}, "
        f"mean={np.mean(action):.4f}"
    )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():

    print("=" * 82)
    print("TA-FDRL-IRF GOVERNANCE BRANCH COVERAGE TEST")
    print("=" * 82)

    print("\nInitializing GovernanceEngine...")

    governance = GovernanceEngine()

    print("GovernanceEngine initialized.")

    scenarios = build_scenarios()

    results = []

    # -----------------------------------------------------------------
    # Execute scenarios
    # -----------------------------------------------------------------

    for index, scenario in enumerate(
        scenarios,
        start=1,
    ):

        print("\n")
        print("-" * 82)
        print(
            f"SCENARIO {index}/{len(scenarios)}: "
            f"{scenario['name']}"
        )
        print("-" * 82)

        action = scenario["action"]

        print(
            f"Trust:             "
            f"{scenario['trust']:.4f}"
        )

        print_vector_summary(
            "Proposed action",
            action,
        )

        decision = governance.evaluate(
            action,
            trust_score=scenario["trust"],
            telemetry=scenario["telemetry"],
        )

        # -------------------------------------------------------------
        # Action comparison
        # -------------------------------------------------------------

        proposed = np.asarray(
            decision.proposed_action,
            dtype=np.float32,
        )

        executed = np.asarray(
            decision.action,
            dtype=np.float32,
        )

        difference = np.abs(
            executed - proposed
        )

        max_difference = float(
            np.max(difference)
        )

        mean_difference = float(
            np.mean(difference)
        )

        # -------------------------------------------------------------
        # Status validation
        # -------------------------------------------------------------

        status_pass = (
            decision.status
            == scenario["expected_status"]
        )

        reason_pass = (
            decision.reason
            == scenario["expected_reason"]
        )

        scenario_pass = (
            status_pass
            and reason_pass
        )

        # -------------------------------------------------------------
        # Output
        # -------------------------------------------------------------

        print(
            f"Risk score:        "
            f"{decision.risk.risk_score:.6f}"
        )

        print(
            f"Risk level:        "
            f"{decision.risk.risk_level}"
        )

        print(
            f"Action anomaly:    "
            f"{decision.risk.action_anomaly:.6f}"
        )

        print(
            f"Policy allowed:    "
            f"{decision.policy.allowed}"
        )

        print(
            f"Policy violations: "
            f"{decision.policy.violations}"
        )

        print(
            f"Risk reasons:      "
            f"{decision.risk.reasons}"
        )

        print(
            f"Expected status:   "
            f"{scenario['expected_status']}"
        )

        print(
            f"Actual status:     "
            f"{decision.status}"
        )

        print(
            f"Expected reason:   "
            f"{scenario['expected_reason']}"
        )

        print(
            f"Actual reason:     "
            f"{decision.reason}"
        )

        print(
            f"Action modified:   "
            f"{decision.modified}"
        )

        print(
            f"Max action delta:  "
            f"{max_difference:.8f}"
        )

        print(
            f"Mean action delta: "
            f"{mean_difference:.8f}"
        )

        print(
            f"STATUS CHECK:      "
            f"{'PASS' if status_pass else 'FAIL'}"
        )

        print(
            f"REASON CHECK:      "
            f"{'PASS' if reason_pass else 'FAIL'}"
        )

        print(
            f"SCENARIO RESULT:   "
            f"{'PASS' if scenario_pass else 'FAIL'}"
        )

        # -------------------------------------------------------------
        # Store result
        # -------------------------------------------------------------

        results.append(
            {
                "name": scenario["name"],
                "trust": float(scenario["trust"]),
                "risk": float(
                    decision.risk.risk_score
                ),
                "risk_level": decision.risk.risk_level,
                "expected_status": scenario[
                    "expected_status"
                ],
                "actual_status": decision.status,
                "expected_reason": scenario[
                    "expected_reason"
                ],
                "actual_reason": decision.reason,
                "modified": bool(
                    decision.modified
                ),
                "max_delta": max_difference,
                "mean_delta": mean_difference,
                "status_pass": status_pass,
                "reason_pass": reason_pass,
                "passed": scenario_pass,
            }
        )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    passed = sum(
        result["passed"]
        for result in results
    )

    failed = (
        len(results)
        - passed
    )

    print("\n")
    print("=" * 82)
    print("BRANCH COVERAGE SUMMARY")
    print("=" * 82)

    print(
        f"Scenarios:          {len(results)}"
    )

    print(
        f"Passed:             {passed}"
    )

    print(
        f"Failed:             {failed}"
    )

    print(
        f"Pass rate:          "
        f"{100.0 * passed / len(results):.2f}%"
    )

    # -----------------------------------------------------------------
    # Decision distribution
    # -----------------------------------------------------------------

    allow_count = sum(
        r["actual_status"] == "ALLOW"
        for r in results
    )

    constrain_count = sum(
        r["actual_status"] == "CONSTRAIN"
        for r in results
    )

    block_count = sum(
        r["actual_status"] == "BLOCK"
        for r in results
    )

    print("\n")
    print("=" * 82)
    print("DECISION BRANCH COVERAGE")
    print("=" * 82)

    print(
        f"ALLOW:             "
        f"{allow_count}"
    )

    print(
        f"CONSTRAIN:         "
        f"{constrain_count}"
    )

    print(
        f"BLOCK:             "
        f"{block_count}"
    )

    # -----------------------------------------------------------------
    # CONSTRAIN-specific analysis
    # -----------------------------------------------------------------

    constrain_results = [
        r
        for r in results
        if r["actual_status"] == "CONSTRAIN"
    ]

    print("\n")
    print("=" * 82)
    print("CONSTRAIN BRANCH ANALYSIS")
    print("=" * 82)

    if not constrain_results:

        print(
            "CONSTRAIN branch was NOT reached."
        )

        print(
            "No action-transformation analysis "
            "can be performed."
        )

    else:

        print(
            f"CONSTRAIN scenarios reached: "
            f"{len(constrain_results)}"
        )

        for result in constrain_results:

            print("\n")
            print(
                f"Scenario: "
                f"{result['name']}"
            )

            print(
                f"Modified: "
                f"{result['modified']}"
            )

            print(
                f"Maximum action delta: "
                f"{result['max_delta']:.8f}"
            )

            print(
                f"Mean action delta: "
                f"{result['mean_delta']:.8f}"
            )

            if result["modified"]:

                print(
                    "SAFE-ENVELOPE EFFECT: "
                    "ACTION WAS MODIFIED"
                )

            else:

                print(
                    "SAFE-ENVELOPE EFFECT: "
                    "NO ACTION MODIFICATION"
                )

                print(
                    "ARCHITECTURAL FINDING: "
                    "The current CONSTRAIN branch "
                    "does not materially transform "
                    "a policy-compliant action."
                )

    # -----------------------------------------------------------------
    # Reason precedence analysis
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 82)
    print("DECISION PRECEDENCE ANALYSIS")
    print("=" * 82)

    print(
        "Current GovernanceEngine precedence:"
    )

    print(
        "1. Policy violation -> BLOCK"
    )

    print(
        "2. HIGH risk -> BLOCK"
    )

    print(
        "3. Trust below minimum -> BLOCK"
    )

    print(
        "4. MEDIUM risk -> CONSTRAIN"
    )

    print(
        "5. Otherwise -> ALLOW"
    )

    print("\n")

    combined_policy_cases = [
        r
        for r in results
        if (
            r["actual_status"] == "BLOCK"
            and r["actual_reason"]
            == "POLICY_VIOLATION"
        )
    ]

    print(
        f"Policy-first BLOCK cases: "
        f"{len(combined_policy_cases)}"
    )

    # -----------------------------------------------------------------
    # Scientific interpretation
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 82)
    print("SCIENTIFIC INTERPRETATION")
    print("=" * 82)

    print(
        "This experiment validates governance "
        "branch reachability using controlled "
        "synthetic scenarios."
    )

    print(
        "The experiment does not modify the SAC "
        "agent, IRF environment, reward function, "
        "or existing scientific benchmarks."
    )

    print(
        "ALLOW demonstrates acceptance of a "
        "policy-compliant low-risk action."
    )

    print(
        "BLOCK demonstrates enforcement of hard "
        "policy, trust, or high-risk conditions."
    )

    print(
        "CONSTRAIN demonstrates whether the current "
        "architecture can transform a medium-risk "
        "action before execution."
    )

    print(
        "A CONSTRAIN decision with zero action delta "
        "indicates that the current constraint "
        "mechanism is logically present but "
        "operationally weak."
    )

    print(
        "Synthetic stress conditions must not be "
        "interpreted as physical-world measurements."
    )

    print("\n")
    print("=" * 82)
    print("GOVERNANCE BRANCH COVERAGE TEST COMPLETE")
    print("=" * 82)


if __name__ == "__main__":
    main()

