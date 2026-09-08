
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
# Helpers
# ---------------------------------------------------------------------

def make_action(value: float) -> np.ndarray:
    """
    Create a uniform 40-dimensional action.
    """

    return np.full(
        ACTION_DIM,
        value,
        dtype=np.float32,
    )


def make_power_violation() -> np.ndarray:
    """
    Valid action range violation for the power portion.

    First 20 dimensions:
        bandwidth

    Last 20 dimensions:
        power

    Policy maximum normalized power = 0.90.
    """

    action = np.zeros(
        ACTION_DIM,
        dtype=np.float32,
    )

    action[20] = 0.95

    return action


def make_high_anomaly_action() -> np.ndarray:
    """
    Action with sufficiently large magnitude to trigger
    the RiskEngine action-anomaly detector.
    """

    return np.ones(
        ACTION_DIM,
        dtype=np.float32,
    )


def make_telemetry(
    queue_value=0.10,
    interference_value=0.10,
):
    """
    Create normalized synthetic telemetry.

    These are controlled stress-test inputs.
    They are NOT measurements from the physical world.
    """

    return {
        "queue": np.full(
            NUM_USERS,
            queue_value,
            dtype=np.float32,
        ),
        "interference": np.full(
            NUM_USERS,
            interference_value,
            dtype=np.float32,
        ),
    }


# ---------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------

def build_scenarios():

    scenarios = [

        {
            "name": "NORMAL_OPERATION",
            "trust": 0.90,
            "action": make_action(0.00),
            "telemetry": make_telemetry(
                queue_value=0.10,
                interference_value=0.10,
            ),
            "expected": "ALLOW",
        },

        {
            "name": "LOW_TRUST",
            "trust": 0.30,
            "action": make_action(0.00),
            "telemetry": make_telemetry(
                queue_value=0.10,
                interference_value=0.10,
            ),
            "expected": "BLOCK",
        },

        {
            "name": "MODERATE_TRUST",
            "trust": 0.55,
            "action": make_action(0.00),
            "telemetry": make_telemetry(
                queue_value=0.10,
                interference_value=0.10,
            ),
            "expected": "BLOCK",
        },

        {
            "name": "HIGH_QUEUE_PRESSURE",
            "trust": 0.90,
            "action": make_action(0.00),
            "telemetry": make_telemetry(
                queue_value=0.90,
                interference_value=0.10,
            ),
            "expected": "ALLOW",
        },

        {
            "name": "HIGH_INTERFERENCE",
            "trust": 0.90,
            "action": make_action(0.00),
            "telemetry": make_telemetry(
                queue_value=0.10,
                interference_value=0.90,
            ),
            "expected": "ALLOW",
        },

        {
            "name": "HIGH_ACTION_ANOMALY",
            "trust": 0.90,
            "action": make_high_anomaly_action(),
            "telemetry": make_telemetry(
                queue_value=0.10,
                interference_value=0.10,
            ),
            "expected": "ALLOW",
        },

        {
            "name": "POWER_POLICY_VIOLATION",
            "trust": 0.90,
            "action": make_power_violation(),
            "telemetry": make_telemetry(
                queue_value=0.10,
                interference_value=0.10,
            ),
            "expected": "BLOCK",
        },

        {
            "name": "LOW_TRUST_HIGH_QUEUE",
            "trust": 0.30,
            "action": make_action(0.00),
            "telemetry": make_telemetry(
                queue_value=0.90,
                interference_value=0.10,
            ),
            "expected": "BLOCK",
        },

        {
            "name": "LOW_TRUST_HIGH_INTERFERENCE",
            "trust": 0.30,
            "action": make_action(0.00),
            "telemetry": make_telemetry(
                queue_value=0.10,
                interference_value=0.90,
            ),
            "expected": "BLOCK",
        },

        {
            "name": "COMBINED_STRESS",
            "trust": 0.30,
            "action": make_high_anomaly_action(),
            "telemetry": make_telemetry(
                queue_value=0.90,
                interference_value=0.90,
            ),
            "expected": "BLOCK",
        },
    ]

    return scenarios


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 78)
    print("TA-FDRL-IRF GOVERNANCE STRESS TEST")
    print("=" * 78)

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

        print("\n" + "-" * 78)
        print(
            f"SCENARIO {index}/{len(scenarios)}"
        )
        print("-" * 78)

        print(
            f"Name:              "
            f"{scenario['name']}"
        )

        print(
            f"Trust score:       "
            f"{scenario['trust']:.4f}"
        )

        print(
            f"Expected status:   "
            f"{scenario['expected']}"
        )

        decision = governance.evaluate(
            scenario["action"],
            trust_score=scenario["trust"],
            telemetry=scenario["telemetry"],
        )

        passed = (
            decision.status
            == scenario["expected"]
        )

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
            f"Governance status: "
            f"{decision.status}"
        )

        print(
            f"Governance reason: "
            f"{decision.reason}"
        )

        print(
            f"Action modified:   "
            f"{decision.modified}"
        )

        print(
            f"TEST RESULT:       "
            f"{'PASS' if passed else 'FAIL'}"
        )

        results.append(
            {
                "name": scenario["name"],
                "expected": scenario["expected"],
                "actual": decision.status,
                "risk": float(
                    decision.risk.risk_score
                ),
                "risk_level": decision.risk.risk_level,
                "trust": float(
                    scenario["trust"]
                ),
                "policy_allowed": bool(
                    decision.policy.allowed
                ),
                "modified": bool(
                    decision.modified
                ),
                "passed": passed,
            }
        )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    passed_count = sum(
        result["passed"]
        for result in results
    )

    failed_count = (
        len(results)
        - passed_count
    )

    print("\n")
    print("=" * 78)
    print("GOVERNANCE STRESS TEST SUMMARY")
    print("=" * 78)

    print(
        f"Total scenarios:    "
        f"{len(results)}"
    )

    print(
        f"Passed:             "
        f"{passed_count}"
    )

    print(
        f"Failed:             "
        f"{failed_count}"
    )

    print(
        f"Pass rate:          "
        f"{100.0 * passed_count / len(results):.2f}%"
    )

    # -----------------------------------------------------------------
    # Decision distribution
    # -----------------------------------------------------------------

    allow_count = sum(
        result["actual"] == "ALLOW"
        for result in results
    )

    constrain_count = sum(
        result["actual"] == "CONSTRAIN"
        for result in results
    )

    block_count = sum(
        result["actual"] == "BLOCK"
        for result in results
    )

    print("\n")
    print("=" * 78)
    print("GOVERNANCE DECISION DISTRIBUTION")
    print("=" * 78)

    print(
        f"ALLOW:              "
        f"{allow_count}"
    )

    print(
        f"CONSTRAIN:          "
        f"{constrain_count}"
    )

    print(
        f"BLOCK:              "
        f"{block_count}"
    )

    # -----------------------------------------------------------------
    # Scenario table
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("SCENARIO RESULTS")
    print("=" * 78)

    print(
        f"{'Scenario':30s} "
        f"{'Trust':>7s} "
        f"{'Risk':>8s} "
        f"{'Expected':>10s} "
        f"{'Actual':>10s} "
        f"{'Result':>8s}"
    )

    print("-" * 78)

    for result in results:

        print(
            f"{result['name']:30s} "
            f"{result['trust']:7.3f} "
            f"{result['risk']:8.3f} "
            f"{result['expected']:>10s} "
            f"{result['actual']:>10s} "
            f"{'PASS' if result['passed'] else 'FAIL':>8s}"
        )

    # -----------------------------------------------------------------
    # Important diagnostic
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("CONSTRAIN MECHANISM DIAGNOSTIC")
    print("=" * 78)

    if constrain_count == 0:

        print(
            "No CONSTRAIN decisions were produced."
        )

        print(
            "This is expected to be investigated."
        )

        print(
            "The current GovernanceEngine can only "
            "produce CONSTRAIN when RiskEngine "
            "returns MEDIUM risk."
        )

        print(
            "The current RiskEngine thresholds may "
            "not overlap with the synthetic operating "
            "conditions."
        )

    else:

        print(
            f"CONSTRAIN decisions observed: "
            f"{constrain_count}"
        )

    # -----------------------------------------------------------------
    # Scientific interpretation
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("SCIENTIFIC INTERPRETATION")
    print("=" * 78)

    print(
        "This experiment validates governance decision "
        "logic under controlled synthetic stress conditions."
    )

    print(
        "The scenarios are synthetic and must not be "
        "interpreted as physical-world measurements."
    )

    print(
        "The existing SAC and IRF scientific benchmark "
        "is not modified by this experiment."
    )

    print(
        "A passing governance test demonstrates that "
        "the software decision hierarchy responds to "
        "specified policy and risk conditions."
    )

    print(
        "It does not by itself demonstrate improved "
        "6G network performance or real-world security."
    )

    print("\n")
    print("=" * 78)
    print("GOVERNANCE STRESS TEST COMPLETE")
    print("=" * 78)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()

