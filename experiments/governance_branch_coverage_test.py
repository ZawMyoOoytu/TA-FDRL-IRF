
"""
TA-FDRL-IRF
Governance Branch Coverage Test

Purpose
-------
Validate GovernanceEngine decision branches independently from the
numerical reachability of the real RiskEngine thresholds.

Branches under test
-------------------
    1. NORMAL_ALLOW
    2. LOW_TRUST_BLOCK
    3. POLICY_VIOLATION_BLOCK
    4. MEDIUM_RISK_CONSTRAIN
    5. HIGH_RISK_BLOCK
    6. COMBINED_STRESS_BLOCK

Architectural distinction
-------------------------
This test isolates GovernanceEngine decision logic.

The real PolicyEngine, RiskEngine, and SafeEnvelope are replaced with
deterministic test doubles through the ACTUAL dependency attributes
used by GovernanceEngine:

    governance.policy
    governance.risk
    governance.safe_envelope

The real GovernanceEngine.evaluate() implementation remains active.

Governance precedence
---------------------
    1. POLICY VIOLATION -> BLOCK
    2. HIGH RISK        -> BLOCK
    3. LOW GOV TRUST    -> BLOCK
    4. MEDIUM RISK      -> CONSTRAIN
    5. OTHERWISE        -> ALLOW

Important
---------
This test does NOT modify:

    - trust/policy.py
    - trust/risk.py
    - trust/governance.py
    - RiskConfig thresholds
    - TrustEngine equations
    - IRF environment dynamics
    - SAC agent behavior

The purpose is to prove that GovernanceEngine decision branches
are reachable and correctly ordered.

RiskEngine numerical calibration/reachability must be tested separately.
"""

from __future__ import annotations

# ============================================================================
# PROJECT ROOT IMPORT PATH
# ============================================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# IMPORTS
# ============================================================================

from dataclasses import dataclass
from typing import Any

import numpy as np

from trust.governance import GovernanceEngine
from trust.policy import PolicyDecision
from trust.risk import RiskAssessment


# ============================================================================
# CONSTANTS
# ============================================================================

ACTION_DIM = 40


# ============================================================================
# DETERMINISTIC POLICY TEST DOUBLE
# ============================================================================

class StubPolicyEngine:
    """
    Deterministic replacement for PolicyEngine.

    GovernanceEngine uses:

        self.policy.evaluate(action)

    Therefore this stub implements evaluate().
    """

    def __init__(
        self,
        allowed: bool = True,
        violations: list[str] | None = None,
    ) -> None:

        self.allowed = bool(allowed)
        self.violations = list(violations or [])

    def evaluate(
        self,
        action: np.ndarray,
    ) -> PolicyDecision:
        """
        Return a deterministic PolicyDecision.

        The actual PolicyDecision requires:

            allowed
            reason
            violations
            action
        """

        normalized_action = np.asarray(
            action,
            dtype=np.float32,
        ).copy()

        if self.allowed:
            reason = "ACTION_APPROVED"
        else:
            reason = "POLICY_VIOLATION"

        return PolicyDecision(
            allowed=self.allowed,
            reason=reason,
            violations=self.violations,
            action=normalized_action,
        )


# ============================================================================
# DETERMINISTIC RISK TEST DOUBLE
# ============================================================================

class StubRiskConfig:
    """
    Minimal RiskConfig-compatible object.

    GovernanceEngine accesses:

        self.risk.config.minimum_trust
    """

    minimum_trust = 0.60


class StubRiskEngine:
    """
    Deterministic replacement for RiskEngine.

    GovernanceEngine uses:

        self.risk.evaluate(...)

    Therefore this stub implements evaluate().
    """

    def __init__(
        self,
        risk_score: float,
        risk_level: str,
        reasons: list[str] | None = None,
    ) -> None:

        self.risk_score = float(risk_score)
        self.risk_level = str(risk_level).upper()
        self.reasons = list(reasons or [])

        self.config = StubRiskConfig()

    def evaluate(
        self,
        action: np.ndarray,
        trust_score: float,
        telemetry: dict[str, Any] | None = None,
    ) -> RiskAssessment:
        """
        Return a deterministic RiskAssessment.

        This bypasses the numerical RiskEngine calculation intentionally.
        The purpose is GovernanceEngine branch testing.
        """

        normalized_action = np.asarray(
            action,
            dtype=np.float32,
        )

        if self.risk_level == "LOW":

            action_anomaly = 0.05

        elif self.risk_level == "MEDIUM":

            action_anomaly = 0.65

        elif self.risk_level == "HIGH":

            action_anomaly = 0.95

        else:

            raise ValueError(
                f"Unsupported test risk level: "
                f"{self.risk_level}"
            )

        return RiskAssessment(
            risk_score=self.risk_score,
            risk_level=self.risk_level,
            action_anomaly=action_anomaly,
            reasons=self.reasons,
            trust_score=float(trust_score),
        )


# ============================================================================
# DETERMINISTIC SAFE ENVELOPE
# ============================================================================

class StubSafeEnvelope:
    """
    Deterministic SafeEnvelope replacement.

    Purpose
    -------
    Ensure that the CONSTRAIN branch visibly modifies the action.

    The actual SafeEnvelope implementation is tested separately.

    This test only verifies that GovernanceEngine:

        MEDIUM risk
            ->
        invokes safe envelope
            ->
        returns CONSTRAIN
            ->
        returns constrained action.
    """

    @dataclass
    class ConstraintResult:
        """
        Minimal result expected by GovernanceEngine.
        """

        action: np.ndarray
        modified: bool

    def apply(
        self,
        action: np.ndarray,
    ) -> ConstraintResult:
        """
        Clip action values to [-0.5, 0.5].
        """

        original = np.asarray(
            action,
            dtype=np.float32,
        ).copy()

        constrained = np.clip(
            original,
            -0.5,
            0.5,
        ).astype(np.float32)

        modified = not np.array_equal(
            original,
            constrained,
        )

        return self.ConstraintResult(
            action=constrained,
            modified=modified,
        )


# ============================================================================
# ACTION HELPERS
# ============================================================================

def make_policy_compliant_action() -> np.ndarray:
    """
    Create a policy-compliant 40-dimensional action.

    dimensions 0:20
        bandwidth-related action

    dimensions 20:40
        power-related action
    """

    action = np.zeros(
        ACTION_DIM,
        dtype=np.float32,
    )

    action[:20] = 0.50
    action[20:] = 0.50

    return action


def make_medium_risk_action() -> np.ndarray:
    """
    Create a policy-compliant action that the test SafeEnvelope
    will visibly modify.

    0.80 is:

        bandwidth <= 1.00  -> policy compliant
        power <= 0.90      -> policy compliant

    SafeEnvelope:

        0.80 -> 0.50
    """

    action = np.zeros(
        ACTION_DIM,
        dtype=np.float32,
    )

    action[:20] = 0.80
    action[20:] = 0.80

    return action


def make_policy_violation_action() -> np.ndarray:
    """
    Create a deliberately policy-invalid action.

    Power dimensions are 1.00, exceeding the configured
    maximum normalized power of 0.90.
    """

    action = np.zeros(
        ACTION_DIM,
        dtype=np.float32,
    )

    action[:20] = 0.50
    action[20:] = 1.00

    return action


# ============================================================================
# GOVERNANCE FACTORY
# ============================================================================

def make_governance_engine(
    *,
    trust_score: float,
    policy_allowed: bool,
    policy_violations: list[str] | None,
    risk_score: float,
    risk_level: str,
    risk_reasons: list[str] | None = None,
) -> GovernanceEngine:
    """
    Construct GovernanceEngine with deterministic test dependencies.

    Actual dependency attributes:

        governance.policy
        governance.risk
        governance.safe_envelope
    """

    governance = GovernanceEngine()

    # ------------------------------------------------------------------------
    # Policy test double
    # ------------------------------------------------------------------------

    governance.policy = StubPolicyEngine(
        allowed=policy_allowed,
        violations=policy_violations,
    )

    # ------------------------------------------------------------------------
    # Risk test double
    # ------------------------------------------------------------------------

    governance.risk = StubRiskEngine(
        risk_score=risk_score,
        risk_level=risk_level,
        reasons=risk_reasons,
    )

    # ------------------------------------------------------------------------
    # Safe envelope test double
    # ------------------------------------------------------------------------

    governance.safe_envelope = StubSafeEnvelope()

    # ------------------------------------------------------------------------
    # Explicit governance trust control
    # ------------------------------------------------------------------------

    governance.trust_engine.trust_score = float(
        trust_score
    )

    governance.trust_engine.environment_trust = float(
        trust_score
    )

    return governance


# ============================================================================
# SCENARIO DEFINITION
# ============================================================================

@dataclass
class Scenario:
    """
    One deterministic governance branch test case.
    """

    name: str

    trust_score: float

    policy_allowed: bool
    policy_violations: list[str]

    risk_score: float
    risk_level: str
    risk_reasons: list[str]

    expected_status: str
    expected_reason: str

    action: np.ndarray


# ============================================================================
# TEST SCENARIOS
# ============================================================================

SCENARIOS: list[Scenario] = [

    # ------------------------------------------------------------------------
    # 1. NORMAL ALLOW
    # ------------------------------------------------------------------------

    Scenario(
        name="NORMAL_ALLOW",

        trust_score=0.90,

        policy_allowed=True,
        policy_violations=[],

        risk_score=0.10,
        risk_level="LOW",
        risk_reasons=[],

        expected_status="ALLOW",
        expected_reason="GOVERNANCE_APPROVED",

        action=make_policy_compliant_action(),
    ),

    # ------------------------------------------------------------------------
    # 2. LOW GOVERNANCE TRUST -> BLOCK
    # ------------------------------------------------------------------------

    Scenario(
        name="LOW_TRUST_BLOCK",

        trust_score=0.30,

        policy_allowed=True,
        policy_violations=[],

        risk_score=0.10,
        risk_level="LOW",
        risk_reasons=["LOW_GOVERNANCE_TRUST"],

        expected_status="BLOCK",
        expected_reason="TRUST_BELOW_MINIMUM",

        action=make_policy_compliant_action(),
    ),

    # ------------------------------------------------------------------------
    # 3. POLICY VIOLATION -> BLOCK
    # ------------------------------------------------------------------------

    Scenario(
        name="POLICY_VIOLATION_BLOCK",

        trust_score=0.90,

        policy_allowed=False,
        policy_violations=[
            "POWER_POLICY_VIOLATION"
        ],

        risk_score=0.10,
        risk_level="LOW",
        risk_reasons=[],

        expected_status="BLOCK",
        expected_reason="POLICY_VIOLATION",

        action=make_policy_violation_action(),
    ),

    # ------------------------------------------------------------------------
    # 4. MEDIUM RISK -> CONSTRAIN
    # ------------------------------------------------------------------------

    Scenario(
        name="MEDIUM_RISK_CONSTRAIN",

        trust_score=0.90,

        policy_allowed=True,
        policy_violations=[],

        risk_score=0.65,
        risk_level="MEDIUM",
        risk_reasons=["MEDIUM_RISK"],

        expected_status="CONSTRAIN",
        expected_reason="MEDIUM_RISK_SAFE_ENVELOPE",

        action=make_medium_risk_action(),
    ),

    # ------------------------------------------------------------------------
    # 5. HIGH RISK -> BLOCK
    # ------------------------------------------------------------------------

    Scenario(
        name="HIGH_RISK_BLOCK",

        trust_score=0.90,

        policy_allowed=True,
        policy_violations=[],

        risk_score=0.85,
        risk_level="HIGH",
        risk_reasons=["HIGH_RISK"],

        expected_status="BLOCK",
        expected_reason="HIGH_RISK",

        action=make_policy_compliant_action(),
    ),

    # ------------------------------------------------------------------------
    # 6. COMBINED STRESS
    #
    # Policy violation + HIGH risk + LOW governance trust.
    #
    # Policy violation must win because it has highest precedence.
    # ------------------------------------------------------------------------

    Scenario(
        name="COMBINED_STRESS_BLOCK",

        trust_score=0.30,

        policy_allowed=False,
        policy_violations=[
            "POWER_POLICY_VIOLATION"
        ],

        risk_score=0.90,
        risk_level="HIGH",
        risk_reasons=[
            "HIGH_QUEUE_PRESSURE",
            "HIGH_INTERFERENCE",
            "HIGH_RISK",
        ],

        expected_status="BLOCK",
        expected_reason="POLICY_VIOLATION",

        action=make_policy_violation_action(),
    ),
]


# ============================================================================
# SCENARIO RUNNER
# ============================================================================

def run_scenario(
    scenario: Scenario,
) -> dict[str, Any]:
    """
    Execute one governance branch scenario.
    """

    governance = make_governance_engine(
        trust_score=scenario.trust_score,
        policy_allowed=scenario.policy_allowed,
        policy_violations=scenario.policy_violations,
        risk_score=scenario.risk_score,
        risk_level=scenario.risk_level,
        risk_reasons=scenario.risk_reasons,
    )

    # ------------------------------------------------------------------------
    # Controlled telemetry
    # ------------------------------------------------------------------------

    telemetry: dict[str, Any] = {
        "queue_pressure": 0.20,
        "interference": 0.20,
        "power": 0.50,
        "sinr": 10.0,
        "trust": scenario.trust_score,
    }

    # ------------------------------------------------------------------------
    # PRE-EXECUTION GOVERNANCE EVALUATION
    # ------------------------------------------------------------------------

    decision = governance.evaluate(
        scenario.action,
        trust_score=scenario.trust_score,
        telemetry=telemetry,
    )

    # ------------------------------------------------------------------------
    # Extract actual decision
    # ------------------------------------------------------------------------

    actual_status = str(
        decision.status
    )

    actual_reason = str(
        decision.reason
    )

    actual_risk_level = str(
        decision.risk.risk_level
    )

    actual_risk_score = float(
        decision.risk.risk_score
    )

    actual_policy_allowed = bool(
        decision.policy.allowed
    )

    # ------------------------------------------------------------------------
    # Status validation
    # ------------------------------------------------------------------------

    status_pass = (
        actual_status
        == scenario.expected_status
    )

    # ------------------------------------------------------------------------
    # Reason validation
    # ------------------------------------------------------------------------

    reason_pass = (
        actual_reason
        == scenario.expected_reason
    )

    # ------------------------------------------------------------------------
    # Risk validation
    # ------------------------------------------------------------------------

    risk_level_pass = (
        actual_risk_level
        == scenario.risk_level
    )

    risk_score_pass = bool(
        np.isclose(
            actual_risk_score,
            scenario.risk_score,
        )
    )

    # ------------------------------------------------------------------------
    # Policy validation
    # ------------------------------------------------------------------------

    policy_pass = (
        actual_policy_allowed
        == scenario.policy_allowed
    )

    # ------------------------------------------------------------------------
    # Decision action validation
    # ------------------------------------------------------------------------

    decision_action = np.asarray(
        decision.action,
        dtype=np.float32,
    )

    proposed_action = np.asarray(
        decision.proposed_action,
        dtype=np.float32,
    )

    action_shape_pass = (
        decision_action.shape
        == (ACTION_DIM,)
    )

    proposed_shape_pass = (
        proposed_action.shape
        == (ACTION_DIM,)
    )

    action_finite_pass = bool(
        np.all(
            np.isfinite(
                decision_action
            )
        )
    )

    # ------------------------------------------------------------------------
    # Safe-envelope validation
    # ------------------------------------------------------------------------

    constraint_pass = True
    constraint_details: dict[str, Any] = {}

    if scenario.expected_status == "CONSTRAIN":

        original_action = np.asarray(
            scenario.action,
            dtype=np.float32,
        )

        # Shape
        shape_pass = (
            decision_action.shape
            == original_action.shape
        )

        # Finite
        finite_pass = bool(
            np.all(
                np.isfinite(
                    decision_action
                )
            )
        )

        # Must be transformed
        transformed_pass = not np.array_equal(
            decision_action,
            original_action,
        )

        # GovernanceDecision.modified
        modified_flag_pass = bool(
            decision.modified
        )

        # Safe envelope should produce 0.50 maximum
        envelope_limit_pass = bool(
            np.all(
                np.abs(decision_action)
                <= 0.50 + 1e-6
            )
        )

        constraint_pass = (
            shape_pass
            and finite_pass
            and transformed_pass
            and modified_flag_pass
            and envelope_limit_pass
        )

        constraint_details = {
            "shape_pass": shape_pass,
            "finite_pass": finite_pass,
            "transformed_pass": transformed_pass,
            "modified_flag_pass": modified_flag_pass,
            "envelope_limit_pass": envelope_limit_pass,
            "original_mean": float(
                np.mean(original_action)
            ),
            "constrained_mean": float(
                np.mean(decision_action)
            ),
        }

    # ------------------------------------------------------------------------
    # BLOCK action validation
    # ------------------------------------------------------------------------

    block_action_pass = True

    if scenario.expected_status == "BLOCK":

        # Governance BLOCK should return a valid action.
        #
        # Exact fallback semantics are tested elsewhere.
        block_action_pass = (
            action_shape_pass
            and action_finite_pass
        )

    # ------------------------------------------------------------------------
    # ALLOW action validation
    # ------------------------------------------------------------------------

    allow_action_pass = True

    if scenario.expected_status == "ALLOW":

        allow_action_pass = (
            action_shape_pass
            and action_finite_pass
            and np.array_equal(
                decision_action,
                proposed_action,
            )
        )

    # ------------------------------------------------------------------------
    # Overall scenario result
    # ------------------------------------------------------------------------

    passed = (
        status_pass
        and reason_pass
        and risk_level_pass
        and risk_score_pass
        and policy_pass
        and action_shape_pass
        and proposed_shape_pass
        and action_finite_pass
        and constraint_pass
        and block_action_pass
        and allow_action_pass
    )

    # ------------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------------

    print()
    print(
        f"SCENARIO: {scenario.name}"
    )
    print("-" * 78)

    print(
        f"Trust              : "
        f"{scenario.trust_score:.4f}"
    )

    print(
        f"Risk score         : "
        f"{actual_risk_score:.6f}"
    )

    print(
        f"Risk level         : "
        f"{actual_risk_level}"
    )

    print(
        f"Policy allowed     : "
        f"{actual_policy_allowed}"
    )

    print(
        f"Policy violations  : "
        f"{scenario.policy_violations}"
    )

    print(
        f"Expected status    : "
        f"{scenario.expected_status}"
    )

    print(
        f"Actual status      : "
        f"{actual_status}"
    )

    print(
        f"Expected reason    : "
        f"{scenario.expected_reason}"
    )

    print(
        f"Actual reason      : "
        f"{actual_reason}"
    )

    print(
        "Policy check       : "
        f"{'PASS' if policy_pass else 'FAIL'}"
    )

    print(
        "Risk injection     : "
        f"{'PASS' if (risk_level_pass and risk_score_pass) else 'FAIL'}"
    )

    print(
        "Decision action    : "
        f"{'PASS' if (action_shape_pass and action_finite_pass) else 'FAIL'}"
    )

    if scenario.expected_status == "CONSTRAIN":

        print(
            "Safe-envelope test : "
            f"{'PASS' if constraint_pass else 'FAIL'}"
        )

        print(
            f"  Original mean    : "
            f"{constraint_details['original_mean']:.4f}"
        )

        print(
            f"  Constrained mean : "
            f"{constraint_details['constrained_mean']:.4f}"
        )

        print(
            f"  Shape            : "
            f"{'PASS' if constraint_details['shape_pass'] else 'FAIL'}"
        )

        print(
            f"  Finite           : "
            f"{'PASS' if constraint_details['finite_pass'] else 'FAIL'}"
        )

        print(
            f"  Transformed      : "
            f"{'PASS' if constraint_details['transformed_pass'] else 'FAIL'}"
        )

        print(
            f"  Modified flag    : "
            f"{'PASS' if constraint_details['modified_flag_pass'] else 'FAIL'}"
        )

        print(
            f"  Envelope limit   : "
            f"{'PASS' if constraint_details['envelope_limit_pass'] else 'FAIL'}"
        )

    print(
        "RESULT             : "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return {
        "name": scenario.name,

        "expected_status": scenario.expected_status,
        "actual_status": actual_status,

        "expected_reason": scenario.expected_reason,
        "actual_reason": actual_reason,

        "expected_risk_level": scenario.risk_level,
        "actual_risk_level": actual_risk_level,

        "expected_risk_score": scenario.risk_score,
        "actual_risk_score": actual_risk_score,

        "expected_policy_allowed": scenario.policy_allowed,
        "actual_policy_allowed": actual_policy_allowed,

        "constraint_pass": constraint_pass,

        "action_shape_pass": action_shape_pass,
        "action_finite_pass": action_finite_pass,

        "passed": passed,
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """
    Execute all governance branch scenarios.
    """

    print("=" * 78)
    print(
        "TA-FDRL-IRF GOVERNANCE BRANCH COVERAGE TEST"
    )
    print("=" * 78)

    print()
    print("Architecture under test:")

    print(
        "  Policy violation       -> BLOCK"
    )

    print(
        "  HIGH risk              -> BLOCK"
    )

    print(
        "  Low governance trust  -> BLOCK"
    )

    print(
        "  MEDIUM risk            -> CONSTRAIN"
    )

    print(
        "  Otherwise              -> ALLOW"
    )

    print()
    print("NOTE:")

    print(
        "This test isolates GovernanceEngine decision branches."
    )

    print(
        "RiskEngine numerical calibration is intentionally tested separately."
    )

    print()
    print(
        "Project root: "
        f"{PROJECT_ROOT}"
    )

    # ------------------------------------------------------------------------
    # Execute scenarios
    # ------------------------------------------------------------------------

    results: list[dict[str, Any]] = []

    for scenario in SCENARIOS:

        results.append(
            run_scenario(
                scenario
            )
        )

    # ------------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------------

    total = len(results)

    passed = sum(
        1
        for result in results
        if result["passed"]
    )

    failed = total - passed

    pass_rate = (
        (passed / total) * 100.0
        if total > 0
        else 0.0
    )

    # ------------------------------------------------------------------------
    # Branch counts
    # ------------------------------------------------------------------------

    allow_count = sum(
        1
        for result in results
        if result["actual_status"] == "ALLOW"
    )

    constrain_count = sum(
        1
        for result in results
        if result["actual_status"] == "CONSTRAIN"
    )

    block_count = sum(
        1
        for result in results
        if result["actual_status"] == "BLOCK"
    )

    # ------------------------------------------------------------------------
    # Branch coverage
    # ------------------------------------------------------------------------

    branch_expectations = {
        "ALLOW": allow_count > 0,
        "CONSTRAIN": constrain_count > 0,
        "BLOCK": block_count > 0,
    }

    all_branches_covered = all(
        branch_expectations.values()
    )

    # ------------------------------------------------------------------------
    # Scenario coverage
    # ------------------------------------------------------------------------

    scenario_expectations = {
        scenario.name: any(
            result["name"] == scenario.name
            and result["passed"]
            for result in results
        )
        for scenario in SCENARIOS
    }

    all_scenarios_passed = all(
        scenario_expectations.values()
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(
        f"Scenarios             : {total}"
    )

    print(
        f"Passed                : {passed}"
    )

    print(
        f"Failed                : {failed}"
    )

    print(
        f"Pass rate             : "
        f"{pass_rate:.1f}%"
    )

    print()

    print(
        f"ALLOW                 : "
        f"{allow_count}"
    )

    print(
        f"CONSTRAIN             : "
        f"{constrain_count}"
    )

    print(
        f"BLOCK                 : "
        f"{block_count}"
    )

    print()
    print("Branch coverage:")

    for branch, covered in branch_expectations.items():

        print(
            f"  {branch:<10} : "
            f"{'COVERED' if covered else 'NOT COVERED'}"
        )

    print()
    print("Scenario validation:")

    for scenario_name, scenario_passed in scenario_expectations.items():

        print(
            f"  {scenario_name:<24} : "
            f"{'PASS' if scenario_passed else 'FAIL'}"
        )

    print()
    print("=" * 78)

    # ------------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------------

    if (
        passed == total
        and all_branches_covered
        and all_scenarios_passed
    ):

        print(
            "RESULT: GOVERNANCE BRANCH COVERAGE PASSED"
        )

        print(
            f"RESULT: {passed}/{total} SCENARIOS PASSED"
        )

        print(
            "RESULT: ALLOW / CONSTRAIN / BLOCK ALL REACHED"
        )

        print(
            "RESULT: MEDIUM-RISK SAFE-ENVELOPE PATH VERIFIED"
        )

        print(
            "RESULT: HIGH-RISK HARD-BLOCK PATH VERIFIED"
        )

        print(
            "RESULT: POLICY-PRECEDENCE PATH VERIFIED"
        )

        print("=" * 78)

        return

    # ------------------------------------------------------------------------
    # Failure
    # ------------------------------------------------------------------------

    print(
        "RESULT: GOVERNANCE BRANCH COVERAGE FAILED"
    )

    print("=" * 78)

    raise SystemExit(1)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

