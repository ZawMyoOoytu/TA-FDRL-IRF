
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import PolicyEngine, PolicyConfig, PolicyDecision
from .risk import RiskEngine, RiskConfig, RiskAssessment
from .safe_envelope import SafeEnvelope, SafeEnvelopeConfig


@dataclass
class GovernanceDecision:
    status: str
    action: np.ndarray
    proposed_action: np.ndarray
    policy: PolicyDecision
    risk: RiskAssessment
    modified: bool
    reason: str


class GovernanceEngine:
    """
    High-level governance layer for the TA-FDRL-IRF agent.

    Decision hierarchy:

        BLOCK
            - hard policy violation
            - invalid action
            - high risk
            - trust below minimum

        CONSTRAIN
            - medium risk
            - action transformed by SafeEnvelope

        ALLOW
            - policy compliant
            - low risk
            - acceptable trust
    """

    def __init__(
        self,
        policy_engine=None,
        risk_engine=None,
        safe_envelope=None,
    ):
        self.policy = (
            policy_engine
            if policy_engine is not None
            else PolicyEngine(PolicyConfig())
        )

        self.risk = (
            risk_engine
            if risk_engine is not None
            else RiskEngine(RiskConfig())
        )

        self.safe_envelope = (
            safe_envelope
            if safe_envelope is not None
            else SafeEnvelope(SafeEnvelopeConfig())
        )

    def evaluate(
        self,
        action,
        trust_score=None,
        telemetry=None,
    ):
        proposed_action = np.asarray(
            action,
            dtype=np.float32,
        ).copy()

        # ---------------------------------------------------------
        # 1. POLICY GATE
        # ---------------------------------------------------------

        policy_decision = self.policy.evaluate(
            proposed_action
        )

        # ---------------------------------------------------------
        # 2. RISK ASSESSMENT
        # ---------------------------------------------------------

        risk_assessment = self.risk.evaluate(
            proposed_action,
            trust_score=trust_score,
            telemetry=telemetry,
        )

        # ---------------------------------------------------------
        # 3. HARD BLOCK: POLICY VIOLATION
        # ---------------------------------------------------------

        if not policy_decision.allowed:
            return GovernanceDecision(
                status="BLOCK",
                action=proposed_action,
                proposed_action=proposed_action,
                policy=policy_decision,
                risk=risk_assessment,
                modified=False,
                reason="POLICY_VIOLATION",
            )

        # ---------------------------------------------------------
        # 4. HARD BLOCK: HIGH RISK
        # ---------------------------------------------------------

        if risk_assessment.risk_level == "HIGH":
            return GovernanceDecision(
                status="BLOCK",
                action=proposed_action,
                proposed_action=proposed_action,
                policy=policy_decision,
                risk=risk_assessment,
                modified=False,
                reason="HIGH_RISK",
            )

        # ---------------------------------------------------------
        # 5. HARD BLOCK: TRUST BELOW MINIMUM
        # ---------------------------------------------------------

        if (
            trust_score is not None
            and trust_score < self.risk.config.minimum_trust
        ):
            return GovernanceDecision(
                status="BLOCK",
                action=proposed_action,
                proposed_action=proposed_action,
                policy=policy_decision,
                risk=risk_assessment,
                modified=False,
                reason="TRUST_BELOW_MINIMUM",
            )

        # ---------------------------------------------------------
        # 6. MEDIUM RISK -> SAFE ENVELOPE
        # ---------------------------------------------------------

        if risk_assessment.risk_level == "MEDIUM":

            constraint = self.safe_envelope.apply(
                proposed_action
            )

            return GovernanceDecision(
                status="CONSTRAIN",
                action=constraint.action,
                proposed_action=proposed_action,
                policy=policy_decision,
                risk=risk_assessment,
                modified=constraint.modified,
                reason="MEDIUM_RISK_SAFE_ENVELOPE",
            )

        # ---------------------------------------------------------
        # 7. LOW RISK -> ALLOW
        # ---------------------------------------------------------

        return GovernanceDecision(
            status="ALLOW",
            action=proposed_action,
            proposed_action=proposed_action,
            policy=policy_decision,
            risk=risk_assessment,
            modified=False,
            reason="GOVERNANCE_APPROVED",
        )

