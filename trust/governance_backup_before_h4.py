from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .policy import PolicyEngine, PolicyConfig, PolicyDecision
from .risk import RiskEngine, RiskConfig, RiskAssessment


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

        ALLOW
            - policy compliant
            - low risk
            - acceptable trust
    """

    def __init__(
        self,
        policy_engine: PolicyEngine | None = None,
        risk_engine: RiskEngine | None = None,
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

    def evaluate(
        self,
        action,
        trust_score: float | None = None,
        telemetry: dict | None = None,
    ) -> GovernanceDecision:

        proposed_action = np.asarray(
            action,
            dtype=np.float32,
        ).copy()

        # ---------------------------------------------------------
        # Policy evaluation
        # ---------------------------------------------------------

        policy_decision = self.policy.evaluate(
            proposed_action
        )

        # ---------------------------------------------------------
        # Risk evaluation
        # ---------------------------------------------------------

        risk_assessment = self.risk.evaluate(
            proposed_action,
            trust_score=trust_score,
            telemetry=telemetry,
        )

        # ---------------------------------------------------------
        # BLOCK conditions
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
        # CONSTRAIN condition
        # ---------------------------------------------------------

        if risk_assessment.risk_level == "MEDIUM":

            constrained_action = np.clip(
                proposed_action,
                self.policy.config.action_min,
                self.policy.config.action_max,
            ).astype(np.float32)

            return GovernanceDecision(
                status="CONSTRAIN",
                action=constrained_action,
                proposed_action=proposed_action,
                policy=policy_decision,
                risk=risk_assessment,
                modified=not np.array_equal(
                    constrained_action,
                    proposed_action,
                ),
                reason="MEDIUM_RISK_CONSTRAINED",
            )

        # ---------------------------------------------------------
        # ALLOW
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