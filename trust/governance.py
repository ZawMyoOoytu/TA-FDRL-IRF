
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import (
    PolicyEngine,
    PolicyConfig,
    PolicyDecision,
)

from .risk import (
    RiskEngine,
    RiskConfig,
    RiskAssessment,
)

from .safe_envelope import (
    SafeEnvelope,
    SafeEnvelopeConfig,
)

from .trust_engine import (
    TrustEngine,
    TrustEngineConfig,
    TrustAssessment,
)


# ============================================================
# GOVERNANCE DECISION
# ============================================================

@dataclass
class GovernanceDecision:
    """
    Result of a PRE-EXECUTION governance evaluation.

    The decision is generated before the proposed action
    reaches the IRF environment.
    """

    status: str

    action: np.ndarray
    proposed_action: np.ndarray

    policy: PolicyDecision
    risk: RiskAssessment

    modified: bool
    reason: str

    trust: TrustAssessment | None = None


# ============================================================
# GOVERNANCE ENGINE
# ============================================================

class GovernanceEngine:
    """
    High-level governance layer for TA-FDRL-IRF.

    Governance pipeline
    -------------------

        Proposed Action
              |
              v
        Trust Assessment
              |
              v
        Policy Evaluation
              |
              v
        Risk Evaluation
              |
        +-----+------+------+
        |            |      |
        v            v      v
      ALLOW      CONSTRAIN  BLOCK
        |            |      |
        +------------+------+
                     |
                     v
              IRF Environment
                     |
                     v
              Execution Result
                     |
                     v
          Post-Execution Feedback
                     |
                     v
               Trust Update


    PRE-EXECUTION
    --------------

    evaluate()

        - reads current environment trust
        - derives governance trust
        - evaluates candidate action
        - evaluates policy
        - evaluates risk
        - does NOT mutate trust
        - does NOT write execution history


    POST-EXECUTION
    --------------

    update_trust()

        - receives actual executed action
        - receives actual reward
        - receives environment trust
        - receives policy result
        - receives risk result
        - records execution feedback exactly once
        - updates governance trust exactly once


    GovernanceEngine does not modify:

        - SAC actor parameters
        - SAC critic parameters
        - replay buffer
        - IRF trust equations
        - IRF environment dynamics
        - SAC learning behavior
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        policy_engine=None,
        risk_engine=None,
        safe_envelope=None,
        trust_engine=None,
    ):
        """
        Initialize the governance subsystem.

        Engines can be injected for controlled experiments,
        deterministic testing, and branch-coverage validation.
        """

        # ----------------------------------------------------
        # Policy Engine
        # ----------------------------------------------------

        self.policy = (
            policy_engine
            if policy_engine is not None
            else PolicyEngine(
                PolicyConfig()
            )
        )

        # ----------------------------------------------------
        # Risk Engine
        # ----------------------------------------------------

        self.risk = (
            risk_engine
            if risk_engine is not None
            else RiskEngine(
                RiskConfig()
            )
        )

        # ----------------------------------------------------
        # Safe Envelope
        # ----------------------------------------------------

        self.safe_envelope = (
            safe_envelope
            if safe_envelope is not None
            else SafeEnvelope(
                SafeEnvelopeConfig()
            )
        )

        # ----------------------------------------------------
        # Trust Engine
        # ----------------------------------------------------

        self.trust_engine = (
            trust_engine
            if trust_engine is not None
            else TrustEngine(
                TrustEngineConfig()
            )
        )

    # ========================================================
    # SAFE NUMERIC HELPER
    # ========================================================

    @staticmethod
    def _safe_float(
        value,
        default=0.0,
    ):
        """
        Convert value to a finite float.

        Invalid or non-finite values are replaced by default.
        """

        try:
            result = float(value)

            if not np.isfinite(result):
                return float(default)

            return result

        except Exception:
            return float(default)

    # ========================================================
    # PRE-EXECUTION GOVERNANCE
    # ========================================================

    def evaluate(
        self,
        action,
        trust_score=None,
        telemetry=None,
    ):
        """
        Evaluate a proposed action before execution.

        This method is strictly PRE-EXECUTION.

        It does not:

            - mutate governance trust
            - write execution history
            - record outcome history
            - update policy history
            - update risk history
        """

        # ----------------------------------------------------
        # Normalize proposed action
        # ----------------------------------------------------

        proposed_action = np.asarray(
            action,
            dtype=np.float32,
        ).copy()

        # ----------------------------------------------------
        # 1. DOMAIN TRUST -> GOVERNANCE TRUST
        # ----------------------------------------------------

        if trust_score is None:

            environment_trust = (
                self.trust_engine.environment_trust
            )

        else:

            environment_trust = self._safe_float(
                trust_score,
                default=0.0,
            )

        trust_assessment = (
            self.trust_engine.evaluate(
                environment_trust=environment_trust,
                action=proposed_action,
            )
        )

        governance_trust = (
            trust_assessment.trust_score
        )

        # ----------------------------------------------------
        # 2. POLICY EVALUATION
        # ----------------------------------------------------

        policy_decision = self.policy.evaluate(
            proposed_action
        )

        # ----------------------------------------------------
        # 3. RISK EVALUATION
        # ----------------------------------------------------

        risk_assessment = self.risk.evaluate(
            proposed_action,
            trust_score=governance_trust,
            telemetry=telemetry,
        )

        # ====================================================
        # 4. POLICY VIOLATION -> HARD BLOCK
        # ====================================================

        if not policy_decision.allowed:

            return GovernanceDecision(
                status="BLOCK",
                action=proposed_action,
                proposed_action=proposed_action,
                policy=policy_decision,
                risk=risk_assessment,
                modified=False,
                reason="POLICY_VIOLATION",
                trust=trust_assessment,
            )

        # ====================================================
        # 5. HIGH RISK -> HARD BLOCK
        # ====================================================

        if risk_assessment.risk_level == "HIGH":

            return GovernanceDecision(
                status="BLOCK",
                action=proposed_action,
                proposed_action=proposed_action,
                policy=policy_decision,
                risk=risk_assessment,
                modified=False,
                reason="HIGH_RISK",
                trust=trust_assessment,
            )

        # ====================================================
        # 6. LOW GOVERNANCE TRUST -> HARD BLOCK
        # ====================================================

        if (
            governance_trust
            < self.risk.config.minimum_trust
        ):

            return GovernanceDecision(
                status="BLOCK",
                action=proposed_action,
                proposed_action=proposed_action,
                policy=policy_decision,
                risk=risk_assessment,
                modified=False,
                reason="TRUST_BELOW_MINIMUM",
                trust=trust_assessment,
            )

        # ====================================================
        # 7. MEDIUM RISK -> SAFE ENVELOPE
        # ====================================================

        if risk_assessment.risk_level == "MEDIUM":

            constraint = (
                self.safe_envelope.apply(
                    proposed_action
                )
            )

            return GovernanceDecision(
                status="CONSTRAIN",
                action=constraint.action,
                proposed_action=proposed_action,
                policy=policy_decision,
                risk=risk_assessment,
                modified=constraint.modified,
                reason="MEDIUM_RISK_SAFE_ENVELOPE",
                trust=trust_assessment,
            )

        # ====================================================
        # 8. LOW RISK -> ALLOW
        # ====================================================

        return GovernanceDecision(
            status="ALLOW",
            action=proposed_action,
            proposed_action=proposed_action,
            policy=policy_decision,
            risk=risk_assessment,
            modified=False,
            reason="GOVERNANCE_APPROVED",
            trust=trust_assessment,
        )

    # ========================================================
    # POST-EXECUTION TRUST FEEDBACK
    # ========================================================

    def update_trust(
        self,
        reward=None,
        environment_trust=None,
        action=None,
        decision=None,
        telemetry=None,
    ):
        """
        Process POST-EXECUTION governance feedback.

        This method must be called AFTER env.step().

        Parameters
        ----------
        reward:
            Actual reward returned by the environment.

        environment_trust:
            Actual/current domain-level trust after execution.

        action:
            ACTUAL EXECUTED ACTION.

            This must be the action that actually reached
            the IRF environment.

        decision:
            GovernanceDecision generated by evaluate().

        telemetry:
            Post-execution environment telemetry.

        Returns
        -------
        TrustAssessment
            Updated governance trust assessment.
        """

        # ----------------------------------------------------
        # Initialize feedback fields
        # ----------------------------------------------------

        policy_allowed = None
        risk_score = None

        # ----------------------------------------------------
        # Extract policy + risk from governance decision
        # ----------------------------------------------------

        if decision is not None:

            policy = getattr(
                decision,
                "policy",
                None,
            )

            risk = getattr(
                decision,
                "risk",
                None,
            )

            # ------------------------------------------------
            # Policy result
            # ------------------------------------------------

            if policy is not None:

                policy_allowed = bool(
                    getattr(
                        policy,
                        "allowed",
                        True,
                    )
                )

            # ------------------------------------------------
            # Risk result
            # ------------------------------------------------

            if risk is not None:

                risk_score = self._safe_float(
                    getattr(
                        risk,
                        "risk_score",
                        0.0,
                    )
                )

        # ----------------------------------------------------
        # Fallback risk extraction from telemetry
        # ----------------------------------------------------

        if (
            risk_score is None
            and telemetry is not None
        ):

            risk_score = telemetry.get(
                "risk_score",
                None,
            )

            if risk_score is not None:

                risk_score = self._safe_float(
                    risk_score
                )

        # ----------------------------------------------------
        # POST-EXECUTION VALIDATION
        # ----------------------------------------------------

        if action is None:

            raise ValueError(
                "Post-execution trust feedback requires "
                "the actual executed action."
            )

        # ----------------------------------------------------
        # Delegate to TrustEngine
        # ----------------------------------------------------
        #
        # IMPORTANT:
        #
        # Do NOT call trust_engine.evaluate() here.
        #
        # evaluate()
        #     = PRE-EXECUTION assessment
        #
        # post_execution_feedback()
        #     = execution-history recording
        #       + governance trust mutation
        #
        # This guarantees one execution event produces
        # exactly one history entry and one trust update.
        # ----------------------------------------------------

        assessment = (
            self.trust_engine.post_execution_feedback(
                environment_trust=environment_trust,
                action=action,
                policy_allowed=policy_allowed,
                risk_score=risk_score,
                reward=reward,
            )
        )

        return assessment

    # ========================================================
    # TRUST STATE
    # ========================================================

    def get_trust_state(self):
        """
        Return the serializable current TrustEngine state.
        """

        return self.trust_engine.get_state()

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        initial_trust=None,
    ):
        """
        Reset governance-level trust state and histories.

        The underlying IRF environment trust equation is not
        modified by this method.
        """

        self.trust_engine.reset(
            initial_trust=initial_trust
        )

