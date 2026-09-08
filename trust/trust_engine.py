from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ============================================================
# TRUST ENGINE CONFIGURATION
# ============================================================

@dataclass
class TrustEngineConfig:
    """
    Governance-level Trust Engine configuration.

    Architectural boundary
    ----------------------
    This engine does NOT replace the scientific trust model
    inside IRFEnvironment.

    IRFEnvironment remains the domain-level trust authority.

    TrustEngine maintains governance-level evidence:

        - environment trust
        - reliability
        - policy compliance
        - behavioral stability
        - execution outcomes
        - governance trust history

    The TrustEngine does not modify:

        - SAC actor
        - SAC critic
        - replay buffer
        - IRF channel model
        - IRF adaptive trust equation
    """

    # --------------------------------------------------------
    # Initial trust
    # --------------------------------------------------------

    initial_trust: float = 1.0

    # --------------------------------------------------------
    # Trust bounds
    # --------------------------------------------------------

    minimum_trust: float = 0.0
    maximum_trust: float = 1.0

    # --------------------------------------------------------
    # Trust memory
    # --------------------------------------------------------

    trust_memory: float = 0.90

    # --------------------------------------------------------
    # Evidence weights
    # --------------------------------------------------------

    environment_trust_weight: float = 0.50
    reliability_weight: float = 0.20
    policy_compliance_weight: float = 0.15
    behavioral_stability_weight: float = 0.15

    # --------------------------------------------------------
    # Outcome classification
    # --------------------------------------------------------

    positive_outcome_threshold: float = 0.0
    negative_outcome_threshold: float = -0.20

    # --------------------------------------------------------
    # Compatibility parameters
    # --------------------------------------------------------

    positive_update_rate: float = 0.02
    negative_update_rate: float = 0.05

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history_size: int = 200


# ============================================================
# TRUST ASSESSMENT
# ============================================================

@dataclass
class TrustAssessment:
    """
    Result returned by TrustEngine evaluation methods.
    """

    trust_score: float

    environment_trust: float

    reliability_score: float

    policy_compliance: float

    behavioral_stability: float

    outcome_score: float

    history_length: int

    reasons: list[str] = field(
        default_factory=list
    )


# ============================================================
# TRUST ENGINE
# ============================================================

class TrustEngine:
    """
    Governance-level Trust Engine.

    Control loop
    ------------
        Observe
            ↓
        Evaluate
            ↓
        Govern
            ↓
        Execute
            ↓
        Feedback
            ↓
        Trust Update
            ↓
        Next Decision

    IMPORTANT HISTORY SEMANTICS
    ---------------------------
    One environment execution step produces exactly one
    execution evidence record.

    PRE-EXECUTION
    -------------
    evaluate()
        - synchronizes current environment trust
        - calculates a provisional behavioral assessment
        - DOES NOT append action/risk/policy histories
        - DOES NOT update governance trust

    POST-EXECUTION
    --------------
    post_execution_feedback()
        - records the actual executed action
        - records policy result
        - records risk
        - records execution outcome
        - recalculates reliability/stability
        - updates governance trust exactly once
        - records trust history exactly once

    Therefore:

        N execution steps
            =
        N action records
        N risk records
        N policy records
        N outcome records
    """

    def __init__(
        self,
        config: TrustEngineConfig | None = None,
    ):
        self.config = (
            config
            if config is not None
            else TrustEngineConfig()
        )

        self.reset()

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        initial_trust: float | None = None,
    ) -> None:
        """
        Reset governance-level trust state.
        """

        if initial_trust is None:
            initial_trust = (
                self.config.initial_trust
            )

        self.trust_score = float(
            np.clip(
                initial_trust,
                self.config.minimum_trust,
                self.config.maximum_trust,
            )
        )

        # ----------------------------------------------------
        # Domain-level trust
        # ----------------------------------------------------

        self.environment_trust = (
            self.trust_score
        )

        # ----------------------------------------------------
        # Governance evidence
        # ----------------------------------------------------

        self.reliability_score = 1.0

        self.policy_compliance = 1.0

        self.behavioral_stability = 1.0

        self.outcome_score = 0.0

        # ----------------------------------------------------
        # Histories
        #
        # IMPORTANT:
        # These histories represent EXECUTION evidence.
        # They must not be written during pre-execution
        # assessment.
        # ----------------------------------------------------

        self.action_history: list[np.ndarray] = []

        self.trust_history: list[float] = [
            self.trust_score
        ]

        self.outcome_history: list[float] = []

        self.policy_history: list[bool] = []

        self.risk_history: list[float] = []

    # ========================================================
    # NUMERIC HELPERS
    # ========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Safely convert a value to a finite float.
        """

        try:
            result = float(value)

            if not np.isfinite(result):
                return float(default)

            return result

        except Exception:
            return float(default)

    @staticmethod
    def _clip01(
        value: float,
    ) -> float:
        """
        Clip scalar to [0, 1].
        """

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    # ========================================================
    # ENVIRONMENT TRUST
    # ========================================================

    def synchronize_environment_trust(
        self,
        environment_trust: float,
    ) -> float:
        """
        Synchronize with domain-level trust from IRFEnvironment.
        """

        trust = self._clip01(
            self._safe_float(
                environment_trust,
                default=0.0,
            )
        )

        self.environment_trust = trust

        return trust

    # ========================================================
    # ACTION HISTORY
    # ========================================================

    def record_action(
        self,
        action: np.ndarray,
    ) -> bool:
        """
        Record an executed action.

        Returns
        -------
        bool
            True if the action was successfully recorded.
        """

        action_array = np.asarray(
            action,
            dtype=np.float32,
        ).copy()

        if action_array.ndim != 1:
            return False

        if not np.all(
            np.isfinite(action_array)
        ):
            return False

        self.action_history.append(
            action_array
        )

        if (
            len(self.action_history)
            > self.config.history_size
        ):
            self.action_history.pop(0)

        return True

    # ========================================================
    # BEHAVIORAL STABILITY
    # ========================================================

    def calculate_behavioral_stability(
        self,
        action: np.ndarray | None = None,
        record: bool = True,
    ) -> float:
        """
        Estimate behavioral stability from action changes.

        Parameters
        ----------
        action:
            Action to optionally evaluate.

        record:
            If True, action is appended to execution history.

            IMPORTANT:
            Pre-execution evaluation must use record=False.

        Result
        ------
        1.0 = highly stable
        0.0 = highly unstable
        """

        if action is not None and record:

            recorded = self.record_action(
                action
            )

            if not recorded:
                self.behavioral_stability = 0.0
                return 0.0

        if len(
            self.action_history
        ) < 2:

            self.behavioral_stability = 1.0

            return 1.0

        previous = (
            self.action_history[-2]
        )

        current = (
            self.action_history[-1]
        )

        if previous.shape != current.shape:

            self.behavioral_stability = 0.0

            return 0.0

        delta = np.abs(
            current - previous
        )

        mean_delta = self._safe_float(
            np.mean(delta)
        )

        stability = (
            1.0
            - np.clip(
                mean_delta,
                0.0,
                1.0,
            )
        )

        self.behavioral_stability = float(
            stability
        )

        return self.behavioral_stability

    # ========================================================
    # POLICY COMPLIANCE
    # ========================================================

    def record_policy_result(
        self,
        policy_allowed: bool,
    ) -> float:
        """
        Record execution-time policy result.

        This method MUST only be called once for each
        governed execution event.
        """

        allowed = bool(
            policy_allowed
        )

        self.policy_history.append(
            allowed
        )

        if (
            len(self.policy_history)
            > self.config.history_size
        ):
            self.policy_history.pop(0)

        if not self.policy_history:

            self.policy_compliance = 1.0

        else:

            self.policy_compliance = float(
                np.mean(
                    np.asarray(
                        self.policy_history,
                        dtype=np.float32,
                    )
                )
            )

        return self.policy_compliance

    # ========================================================
    # RISK HISTORY
    # ========================================================

    def record_risk(
        self,
        risk_score: float,
    ) -> float:
        """
        Record execution-time governance risk.
        """

        risk = self._clip01(
            self._safe_float(
                risk_score
            )
        )

        self.risk_history.append(
            risk
        )

        if (
            len(self.risk_history)
            > self.config.history_size
        ):
            self.risk_history.pop(0)

        return risk

    # ========================================================
    # RELIABILITY
    # ========================================================

    def calculate_reliability(
        self,
    ) -> float:
        """
        Calculate reliability from historical risk.

            reliability = 1 - mean(historical risk)
        """

        if not self.risk_history:

            self.reliability_score = 1.0

            return 1.0

        mean_risk = self._safe_float(
            np.mean(
                np.asarray(
                    self.risk_history,
                    dtype=np.float32,
                )
            )
        )

        reliability = (
            1.0
            - np.clip(
                mean_risk,
                0.0,
                1.0,
            )
        )

        self.reliability_score = float(
            reliability
        )

        return self.reliability_score

    # ========================================================
    # OUTCOME FEEDBACK
    # ========================================================

    def record_outcome(
        self,
        reward: float,
    ) -> float:
        """
        Convert execution reward into bounded outcome evidence.

            positive reward -> +1
            neutral reward  ->  0
            negative reward -> -1

        This method records evidence only.

        It does NOT directly modify trust_score.
        """

        reward_value = self._safe_float(
            reward
        )

        if (
            reward_value
            > self.config.positive_outcome_threshold
        ):

            outcome = 1.0

        elif (
            reward_value
            < self.config.negative_outcome_threshold
        ):

            outcome = -1.0

        else:

            outcome = 0.0

        self.outcome_history.append(
            outcome
        )

        if (
            len(self.outcome_history)
            > self.config.history_size
        ):
            self.outcome_history.pop(0)

        if self.outcome_history:

            self.outcome_score = self._safe_float(
                np.mean(
                    np.asarray(
                        self.outcome_history,
                        dtype=np.float32,
                    )
                )
            )

        else:

            self.outcome_score = 0.0

        return outcome

    # ========================================================
    # COMPOSITE TRUST
    # ========================================================

    def _calculate_composite_trust(
        self,
    ) -> float:
        """
        Calculate governance trust from bounded evidence.

            T_composite =
                w_env      * environment_trust
              + w_reliable * reliability
              + w_policy   * policy_compliance
              + w_behavior * behavioral_stability

        Outcome evidence remains separately observable.

        This preserves the current Phase-1 trust model and
        avoids mixing arbitrary reward scales directly into
        the trust equation.
        """

        weights = np.asarray(
            [
                self.config.environment_trust_weight,
                self.config.reliability_weight,
                self.config.policy_compliance_weight,
                self.config.behavioral_stability_weight,
            ],
            dtype=np.float64,
        )

        values = np.asarray(
            [
                self.environment_trust,
                self.reliability_score,
                self.policy_compliance,
                self.behavioral_stability,
            ],
            dtype=np.float64,
        )

        weight_sum = float(
            np.sum(weights)
        )

        if weight_sum <= 0.0:
            return self.environment_trust

        composite_trust = float(
            np.dot(
                weights,
                values,
            )
            / weight_sum
        )

        return self._clip01(
            composite_trust
        )

    # ========================================================
    # TRUST REASONS
    # ========================================================

    def _build_reasons(
        self,
    ) -> list[str]:
        """
        Build human-readable governance trust reasons.
        """

        reasons: list[str] = []

        if self.trust_score < 0.60:

            reasons.append(
                "LOW_GOVERNANCE_TRUST"
            )

        if self.reliability_score < 0.50:

            reasons.append(
                "LOW_RELIABILITY"
            )

        if self.policy_compliance < 0.80:

            reasons.append(
                "POLICY_COMPLIANCE_DEGRADED"
            )

        if self.behavioral_stability < 0.50:

            reasons.append(
                "BEHAVIORAL_INSTABILITY"
            )

        if self.outcome_score < -0.50:

            reasons.append(
                "NEGATIVE_EXECUTION_OUTCOMES"
            )

        return reasons

    # ========================================================
    # TRUST UPDATE
    # ========================================================

    def _update_governance_trust(
        self,
    ) -> None:
        """
        Update governance trust exactly once.

            T_t =
                memory * T_(t-1)
                +
                (1-memory) * T_composite
        """

        composite_trust = (
            self._calculate_composite_trust()
        )

        memory = float(
            np.clip(
                self.config.trust_memory,
                0.0,
                1.0,
            )
        )

        self.trust_score = (
            memory * self.trust_score
            + (1.0 - memory)
            * composite_trust
        )

        self.trust_score = float(
            np.clip(
                self.trust_score,
                self.config.minimum_trust,
                self.config.maximum_trust,
            )
        )

    # ========================================================
    # TRUST HISTORY
    # ========================================================

    def _record_trust_history(
        self,
    ) -> None:
        """
        Record one post-update governance trust state.
        """

        self.trust_history.append(
            self.trust_score
        )

        if (
            len(self.trust_history)
            > self.config.history_size
        ):
            self.trust_history.pop(0)

    # ========================================================
    # ASSESSMENT BUILDER
    # ========================================================

    def _build_assessment(
        self,
        reasons: list[str] | None = None,
    ) -> TrustAssessment:
        """
        Build a JSON/telemetry-friendly TrustAssessment.
        """

        if reasons is None:
            reasons = self._build_reasons()

        return TrustAssessment(
            trust_score=float(
                self.trust_score
            ),

            environment_trust=float(
                self.environment_trust
            ),

            reliability_score=float(
                self.reliability_score
            ),

            policy_compliance=float(
                self.policy_compliance
            ),

            behavioral_stability=float(
                self.behavioral_stability
            ),

            outcome_score=float(
                self.outcome_score
            ),

            history_length=len(
                self.trust_history
            ),

            reasons=list(reasons),
        )

    # ========================================================
    # PRE-EXECUTION EVALUATION
    # ========================================================

    def evaluate(
        self,
        environment_trust: float | None = None,
        action: np.ndarray | None = None,
        policy_allowed: bool | None = None,
        risk_score: float | None = None,
        reward: float | None = None,
    ) -> TrustAssessment:
        """
        PRE-EXECUTION governance trust assessment.

        IMPORTANT
        ---------
        This method is intentionally READ/ASSESS oriented.

        It does NOT append:

            - action_history
            - risk_history
            - policy_history
            - outcome_history

        It does NOT update governance trust.

        Therefore calling evaluate() multiple times before
        execution does not create fake execution history.

        The supplied action/policy/risk values are used only
        for a provisional assessment.
        """

        # ----------------------------------------------------
        # Environment trust
        # ----------------------------------------------------

        if environment_trust is not None:

            self.synchronize_environment_trust(
                environment_trust
            )

        # ----------------------------------------------------
        # Provisional evidence
        #
        # Do NOT write histories here.
        # ----------------------------------------------------

        provisional_reliability = (
            self.reliability_score
        )

        provisional_policy = (
            self.policy_compliance
        )

        provisional_stability = (
            self.behavioral_stability
        )

        # ----------------------------------------------------
        # Provisional risk assessment
        # ----------------------------------------------------

        if risk_score is not None:

            risk = self._clip01(
                self._safe_float(
                    risk_score
                )
            )

            # Use the incoming risk for a provisional
            # reliability estimate without recording it.
            if self.risk_history:

                previous_mean_risk = self._safe_float(
                    np.mean(
                        np.asarray(
                            self.risk_history,
                            dtype=np.float32,
                        )
                    )
                )

                provisional_reliability = self._clip01(
                    1.0
                    - (
                        (
                            previous_mean_risk
                            * len(self.risk_history)
                        )
                        + risk
                    )
                    / (
                        len(self.risk_history)
                        + 1
                    )
                )

            else:

                provisional_reliability = (
                    1.0 - risk
                )

        # ----------------------------------------------------
        # Provisional policy evidence
        # ----------------------------------------------------

        if policy_allowed is not None:

            allowed = 1.0 if bool(
                policy_allowed
            ) else 0.0

            if self.policy_history:

                provisional_policy = self._clip01(
                    (
                        float(
                            np.sum(
                                np.asarray(
                                    self.policy_history,
                                    dtype=np.float32,
                                )
                            )
                        )
                        + allowed
                    )
                    / (
                        len(self.policy_history)
                        + 1
                    )
                )

            else:

                provisional_policy = allowed

        # ----------------------------------------------------
        # Provisional behavioral stability
        # ----------------------------------------------------

        if action is not None:

            action_array = np.asarray(
                action,
                dtype=np.float32,
            )

            if (
                action_array.ndim == 1
                and np.all(
                    np.isfinite(action_array)
                )
            ):

                if self.action_history:

                    previous = (
                        self.action_history[-1]
                    )

                    if previous.shape == action_array.shape:

                        delta = np.abs(
                            action_array
                            - previous
                        )

                        mean_delta = self._safe_float(
                            np.mean(delta)
                        )

                        provisional_stability = (
                            1.0
                            - np.clip(
                                mean_delta,
                                0.0,
                                1.0,
                            )
                        )

        # ----------------------------------------------------
        # Provisional composite
        # ----------------------------------------------------

        weights = np.asarray(
            [
                self.config.environment_trust_weight,
                self.config.reliability_weight,
                self.config.policy_compliance_weight,
                self.config.behavioral_stability_weight,
            ],
            dtype=np.float64,
        )

        values = np.asarray(
            [
                self.environment_trust,
                provisional_reliability,
                provisional_policy,
                provisional_stability,
            ],
            dtype=np.float64,
        )

        weight_sum = float(
            np.sum(weights)
        )

        if weight_sum > 0.0:

            provisional_composite = float(
                np.dot(
                    weights,
                    values,
                )
                / weight_sum
            )

        else:

            provisional_composite = (
                self.environment_trust
            )

        provisional_composite = self._clip01(
            provisional_composite
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Do not change self.trust_score during pre-execution
        # evaluation.
        #
        # We return the current governance trust.
        # ----------------------------------------------------

        reasons: list[str] = []

        if self.trust_score < 0.60:

            reasons.append(
                "LOW_GOVERNANCE_TRUST"
            )

        if provisional_reliability < 0.50:

            reasons.append(
                "LOW_RELIABILITY"
            )

        if provisional_policy < 0.80:

            reasons.append(
                "POLICY_COMPLIANCE_DEGRADED"
            )

        if provisional_stability < 0.50:

            reasons.append(
                "BEHAVIORAL_INSTABILITY"
            )

        if (
            reward is not None
            and self._safe_float(reward)
            < self.config.negative_outcome_threshold
        ):

            reasons.append(
                "NEGATIVE_PROVISIONAL_OUTCOME"
            )

        # ----------------------------------------------------
        # Return current trust state.
        #
        # Provisional composite is intentionally not stored
        # as governance trust.
        # ----------------------------------------------------

        return TrustAssessment(
            trust_score=float(
                self.trust_score
            ),

            environment_trust=float(
                self.environment_trust
            ),

            reliability_score=float(
                provisional_reliability
            ),

            policy_compliance=float(
                provisional_policy
            ),

            behavioral_stability=float(
                provisional_stability
            ),

            outcome_score=float(
                self.outcome_score
            ),

            history_length=len(
                self.trust_history
            ),

            reasons=reasons,
        )

    # ========================================================
    # POST-EXECUTION FEEDBACK
    # ========================================================

    def post_execution_feedback(
        self,
        reward: float | None = None,
        environment_trust: float | None = None,
        action: np.ndarray | None = None,
        policy_allowed: bool | None = None,
        risk_score: float | None = None,
    ) -> TrustAssessment:
        """
        POST-EXECUTION governance update.

        Exactly one call represents one completed governance
        execution event.

        This method:

            1. records executed action
            2. records policy result
            3. records risk
            4. records outcome
            5. recalculates behavioral stability
            6. recalculates reliability
            7. updates governance trust exactly once
            8. records trust history exactly once
        """

        # ----------------------------------------------------
        # Environment trust
        # ----------------------------------------------------

        if environment_trust is not None:

            self.synchronize_environment_trust(
                environment_trust
            )

        # ----------------------------------------------------
        # ACTION
        #
        # Exactly one action history entry per execution.
        # ----------------------------------------------------

        if action is not None:

            recorded = self.record_action(
                action
            )

            if not recorded:

                self.behavioral_stability = 0.0

            else:

                self.calculate_behavioral_stability(
                    action=None,
                    record=False,
                )

        # ----------------------------------------------------
        # POLICY
        #
        # Exactly one policy history entry per execution.
        # ----------------------------------------------------

        if policy_allowed is not None:

            self.record_policy_result(
                policy_allowed
            )

        # ----------------------------------------------------
        # RISK
        #
        # Exactly one risk history entry per execution.
        # ----------------------------------------------------

        if risk_score is not None:

            self.record_risk(
                risk_score
            )

        # ----------------------------------------------------
        # RELIABILITY
        # ----------------------------------------------------

        self.calculate_reliability()

        # ----------------------------------------------------
        # OUTCOME
        #
        # Exactly one outcome history entry per execution.
        # ----------------------------------------------------

        if reward is not None:

            self.record_outcome(
                reward
            )

        # ----------------------------------------------------
        # TRUST UPDATE
        #
        # Exactly once per execution.
        # ----------------------------------------------------

        self._update_governance_trust()

        # ----------------------------------------------------
        # TRUST HISTORY
        #
        # Exactly one new trust state per execution.
        # ----------------------------------------------------

        self._record_trust_history()

        # ----------------------------------------------------
        # Reasons
        # ----------------------------------------------------

        reasons = self._build_reasons()

        # ----------------------------------------------------
        # Assessment
        # ----------------------------------------------------

        return self._build_assessment(
            reasons=reasons
        )

    # ========================================================
    # STATE
    # ========================================================

    def get_state(
        self,
    ) -> dict[str, Any]:
        """
        Return current TrustEngine state.

        Values are JSON/telemetry friendly.
        """

        return {
            "trust_score": float(
                self.trust_score
            ),

            "environment_trust": float(
                self.environment_trust
            ),

            "reliability_score": float(
                self.reliability_score
            ),

            "policy_compliance": float(
                self.policy_compliance
            ),

            "behavioral_stability": float(
                self.behavioral_stability
            ),

            "outcome_score": float(
                self.outcome_score
            ),

            "history_length": len(
                self.trust_history
            ),

            "action_history_length": len(
                self.action_history
            ),

            "risk_history_length": len(
                self.risk_history
            ),

            "policy_history_length": len(
                self.policy_history
            ),

            "outcome_history_length": len(
                self.outcome_history
            ),
        }