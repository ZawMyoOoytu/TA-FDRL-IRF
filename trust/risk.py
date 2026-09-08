from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskConfig:
    """
    Governance-level risk configuration.

    Risk is evaluated independently from the SAC learning core.
    """

    # -----------------------------------------------------
    # Risk thresholds
    # -----------------------------------------------------

    low_risk_threshold: float = 0.30
    medium_risk_threshold: float = 0.60
    high_risk_threshold: float = 0.80

    # -----------------------------------------------------
    # Trust
    # -----------------------------------------------------

    minimum_trust: float = 0.60

    # -----------------------------------------------------
    # Action anomaly
    # -----------------------------------------------------

    action_anomaly_threshold: float = 0.80

    # -----------------------------------------------------
    # Expected dimensions
    # -----------------------------------------------------

    expected_action_dim: int = 40
    expected_state_dim: int = 100


@dataclass
class RiskAssessment:
    """
    Risk assessment result.
    """

    risk_score: float

    risk_level: str

    trust_score: float

    action_anomaly: float

    reasons: list[str]


class RiskEngine:
    """
    Governance-level risk evaluator.

    This engine does not modify the SAC agent or environment.

    It evaluates:
        1. Agent action magnitude/anomaly
        2. Current trust
        3. Optional environment telemetry
    """

    def __init__(
        self,
        config: RiskConfig | None = None,
    ):

        self.config = (
            config
            if config is not None
            else RiskConfig()
        )

    # =====================================================
    # Public API
    # =====================================================

    def evaluate(
        self,
        action,
        trust_score: float | None = None,
        telemetry: dict | None = None,
    ) -> RiskAssessment:
        """
        Evaluate governance-level risk.

        Parameters
        ----------
        action:
            SAC proposed action in [-1, 1].

        trust_score:
            Governance-level trust score.

        telemetry:
            Optional IRF environment telemetry.
        """

        action_array = np.asarray(
            action,
            dtype=np.float32,
        )

        reasons: list[str] = []

        # -------------------------------------------------
        # Validate action shape
        # -------------------------------------------------

        if action_array.ndim != 1:

            return RiskAssessment(
                risk_score=1.0,
                risk_level="HIGH",
                trust_score=0.0,
                action_anomaly=1.0,
                reasons=[
                    "ACTION_NOT_1D"
                ],
            )

        if (
            action_array.shape[0]
            != self.config.expected_action_dim
        ):

            return RiskAssessment(
                risk_score=1.0,
                risk_level="HIGH",
                trust_score=0.0,
                action_anomaly=1.0,
                reasons=[
                    "ACTION_DIMENSION_MISMATCH"
                ],
            )

        # -------------------------------------------------
        # Validate finite values
        # -------------------------------------------------

        if not np.all(
            np.isfinite(action_array)
        ):

            return RiskAssessment(
                risk_score=1.0,
                risk_level="HIGH",
                trust_score=0.0,
                action_anomaly=1.0,
                reasons=[
                    "NON_FINITE_ACTION"
                ],
            )

        # -------------------------------------------------
        # Trust
        # -------------------------------------------------

        if trust_score is None:

            trust = 1.0

        else:

            trust = float(
                np.clip(
                    trust_score,
                    0.0,
                    1.0,
                )
            )

        # -------------------------------------------------
        # Action anomaly
        # -------------------------------------------------
        #
        # SAC normally produces actions in [-1, 1].
        #
        # We measure how strongly the action approaches
        # the extreme boundaries.
        # -------------------------------------------------

        action_magnitude = np.abs(
            action_array
        )

        action_anomaly = float(
            np.mean(
                np.clip(
                    (
                        action_magnitude
                        - 0.70
                    )
                    / 0.30,
                    0.0,
                    1.0,
                )
            )
        )

        if (
            action_anomaly
            >= self.config.action_anomaly_threshold
        ):

            reasons.append(
                "HIGH_ACTION_ANOMALY"
            )

        # -------------------------------------------------
        # Trust risk
        # -------------------------------------------------

        trust_risk = (
            1.0 - trust
        )

        if (
            trust
            < self.config.minimum_trust
        ):

            reasons.append(
                "LOW_TRUST"
            )

        # -------------------------------------------------
        # Telemetry risk
        # -------------------------------------------------

        telemetry_risk = 0.0

        if telemetry is not None:

            # Queue pressure
            queue = telemetry.get(
                "queue"
            )

            if queue is not None:

                queue_array = np.asarray(
                    queue,
                    dtype=np.float32,
                )

                if queue_array.size > 0:

                    queue_pressure = float(
                        np.clip(
                            np.mean(
                                queue_array
                            ),
                            0.0,
                            1.0,
                        )
                    )

                    telemetry_risk = max(
                        telemetry_risk,
                        queue_pressure,
                    )

                    if queue_pressure > 0.80:

                        reasons.append(
                            "HIGH_QUEUE_PRESSURE"
                        )

            # Interference
            interference = telemetry.get(
                "interference"
            )

            if interference is not None:

                interference_array = np.asarray(
                    interference,
                    dtype=np.float32,
                )

                if (
                    interference_array.size > 0
                    and np.all(
                        np.isfinite(
                            interference_array
                        )
                    )
                ):

                    interference_pressure = float(
                        np.clip(
                            np.mean(
                                np.abs(
                                    interference_array
                                )
                            ),
                            0.0,
                            1.0,
                        )
                    )

                    telemetry_risk = max(
                        telemetry_risk,
                        interference_pressure,
                    )

                    if (
                        interference_pressure
                        > 0.80
                    ):

                        reasons.append(
                            "HIGH_INTERFERENCE"
                        )

        # -------------------------------------------------
        # Combined risk
        # -----------------------------------------------------
        #
        # Governance risk is intentionally independent
        # from the SAC reward.
        # -----------------------------------------------------

        risk_score = (
            0.45 * trust_risk
            + 0.35 * action_anomaly
            + 0.20 * telemetry_risk
        )

        risk_score = float(
            np.clip(
                risk_score,
                0.0,
                1.0,
            )
        )

        # -------------------------------------------------
        # Risk level
        # -------------------------------------------------

        if (
            risk_score
            >= self.config.high_risk_threshold
        ):

            risk_level = "HIGH"

        elif (
            risk_score
            >= self.config.medium_risk_threshold
        ):

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"

        return RiskAssessment(
            risk_score=risk_score,
            risk_level=risk_level,
            trust_score=trust,
            action_anomaly=action_anomaly,
            reasons=reasons,
        )