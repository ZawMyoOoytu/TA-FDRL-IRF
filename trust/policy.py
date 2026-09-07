from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PolicyConfig:
    """
    Governance policy configuration.

    These constraints operate outside the SAC learning core.
    They validate the action proposed by the SAC agent before
    the action is executed by the IRF environment.
    """

    # -----------------------------------------------------
    # Action constraints
    # -----------------------------------------------------

    action_min: float = -1.0
    action_max: float = 1.0

    # -----------------------------------------------------
    # Trust policy
    # -----------------------------------------------------

    minimum_trust: float = 0.60

    # -----------------------------------------------------
    # Risk policy
    # -----------------------------------------------------

    maximum_risk: float = 0.80

    # -----------------------------------------------------
    # Power policy
    #
    # The environment maps the last 20 action values
    # from [-1, 1] to [0, max_power_w].
    #
    # Therefore this policy is expressed in normalized
    # action space rather than modifying the environment.
    # -----------------------------------------------------

    maximum_normalized_power: float = 0.90

    # -----------------------------------------------------
    # Bandwidth policy
    #
    # Bandwidth actions are also represented in normalized
    # action space.
    # -----------------------------------------------------

    maximum_bandwidth_action: float = 1.00

    # -----------------------------------------------------
    # Action dimension
    # -----------------------------------------------------

    expected_action_dim: int = 40


@dataclass
class PolicyDecision:
    """
    Result of policy validation.
    """

    allowed: bool

    action: np.ndarray

    violations: list[str] = field(
        default_factory=list
    )

    reason: str = ""


class PolicyEngine:
    """
    Policy validation layer for TA-FDRL-IRF.

    Important:
        This class does NOT modify the SAC actor,
        critic, replay buffer, or learning algorithm.

    It validates an action proposed by the agent before
    execution by the IRF environment.
    """

    def __init__(
        self,
        config: PolicyConfig | None = None,
    ):

        self.config = (
            config
            if config is not None
            else PolicyConfig()
        )

    # =====================================================
    # Public API
    # =====================================================

    def evaluate(
        self,
        action,
    ) -> PolicyDecision:
        """
        Evaluate a proposed action against governance policy.
        """

        action_array = np.asarray(
            action,
            dtype=np.float32,
        ).copy()

        violations: list[str] = []

        # -------------------------------------------------
        # Dimension validation
        # -------------------------------------------------

        if action_array.ndim != 1:

            violations.append(
                "ACTION_NOT_1D"
            )

            return PolicyDecision(
                allowed=False,
                action=action_array,
                violations=violations,
                reason="ACTION_NOT_1D",
            )

        if (
            action_array.shape[0]
            != self.config.expected_action_dim
        ):

            violations.append(
                "ACTION_DIMENSION_MISMATCH"
            )

            return PolicyDecision(
                allowed=False,
                action=action_array,
                violations=violations,
                reason="ACTION_DIMENSION_MISMATCH",
            )

        # -------------------------------------------------
        # Finite-value validation
        # -------------------------------------------------

        if not np.all(
            np.isfinite(action_array)
        ):

            violations.append(
                "NON_FINITE_ACTION"
            )

            return PolicyDecision(
                allowed=False,
                action=action_array,
                violations=violations,
                reason="NON_FINITE_ACTION",
            )

        # -------------------------------------------------
        # Action range
        # -------------------------------------------------

        if np.any(
            action_array
            < self.config.action_min
        ):

            violations.append(
                "ACTION_BELOW_MINIMUM"
            )

        if np.any(
            action_array
            > self.config.action_max
        ):

            violations.append(
                "ACTION_ABOVE_MAXIMUM"
            )

        # -------------------------------------------------
        # Bandwidth policy
        #
        # First 20 dimensions.
        # -------------------------------------------------

        bandwidth_actions = (
            action_array[:20]
        )

        if np.any(
            np.abs(bandwidth_actions)
            > self.config.maximum_bandwidth_action
        ):

            violations.append(
                "BANDWIDTH_POLICY_VIOLATION"
            )

        # -------------------------------------------------
        # Power policy
        #
        # Last 20 dimensions.
        # -------------------------------------------------

        power_actions = (
            action_array[20:]
        )

        if np.any(
            power_actions
            > self.config.maximum_normalized_power
        ):

            violations.append(
                "POWER_POLICY_VIOLATION"
            )

        # -------------------------------------------------
        # Final decision
        # -------------------------------------------------

        allowed = (
            len(violations) == 0
        )

        if allowed:

            reason = (
                "ACTION_APPROVED"
            )

        else:

            reason = (
                "POLICY_VIOLATION"
            )

        return PolicyDecision(
            allowed=allowed,
            action=action_array,
            violations=violations,
            reason=reason,
        )