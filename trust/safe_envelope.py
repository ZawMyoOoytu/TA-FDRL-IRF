
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafeEnvelopeConfig:
    """
    Engineering safety envelope for medium-risk actions.

    Action layout:
        [0:20]   -> bandwidth allocation actions
        [20:40]  -> power allocation actions

    NOTE:
        These limits are initial engineering parameters.
        They are NOT claimed to be scientifically optimal.
    """

    action_min: float = -1.0
    action_max: float = 1.0

    bandwidth_limit: float = 0.85
    power_limit: float = 0.75

    expected_action_dim: int = 40


@dataclass
class ConstraintResult:
    action: np.ndarray
    modified: bool
    max_delta: float
    mean_delta: float
    violations_before: list[str]
    violations_after: list[str]


class SafeEnvelope:
    """
    Applies a configurable safety envelope to an action.

    The operator is intended for MEDIUM-risk governance decisions.

    It does not decide whether an action should be blocked.
    It only transforms an otherwise executable action into
    a safer bounded action.
    """

    def __init__(self, config: SafeEnvelopeConfig | None = None):
        self.config = (
            config if config is not None else SafeEnvelopeConfig()
        )

    def _validate(self, action: np.ndarray) -> list[str]:
        violations: list[str] = []

        if action.ndim != 1:
            violations.append("ACTION_NOT_1D")
            return violations

        if action.shape[0] != self.config.expected_action_dim:
            violations.append("ACTION_DIMENSION_MISMATCH")
            return violations

        if not np.all(np.isfinite(action)):
            violations.append("NON_FINITE_ACTION")
            return violations

        if np.any(action < self.config.action_min):
            violations.append("ACTION_BELOW_MINIMUM")

        if np.any(action > self.config.action_max):
            violations.append("ACTION_ABOVE_MAXIMUM")

        bandwidth = action[:20]
        power = action[20:]

        if np.any(bandwidth > self.config.bandwidth_limit):
            violations.append("BANDWIDTH_ENVELOPE_EXCEEDED")

        if np.any(power > self.config.power_limit):
            violations.append("POWER_ENVELOPE_EXCEEDED")

        return violations

    def apply(self, action) -> ConstraintResult:
        proposed = np.asarray(action, dtype=np.float32).copy()

        violations_before = self._validate(proposed)

        # Preserve the original action if the input is structurally invalid.
        if proposed.ndim != 1:
            return ConstraintResult(
                action=proposed,
                modified=False,
                max_delta=0.0,
                mean_delta=0.0,
                violations_before=violations_before,
                violations_after=violations_before.copy(),
            )

        if proposed.shape[0] != self.config.expected_action_dim:
            return ConstraintResult(
                action=proposed,
                modified=False,
                max_delta=0.0,
                mean_delta=0.0,
                violations_before=violations_before,
                violations_after=violations_before.copy(),
            )

        if not np.all(np.isfinite(proposed)):
            return ConstraintResult(
                action=proposed,
                modified=False,
                max_delta=0.0,
                mean_delta=0.0,
                violations_before=violations_before,
                violations_after=violations_before.copy(),
            )

        constrained = proposed.copy()

        # Global action safety.
        constrained = np.clip(
            constrained,
            self.config.action_min,
            self.config.action_max,
        )

        # Bandwidth envelope.
        constrained[:20] = np.clip(
            constrained[:20],
            self.config.action_min,
            self.config.bandwidth_limit,
        )

        # Power envelope.
        constrained[20:] = np.clip(
            constrained[20:],
            self.config.action_min,
            self.config.power_limit,
        )

        delta = np.abs(constrained - proposed)

        max_delta = float(np.max(delta))
        mean_delta = float(np.mean(delta))
        modified = bool(max_delta > 0.0)

        violations_after = self._validate(constrained)

        return ConstraintResult(
            action=constrained.astype(np.float32),
            modified=modified,
            max_delta=max_delta,
            mean_delta=mean_delta,
            violations_before=violations_before,
            violations_after=violations_after,
        )

