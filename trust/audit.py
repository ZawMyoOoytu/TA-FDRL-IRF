from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass
class AuditRecord:
    """
    Immutable-style governance runtime audit record.
    """

    timestamp: str

    step: int

    governance_status: str

    reason: str

    trust_score: float

    risk_score: float

    risk_level: str

    policy_allowed: bool

    policy_violations: list[str]

    modified: bool

    fallback_used: bool

    max_action_delta: float

    mean_action_delta: float

    reward: float

    execution_success: bool


class AuditLogger:
    """
    Lightweight in-memory audit logger.

    Phase-2A deliberately keeps audit storage in memory.
    A persistent database can be added later without changing
    the governance decision interface.
    """

    def __init__(self):
        self.records: list[AuditRecord] = []

    def reset(self) -> None:
        self.records.clear()

    def log(
        self,
        *,
        step: int,
        governance_status: str,
        reason: str,
        trust_score: float,
        risk_score: float,
        risk_level: str,
        policy_allowed: bool,
        policy_violations: list[str],
        modified: bool,
        fallback_used: bool,
        max_action_delta: float,
        mean_action_delta: float,
        reward: float,
        execution_success: bool,
    ) -> AuditRecord:

        record = AuditRecord(
            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),

            step=int(step),

            governance_status=str(
                governance_status
            ),

            reason=str(
                reason
            ),

            trust_score=float(
                trust_score
            ),

            risk_score=float(
                risk_score
            ),

            risk_level=str(
                risk_level
            ),

            policy_allowed=bool(
                policy_allowed
            ),

            policy_violations=list(
                policy_violations
            ),

            modified=bool(
                modified
            ),

            fallback_used=bool(
                fallback_used
            ),

            max_action_delta=float(
                max_action_delta
            ),

            mean_action_delta=float(
                mean_action_delta
            ),

            reward=float(
                reward
            ),

            execution_success=bool(
                execution_success
            ),
        )

        self.records.append(
            record
        )

        return record

    def latest(self) -> AuditRecord | None:
        if not self.records:
            return None

        return self.records[-1]

    def count(self) -> int:
        return len(
            self.records
        )

    def summary(self) -> dict[str, Any]:

        if not self.records:
            return {
                "total": 0,
                "allow": 0,
                "constrain": 0,
                "block": 0,
                "mean_trust": 0.0,
                "mean_risk": 0.0,
            }

        statuses = [
            record.governance_status
            for record in self.records
        ]

        return {
            "total": len(
                self.records
            ),

            "allow": statuses.count(
                "ALLOW"
            ),

            "constrain": statuses.count(
                "CONSTRAIN"
            ),

            "block": statuses.count(
                "BLOCK"
            ),

            "mean_trust": float(
                np.mean(
                    [
                        record.trust_score
                        for record in self.records
                    ]
                )
            ),

            "mean_risk": float(
                np.mean(
                    [
                        record.risk_score
                        for record in self.records
                    ]
                )
            ),
        }

    def export_records(self) -> list[dict[str, Any]]:
        """
        Return JSON-serializable audit records.
        """

        return [
            asdict(record)
            for record in self.records
        ]