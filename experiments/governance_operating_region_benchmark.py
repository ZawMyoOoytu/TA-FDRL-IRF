
"""
TA-FDRL-IRF
Governance Operating Region Benchmark
H5-B — Final Corrected Research-Grade Version

Purpose
-------
Measure the REAL numerical operating regions of the governance layer
using:

    - Real TrustEngine
    - Real PolicyEngine
    - Real RiskEngine
    - Real SafeEnvelope
    - Real GovernanceEngine

This benchmark does NOT inject artificial RiskAssessment objects.

The benchmark controls the INITIAL governance trust state for every
operating point so that:

    controlled trust
        ==
    initial governance trust
        ==
    effective trust used by RiskEngine

This is required because GovernanceEngine.evaluate() obtains governance
trust from TrustEngine before passing it to RiskEngine.

Therefore every operating point receives a FRESH TrustEngine with:

    TrustEngineConfig(initial_trust=controlled_trust)

This preserves the real architecture while making the operating-point
experiment scientifically controlled.

Risk equation
-------------

    risk =
        0.45 * trust_risk
        + 0.35 * action_anomaly
        + 0.20 * telemetry_risk

where:

    trust_risk = 1 - trust

    action_anomaly =
        clip((abs(action) - 0.70) / 0.30, 0, 1)

    telemetry_risk =
        max(
            mean(clip(queue, 0, 1)),
            mean(clip(abs(interference), 0, 1))
        )

Risk levels
-----------

    LOW       : risk < 0.60
    MEDIUM    : 0.60 <= risk < 0.80
    HIGH      : risk >= 0.80

Governance precedence
---------------------

    1. POLICY_VIOLATION
        -> BLOCK

    2. HIGH_RISK
        -> BLOCK

    3. TRUST_BELOW_MINIMUM
        -> BLOCK

    4. MEDIUM_RISK_SAFE_ENVELOPE
        -> CONSTRAIN

    5. GOVERNANCE_APPROVED
        -> ALLOW

Important validation principle
------------------------------

Governance status is NOT determined by risk level alone.

For example:

    LOW risk + policy violation
        -> BLOCK

    LOW risk + trust < 0.60
        -> BLOCK

    MEDIUM risk + policy compliant + trust >= 0.60
        -> CONSTRAIN

    HIGH risk + policy compliant
        -> BLOCK

Therefore validation functions explicitly respect governance
precedence rather than assuming:

    risk level == governance status

Scientific purpose
------------------

This benchmark answers:

    "Under the current real governance equations and thresholds,
     what combinations of trust, action magnitude, queue pressure,
     and interference pressure produce ALLOW, CONSTRAIN, and BLOCK?"

This is a governance calibration / operating-region experiment.

It does NOT:

    - modify GovernanceEngine
    - modify RiskEngine
    - modify PolicyEngine
    - modify TrustEngine
    - modify SafeEnvelope
    - modify SAC
    - modify IRFEnvironment
    - train SAC
    - perform post-execution trust updates
    - represent physical-world 6G measurements
"""


from __future__ import annotations


# ============================================================================
# PROJECT ROOT
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

import numpy as np

from trust.governance import GovernanceEngine
from trust.trust_engine import (
    TrustEngine,
    TrustEngineConfig,
)


# ============================================================================
# CONSTANTS
# ============================================================================

ACTION_DIM = 40
NUM_USERS = 20

# ---------------------------------------------------------------------------
# Governance thresholds mirrored from the REAL RiskConfig.
#
# These constants are used only for independent validation/reporting.
# They do not replace the actual GovernanceEngine configuration.
# ---------------------------------------------------------------------------

RISK_LOW_THRESHOLD = 0.60
RISK_HIGH_THRESHOLD = 0.80
MINIMUM_TRUST = 0.60


# ============================================================================
# CONTROLLED OPERATING GRID
# ============================================================================

TRUST_LEVELS = [
    1.00,
    0.90,
    0.80,
    0.70,
    0.60,
    0.50,
    0.40,
    0.30,
    0.20,
    0.10,
]


ACTION_LEVELS = [
    0.00,
    0.50,
    0.70,
    0.80,
    0.90,
    1.00,
]


TELEMETRY_LEVELS = [
    0.00,
    0.40,
    0.60,
    0.80,
    0.90,
    1.00,
]


# ============================================================================
# RESULT DATA STRUCTURE
# ============================================================================

@dataclass
class BenchmarkResult:
    """
    One controlled governance operating-point observation.
    """

    # Controlled input trust.
    trust: float

    # Actual governance trust returned by TrustEngine.
    governance_trust: float

    # Environment/domain trust returned by TrustEngine.
    environment_trust: float

    # Controlled action magnitude.
    action_magnitude: float

    # Controlled queue pressure.
    queue_pressure: float

    # Controlled interference pressure.
    interference_pressure: float

    # Actual RiskEngine output.
    risk_score: float

    # Independently calculated theoretical risk.
    expected_risk: float

    # Actual RiskEngine classification.
    risk_level: str

    # Actual GovernanceEngine decision.
    governance_status: str

    # Actual GovernanceEngine reason.
    governance_reason: str

    # Whether SafeEnvelope modified the action.
    action_modified: bool

    # Actual PolicyEngine decision.
    policy_allowed: bool

    # Absolute numerical difference between actual and expected risk.
    formula_error: float


# ============================================================================
# ACTION CONSTRUCTION
# ============================================================================

def make_action(
    magnitude: float,
) -> np.ndarray:
    """
    Construct a 40-dimensional controlled action.

    All dimensions receive the same absolute magnitude.

    PolicyEngine boundary:

        dimensions [0:20]
            -> bandwidth

        dimensions [20:40]
            -> normalized power

    Current PolicyEngine allows:

        normalized power <= 0.90

    Therefore:

        magnitude <= 0.90
            -> policy compliant

        magnitude = 1.00
            -> policy violation
            -> BLOCK through POLICY_VIOLATION precedence
    """

    return np.full(
        ACTION_DIM,
        float(magnitude),
        dtype=np.float32,
    )


# ============================================================================
# TELEMETRY CONSTRUCTION
# ============================================================================

def make_telemetry(
    queue_pressure: float,
    interference_pressure: float,
) -> dict:
    """
    Construct deterministic telemetry compatible with RiskEngine.

    RiskEngine consumes:

        telemetry["queue"]
        telemetry["interference"]
    """

    queue = np.full(
        NUM_USERS,
        float(queue_pressure),
        dtype=np.float32,
    )

    interference = np.full(
        NUM_USERS,
        float(interference_pressure),
        dtype=np.float32,
    )

    return {
        "queue": queue,
        "interference": interference,
    }


# ============================================================================
# GOVERNANCE ENGINE FACTORY
# ============================================================================

def create_governance(
    initial_trust: float,
) -> GovernanceEngine:
    """
    Create a REAL GovernanceEngine with an isolated TrustEngine.

    Each operating point receives a fresh TrustEngine.

    This prevents cross-point state contamination.

    No artificial RiskAssessment is injected.
    """

    trust_engine = TrustEngine(
        TrustEngineConfig(
            initial_trust=float(initial_trust),
        )
    )

    return GovernanceEngine(
        trust_engine=trust_engine,
    )


# ============================================================================
# SINGLE OPERATING POINT
# ============================================================================

def evaluate_point(
    trust: float,
    action_magnitude: float,
    queue_pressure: float,
    interference_pressure: float,
) -> BenchmarkResult:
    """
    Evaluate one completely isolated operating point.

    A fresh GovernanceEngine and TrustEngine are created for every point.
    """

    # ------------------------------------------------------------------------
    # Fresh governance state
    # ------------------------------------------------------------------------

    governance = create_governance(
        initial_trust=trust,
    )

    # ------------------------------------------------------------------------
    # Controlled inputs
    # ------------------------------------------------------------------------

    action = make_action(
        action_magnitude,
    )

    telemetry = make_telemetry(
        queue_pressure,
        interference_pressure,
    )

    # ------------------------------------------------------------------------
    # REAL GovernanceEngine evaluation
    # ------------------------------------------------------------------------

    decision = governance.evaluate(
        action,
        trust_score=trust,
        telemetry=telemetry,
    )

    # ------------------------------------------------------------------------
    # Extract actual TrustEngine state
    # ------------------------------------------------------------------------

    if decision.trust is not None:

        governance_trust = float(
            decision.trust.trust_score
        )

        environment_trust = float(
            decision.trust.environment_trust
        )

    else:

        governance_trust = float(
            governance.trust_engine.trust_score
        )

        environment_trust = float(
            governance.trust_engine.environment_trust
        )

    # ------------------------------------------------------------------------
    # Actual RiskEngine output
    # ------------------------------------------------------------------------

    risk_score = float(
        decision.risk.risk_score
    )

    risk_level = str(
        decision.risk.risk_level
    )

    # ------------------------------------------------------------------------
    # Independent theoretical calculation
    # ------------------------------------------------------------------------

    # Use the ACTUAL governance trust consumed by RiskEngine.
    #
    # In this controlled experiment:
    #
    #     governance_trust == trust
    #
    # The explicit use of governance_trust here makes the validation
    # architecture transparent and prevents future silent mismatches.

    expected_risk = theoretical_risk(
        trust=governance_trust,
        action_magnitude=action_magnitude,
        queue_pressure=queue_pressure,
        interference_pressure=interference_pressure,
    )

    formula_error = abs(
        risk_score - expected_risk
    )

    # ------------------------------------------------------------------------
    # Return observation
    # ------------------------------------------------------------------------

    return BenchmarkResult(
        trust=float(trust),

        governance_trust=governance_trust,

        environment_trust=environment_trust,

        action_magnitude=float(
            action_magnitude
        ),

        queue_pressure=float(
            queue_pressure
        ),

        interference_pressure=float(
            interference_pressure
        ),

        risk_score=risk_score,

        expected_risk=expected_risk,

        risk_level=risk_level,

        governance_status=str(
            decision.status
        ),

        governance_reason=str(
            decision.reason
        ),

        action_modified=bool(
            decision.modified
        ),

        policy_allowed=bool(
            decision.policy.allowed
        ),

        formula_error=float(
            formula_error
        ),
    )


# ============================================================================
# THEORETICAL RISK EQUATION
# ============================================================================

def theoretical_risk(
    trust: float,
    action_magnitude: float,
    queue_pressure: float,
    interference_pressure: float,
) -> float:
    """
    Independently reproduce the REAL RiskEngine numerical equation.

    Equation:

        trust_risk = 1 - trust

        action_anomaly =
            clip(
                (abs(action) - 0.70) / 0.30,
                0,
                1
            )

        telemetry_risk =
            max(queue, interference)

        risk =
            0.45 * trust_risk
            + 0.35 * action_anomaly
            + 0.20 * telemetry_risk
    """

    # ------------------------------------------------------------------------
    # Trust term
    # ------------------------------------------------------------------------

    trust_clipped = float(
        np.clip(
            trust,
            0.0,
            1.0,
        )
    )

    trust_risk = (
        1.0 - trust_clipped
    )

    # ------------------------------------------------------------------------
    # Action anomaly term
    # ------------------------------------------------------------------------

    action_value = float(
        np.clip(
            abs(action_magnitude),
            0.0,
            1.0,
        )
    )

    action_anomaly = float(
        np.clip(
            (
                action_value - 0.70
            )
            / 0.30,
            0.0,
            1.0,
        )
    )

    # ------------------------------------------------------------------------
    # Telemetry term
    # ------------------------------------------------------------------------

    queue = float(
        np.clip(
            queue_pressure,
            0.0,
            1.0,
        )
    )

    interference = float(
        np.clip(
            interference_pressure,
            0.0,
            1.0,
        )
    )

    telemetry_risk = max(
        queue,
        interference,
    )

    # ------------------------------------------------------------------------
    # Combined risk
    # ------------------------------------------------------------------------

    score = (
        0.45 * trust_risk
        + 0.35 * action_anomaly
        + 0.20 * telemetry_risk
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


# ============================================================================
# THEORETICAL RISK LEVEL
# ============================================================================

def theoretical_risk_level(
    risk_score: float,
) -> str:
    """
    Determine expected risk level from the configured thresholds.
    """

    risk = float(
        np.clip(
            risk_score,
            0.0,
            1.0,
        )
    )

    if risk >= RISK_HIGH_THRESHOLD:
        return "HIGH"

    if risk >= RISK_LOW_THRESHOLD:
        return "MEDIUM"

    return "LOW"


# ============================================================================
# FORMULA VALIDATION
# ============================================================================

def validate_formula(
    result: BenchmarkResult,
    tolerance: float = 1e-6,
) -> bool:
    """
    Verify actual RiskEngine output against the independent equation.
    """

    return bool(
        np.isclose(
            result.risk_score,
            result.expected_risk,
            atol=tolerance,
            rtol=0.0,
        )
    )


# ============================================================================
# EXTREME OPERATING POINTS
# ============================================================================

def evaluate_extreme_cases() -> bool:
    """
    Evaluate analytically important operating boundaries.

    These cases validate:

        - ideal operation
        - low-trust blocking
        - policy precedence
        - telemetry contribution
        - high-risk blocking

    Returns
    -------
    bool
        True when all numerical and trust-initialization checks pass.
    """

    cases = [
        (
            "IDEAL",
            1.00,
            0.00,
            0.00,
            0.00,
        ),
        (
            "LOW_TRUST_ONLY",
            0.30,
            0.00,
            0.00,
            0.00,
        ),
        (
            "MAX_ACTION_ONLY",
            1.00,
            1.00,
            0.00,
            0.00,
        ),
        (
            "MAX_TELEMETRY_ONLY",
            1.00,
            0.00,
            1.00,
            1.00,
        ),
        (
            "ALL_MAX",
            0.00,
            0.90,
            1.00,
            1.00,
        ),
    ]

    print()
    print("=" * 78)
    print("EXTREME OPERATING POINTS")
    print("=" * 78)

    all_passed = True

    for (
        name,
        trust,
        action_magnitude,
        queue_pressure,
        interference_pressure,
    ) in cases:

        result = evaluate_point(
            trust=trust,
            action_magnitude=action_magnitude,
            queue_pressure=queue_pressure,
            interference_pressure=interference_pressure,
        )

        formula_pass = validate_formula(
            result
        )

        governance_trust_match = np.isclose(
            result.governance_trust,
            trust,
            atol=1e-6,
            rtol=0.0,
        )

        if not formula_pass:
            all_passed = False

        if not governance_trust_match:
            all_passed = False

        print()
        print(name)

        print(
            f"  Environment trust : "
            f"{result.environment_trust:.6f}"
        )

        print(
            f"  Governance trust  : "
            f"{result.governance_trust:.6f}"
        )

        print(
            f"  Action magnitude  : "
            f"{action_magnitude:.2f}"
        )

        print(
            f"  Queue             : "
            f"{queue_pressure:.2f}"
        )

        print(
            f"  Interference      : "
            f"{interference_pressure:.2f}"
        )

        print(
            f"  Expected risk     : "
            f"{result.expected_risk:.6f}"
        )

        print(
            f"  Actual risk       : "
            f"{result.risk_score:.6f}"
        )

        print(
            f"  Risk level        : "
            f"{result.risk_level}"
        )

        print(
            f"  Governance        : "
            f"{result.governance_status}"
        )

        print(
            f"  Reason            : "
            f"{result.governance_reason}"
        )

        print(
            f"  Policy allowed    : "
            f"{result.policy_allowed}"
        )

        print(
            f"  Formula validation: "
            f"{'PASS' if formula_pass else 'FAIL'}"
        )

        print(
            f"  Trust initialization: "
            f"{'PASS' if governance_trust_match else 'FAIL'}"
        )

    return all_passed


# ============================================================================
# FULL CONTROLLED BENCHMARK
# ============================================================================

def run_benchmark() -> list[BenchmarkResult]:
    """
    Execute the complete controlled operating grid.

    Grid size:

        10 trust levels
        × 6 action levels
        × 6 queue levels
        × 6 interference levels

        = 2160 operating points
    """

    results: list[BenchmarkResult] = []

    total_points = (
        len(TRUST_LEVELS)
        * len(ACTION_LEVELS)
        * len(TELEMETRY_LEVELS)
        * len(TELEMETRY_LEVELS)
    )

    completed = 0

    print(
        f"Total operating points: "
        f"{total_points}"
    )

    print()

    for trust in TRUST_LEVELS:

        for action_magnitude in ACTION_LEVELS:

            for queue_pressure in TELEMETRY_LEVELS:

                for interference_pressure in TELEMETRY_LEVELS:

                    result = evaluate_point(
                        trust=trust,
                        action_magnitude=action_magnitude,
                        queue_pressure=queue_pressure,
                        interference_pressure=interference_pressure,
                    )

                    results.append(
                        result
                    )

                    completed += 1

    print(
        f"Completed operating points: "
        f"{completed}"
    )

    return results


# ============================================================================
# RISK FORMULA SUMMARY
# ============================================================================

def summarize_formula_validation(
    results: list[BenchmarkResult],
) -> bool:
    """
    Validate all numerical RiskEngine outputs.
    """

    passed = 0
    failed = 0
    max_error = 0.0

    for result in results:

        error = abs(
            result.risk_score
            - result.expected_risk
        )

        max_error = max(
            max_error,
            error,
        )

        if validate_formula(
            result
        ):

            passed += 1

        else:

            failed += 1

    print()
    print("=" * 78)
    print("RISK FORMULA VALIDATION")
    print("=" * 78)

    print(
        f"Points checked       : "
        f"{len(results)}"
    )

    print(
        f"Formula matches      : "
        f"{passed}"
    )

    print(
        f"Formula mismatches   : "
        f"{failed}"
    )

    print(
        f"Maximum absolute err : "
        f"{max_error:.12f}"
    )

    passed_all = (
        failed == 0
    )

    print(
        "RESULT                : "
        f"{'PASS' if passed_all else 'FAIL'}"
    )

    return passed_all


# ============================================================================
# TRUST INITIALIZATION SUMMARY
# ============================================================================

def summarize_trust_initialization(
    results: list[BenchmarkResult],
) -> bool:
    """
    Validate that every controlled trust value is correctly represented
    by both:

        - governance trust
        - environment trust

    This guards against the original H5-B failure in which controlled
    environment trust could differ from internal governance trust.
    """

    governance_mismatches = []
    environment_mismatches = []

    for index, result in enumerate(results):

        if not np.isclose(
            result.trust,
            result.governance_trust,
            atol=1e-6,
            rtol=0.0,
        ):

            governance_mismatches.append(
                (
                    index,
                    result.trust,
                    result.governance_trust,
                )
            )

        if not np.isclose(
            result.trust,
            result.environment_trust,
            atol=1e-6,
            rtol=0.0,
        ):

            environment_mismatches.append(
                (
                    index,
                    result.trust,
                    result.environment_trust,
                )
            )

    print()
    print("=" * 78)
    print("TRUST INITIALIZATION VALIDATION")
    print("=" * 78)

    print(
        f"Points checked           : "
        f"{len(results)}"
    )

    print(
        f"Governance trust matches : "
        f"{len(results) - len(governance_mismatches)}"
    )

    print(
        f"Governance mismatches    : "
        f"{len(governance_mismatches)}"
    )

    print(
        f"Environment trust matches: "
        f"{len(results) - len(environment_mismatches)}"
    )

    print(
        f"Environment mismatches   : "
        f"{len(environment_mismatches)}"
    )

    if governance_mismatches:

        first = governance_mismatches[0]

        print(
            "First governance mismatch: "
            f"index={first[0]}, "
            f"input={first[1]:.6f}, "
            f"governance={first[2]:.6f}"
        )

    if environment_mismatches:

        first = environment_mismatches[0]

        print(
            "First environment mismatch: "
            f"index={first[0]}, "
            f"input={first[1]:.6f}, "
            f"environment={first[2]:.6f}"
        )

    passed = (
        len(governance_mismatches) == 0
        and len(environment_mismatches) == 0
    )

    print(
        "RESULT                : "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return passed


# ============================================================================
# GOVERNANCE OPERATING REGION
# ============================================================================

def summarize_regions(
    results: list[BenchmarkResult],
) -> tuple[int, int, int]:
    """
    Summarize ALLOW / CONSTRAIN / BLOCK operating regions.
    """

    total = len(results)

    allow = sum(
        1
        for result in results
        if result.governance_status == "ALLOW"
    )

    constrain = sum(
        1
        for result in results
        if result.governance_status == "CONSTRAIN"
    )

    block = sum(
        1
        for result in results
        if result.governance_status == "BLOCK"
    )

    print()
    print("=" * 78)
    print("GOVERNANCE OPERATING REGION")
    print("=" * 78)

    print(
        f"Total points          : "
        f"{total}"
    )

    print(
        f"ALLOW                 : "
        f"{allow:6d} "
        f"({allow / total * 100:6.2f}%)"
    )

    print(
        f"CONSTRAIN             : "
        f"{constrain:6d} "
        f"({constrain / total * 100:6.2f}%)"
    )

    print(
        f"BLOCK                 : "
        f"{block:6d} "
        f"({block / total * 100:6.2f}%)"
    )

    print()

    print(
        "[PASS] ALLOW operating region exists"
        if allow > 0
        else
        "[FAIL] ALLOW operating region missing"
    )

    print(
        "[PASS] CONSTRAIN operating region exists"
        if constrain > 0
        else
        "[FAIL] CONSTRAIN operating region missing"
    )

    print(
        "[PASS] BLOCK operating region exists"
        if block > 0
        else
        "[FAIL] BLOCK operating region missing"
    )

    return (
        allow,
        constrain,
        block,
    )


# ============================================================================
# RISK LEVEL DISTRIBUTION
# ============================================================================

def summarize_risk_levels(
    results: list[BenchmarkResult],
) -> tuple[int, int, int]:
    """
    Summarize LOW / MEDIUM / HIGH risk distribution.
    """

    total = len(results)

    low = sum(
        1
        for result in results
        if result.risk_level == "LOW"
    )

    medium = sum(
        1
        for result in results
        if result.risk_level == "MEDIUM"
    )

    high = sum(
        1
        for result in results
        if result.risk_level == "HIGH"
    )

    print()
    print("=" * 78)
    print("RISK LEVEL DISTRIBUTION")
    print("=" * 78)

    print(
        f"LOW                   : "
        f"{low:6d} "
        f"({low / total * 100:6.2f}%)"
    )

    print(
        f"MEDIUM                : "
        f"{medium:6d} "
        f"({medium / total * 100:6.2f}%)"
    )

    print(
        f"HIGH                  : "
        f"{high:6d} "
        f"({high / total * 100:6.2f}%)"
    )

    return (
        low,
        medium,
        high,
    )


# ============================================================================
# RISK SCORE RANGE
# ============================================================================

def summarize_risk_range(
    results: list[BenchmarkResult],
) -> None:
    """
    Report numerical risk statistics.
    """

    scores = np.asarray(
        [
            result.risk_score
            for result in results
        ],
        dtype=np.float64,
    )

    print()
    print("=" * 78)
    print("RISK SCORE RANGE")
    print("=" * 78)

    print(
        f"Minimum               : "
        f"{np.min(scores):.6f}"
    )

    print(
        f"Maximum               : "
        f"{np.max(scores):.6f}"
    )

    print(
        f"Mean                  : "
        f"{np.mean(scores):.6f}"
    )

    print(
        f"Median                : "
        f"{np.median(scores):.6f}"
    )


# ============================================================================
# RISK × GOVERNANCE CROSS TABULATION
# ============================================================================

def summarize_cross_tab(
    results: list[BenchmarkResult],
) -> None:
    """
    Print risk-level × governance-status cross-tabulation.
    """

    combinations = [
        ("LOW", "ALLOW"),
        ("LOW", "CONSTRAIN"),
        ("LOW", "BLOCK"),
        ("MEDIUM", "ALLOW"),
        ("MEDIUM", "CONSTRAIN"),
        ("MEDIUM", "BLOCK"),
        ("HIGH", "ALLOW"),
        ("HIGH", "CONSTRAIN"),
        ("HIGH", "BLOCK"),
    ]

    print()
    print("=" * 78)
    print("RISK LEVEL x GOVERNANCE STATUS")
    print("=" * 78)

    for risk_level, status in combinations:

        count = sum(
            1
            for result in results
            if (
                result.risk_level == risk_level
                and result.governance_status == status
            )
        )

        print(
            f"{risk_level:<8} -> "
            f"{status:<10} : "
            f"{count}"
        )


# ============================================================================
# GOVERNANCE REASON DISTRIBUTION
# ============================================================================

def summarize_reasons(
    results: list[BenchmarkResult],
) -> dict[str, int]:
    """
    Count actual GovernanceEngine decision reasons.
    """

    counts: dict[str, int] = {}

    for result in results:

        reason = result.governance_reason

        counts[reason] = (
            counts.get(reason, 0)
            + 1
        )

    print()
    print("=" * 78)
    print("GOVERNANCE REASON DISTRIBUTION")
    print("=" * 78)

    for reason in sorted(counts):

        print(
            f"{reason:<32} : "
            f"{counts[reason]}"
        )

    return counts


# ============================================================================
# POLICY DISTRIBUTION
# ============================================================================

def summarize_policy(
    results: list[BenchmarkResult],
) -> tuple[int, int]:
    """
    Summarize policy-approved and policy-rejected operating points.
    """

    allowed = sum(
        1
        for result in results
        if result.policy_allowed
    )

    rejected = sum(
        1
        for result in results
        if not result.policy_allowed
    )

    print()
    print("=" * 78)
    print("POLICY OPERATING REGION")
    print("=" * 78)

    print(
        f"Policy allowed       : "
        f"{allowed}"
    )

    print(
        f"Policy rejected      : "
        f"{rejected}"
    )

    return (
        allowed,
        rejected,
    )


# ============================================================================
# CONSTRAIN REGION VALIDATION
# ============================================================================

def validate_constrain_region(
    results: list[BenchmarkResult],
) -> bool:
    """
    Validate the MEDIUM-risk -> CONSTRAIN path.

    A valid CONSTRAIN point must satisfy:

        risk_level == MEDIUM
        governance_status == CONSTRAIN
        policy_allowed == True
        action_modified == True
        trust >= minimum trust
    """

    candidates = [
        result
        for result in results
        if (
            result.risk_level == "MEDIUM"
            and result.governance_status == "CONSTRAIN"
        )
    ]

    valid = [
        result
        for result in candidates
        if (
            result.policy_allowed
            and result.action_modified
            and result.governance_trust
            >= MINIMUM_TRUST
        )
    ]

    print()
    print("=" * 78)
    print("CONSTRAIN REGION VALIDATION")
    print("=" * 78)

    print(
        f"MEDIUM-risk points   : "
        f"{len(candidates)}"
    )

    print(
        f"Valid CONSTRAIN      : "
        f"{len(valid)}"
    )

    passed = (
        len(candidates) > 0
        and len(valid) == len(candidates)
    )

    print(
        "RESULT                : "
        f"{'PASS' if passed else 'FAIL'}"
    )

    if valid:

        example = valid[0]

        print()
        print("Example valid CONSTRAIN point:")

        print(
            f"  Trust             : "
            f"{example.trust:.2f}"
        )

        print(
            f"  Action magnitude  : "
            f"{example.action_magnitude:.2f}"
        )

        print(
            f"  Queue             : "
            f"{example.queue_pressure:.2f}"
        )

        print(
            f"  Interference      : "
            f"{example.interference_pressure:.2f}"
        )

        print(
            f"  Risk              : "
            f"{example.risk_score:.6f}"
        )

        print(
            f"  Status            : "
            f"{example.governance_status}"
        )

        print(
            f"  Reason            : "
            f"{example.governance_reason}"
        )

        print(
            f"  Action modified   : "
            f"{example.action_modified}"
        )

    return passed


# ============================================================================
# HIGH-RISK REGION VALIDATION
# ============================================================================

def validate_high_risk_region(
    results: list[BenchmarkResult],
) -> bool:
    """
    Validate HIGH-risk governance behavior.

    Policy violation has higher precedence than HIGH_RISK.

    Therefore only policy-compliant HIGH-risk points are used to
    validate the HIGH_RISK -> BLOCK path.
    """

    high_points = [
        result
        for result in results
        if result.risk_level == "HIGH"
    ]

    policy_compliant_high = [
        result
        for result in high_points
        if result.policy_allowed
    ]

    correct = [
        result
        for result in policy_compliant_high
        if (
            result.governance_status == "BLOCK"
            and result.governance_reason
            == "HIGH_RISK"
        )
    ]

    print()
    print("=" * 78)
    print("HIGH-RISK VALIDATION")
    print("=" * 78)

    print(
        f"HIGH-risk points     : "
        f"{len(high_points)}"
    )

    print(
        f"Policy-compliant HIGH: "
        f"{len(policy_compliant_high)}"
    )

    print(
        f"Correct HIGH blocks  : "
        f"{len(correct)}"
    )

    passed = (
        len(high_points) > 0
        and len(policy_compliant_high) > 0
        and len(correct)
        == len(policy_compliant_high)
    )

    print(
        "RESULT                : "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return passed


# ============================================================================
# LOW-TRUST GOVERNANCE VALIDATION
# ============================================================================

def validate_low_trust_region(
    results: list[BenchmarkResult],
) -> bool:
    """
    Validate the low-trust governance boundary while respecting
    GovernanceEngine precedence.

    Governance precedence:

        1. POLICY_VIOLATION
        2. HIGH_RISK
        3. TRUST_BELOW_MINIMUM
        4. MEDIUM_RISK_SAFE_ENVELOPE
        5. GOVERNANCE_APPROVED

    Therefore, NOT every low-trust point is expected to have:

        TRUST_BELOW_MINIMUM

    because some low-trust points can legitimately be intercepted
    earlier by:

        POLICY_VIOLATION
        HIGH_RISK

    The specific trust-boundary validation therefore selects only
    low-trust points for which no higher-precedence condition applies.

    Eligible point:

        trust < 0.60
        AND policy_allowed == True
        AND risk_level != HIGH

    Expected:

        BLOCK
        TRUST_BELOW_MINIMUM
        governance_trust < 0.60
    """

    # ------------------------------------------------------------------------
    # All low-trust operating points
    # ------------------------------------------------------------------------

    low_trust_points = [
        result
        for result in results
        if result.trust < MINIMUM_TRUST
    ]

    # ------------------------------------------------------------------------
    # Points where trust precedence is actually eligible.
    #
    # POLICY_VIOLATION and HIGH_RISK are intentionally excluded because
    # they have higher precedence in GovernanceEngine.
    # ------------------------------------------------------------------------

    eligible_points = [
        result
        for result in low_trust_points
        if (
            result.policy_allowed
            and result.risk_level != "HIGH"
        )
    ]

    # ------------------------------------------------------------------------
    # Correct trust-boundary decisions
    # ------------------------------------------------------------------------

    correct = [
        result
        for result in eligible_points
        if (
            result.governance_status == "BLOCK"
            and result.governance_reason
            == "TRUST_BELOW_MINIMUM"
            and result.governance_trust
            < MINIMUM_TRUST
        )
    ]

    # ------------------------------------------------------------------------
    # Validation report
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("LOW-TRUST GOVERNANCE VALIDATION")
    print("=" * 78)

    print(
        f"Low-trust points             : "
        f"{len(low_trust_points)}"
    )

    print(
        f"Eligible trust-boundary pts  : "
        f"{len(eligible_points)}"
    )

    print(
        f"Correct trust blocks         : "
        f"{len(correct)}"
    )

    # ------------------------------------------------------------------------
    # Scientific invariant
    # ------------------------------------------------------------------------

    passed = (
        len(eligible_points) > 0
        and len(correct) == len(eligible_points)
    )

    print(
        "RESULT                       : "
        f"{'PASS' if passed else 'FAIL'}"
    )

    # ------------------------------------------------------------------------
    # Representative valid point
    # ------------------------------------------------------------------------

    if correct:

        example = correct[0]

        print()
        print("Example valid LOW-TRUST point:")

        print(
            f"  Controlled trust       : "
            f"{example.trust:.2f}"
        )

        print(
            f"  Governance trust       : "
            f"{example.governance_trust:.2f}"
        )

        print(
            f"  Action magnitude       : "
            f"{example.action_magnitude:.2f}"
        )

        print(
            f"  Queue pressure         : "
            f"{example.queue_pressure:.2f}"
        )

        print(
            f"  Interference pressure  : "
            f"{example.interference_pressure:.2f}"
        )

        print(
            f"  Risk                   : "
            f"{example.risk_score:.6f}"
        )

        print(
            f"  Risk level             : "
            f"{example.risk_level}"
        )

        print(
            f"  Governance status      : "
            f"{example.governance_status}"
        )

        print(
            f"  Governance reason      : "
            f"{example.governance_reason}"
        )

    return passed


# ============================================================================
# POLICY PRECEDENCE VALIDATION
# ============================================================================

def validate_policy_precedence(
    results: list[BenchmarkResult],
) -> bool:
    """
    Validate that every actual policy violation produces BLOCK with
    POLICY_VIOLATION precedence.
    """

    policy_violations = [
        result
        for result in results
        if not result.policy_allowed
    ]

    correct = [
        result
        for result in policy_violations
        if (
            result.governance_status == "BLOCK"
            and result.governance_reason
            == "POLICY_VIOLATION"
        )
    ]

    print()
    print("=" * 78)
    print("POLICY-PRECEDENCE VALIDATION")
    print("=" * 78)

    print(
        f"Policy violations     : "
        f"{len(policy_violations)}"
    )

    print(
        f"Correct policy blocks : "
        f"{len(correct)}"
    )

    passed = (
        len(policy_violations) > 0
        and len(correct)
        == len(policy_violations)
    )

    print(
        "RESULT                : "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return passed


# ============================================================================
# THRESHOLD REPORT
# ============================================================================

def print_threshold_analysis() -> None:
    """
    Print the governance thresholds used for independent validation.
    """

    print()
    print("=" * 78)
    print("CONFIGURED GOVERNANCE THRESHOLDS")
    print("=" * 78)

    print("Risk thresholds:")

    print(
        f"  LOW       : risk < "
        f"{RISK_LOW_THRESHOLD:.2f}"
    )

    print(
        f"  MEDIUM    : "
        f"{RISK_LOW_THRESHOLD:.2f} <= risk < "
        f"{RISK_HIGH_THRESHOLD:.2f}"
    )

    print(
        f"  HIGH      : risk >= "
        f"{RISK_HIGH_THRESHOLD:.2f}"
    )

    print()

    print("Trust threshold:")

    print(
        f"  minimum trust = "
        f"{MINIMUM_TRUST:.2f}"
    )

    print()

    print("Risk equation:")

    print(
        "  risk = "
        "0.45 * trust_risk "
        "+ 0.35 * action_anomaly "
        "+ 0.20 * telemetry_risk"
    )

    print()

    print("Action anomaly:")

    print(
        "  anomaly = "
        "clip((abs(action) - 0.70) / 0.30, 0, 1)"
    )

    print()

    print("Governance precedence:")

    print(
        "  1. POLICY_VIOLATION"
    )

    print(
        "  2. HIGH_RISK"
    )

    print(
        "  3. TRUST_BELOW_MINIMUM"
    )

    print(
        "  4. MEDIUM_RISK_SAFE_ENVELOPE"
    )

    print(
        "  5. GOVERNANCE_APPROVED"
    )

    print()

    print("Important:")

    print(
        "  Numerical risk level and governance status are not identical."
    )

    print(
        "  LOW risk can still produce BLOCK because of policy violation"
    )

    print(
        "  or governance trust below the minimum threshold."
    )


# ============================================================================
# FINAL VALIDATION
# ============================================================================

def final_validation(
    results: list[BenchmarkResult],
    formula_pass: bool,
    trust_initialization_pass: bool,
    allow: int,
    constrain: int,
    block: int,
    constrain_pass: bool,
    high_risk_pass: bool,
    low_trust_pass: bool,
    policy_precedence_pass: bool,
) -> bool:
    """
    Perform final research-benchmark validation.
    """

    statuses = {
        result.governance_status
        for result in results
    }

    risk_levels = {
        result.risk_level
        for result in results
    }

    # ------------------------------------------------------------------------
    # Governance regions
    # ------------------------------------------------------------------------

    allow_pass = (
        "ALLOW" in statuses
        and allow > 0
    )

    constrain_region_pass = (
        "CONSTRAIN" in statuses
        and constrain > 0
        and constrain_pass
    )

    block_pass = (
        "BLOCK" in statuses
        and block > 0
    )

    # ------------------------------------------------------------------------
    # Risk regions
    # ------------------------------------------------------------------------

    medium_present = (
        "MEDIUM" in risk_levels
    )

    high_present = (
        "HIGH" in risk_levels
    )

    # ------------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL VALIDATION")
    print("=" * 78)

    print(
        "Real RiskEngine formula : "
        f"{'PASS' if formula_pass else 'FAIL'}"
    )

    print(
        "Trust initialization    : "
        f"{'PASS' if trust_initialization_pass else 'FAIL'}"
    )

    print(
        "ALLOW region            : "
        f"{'PASS' if allow_pass else 'FAIL'}"
    )

    print(
        "CONSTRAIN region        : "
        f"{'PASS' if constrain_region_pass else 'FAIL'}"
    )

    print(
        "BLOCK region            : "
        f"{'PASS' if block_pass else 'FAIL'}"
    )

    print(
        "MEDIUM risk region      : "
        f"{'PASS' if medium_present else 'FAIL'}"
    )

    print(
        "HIGH risk region        : "
        f"{'PASS' if high_present else 'FAIL'}"
    )

    print(
        "CONSTRAIN validation    : "
        f"{'PASS' if constrain_pass else 'FAIL'}"
    )

    print(
        "HIGH-risk validation    : "
        f"{'PASS' if high_risk_pass else 'FAIL'}"
    )

    print(
        "Low-trust validation    : "
        f"{'PASS' if low_trust_pass else 'FAIL'}"
    )

    print(
        "Policy precedence       : "
        f"{'PASS' if policy_precedence_pass else 'FAIL'}"
    )

    # ------------------------------------------------------------------------
    # Final benchmark condition
    # ------------------------------------------------------------------------

    benchmark_passed = all(
        [
            formula_pass,
            trust_initialization_pass,
            allow_pass,
            constrain_region_pass,
            block_pass,
            medium_present,
            high_present,
            constrain_pass,
            high_risk_pass,
            low_trust_pass,
            policy_precedence_pass,
        ]
    )

    print()

    if benchmark_passed:

        print(
            "RESULT: GOVERNANCE OPERATING REGION BENCHMARK PASSED"
        )

        print(
            "RESULT: REAL RISK EQUATION VALIDATED"
        )

        print(
            "RESULT: CONTROLLED TRUST INITIALIZATION VALIDATED"
        )

        print(
            "RESULT: ALLOW / CONSTRAIN / BLOCK REGIONS OBSERVED"
        )

        print(
            "RESULT: MEDIUM-RISK SAFE-ENVELOPE REGION VALIDATED"
        )

        print(
            "RESULT: HIGH-RISK BLOCK REGION VALIDATED"
        )

        print(
            "RESULT: LOW-TRUST GOVERNANCE BOUNDARY VALIDATED"
        )

        print(
            "RESULT: POLICY-PRECEDENCE BLOCK REGION VALIDATED"
        )

    else:

        print(
            "RESULT: GOVERNANCE OPERATING REGION BENCHMARK FAILED"
        )

    print("=" * 78)

    return benchmark_passed


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """
    Execute the complete H5-B governance operating-region experiment.
    """

    print("=" * 78)

    print(
        "TA-FDRL-IRF GOVERNANCE OPERATING REGION BENCHMARK"
    )

    print("=" * 78)

    # ------------------------------------------------------------------------
    # Real components
    # ------------------------------------------------------------------------

    print()
    print("REAL COMPONENTS:")

    print(
        "  TrustEngine        : REAL"
    )

    print(
        "  PolicyEngine       : REAL"
    )

    print(
        "  RiskEngine         : REAL"
    )

    print(
        "  SafeEnvelope       : REAL"
    )

    print(
        "  GovernanceEngine   : REAL"
    )

    print()

    print(
        "No artificial RiskAssessment injection is used."
    )

    # ------------------------------------------------------------------------
    # Experimental control
    # ------------------------------------------------------------------------

    print()
    print(
        "CONTROL METHOD:"
    )

    print(
        "  Every operating point receives a fresh TrustEngine."
    )

    print(
        "  initial_trust = controlled environment trust."
    )

    print(
        "  governance trust is independently verified."
    )

    print(
        "  No post-execution trust update is performed."
    )

    print(
        "  This is a PRE-EXECUTION operating-region benchmark."
    )

    # ------------------------------------------------------------------------
    # Threshold report
    # ------------------------------------------------------------------------

    print_threshold_analysis()

    # ------------------------------------------------------------------------
    # Extreme cases
    # ------------------------------------------------------------------------

    extreme_pass = evaluate_extreme_cases()

    # ------------------------------------------------------------------------
    # Full grid
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("CONTROLLED OPERATING GRID")
    print("=" * 78)

    results = run_benchmark()

    # ------------------------------------------------------------------------
    # Formula validation
    # ------------------------------------------------------------------------

    formula_pass = summarize_formula_validation(
        results
    )

    # ------------------------------------------------------------------------
    # Trust initialization
    # ------------------------------------------------------------------------

    trust_initialization_pass = (
        summarize_trust_initialization(
            results
        )
    )

    # ------------------------------------------------------------------------
    # Operating regions
    # ------------------------------------------------------------------------

    allow, constrain, block = summarize_regions(
        results
    )

    # ------------------------------------------------------------------------
    # Risk distribution
    # ------------------------------------------------------------------------

    summarize_risk_levels(
        results
    )

    # ------------------------------------------------------------------------
    # Risk range
    # ------------------------------------------------------------------------

    summarize_risk_range(
        results
    )

    # ------------------------------------------------------------------------
    # Risk × Governance cross-tab
    # ------------------------------------------------------------------------

    summarize_cross_tab(
        results
    )

    # ------------------------------------------------------------------------
    # Governance reasons
    # ------------------------------------------------------------------------

    summarize_reasons(
        results
    )

    # ------------------------------------------------------------------------
    # Policy distribution
    # ------------------------------------------------------------------------

    summarize_policy(
        results
    )

    # ------------------------------------------------------------------------
    # Governance branch validations
    # ------------------------------------------------------------------------

    constrain_pass = validate_constrain_region(
        results
    )

    high_risk_pass = validate_high_risk_region(
        results
    )

    low_trust_pass = validate_low_trust_region(
        results
    )

    policy_precedence_pass = (
        validate_policy_precedence(
            results
        )
    )

    # ------------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------------

    benchmark_passed = final_validation(
        results=results,

        formula_pass=(
            formula_pass
            and extreme_pass
        ),

        trust_initialization_pass=(
            trust_initialization_pass
        ),

        allow=allow,

        constrain=constrain,

        block=block,

        constrain_pass=constrain_pass,

        high_risk_pass=high_risk_pass,

        low_trust_pass=low_trust_pass,

        policy_precedence_pass=(
            policy_precedence_pass
        ),
    )

    # ------------------------------------------------------------------------
    # Process exit status
    # ------------------------------------------------------------------------

    if not benchmark_passed:

        raise SystemExit(1)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

