"""
TA-FDRL-IRF
H5-B Governance Operating-Region Benchmark

Purpose
-------
Evaluate governance execution behavior across controlled operating regions:

    ALLOW
    CONSTRAIN
    BLOCK

This benchmark does NOT modify:
    - SAC architecture
    - IRF environment
    - reward weights
    - Phase-4A scientific results
    - trained checkpoint

It evaluates the existing governance layer under controlled scenarios.

Important
---------
H5-B is an execution-level governance benchmark.

Trust, action, queue, and interference conditions are deliberately
controlled to exercise specific governance operating regions.

The controlled telemetry is passed directly to the GovernanceEngine.
The resulting governed action is then executed inside the real
TA-FDRL-IRF environment.

Therefore:

    Controlled Scenario
            ↓
    Governance Evaluation
            ↓
    Executed Action
            ↓
    IRF Environment
            ↓
    Reward / Environment Response

This benchmark is NOT a physical-world 6G measurement.
"""


from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch


# ============================================================================
# PROJECT ROOT BOOTSTRAP
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

from agents.sac_agent import SACAgent
from environment.irf_env import IRFEnvironment
from trust.governance import GovernanceEngine


# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "best_sac_irf_phase2.pt"
)

JSON_RESULT = (
    PROJECT_ROOT
    / "results"
    / "governance_operating_region_benchmark.json"
)

TEXT_RESULT = (
    PROJECT_ROOT
    / "results"
    / "governance_operating_region_benchmark.txt"
)


# ---------------------------------------------------------------------
# Model / environment dimensions
# ---------------------------------------------------------------------

STATE_DIM = 100
ACTION_DIM = 40

NUM_USERS = 20
STATE_FEATURES_PER_USER = 5

MAX_STEPS = 200


# ---------------------------------------------------------------------
# Controlled benchmark seeds
# ---------------------------------------------------------------------

SEEDS = [
    11,
    22,
    33,
    44,
    55,
]


# ============================================================================
# CONTROLLED GOVERNANCE SCENARIOS
# ============================================================================

SCENARIOS = [

    # -----------------------------------------------------------------
    # Scenario 1
    # -----------------------------------------------------------------
    {
        "name": "SAFE_ALLOW",

        # Governance trust
        "trust": 0.90,

        # 40-dimensional action baseline
        "action_value": 0.00,

        # Controlled telemetry
        "queue": 0.10,
        "interference": 0.10,

        # Expected governance decision
        "expected_status": "ALLOW",
    },


    # -----------------------------------------------------------------
    # Scenario 2
    # -----------------------------------------------------------------
    {
        "name": "MEDIUM_RISK_CONSTRAIN",

        # Exactly at minimum trust boundary.
        # Risk is raised through controlled telemetry and action anomaly.
        "trust": 0.60,

        # High proposed action magnitude.
        # Safe Envelope should reduce this.
        "action_value": 0.89,

        # Deliberately controlled high telemetry pressure.
        "queue": 1.00,
        "interference": 1.00,

        # Expected governance decision
        "expected_status": "CONSTRAIN",
    },


    # -----------------------------------------------------------------
    # Scenario 3
    # -----------------------------------------------------------------
    {
        "name": "HIGH_RISK_BLOCK",

        # Low trust contribution
        "trust": 0.15,

        # High action anomaly
        "action_value": 0.89,

        # High telemetry pressure
        "queue": 1.00,
        "interference": 1.00,

        # Expected governance decision
        "expected_status": "BLOCK",
    },


    # -----------------------------------------------------------------
    # Scenario 4
    # -----------------------------------------------------------------
    {
        "name": "LOW_TRUST_BLOCK",

        # Explicit trust violation
        "trust": 0.30,

        # Otherwise benign action
        "action_value": 0.00,

        # Low telemetry pressure
        "queue": 0.10,
        "interference": 0.10,

        # Expected governance decision
        "expected_status": "BLOCK",
    },


    # -----------------------------------------------------------------
    # Scenario 5
    # -----------------------------------------------------------------
    {
        "name": "POLICY_VIOLATION_BLOCK",

        # High trust alone should not bypass policy enforcement.
        "trust": 0.90,

        # Normal baseline action
        "action_value": 0.00,

        # Low telemetry pressure
        "queue": 0.10,
        "interference": 0.10,

        # Explicit policy violation
        "power_override": 0.95,

        # Expected governance decision
        "expected_status": "BLOCK",
    },
]


# ============================================================================
# RANDOMNESS CONTROL
# ============================================================================

def set_seed(seed: int) -> None:
    """
    Set deterministic RNG state where supported.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# ENVIRONMENT
# ============================================================================

def make_environment(seed: int) -> IRFEnvironment:
    """
    Create an IRF environment and attempt to seed it.
    """

    env = IRFEnvironment()

    try:
        env.seed(seed)
    except Exception:
        pass

    return env


def reset_environment(env, seed: int):
    """
    Support both modern and legacy reset APIs.
    """

    try:

        result = env.reset(
            seed=seed
        )

    except TypeError:

        try:
            env.seed(seed)
        except Exception:
            pass

        result = env.reset()

    # Gymnasium-style reset:
    # (observation, info)

    if isinstance(result, tuple):
        return result[0]

    return result


# ============================================================================
# SAC CHECKPOINT
# ============================================================================

def load_agent() -> SACAgent:
    """
    Load the existing Phase-2 SAC checkpoint.

    The checkpoint itself is not modified.
    """

    agent = SACAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
    )

    agent.load(
        str(CHECKPOINT)
    )

    return agent


# ============================================================================
# OPTIONAL ENVIRONMENT TELEMETRY EXTRACTION
# ============================================================================

def telemetry_from_state(
    state: np.ndarray,
) -> dict:
    """
    Extract queue and interference from the IRF state.

    This function is retained for diagnostic/reference purposes.

    H5-B controlled scenarios intentionally use controlled telemetry
    instead of this environment-derived telemetry when evaluating
    governance operating regions.
    """

    state_array = np.asarray(
        state,
        dtype=np.float32,
    )

    if state_array.size != STATE_DIM:
        return {}

    user_matrix = state_array.reshape(
        NUM_USERS,
        STATE_FEATURES_PER_USER,
    )

    return {
        "queue": user_matrix[:, 2],
        "interference": user_matrix[:, 1],
    }


# ============================================================================
# CONTROLLED TELEMETRY
# ============================================================================

def build_controlled_telemetry(
    scenario: dict,
) -> dict:
    """
    Build deterministic telemetry corresponding to the selected
    governance operating region.

    This is the critical H5-B correction.

    The scenario's queue/interference values are passed directly
    to the RiskEngine through GovernanceEngine.
    """

    queue_value = float(
        scenario["queue"]
    )

    interference_value = float(
        scenario["interference"]
    )

    return {
        "queue": np.full(
            NUM_USERS,
            queue_value,
            dtype=np.float32,
        ),

        "interference": np.full(
            NUM_USERS,
            interference_value,
            dtype=np.float32,
        ),
    }


# ============================================================================
# ENVIRONMENT STEP
# ============================================================================

def step_environment(
    env,
    action,
):
    """
    Support both Gymnasium 5-value and legacy 4-value step APIs.
    """

    result = env.step(
        action
    )

    # Gymnasium:
    #
    # observation,
    # reward,
    # terminated,
    # truncated,
    # info

    if len(result) == 5:

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = result

        done = bool(
            terminated or truncated
        )

    # Legacy Gym:
    #
    # observation,
    # reward,
    # done,
    # info

    elif len(result) == 4:

        (
            next_state,
            reward,
            done,
            info,
        ) = result

    else:

        raise RuntimeError(
            "Unexpected environment step output length: "
            f"{len(result)}"
        )

    if not isinstance(info, dict):
        info = {}

    return (
        next_state,
        float(reward),
        bool(done),
        info,
    )


# ============================================================================
# CONTROLLED ACTION
# ============================================================================

def build_controlled_action(
    scenario: dict,
    proposed_action: np.ndarray,
) -> np.ndarray:
    """
    Construct the controlled action used to isolate a governance
    operating region.

    The SAC proposal is retained as an upstream controller signal,
    but the benchmark deliberately replaces its values with
    scenario-controlled values.

    This avoids retraining the controller or changing the scientific core.
    """

    action = np.asarray(
        proposed_action,
        dtype=np.float32,
    ).copy()

    # Start with the scenario-wide action magnitude.
    action[:] = float(
        scenario["action_value"]
    )

    # Explicit policy violation for the policy test.
    if "power_override" in scenario:

        action[20:] = float(
            scenario["power_override"]
        )

    return action


# ============================================================================
# GOVERNANCE STATUS VALIDATION
# ============================================================================

def validate_status(
    actual_status: str,
    expected_status: str,
) -> bool:
    """
    Return True when governance produces the expected branch.
    """

    return actual_status == expected_status


# ============================================================================
# SCENARIO BENCHMARK
# ============================================================================

def run_scenario(
    agent,
    scenario: dict,
    seed: int,
) -> dict:
    """
    Execute one controlled governance scenario.

    Each scenario runs for up to MAX_STEPS inside the actual IRF
    environment.
    """

    set_seed(
        seed
    )

    env = make_environment(
        seed
    )

    state = reset_environment(
        env,
        seed,
    )

    # One governance engine per scenario run.
    governance = GovernanceEngine()

    # -----------------------------------------------------------------
    # Aggregates
    # -----------------------------------------------------------------

    total_reward = 0.0

    steps = 0

    allow_count = 0
    constrain_count = 0
    block_count = 0

    modified_count = 0

    action_deltas = []

    risks = []

    trusts = []

    branch_mismatch_count = 0

    # -----------------------------------------------------------------
    # Store reason frequencies for scientific diagnostics.
    # -----------------------------------------------------------------

    reason_counts = {}

    # -----------------------------------------------------------------
    # Step loop
    # -----------------------------------------------------------------

    while steps < MAX_STEPS:

        # =============================================================
        # 1. SAC PROPOSES AN ACTION
        # =============================================================

        proposed_action = agent.select_action(
            state,
            evaluate=True,
        )

        proposed_action = np.asarray(
            proposed_action,
            dtype=np.float32,
        )

        # =============================================================
        # 2. BUILD CONTROLLED ACTION
        # =============================================================

        controlled_action = build_controlled_action(
            scenario,
            proposed_action,
        )

        # =============================================================
        # 3. BUILD CONTROLLED TELEMETRY
        #
        # IMPORTANT:
        # This intentionally uses scenario-controlled telemetry.
        # =============================================================

        telemetry = build_controlled_telemetry(
            scenario
        )

        # =============================================================
        # 4. CONTROLLED TRUST
        # =============================================================

        trust_score = float(
            scenario["trust"]
        )

        # =============================================================
        # 5. GOVERNANCE DECISION
        # =============================================================

        decision = governance.evaluate(
            action=controlled_action,
            trust_score=trust_score,
            telemetry=telemetry,
        )

        # =============================================================
        # 6. EXTRACT EXECUTED ACTION
        # =============================================================

        executed_action = np.asarray(
            decision.action,
            dtype=np.float32,
        )

        # =============================================================
        # 7. GOVERNANCE BRANCH ACCOUNTING
        # =============================================================

        if decision.status == "ALLOW":

            allow_count += 1

        elif decision.status == "CONSTRAIN":

            constrain_count += 1

        elif decision.status == "BLOCK":

            block_count += 1

        else:

            raise RuntimeError(
                "Unknown governance status: "
                f"{decision.status}"
            )

        # =============================================================
        # 8. GOVERNANCE REASON ACCOUNTING
        # =============================================================

        reason = str(
            decision.reason
        )

        reason_counts[reason] = (
            reason_counts.get(
                reason,
                0,
            )
            + 1
        )

        # =============================================================
        # 9. ACTION MODIFICATION
        # =============================================================

        delta = float(
            np.max(
                np.abs(
                    executed_action
                    - controlled_action
                )
            )
        )

        action_deltas.append(
            delta
        )

        if delta > 1e-8:

            modified_count += 1

        # =============================================================
        # 10. RISK / TRUST RECORDING
        # =============================================================

        risks.append(
            float(
                decision.risk.risk_score
            )
        )

        trusts.append(
            float(
                decision.risk.trust_score
            )
        )

        # =============================================================
        # 11. EXPECTED BRANCH VALIDATION
        # =============================================================

        if not validate_status(
            decision.status,
            scenario["expected_status"],
        ):

            branch_mismatch_count += 1

        # =============================================================
        # 12. EXECUTE GOVERNED ACTION IN IRF
        # =============================================================

        (
            next_state,
            reward,
            done,
            info,
        ) = step_environment(
            env,
            executed_action,
        )

        total_reward += reward

        steps += 1

        state = next_state

        if done:
            break

    # -----------------------------------------------------------------
    # Determine dominant/primary branch
    # -----------------------------------------------------------------

    branch_counts = {
        "ALLOW": allow_count,
        "CONSTRAIN": constrain_count,
        "BLOCK": block_count,
    }

    actual_primary_status = max(
        branch_counts,
        key=branch_counts.get,
    )

    # -----------------------------------------------------------------
    # Return scenario result
    # -----------------------------------------------------------------

    return {
        "scenario": scenario["name"],

        "seed": seed,

        "expected_status": scenario[
            "expected_status"
        ],

        "actual_primary_status": (
            actual_primary_status
        ),

        "steps": steps,

        "total_reward": (
            total_reward
        ),

        "mean_reward": (
            total_reward / steps
            if steps > 0
            else 0.0
        ),

        "mean_trust": (
            mean(trusts)
            if trusts
            else 0.0
        ),

        "mean_risk": (
            mean(risks)
            if risks
            else 0.0
        ),

        "allow_count": allow_count,

        "constrain_count": (
            constrain_count
        ),

        "block_count": block_count,

        "modified_count": (
            modified_count
        ),

        "modification_rate": (
            modified_count / steps
            if steps > 0
            else 0.0
        ),

        "mean_action_delta": (
            mean(action_deltas)
            if action_deltas
            else 0.0
        ),

        "max_action_delta": (
            max(action_deltas)
            if action_deltas
            else 0.0
        ),

        "branch_mismatch_count": (
            branch_mismatch_count
        ),

        "branch_accuracy": (
            1.0
            if branch_mismatch_count == 0
            else 0.0
        ),

        "reason_counts": (
            reason_counts
        ),
    }


# ============================================================================
# SUMMARY
# ============================================================================

def summarize(
    results: list[dict],
) -> dict:
    """
    Aggregate all scenario/seed results.
    """

    grouped = {}

    for scenario in SCENARIOS:

        name = scenario["name"]

        rows = [
            row
            for row in results
            if row["scenario"] == name
        ]

        if not rows:
            continue

        total_steps = sum(
            row["steps"]
            for row in rows
        )

        # -------------------------------------------------------------
        # Aggregate reason counts
        # -------------------------------------------------------------

        combined_reasons = {}

        for row in rows:

            for reason, count in row[
                "reason_counts"
            ].items():

                combined_reasons[reason] = (
                    combined_reasons.get(
                        reason,
                        0,
                    )
                    + count
                )

        # -------------------------------------------------------------
        # Scenario summary
        # -------------------------------------------------------------

        grouped[name] = {

            "runs": len(rows),

            "expected_status": (
                scenario["expected_status"]
            ),

            "mean_total_reward": mean(
                r["total_reward"]
                for r in rows
            ),

            "mean_reward_per_step": mean(
                r["mean_reward"]
                for r in rows
            ),

            "mean_trust": mean(
                r["mean_trust"]
                for r in rows
            ),

            "mean_risk": mean(
                r["mean_risk"]
                for r in rows
            ),

            "allow_rate": (
                sum(
                    r["allow_count"]
                    for r in rows
                )
                / total_steps
                if total_steps > 0
                else 0.0
            ),

            "constrain_rate": (
                sum(
                    r["constrain_count"]
                    for r in rows
                )
                / total_steps
                if total_steps > 0
                else 0.0
            ),

            "block_rate": (
                sum(
                    r["block_count"]
                    for r in rows
                )
                / total_steps
                if total_steps > 0
                else 0.0
            ),

            "modification_rate": (
                sum(
                    r["modified_count"]
                    for r in rows
                )
                / total_steps
                if total_steps > 0
                else 0.0
            ),

            "mean_action_delta": mean(
                r["mean_action_delta"]
                for r in rows
            ),

            "max_action_delta": max(
                r["max_action_delta"]
                for r in rows
            ),

            "branch_mismatches": sum(
                r["branch_mismatch_count"]
                for r in rows
            ),

            "branch_accuracy": (
                1.0
                if sum(
                    r["branch_mismatch_count"]
                    for r in rows
                ) == 0
                else
                1.0
                -
                (
                    sum(
                        r[
                            "branch_mismatch_count"
                        ]
                        for r in rows
                    )
                    / total_steps
                )
            ),

            "reason_counts": (
                combined_reasons
            ),
        }

    # -----------------------------------------------------------------
    # Overall statistics
    # -----------------------------------------------------------------

    total_steps = sum(
        row["steps"]
        for row in results
    )

    total_mismatches = sum(
        row["branch_mismatch_count"]
        for row in results
    )

    total_modifications = sum(
        row["modified_count"]
        for row in results
    )

    overall_branch_accuracy = (
        1.0
        - (
            total_mismatches
            / total_steps
        )
        if total_steps > 0
        else 0.0
    )

    overall_modification_rate = (
        total_modifications
        / total_steps
        if total_steps > 0
        else 0.0
    )

    return {

        "scenario_summary": grouped,

        "total_runs": len(results),

        "total_steps": total_steps,

        "total_branch_mismatches": (
            total_mismatches
        ),

        "branch_accuracy": (
            overall_branch_accuracy
        ),

        "total_modifications": (
            total_modifications
        ),

        "overall_modification_rate": (
            overall_modification_rate
        ),
    }


# ============================================================================
# REPORT GENERATION
# ============================================================================

def build_report(
    all_results: list[dict],
    summary: dict,
) -> dict:
    """
    Build machine-readable benchmark report.
    """

    return {

        "benchmark": (
            "H5-B Governance "
            "Operating-Region Benchmark"
        ),

        "project": (
            "TA-FDRL-IRF"
        ),

        "purpose": (
            "Controlled execution-level "
            "evaluation of governance "
            "operating regions."
        ),

        "scientific_core_modified": False,

        "checkpoint": str(
            CHECKPOINT
        ),

        "state_dim": STATE_DIM,

        "action_dim": ACTION_DIM,

        "num_users": NUM_USERS,

        "max_steps": MAX_STEPS,

        "seeds": SEEDS,

        "scenarios": SCENARIOS,

        "results": all_results,

        "summary": summary,

        "scientific_interpretation": {

            "note_1": (
                "H5-B uses controlled trust, "
                "action, queue, and interference "
                "conditions to exercise specific "
                "governance operating regions."
            ),

            "note_2": (
                "Controlled telemetry is passed "
                "directly to the GovernanceEngine "
                "for operating-region evaluation."
            ),

            "note_3": (
                "The governed action is subsequently "
                "executed inside the existing IRF "
                "environment."
            ),

            "note_4": (
                "The benchmark does not claim that "
                "the controlled conditions represent "
                "measured physical-world 6G conditions."
            ),

            "note_5": (
                "The existing SAC checkpoint, "
                "SAC architecture, IRF environment, "
                "reward function, and Phase-4A results "
                "remain unchanged."
            ),

            "note_6": (
                "Performance differences observed "
                "under controlled scenarios represent "
                "governance execution effects and "
                "should not be interpreted as evidence "
                "of real-world network performance."
            ),
        },
    }


# ============================================================================
# TEXT REPORT
# ============================================================================

def write_text_report(
    summary: dict,
) -> None:
    """
    Write human-readable scientific benchmark report.
    """

    with open(
        TEXT_RESULT,
        "w",
        encoding="utf-8",
    ) as f:

        # -------------------------------------------------------------
        # Header
        # -------------------------------------------------------------

        f.write(
            "=" * 78
            + "\n"
        )

        f.write(
            "TA-FDRL-IRF H5-B GOVERNANCE "
            "OPERATING-REGION BENCHMARK\n"
        )

        f.write(
            "=" * 78
            + "\n\n"
        )

        # -------------------------------------------------------------
        # Configuration
        # -------------------------------------------------------------

        f.write(
            "BENCHMARK CONFIGURATION\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            f"Project root: {PROJECT_ROOT}\n"
        )

        f.write(
            f"Checkpoint: {CHECKPOINT}\n"
        )

        f.write(
            f"Seeds: {SEEDS}\n"
        )

        f.write(
            f"Max steps: {MAX_STEPS}\n"
        )

        f.write(
            f"State dimension: {STATE_DIM}\n"
        )

        f.write(
            f"Action dimension: {ACTION_DIM}\n"
        )

        f.write(
            f"Users: {NUM_USERS}\n"
        )

        f.write(
            "\n"
        )

        # -------------------------------------------------------------
        # Scenario results
        # -------------------------------------------------------------

        f.write(
            "SCENARIO RESULTS\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        for (
            name,
            data,
        ) in summary[
            "scenario_summary"
        ].items():

            f.write(
                f"\n{name}\n"
            )

            f.write(
                f"Expected status: "
                f"{data['expected_status']}\n"
            )

            f.write(
                f"Runs: "
                f"{data['runs']}\n"
            )

            f.write(
                f"Mean total reward: "
                f"{data['mean_total_reward']:.6f}\n"
            )

            f.write(
                f"Mean reward/step: "
                f"{data['mean_reward_per_step']:.6f}\n"
            )

            f.write(
                f"Mean trust: "
                f"{data['mean_trust']:.6f}\n"
            )

            f.write(
                f"Mean risk: "
                f"{data['mean_risk']:.6f}\n"
            )

            f.write(
                f"ALLOW rate: "
                f"{data['allow_rate']:.2%}\n"
            )

            f.write(
                f"CONSTRAIN rate: "
                f"{data['constrain_rate']:.2%}\n"
            )

            f.write(
                f"BLOCK rate: "
                f"{data['block_rate']:.2%}\n"
            )

            f.write(
                f"Modification rate: "
                f"{data['modification_rate']:.2%}\n"
            )

            f.write(
                f"Mean action delta: "
                f"{data['mean_action_delta']:.6f}\n"
            )

            f.write(
                f"Maximum action delta: "
                f"{data['max_action_delta']:.6f}\n"
            )

            f.write(
                f"Branch mismatches: "
                f"{data['branch_mismatches']}\n"
            )

            f.write(
                f"Branch accuracy: "
                f"{data['branch_accuracy']:.2%}\n"
            )

            f.write(
                "Reason counts:\n"
            )

            for (
                reason,
                count,
            ) in data[
                "reason_counts"
            ].items():

                f.write(
                    f"  {reason}: {count}\n"
                )

        # -------------------------------------------------------------
        # Overall validation
        # -------------------------------------------------------------

        f.write(
            "\n"
        )

        f.write(
            "OVERALL GOVERNANCE VALIDATION\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            f"Total runs: "
            f"{summary['total_runs']}\n"
        )

        f.write(
            f"Total steps: "
            f"{summary['total_steps']}\n"
        )

        f.write(
            f"Branch mismatches: "
            f"{summary['total_branch_mismatches']}\n"
        )

        f.write(
            f"Branch accuracy: "
            f"{summary['branch_accuracy']:.2%}\n"
        )

        f.write(
            f"Total action modifications: "
            f"{summary['total_modifications']}\n"
        )

        f.write(
            f"Overall modification rate: "
            f"{summary['overall_modification_rate']:.2%}\n"
        )

        # -------------------------------------------------------------
        # Scientific interpretation
        # -------------------------------------------------------------

        f.write(
            "\n"
        )

        f.write(
            "SCIENTIFIC INTERPRETATION\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            "H5-B evaluates governance execution "
            "behavior across controlled operating "
            "regions without modifying the existing "
            "SAC or IRF scientific core.\n\n"
        )

        f.write(
            "SAFE_ALLOW evaluates direct execution "
            "when trust and risk conditions are "
            "acceptable.\n\n"
        )

        f.write(
            "MEDIUM_RISK_CONSTRAIN evaluates the "
            "Safe Envelope intervention under a "
            "controlled medium-risk condition. "
            "Action modification is explicitly "
            "measured.\n\n"
        )

        f.write(
            "HIGH_RISK_BLOCK evaluates high-risk "
            "enforcement and verifies that unsafe "
            "actions are blocked.\n\n"
        )

        f.write(
            "LOW_TRUST_BLOCK evaluates the explicit "
            "minimum-trust governance gate.\n\n"
        )

        f.write(
            "POLICY_VIOLATION_BLOCK evaluates policy "
            "precedence independently of high trust.\n\n"
        )

        f.write(
            "The controlled conditions are intended "
            "to establish execution-level governance "
            "behavior. They are not physical-world "
            "6G measurements.\n\n"
        )

        f.write(
            "The benchmark intentionally preserves "
            "the existing SAC checkpoint, SAC "
            "architecture, IRF environment, reward "
            "function, and Phase-4A scientific "
            "results.\n"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print(
        "=" * 78
    )

    print(
        "TA-FDRL-IRF H5-B GOVERNANCE "
        "OPERATING-REGION BENCHMARK"
    )

    print(
        "=" * 78
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Checkpoint:   {CHECKPOINT}"
    )

    print(
        f"Seeds:        {SEEDS}"
    )

    print(
        f"Max steps:    {MAX_STEPS}"
    )

    print()

    # -----------------------------------------------------------------
    # Check checkpoint
    # -----------------------------------------------------------------

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT}"
        )

    all_results = []

    # -----------------------------------------------------------------
    # Scenario loop
    # -----------------------------------------------------------------

    for scenario in SCENARIOS:

        print(
            "-" * 78
        )

        print(
            f"SCENARIO: "
            f"{scenario['name']} "
            f"→ expected "
            f"{scenario['expected_status']}"
        )

        print(
            "-" * 78
        )

        # -------------------------------------------------------------
        # Seed loop
        # -------------------------------------------------------------

        for seed in SEEDS:

            # Fresh checkpoint-backed SAC agent
            agent = load_agent()

            result = run_scenario(
                agent=agent,
                scenario=scenario,
                seed=seed,
            )

            all_results.append(
                result
            )

            print(
                f"seed={seed} "
                f"reward={result['total_reward']:.6f} "
                f"trust={result['mean_trust']:.4f} "
                f"risk={result['mean_risk']:.4f} "
                f"ALLOW={result['allow_count']} "
                f"CONSTRAIN={result['constrain_count']} "
                f"BLOCK={result['block_count']} "
                f"modified={result['modified_count']} "
                f"delta={result['max_action_delta']:.4f}"
            )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    summary = summarize(
        all_results
    )

    # -----------------------------------------------------------------
    # Machine-readable report
    # -----------------------------------------------------------------

    report = build_report(
        all_results,
        summary,
    )

    with open(
        JSON_RESULT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Human-readable report
    # -----------------------------------------------------------------

    write_text_report(
        summary
    )

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------

    print()

    print(
        "=" * 78
    )

    print(
        "H5-B SUMMARY"
    )

    print(
        "=" * 78
    )

    for (
        name,
        data,
    ) in summary[
        "scenario_summary"
    ].items():

        print(
            f"{name:30s} "
            f"expected={data['expected_status']:9s} "
            f"ALLOW={data['allow_rate']:.1%} "
            f"CONSTRAIN={data['constrain_rate']:.1%} "
            f"BLOCK={data['block_rate']:.1%} "
            f"modified={data['modification_rate']:.1%}"
        )

    print()

    print(
        "Branch accuracy: "
        f"{summary['branch_accuracy']:.2%}"
    )

    print(
        "Overall modification rate: "
        f"{summary['overall_modification_rate']:.2%}"
    )

    print()

    print(
        f"JSON report: "
        f"{JSON_RESULT}"
    )

    print(
        f"Text report: "
        f"{TEXT_RESULT}"
    )

    print()

    if summary["branch_accuracy"] == 1.0:

        print(
            "H5-B GOVERNANCE OPERATING-REGION "
            "BENCHMARK PASSED"
        )

    else:

        print(
            "H5-B GOVERNANCE OPERATING-REGION "
            "BENCHMARK COMPLETED WITH BRANCH MISMATCHES"
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()