from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from agents.sac_agent import SACAgent
from environment.irf_env import IRFConfig, IRFEnvironment
from trust.governance import GovernanceEngine


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "best_sac_irf_phase3_adaptive_trust"
    / "sac_phase3_adaptive_trust.pt"
)


# ============================================================
# CONSTANTS
# ============================================================

NUM_USERS = 20
NUM_RIS_ELEMENTS = 64

STATE_DIM = 100
ACTION_DIM = 40

MAX_STEPS = 200

DEFAULT_SEED = 42


# ============================================================
# PHASE-3 ENVIRONMENT
# ============================================================

def create_environment(
    seed: int = DEFAULT_SEED,
) -> IRFEnvironment:
    """
    Create the Phase-3 IRF environment.

    IMPORTANT
    ---------
    This configuration intentionally preserves the existing
    research environment and adaptive-trust formulation.

    Governance is an external control layer. It does not replace
    the domain-level trust dynamics implemented inside IRFEnvironment.
    """

    config = IRFConfig(
        # ----------------------------------------------------
        # Network
        # ----------------------------------------------------
        num_users=NUM_USERS,
        num_ris_elements=NUM_RIS_ELEMENTS,
        bandwidth_hz=100e6,
        carrier_frequency_hz=28e9,

        # ----------------------------------------------------
        # Power
        # ----------------------------------------------------
        max_power_w=1.0,
        circuit_power_w=0.1,

        # ----------------------------------------------------
        # Noise
        # ----------------------------------------------------
        noise_figure_db=7.0,
        noise_density_dbm_hz=-174.0,

        # ----------------------------------------------------
        # Episode
        # ----------------------------------------------------
        max_steps=MAX_STEPS,

        # ----------------------------------------------------
        # Phase-3 control
        # ----------------------------------------------------
        optimize_ris=False,
        fixed_trust=False,

        # ----------------------------------------------------
        # Adaptive Trust
        # ----------------------------------------------------
        trust_memory=0.90,
        trust_learning_rate=0.10,
        trust_target_rate_bps=1e7,

        trust_service_weight=0.35,
        trust_interference_weight=0.20,
        trust_queue_weight=0.25,
        trust_instability_weight=0.20,

        trust_neutral_point=0.50,
        trust_floor=0.0,

        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------
        se_weight=0.45,
        ee_weight=0.20,

        trust_weight=0.15,
        trust_delta_weight=0.10,

        interference_penalty=0.05,
        power_penalty=0.03,
        queue_penalty=0.07,

        # ----------------------------------------------------
        # Normalization
        # ----------------------------------------------------
        se_target=0.20,
        ee_target=1e6,

        # ----------------------------------------------------
        # RNG
        # ----------------------------------------------------
        seed=seed,
    )

    env = IRFEnvironment(config)

    return env


# ============================================================
# SAC AGENT
# ============================================================

def create_agent(
    env: IRFEnvironment,
) -> SACAgent:
    """
    Create the SAC agent using the same architecture expected
    by the Phase-3 checkpoint.
    """

    agent = SACAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,

        hidden_dim=256,

        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,

        gamma=0.99,
        tau=0.005,

        buffer_size=100_000,
        batch_size=256,
    )

    return agent


def load_agent(
    agent: SACAgent,
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> None:
    """
    Load the trained SAC checkpoint.

    The checkpoint is required for the live research runtime.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "\n"
            "SAC checkpoint not found.\n"
            f"Expected path:\n{checkpoint_path}\n"
        )

    agent.load(
        str(checkpoint_path)
    )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    agent.actor.eval()
    agent.critic.eval()
    agent.target_critic.eval()


# ============================================================
# ACTION VALIDATION
# ============================================================

def validate_action(
    action: np.ndarray,
    action_dim: int,
    name: str = "action",
) -> np.ndarray:
    """
    Validate and normalize an action before it reaches the
    governance engine or environment.
    """

    action = np.asarray(
        action,
        dtype=np.float32,
    )

    if action.ndim != 1:
        raise ValueError(
            f"{name} must be 1-dimensional. "
            f"Received shape={action.shape}"
        )

    if action.shape[0] != action_dim:
        raise ValueError(
            f"{name} dimension mismatch. "
            f"Expected {action_dim}, "
            f"received {action.shape[0]}"
        )

    if not np.all(
        np.isfinite(action)
    ):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return action


# ============================================================
# SAFE FALLBACK
# ============================================================

def make_block_fallback(
    action_dim: int,
) -> np.ndarray:
    """
    Create a neutral zero-action fallback for BLOCK decisions.

    BLOCK is enforced by the runtime here.

    GovernanceEngine decides BLOCK.
    live_runner prevents the blocked action from reaching
    IRFEnvironment.
    """

    return np.zeros(
        action_dim,
        dtype=np.float32,
    )


# ============================================================
# SAFE NUMERIC HELPERS
# ============================================================

def safe_mean(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Safely calculate the mean of a scalar/array-like value.
    """

    try:
        array = np.asarray(
            value,
            dtype=np.float64,
        )

        if array.size == 0:
            return float(default)

        finite = array[
            np.isfinite(array)
        ]

        if finite.size == 0:
            return float(default)

        return float(
            np.mean(finite)
        )

    except Exception:
        return float(default)


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert a value to a finite float.
    """

    try:
        result = float(value)

        if not np.isfinite(result):
            return float(default)

        return result

    except Exception:
        return float(default)


# ============================================================
# TELEMETRY EXTRACTION
# ============================================================

def get_telemetry(
    env: IRFEnvironment,
    info: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract telemetry from the actual IRF environment.

    The function intentionally preserves both:
        - environment-level arrays
        - info-level scalar metrics

    This makes the telemetry useful for both governance and
    dashboard visualization.
    """

    # --------------------------------------------------------
    # Trust
    # --------------------------------------------------------

    trust_source = getattr(
        env,
        "trust",
        info.get(
            "trust",
            0.0,
        ),
    )

    trust = safe_mean(
        trust_source,
        default=safe_float(
            info.get("trust", 0.0)
        ),
    )

    # --------------------------------------------------------
    # Queue
    # --------------------------------------------------------

    queue_source = getattr(
        env,
        "queue",
        info.get(
            "queue",
            0.0,
        ),
    )

    queue = safe_mean(
        queue_source,
        default=safe_float(
            info.get("queue", 0.0)
        ),
    )

    # --------------------------------------------------------
    # Interference
    # --------------------------------------------------------

    interference_source = getattr(
        env,
        "interference",
        info.get(
            "interference",
            0.0,
        ),
    )

    interference = safe_mean(
        np.abs(
            np.asarray(
                interference_source,
                dtype=np.float64,
            )
        ),
        default=safe_float(
            info.get(
                "interference",
                0.0,
            )
        ),
    )

    # --------------------------------------------------------
    # Power
    # --------------------------------------------------------

    power_source = getattr(
        env,
        "power",
        info.get(
            "power",
            0.0,
        ),
    )

    power = safe_mean(
        power_source,
        default=safe_float(
            info.get("power", 0.0)
        ),
    )

    # --------------------------------------------------------
    # SINR
    # --------------------------------------------------------

    sinr_source = getattr(
        env,
        "sinr",
        info.get(
            "sinr",
            0.0,
        ),
    )

    sinr = safe_mean(
        sinr_source,
        default=safe_float(
            info.get("sinr", 0.0)
        ),
    )

    # --------------------------------------------------------
    # Info metrics
    # --------------------------------------------------------

    spectral_efficiency = safe_float(
        info.get(
            "spectral_efficiency",
            0.0,
        )
    )

    energy_efficiency = safe_float(
        info.get(
            "energy_efficiency",
            0.0,
        )
    )

    rate = safe_float(
        info.get(
            "rate",
            0.0,
        )
    )

    behavior_score = safe_float(
        info.get(
            "behavior_score",
            0.0,
        )
    )

    trust_delta = safe_float(
        info.get(
            "trust_delta",
            0.0,
        )
    )

    arrival_mean = safe_float(
        info.get(
            "arrival_mean",
            0.0,
        )
    )

    arrival_sum = safe_float(
        info.get(
            "arrival_sum",
            0.0,
        )
    )

    # --------------------------------------------------------
    # Interference ratio
    # --------------------------------------------------------

    interference_ratio = safe_float(
        info.get(
            "interference_ratio",
            info.get(
                "interference",
                0.0,
            ),
        )
    )

    # --------------------------------------------------------
    # Return telemetry
    # --------------------------------------------------------

    return {
        "trust": trust,

        "queue": queue,

        "interference": interference,

        "power": power,

        "sinr": sinr,

        "spectral_efficiency": spectral_efficiency,

        "energy_efficiency": energy_efficiency,

        "rate": rate,

        "behavior_score": behavior_score,

        "trust_delta": trust_delta,

        "arrival_mean": arrival_mean,

        "arrival_sum": arrival_sum,

        "interference_ratio": interference_ratio,
    }


# ============================================================
# PRE-ACTION TELEMETRY
# ============================================================

def get_pre_action_telemetry(
    env: IRFEnvironment,
) -> dict[str, Any]:
    """
    Collect only the telemetry required by the governance
    engine before executing an action.

    This intentionally avoids modifying the environment.
    """

    queue = np.asarray(
        getattr(
            env,
            "queue",
            np.zeros(
                env.num_users,
                dtype=np.float32,
            ),
        ),
        dtype=np.float32,
    )

    interference = np.asarray(
        getattr(
            env,
            "interference",
            np.zeros(
                env.num_users,
                dtype=np.float32,
            ),
        ),
        dtype=np.float32,
    )

    return {
        "queue": queue.copy(),

        "interference": interference.copy(),
    }


def get_current_trust(
    env: IRFEnvironment,
) -> float:
    """
    Read the current domain-level trust maintained by
    IRFEnvironment.

    This is the authoritative trust signal for the current
    Phase-3 runtime.
    """

    trust = getattr(
        env,
        "trust",
        np.ones(
            env.num_users,
            dtype=np.float32,
        ),
    )

    return safe_mean(
        trust,
        default=1.0,
    )


# ============================================================
# GOVERNANCE DECISION
# ============================================================

def evaluate_governance(
    governance: GovernanceEngine,
    proposed_action: np.ndarray,
    env: IRFEnvironment,
) -> Any:
    """
    Evaluate a proposed SAC action through the GovernanceEngine.
    """

    current_trust = get_current_trust(
        env
    )

    current_telemetry = get_pre_action_telemetry(
        env
    )

    decision = governance.evaluate(
        proposed_action,
        trust_score=current_trust,
        telemetry=current_telemetry,
    )

    return decision


# ============================================================
# EXECUTION CONTROL
# ============================================================

def resolve_executed_action(
    decision: Any,
    proposed_action: np.ndarray,
    action_dim: int,
) -> tuple[np.ndarray, bool]:
    """
    Convert a governance decision into the actual action that
    is allowed to reach the IRF environment.

    Returns
    -------
    executed_action
    fallback_used
    """

    status = str(
        getattr(
            decision,
            "status",
            "",
        )
    ).upper()

    # --------------------------------------------------------
    # BLOCK
    # --------------------------------------------------------

    if status == "BLOCK":

        executed_action = make_block_fallback(
            action_dim
        )

        return (
            executed_action,
            True,
        )

    # --------------------------------------------------------
    # ALLOW / CONSTRAIN / other non-block status
    # --------------------------------------------------------

    decision_action = getattr(
        decision,
        "action",
        proposed_action,
    )

    executed_action = validate_action(
        decision_action,
        action_dim=action_dim,
        name="governed action",
    )

    return (
        executed_action,
        False,
    )


# ============================================================
# GOVERNANCE RECORD
# ============================================================

def build_governance_record(
    decision: Any,
    proposed_action: np.ndarray,
    executed_action: np.ndarray,
    reward: float,
    telemetry: dict[str, Any],
    done: bool,
    fallback_used: bool,
    env: IRFEnvironment,
) -> dict[str, Any]:
    """
    Build a dashboard-safe governance record.

    The record contains enough information to inspect:

        SAC proposal
            ↓
        Governance decision
            ↓
        Actual executed action
            ↓
        IRF outcome
    """

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    risk = getattr(
        decision,
        "risk",
        None,
    )

    risk_score = safe_float(
        getattr(
            risk,
            "risk_score",
            0.0,
        )
    )

    risk_level = str(
        getattr(
            risk,
            "risk_level",
            "UNKNOWN",
        )
    )

    trust_score = safe_float(
        getattr(
            risk,
            "trust_score",
            get_current_trust(env),
        )
    )

    action_anomaly = safe_float(
        getattr(
            risk,
            "action_anomaly",
            0.0,
        )
    )

    # --------------------------------------------------------
    # Policy
    # --------------------------------------------------------

    policy = getattr(
        decision,
        "policy",
        None,
    )

    policy_allowed = bool(
        getattr(
            policy,
            "allowed",
            True,
        )
    )

    policy_violations = list(
        getattr(
            policy,
            "violations",
            [],
        )
        or []
    )

    # --------------------------------------------------------
    # Action delta
    # --------------------------------------------------------

    action_delta = np.abs(
        executed_action
        - proposed_action
    )

    # --------------------------------------------------------
    # Governance status
    # --------------------------------------------------------

    governance_status = str(
        getattr(
            decision,
            "status",
            "UNKNOWN",
        )
    ).upper()

    reason = str(
        getattr(
            decision,
            "reason",
            "",
        )
    )

    modified = bool(
        getattr(
            decision,
            "modified",
            False,
        )
    )

    # BLOCK fallback itself is also a modification of the
    # proposed action, even if the GovernanceEngine did not
    # explicitly mark it as modified.
    if fallback_used:
        modified = True

    # --------------------------------------------------------
    # Step
    # --------------------------------------------------------

    step_count = int(
        getattr(
            env,
            "step_count",
            0,
        )
    )

    # --------------------------------------------------------
    # Record
    # --------------------------------------------------------

    return {
        # Runtime
        "step": step_count,
        "done": bool(done),

        # Reward
        "reward": safe_float(
            reward
        ),

        # Governance
        "governance": governance_status,
        "reason": reason,

        # Risk
        "risk_score": risk_score,
        "risk_level": risk_level,
        "trust_score": trust_score,
        "action_anomaly": action_anomaly,

        # Policy
        "policy_allowed": policy_allowed,
        "policy_violations": policy_violations,

        # Constraint / fallback
        "modified": modified,
        "fallback_used": bool(
            fallback_used
        ),

        # Action difference
        "max_action_delta": safe_float(
            np.max(action_delta)
        ),

        "mean_action_delta": safe_float(
            np.mean(action_delta)
        ),

        # Actions
        "proposed_action": proposed_action.copy(),

        "executed_action": executed_action.copy(),

        # Outcome telemetry
        "telemetry": telemetry,
    }


# ============================================================
# SINGLE LIVE STEP
# ============================================================

def run_step(
    env: IRFEnvironment,
    agent: SACAgent,
    governance: GovernanceEngine,
    state: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Execute exactly ONE live closed-loop step.

    Pipeline
    --------
        State
          ↓
        SAC
          ↓
        Proposed Action
          ↓
        Governance
          ├── ALLOW
          ├── CONSTRAIN
          └── BLOCK
          ↓
        Executed Action
          ↓
        IRF Environment
          ↓
        Reward + Telemetry
    """

    # ========================================================
    # 1. Validate incoming state
    # ========================================================

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    if state.ndim != 1:
        raise ValueError(
            f"State must be 1-dimensional. "
            f"Received shape={state.shape}"
        )

    if state.shape[0] != env.state_dim:
        raise ValueError(
            f"State dimension mismatch. "
            f"Expected {env.state_dim}, "
            f"received {state.shape[0]}"
        )

    if not np.all(
        np.isfinite(state)
    ):
        raise ValueError(
            "State contains NaN or infinite values."
        )

    # ========================================================
    # 2. SAC inference
    # ========================================================

    proposed_action = agent.select_action(
        state,
        evaluate=True,
    )

    proposed_action = validate_action(
        proposed_action,
        action_dim=env.action_dim,
        name="SAC proposed action",
    )

    # ========================================================
    # 3. Governance evaluation
    # ========================================================

    decision = evaluate_governance(
        governance=governance,
        proposed_action=proposed_action,
        env=env,
    )

    # ========================================================
    # 4. Resolve executable action
    # ========================================================

    (
        executed_action,
        fallback_used,
    ) = resolve_executed_action(
        decision=decision,
        proposed_action=proposed_action,
        action_dim=env.action_dim,
    )

    executed_action = validate_action(
        executed_action,
        action_dim=env.action_dim,
        name="executed action",
    )

    # ========================================================
    # 5. Execute actual IRF environment step
    # ========================================================

    (
        next_state,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(
        executed_action
    )

    done = bool(
        terminated
        or truncated
    )

    # ========================================================
    # 6. Validate next state
    # ========================================================

    next_state = np.asarray(
        next_state,
        dtype=np.float32,
    )

    if next_state.ndim != 1:
        raise ValueError(
            "Environment returned an invalid next_state "
            f"shape={next_state.shape}"
        )

    if next_state.shape[0] != env.state_dim:
        raise ValueError(
            "Environment next_state dimension mismatch. "
            f"Expected {env.state_dim}, "
            f"received {next_state.shape[0]}"
        )

    if not np.all(
        np.isfinite(next_state)
    ):
        raise ValueError(
            "Environment returned NaN or infinite "
            "values in next_state."
        )

    # ========================================================
    # 7. Post-action telemetry
    # ========================================================

    telemetry = get_telemetry(
        env,
        info,
    )

    # ========================================================
    # 8. Governance / action record
    # ========================================================

    record = build_governance_record(
        decision=decision,
        proposed_action=proposed_action,
        executed_action=executed_action,
        reward=reward,
        telemetry=telemetry,
        done=done,
        fallback_used=fallback_used,
        env=env,
    )

    # ========================================================
    # 9. Return
    # ========================================================

    return (
        next_state,
        record,
    )


# ============================================================
# LIVE SESSION
# ============================================================

def create_live_session(
    seed: int = DEFAULT_SEED,
) -> tuple[
    IRFEnvironment,
    SACAgent,
    GovernanceEngine,
    np.ndarray,
]:
    """
    Create a fresh SAC + Governance + IRF live session.

    Returns
    -------
    env
        IRF environment

    agent
        Loaded SAC agent

    governance
        GovernanceEngine

    state
        Initial environment state
    """

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = create_environment(
        seed=seed
    )

    # --------------------------------------------------------
    # SAC
    # --------------------------------------------------------

    agent = create_agent(
        env
    )

    load_agent(
        agent
    )

    # --------------------------------------------------------
    # Governance
    # --------------------------------------------------------

    governance = GovernanceEngine()

    # --------------------------------------------------------
    # Environment reset
    # --------------------------------------------------------

    state = env.reset(
        seed=seed
    )

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Initial state validation
    # --------------------------------------------------------

    if state.ndim != 1:
        raise ValueError(
            f"Initial state must be 1-dimensional. "
            f"Received shape={state.shape}"
        )

    if state.shape[0] != env.state_dim:
        raise ValueError(
            f"Initial state dimension mismatch. "
            f"Expected {env.state_dim}, "
            f"received {state.shape[0]}"
        )

    if not np.all(
        np.isfinite(state)
    ):
        raise ValueError(
            "Initial state contains NaN or infinite values."
        )

    return (
        env,
        agent,
        governance,
        state,
    )


# ============================================================
# OPTIONAL FULL EPISODE RUNNER
# ============================================================

def run_episode(
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """
    Run a complete governed SAC episode.

    This helper is useful for research testing and does not
    modify SAC weights.
    """

    (
        env,
        agent,
        governance,
        state,
    ) = create_live_session(
        seed=seed
    )

    records: list[dict[str, Any]] = []

    for _ in range(
        env.max_steps
    ):
        (
            state,
            record,
        ) = run_step(
            env=env,
            agent=agent,
            governance=governance,
            state=state,
        )

        records.append(
            record
        )

        if record["done"]:
            break

    return records


# ============================================================
# SUMMARY
# ============================================================

def summarize_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate a compact summary from governed runtime records.
    """

    if not records:
        return {
            "steps": 0,
            "total_reward": 0.0,
            "mean_reward": 0.0,
            "allow_count": 0,
            "constrain_count": 0,
            "block_count": 0,
            "fallback_count": 0,
            "mean_risk": 0.0,
            "mean_trust": 0.0,
        }

    rewards = np.asarray(
        [
            record.get(
                "reward",
                0.0,
            )
            for record in records
        ],
        dtype=np.float64,
    )

    risks = np.asarray(
        [
            record.get(
                "risk_score",
                0.0,
            )
            for record in records
        ],
        dtype=np.float64,
    )

    trusts = np.asarray(
        [
            record.get(
                "trust_score",
                0.0,
            )
            for record in records
        ],
        dtype=np.float64,
    )

    statuses = [
        str(
            record.get(
                "governance",
                "UNKNOWN",
            )
        ).upper()
        for record in records
    ]

    return {
        "steps": len(records),

        "total_reward": safe_float(
            np.sum(rewards)
        ),

        "mean_reward": safe_float(
            np.mean(rewards)
        ),

        "allow_count": statuses.count(
            "ALLOW"
        ),

        "constrain_count": statuses.count(
            "CONSTRAIN"
        ),

        "block_count": statuses.count(
            "BLOCK"
        ),

        "fallback_count": sum(
            bool(
                record.get(
                    "fallback_used",
                    False,
                )
            )
            for record in records
        ),

        "mean_risk": safe_float(
            np.mean(risks)
        ),

        "mean_trust": safe_float(
            np.mean(trusts)
        ),
    }