from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from environment.irf_env import IRFConfig, IRFEnvironment
from agents.sac_agent import SACAgent
from trust.governance import GovernanceEngine


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "best_sac_irf_phase3_adaptive_trust"
    / "sac_phase3_adaptive_trust.pt"
)


# =========================================================
# PHASE-3 CONFIGURATION
# =========================================================

def create_environment(seed: int = 42) -> IRFEnvironment:
    """
    Create the same Phase-3 environment configuration
    used by train_sac.py.
    """

    config = IRFConfig(
        # Network
        num_users=20,
        num_ris_elements=64,
        bandwidth_hz=100e6,
        carrier_frequency_hz=28e9,

        # Power
        max_power_w=1.0,
        circuit_power_w=0.1,

        # Noise
        noise_figure_db=7.0,
        noise_density_dbm_hz=-174.0,

        # Episode
        max_steps=200,

        # Phase-3
        optimize_ris=False,
        fixed_trust=False,

        # Adaptive Trust
        trust_memory=0.90,
        trust_learning_rate=0.10,
        trust_target_rate_bps=1e7,

        trust_service_weight=0.35,
        trust_interference_weight=0.20,
        trust_queue_weight=0.25,
        trust_instability_weight=0.20,

        trust_neutral_point=0.50,
        trust_floor=0.0,

        # Reward
        se_weight=0.45,
        ee_weight=0.20,
        trust_weight=0.15,
        trust_delta_weight=0.10,
        interference_penalty=0.05,
        power_penalty=0.03,
        queue_penalty=0.07,

        # Normalization
        se_target=0.20,
        ee_target=1e6,

        seed=seed,
    )

    return IRFEnvironment(config)


# =========================================================
# SAC
# =========================================================

def create_agent(
    env: IRFEnvironment,
) -> SACAgent:
    """
    Create SAC with the same architecture used during
    Phase-3 training.
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
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAC checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    agent.load(
        str(checkpoint_path)
    )

    agent.actor.eval()
    agent.critic.eval()
    agent.target_critic.eval()


# =========================================================
# SAFE FALLBACK
# =========================================================

def make_block_fallback(
    action_dim: int,
) -> np.ndarray:
    """
    Explicit zero-action fallback for BLOCK decisions.

    The GovernanceEngine itself reports the proposed action
    when BLOCK occurs, so the dashboard runtime explicitly
    replaces it with a neutral zero action before env.step().
    """

    return np.zeros(
        action_dim,
        dtype=np.float32,
    )


# =========================================================
# TELEMETRY
# =========================================================

def get_telemetry(
    env: IRFEnvironment,
    info: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract live telemetry from the actual environment.

    Both scalar info values and current environment arrays
    are retained where available.
    """

    trust = float(
        np.mean(
            getattr(
                env,
                "trust",
                [info.get("trust", 0.0)],
            )
        )
    )

    queue_array = np.asarray(
        getattr(
            env,
            "queue",
            [info.get("queue", 0.0)],
        ),
        dtype=np.float32,
    )

    interference_array = np.asarray(
        getattr(
            env,
            "interference",
            [info.get("interference", 0.0)],
        ),
        dtype=np.float32,
    )

    power_array = np.asarray(
        getattr(
            env,
            "power",
            [info.get("power", 0.0)],
        ),
        dtype=np.float32,
    )

    sinr_array = np.asarray(
        getattr(
            env,
            "sinr",
            [info.get("sinr", 0.0)],
        ),
        dtype=np.float32,
    )

    return {
        "trust": trust,

        "queue": float(
            np.mean(queue_array)
        ),

        "interference": float(
            np.mean(
                np.abs(interference_array)
            )
        ),

        "power": float(
            np.mean(power_array)
        ),

        "sinr": float(
            np.mean(sinr_array)
        ),

        "spectral_efficiency": float(
            info.get(
                "spectral_efficiency",
                0.0,
            )
        ),

        "energy_efficiency": float(
            info.get(
                "energy_efficiency",
                0.0,
            )
        ),

        "rate": float(
            info.get(
                "rate",
                0.0,
            )
        ),

        "behavior_score": float(
            info.get(
                "behavior_score",
                0.0,
            )
        ),

        "trust_delta": float(
            info.get(
                "trust_delta",
                0.0,
            )
        ),

        "arrival_mean": float(
            info.get(
                "arrival_mean",
                0.0,
            )
        ),

        "arrival_sum": float(
            info.get(
                "arrival_sum",
                0.0,
            )
        ),

        "interference_ratio": float(
            info.get(
                "interference",
                0.0,
            )
        ),
    }


# =========================================================
# SINGLE LIVE STEP
# =========================================================

def run_step(
    env: IRFEnvironment,
    agent: SACAgent,
    governance: GovernanceEngine,
    state: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Execute ONE real SAC -> Governance -> IRF step.
    """

    # -----------------------------------------------------
    # 1. SAC inference
    # -----------------------------------------------------

    proposed_action = agent.select_action(
        state,
        evaluate=True,
    )

    proposed_action = np.asarray(
        proposed_action,
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # 2. Current telemetry before action
    # -----------------------------------------------------

    current_telemetry = {
        "queue": np.asarray(
            getattr(
                env,
                "queue",
                np.zeros(env.num_users),
            ),
            dtype=np.float32,
        ),

        "interference": np.asarray(
            getattr(
                env,
                "interference",
                np.zeros(env.num_users),
            ),
            dtype=np.float32,
        ),
    }

    current_trust = float(
        np.mean(
            getattr(
                env,
                "trust",
                np.ones(env.num_users),
            )
        )
    )

    # -----------------------------------------------------
    # 3. Governance
    # -----------------------------------------------------

    decision = governance.evaluate(
        proposed_action,
        trust_score=current_trust,
        telemetry=current_telemetry,
    )

    # -----------------------------------------------------
    # 4. Decide executed action
    # -----------------------------------------------------

    if decision.status == "BLOCK":

        executed_action = make_block_fallback(
            env.action_dim
        )

        fallback_used = True

    else:

        executed_action = np.asarray(
            decision.action,
            dtype=np.float32,
        )

        fallback_used = False

    # -----------------------------------------------------
    # 5. Actual IRF environment step
    # -----------------------------------------------------

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
        terminated or truncated
    )

    # -----------------------------------------------------
    # 6. Live telemetry
    # -----------------------------------------------------

    telemetry = get_telemetry(
        env,
        info,
    )

    # -----------------------------------------------------
    # 7. Action delta
    # -----------------------------------------------------

    action_delta = np.abs(
        executed_action
        - proposed_action
    )

    # -----------------------------------------------------
    # 8. Dashboard record
    # -----------------------------------------------------

    record = {
        "step": int(
            getattr(
                env,
                "step_count",
                0,
            )
        ),

        "reward": float(
            reward
        ),

        "governance": decision.status,

        "reason": decision.reason,

        "risk_score": float(
            decision.risk.risk_score
        ),

        "risk_level": decision.risk.risk_level,

        "trust_score": float(
            decision.risk.trust_score
        ),

        "action_anomaly": float(
            decision.risk.action_anomaly
        ),

        "policy_allowed": bool(
            decision.policy.allowed
        ),

        "policy_violations": list(
            decision.policy.violations
        ),

        "modified": bool(
            decision.modified
        ),

        "fallback_used": fallback_used,

        "max_action_delta": float(
            np.max(action_delta)
        ),

        "mean_action_delta": float(
            np.mean(action_delta)
        ),

        "proposed_action": proposed_action.copy(),

        "executed_action": executed_action.copy(),

        "telemetry": telemetry,

        "done": done,
    }

    return (
        next_state,
        record,
    )


# =========================================================
# LIVE EPISODE
# =========================================================

def create_live_session(
    seed: int = 42,
) -> tuple[
    IRFEnvironment,
    SACAgent,
    GovernanceEngine,
    np.ndarray,
]:
    """
    Initialize a fresh live simulation session.
    """

    env = create_environment(
        seed=seed
    )

    agent = create_agent(
        env
    )

    load_agent(
        agent
    )

    governance = GovernanceEngine()

    state = env.reset(
        seed=seed
    )

    return (
        env,
        agent,
        governance,
        state,
    )