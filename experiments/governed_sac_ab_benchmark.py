
from __future__ import annotations

# ================================================================
# PROJECT ROOT IMPORT BOOTSTRAP
# ================================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ================================================================
# IMPORTS
# ================================================================

import json
import random

import numpy as np
import torch

from agents.sac_agent import SACAgent
from environment.irf_env import IRFEnvironment
from trust.governance import GovernanceEngine


# ================================================================
# CONFIGURATION
# ================================================================

CHECKPOINT = PROJECT_ROOT / "results" / "best_sac_irf_phase2.pt"

RESULT_FILE = (
    PROJECT_ROOT
    / "results"
    / "governed_sac_ab_benchmark.json"
)

TEXT_RESULT_FILE = (
    PROJECT_ROOT
    / "results"
    / "governed_sac_ab_benchmark.txt"
)

SEEDS = [11, 22, 33, 44, 55]

EPISODES_PER_SEED = 5

MAX_STEPS = 200

STATE_DIM = 100
ACTION_DIM = 40

NUM_USERS = 20
STATE_FEATURES_PER_USER = 5


# ================================================================
# REPRODUCIBILITY
# ================================================================

def set_seed(seed: int) -> None:
    """
    Set the major random number generators used by the benchmark.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================================================================
# ENVIRONMENT FACTORY
# ================================================================

def make_environment(seed: int):
    """
    Create the existing IRF environment.

    The scientific IRF environment itself is not modified.
    """

    env = IRFEnvironment()

    # Some project environments expose seed().
    if hasattr(env, "seed"):
        try:
            env.seed(seed)
        except Exception:
            pass

    return env


# ================================================================
# AGENT
# ================================================================

def load_agent() -> SACAgent:
    """
    Load the existing Phase-2 SAC checkpoint.

    No retraining is performed.
    """

    agent = SACAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
    )

    agent.load(str(CHECKPOINT))

    return agent


# ================================================================
# RESET COMPATIBILITY
# ================================================================

def reset_environment(env, seed: int | None = None):
    """
    Handle both modern Gym-style reset(seed=...)
    and the existing project reset() interface.
    """

    if seed is not None:

        try:
            result = env.reset(seed=seed)
        except TypeError:
            result = env.reset()

            if hasattr(env, "seed"):
                try:
                    env.seed(seed)
                except Exception:
                    pass
        except Exception:
            result = env.reset()

    else:
        result = env.reset()

    if isinstance(result, tuple):

        state = result[0]
        info = result[1] if len(result) > 1 else {}

        return (
            np.asarray(state, dtype=np.float32),
            info,
        )

    return (
        np.asarray(result, dtype=np.float32),
        {},
    )


# ================================================================
# TELEMETRY EXTRACTION
# ================================================================

def telemetry_from_state(state):
    """
    Existing TA-FDRL-IRF state layout:

        User 1:
            SINR
            interference
            queue
            power
            trust

        User 2:
            SINR
            interference
            queue
            power
            trust

        ...

        User 20

    Therefore:

        20 users × 5 features = 100 state dimensions.
    """

    state_array = np.asarray(
        state,
        dtype=np.float32,
    )

    if state_array.shape != (STATE_DIM,):
        raise ValueError(
            f"Unexpected state shape: "
            f"{state_array.shape}; "
            f"expected {(STATE_DIM,)}"
        )

    features = state_array.reshape(
        NUM_USERS,
        STATE_FEATURES_PER_USER,
    )

    return {
        "sinr": features[:, 0],
        "interference": features[:, 1],
        "queue": features[:, 2],
        "power": features[:, 3],
        "trust": features[:, 4],
    }


# ================================================================
# ENVIRONMENT STEP COMPATIBILITY
# ================================================================

def step_environment(env, action):
    """
    Handle both:

        Gymnasium:
            observation, reward, terminated, truncated, info

        Older Gym/project API:
            observation, reward, done, info
    """

    result = env.step(action)

    if len(result) == 5:

        next_state, reward, terminated, truncated, info = result

        done = bool(
            terminated or truncated
        )

        return (
            np.asarray(next_state, dtype=np.float32),
            float(reward),
            done,
            info,
        )

    if len(result) == 4:

        next_state, reward, done, info = result

        return (
            np.asarray(next_state, dtype=np.float32),
            float(reward),
            bool(done),
            info,
        )

    raise RuntimeError(
        "Unexpected environment step output "
        f"length: {len(result)}"
    )


# ================================================================
# INFO METRIC EXTRACTION
# ================================================================

def extract_info_metric(
    info,
    possible_keys,
):
    """
    Extract a scalar metric from environment info.

    The function is intentionally tolerant because the existing
    IRF environment may expose different metric names.

    Returns None when the metric is not available.
    """

    if not isinstance(info, dict):
        return None

    for key in possible_keys:

        if key not in info:
            continue

        value = info[key]

        try:

            array = np.asarray(
                value,
                dtype=np.float64,
            )

            if array.size == 0:
                continue

            if not np.all(np.isfinite(array)):
                continue

            return float(np.mean(array))

        except Exception:
            continue

    return None


# ================================================================
# EPISODE RUNNER
# ================================================================

def run_episode(
    agent: SACAgent,
    env,
    governed: bool,
    governance_engine: GovernanceEngine | None = None,
    episode_seed: int | None = None,
):
    """
    Run exactly one episode.

    Baseline:

        SAC -> IRF

    Governed:

        SAC -> Policy -> Risk -> Safe Envelope -> IRF
    """

    state, reset_info = reset_environment(
        env,
        seed=episode_seed,
    )

    if state.shape != (STATE_DIM,):
        raise ValueError(
            f"Unexpected reset state shape: {state.shape}"
        )

    total_reward = 0.0

    rewards = []

    trusts = []

    risks = []

    spectral_efficiencies = []

    energy_efficiencies = []

    allow_count = 0
    constrain_count = 0
    block_count = 0

    modification_count = 0

    action_deltas = []

    executed_steps = 0

    for _ in range(MAX_STEPS):

        # --------------------------------------------------------
        # SAC ACTION
        # --------------------------------------------------------

        proposed_action = agent.select_action(
            state,
            evaluate=True,
        )

        proposed_action = np.asarray(
            proposed_action,
            dtype=np.float32,
        ).copy()

        if proposed_action.shape != (ACTION_DIM,):
            raise ValueError(
                "Unexpected SAC action shape: "
                f"{proposed_action.shape}; "
                f"expected {(ACTION_DIM,)}"
            )

        executed_action = proposed_action.copy()

        # --------------------------------------------------------
        # TELEMETRY
        # --------------------------------------------------------

        telemetry = telemetry_from_state(state)

        current_trust = float(
            np.mean(
                telemetry["trust"]
            )
        )

        trusts.append(current_trust)

        # --------------------------------------------------------
        # GOVERNANCE
        # --------------------------------------------------------

        if governed:

            if governance_engine is None:
                raise ValueError(
                    "GovernanceEngine is required "
                    "for governed execution."
                )

            decision = governance_engine.evaluate(
                proposed_action,
                trust_score=current_trust,
                telemetry=telemetry,
            )

            # Governance-selected action.
            executed_action = np.asarray(
                decision.action,
                dtype=np.float32,
            ).copy()

            if decision.status == "ALLOW":

                allow_count += 1

            elif decision.status == "CONSTRAIN":

                constrain_count += 1

            elif decision.status == "BLOCK":

                block_count += 1

                # ------------------------------------------------
                # Explicit emergency fallback.
                #
                # A blocked action is never sent to the IRF
                # environment.
                # ------------------------------------------------

                executed_action = np.zeros(
                    ACTION_DIM,
                    dtype=np.float32,
                )

            else:

                raise RuntimeError(
                    "Unknown governance status: "
                    f"{decision.status}"
                )

            if decision.modified:
                modification_count += 1

            delta = np.abs(
                executed_action - proposed_action
            )

            action_deltas.append(
                float(np.mean(delta))
            )

            risks.append(
                float(
                    decision.risk.risk_score
                )
            )

        # --------------------------------------------------------
        # BASELINE
        # --------------------------------------------------------

        else:

            # Baseline does not apply governance.
            #
            # We still record trust telemetry so that the
            # scientific operating conditions can be compared.

            pass

        # --------------------------------------------------------
        # ENVIRONMENT STEP
        # --------------------------------------------------------

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

        rewards.append(reward)

        executed_steps += 1

        # --------------------------------------------------------
        # OPTIONAL SE / EE METRICS
        # --------------------------------------------------------

        se = extract_info_metric(
            info,
            [
                "spectral_efficiency",
                "spectral_efficiency_bps_hz",
                "se",
                "SE",
            ],
        )

        ee = extract_info_metric(
            info,
            [
                "energy_efficiency",
                "energy_efficiency_bits_joule",
                "ee",
                "EE",
            ],
        )

        if se is not None:
            spectral_efficiencies.append(se)

        if ee is not None:
            energy_efficiencies.append(ee)

        # --------------------------------------------------------
        # NEXT STATE
        # --------------------------------------------------------

        state = next_state

        if state.shape != (STATE_DIM,):
            raise ValueError(
                "Unexpected next-state shape: "
                f"{state.shape}"
            )

        if done:
            break

    # ============================================================
    # EPISODE RESULT
    # ============================================================

    result = {
        "total_reward": float(
            total_reward
        ),

        "mean_reward": float(
            np.mean(rewards)
        ) if rewards else 0.0,

        "steps": int(
            executed_steps
        ),

        "mean_trust": float(
            np.mean(trusts)
        ) if trusts else 0.0,

        "min_trust": float(
            np.min(trusts)
        ) if trusts else 0.0,

        "mean_risk": float(
            np.mean(risks)
        ) if risks else 0.0,

        "max_risk": float(
            np.max(risks)
        ) if risks else 0.0,

        "allow_count": int(
            allow_count
        ),

        "constrain_count": int(
            constrain_count
        ),

        "block_count": int(
            block_count
        ),

        "modification_count": int(
            modification_count
        ),

        "mean_action_delta": float(
            np.mean(action_deltas)
        ) if action_deltas else 0.0,

        "max_action_delta": float(
            np.max(action_deltas)
        ) if action_deltas else 0.0,

        "mean_spectral_efficiency": (
            float(
                np.mean(
                    spectral_efficiencies
                )
            )
            if spectral_efficiencies
            else None
        ),

        "mean_energy_efficiency": (
            float(
                np.mean(
                    energy_efficiencies
                )
            )
            if energy_efficiencies
            else None
        ),

        "se_measurements": int(
            len(spectral_efficiencies)
        ),

        "ee_measurements": int(
            len(energy_efficiencies)
        ),
    }

    return result


# ================================================================
# AGGREGATION
# ================================================================

def mean_metric(
    records,
    key,
):
    """
    Mean of a numeric metric.

    None values are ignored.
    """

    values = []

    for record in records:

        value = record.get(key)

        if value is None:
            continue

        try:

            value = float(value)

            if np.isfinite(value):
                values.append(value)

        except Exception:
            continue

    if not values:
        return None

    return float(
        np.mean(values)
    )


def sum_metric(
    records,
    key,
):
    """
    Sum an integer/count metric.
    """

    return int(
        sum(
            int(record.get(key, 0))
            for record in records
        )
    )


# ================================================================
# GROUP SUMMARY
# ================================================================

def summarize_group(
    records,
    governed: bool,
):
    """
    Produce aggregate statistics across all episodes.
    """

    summary = {
        "episodes": len(records),

        "mean_total_reward": mean_metric(
            records,
            "total_reward",
        ),

        "mean_reward_per_step": mean_metric(
            records,
            "mean_reward",
        ),

        "mean_steps": mean_metric(
            records,
            "steps",
        ),

        "mean_trust": mean_metric(
            records,
            "mean_trust",
        ),

        "mean_episode_min_trust": mean_metric(
            records,
            "min_trust",
        ),

        "mean_spectral_efficiency": mean_metric(
            records,
            "mean_spectral_efficiency",
        ),

        "mean_energy_efficiency": mean_metric(
            records,
            "mean_energy_efficiency",
        ),
    }

    if governed:

        total_steps = sum_metric(
            records,
            "steps",
        )

        allow_count = sum_metric(
            records,
            "allow_count",
        )

        constrain_count = sum_metric(
            records,
            "constrain_count",
        )

        block_count = sum_metric(
            records,
            "block_count",
        )

        modification_count = sum_metric(
            records,
            "modification_count",
        )

        summary.update(
            {
                "mean_risk": mean_metric(
                    records,
                    "mean_risk",
                ),

                "mean_episode_max_risk": mean_metric(
                    records,
                    "max_risk",
                ),

                "allow_count": allow_count,

                "constrain_count": constrain_count,

                "block_count": block_count,

                "modification_count": modification_count,

                "allow_rate": (
                    allow_count / total_steps
                    if total_steps > 0
                    else 0.0
                ),

                "constrain_rate": (
                    constrain_count / total_steps
                    if total_steps > 0
                    else 0.0
                ),

                "block_rate": (
                    block_count / total_steps
                    if total_steps > 0
                    else 0.0
                ),

                "modification_rate": (
                    modification_count / total_steps
                    if total_steps > 0
                    else 0.0
                ),

                "mean_action_delta": mean_metric(
                    records,
                    "mean_action_delta",
                ),

                "mean_max_action_delta": mean_metric(
                    records,
                    "max_action_delta",
                ),
            }
        )

    return summary


# ================================================================
# MAIN BENCHMARK
# ================================================================

def main():

    print("=" * 78)
    print(
        "TA-FDRL-IRF GOVERNED SAC A/B BENCHMARK"
    )
    print("=" * 78)

    print()
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
        f"Episodes/seed:{EPISODES_PER_SEED}"
    )

    print(
        f"Max steps:    {MAX_STEPS}"
    )

    print()

    # ============================================================
    # CHECKPOINT
    # ============================================================

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            "SAC checkpoint not found:\n"
            f"{CHECKPOINT}"
        )

    # ============================================================
    # RESULT CONTAINERS
    # ============================================================

    all_results = {
        "baseline": [],
        "governed": [],
    }

    # ============================================================
    # MATCHED A/B EXPERIMENT
    # ============================================================

    for seed in SEEDS:

        print()
        print("-" * 78)
        print(
            f"SEED {seed}"
        )
        print("-" * 78)

        # ========================================================
        # BASELINE
        # ========================================================

        for episode_index in range(
            EPISODES_PER_SEED
        ):

            episode_seed = (
                seed * 1000
                + episode_index
            )

            print(
                f"\nBaseline "
                f"episode "
                f"{episode_index + 1}/"
                f"{EPISODES_PER_SEED} "
                f"(seed={episode_seed})"
            )

            set_seed(
                episode_seed
            )

            baseline_agent = load_agent()

            baseline_env = make_environment(
                episode_seed
            )

            baseline_result = run_episode(
                baseline_agent,
                baseline_env,
                governed=False,
                governance_engine=None,
                episode_seed=episode_seed,
            )

            record = {
                "seed": seed,
                "episode": episode_index + 1,
                "episode_seed": episode_seed,
                **baseline_result,
            }

            all_results[
                "baseline"
            ].append(record)

            print(
                "  reward="
                f"{baseline_result['total_reward']:.6f} "
                "mean_reward="
                f"{baseline_result['mean_reward']:.6f} "
                "trust="
                f"{baseline_result['mean_trust']:.6f}"
            )

        # ========================================================
        # GOVERNED
        # ========================================================

        for episode_index in range(
            EPISODES_PER_SEED
        ):

            episode_seed = (
                seed * 1000
                + episode_index
            )

            print(
                f"\nGoverned "
                f"episode "
                f"{episode_index + 1}/"
                f"{EPISODES_PER_SEED} "
                f"(seed={episode_seed})"
            )

            set_seed(
                episode_seed
            )

            governed_agent = load_agent()

            governed_env = make_environment(
                episode_seed
            )

            # Instantiate GovernanceEngine once per episode,
            # not once per environment step.
            governance_engine = (
                GovernanceEngine()
            )

            governed_result = run_episode(
                governed_agent,
                governed_env,
                governed=True,
                governance_engine=governance_engine,
                episode_seed=episode_seed,
            )

            record = {
                "seed": seed,
                "episode": episode_index + 1,
                "episode_seed": episode_seed,
                **governed_result,
            }

            all_results[
                "governed"
            ].append(record)

            print(
                "  reward="
                f"{governed_result['total_reward']:.6f} "
                "mean_reward="
                f"{governed_result['mean_reward']:.6f} "
                "trust="
                f"{governed_result['mean_trust']:.6f}"
            )

            print(
                "  governance: "
                f"ALLOW={governed_result['allow_count']} "
                f"CONSTRAIN={governed_result['constrain_count']} "
                f"BLOCK={governed_result['block_count']}"
            )

    # ============================================================
    # SUMMARY
    # ============================================================

    baseline_summary = summarize_group(
        all_results["baseline"],
        governed=False,
    )

    governed_summary = summarize_group(
        all_results["governed"],
        governed=True,
    )

    # ============================================================
    # PERFORMANCE COMPARISON
    # ============================================================

    baseline_reward = (
        baseline_summary[
            "mean_total_reward"
        ]
    )

    governed_reward = (
        governed_summary[
            "mean_total_reward"
        ]
    )

    if (
        baseline_reward is not None
        and baseline_reward != 0
        and governed_reward is not None
    ):

        reward_change_pct = (
            (
                governed_reward
                - baseline_reward
            )
            / abs(baseline_reward)
            * 100.0
        )

    else:

        reward_change_pct = None

    comparison = {
        "reward_change_percent": (
            float(reward_change_pct)
            if reward_change_pct is not None
            else None
        ),

        "interpretation": (
            "Governance overhead or intervention "
            "effect under the tested conditions. "
            "Not evidence of universal optimality."
        ),
    }

    # ============================================================
    # CONFIGURATION RECORD
    # ============================================================

    output = {

        "benchmark": {
            "name": (
                "TA-FDRL-IRF Governed SAC "
                "A/B Benchmark"
            ),

            "type": (
                "controlled_matched_seed_evaluation"
            ),

            "scientific_core_modified": False,

            "sac_retrained": False,

            "reward_function_modified": False,
        },

        "configuration": {

            "project_root": str(
                PROJECT_ROOT
            ),

            "checkpoint": str(
                CHECKPOINT
            ),

            "seeds": SEEDS,

            "episodes_per_seed": (
                EPISODES_PER_SEED
            ),

            "total_episodes_per_condition": (
                len(SEEDS)
                * EPISODES_PER_SEED
            ),

            "max_steps": MAX_STEPS,

            "state_dim": STATE_DIM,

            "action_dim": ACTION_DIM,

            "num_users": NUM_USERS,

            "state_features_per_user": (
                STATE_FEATURES_PER_USER
            ),

            "safe_envelope": {

                "bandwidth_limit": 0.85,

                "power_limit": 0.75,
            },

            "governance_thresholds": {

                "minimum_trust": 0.60,

                "maximum_risk": 0.80,
            },
        },

        "system_paths": {

            "baseline": (
                "SAC -> IRF"
            ),

            "governed": (
                "SAC -> Policy -> Risk "
                "-> Safe Envelope -> IRF"
            ),
        },

        "results": all_results,

        "summary": {

            "baseline": baseline_summary,

            "governed": governed_summary,

            "comparison": comparison,
        },
    }

    # ============================================================
    # SAVE JSON
    # ============================================================

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_FILE.write_text(
        json.dumps(
            output,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # TEXT REPORT
    # ============================================================

    with TEXT_RESULT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write("=" * 78 + "\n")

        f.write(
            "TA-FDRL-IRF GOVERNED SAC A/B BENCHMARK\n"
        )

        f.write("=" * 78 + "\n\n")

        f.write(
            "CONTROLLED COMPARISON\n"
        )

        f.write(
            "Baseline: SAC -> IRF\n"
        )

        f.write(
            "Governed: SAC -> Policy -> Risk -> "
            "Safe Envelope -> IRF\n\n"
        )

        f.write(
            f"Seeds: {SEEDS}\n"
        )

        f.write(
            f"Episodes per seed: "
            f"{EPISODES_PER_SEED}\n"
        )

        f.write(
            f"Total episodes per condition: "
            f"{len(SEEDS) * EPISODES_PER_SEED}\n"
        )

        f.write(
            f"Max steps: {MAX_STEPS}\n"
        )

        f.write(
            f"State dimension: {STATE_DIM}\n"
        )

        f.write(
            f"Action dimension: {ACTION_DIM}\n\n"
        )

        # --------------------------------------------------------
        # BASELINE
        # --------------------------------------------------------

        f.write(
            "BASELINE SUMMARY\n"
        )

        f.write("-" * 78 + "\n")

        f.write(
            f"Mean total reward: "
            f"{baseline_summary['mean_total_reward']:.6f}\n"
        )

        f.write(
            f"Mean reward/step: "
            f"{baseline_summary['mean_reward_per_step']:.6f}\n"
        )

        f.write(
            f"Mean trust: "
            f"{baseline_summary['mean_trust']:.6f}\n"
        )

        f.write(
            f"Mean episode minimum trust: "
            f"{baseline_summary['mean_episode_min_trust']:.6f}\n"
        )

        if (
            baseline_summary[
                "mean_spectral_efficiency"
            ]
            is not None
        ):

            f.write(
                f"Mean spectral efficiency: "
                f"{baseline_summary['mean_spectral_efficiency']:.6f}\n"
            )

        else:

            f.write(
                "Mean spectral efficiency: "
                "NOT EXPOSED BY ENVIRONMENT INFO\n"
            )

        if (
            baseline_summary[
                "mean_energy_efficiency"
            ]
            is not None
        ):

            f.write(
                f"Mean energy efficiency: "
                f"{baseline_summary['mean_energy_efficiency']:.6f}\n"
            )

        else:

            f.write(
                "Mean energy efficiency: "
                "NOT EXPOSED BY ENVIRONMENT INFO\n"
            )

        f.write("\n")

        # --------------------------------------------------------
        # GOVERNED
        # --------------------------------------------------------

        f.write(
            "GOVERNED SUMMARY\n"
        )

        f.write("-" * 78 + "\n")

        f.write(
            f"Mean total reward: "
            f"{governed_summary['mean_total_reward']:.6f}\n"
        )

        f.write(
            f"Mean reward/step: "
            f"{governed_summary['mean_reward_per_step']:.6f}\n"
        )

        f.write(
            f"Mean trust: "
            f"{governed_summary['mean_trust']:.6f}\n"
        )

        f.write(
            f"Mean episode minimum trust: "
            f"{governed_summary['mean_episode_min_trust']:.6f}\n"
        )

        f.write(
            f"Mean risk: "
            f"{governed_summary['mean_risk']:.6f}\n"
        )

        f.write(
            f"Mean episode maximum risk: "
            f"{governed_summary['mean_episode_max_risk']:.6f}\n"
        )

        f.write(
            f"ALLOW count: "
            f"{governed_summary['allow_count']}\n"
        )

        f.write(
            f"CONSTRAIN count: "
            f"{governed_summary['constrain_count']}\n"
        )

        f.write(
            f"BLOCK count: "
            f"{governed_summary['block_count']}\n"
        )

        f.write(
            f"ALLOW rate: "
            f"{governed_summary['allow_rate'] * 100.0:.4f}%\n"
        )

        f.write(
            f"CONSTRAIN rate: "
            f"{governed_summary['constrain_rate'] * 100.0:.4f}%\n"
        )

        f.write(
            f"BLOCK rate: "
            f"{governed_summary['block_rate'] * 100.0:.4f}%\n"
        )

        f.write(
            f"Action modification count: "
            f"{governed_summary['modification_count']}\n"
        )

        f.write(
            f"Action modification rate: "
            f"{governed_summary['modification_rate'] * 100.0:.4f}%\n"
        )

        f.write(
            f"Mean action delta: "
            f"{governed_summary['mean_action_delta']:.6f}\n"
        )

        f.write(
            f"Mean maximum action delta: "
            f"{governed_summary['mean_max_action_delta']:.6f}\n"
        )

        if (
            governed_summary[
                "mean_spectral_efficiency"
            ]
            is not None
        ):

            f.write(
                f"Mean spectral efficiency: "
                f"{governed_summary['mean_spectral_efficiency']:.6f}\n"
            )

        else:

            f.write(
                "Mean spectral efficiency: "
                "NOT EXPOSED BY ENVIRONMENT INFO\n"
            )

        if (
            governed_summary[
                "mean_energy_efficiency"
            ]
            is not None
        ):

            f.write(
                f"Mean energy efficiency: "
                f"{governed_summary['mean_energy_efficiency']:.6f}\n"
            )

        else:

            f.write(
                "Mean energy efficiency: "
                "NOT EXPOSED BY ENVIRONMENT INFO\n"
            )

        f.write("\n")

        # --------------------------------------------------------
        # COMPARISON
        # --------------------------------------------------------

        f.write(
            "A/B PERFORMANCE COMPARISON\n"
        )

        f.write("-" * 78 + "\n")

        f.write(
            f"Baseline mean reward: "
            f"{baseline_reward:.6f}\n"
        )

        f.write(
            f"Governed mean reward: "
            f"{governed_reward:.6f}\n"
        )

        if reward_change_pct is not None:

            f.write(
                f"Reward change: "
                f"{reward_change_pct:.4f}%\n"
            )

        else:

            f.write(
                "Reward change: NOT AVAILABLE\n"
            )

        f.write("\n")

        # --------------------------------------------------------
        # SCIENTIFIC INTERPRETATION
        # --------------------------------------------------------

        f.write(
            "SCIENTIFIC INTERPRETATION\n"
        )

        f.write("-" * 78 + "\n")

        f.write(
            "This benchmark compares the existing Phase-2 "
            "SAC controller against the same controller wrapped "
            "by the governance layer.\n\n"
        )

        f.write(
            "Both conditions use the same SAC checkpoint, "
            "matched episode seeds, identical state and action "
            "dimensions, and the same maximum episode length.\n\n"
        )

        f.write(
            "The baseline executes SAC actions directly in "
            "the IRF environment.\n\n"
        )

        f.write(
            "The governed condition evaluates each proposed "
            "action through policy enforcement, risk assessment, "
            "and the medium-risk Safe Envelope before execution.\n\n"
        )

        f.write(
            "High-risk or policy-violating actions are blocked "
            "and replaced by an explicit zero-action fallback. "
            "Medium-risk actions may be constrained by the "
            "Safe Envelope.\n\n"
        )

        f.write(
            "The benchmark does not retrain the SAC agent and "
            "does not modify the original IRF reward function.\n\n"
        )

        f.write(
            "Any observed reward difference therefore represents "
            "the measured performance effect of governance "
            "intervention under the tested conditions.\n\n"
        )

        f.write(
            "A performance reduction is not automatically a "
            "failure: governance may trade some control performance "
            "for safety, trust compliance, or risk reduction.\n\n"
        )

        f.write(
            "This benchmark is an engineering evaluation of the "
            "integrated control loop and must not be interpreted "
            "as evidence of universal optimality or real-world "
            "6G deployment performance.\n"
        )

    # ============================================================
    # CONSOLE SUMMARY
    # ============================================================

    print()
    print("=" * 78)
    print(
        "BENCHMARK SUMMARY"
    )
    print("=" * 78)

    print(
        f"Baseline mean reward: "
        f"{baseline_reward:.6f}"
    )

    print(
        f"Governed mean reward: "
        f"{governed_reward:.6f}"
    )

    if reward_change_pct is not None:

        print(
            f"Reward change: "
            f"{reward_change_pct:.4f}%"
        )

    print()

    print(
        f"Baseline mean trust: "
        f"{baseline_summary['mean_trust']:.6f}"
    )

    print(
        f"Governed mean trust: "
        f"{governed_summary['mean_trust']:.6f}"
    )

    print(
        f"Governed mean risk: "
        f"{governed_summary['mean_risk']:.6f}"
    )

    print()

    print(
        f"ALLOW: "
        f"{governed_summary['allow_count']} "
        f"({governed_summary['allow_rate'] * 100.0:.2f}%)"
    )

    print(
        f"CONSTRAIN: "
        f"{governed_summary['constrain_count']} "
        f"({governed_summary['constrain_rate'] * 100.0:.2f}%)"
    )

    print(
        f"BLOCK: "
        f"{governed_summary['block_count']} "
        f"({governed_summary['block_rate'] * 100.0:.2f}%)"
    )

    print()

    print(
        f"Action modification: "
        f"{governed_summary['modification_count']} "
        f"({governed_summary['modification_rate'] * 100.0:.2f}%)"
    )

    print()

    print(
        f"JSON report: "
        f"{RESULT_FILE}"
    )

    print(
        f"Text report: "
        f"{TEXT_RESULT_FILE}"
    )

    print()
    print(
        "GOVERNED SAC A/B BENCHMARK COMPLETE"
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()

