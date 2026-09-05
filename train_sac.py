
# =========================================================
# train_sac.py
#
# TA-FDRL-IRF
# Trust-Aware Adaptive Federated Deep Reinforcement
# Learning for Intelligent Radio Fabric in 6G Networks
#
# Phase-3:
# SAC + Adaptive Trust
#
# Configuration:
#   - Fixed RIS
#   - Adaptive / Dynamic Trust
#   - SAC continuous-control agent
#   - Random-policy baseline
#   - Periodic deterministic evaluation
#   - Best-model checkpointing
#   - Training/evaluation metric export
#
# Compatible with:
#   environment/irf_env.py
#   agents/sac_agent.py
# =========================================================

from __future__ import annotations

import os
import random
from typing import Dict, List

import numpy as np
import torch

from environment.irf_env import (
    IRFConfig,
    IRFEnvironment,
)

from agents.sac_agent import (
    SACAgent,
)


# =========================================================
# GLOBAL CONFIGURATION
# =========================================================

SEED = 42

NUM_TRAIN_EPISODES = 300

EVAL_INTERVAL = 10

EVAL_EPISODES = 5

FINAL_EVAL_EPISODES = 10

STEPS_PER_EPISODE = 200


# =========================================================
# RESULTS
# =========================================================

RESULTS_DIR = "results"

BEST_MODEL_DIR = os.path.join(
    RESULTS_DIR,
    "best_sac_irf_phase3_adaptive_trust",
)

BEST_MODEL_PATH = os.path.join(
    BEST_MODEL_DIR,
    "sac_phase3_adaptive_trust.pt",
)

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "phase3_summary.txt",
)


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_global_seed(seed: int) -> None:
    """
    Set Python, NumPy and PyTorch random seeds.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def mean_last(
    values: List[float],
    window: int = 10,
) -> float:
    """
    Mean of the most recent values.
    """

    if not values:
        return 0.0

    start = max(
        0,
        len(values) - window,
    )

    return float(
        np.mean(
            values[start:]
        )
    )


def mean_first(
    values: List[float],
    window: int = 10,
) -> float:
    """
    Mean of the first values.
    """

    if not values:
        return 0.0

    return float(
        np.mean(
            values[:window]
        )
    )


def best_window_average(
    values: List[float],
    window: int = 10,
) -> float:
    """
    Best rolling/window average.
    """

    if not values:
        return 0.0

    if len(values) <= window:

        return float(
            np.mean(values)
        )

    window_means = []

    for start in range(
        0,
        len(values) - window + 1,
    ):

        current_mean = float(
            np.mean(
                values[
                    start:
                    start + window
                ]
            )
        )

        window_means.append(
            current_mean
        )

    return float(
        max(window_means)
    )


def safe_mean(
    values: List[float],
) -> float:
    """
    Numerically safe mean.
    """

    if not values:
        return 0.0

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array = np.nan_to_num(
        array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return float(
        np.mean(array)
    )


# =========================================================
# ENVIRONMENT FACTORY
# =========================================================

def create_environment(
    seed: int = SEED,
) -> IRFEnvironment:
    """
    Create the Phase-3 TA-FDRL-IRF environment.

    Phase-3 configuration:

        RIS:
            fixed

        Trust:
            adaptive

        State:
            100 dimensions

        Action:
            40 dimensions
    """

    config = IRFConfig(

        # -------------------------------------------------
        # Network
        # -------------------------------------------------

        num_users=20,

        num_ris_elements=64,

        bandwidth_hz=100e6,

        carrier_frequency_hz=28e9,

        # -------------------------------------------------
        # Power
        # -------------------------------------------------

        max_power_w=1.0,

        circuit_power_w=0.1,

        # -------------------------------------------------
        # Noise
        # -------------------------------------------------

        noise_figure_db=7.0,

        noise_density_dbm_hz=-174.0,

        # -------------------------------------------------
        # Episode
        # -------------------------------------------------

        max_steps=STEPS_PER_EPISODE,

        # -------------------------------------------------
        # Phase-3
        # -------------------------------------------------

        optimize_ris=False,

        fixed_trust=False,

        # -------------------------------------------------
        # Adaptive Trust
        # -------------------------------------------------

        trust_memory=0.90,

        trust_learning_rate=0.10,

        trust_target_rate_bps=1e7,

        trust_service_weight=0.35,

        trust_interference_weight=0.20,

        trust_queue_weight=0.25,

        trust_instability_weight=0.20,

        trust_neutral_point=0.50,

        trust_floor=0.0,

        # -------------------------------------------------
        # Reward
        # -------------------------------------------------

        se_weight=0.45,

        ee_weight=0.20,

        trust_weight=0.15,

        trust_delta_weight=0.10,

        interference_penalty=0.05,

        power_penalty=0.03,

        queue_penalty=0.07,

        # -------------------------------------------------
        # Reward normalization
        # -------------------------------------------------

        se_target=0.20,

        ee_target=1e6,

        # -------------------------------------------------
        # Seed
        # -------------------------------------------------

        seed=seed,
    )

    return IRFEnvironment(
        config
    )


# =========================================================
# METRIC EXTRACTION
# =========================================================

def extract_metrics(
    info: Dict,
) -> Dict[str, float]:
    """
    Extract all important metrics from environment info.
    """

    return {

        # -------------------------------------------------
        # Communication
        # -------------------------------------------------

        "se": float(
            info.get(
                "spectral_efficiency",
                0.0,
            )
        ),

        "ee": float(
            info.get(
                "energy_efficiency",
                0.0,
            )
        ),

        "total_rate": float(
            info.get(
                "total_rate",
                0.0,
            )
        ),

        "sinr": float(
            info.get(
                "mean_sinr",
                0.0,
            )
        ),

        # -------------------------------------------------
        # Trust
        # -------------------------------------------------

        "trust": float(
            info.get(
                "mean_trust",
                0.0,
            )
        ),

        "min_trust": float(
            info.get(
                "min_trust",
                0.0,
            )
        ),

        "max_trust": float(
            info.get(
                "max_trust",
                0.0,
            )
        ),

        "trust_delta": float(
            info.get(
                "trust_delta",
                0.0,
            )
        ),

        "behavior_score": float(
            info.get(
                "mean_behavior_score",
                0.0,
            )
        ),

        # -------------------------------------------------
        # Resource
        # -------------------------------------------------

        "power": float(
            info.get(
                "mean_power",
                0.0,
            )
        ),

        "total_power": float(
            info.get(
                "total_power",
                0.0,
            )
        ),

        # -------------------------------------------------
        # Queue / interference
        # -------------------------------------------------

        "queue": float(
            info.get(
                "mean_queue",
                0.0,
            )
        ),

        "interference": float(
            info.get(
                "interference_ratio",
                0.0,
            )
        ),

        # -------------------------------------------------
        # Normalized reward components
        # -------------------------------------------------

        "se_normalized": float(
            info.get(
                "se_normalized",
                0.0,
            )
        ),

        "ee_normalized": float(
            info.get(
                "ee_normalized",
                0.0,
            )
        ),

        "queue_normalized": float(
            info.get(
                "queue_normalized",
                0.0,
            )
        ),

        "power_normalized": float(
            info.get(
                "power_normalized",
                0.0,
            )
        ),

        # -------------------------------------------------
        # Reward
        # -------------------------------------------------

        "reward": float(
            info.get(
                "reward",
                0.0,
            )
        ),
    }


# =========================================================
# EPISODE ROLLOUT
# =========================================================

def run_episode(
    env: IRFEnvironment,
    action_selector,
    steps: int = STEPS_PER_EPISODE,
) -> Dict[str, float]:
    """
    Execute one complete episode.

    action_selector(state) must return an action.
    """

    state = env.reset()

    initial_trust = float(
        np.mean(env.trust)
    )

    episode_reward = 0.0

    step_rewards = []

    last_info = {}

    trust_trajectory = []

    behavior_trajectory = []

    queue_trajectory = []

    interference_trajectory = []

    se_trajectory = []

    ee_trajectory = []

    for _ in range(steps):

        action = action_selector(
            state
        )

        (
            next_state,
            reward,
            done,
            info,
        ) = env.step(action)

        episode_reward += float(
            reward
        )

        step_rewards.append(
            float(reward)
        )

        state = next_state

        last_info = info

        metrics = extract_metrics(
            info
        )

        trust_trajectory.append(
            metrics["trust"]
        )

        behavior_trajectory.append(
            metrics["behavior_score"]
        )

        queue_trajectory.append(
            metrics["queue"]
        )

        interference_trajectory.append(
            metrics["interference"]
        )

        se_trajectory.append(
            metrics["se"]
        )

        ee_trajectory.append(
            metrics["ee"]
        )

        if done:
            break

    final_metrics = extract_metrics(
        last_info
    )

    final_trust = final_metrics[
        "trust"
    ]

    return {

        "episode_reward":
            float(
                episode_reward
            ),

        "per_step_reward":
            safe_mean(
                step_rewards
            ),

        "se":
            final_metrics["se"],

        "ee":
            final_metrics["ee"],

        "total_rate":
            final_metrics["total_rate"],

        "sinr":
            final_metrics["sinr"],

        "initial_trust":
            initial_trust,

        "trust":
            final_trust,

        "trust_change":
            final_trust
            - initial_trust,

        "trust_delta":
            final_metrics[
                "trust_delta"
            ],

        "behavior_score":
            final_metrics[
                "behavior_score"
            ],

        "min_trust":
            final_metrics[
                "min_trust"
            ],

        "max_trust":
            final_metrics[
                "max_trust"
            ],

        "queue":
            final_metrics["queue"],

        "power":
            final_metrics["power"],

        "total_power":
            final_metrics[
                "total_power"
            ],

        "interference":
            final_metrics[
                "interference"
            ],

        "se_normalized":
            final_metrics[
                "se_normalized"
            ],

        "ee_normalized":
            final_metrics[
                "ee_normalized"
            ],

        "trust_trajectory":
            trust_trajectory,

        "behavior_trajectory":
            behavior_trajectory,

        "queue_trajectory":
            queue_trajectory,

        "interference_trajectory":
            interference_trajectory,

        "se_trajectory":
            se_trajectory,

        "ee_trajectory":
            ee_trajectory,
    }


# =========================================================
# RANDOM BASELINE
# =========================================================

def evaluate_random_policy(
    env: IRFEnvironment,
    episodes: int = 5,
) -> Dict[str, float]:
    """
    Evaluate a uniformly random policy.
    """

    episode_rewards = []

    per_step_rewards = []

    se_values = []

    ee_values = []

    total_rate_values = []

    sinr_values = []

    initial_trust_values = []

    final_trust_values = []

    trust_change_values = []

    trust_delta_values = []

    behavior_values = []

    queue_values = []

    power_values = []

    interference_values = []

    for episode in range(
        episodes
    ):

        env.reset(
            seed=10000 + episode
        )

        initial_trust = float(
            np.mean(env.trust)
        )

        episode_reward = 0.0

        step_rewards = []

        last_info = {}

        for _ in range(
            STEPS_PER_EPISODE
        ):

            action = np.random.uniform(
                -1.0,
                1.0,
                size=env.action_dim,
            )

            (
                next_state,
                reward,
                done,
                info,
            ) = env.step(action)

            episode_reward += float(
                reward
            )

            step_rewards.append(
                float(reward)
            )

            last_info = info

            if done:
                break

        metrics = extract_metrics(
            last_info
        )

        final_trust = metrics[
            "trust"
        ]

        episode_rewards.append(
            episode_reward
        )

        per_step_rewards.append(
            safe_mean(
                step_rewards
            )
        )

        se_values.append(
            metrics["se"]
        )

        ee_values.append(
            metrics["ee"]
        )

        total_rate_values.append(
            metrics["total_rate"]
        )

        sinr_values.append(
            metrics["sinr"]
        )

        initial_trust_values.append(
            initial_trust
        )

        final_trust_values.append(
            final_trust
        )

        trust_change_values.append(
            final_trust
            - initial_trust
        )

        trust_delta_values.append(
            metrics["trust_delta"]
        )

        behavior_values.append(
            metrics["behavior_score"]
        )

        queue_values.append(
            metrics["queue"]
        )

        power_values.append(
            metrics["power"]
        )

        interference_values.append(
            metrics["interference"]
        )

    return {

        "episode_reward":
            safe_mean(
                episode_rewards
            ),

        "per_step_reward":
            safe_mean(
                per_step_rewards
            ),

        "se":
            safe_mean(
                se_values
            ),

        "ee":
            safe_mean(
                ee_values
            ),

        "total_rate":
            safe_mean(
                total_rate_values
            ),

        "sinr":
            safe_mean(
                sinr_values
            ),

        "initial_trust":
            safe_mean(
                initial_trust_values
            ),

        "trust":
            safe_mean(
                final_trust_values
            ),

        "trust_change":
            safe_mean(
                trust_change_values
            ),

        "trust_delta":
            safe_mean(
                trust_delta_values
            ),

        "behavior_score":
            safe_mean(
                behavior_values
            ),

        "queue":
            safe_mean(
                queue_values
            ),

        "power":
            safe_mean(
                power_values
            ),

        "interference":
            safe_mean(
                interference_values
            ),
    }


# =========================================================
# SAC POLICY EVALUATION
# =========================================================

def evaluate_sac_policy(
    agent: SACAgent,
    env: IRFEnvironment,
    episodes: int = 5,
) -> Dict[str, float]:
    """
    Evaluate SAC deterministically.

    The SAC agent must support:

        select_action(
            state,
            evaluate=True
        )
    """

    episode_rewards = []

    per_step_rewards = []

    se_values = []

    ee_values = []

    total_rate_values = []

    sinr_values = []

    initial_trust_values = []

    final_trust_values = []

    trust_change_values = []

    trust_delta_values = []

    behavior_values = []

    min_trust_values = []

    max_trust_values = []

    queue_values = []

    power_values = []

    total_power_values = []

    interference_values = []

    for episode in range(
        episodes
    ):

        env.reset(
            seed=20000 + episode
        )

        initial_trust = float(
            np.mean(env.trust)
        )

        state = env._get_state()

        episode_reward = 0.0

        step_rewards = []

        last_info = {}

        for _ in range(
            STEPS_PER_EPISODE
        ):

            action = agent.select_action(
                state,
                evaluate=True,
            )

            (
                next_state,
                reward,
                done,
                info,
            ) = env.step(action)

            episode_reward += float(
                reward
            )

            step_rewards.append(
                float(reward)
            )

            state = next_state

            last_info = info

            if done:
                break

        metrics = extract_metrics(
            last_info
        )

        final_trust = metrics[
            "trust"
        ]

        episode_rewards.append(
            episode_reward
        )

        per_step_rewards.append(
            safe_mean(
                step_rewards
            )
        )

        se_values.append(
            metrics["se"]
        )

        ee_values.append(
            metrics["ee"]
        )

        total_rate_values.append(
            metrics["total_rate"]
        )

        sinr_values.append(
            metrics["sinr"]
        )

        initial_trust_values.append(
            initial_trust
        )

        final_trust_values.append(
            final_trust
        )

        trust_change_values.append(
            final_trust
            - initial_trust
        )

        trust_delta_values.append(
            metrics["trust_delta"]
        )

        behavior_values.append(
            metrics["behavior_score"]
        )

        min_trust_values.append(
            metrics["min_trust"]
        )

        max_trust_values.append(
            metrics["max_trust"]
        )

        queue_values.append(
            metrics["queue"]
        )

        power_values.append(
            metrics["power"]
        )

        total_power_values.append(
            metrics["total_power"]
        )

        interference_values.append(
            metrics["interference"]
        )

    return {

        "episode_reward":
            safe_mean(
                episode_rewards
            ),

        "per_step_reward":
            safe_mean(
                per_step_rewards
            ),

        "se":
            safe_mean(
                se_values
            ),

        "ee":
            safe_mean(
                ee_values
            ),

        "total_rate":
            safe_mean(
                total_rate_values
            ),

        "sinr":
            safe_mean(
                sinr_values
            ),

        "initial_trust":
            safe_mean(
                initial_trust_values
            ),

        "trust":
            safe_mean(
                final_trust_values
            ),

        "trust_change":
            safe_mean(
                trust_change_values
            ),

        "trust_delta":
            safe_mean(
                trust_delta_values
            ),

        "behavior_score":
            safe_mean(
                behavior_values
            ),

        "min_trust":
            safe_mean(
                min_trust_values
            ),

        "max_trust":
            safe_mean(
                max_trust_values
            ),

        "queue":
            safe_mean(
                queue_values
            ),

        "power":
            safe_mean(
                power_values
            ),

        "total_power":
            safe_mean(
                total_power_values
            ),

        "interference":
            safe_mean(
                interference_values
            ),
    }


# =========================================================
# PRINT EVALUATION
# =========================================================

def print_evaluation(
    title: str,
    result: Dict[str, float],
) -> None:
    """
    Print evaluation metrics.
    """

    print()

    print(title)

    print(
        f"Episode reward : "
        f"{result['episode_reward']:.6f}"
    )

    print(
        f"Per-step reward: "
        f"{result['per_step_reward']:.6f}"
    )

    print(
        f"SE             : "
        f"{result['se']:.6f}"
    )

    print(
        f"EE             : "
        f"{result['ee']:.4e}"
    )

    print(
        f"Initial trust  : "
        f"{result['initial_trust']:.6f}"
    )

    print(
        f"Final trust    : "
        f"{result['trust']:.6f}"
    )

    print(
        f"Trust change   : "
        f"{result['trust_change']:+.6f}"
    )

    print(
        f"Trust delta    : "
        f"{result['trust_delta']:+.8f}"
    )

    print(
        f"Behavior score : "
        f"{result['behavior_score']:.6f}"
    )

    print(
        f"Queue          : "
        f"{result['queue']:.6f}"
    )

    print(
        f"Power          : "
        f"{result['power']:.6f}"
    )

    print(
        f"Interference   : "
        f"{result['interference']:.6f}"
    )


# =========================================================
# SAVE ARRAY
# =========================================================

def save_array(
    filename: str,
    values,
) -> None:
    """
    Save NumPy array to results directory.
    """

    np.save(
        os.path.join(
            RESULTS_DIR,
            filename,
        ),
        np.asarray(
            values
        ),
    )


# =========================================================
# MAIN
# =========================================================

def main():

    # =====================================================
    # SETUP
    # =====================================================

    set_global_seed(
        SEED
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    os.makedirs(
        BEST_MODEL_DIR,
        exist_ok=True,
    )

    print(
        "=" * 72
    )

    print(
        "TA-FDRL-IRF | SAC Phase-3 Adaptive Trust"
    )

    print(
        "=" * 72
    )

    # =====================================================
    # ENVIRONMENT
    # =====================================================

    env = create_environment(
        SEED
    )

    print()

    print(
        f"Device: "
        f"initializing SAC..."
    )

    print(
        f"State dimension: "
        f"{env.state_dim}"
    )

    print(
        f"Action dimension: "
        f"{env.action_dim}"
    )

    print(
        f"Users: "
        f"{env.num_users}"
    )

    print(
        f"RIS elements: "
        f"{env.num_ris}"
    )

    print(
        f"RIS optimization: "
        f"{env.cfg.optimize_ris}"
    )

    print(
        f"Fixed trust: "
        f"{env.cfg.fixed_trust}"
    )

    print(
        f"Trust memory: "
        f"{env.cfg.trust_memory}"
    )

    print(
        f"Trust learning rate: "
        f"{env.cfg.trust_learning_rate}"
    )

    print(
        f"Trust target rate: "
        f"{env.cfg.trust_target_rate_bps:.2e} bps"
    )

    print(
        f"Trust neutral point: "
        f"{env.cfg.trust_neutral_point}"
    )

    print(
        f"Training episodes: "
        f"{NUM_TRAIN_EPISODES}"
    )

    print(
        f"Steps per episode: "
        f"{STEPS_PER_EPISODE}"
    )

    print(
        "=" * 72
    )

    # =====================================================
    # SAC AGENT
    # =====================================================

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

    print(
        f"Device: "
        f"{agent.device}"
    )

    print(
        "=" * 72
    )

    # =====================================================
    # RANDOM BASELINE
    # =====================================================

    print()

    print(
        "Evaluating random-policy baseline..."
    )

    random_baseline = (
        evaluate_random_policy(
            env,
            episodes=EVAL_EPISODES,
        )
    )

    print_evaluation(
        "RANDOM BASELINE",
        random_baseline,
    )

    print()

    print(
        "=" * 72
    )

    # =====================================================
    # TRAINING HISTORIES
    # =====================================================

    episode_rewards = []

    per_step_rewards = []

    se_history = []

    ee_history = []

    total_rate_history = []

    sinr_history = []

    trust_history = []

    initial_trust_history = []

    trust_change_history = []

    trust_delta_history = []

    behavior_history = []

    min_trust_history = []

    max_trust_history = []

    queue_history = []

    power_history = []

    total_power_history = []

    interference_history = []

    # =====================================================
    # EVALUATION HISTORIES
    # =====================================================

    evaluation_episodes = []

    evaluation_rewards = []

    evaluation_per_step = []

    evaluation_se = []

    evaluation_ee = []

    evaluation_trust = []

    evaluation_initial_trust = []

    evaluation_trust_change = []

    evaluation_trust_delta = []

    evaluation_behavior = []

    evaluation_queue = []

    evaluation_power = []

    evaluation_interference = []

    # =====================================================
    # BEST MODEL
    # =====================================================

    best_eval_reward = -np.inf

    best_eval_episode = 0

    # =====================================================
    # TRAINING LOOP
    # =====================================================

    for episode in range(
        1,
        NUM_TRAIN_EPISODES + 1,
    ):

        # -------------------------------------------------
        # Reset
        # -------------------------------------------------

        state = env.reset(
            seed=SEED + episode
        )

        initial_trust = float(
            np.mean(env.trust)
        )

        episode_reward = 0.0

        step_rewards = []

        last_info = {}

        updates = 0

        # -------------------------------------------------
        # Episode
        # -------------------------------------------------

        for _ in range(
            STEPS_PER_EPISODE
        ):

            action = agent.select_action(
                state,
                evaluate=False,
            )

            (
                next_state,
                reward,
                done,
                info,
            ) = env.step(action)

            # ---------------------------------------------
            # Replay buffer
            # ---------------------------------------------

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            # ---------------------------------------------
            # SAC update
            # ---------------------------------------------

            update_result = agent.update()

            if update_result is not None:

                updates += 1

            # ---------------------------------------------
            # Accumulate
            # ---------------------------------------------

            episode_reward += float(
                reward
            )

            step_rewards.append(
                float(reward)
            )

            state = next_state

            last_info = info

            if done:
                break

        # =================================================
        # Episode metrics
        # =================================================

        metrics = extract_metrics(
            last_info
        )

        final_trust = metrics[
            "trust"
        ]

        trust_change = (
            final_trust
            - initial_trust
        )

        per_step_reward = safe_mean(
            step_rewards
        )

        # -------------------------------------------------
        # Save histories
        # -------------------------------------------------

        episode_rewards.append(
            episode_reward
        )

        per_step_rewards.append(
            per_step_reward
        )

        se_history.append(
            metrics["se"]
        )

        ee_history.append(
            metrics["ee"]
        )

        total_rate_history.append(
            metrics["total_rate"]
        )

        sinr_history.append(
            metrics["sinr"]
        )

        trust_history.append(
            final_trust
        )

        initial_trust_history.append(
            initial_trust
        )

        trust_change_history.append(
            trust_change
        )

        trust_delta_history.append(
            metrics["trust_delta"]
        )

        behavior_history.append(
            metrics["behavior_score"]
        )

        min_trust_history.append(
            metrics["min_trust"]
        )

        max_trust_history.append(
            metrics["max_trust"]
        )

        queue_history.append(
            metrics["queue"]
        )

        power_history.append(
            metrics["power"]
        )

        total_power_history.append(
            metrics["total_power"]
        )

        interference_history.append(
            metrics["interference"]
        )

        # =================================================
        # PERIODIC EVALUATION
        # =================================================

        if (
            episode == 1
            or episode % EVAL_INTERVAL == 0
        ):

            evaluation = (
                evaluate_sac_policy(
                    agent,
                    env,
                    episodes=EVAL_EPISODES,
                )
            )

            evaluation_episodes.append(
                episode
            )

            evaluation_rewards.append(
                evaluation[
                    "episode_reward"
                ]
            )

            evaluation_per_step.append(
                evaluation[
                    "per_step_reward"
                ]
            )

            evaluation_se.append(
                evaluation["se"]
            )

            evaluation_ee.append(
                evaluation["ee"]
            )

            evaluation_trust.append(
                evaluation["trust"]
            )

            evaluation_initial_trust.append(
                evaluation[
                    "initial_trust"
                ]
            )

            evaluation_trust_change.append(
                evaluation[
                    "trust_change"
                ]
            )

            evaluation_trust_delta.append(
                evaluation[
                    "trust_delta"
                ]
            )

            evaluation_behavior.append(
                evaluation[
                    "behavior_score"
                ]
            )

            evaluation_queue.append(
                evaluation["queue"]
            )

            evaluation_power.append(
                evaluation["power"]
            )

            evaluation_interference.append(
                evaluation[
                    "interference"
                ]
            )

            # -------------------------------------------------
            # Best model selection
            #
            # Use evaluation per-step reward rather than
            # raw episode reward so different episode
            # lengths remain comparable.
            # -------------------------------------------------

            current_eval_score = (
                evaluation[
                    "per_step_reward"
                ]
            )

            if (
                current_eval_score
                > best_eval_reward
            ):

                best_eval_reward = (
                    current_eval_score
                )

                best_eval_episode = (
                    episode
                )

                agent.save(
                    BEST_MODEL_PATH
                )

                best_marker = (
                    " ★ BEST"
                )

            else:

                best_marker = ""

            # -------------------------------------------------
            # Training log
            # -------------------------------------------------

            print(
                f"Episode {episode:03d} | "
                f"Reward={episode_reward:9.4f} | "
                f"Avg10={mean_last(episode_rewards):9.4f} | "
                f"SE={metrics['se']:.4f} | "
                f"EE={metrics['ee']:.4e} | "
                f"Trust={final_trust:.4f} | "
                f"dTrust={trust_change:+.4f} | "
                f"Beh={metrics['behavior_score']:.4f} | "
                f"Queue={metrics['queue']:.4f} | "
                f"Eval={evaluation['per_step_reward']:.6f} | "
                f"Updates={updates}"
                f"{best_marker}"
            )

    # =====================================================
    # FINAL SAC EVALUATION
    # =====================================================

    print()

    print(
        "=" * 72
    )

    print(
        "Final SAC evaluation..."
    )

    final_eval = (
        evaluate_sac_policy(
            agent,
            env,
            episodes=FINAL_EVAL_EPISODES,
        )
    )

    # =====================================================
    # IMPROVEMENT
    # =====================================================

    random_step = (
        random_baseline[
            "per_step_reward"
        ]
    )

    sac_step = (
        final_eval[
            "per_step_reward"
        ]
    )

    improvement = (
        (
            sac_step
            - random_step
        )
        / max(
            abs(random_step),
            1e-12,
        )
        * 100.0
    )

    # =====================================================
    # FINAL REPORT
    # =====================================================

    print_evaluation(
        "RANDOM BASELINE",
        random_baseline,
    )

    print()

    print_evaluation(
        "SAC FINAL",
        final_eval,
    )

    print()

    print(
        f"SAC improvement over random: "
        f"{improvement:.2f}%"
    )

    # =====================================================
    # TRAINING STATISTICS
    # =====================================================

    first_10_avg = mean_first(
        episode_rewards,
        10,
    )

    last_10_avg = mean_last(
        episode_rewards,
        10,
    )

    best_10_avg = best_window_average(
        episode_rewards,
        10,
    )

    # =====================================================
    # SAVE TRAINING ARRAYS
    # =====================================================

    save_array(
        "phase3_training_episode_rewards.npy",
        episode_rewards,
    )

    save_array(
        "phase3_training_per_step_rewards.npy",
        per_step_rewards,
    )

    save_array(
        "phase3_training_se.npy",
        se_history,
    )

    save_array(
        "phase3_training_ee.npy",
        ee_history,
    )

    save_array(
        "phase3_training_total_rate.npy",
        total_rate_history,
    )

    save_array(
        "phase3_training_sinr.npy",
        sinr_history,
    )

    save_array(
        "phase3_training_trust.npy",
        trust_history,
    )

    save_array(
        "phase3_training_initial_trust.npy",
        initial_trust_history,
    )

    save_array(
        "phase3_training_trust_change.npy",
        trust_change_history,
    )

    save_array(
        "phase3_training_trust_delta.npy",
        trust_delta_history,
    )

    save_array(
        "phase3_training_behavior_score.npy",
        behavior_history,
    )

    save_array(
        "phase3_training_min_trust.npy",
        min_trust_history,
    )

    save_array(
        "phase3_training_max_trust.npy",
        max_trust_history,
    )

    save_array(
        "phase3_training_queue.npy",
        queue_history,
    )

    save_array(
        "phase3_training_power.npy",
        power_history,
    )

    save_array(
        "phase3_training_total_power.npy",
        total_power_history,
    )

    save_array(
        "phase3_training_interference.npy",
        interference_history,
    )

    # =====================================================
    # SAVE EVALUATION ARRAYS
    # =====================================================

    save_array(
        "phase3_evaluation_episodes.npy",
        evaluation_episodes,
    )

    save_array(
        "phase3_evaluation_rewards.npy",
        evaluation_rewards,
    )

    save_array(
        "phase3_evaluation_per_step.npy",
        evaluation_per_step,
    )

    save_array(
        "phase3_evaluation_se.npy",
        evaluation_se,
    )

    save_array(
        "phase3_evaluation_ee.npy",
        evaluation_ee,
    )

    save_array(
        "phase3_evaluation_trust.npy",
        evaluation_trust,
    )

    save_array(
        "phase3_evaluation_initial_trust.npy",
        evaluation_initial_trust,
    )

    save_array(
        "phase3_evaluation_trust_change.npy",
        evaluation_trust_change,
    )

    save_array(
        "phase3_evaluation_trust_delta.npy",
        evaluation_trust_delta,
    )

    save_array(
        "phase3_evaluation_behavior_score.npy",
        evaluation_behavior,
    )

    save_array(
        "phase3_evaluation_queue.npy",
        evaluation_queue,
    )

    save_array(
        "phase3_evaluation_power.npy",
        evaluation_power,
    )

    save_array(
        "phase3_evaluation_interference.npy",
        evaluation_interference,
    )

    # =====================================================
    # SAVE SUMMARY
    # =====================================================

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "TA-FDRL-IRF Phase-3 SAC Benchmark\n"
        )

        f.write(
            "====================================\n\n"
        )

        # -------------------------------------------------
        # Configuration
        # -------------------------------------------------

        f.write(
            "Configuration\n"
        )

        f.write(
            "-------------\n"
        )

        f.write(
            f"Seed: {SEED}\n"
        )

        f.write(
            f"State dimension: "
            f"{env.state_dim}\n"
        )

        f.write(
            f"Action dimension: "
            f"{env.action_dim}\n"
        )

        f.write(
            f"Number of users: "
            f"{env.num_users}\n"
        )

        f.write(
            f"RIS elements: "
            f"{env.num_ris}\n"
        )

        f.write(
            f"RIS optimization: "
            f"{env.cfg.optimize_ris}\n"
        )

        f.write(
            f"Fixed trust: "
            f"{env.cfg.fixed_trust}\n"
        )

        f.write(
            f"Trust memory: "
            f"{env.cfg.trust_memory}\n"
        )

        f.write(
            f"Trust learning rate: "
            f"{env.cfg.trust_learning_rate}\n"
        )

        f.write(
            f"Trust target rate (bps): "
            f"{env.cfg.trust_target_rate_bps}\n"
        )

        f.write(
            f"Trust neutral point: "
            f"{env.cfg.trust_neutral_point}\n"
        )

        f.write(
            f"Trust floor: "
            f"{env.cfg.trust_floor}\n"
        )

        f.write(
            f"SE target: "
            f"{env.cfg.se_target}\n"
        )

        f.write(
            f"EE target: "
            f"{env.cfg.ee_target}\n"
        )

        f.write(
            f"Training episodes: "
            f"{NUM_TRAIN_EPISODES}\n"
        )

        f.write(
            f"Steps per episode: "
            f"{STEPS_PER_EPISODE}\n"
        )

        f.write(
            f"Evaluation interval: "
            f"{EVAL_INTERVAL}\n"
        )

        f.write(
            f"Evaluation episodes: "
            f"{EVAL_EPISODES}\n"
        )

        f.write(
            f"Final evaluation episodes: "
            f"{FINAL_EVAL_EPISODES}\n\n"
        )

        # -------------------------------------------------
        # Random baseline
        # -------------------------------------------------

        f.write(
            "Random baseline\n"
        )

        f.write(
            "---------------\n"
        )

        f.write(
            f"Episode reward: "
            f"{random_baseline['episode_reward']}\n"
        )

        f.write(
            f"Per-step reward: "
            f"{random_baseline['per_step_reward']}\n"
        )

        f.write(
            f"SE: "
            f"{random_baseline['se']}\n"
        )

        f.write(
            f"EE: "
            f"{random_baseline['ee']}\n"
        )

        f.write(
            f"Initial trust: "
            f"{random_baseline['initial_trust']}\n"
        )

        f.write(
            f"Final trust: "
            f"{random_baseline['trust']}\n"
        )

        f.write(
            f"Trust change: "
            f"{random_baseline['trust_change']}\n"
        )

        f.write(
            f"Trust delta: "
            f"{random_baseline['trust_delta']}\n"
        )

        f.write(
            f"Behavior score: "
            f"{random_baseline['behavior_score']}\n"
        )

        f.write(
            f"Queue: "
            f"{random_baseline['queue']}\n"
        )

        f.write(
            f"Power: "
            f"{random_baseline['power']}\n"
        )

        f.write(
            f"Interference: "
            f"{random_baseline['interference']}\n\n"
        )

        # -------------------------------------------------
        # Final SAC
        # -------------------------------------------------

        f.write(
            "Final SAC\n"
        )

        f.write(
            "---------\n"
        )

        f.write(
            f"Episode reward: "
            f"{final_eval['episode_reward']}\n"
        )

        f.write(
            f"Per-step reward: "
            f"{final_eval['per_step_reward']}\n"
        )

        f.write(
            f"SE: "
            f"{final_eval['se']}\n"
        )

        f.write(
            f"EE: "
            f"{final_eval['ee']}\n"
        )

        f.write(
            f"Initial trust: "
            f"{final_eval['initial_trust']}\n"
        )

        f.write(
            f"Final trust: "
            f"{final_eval['trust']}\n"
        )

        f.write(
            f"Trust change: "
            f"{final_eval['trust_change']}\n"
        )

        f.write(
            f"Trust delta: "
            f"{final_eval['trust_delta']}\n"
        )

        f.write(
            f"Behavior score: "
            f"{final_eval['behavior_score']}\n"
        )

        f.write(
            f"Minimum trust: "
            f"{final_eval['min_trust']}\n"
        )

        f.write(
            f"Maximum trust: "
            f"{final_eval['max_trust']}\n"
        )

        f.write(
            f"Queue: "
            f"{final_eval['queue']}\n"
        )

        f.write(
            f"Power: "
            f"{final_eval['power']}\n"
        )

        f.write(
            f"Interference: "
            f"{final_eval['interference']}\n\n"
        )

        # -------------------------------------------------
        # Comparison
        # -------------------------------------------------

        f.write(
            "Comparison\n"
        )

        f.write(
            "----------\n"
        )

        f.write(
            f"SAC improvement over random (%): "
            f"{improvement}\n\n"
        )

        # -------------------------------------------------
        # Training statistics
        # -------------------------------------------------

        f.write(
            "Training statistics\n"
        )

        f.write(
            "-------------------\n"
        )

        f.write(
            f"First-10 average reward: "
            f"{first_10_avg}\n"
        )

        f.write(
            f"Last-10 average reward: "
            f"{last_10_avg}\n"
        )

        f.write(
            f"Best-10 average reward: "
            f"{best_10_avg}\n"
        )

        f.write(
            f"Best evaluation per-step reward: "
            f"{best_eval_reward}\n"
        )

        f.write(
            f"Best evaluation episode: "
            f"{best_eval_episode}\n\n"
        )

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        f.write(
            "Model\n"
        )

        f.write(
            "-----\n"
        )

        f.write(
            f"Best model path: "
            f"{BEST_MODEL_PATH}\n"
        )

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print()

    print(
        "=" * 72
    )

    print(
        "Training complete."
    )

    print()

    print(
        f"First-10 average reward: "
        f"{first_10_avg:.6f}"
    )

    print(
        f"Last-10 average reward: "
        f"{last_10_avg:.6f}"
    )

    print(
        f"Best-10 average reward: "
        f"{best_10_avg:.6f}"
    )

    print()

    print(
        f"Final SAC trust: "
        f"{final_eval['trust']:.6f}"
    )

    print(
        f"Final SAC trust change: "
        f"{final_eval['trust_change']:+.6f}"
    )

    print(
        f"Final SAC trust delta: "
        f"{final_eval['trust_delta']:+.8f}"
    )

    print(
        f"Final behavior score: "
        f"{final_eval['behavior_score']:.6f}"
    )

    print()

    print(
        f"SAC improvement over random: "
        f"{improvement:.2f}%"
    )

    print()

    print(
        "Best model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print()

    print(
        "Summary:"
    )

    print(
        SUMMARY_PATH
    )

    print()

    print(
        "Results directory:"
    )

    print(
        RESULTS_DIR
    )

    print(
        "=" * 72
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()

