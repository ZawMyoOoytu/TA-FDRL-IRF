
# =========================================================
# irf_env.py
#
# TA-FDRL-IRF
# Trust-Aware Adaptive Federated Deep Reinforcement
# Learning for Intelligent Radio Fabric in 6G Networks
#
# Phase-3:
# SAC + Adaptive Trust
#
# Fixed RIS
# Dynamic / Adaptive Trust
#
# Corrected Phase-3 Environment
#
# Main improvements:
#   1. Bidirectional adaptive trust update
#   2. Explicit trust-delta reward
#   3. Configurable target service rate
#   4. Trust behavior score tracking
#   5. Trust trajectory tracking
#   6. Queue/interference/stability tracking
#   7. Numerical safety
#   8. Phase-2-compatible interface
# =========================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# =========================================================
# CONFIGURATION
# =========================================================

@dataclass
class IRFConfig:

    # =====================================================
    # Network
    # =====================================================

    num_users: int = 20

    num_ris_elements: int = 64

    bandwidth_hz: float = 100e6

    carrier_frequency_hz: float = 28e9

    # =====================================================
    # Power
    # =====================================================

    max_power_w: float = 1.0

    circuit_power_w: float = 0.1

    # =====================================================
    # Noise
    # =====================================================

    noise_figure_db: float = 7.0

    noise_density_dbm_hz: float = -174.0

    # =====================================================
    # Episode
    # =====================================================

    max_steps: int = 200

    # =====================================================
    # Phase Control
    # =====================================================

    # False = fixed RIS
    # True  = RIS optimization enabled
    optimize_ris: bool = False

    # True  = trust remains at 1.0
    # False = adaptive trust
    fixed_trust: bool = False

    # =====================================================
    # Adaptive Trust
    # =====================================================

    # Historical trust contribution.
    #
    # Larger value:
    #   slower trust adaptation
    #
    # Smaller value:
    #   faster trust adaptation
    trust_memory: float = 0.90

    # Trust learning rate.
    trust_learning_rate: float = 0.10

    # Target service rate per user.
    #
    # 1e7 = 10 Mbps.
    #
    # Service quality:
    #
    #     rate / target_rate
    #
    trust_target_rate_bps: float = 1e7

    # -----------------------------------------------------
    # Trust behavior weights
    # -----------------------------------------------------

    trust_service_weight: float = 0.35

    trust_interference_weight: float = 0.20

    trust_queue_weight: float = 0.25

    trust_instability_weight: float = 0.20

    # -----------------------------------------------------
    # Trust reference point
    #
    # behavior > 0.5:
    #       positive trust movement
    #
    # behavior < 0.5:
    #       negative trust movement
    #
    # behavior = 0.5:
    #       approximately stable
    # -----------------------------------------------------

    trust_neutral_point: float = 0.50

    # Optional trust floor.
    #
    # 0.0 means no artificial lower bound.
    # This is kept as a configuration parameter so
    # experiments can explicitly define a minimum trust.
    trust_floor: float = 0.0

    # =====================================================
    # Reward
    # =====================================================

    se_weight: float = 0.45

    ee_weight: float = 0.20

    trust_weight: float = 0.15

    # NEW:
    # Direct reward for improving trust.
    trust_delta_weight: float = 0.10

    interference_penalty: float = 0.05

    power_penalty: float = 0.03

    queue_penalty: float = 0.07

    # =====================================================
    # Reward normalization
    # =====================================================

    # SE target.
    #
    # SE normalized as:
    #
    #     SE / se_target
    #
    se_target: float = 0.20

    # EE target.
    #
    # EE normalized as:
    #
    #     EE / ee_target
    #
    ee_target: float = 1e6

    # =====================================================
    # Random Seed
    # =====================================================

    seed: int = 42


# =========================================================
# ENVIRONMENT
# =========================================================

class IRFEnvironment:
    """
    Trust-Aware Adaptive Federated Deep Reinforcement
    Learning for Intelligent Radio Fabric in 6G Networks.

    Phase-3 environment:

        SAC
        +
        Adaptive Trust
        +
        Fixed RIS

    -----------------------------------------------------
    State
    -----------------------------------------------------

    Five features per user:

        1. SINR
        2. Interference
        3. Queue
        4. Power
        5. Trust

    For N users:

        state_dim = 5N

    -----------------------------------------------------
    Action
    -----------------------------------------------------

    Base action:

        bandwidth allocation: N
        power allocation:     N

    Total:

        2N

    If RIS optimization is enabled:

        + num_ris_elements

    -----------------------------------------------------
    Trust
    -----------------------------------------------------

    Trust is dynamically updated from:

        - service quality
        - interference quality
        - queue quality
        - temporal stability

    Unlike the original Phase-3 formulation, the
    trust update is centered around a neutral point.

        behavior_score > neutral point
            -> trust increases

        behavior_score < neutral point
            -> trust decreases

        behavior_score ~= neutral point
            -> trust remains approximately stable

    -----------------------------------------------------
    Reward
    -----------------------------------------------------

        R =
            w_SE * SE
          + w_EE * EE
          + w_T  * Trust
          + w_dT * DeltaTrust
          - w_I * Interference
          - w_P * Power
          - w_Q * Queue

    This explicitly encourages the SAC agent to improve
    both communication performance and trust.
    """

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        config: IRFConfig | None = None,
    ):

        self.cfg = (
            config
            if config is not None
            else IRFConfig()
        )

        self.num_users = self.cfg.num_users

        self.num_ris = self.cfg.num_ris_elements

        # -------------------------------------------------
        # Random generator
        # -------------------------------------------------

        self.rng = np.random.default_rng(
            self.cfg.seed
        )

        # -------------------------------------------------
        # Episode state
        # -------------------------------------------------

        self.step_count = 0

        # -------------------------------------------------
        # Noise
        # -------------------------------------------------

        self.noise_power_w = (
            self._calculate_noise_power()
        )

        # -------------------------------------------------
        # Channel variables
        # -------------------------------------------------

        self.direct_channel = None

        self.bs_ris_channel = None

        self.ris_user_channel = None

        # -------------------------------------------------
        # State variables
        # -------------------------------------------------

        self.sinr = None

        self.interference = None

        self.queue = None

        self.power = None

        self.trust = None

        # -------------------------------------------------
        # Previous variables
        # -------------------------------------------------

        self.previous_queue = None

        self.previous_sinr = None

        self.previous_interference = None

        self.previous_trust = None

        # -------------------------------------------------
        # Current trust diagnostics
        # -------------------------------------------------

        self.behavior_score = None

        self.trust_delta = None

        self.mean_trust = 0.0

    # =====================================================
    # RESET
    # =====================================================

    def reset(
        self,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Reset the environment.

        A new channel realization and new adaptive-trust
        state are generated.
        """

        if seed is not None:

            self.rng = np.random.default_rng(
                seed
            )

        self.step_count = 0

        # -------------------------------------------------
        # Generate channel
        # -------------------------------------------------

        self._generate_channels()

        # -------------------------------------------------
        # Initial SINR
        # -------------------------------------------------

        self.sinr = np.ones(
            self.num_users,
            dtype=np.float64,
        )

        # -------------------------------------------------
        # Initial interference
        # -------------------------------------------------

        self.interference = np.zeros(
            self.num_users,
            dtype=np.float64,
        )

        # -------------------------------------------------
        # Initial queues
        # -------------------------------------------------

        self.queue = self.rng.uniform(
            0.10,
            0.30,
            size=self.num_users,
        )

        # -------------------------------------------------
        # Initial power
        # -------------------------------------------------

        self.power = np.full(
            self.num_users,
            0.50,
            dtype=np.float64,
        )

        # -------------------------------------------------
        # Initial trust
        # -------------------------------------------------

        if self.cfg.fixed_trust:

            self.trust = np.ones(
                self.num_users,
                dtype=np.float64,
            )

        else:

            # Heterogeneous initial trust.
            self.trust = self.rng.uniform(
                0.75,
                0.95,
                size=self.num_users,
            )

        # -------------------------------------------------
        # Historical variables
        # -------------------------------------------------

        self.previous_queue = (
            self.queue.copy()
        )

        self.previous_sinr = (
            self.sinr.copy()
        )

        self.previous_interference = (
            self.interference.copy()
        )

        self.previous_trust = (
            self.trust.copy()
        )

        # -------------------------------------------------
        # Trust diagnostics
        # -------------------------------------------------

        self.behavior_score = np.full(
            self.num_users,
            self.cfg.trust_neutral_point,
            dtype=np.float64,
        )

        self.trust_delta = np.zeros(
            self.num_users,
            dtype=np.float64,
        )

        self.mean_trust = float(
            np.mean(self.trust)
        )

        return self._get_state()

    # =====================================================
    # CHANNEL GENERATION
    # =====================================================

    def _generate_channels(self):

        # -------------------------------------------------
        # Direct BS-user channel
        # -------------------------------------------------

        self.direct_channel = (
            self.rng.normal(
                size=self.num_users
            )
            + 1j
            * self.rng.normal(
                size=self.num_users
            )
        ) / np.sqrt(2.0)

        # -------------------------------------------------
        # BS-RIS channel
        # -------------------------------------------------

        self.bs_ris_channel = (
            self.rng.normal(
                size=self.num_ris
            )
            + 1j
            * self.rng.normal(
                size=self.num_ris
            )
        ) / np.sqrt(2.0)

        # -------------------------------------------------
        # RIS-user channel
        # -------------------------------------------------

        self.ris_user_channel = (
            self.rng.normal(
                size=(
                    self.num_users,
                    self.num_ris,
                )
            )
            + 1j
            * self.rng.normal(
                size=(
                    self.num_users,
                    self.num_ris,
                )
            )
        ) / np.sqrt(2.0)

    # =====================================================
    # EFFECTIVE CHANNEL
    # =====================================================

    def _effective_channel(
        self,
        ris_phase: np.ndarray,
    ) -> np.ndarray:

        phi = np.exp(
            1j * ris_phase
        )

        reflected = np.sum(
            self.ris_user_channel
            * phi[None, :]
            * self.bs_ris_channel[None, :],
            axis=1,
        )

        h_eff = (
            self.direct_channel
            + reflected
        )

        return h_eff

    # =====================================================
    # NOISE
    # =====================================================

    def _calculate_noise_power(
        self,
    ) -> float:

        noise_dbm = (
            self.cfg.noise_density_dbm_hz
            + 10.0
            * np.log10(
                self.cfg.bandwidth_hz
            )
            + self.cfg.noise_figure_db
        )

        return 10.0 ** (
            (noise_dbm - 30.0)
            / 10.0
        )

    # =====================================================
    # SERVICE QUALITY
    # =====================================================

    def _calculate_service_quality(
        self,
        rates: np.ndarray,
    ) -> np.ndarray:
        """
        Convert user rate into normalized service quality.

        target = configurable target rate per user.

        Example:

            rate = 10 Mbps
            target = 10 Mbps
            quality = 1.0
        """

        target = max(
            self.cfg.trust_target_rate_bps,
            1e-12,
        )

        service_quality = np.clip(
            rates / target,
            0.0,
            1.0,
        )

        return service_quality

    # =====================================================
    # ADAPTIVE TRUST
    # =====================================================

    def _update_trust(
        self,
        rates: np.ndarray,
        interference_ratio: float,
    ) -> None:
        """
        Update adaptive trust.

        Trust is based on four components:

            service quality
            interference quality
            queue quality
            stability quality

        A neutral point is introduced.

        behavior_score > neutral:
            positive trust movement

        behavior_score < neutral:
            negative trust movement

        This prevents the trust state from automatically
        drifting toward a low behavior-score equilibrium.
        """

        # -------------------------------------------------
        # Fixed trust mode
        # -------------------------------------------------

        if self.cfg.fixed_trust:

            self.trust = np.ones(
                self.num_users,
                dtype=np.float64,
            )

            self.behavior_score = np.ones(
                self.num_users,
                dtype=np.float64,
            )

            self.trust_delta = np.zeros(
                self.num_users,
                dtype=np.float64,
            )

            self.mean_trust = 1.0

            return

        # -------------------------------------------------
        # 1. Service quality
        # -------------------------------------------------

        service_quality = (
            self._calculate_service_quality(
                rates
            )
        )

        # -------------------------------------------------
        # 2. Queue quality
        # -------------------------------------------------

        queue_quality = np.clip(
            1.0 - self.queue,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # 3. Interference quality
        # -------------------------------------------------

        interference_quality = np.clip(
            1.0 - interference_ratio,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # 4. Temporal stability
        # -------------------------------------------------

        queue_change = np.abs(
            self.queue
            - self.previous_queue
        )

        stability_quality = np.clip(
            1.0 - queue_change,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Composite behavior score
        # -------------------------------------------------

        behavior_score = (

            self.cfg.trust_service_weight
            * service_quality

            + self.cfg.trust_interference_weight
            * interference_quality

            + self.cfg.trust_queue_weight
            * queue_quality

            + self.cfg.trust_instability_weight
            * stability_quality
        )

        behavior_score = np.clip(
            behavior_score,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Save behavior score
        # -------------------------------------------------

        self.behavior_score = (
            behavior_score.copy()
        )

        # -------------------------------------------------
        # Trust movement
        # -------------------------------------------------

        neutral = float(
            self.cfg.trust_neutral_point
        )

        centered_behavior = (
            behavior_score
            - neutral
        )

        # -------------------------------------------------
        # Learning-rate trust movement
        # -------------------------------------------------

        raw_delta = (
            self.cfg.trust_learning_rate
            * centered_behavior
        )

        # -------------------------------------------------
        # Historical smoothing
        #
        # Equivalent to:
        #
        # trust_new =
        #     memory * old
        #     +
        #     learning * target
        #
        # but centered around current trust.
        # -------------------------------------------------

        old_trust = self.trust.copy()

        self.trust = (
            old_trust
            + (
                1.0
                - self.cfg.trust_memory
            )
            * raw_delta
        )

        # -------------------------------------------------
        # Trust delta
        # -------------------------------------------------

        self.trust_delta = (
            self.trust
            - old_trust
        )

        # -------------------------------------------------
        # Numerical safety
        # -------------------------------------------------

        floor = float(
            np.clip(
                self.cfg.trust_floor,
                0.0,
                1.0,
            )
        )

        self.trust = np.clip(
            self.trust,
            floor,
            1.0,
        )

        # -------------------------------------------------
        # Recalculate actual delta after clipping
        # -------------------------------------------------

        self.trust_delta = (
            self.trust
            - old_trust
        )

        # -------------------------------------------------
        # Mean trust
        # -------------------------------------------------

        self.mean_trust = float(
            np.mean(
                self.trust
            )
        )

    # =====================================================
    # STEP
    # =====================================================

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        float,
        bool,
        dict,
    ]:
        """
        Execute one environment step.
        """

        action = np.asarray(
            action,
            dtype=np.float64,
        ).reshape(-1)

        # =================================================
        # Validate action
        # =================================================

        if action.size != self.action_dim:

            raise ValueError(
                f"Invalid action dimension. "
                f"Expected {self.action_dim}, "
                f"received {action.size}."
            )

        # =================================================
        # BANDWIDTH ACTION
        # =================================================

        bandwidth_raw = action[
            :self.num_users
        ]

        bandwidth_weights = (
            bandwidth_raw + 1.0
        )

        bandwidth_weights = np.maximum(
            bandwidth_weights,
            1e-6,
        )

        bandwidth_sum = float(
            np.sum(
                bandwidth_weights
            )
        )

        if bandwidth_sum <= 0.0:

            bandwidth_weights = np.ones(
                self.num_users,
                dtype=np.float64,
            )

            bandwidth_sum = float(
                self.num_users
            )

        bandwidth_alloc = (
            bandwidth_weights
            / bandwidth_sum
        )

        user_bandwidth = (
            bandwidth_alloc
            * self.cfg.bandwidth_hz
        )

        # =================================================
        # POWER ACTION
        # =================================================

        power_raw = action[
            self.num_users:
            2 * self.num_users
        ]

        power_alloc = (
            np.clip(
                power_raw,
                -1.0,
                1.0,
            )
            + 1.0
        ) / 2.0

        power_alloc *= (
            self.cfg.max_power_w
        )

        power_alloc = np.clip(
            power_alloc,
            0.0,
            self.cfg.max_power_w,
        )

        self.power = (
            power_alloc.copy()
        )

        # =================================================
        # RIS ACTION
        # =================================================

        if self.cfg.optimize_ris:

            ris_start = (
                2 * self.num_users
            )

            ris_end = (
                ris_start
                + self.num_ris
            )

            ris_raw = action[
                ris_start:ris_end
            ]

            ris_phase = (
                np.clip(
                    ris_raw,
                    -1.0,
                    1.0,
                )
                * np.pi
            )

        else:

            # Fixed RIS configuration.
            ris_phase = np.zeros(
                self.num_ris,
                dtype=np.float64,
            )

        # =================================================
        # CHANNEL
        # =================================================

        h_eff = self._effective_channel(
            ris_phase
        )

        channel_gain = (
            np.abs(h_eff) ** 2
        )

        channel_gain = np.maximum(
            channel_gain,
            1e-12,
        )

        # =================================================
        # RECEIVED POWER
        # =================================================

        received_power = (
            power_alloc
            * channel_gain
        )

        received_power = np.maximum(
            received_power,
            0.0,
        )

        # =================================================
        # INTERFERENCE
        # =================================================

        total_received_power = float(
            np.sum(
                received_power
            )
        )

        self.interference = np.maximum(
            total_received_power
            - received_power,
            0.0,
        )

        # =================================================
        # SINR
        # =================================================

        denominator = (
            self.interference
            + self.noise_power_w
            + 1e-12
        )

        self.sinr = (
            received_power
            / denominator
        )

        self.sinr = np.nan_to_num(
            self.sinr,
            nan=0.0,
            posinf=1e6,
            neginf=0.0,
        )

        # =================================================
        # RATE
        # =================================================

        rates = (
            user_bandwidth
            * np.log2(
                1.0 + self.sinr
            )
        )

        rates = np.nan_to_num(
            rates,
            nan=0.0,
            posinf=1e12,
            neginf=0.0,
        )

        total_rate = float(
            np.sum(
                rates
            )
        )

        # =================================================
        # SPECTRAL EFFICIENCY
        # =================================================

        spectral_efficiency = (
            total_rate
            / max(
                self.cfg.bandwidth_hz,
                1e-12,
            )
        )

        # =================================================
        # TOTAL POWER
        # =================================================

        total_power = (
            float(
                np.sum(
                    power_alloc
                )
            )
            + self.cfg.circuit_power_w
        )

        # =================================================
        # ENERGY EFFICIENCY
        # =================================================

        energy_efficiency = (
            total_rate
            / max(
                total_power,
                1e-12,
            )
        )

        # =================================================
        # INTERFERENCE RATIO
        # =================================================

        signal_reference = (
            float(
                np.mean(
                    received_power
                )
            )
            + 1e-12
        )

        interference_reference = (
            float(
                np.mean(
                    self.interference
                )
            )
        )

        interference_ratio = (
            interference_reference
            / signal_reference
        )

        interference_ratio = float(
            np.clip(
                interference_ratio,
                0.0,
                1.0,
            )
        )

        # =================================================
        # SAVE PREVIOUS VARIABLES
        # =================================================

        self.previous_queue = (
            self.queue.copy()
        )

        self.previous_sinr = (
            self.sinr.copy()
        )

        self.previous_interference = (
            self.interference.copy()
        )

        self.previous_trust = (
            self.trust.copy()
        )

        # =================================================
        # QUEUE DYNAMICS
        # =================================================

        arrivals = self.rng.uniform(
            0.0,
            0.005,
            size=self.num_users,
        )

        service = np.clip(
            rates / 1e8,
            0.0,
            0.05,
        )

        self.queue = np.clip(
            self.queue
            + arrivals
            - service,
            0.0,
            1.0,
        )

        # =================================================
        # ADAPTIVE TRUST
        # =================================================

        self._update_trust(
            rates=rates,
            interference_ratio=interference_ratio,
        )

        # =================================================
        # METRICS
        # =================================================

        mean_trust = float(
            np.mean(
                self.trust
            )
        )

        mean_trust_delta = float(
            np.mean(
                self.trust_delta
            )
        )

        mean_behavior_score = float(
            np.mean(
                self.behavior_score
            )
        )

        # =================================================
        # REWARD NORMALIZATION
        # =================================================

        # -------------------------------------------------
        # Spectral efficiency
        # -------------------------------------------------

        se_norm = np.clip(
            spectral_efficiency
            / max(
                self.cfg.se_target,
                1e-12,
            ),
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Energy efficiency
        # -------------------------------------------------

        ee_norm = np.clip(
            energy_efficiency
            / max(
                self.cfg.ee_target,
                1e-12,
            ),
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Power
        # -------------------------------------------------

        power_norm = np.clip(
            float(
                np.mean(
                    power_alloc
                )
            )
            / max(
                self.cfg.max_power_w,
                1e-12,
            ),
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Queue
        # -------------------------------------------------

        queue_norm = np.clip(
            float(
                np.mean(
                    self.queue
                )
            ),
            0.0,
            1.0,
        )

        # =================================================
        # TRUST-AWARE REWARD
        # =================================================

        reward = (

            # Communication performance
            self.cfg.se_weight
            * se_norm

            # Energy performance
            + self.cfg.ee_weight
            * ee_norm

            # Absolute trust
            + self.cfg.trust_weight
            * mean_trust

            # Trust improvement
            + self.cfg.trust_delta_weight
            * mean_trust_delta

            # Interference penalty
            - self.cfg.interference_penalty
            * interference_ratio

            # Power penalty
            - self.cfg.power_penalty
            * power_norm

            # Queue penalty
            - self.cfg.queue_penalty
            * queue_norm
        )

        # =================================================
        # Numerical safety
        # =================================================

        reward = float(
            np.nan_to_num(
                reward,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
        )

        reward = float(
            np.clip(
                reward,
                -1.0,
                1.0,
            )
        )

        # =================================================
        # STEP COUNTER
        # =================================================

        self.step_count += 1

        terminated = (
            self.step_count
            >= self.cfg.max_steps
        )

        # =================================================
        # STATE
        # =================================================

        state = self._get_state()

        # =================================================
        # INFO
        # =================================================

        info = {

            # -------------------------------------------------
            # Communication
            # -------------------------------------------------

            "spectral_efficiency":
                float(
                    spectral_efficiency
                ),

            "energy_efficiency":
                float(
                    energy_efficiency
                ),

            "total_rate":
                float(
                    total_rate
                ),

            "mean_sinr":
                float(
                    np.mean(
                        self.sinr
                    )
                ),

            # -------------------------------------------------
            # Trust
            # -------------------------------------------------

            "mean_trust":
                float(
                    mean_trust
                ),

            "min_trust":
                float(
                    np.min(
                        self.trust
                    )
                ),

            "max_trust":
                float(
                    np.max(
                        self.trust
                    )
                ),

            "trust_delta":
                float(
                    mean_trust_delta
                ),

            "mean_behavior_score":
                float(
                    mean_behavior_score
                ),

            # -------------------------------------------------
            # Resource
            # -------------------------------------------------

            "mean_power":
                float(
                    np.mean(
                        power_alloc
                    )
                ),

            "total_power":
                float(
                    total_power
                ),

            # -------------------------------------------------
            # Queue / interference
            # -------------------------------------------------

            "mean_queue":
                float(
                    np.mean(
                        self.queue
                    )
                ),

            "interference_ratio":
                float(
                    interference_ratio
                ),

            # -------------------------------------------------
            # Reward
            # -------------------------------------------------

            "se_normalized":
                float(
                    se_norm
                ),

            "ee_normalized":
                float(
                    ee_norm
                ),

            "queue_normalized":
                float(
                    queue_norm
                ),

            "power_normalized":
                float(
                    power_norm
                ),

            "reward":
                float(
                    reward
                ),
        }

        return (
            state,
            reward,
            terminated,
            info,
        )

    # =====================================================
    # STATE
    # =====================================================

    def _get_state(
        self,
    ) -> np.ndarray:
        """
        Construct normalized state vector.
        """

        # -------------------------------------------------
        # SINR
        # -------------------------------------------------

        sinr_state = np.tanh(
            np.log1p(
                np.maximum(
                    self.sinr,
                    0.0,
                )
            )
        )

        # -------------------------------------------------
        # Interference
        # -------------------------------------------------

        interference_state = np.tanh(
            np.log1p(
                np.maximum(
                    self.interference,
                    0.0,
                )
            )
        )

        # -------------------------------------------------
        # Queue
        # -------------------------------------------------

        queue_state = np.clip(
            self.queue,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Power
        # -------------------------------------------------

        power_state = np.clip(
            self.power
            / max(
                self.cfg.max_power_w,
                1e-12,
            ),
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Trust
        # -------------------------------------------------

        trust_state = np.clip(
            self.trust,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # State vector
        # -------------------------------------------------

        state = np.concatenate(
            [
                sinr_state,
                interference_state,
                queue_state,
                power_state,
                trust_state,
            ]
        )

        # -------------------------------------------------
        # Numerical safety
        # -------------------------------------------------

        state = np.nan_to_num(
            state,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        return state.astype(
            np.float32
        )

    # =====================================================
    # STATE DIMENSION
    # =====================================================

    @property
    def state_dim(
        self,
    ) -> int:

        return (
            self.num_users
            * 5
        )

    # =====================================================
    # ACTION DIMENSION
    # =====================================================

    @property
    def action_dim(
        self,
    ) -> int:

        base_action_dim = (
            self.num_users
            * 2
        )

        if self.cfg.optimize_ris:

            return (
                base_action_dim
                + self.num_ris
            )

        return base_action_dim

