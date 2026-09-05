from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


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
    # Phase control
    # =====================================================

    # Phase 2:
    # False = fixed RIS
    # True  = RIS becomes part of action
    optimize_ris: bool = False

    # Phase 2:
    # True = trust fixed at 1.0
    #
    # Later:
    # False = dynamic trust
    fixed_trust: bool = True

    # =====================================================
    # Reward
    # =====================================================

    se_weight: float = 0.55
    ee_weight: float = 0.25
    trust_weight: float = 0.10

    interference_penalty: float = 0.05
    power_penalty: float = 0.03
    queue_penalty: float = 0.02

    # =====================================================
    # Random seed
    # =====================================================

    seed: int = 42


class IRFEnvironment:

    """
    Trust-Aware Adaptive Federated Deep Reinforcement
    Learning for Intelligent Radio Fabric in 6G Networks.

    Phase-2 SAC baseline.

    State:
        5 features per user

        [SINR,
         interference,
         queue,
         power,
         trust]

        20 users * 5 = 100

    Phase-2 Action:

        [bandwidth allocation x 20,
         power allocation x 20]

        = 40 dimensions

    Phase-2 RIS:
        Fixed.

    Phase-2 Trust:
        Fixed at 1.0.
    """

    def __init__(
        self,
        config: IRFConfig | None = None,
    ):

        self.cfg = (
            config
            if config is not None
            else IRFConfig()
        )

        self.num_users = (
            self.cfg.num_users
        )

        self.num_ris = (
            self.cfg.num_ris_elements
        )

        self.rng = np.random.default_rng(
            self.cfg.seed
        )

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

    # =====================================================
    # RESET
    # =====================================================

    def reset(
        self,
        seed: int | None = None,
    ) -> np.ndarray:

        if seed is not None:

            self.rng = np.random.default_rng(
                seed
            )

        self.step_count = 0

        # -------------------------------------------------
        # New channel realization
        # -------------------------------------------------

        self._generate_channels()

        # -------------------------------------------------
        # Initial state
        # -------------------------------------------------

        self.sinr = np.ones(
            self.num_users,
            dtype=np.float64,
        )

        self.interference = np.zeros(
            self.num_users,
            dtype=np.float64,
        )

        self.queue = self.rng.uniform(
            0.10,
            0.30,
            size=self.num_users,
        )

        self.power = np.full(
            self.num_users,
            0.50,
            dtype=np.float64,
        )

        # -------------------------------------------------
        # Fixed trust baseline
        # -------------------------------------------------

        self.trust = np.ones(
            self.num_users,
            dtype=np.float64,
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

        action = np.asarray(
            action,
            dtype=np.float64,
        ).reshape(-1)

        # -------------------------------------------------
        # Validate action
        # -------------------------------------------------

        if action.size != self.action_dim:

            raise ValueError(
                f"Invalid action dimension. "
                f"Expected {self.action_dim}, "
                f"received {action.size}."
            )

        # =================================================
        # ACTION: BANDWIDTH
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

        bandwidth_alloc = (
            bandwidth_weights
            / bandwidth_weights.sum()
        )

        user_bandwidth = (
            bandwidth_alloc
            * self.cfg.bandwidth_hz
        )

        # =================================================
        # ACTION: POWER
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
        # ACTION: RIS
        # =================================================

        if self.cfg.optimize_ris:

            ris_raw = action[
                2 * self.num_users:
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

            # ---------------------------------------------
            # Fixed RIS
            # ---------------------------------------------

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

        total_received_power = (
            np.sum(received_power)
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
            np.sum(rates)
        )

        # =================================================
        # SPECTRAL EFFICIENCY
        # =================================================

        spectral_efficiency = (
            total_rate
            / self.cfg.bandwidth_hz
        )

        # =================================================
        # ENERGY EFFICIENCY
        # =================================================

        total_power = (
            float(np.sum(power_alloc))
            + self.cfg.circuit_power_w
        )

        energy_efficiency = (
            total_rate
            / max(
                total_power,
                1e-12,
            )
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
        # TRUST
        # =================================================

        if self.cfg.fixed_trust:

            self.trust = np.ones(
                self.num_users,
                dtype=np.float64,
            )

        else:

            behavior_quality = (
                1.0
                - self.queue
            )

            self.trust = np.clip(
                0.95 * self.trust
                + 0.05 * behavior_quality,
                0.0,
                1.0,
            )

        mean_trust = float(
            np.mean(self.trust)
        )

        # =================================================
        # REWARD NORMALIZATION
        # =================================================

        # -------------------------------------------------
        # SE
        #
        # Current environment typically produces SE around
        # 0.01-0.2.
        # -------------------------------------------------

        se_norm = np.clip(
            spectral_efficiency / 0.20,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # EE
        #
        # Current environment typically produces EE around
        # 1e5 - 1e6.
        # -------------------------------------------------

        ee_norm = np.clip(
            energy_efficiency / 1e6,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Interference ratio
        # -------------------------------------------------

        signal_reference = (
            float(
                np.mean(
                    received_power
                )
            )
            + 1e-12
        )

        interference_ratio = (
            float(
                np.mean(
                    self.interference
                )
            )
            / signal_reference
        )

        interference_norm = np.clip(
            interference_ratio,
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
        # FINAL REWARD
        # =================================================

        reward = (

            self.cfg.se_weight
            * se_norm

            + self.cfg.ee_weight
            * ee_norm

            + self.cfg.trust_weight
            * mean_trust

            - self.cfg.interference_penalty
            * interference_norm

            - self.cfg.power_penalty
            * power_norm

            - self.cfg.queue_penalty
            * queue_norm
        )

        # -------------------------------------------------
        # Hard numerical safety
        # -------------------------------------------------

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

            "mean_trust":
                float(
                    mean_trust
                ),

            "mean_power":
                float(
                    np.mean(
                        power_alloc
                    )
                ),

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

            "reward":
                reward,
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

        state = np.concatenate(
            [
                sinr_state,
                interference_state,
                queue_state,
                power_state,
                trust_state,
            ]
        )

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
    # DIMENSIONS
    # =====================================================

    @property
    def state_dim(self) -> int:

        return (
            self.num_users * 5
        )

    @property
    def action_dim(self) -> int:

        base_action_dim = (
            self.num_users * 2
        )

        if self.cfg.optimize_ris:

            return (
                base_action_dim
                + self.num_ris
            )

        return base_action_dim