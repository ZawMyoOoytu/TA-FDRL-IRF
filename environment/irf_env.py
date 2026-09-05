# =========================================================
# irf_env.py
#
# TA-FDRL-IRF
# Trust-Aware Adaptive Federated Deep Reinforcement
# Learning for Intelligent Radio Fabric in 6G Networks
#
# Phase-4A:
# Polarization-Aware PHY
#
# SAC architecture:
#     UNCHANGED
#
# State:
#     100
#
# Action:
#     40
#
# RIS:
#     Fixed
#
# Trust:
#     Adaptive
#
# Main Phase-4A addition:
#
#     H_p =
#     [ h_VV  h_VH ]
#     [ h_HV  h_HH ]
#
#     Polarization-aware effective channel
#
# Research-grade RNG architecture:
#
#     Channel RNG
#         -> scalar channels
#         -> polarization channels
#
#     Dynamics RNG
#         -> initial queue
#         -> initial trust
#         -> queue arrivals
#
# This prevents polarization-specific random channel generation
# from changing the stochastic environment dynamics.
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
    # Polarization — Phase 4A
    # =====================================================

    polarization_enabled: bool = True

    # Cross-polarization coupling.
    #
    # 0.0:
    #     no cross-polarization coupling
    #
    # 1.0:
    #     strong cross-polarization coupling
    #
    # Phase-4A baseline:
    #     0.15
    #

    cross_polarization_factor: float = 0.15

    # Relative V/H polarization strengths.

    polarization_v_strength: float = 1.0

    polarization_h_strength: float = 1.0

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

    trust_memory: float = 0.90

    trust_learning_rate: float = 0.10

    # Target service rate per user.
    #
    # 1e7 = 10 Mbps.

    trust_target_rate_bps: float = 1e7

    # =====================================================
    # Trust behavior weights
    # =====================================================

    trust_service_weight: float = 0.35

    trust_interference_weight: float = 0.20

    trust_queue_weight: float = 0.25

    trust_instability_weight: float = 0.20

    # =====================================================
    # Trust reference point
    # =====================================================

    trust_neutral_point: float = 0.50

    trust_floor: float = 0.0

    # =====================================================
    # Reward
    # =====================================================

    se_weight: float = 0.45

    ee_weight: float = 0.20

    trust_weight: float = 0.15

    trust_delta_weight: float = 0.10

    interference_penalty: float = 0.05

    power_penalty: float = 0.03

    queue_penalty: float = 0.07

    # =====================================================
    # Reward normalization
    # =====================================================

    se_target: float = 0.20

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
    TA-FDRL-IRF Phase-4A environment.

    Polarization-aware PHY
    + Adaptive Trust
    + Fixed RIS

    -----------------------------------------------------
    State
    -----------------------------------------------------

    Five features per user:

        1. SINR
        2. Interference
        3. Queue
        4. Power
        5. Trust

    For 20 users:

        state_dim = 20 * 5
                  = 100

    -----------------------------------------------------
    Action
    -----------------------------------------------------

    Bandwidth allocation:

        20

    Power allocation:

        20

    Total:

        action_dim = 40

    RIS optimization is disabled in Phase-4A.

    -----------------------------------------------------
    Polarization
    -----------------------------------------------------

    Each channel is represented by:

        H_p =
        [ h_VV  h_VH ]
        [ h_HV  h_HH ]

    where:

        VV = V transmit -> V receive
        VH = H transmit -> V receive
        HV = V transmit -> H receive
        HH = H transmit -> H receive

    Phase-4A uses V-polarized transmission as
    the reference transmission vector.

    -----------------------------------------------------
    Effective polarization channel
    -----------------------------------------------------

        H_eff =
            H_direct
            +
            sum_n(
                phi_n *
                H_RIS_user,n *
                H_BS_RIS,n
            )

    -----------------------------------------------------
    Trust
    -----------------------------------------------------

    Adaptive trust is calculated from:

        service quality
        interference quality
        queue quality
        temporal stability

    -----------------------------------------------------
    Reward
    -----------------------------------------------------

        R =
            w_SE * normalized_SE
          + w_EE * normalized_EE
          + w_T  * mean_trust
          + w_dT * trust_delta
          - w_I * interference
          - w_P * power
          - w_Q * queue

    -----------------------------------------------------
    RNG Architecture
    -----------------------------------------------------

    Channel RNG:
        Used exclusively for channel realizations.

    Dynamics RNG:
        Used exclusively for stochastic environment dynamics.

    The two streams are independent and deterministic.

    This allows fair Polarization OFF vs ON comparison.
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

        # =================================================
        # Research-grade independent RNG streams
        # =================================================

        self.base_seed = int(
            self.cfg.seed
        )

        # -------------------------------------------------
        # Channel RNG
        #
        # Used only for:
        #
        #   scalar channels
        #   polarization channels
        # -------------------------------------------------

        self.channel_rng = np.random.default_rng(
            self.base_seed + 1000
        )

        # -------------------------------------------------
        # Dynamics RNG
        #
        # Used only for:
        #
        #   initial queue
        #   initial trust
        #   queue arrivals
        # -------------------------------------------------

        self.dynamics_rng = np.random.default_rng(
            self.base_seed + 2000
        )

        # -------------------------------------------------
        # Backward compatibility
        #
        # Existing code that references self.rng
        # continues to operate.
        #
        # self.rng now maps to dynamics RNG.
        # -------------------------------------------------

        self.rng = self.dynamics_rng

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
        # Scalar channel variables
        #
        # Kept for Phase-3 compatibility.
        # -------------------------------------------------

        self.direct_channel = None

        self.bs_ris_channel = None

        self.ris_user_channel = None

        # -------------------------------------------------
        # Polarization channel variables
        #
        # direct:
        #
        #   (users, rx_pol, tx_pol)
        #
        # BS -> RIS:
        #
        #   (ris, rx_pol, tx_pol)
        #
        # RIS -> user:
        #
        #   (users, ris, rx_pol, tx_pol)
        # -------------------------------------------------

        self.direct_polarization_channel = None

        self.bs_ris_polarization_channel = None

        self.ris_user_polarization_channel = None

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
        # Trust diagnostics
        # -------------------------------------------------

        self.behavior_score = None

        self.trust_delta = None

        self.mean_trust = 0.0

        # -------------------------------------------------
        # Performance diagnostics
        # -------------------------------------------------

        self.rates = None

        self.spectral_efficiency = 0.0

        self.energy_efficiency = 0.0

        self.mean_queue = 0.0

        self.mean_power = 0.0

        self.interference_ratio = 0.0

        # -------------------------------------------------
        # Research diagnostics
        #
        # Useful for controlled OFF vs ON experiments.
        # -------------------------------------------------

        self.last_arrivals = None

    # =====================================================
    # RANDOM SEED CONTROL
    # =====================================================

    def set_seed(
        self,
        seed: int,
    ) -> None:

        """
        Configure independent deterministic RNG streams.

        Channel RNG:
            Used exclusively for channel realizations.

        Dynamics RNG:
            Used exclusively for queue/trust/dynamics.

        Seed offsets:

            channel = seed + 1000
            dynamics = seed + 2000

        The offsets ensure independent random streams while
        preserving deterministic reproducibility.
        """

        self.base_seed = int(seed)

        # -------------------------------------------------
        # Independent channel stream
        # -------------------------------------------------

        self.channel_rng = np.random.default_rng(
            self.base_seed + 1000
        )

        # -------------------------------------------------
        # Independent dynamics stream
        # -------------------------------------------------

        self.dynamics_rng = np.random.default_rng(
            self.base_seed + 2000
        )

        # -------------------------------------------------
        # Backward compatibility
        # -------------------------------------------------

        self.rng = self.dynamics_rng

    # =====================================================
    # RESET
    # =====================================================

    def reset(
        self,
        seed: int | None = None,
    ) -> np.ndarray:

        """
        Reset environment.

        A new channel realization and trust state
        are generated.

        If a seed is supplied, both independent RNG
        streams are deterministically reinitialized.
        """

        if seed is not None:

            self.set_seed(
                seed
            )

        self.step_count = 0

        # -------------------------------------------------
        # Generate channels
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
        # Initial queue
        #
        # IMPORTANT:
        # Dynamics RNG only.
        # -------------------------------------------------

        self.queue = self.dynamics_rng.uniform(
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
        #
        # IMPORTANT:
        # Dynamics RNG only.
        # -------------------------------------------------

        if self.cfg.fixed_trust:

            self.trust = np.ones(
                self.num_users,
                dtype=np.float64,
            )

        else:

            self.trust = self.dynamics_rng.uniform(
                0.75,
                0.95,
                size=self.num_users,
            )

        # -------------------------------------------------
        # Previous variables
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

        # -------------------------------------------------
        # Performance diagnostics
        # -------------------------------------------------

        self.rates = np.zeros(
            self.num_users,
            dtype=np.float64,
        )

        self.spectral_efficiency = 0.0

        self.energy_efficiency = 0.0

        self.mean_queue = float(
            np.mean(self.queue)
        )

        self.mean_power = float(
            np.mean(self.power)
        )

        self.interference_ratio = 0.0

        # -------------------------------------------------
        # Research diagnostics
        # -------------------------------------------------

        self.last_arrivals = np.zeros(
            self.num_users,
            dtype=np.float64,
        )

        return self._get_state()

    # =====================================================
    # CHANNEL GENERATION
    # =====================================================

    def _generate_channels(self):

        """
        Generate all PHY/channel realizations.

        IMPORTANT:

        Every channel-related random sample comes
        exclusively from channel_rng.

        This prevents queue/trust dynamics from being
        affected by additional polarization random draws.
        """

        # =================================================
        # Phase-3 scalar channels
        #
        # Kept for compatibility and fallback.
        # =================================================

        self.direct_channel = (
            self.channel_rng.normal(
                size=self.num_users
            )
            + 1j
            * self.channel_rng.normal(
                size=self.num_users
            )
        ) / np.sqrt(2.0)

        self.bs_ris_channel = (
            self.channel_rng.normal(
                size=self.num_ris
            )
            + 1j
            * self.channel_rng.normal(
                size=self.num_ris
            )
        ) / np.sqrt(2.0)

        self.ris_user_channel = (
            self.channel_rng.normal(
                size=(
                    self.num_users,
                    self.num_ris,
                )
            )
            + 1j
            * self.channel_rng.normal(
                size=(
                    self.num_users,
                    self.num_ris,
                )
            )
        ) / np.sqrt(2.0)

        # =================================================
        # Phase-4A polarization channels
        # =================================================

        if not self.cfg.polarization_enabled:

            self.direct_polarization_channel = None

            self.bs_ris_polarization_channel = None

            self.ris_user_polarization_channel = None

            return

        # -------------------------------------------------
        # Polarization parameters
        # -------------------------------------------------

        v_strength = (
            self.cfg.polarization_v_strength
        )

        h_strength = (
            self.cfg.polarization_h_strength
        )

        xpol = (
            self.cfg.cross_polarization_factor
        )

        # =================================================
        # Direct BS -> User
        #
        # Shape:
        #
        #   (users, 2, 2)
        #
        # Ordering:
        #
        #   [rx polarization, tx polarization]
        #
        #   0 = V
        #   1 = H
        # =================================================

        self.direct_polarization_channel = (
            self.channel_rng.normal(
                size=(
                    self.num_users,
                    2,
                    2,
                )
            )
            + 1j
            * self.channel_rng.normal(
                size=(
                    self.num_users,
                    2,
                    2,
                )
            )
        ) / np.sqrt(2.0)

        # -------------------------------------------------
        # Co-polarized components
        # -------------------------------------------------

        self.direct_polarization_channel[
            :, 0, 0
        ] *= v_strength

        self.direct_polarization_channel[
            :, 1, 1
        ] *= h_strength

        # -------------------------------------------------
        # Cross-polarized components
        # -------------------------------------------------

        self.direct_polarization_channel[
            :, 0, 1
        ] *= xpol

        self.direct_polarization_channel[
            :, 1, 0
        ] *= xpol

        # =================================================
        # BS -> RIS
        #
        # Shape:
        #
        #   (ris, 2, 2)
        # =================================================

        self.bs_ris_polarization_channel = (
            self.channel_rng.normal(
                size=(
                    self.num_ris,
                    2,
                    2,
                )
            )
            + 1j
            * self.channel_rng.normal(
                size=(
                    self.num_ris,
                    2,
                    2,
                )
            )
        ) / np.sqrt(2.0)

        self.bs_ris_polarization_channel[
            :, 0, 0
        ] *= v_strength

        self.bs_ris_polarization_channel[
            :, 1, 1
        ] *= h_strength

        self.bs_ris_polarization_channel[
            :, 0, 1
        ] *= xpol

        self.bs_ris_polarization_channel[
            :, 1, 0
        ] *= xpol

        # =================================================
        # RIS -> User
        #
        # Shape:
        #
        #   (users, ris, 2, 2)
        # =================================================

        self.ris_user_polarization_channel = (
            self.channel_rng.normal(
                size=(
                    self.num_users,
                    self.num_ris,
                    2,
                    2,
                )
            )
            + 1j
            * self.channel_rng.normal(
                size=(
                    self.num_users,
                    self.num_ris,
                    2,
                    2,
                )
            )
        ) / np.sqrt(2.0)

        self.ris_user_polarization_channel[
            :, :, 0, 0
        ] *= v_strength

        self.ris_user_polarization_channel[
            :, :, 1, 1
        ] *= h_strength

        self.ris_user_polarization_channel[
            :, :, 0, 1
        ] *= xpol

        self.ris_user_polarization_channel[
            :, :, 1, 0
        ] *= xpol

    # =====================================================
    # SCALAR EFFECTIVE CHANNEL
    # =====================================================

    def _effective_channel(
        self,
        ris_phase: np.ndarray,
    ) -> np.ndarray:

        """
        Phase-3 scalar effective channel.

        Used only when polarization is disabled.
        """

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
    # POLARIZATION EFFECTIVE CHANNEL
    # =====================================================

    def _effective_polarization_channel(
        self,
        ris_phase: np.ndarray,
    ) -> np.ndarray:

        """
        Compute polarization-aware effective channel.

        Returns:

            shape = (num_users, 2, 2)

        Matrix for user u:

            H_u =
            [ h_VV  h_VH ]
            [ h_HV  h_HH ]

        where:

            h_VV:
                V -> V

            h_VH:
                H -> V

            h_HV:
                V -> H

            h_HH:
                H -> H
        """

        if not self.cfg.polarization_enabled:

            raise RuntimeError(
                "Polarization is disabled."
            )

        phi = np.exp(
            1j * ris_phase
        )

        # -------------------------------------------------
        # Direct component
        # -------------------------------------------------

        h_direct = (
            self.direct_polarization_channel
        )

        # -------------------------------------------------
        # Reflected component
        # -------------------------------------------------

        reflected = np.zeros(
            (
                self.num_users,
                2,
                2,
            ),
            dtype=np.complex128,
        )

        # -------------------------------------------------
        # RIS elements
        # -------------------------------------------------

        for n in range(self.num_ris):

            # RIS-user matrix:
            #
            # (users, rx_pol, ris_pol)

            h_ru = (
                self.ris_user_polarization_channel[
                    :,
                    n,
                    :,
                    :
                ]
            )

            # BS-RIS matrix:
            #
            # (ris_pol, tx_pol)

            h_br = (
                self.bs_ris_polarization_channel[
                    n,
                    :,
                    :
                ]
            )

            # Matrix multiplication:
            #
            # H_ru @ H_br
            #
            # Result:
            #
            # (users, rx_pol, tx_pol)

            reflected += (
                phi[n]
                * np.einsum(
                    "uij,jk->uik",
                    h_ru,
                    h_br,
                )
            )

        # -------------------------------------------------
        # Total effective polarization channel
        # -------------------------------------------------

        h_eff = (
            h_direct
            + reflected
        )

        return h_eff

    # =====================================================
    # POLARIZATION GAIN
    # =====================================================

    def _polarization_channel_gain(
        self,
        ris_phase: np.ndarray,
    ) -> np.ndarray:

        """
        Convert 2x2 polarization channel into
        scalar received channel gain.

        Phase-4A uses V-polarized transmission.

        Transmit vector:

            [1]
            [0]

        Therefore:

            received = H_eff @ tx

        and total received power is:

            |V_received|²
            +
            |H_received|²
        """

        h_pol = (
            self._effective_polarization_channel(
                ris_phase
            )
        )

        # -------------------------------------------------
        # V-polarized transmit vector
        # -------------------------------------------------

        tx_vector = np.array(
            [1.0, 0.0],
            dtype=np.complex128,
        )

        # -------------------------------------------------
        # Received polarization vector
        # -------------------------------------------------

        received_pol = np.einsum(
            "uij,j->ui",
            h_pol,
            tx_vector,
        )

        # -------------------------------------------------
        # Total received gain
        # -------------------------------------------------

        channel_gain = np.sum(
            np.abs(received_pol) ** 2,
            axis=1,
        )

        return np.maximum(
            channel_gain,
            1e-12,
        )

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

        # =================================================
        # Fixed trust
        # =================================================

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

        # =================================================
        # 1. Service quality
        # =================================================

        service_quality = (
            self._calculate_service_quality(
                rates
            )
        )

        # =================================================
        # 2. Queue quality
        # =================================================

        queue_quality = np.clip(
            1.0 - self.queue,
            0.0,
            1.0,
        )

        # =================================================
        # 3. Interference quality
        # =================================================

        interference_quality = np.clip(
            1.0 - interference_ratio,
            0.0,
            1.0,
        )

        # =================================================
        # 4. Temporal stability
        # =================================================

        queue_change = np.abs(
            self.queue
            - self.previous_queue
        )

        stability_quality = np.clip(
            1.0 - queue_change,
            0.0,
            1.0,
        )

        # =================================================
        # Composite behavior score
        # =================================================

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

        self.behavior_score = (
            behavior_score.copy()
        )

        # =================================================
        # Trust movement relative to neutral point
        # =================================================

        centered_behavior = (
            behavior_score
            - self.cfg.trust_neutral_point
        )

        # =================================================
        # Adaptive trust update
        # =================================================

        old_trust = (
            self.trust.copy()
        )

        self.trust = (
            self.cfg.trust_memory
            * self.trust
            +
            (
                1.0
                - self.cfg.trust_memory
            )
            * (
                self.trust
                + self.cfg.trust_learning_rate
                * centered_behavior
            )
        )

        # -------------------------------------------------
        # Trust floor
        # -------------------------------------------------

        self.trust = np.maximum(
            self.trust,
            self.cfg.trust_floor,
        )

        self.trust = np.clip(
            self.trust,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Trust delta
        # -------------------------------------------------

        self.trust_delta = (
            self.trust
            - old_trust
        )

        self.mean_trust = float(
            np.mean(self.trust)
        )

    # =====================================================
    # ACTION PROCESSING
    # =====================================================

    def _process_action(
        self,
        action: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:

        action = np.asarray(
            action,
            dtype=np.float64,
        )

        base_action_dim = (
            self.num_users * 2
        )

        if self.cfg.optimize_ris:

            expected_dim = (
                base_action_dim
                + self.num_ris
            )

        else:

            expected_dim = (
                base_action_dim
            )

        if action.shape != (
            expected_dim,
        ):

            raise ValueError(
                "Invalid action shape. "
                f"Expected ({expected_dim},), "
                f"got {action.shape}."
            )

        # -------------------------------------------------
        # Clip action
        # -------------------------------------------------

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        # -------------------------------------------------
        # Bandwidth
        # -------------------------------------------------

        bandwidth_raw = (
            action[
                :self.num_users
            ]
            + 1.0
        )

        bandwidth_raw = np.maximum(
            bandwidth_raw,
            1e-12,
        )

        bandwidth = (
            bandwidth_raw
            / np.sum(bandwidth_raw)
            * self.cfg.bandwidth_hz
        )

        # -------------------------------------------------
        # Power
        # -------------------------------------------------

        power_action = action[
            self.num_users:
            self.num_users * 2
        ]

        power = (
            power_action
            + 1.0
        ) / 2.0

        power *= (
            self.cfg.max_power_w
        )

        # -------------------------------------------------
        # RIS phase
        # -------------------------------------------------

        if self.cfg.optimize_ris:

            ris_action = action[
                self.num_users * 2:
            ]

            ris_phase = (
                ris_action
                * np.pi
            )

        else:

            ris_phase = np.zeros(
                self.num_ris,
                dtype=np.float64,
            )

        return (
            bandwidth,
            power,
            ris_phase,
        )

    # =====================================================
    # STEP
    # =====================================================

    def step(
        self,
        action: np.ndarray,
    ):

        (
            bandwidth,
            power,
            ris_phase,
        ) = self._process_action(
            action
        )

        # -------------------------------------------------
        # Channel gain
        # -------------------------------------------------

        if self.cfg.polarization_enabled:

            channel_gain = (
                self._polarization_channel_gain(
                    ris_phase
                )
            )

        else:

            h_eff = (
                self._effective_channel(
                    ris_phase
                )
            )

            channel_gain = (
                np.abs(h_eff) ** 2
            )

        channel_gain = np.maximum(
            channel_gain,
            1e-12,
        )

        # -------------------------------------------------
        # Received power
        # -------------------------------------------------

        received_power = (
            power
            * channel_gain
        )

        # -------------------------------------------------
        # Interference
        # -------------------------------------------------

        total_received_power = (
            np.sum(received_power)
        )

        interference = (
            total_received_power
            - received_power
        )

        interference = np.maximum(
            interference,
            0.0,
        )

        # -------------------------------------------------
        # SINR
        # -------------------------------------------------

        denominator = (
            interference
            + self.noise_power_w
        )

        self.sinr = (
            received_power
            / np.maximum(
                denominator,
                1e-15,
            )
        )

        self.sinr = np.maximum(
            self.sinr,
            0.0,
        )

        # -------------------------------------------------
        # Rates
        # -------------------------------------------------

        rates = (
            bandwidth
            * np.log2(
                1.0
                + self.sinr
            )
        )

        rates = np.maximum(
            rates,
            0.0,
        )

        self.rates = rates.copy()

        # -------------------------------------------------
        # Spectral efficiency
        # -------------------------------------------------

        total_rate = (
            np.sum(rates)
        )

        self.spectral_efficiency = (
            total_rate
            / max(
                self.cfg.bandwidth_hz,
                1e-12,
            )
        )

        # -------------------------------------------------
        # Energy efficiency
        # -------------------------------------------------

        total_power = (
            np.sum(power)
            + self.cfg.circuit_power_w
        )

        self.energy_efficiency = (
            total_rate
            / max(
                total_power,
                1e-12,
            )
        )

        # -------------------------------------------------
        # Interference ratio
        # -------------------------------------------------

        interference_total = (
            np.sum(interference)
        )

        received_total = (
            np.sum(received_power)
        )

        self.interference_ratio = (
            interference_total
            / max(
                received_total,
                1e-12,
            )
        )

        self.interference_ratio = float(
            np.clip(
                self.interference_ratio,
                0.0,
                1.0,
            )
        )

        # =================================================
        # Queue dynamics
        #
        # IMPORTANT:
        #
        # Queue arrivals use dynamics_rng only.
        #
        # This ensures polarization channel generation
        # cannot alter the stochastic traffic sequence.
        # =================================================

        arrivals = self.dynamics_rng.uniform(
            0.0,
            0.05,
            size=self.num_users,
        )

        self.last_arrivals = (
            arrivals.copy()
        )

        normalized_service = (
            rates
            / max(
                self.cfg.trust_target_rate_bps,
                1e-12,
            )
        )

        service_fraction = np.clip(
            normalized_service,
            0.0,
            1.0,
        )

        service = (
            0.05
            * service_fraction
        )

        self.queue = (
            self.queue
            + arrivals
            - service
        )

        self.queue = np.clip(
            self.queue,
            0.0,
            1.0,
        )

        # =================================================
        # Trust update
        # =================================================

        self._update_trust(
            rates,
            self.interference_ratio,
        )

        # =================================================
        # Reward normalization
        # =================================================

        normalized_se = np.clip(
            self.spectral_efficiency
            / max(
                self.cfg.se_target,
                1e-12,
            ),
            0.0,
            1.0,
        )

        normalized_ee = np.clip(
            self.energy_efficiency
            / max(
                self.cfg.ee_target,
                1e-12,
            ),
            0.0,
            1.0,
        )

        normalized_interference = (
            self.interference_ratio
        )

        normalized_power = (
            np.mean(power)
            / max(
                self.cfg.max_power_w,
                1e-12,
            )
        )

        normalized_queue = (
            np.mean(self.queue)
        )

        normalized_trust_delta = float(
            np.mean(
                self.trust_delta
            )
        )

        # =================================================
        # Reward
        # =================================================

        reward = (

            self.cfg.se_weight
            * normalized_se

            + self.cfg.ee_weight
            * normalized_ee

            + self.cfg.trust_weight
            * self.mean_trust

            + self.cfg.trust_delta_weight
            * normalized_trust_delta

            - self.cfg.interference_penalty
            * normalized_interference

            - self.cfg.power_penalty
            * normalized_power

            - self.cfg.queue_penalty
            * normalized_queue
        )

        reward = float(
            reward
        )

        # =================================================
        # Previous variables
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
        # Diagnostics
        # =================================================

        self.mean_queue = float(
            np.mean(self.queue)
        )

        self.mean_power = float(
            np.mean(power)
        )

        # =================================================
        # Step count
        # =================================================

        self.step_count += 1

        terminated = (
            self.step_count
            >= self.cfg.max_steps
        )

        truncated = False

        # =================================================
        # Next state
        # =================================================

        next_state = (
            self._get_state()
        )

        # =================================================
        # Info
        # =================================================

        info = {

            "spectral_efficiency":
                float(
                    self.spectral_efficiency
                ),

            "energy_efficiency":
                float(
                    self.energy_efficiency
                ),

            "rate":
                float(
                    total_rate
                ),

            "sinr":
                float(
                    np.mean(self.sinr)
                ),

            "trust":
                float(
                    self.mean_trust
                ),

            "behavior_score":
                float(
                    np.mean(
                        self.behavior_score
                    )
                ),

            "trust_delta":
                float(
                    np.mean(
                        self.trust_delta
                    )
                ),

            "queue":
                float(
                    self.mean_queue
                ),

            "power":
                float(
                    self.mean_power
                ),

            "interference":
                float(
                    self.interference_ratio
                ),

            "polarization_enabled":
                bool(
                    self.cfg.polarization_enabled
                ),

            # -------------------------------------------------
            # Research diagnostics
            # -------------------------------------------------

            "arrival_mean":
                float(
                    np.mean(
                        self.last_arrivals
                    )
                ),

            "arrival_sum":
                float(
                    np.sum(
                        self.last_arrivals
                    )
                ),
        }

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        )

    # =====================================================
    # STATE
    # =====================================================

    def _get_state(
        self,
    ) -> np.ndarray:

        # -------------------------------------------------
        # Normalize SINR
        # -------------------------------------------------

        sinr_normalized = np.clip(
            self.sinr
            / 10.0,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Normalize interference
        # -------------------------------------------------

        interference_normalized = np.clip(
            self.interference
            / max(
                np.max(
                    self.interference
                ),
                1e-12,
            ),
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Queue
        # -------------------------------------------------

        queue_normalized = np.clip(
            self.queue,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Power
        # -------------------------------------------------

        power_normalized = np.clip(
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

        trust_normalized = np.clip(
            self.trust,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Concatenate
        # -------------------------------------------------

        state = np.concatenate(
            [
                sinr_normalized,
                interference_normalized,
                queue_normalized,
                power_normalized,
                trust_normalized,
            ]
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

        base_dim = (
            self.num_users * 2
        )

        if self.cfg.optimize_ris:

            return (
                base_dim
                + self.num_ris
            )

        return base_dim


# =========================================================
# END OF FILE
# =========================================================