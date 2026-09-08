"""
TA-FDRL-IRF
Phase-4B.2
Polarization-Aware Intelligent Radio Fabric Environment

Research-grade simulation environment for:
- 6G Intelligent Radio Fabric (IRF)
- Polarization-aware PHY
- Adaptive trust dynamics
- Governance-compatible telemetry
- Optional polarization-aware observations

Phase-4B.2:
    State:
        Base:
            [SINR, interference, queue, power, trust] x users
            = 20 x 5 = 100

        Optional polarization observations:
            [polarization_quality, cross_polarization_ratio] x users
            = 20 x 2 = 40

        Total:
            100 + 40 = 140

    Action:
        bandwidth allocation x users
        power allocation x users

        = 20 + 20
        = 40

    RIS optimization remains disabled in Phase-4B.2 by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class IRFConfig:
    # ------------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------------

    num_users: int = 20
    num_ris_elements: int = 64

    bandwidth_hz: float = 100e6
    carrier_frequency_hz: float = 28e9

    # ------------------------------------------------------------------------
    # Polarization PHY
    # ------------------------------------------------------------------------

    polarization_enabled: bool = True

    # Phase-4B.2:
    # Expose polarization information to the RL observation/state.
    polarization_state_enabled: bool = False

    cross_polarization_factor: float = 0.15

    polarization_v_strength: float = 1.0
    polarization_h_strength: float = 1.0

    # ------------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------------

    max_power_w: float = 1.0
    circuit_power_w: float = 0.1

    # ------------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------------

    noise_figure_db: float = 7.0
    noise_density_dbm_hz: float = -174.0

    # ------------------------------------------------------------------------
    # Episode
    # ------------------------------------------------------------------------

    max_steps: int = 200

    # ------------------------------------------------------------------------
    # RIS
    # ------------------------------------------------------------------------

    optimize_ris: bool = False

    # ------------------------------------------------------------------------
    # Trust
    # ------------------------------------------------------------------------

    fixed_trust: bool = False

    trust_memory: float = 0.90
    trust_learning_rate: float = 0.10

    trust_target_rate_bps: float = 1e7

    trust_service_weight: float = 0.35
    trust_interference_weight: float = 0.20
    trust_queue_weight: float = 0.25
    trust_instability_weight: float = 0.20

    trust_neutral_point: float = 0.50
    trust_floor: float = 0.0

    # ------------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------------

    se_weight: float = 0.45
    ee_weight: float = 0.20
    trust_weight: float = 0.15
    trust_delta_weight: float = 0.10

    interference_penalty: float = 0.05
    power_penalty: float = 0.03
    queue_penalty: float = 0.07

    # ------------------------------------------------------------------------
    # Reward normalization targets
    # ------------------------------------------------------------------------

    se_target: float = 0.20
    ee_target: float = 1e6

    # ------------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------------

    seed: int = 42


# ============================================================================
# Environment
# ============================================================================


class IRFEnvironment:
    """
    Polarization-aware 6G IRF environment.

    The environment maintains:

        - direct channel
        - BS -> RIS channel
        - RIS -> user channel
        - optional polarization channel matrices
        - bandwidth allocation
        - transmit power
        - interference
        - queue
        - trust
        - behavior score
        - PHY telemetry

    RNG architecture:

        channel_rng  = seed + 1000
        dynamics_rng = seed + 2000

    This prevents channel generation from consuming the same RNG stream
    used by queue/trust dynamics.
    """

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(self, cfg: IRFConfig | None = None):

        self.cfg = cfg if cfg is not None else IRFConfig()

        # ---------------------------------------------------------------------
        # Dimensions
        # ---------------------------------------------------------------------

        self.num_users = int(self.cfg.num_users)
        self.num_ris_elements = int(self.cfg.num_ris_elements)

        self.base_state_dim = self.num_users * 5

        self.polarization_observation_dim = (
            self.num_users * 2
            if self.cfg.polarization_state_enabled
            else 0
        )

        self.state_dim = (
            self.base_state_dim
            + self.polarization_observation_dim
        )

        self.base_action_dim = self.num_users * 2

        self.ris_action_dim = (
            self.num_ris_elements
            if self.cfg.optimize_ris
            else 0
        )

        self.action_dim = (
            self.base_action_dim
            + self.ris_action_dim
        )

        # ---------------------------------------------------------------------
        # Independent RNG streams
        # ---------------------------------------------------------------------

        self.seed = int(self.cfg.seed)

        self.channel_rng = np.random.default_rng(
            self.seed + 1000
        )

        self.dynamics_rng = np.random.default_rng(
            self.seed + 2000
        )

        # Backward-compatible alias.
        self.rng = self.dynamics_rng

        # ---------------------------------------------------------------------
        # Internal simulation state
        # ---------------------------------------------------------------------

        self.current_step = 0

        self.bandwidth = np.full(
            self.num_users,
            self.cfg.bandwidth_hz / self.num_users,
            dtype=np.float64,
        )

        self.power = np.full(
            self.num_users,
            self.cfg.max_power_w / self.num_users,
            dtype=np.float64,
        )

        self.interference = np.zeros(
            self.num_users,
            dtype=np.float64,
        )

        self.queue = np.zeros(
            self.num_users,
            dtype=np.float64,
        )

        self.trust = np.full(
            self.num_users,
            0.85,
            dtype=np.float64,
        )

        self.behavior_score = np.full(
            self.num_users,
            0.85,
            dtype=np.float64,
        )

        self.trust_delta = np.zeros(
            self.num_users,
            dtype=np.float64,
        )

        # ---------------------------------------------------------------------
        # Channel state
        # ---------------------------------------------------------------------

        self.direct_channel = np.zeros(
            self.num_users,
            dtype=np.complex128,
        )

        self.bs_ris_channel = np.zeros(
            self.num_ris_elements,
            dtype=np.complex128,
        )

        self.ris_user_channel = np.zeros(
            (
                self.num_users,
                self.num_ris_elements,
            ),
            dtype=np.complex128,
        )

        # ---------------------------------------------------------------------
        # Polarization channels
        #
        # Shape:
        #
        # direct:
        #     (users, 2, 2)
        #
        # BS -> RIS:
        #     (ris, 2, 2)
        #
        # RIS -> users:
        #     (users, ris, 2, 2)
        # ---------------------------------------------------------------------

        self.direct_channel_p = np.zeros(
            (
                self.num_users,
                2,
                2,
            ),
            dtype=np.complex128,
        )

        self.bs_ris_channel_p = np.zeros(
            (
                self.num_ris_elements,
                2,
                2,
            ),
            dtype=np.complex128,
        )

        self.ris_user_channel_p = np.zeros(
            (
                self.num_users,
                self.num_ris_elements,
                2,
                2,
            ),
            dtype=np.complex128,
        )

        # ---------------------------------------------------------------------
        # Polarization diagnostics
        # ---------------------------------------------------------------------

        self.polarization_quality = np.ones(
            self.num_users,
            dtype=np.float64,
        )

        self.cross_polarization_ratio = np.zeros(
            self.num_users,
            dtype=np.float64,
        )

        # ---------------------------------------------------------------------
        # Performance metrics
        # ---------------------------------------------------------------------

        self.spectral_efficiency = 0.0
        self.energy_efficiency = 0.0

        self.total_rate = 0.0
        self.mean_sinr = 0.0

        self.mean_trust = 0.0
        self.mean_behavior_score = 0.0

        self.mean_queue = 0.0
        self.mean_power = 0.0

        self.radiated_power = 0.0
        self.total_power = 0.0

        self.interference_ratio = 0.0

        # ---------------------------------------------------------------------
        # Previous-step values
        # ---------------------------------------------------------------------

        self.previous_trust = self.trust.copy()
        self.previous_queue = self.queue.copy()
        self.previous_power = self.power.copy()

    # =========================================================================
    # Channel generation
    # =========================================================================

    def _generate_channels(self) -> None:
        """
        Generate deterministic channel state using channel_rng only.
        """

        # ---------------------------------------------------------------------
        # Scalar channel model
        # ---------------------------------------------------------------------

        self.direct_channel = (
            self.channel_rng.normal(
                0.0,
                1.0 / np.sqrt(2.0),
                self.num_users,
            )
            + 1j
            * self.channel_rng.normal(
                0.0,
                1.0 / np.sqrt(2.0),
                self.num_users,
            )
        )

        self.bs_ris_channel = (
            self.channel_rng.normal(
                0.0,
                1.0 / np.sqrt(2.0),
                self.num_ris_elements,
            )
            + 1j
            * self.channel_rng.normal(
                0.0,
                1.0 / np.sqrt(2.0),
                self.num_ris_elements,
            )
        )

        self.ris_user_channel = (
            self.channel_rng.normal(
                0.0,
                1.0 / np.sqrt(2.0),
                (
                    self.num_users,
                    self.num_ris_elements,
                ),
            )
            + 1j
            * self.channel_rng.normal(
                0.0,
                1.0 / np.sqrt(2.0),
                (
                    self.num_users,
                    self.num_ris_elements,
                ),
            )
        )

        # ---------------------------------------------------------------------
        # Polarization channel model
        # ---------------------------------------------------------------------

        if not self.cfg.polarization_enabled:
            self.direct_channel_p.fill(0.0)
            self.bs_ris_channel_p.fill(0.0)
            self.ris_user_channel_p.fill(0.0)
            return

        # Cross-polarization coupling.
        #
        # Diagonal terms:
        #     VV
        #     HH
        #
        # Off-diagonal:
        #     VH
        #     HV
        #
        # The off-diagonal terms are scaled by
        # cross_polarization_factor.
        # ---------------------------------------------------------------------

        xpol = float(
            np.clip(
                self.cfg.cross_polarization_factor,
                0.0,
                1.0,
            )
        )

        v_strength = float(
            max(
                self.cfg.polarization_v_strength,
                0.0,
            )
        )

        h_strength = float(
            max(
                self.cfg.polarization_h_strength,
                0.0,
            )
        )

        # Direct channel.
        self.direct_channel_p = (
            self.channel_rng.normal(
                0.0,
                1.0 / np.sqrt(2.0),
                (
                    self.num_users,
                    2,
                    2,
                ),
            )
            + 1j
            * self.channel_rng.normal(
                0.0,
                1.0 / np.sqrt(2.0),
                (
                    self.num_users,
                    2,
                    2,
                ),
            )
        )

        # BS-RIS channel.
        self.bs_ris_channel_p = (
            self.channel_rng.normal(
                0.0,
                1.0 / np.sqrt(2.0),
                (
                    self.num_ris_elements,
                    2,
                    2,
                ),
            )
            + 1j
            * self.channel_rng.normal(
                0.0,
                1.0 / np.sqrt(2.0),
                (
                    self.num_ris_elements,
                    2,
                    2,
                ),
            )
        )

        # RIS-user channel.
        self.ris_user_channel_p = (
            self.channel_rng.normal(
                0.0,
                1.0 / np.sqrt(2.0),
                (
                    self.num_users,
                    self.num_ris_elements,
                    2,
                    2,
                ),
            )
            + 1j
            * self.channel_rng.normal(
                0.0,
                1.0 / np.sqrt(2.0),
                (
                    self.num_users,
                    self.num_ris_elements,
                    2,
                    2,
                ),
            )
        )

        # ---------------------------------------------------------------------
        # Apply polarization strengths and cross-polarization scaling.
        # ---------------------------------------------------------------------

        # Direct
        self.direct_channel_p[:, 0, 0] *= v_strength
        self.direct_channel_p[:, 1, 1] *= h_strength

        self.direct_channel_p[:, 0, 1] *= xpol
        self.direct_channel_p[:, 1, 0] *= xpol

        # BS -> RIS
        self.bs_ris_channel_p[:, 0, 0] *= v_strength
        self.bs_ris_channel_p[:, 1, 1] *= h_strength

        self.bs_ris_channel_p[:, 0, 1] *= xpol
        self.bs_ris_channel_p[:, 1, 0] *= xpol

        # RIS -> user
        self.ris_user_channel_p[:, :, 0, 0] *= v_strength
        self.ris_user_channel_p[:, :, 1, 1] *= h_strength

        self.ris_user_channel_p[:, :, 0, 1] *= xpol
        self.ris_user_channel_p[:, :, 1, 0] *= xpol

    # =========================================================================
    # Polarization channel
    # =========================================================================

    def _effective_polarization_channel(
        self,
        ris_phase: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate effective 2x2 polarization channel for every user.

        H_eff[user]
            =
            H_direct[user]
            +
            sum_n(
                phi_n
                * H_RIS_user[user,n]
                * H_BS_RIS[n]
            )

        Returns:
            ndarray shape (num_users, 2, 2)
        """

        ris_phase = np.asarray(
            ris_phase,
            dtype=np.complex128,
        )

        if ris_phase.shape != (
            self.num_ris_elements,
        ):
            raise ValueError(
                "RIS phase vector has invalid shape: "
                f"{ris_phase.shape}; expected "
                f"({self.num_ris_elements},)"
            )

        # Matrix multiplication:
        #
        # H_RIS_user[u,n] @ H_BS_RIS[n]
        #
        # Result:
        #     (users, ris, 2, 2)
        #
        cascaded = np.einsum(
            "unij,njk->unik",
            self.ris_user_channel_p,
            self.bs_ris_channel_p,
            optimize=True,
        )

        effective = (
            self.direct_channel_p.copy()
            + np.einsum(
                "n,unik->uik",
                ris_phase,
                cascaded,
                optimize=True,
            )
        )

        return effective

    # =========================================================================
    # Polarization diagnostics
    # =========================================================================

    def _polarization_diagnostics(
        self,
        ris_phase: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute polarization quality and cross-polarization ratio.

        Reference transmitter:
            x_tx = [1, 0]^T

        Therefore:
            y = H_eff @ x_tx

        which corresponds to the first column of H_eff.

        V received power:
            |y_V|^2

        H received power:
            |y_H|^2

        PQI:
            V_power / total_power

        XPI:
            H_power / total_power

        Returns:
            polarization_quality,
            cross_polarization_ratio
        """

        effective = self._effective_polarization_channel(
            ris_phase
        )

        # V-polarized transmit reference [1, 0]^T.
        y_v = effective[:, 0, 0]
        y_h = effective[:, 1, 0]

        v_power = np.abs(y_v) ** 2
        h_power = np.abs(y_h) ** 2

        total_power = v_power + h_power

        safe_total = np.maximum(
            total_power,
            1e-12,
        )

        pqi = np.clip(
            v_power / safe_total,
            0.0,
            1.0,
        )

        xpi = np.clip(
            h_power / safe_total,
            0.0,
            1.0,
        )

        return (
            pqi.astype(np.float64),
            xpi.astype(np.float64),
        )

    # =========================================================================
    # Polarization channel gain
    # =========================================================================

    def _polarization_channel_gain(
        self,
        ris_phase: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate total polarization-aware channel power gain.

        G_p =
            |y_V|^2 + |y_H|^2

        for V-polarized transmission.
        """

        effective = self._effective_polarization_channel(
            ris_phase
        )

        y_v = effective[:, 0, 0]
        y_h = effective[:, 1, 0]

        gain = (
            np.abs(y_v) ** 2
            + np.abs(y_h) ** 2
        )

        return np.maximum(
            gain.astype(np.float64),
            1e-12,
        )

    # =========================================================================
    # Scalar channel gain
    # =========================================================================

    def _scalar_channel_gain(
        self,
        ris_phase: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate scalar effective channel power gain.
        """

        ris_phase = np.asarray(
            ris_phase,
            dtype=np.complex128,
        )

        cascaded = (
            self.ris_user_channel
            * self.bs_ris_channel[np.newaxis, :]
        )

        reflected = np.sum(
            ris_phase[np.newaxis, :]
            * cascaded,
            axis=1,
        )

        effective = (
            self.direct_channel
            + reflected
        )

        gain = np.abs(effective) ** 2

        return np.maximum(
            gain.astype(np.float64),
            1e-12,
        )

    # =========================================================================
    # Noise
    # =========================================================================

    def _noise_power_w(self) -> float:
        """
        Thermal noise:

            N = kTB

        using the configured noise spectral density in dBm/Hz.

        Since bandwidth is system-wide, this returns total noise power.
        """

        noise_density_dbm_hz = (
            self.cfg.noise_density_dbm_hz
        )

        noise_figure_db = (
            self.cfg.noise_figure_db
        )

        noise_dbm = (
            noise_density_dbm_hz
            + 10.0
            * np.log10(
                max(
                    self.cfg.bandwidth_hz,
                    1e-12,
                )
            )
            + noise_figure_db
        )

        # dBm -> W
        noise_w = (
            10.0
            ** (
                (noise_dbm - 30.0)
                / 10.0
            )
        )

        return float(
            max(
                noise_w,
                1e-15,
            )
        )

    # =========================================================================
    # Reset
    # =========================================================================

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> np.ndarray:

        if seed is not None:
            self.seed = int(seed)

        # Re-create independent RNG streams.
        self.channel_rng = np.random.default_rng(
            self.seed + 1000
        )

        self.dynamics_rng = np.random.default_rng(
            self.seed + 2000
        )

        self.rng = self.dynamics_rng

        self.current_step = 0

        # ---------------------------------------------------------------------
        # Reset system state
        # ---------------------------------------------------------------------

        self.bandwidth = np.full(
            self.num_users,
            self.cfg.bandwidth_hz / self.num_users,
            dtype=np.float64,
        )

        self.power = np.full(
            self.num_users,
            self.cfg.max_power_w / self.num_users,
            dtype=np.float64,
        )

        self.interference.fill(0.0)

        self.queue.fill(0.0)

        if self.cfg.fixed_trust:
            self.trust.fill(
                self.cfg.trust_neutral_point
            )
        else:
            self.trust = np.clip(
                self.dynamics_rng.normal(
                    0.85,
                    0.03,
                    self.num_users,
                ),
                0.0,
                1.0,
            )

        self.behavior_score = self.trust.copy()

        self.trust_delta.fill(0.0)

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------

        self.spectral_efficiency = 0.0
        self.energy_efficiency = 0.0

        self.total_rate = 0.0
        self.mean_sinr = 0.0

        self.mean_trust = float(
            np.mean(self.trust)
        )

        self.mean_behavior_score = float(
            np.mean(self.behavior_score)
        )

        self.mean_queue = float(
            np.mean(self.queue)
        )

        self.mean_power = float(
            np.mean(self.power)
        )

        self.radiated_power = float(
            np.sum(self.power)
        )

        self.total_power = (
            self.radiated_power
            + self.cfg.circuit_power_w
        )

        self.interference_ratio = 0.0

        # ---------------------------------------------------------------------
        # Generate channel state
        # ---------------------------------------------------------------------

        self._generate_channels()

        # ---------------------------------------------------------------------
        # Fixed RIS phase in Phase-4B.2
        # ---------------------------------------------------------------------

        ris_phase = self._default_ris_phase()

        if self.cfg.polarization_enabled:
            (
                self.polarization_quality,
                self.cross_polarization_ratio,
            ) = self._polarization_diagnostics(
                ris_phase
            )
        else:
            self.polarization_quality.fill(1.0)
            self.cross_polarization_ratio.fill(0.0)

        # ---------------------------------------------------------------------
        # Previous values
        # ---------------------------------------------------------------------

        self.previous_trust = self.trust.copy()
        self.previous_queue = self.queue.copy()
        self.previous_power = self.power.copy()

        return self._get_state()

    # =========================================================================
    # Default RIS phase
    # =========================================================================

    def _default_ris_phase(self) -> np.ndarray:
        """
        Fixed RIS phase used when optimize_ris=False.
        """

        return np.ones(
            self.num_ris_elements,
            dtype=np.complex128,
        )

    # =========================================================================
    # Action processing
    # =========================================================================

    def _process_action(
        self,
        action: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Decode and validate action.

        Action layout:

            [bandwidth_0 ... bandwidth_N-1,
             power_0     ... power_N-1,
             optional RIS phase actions]
        """

        action = np.asarray(
            action,
            dtype=np.float64,
        ).reshape(-1)

        if action.size != self.action_dim:
            raise ValueError(
                "Invalid action dimension: "
                f"received {action.size}, "
                f"expected {self.action_dim}"
            )

        # ---------------------------------------------------------------------
        # Bandwidth
        # ---------------------------------------------------------------------

        bandwidth_raw = action[
            : self.num_users
        ]

        # Map [-1,1] -> [0,1].
        bandwidth_weight = (
            np.clip(
                bandwidth_raw,
                -1.0,
                1.0,
            )
            + 1.0
        ) / 2.0

        # Avoid all-zero allocation.
        bandwidth_weight += 1e-6

        bandwidth = (
            bandwidth_weight
            / np.sum(bandwidth_weight)
            * self.cfg.bandwidth_hz
        )

        # ---------------------------------------------------------------------
        # Power
        # ---------------------------------------------------------------------

        power_raw = action[
            self.num_users :
            2 * self.num_users
        ]

        power_fraction = (
            np.clip(
                power_raw,
                -1.0,
                1.0,
            )
            + 1.0
        ) / 2.0

        power = (
            power_fraction
            * self.cfg.max_power_w
        )

        # ---------------------------------------------------------------------
        # RIS phase
        # ---------------------------------------------------------------------

        if self.cfg.optimize_ris:

            ris_raw = action[
                2 * self.num_users :
            ]

            ris_phase = np.exp(
                1j
                * np.pi
                * np.clip(
                    ris_raw,
                    -1.0,
                    1.0,
                )
            )

        else:

            ris_phase = self._default_ris_phase()

        return (
            bandwidth.astype(np.float64),
            power.astype(np.float64),
            ris_phase,
        )

    # =========================================================================
    # Trust update
    # =========================================================================

    def _update_trust(
        self,
        rates: np.ndarray,
        sinr: np.ndarray,
    ) -> None:
        """
        Adaptive trust update.

        Components:

            service quality
            interference quality
            queue quality
            stability quality
        """

        if self.cfg.fixed_trust:

            self.trust_delta = np.zeros(
                self.num_users,
                dtype=np.float64,
            )

            self.behavior_score = self.trust.copy()

            self.mean_trust = float(
                np.mean(self.trust)
            )

            self.mean_behavior_score = float(
                np.mean(self.behavior_score)
            )

            return

        # ---------------------------------------------------------------------
        # Service quality
        # ---------------------------------------------------------------------

        target = max(
            self.cfg.trust_target_rate_bps,
            1e-12,
        )

        service_score = np.clip(
            rates / target,
            0.0,
            1.0,
        )

        # ---------------------------------------------------------------------
        # Interference quality
        # ---------------------------------------------------------------------

        interference_safe = np.maximum(
            self.interference,
            1e-12,
        )

        interference_score = np.clip(
            1.0
            / (
                1.0
                + interference_safe
                / np.maximum(
                    np.mean(interference_safe),
                    1e-12,
                )
            ),
            0.0,
            1.0,
        )

        # ---------------------------------------------------------------------
        # Queue quality
        # ---------------------------------------------------------------------

        queue_scale = max(
            np.mean(self.queue)
            + 1e-6,
            1e-6,
        )

        queue_score = np.clip(
            1.0
            / (
                1.0
                + self.queue
                / queue_scale
            ),
            0.0,
            1.0,
        )

        # ---------------------------------------------------------------------
        # Stability
        # ---------------------------------------------------------------------

        if np.all(
            np.isfinite(
                self.previous_power
            )
        ):

            power_change = np.abs(
                self.power
                - self.previous_power
            )

            max_power = max(
                self.cfg.max_power_w,
                1e-12,
            )

            stability_score = np.clip(
                1.0
                - power_change
                / max_power,
                0.0,
                1.0,
            )

        else:

            stability_score = np.ones(
                self.num_users,
                dtype=np.float64,
            )

        # ---------------------------------------------------------------------
        # Behavior score
        # ---------------------------------------------------------------------

        self.behavior_score = np.clip(
            self.cfg.trust_service_weight
            * service_score
            + self.cfg.trust_interference_weight
            * interference_score
            + self.cfg.trust_queue_weight
            * queue_score
            + self.cfg.trust_instability_weight
            * stability_score,
            0.0,
            1.0,
        )

        # ---------------------------------------------------------------------
        # Adaptive trust
        # ---------------------------------------------------------------------

        old_trust = self.trust.copy()

        self.trust = (
            self.cfg.trust_memory
            * self.trust
            + (
                1.0
                - self.cfg.trust_memory
            )
            * self.behavior_score
        )

        # Additional learning-rate interpolation.
        self.trust = (
            (1.0 - self.cfg.trust_learning_rate)
            * old_trust
            + self.cfg.trust_learning_rate
            * self.trust
        )

        self.trust = np.maximum(
            self.trust,
            self.cfg.trust_floor,
        )

        self.trust = np.clip(
            self.trust,
            0.0,
            1.0,
        )

        self.trust_delta = (
            self.trust
            - old_trust
        )

        self.mean_trust = float(
            np.mean(self.trust)
        )

        self.mean_behavior_score = float(
            np.mean(self.behavior_score)
        )

    # =========================================================================
    # Reward
    # =========================================================================

    def _calculate_reward(
        self,
        spectral_efficiency: float,
        energy_efficiency: float,
    ) -> float:
        """
        Multi-objective reward.

        Components:

            spectral efficiency
            energy efficiency
            trust
            trust delta
            interference penalty
            power penalty
            queue penalty
        """

        se_normalized = (
            spectral_efficiency
            / max(
                self.cfg.se_target,
                1e-12,
            )
        )

        ee_normalized = (
            energy_efficiency
            / max(
                self.cfg.ee_target,
                1e-12,
            )
        )

        trust_component = self.mean_trust

        trust_delta_component = float(
            np.mean(
                self.trust_delta
            )
        )

        interference_penalty = (
            self.interference_ratio
        )

        power_penalty = (
            self.mean_power
            / max(
                self.cfg.max_power_w,
                1e-12,
            )
        )

        queue_penalty = float(
            np.mean(self.queue)
        )

        reward = (
            self.cfg.se_weight
            * se_normalized
            + self.cfg.ee_weight
            * ee_normalized
            + self.cfg.trust_weight
            * trust_component
            + self.cfg.trust_delta_weight
            * trust_delta_component
            - self.cfg.interference_penalty
            * interference_penalty
            - self.cfg.power_penalty
            * power_penalty
            - self.cfg.queue_penalty
            * queue_penalty
        )

        return float(
            np.nan_to_num(
                reward,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )

    # =========================================================================
    # State
    # =========================================================================

    def _get_state(self) -> np.ndarray:
        """
        Construct RL observation.

        Base state per user:

            normalized SINR
            normalized interference
            queue
            normalized power
            trust

        If polarization_state_enabled:

            polarization quality
            cross polarization ratio

        Therefore:

            scalar PHY:
                20 x 5 = 100

            polarization-aware:
                20 x 7 = 140
        """

        # ---------------------------------------------------------------------
        # SINR
        # ---------------------------------------------------------------------

        sinr_safe = np.maximum(
            getattr(
                self,
                "_current_sinr",
                np.zeros(self.num_users),
            ),
            0.0,
        )

        # Log compression.
        sinr_normalized = np.clip(
            np.log1p(sinr_safe)
            / 20.0,
            0.0,
            1.0,
        )

        # ---------------------------------------------------------------------
        # Interference
        # ---------------------------------------------------------------------

        interference_scale = max(
            self.cfg.max_power_w,
            1e-12,
        )

        interference_normalized = np.clip(
            self.interference
            / interference_scale,
            0.0,
            1.0,
        )

        # ---------------------------------------------------------------------
        # Queue
        # ---------------------------------------------------------------------

        queue_normalized = np.clip(
            self.queue,
            0.0,
            1.0,
        )

        # ---------------------------------------------------------------------
        # Power
        # ---------------------------------------------------------------------

        power_normalized = np.clip(
            self.power
            / max(
                self.cfg.max_power_w,
                1e-12,
            ),
            0.0,
            1.0,
        )

        # ---------------------------------------------------------------------
        # Trust
        # ---------------------------------------------------------------------

        trust_normalized = np.clip(
            self.trust,
            0.0,
            1.0,
        )

        # ---------------------------------------------------------------------
        # Base state
        # ---------------------------------------------------------------------

        base_features = np.stack(
            [
                sinr_normalized,
                interference_normalized,
                queue_normalized,
                power_normalized,
                trust_normalized,
            ],
            axis=1,
        )

        # ---------------------------------------------------------------------
        # Optional polarization state
        # ---------------------------------------------------------------------

        if self.cfg.polarization_state_enabled:

            pqi = np.clip(
                self.polarization_quality,
                0.0,
                1.0,
            )

            xpi = np.clip(
                self.cross_polarization_ratio,
                0.0,
                1.0,
            )

            state_features = np.concatenate(
                [
                    base_features,
                    pqi[:, np.newaxis],
                    xpi[:, np.newaxis],
                ],
                axis=1,
            )

        else:

            state_features = base_features

        state = state_features.reshape(-1)

        state = np.nan_to_num(
            state,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        return state.astype(
            np.float32
        )

    # =========================================================================
    # Step
    # =========================================================================

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        float,
        bool,
        bool,
        Dict[str, Any],
    ]:
        """
        Execute one environment step.

        Gymnasium API:

            observation,
            reward,
            terminated,
            truncated,
            info
        """

        # ---------------------------------------------------------------------
        # Decode action
        # ---------------------------------------------------------------------

        (
            bandwidth,
            power,
            ris_phase,
        ) = self._process_action(
            action
        )

        # ---------------------------------------------------------------------
        # Commit action
        # ---------------------------------------------------------------------

        self.bandwidth = bandwidth
        self.power = power

        # ---------------------------------------------------------------------
        # Channel gain
        # ---------------------------------------------------------------------

        if self.cfg.polarization_enabled:

            (
                self.polarization_quality,
                self.cross_polarization_ratio,
            ) = self._polarization_diagnostics(
                ris_phase
            )

            channel_gain = (
                self._polarization_channel_gain(
                    ris_phase
                )
            )

        else:

            channel_gain = (
                self._scalar_channel_gain(
                    ris_phase
                )
            )

            self.polarization_quality = (
                np.ones(
                    self.num_users,
                    dtype=np.float64,
                )
            )

            self.cross_polarization_ratio = (
                np.zeros(
                    self.num_users,
                    dtype=np.float64,
                )
            )

        # ---------------------------------------------------------------------
        # Received power
        # ---------------------------------------------------------------------

        received_power = (
            self.power
            * channel_gain
        )

        received_power = np.maximum(
            received_power,
            0.0,
        )

        # ---------------------------------------------------------------------
        # Interference
        # ---------------------------------------------------------------------
        #
        # Each user's total received power contains its own desired signal
        # plus contributions from other users.
        #
        # This simplified shared-channel model derives aggregate interference
        # from total received power.
        # ---------------------------------------------------------------------

        total_received_power = float(
            np.sum(received_power)
        )

        own_received_power = (
            received_power
        )

        total_noise = (
            self._noise_power_w()
        )

        aggregate_interference = max(
            total_received_power
            - float(
                np.mean(
                    own_received_power
                )
            ),
            0.0,
        )

        # Distribute aggregate interference across users.
        #
        # This keeps interference per-user finite and deterministic.
        self.interference = np.full(
            self.num_users,
            aggregate_interference
            / max(
                self.num_users,
                1,
            ),
            dtype=np.float64,
        )

        # ---------------------------------------------------------------------
        # SINR
        # ---------------------------------------------------------------------

        sinr = (
            own_received_power
            / (
                self.interference
                + total_noise
            )
        )

        sinr = np.nan_to_num(
            sinr,
            nan=0.0,
            posinf=1e12,
            neginf=0.0,
        )

        sinr = np.maximum(
            sinr,
            0.0,
        )

        self._current_sinr = sinr.copy()

        # ---------------------------------------------------------------------
        # Rate
        # ---------------------------------------------------------------------

        rates = (
            self.bandwidth
            * np.log2(
                1.0 + sinr
            )
        )

        rates = np.nan_to_num(
            rates,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        rates = np.maximum(
            rates,
            0.0,
        )

        total_rate = float(
            np.sum(rates)
        )

        self.total_rate = total_rate

        # ---------------------------------------------------------------------
        # Spectral efficiency
        # ---------------------------------------------------------------------

        self.spectral_efficiency = (
            total_rate
            / max(
                self.cfg.bandwidth_hz,
                1e-12,
            )
        )

        # ---------------------------------------------------------------------
        # Power accounting
        # ---------------------------------------------------------------------
        #
        # radiated_power:
        #     sum of transmit powers
        #
        # total_power:
        #     radiated + circuit
        #
        # EE uses total_power.
        # ---------------------------------------------------------------------

        self.radiated_power = float(
            np.sum(self.power)
        )

        self.total_power = (
            self.radiated_power
            + float(
                self.cfg.circuit_power_w
            )
        )

        self.energy_efficiency = (
            total_rate
            / max(
                self.total_power,
                1e-12,
            )
        )

        # ---------------------------------------------------------------------
        # Mean SINR
        # ---------------------------------------------------------------------

        self.mean_sinr = float(
            np.mean(sinr)
        )

        # ---------------------------------------------------------------------
        # Interference ratio
        # ---------------------------------------------------------------------

        total_interference = float(
            np.sum(self.interference)
        )

        total_signal = float(
            np.sum(received_power)
        )

        self.interference_ratio = float(
            np.clip(
                total_interference
                / max(
                    total_signal,
                    1e-12,
                ),
                0.0,
                1.0,
            )
        )

        # ---------------------------------------------------------------------
        # Queue dynamics
        # ---------------------------------------------------------------------

        arrivals = self.dynamics_rng.uniform(
            0.0,
            0.05,
            self.num_users,
        )

        service_ratio = np.clip(
            rates
            / max(
                self.cfg.trust_target_rate_bps,
                1e-12,
            ),
            0.0,
            1.0,
        )

        service = (
            0.05
            * service_ratio
        )

        self.queue = np.clip(
            self.queue
            + arrivals
            - service,
            0.0,
            1.0,
        )

        # ---------------------------------------------------------------------
        # Trust update
        # ---------------------------------------------------------------------

        self._update_trust(
            rates,
            sinr,
        )

        # ---------------------------------------------------------------------
        # Aggregate metrics
        # ---------------------------------------------------------------------

        self.mean_queue = float(
            np.mean(self.queue)
        )

        self.mean_power = float(
            np.mean(self.power)
        )

        self.mean_trust = float(
            np.mean(self.trust)
        )

        self.mean_behavior_score = float(
            np.mean(
                self.behavior_score
            )
        )

        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------

        reward = self._calculate_reward(
            self.spectral_efficiency,
            self.energy_efficiency,
        )

        # ---------------------------------------------------------------------
        # Update previous state
        # ---------------------------------------------------------------------

        self.previous_trust = (
            self.trust.copy()
        )

        self.previous_queue = (
            self.queue.copy()
        )

        self.previous_power = (
            self.power.copy()
        )

        # ---------------------------------------------------------------------
        # Step counter
        # ---------------------------------------------------------------------

        self.current_step += 1

        terminated = (
            self.current_step
            >= self.cfg.max_steps
        )

        truncated = False

        # ---------------------------------------------------------------------
        # Polarization aggregate diagnostics
        # ---------------------------------------------------------------------

        mean_pqi = float(
            np.mean(
                self.polarization_quality
            )
        )

        mean_xpi = float(
            np.mean(
                self.cross_polarization_ratio
            )
        )

        min_pqi = float(
            np.min(
                self.polarization_quality
            )
        )

        max_pqi = float(
            np.max(
                self.polarization_quality
            )
        )

        min_xpi = float(
            np.min(
                self.cross_polarization_ratio
            )
        )

        max_xpi = float(
            np.max(
                self.cross_polarization_ratio
            )
        )

        # ---------------------------------------------------------------------
        # Telemetry
        #
        # IMPORTANT:
        # These canonical keys are required by:
        #
        # experiments/phase4b2_sac_training.py
        #
        # Do not remove them.
        # ---------------------------------------------------------------------

        info: Dict[str, Any] = {

            # =================================================================
            # Canonical Phase-4B.2 training telemetry
            # =================================================================

            "spectral_efficiency": float(
                self.spectral_efficiency
            ),

            "energy_efficiency": float(
                self.energy_efficiency
            ),

            "mean_trust": float(
                self.mean_trust
            ),

            "mean_behavior_score": float(
                self.mean_behavior_score
            ),

            "mean_queue": float(
                self.mean_queue
            ),

            "mean_power": float(
                self.mean_power
            ),

            "total_power": float(
                self.total_power
            ),

            "interference_ratio": float(
                self.interference_ratio
            ),

            "mean_pqi": float(
                mean_pqi
            ),

            "mean_xpi": float(
                mean_xpi
            ),

            # =================================================================
            # Additional useful telemetry
            # =================================================================

            "total_rate": float(
                self.total_rate
            ),

            "mean_sinr": float(
                self.mean_sinr
            ),

            "radiated_power": float(
                self.radiated_power
            ),

            "arrival_mean": float(
                np.mean(arrivals)
            ),

            "arrival_sum": float(
                np.sum(arrivals)
            ),

            "state_interference_mean": float(
                np.mean(self.interference)
            ),

            "state_power_mean": float(
                np.mean(self.power)
            ),

            # =================================================================
            # Backward-compatible names
            # =================================================================

            "rate": rates.copy(),

            "sinr": sinr.copy(),

            "trust": self.trust.copy(),

            "behavior_score": (
                self.behavior_score.copy()
            ),

            "trust_delta": (
                self.trust_delta.copy()
            ),

            "queue": self.queue.copy(),

            "power": self.power.copy(),

            "interference": (
                self.interference.copy()
            ),

            # =================================================================
            # Polarization telemetry
            # =================================================================

            "polarization_enabled": bool(
                self.cfg.polarization_enabled
            ),

            "polarization_state_enabled": bool(
                self.cfg.polarization_state_enabled
            ),

            "cross_polarization_factor": float(
                self.cfg.cross_polarization_factor
            ),

            "mean_polarization_quality": float(
                mean_pqi
            ),

            "mean_cross_polarization_ratio": float(
                mean_xpi
            ),

            "min_polarization_quality": float(
                min_pqi
            ),

            "max_polarization_quality": float(
                max_pqi
            ),

            "min_cross_polarization_ratio": float(
                min_xpi
            ),

            "max_cross_polarization_ratio": float(
                max_xpi
            ),

            # Per-user arrays for research analysis.
            "polarization_quality": (
                self.polarization_quality.copy()
            ),

            "cross_polarization_ratio": (
                self.cross_polarization_ratio.copy()
            ),

            # =================================================================
            # Configuration / state metadata
            # =================================================================

            "num_users": int(
                self.num_users
            ),

            "num_ris_elements": int(
                self.num_ris_elements
            ),

            "state_dim": int(
                self.state_dim
            ),

            "action_dim": int(
                self.action_dim
            ),

            "optimize_ris": bool(
                self.cfg.optimize_ris
            ),

            "fixed_trust": bool(
                self.cfg.fixed_trust
            ),

            "step": int(
                self.current_step
            ),
        }

        # ---------------------------------------------------------------------
        # Next state
        # ---------------------------------------------------------------------

        next_state = self._get_state()

        return (
            next_state,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )


# ============================================================================
# Factory
# ============================================================================


def create_environment(
    cfg: IRFConfig | None = None,
) -> IRFEnvironment:
    """
    Standard environment factory.

    This is kept for compatibility with:
        - training scripts
        - evaluation scripts
        - Streamlit dashboard
        - governance closed-loop experiments
    """

    if cfg is None:
        cfg = IRFConfig()

    return IRFEnvironment(cfg)


# ============================================================================
# Simple validation helper
# ============================================================================


def validate_environment(
    cfg: IRFConfig | None = None,
) -> Dict[str, Any]:
    """
    Lightweight environment validation.

    Returns a dictionary suitable for debugging/research logs.
    """

    if cfg is None:
        cfg = IRFConfig()

    env = IRFEnvironment(cfg)

    state = env.reset(
        seed=cfg.seed
    )

    zero_action = np.zeros(
        env.action_dim,
        dtype=np.float32,
    )

    (
        next_state,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(
        zero_action
    )

    required_keys = [
        "spectral_efficiency",
        "energy_efficiency",
        "mean_trust",
        "mean_behavior_score",
        "mean_queue",
        "mean_power",
        "total_power",
        "interference_ratio",
        "mean_pqi",
        "mean_xpi",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in info
    ]

    if missing_keys:
        raise RuntimeError(
            "Required Phase-4B.2 telemetry missing: "
            + ", ".join(missing_keys)
        )

    return {
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "state_shape": tuple(
            state.shape
        ),
        "next_state_shape": tuple(
            next_state.shape
        ),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "required_telemetry_ok": True,
        "mean_pqi": float(
            info["mean_pqi"]
        ),
        "mean_xpi": float(
            info["mean_xpi"]
        ),
        "total_power": float(
            info["total_power"]
        ),
    }


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":

    cfg = IRFConfig(
        polarization_enabled=True,
        polarization_state_enabled=True,
        optimize_ris=False,
        fixed_trust=False,
    )

    result = validate_environment(
        cfg
    )

    print("=" * 72)
    print(
        "TA-FDRL-IRF | Phase-4B.2 Environment Validation"
    )
    print("=" * 72)

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )

    print("=" * 72)
    print("VALIDATION PASSED")
    print("=" * 72)