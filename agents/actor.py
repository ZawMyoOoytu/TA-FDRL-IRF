from __future__ import annotations

import torch
import torch.nn as nn

from torch.distributions import Normal


class Actor(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                state_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.ReLU(),
        )

        self.mean = nn.Linear(
            hidden_dim,
            action_dim,
        )

        self.log_std = nn.Linear(
            hidden_dim,
            action_dim,
        )

        self.log_std_min = -5.0
        self.log_std_max = 1.0

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        state: torch.Tensor,
    ):

        x = self.network(
            state
        )

        mean = self.mean(
            x
        )

        log_std = self.log_std(
            x
        )

        log_std = torch.clamp(
            log_std,
            self.log_std_min,
            self.log_std_max,
        )

        std = torch.exp(
            log_std
        )

        return (
            mean,
            std,
        )

    # =====================================================
    # Sample
    # =====================================================

    def sample(
        self,
        state: torch.Tensor,
    ):

        mean, std = self.forward(
            state
        )

        distribution = Normal(
            mean,
            std,
        )

        raw_action = (
            distribution.rsample()
        )

        action = torch.tanh(
            raw_action
        )

        log_prob = (
            distribution.log_prob(
                raw_action
            )
        )

        # -------------------------------------------------
        # Tanh correction
        # -------------------------------------------------

        log_prob -= torch.log(
            1.0
            - action.pow(2)
            + 1e-6
        )

        log_prob = log_prob.sum(
            dim=-1,
            keepdim=True,
        )

        deterministic_action = (
            torch.tanh(
                mean
            )
        )

        return (
            action,
            log_prob,
            deterministic_action,
        )