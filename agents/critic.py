from __future__ import annotations

import torch

import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                state_dim + action_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):

        x = torch.cat(
            [
                state,
                action,
            ],
            dim=-1,
        )

        return self.network(
            x
        )


class Critic(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):

        super().__init__()

        self.q1 = QNetwork(
            state_dim,
            action_dim,
            hidden_dim,
        )

        self.q2 = QNetwork(
            state_dim,
            action_dim,
            hidden_dim,
        )

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):

        q1 = self.q1(
            state,
            action,
        )

        q2 = self.q2(
            state,
            action,
        )

        return (
            q1,
            q2,
        )

    # =====================================================
    # Q1 only
    # =====================================================

    def q1_forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):

        return self.q1(
            state,
            action,
        )