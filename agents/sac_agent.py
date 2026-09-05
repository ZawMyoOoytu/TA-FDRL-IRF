from __future__ import annotations

import copy
import random

from collections import deque

import numpy as np

import torch
import torch.nn.functional as F

import torch.optim as optim

from agents.actor import Actor
from agents.critic import Critic


class ReplayBuffer:

    def __init__(
        self,
        capacity: int = 100_000,
    ):

        self.buffer = deque(
            maxlen=capacity
        )

    # =====================================================
    # Add
    # =====================================================

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        self.buffer.append(
            (
                np.asarray(
                    state,
                    dtype=np.float32,
                ),

                np.asarray(
                    action,
                    dtype=np.float32,
                ),

                float(reward),

                np.asarray(
                    next_state,
                    dtype=np.float32,
                ),

                float(done),
            )
        )

    # =====================================================
    # Sample
    # =====================================================

    def sample(
        self,
        batch_size: int,
    ):

        batch = random.sample(
            self.buffer,
            batch_size,
        )

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = zip(*batch)

        return (

            np.asarray(
                states,
                dtype=np.float32,
            ),

            np.asarray(
                actions,
                dtype=np.float32,
            ),

            np.asarray(
                rewards,
                dtype=np.float32,
            ),

            np.asarray(
                next_states,
                dtype=np.float32,
            ),

            np.asarray(
                dones,
                dtype=np.float32,
            ),
        )

    def __len__(self):

        return len(
            self.buffer
        )


class SACAgent:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,

        hidden_dim: int = 256,

        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,

        gamma: float = 0.99,
        tau: float = 0.005,

        buffer_size: int = 100_000,
        batch_size: int = 256,

        device: str | None = None,
    ):

        # =================================================
        # Device
        # =================================================

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(
            device
        )

        self.gamma = gamma
        self.tau = tau

        self.batch_size = (
            batch_size
        )

        # =================================================
        # Actor
        # =================================================

        self.actor = Actor(
            state_dim,
            action_dim,
            hidden_dim,
        ).to(
            self.device
        )

        # =================================================
        # Critic
        # =================================================

        self.critic = Critic(
            state_dim,
            action_dim,
            hidden_dim,
        ).to(
            self.device
        )

        # =================================================
        # Target critic
        # =================================================

        self.target_critic = (
            copy.deepcopy(
                self.critic
            ).to(
                self.device
            )
        )

        for parameter in (
            self.target_critic.parameters()
        ):

            parameter.requires_grad = False

        # =================================================
        # Optimizers
        # =================================================

        self.actor_optimizer = (
            optim.Adam(
                self.actor.parameters(),
                lr=actor_lr,
            )
        )

        self.critic_optimizer = (
            optim.Adam(
                self.critic.parameters(),
                lr=critic_lr,
            )
        )

        # =================================================
        # Entropy temperature
        # =================================================

        self.target_entropy = (
            -float(action_dim)
        )

        self.log_alpha = torch.zeros(
            1,
            requires_grad=True,
            device=self.device,
        )

        self.alpha_optimizer = (
            optim.Adam(
                [self.log_alpha],
                lr=alpha_lr,
            )
        )

        # =================================================
        # Replay buffer
        # =================================================

        self.replay_buffer = (
            ReplayBuffer(
                buffer_size
            )
        )

    # =====================================================
    # Alpha
    # =====================================================

    @property
    def alpha(self):

        return self.log_alpha.exp()

    # =====================================================
    # Action
    # =====================================================

    def select_action(
        self,
        state,
        evaluate: bool = False,
    ):

        state_tensor = (
            torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            )
            .unsqueeze(0)
        )

        with torch.no_grad():

            (
                action,
                _,
                deterministic_action,
            ) = self.actor.sample(
                state_tensor
            )

        if evaluate:

            selected_action = (
                deterministic_action
            )

        else:

            selected_action = (
                action
            )

        selected_action = (
            torch.nan_to_num(
                selected_action,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
        )

        selected_action = torch.clamp(
            selected_action,
            -1.0,
            1.0,
        )

        return (
            selected_action
            .squeeze(0)
            .cpu()
            .numpy()
        )

    # =====================================================
    # Remember
    # =====================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    # =====================================================
    # Update
    # =====================================================

    def update(self):

        if (
            len(self.replay_buffer)
            < self.batch_size
        ):

            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.float32,
            device=self.device,
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)

        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)

        # =================================================
        # Target
        # =================================================

        with torch.no_grad():

            (
                next_actions,
                next_log_prob,
                _,
            ) = self.actor.sample(
                next_states
            )

            (
                target_q1,
                target_q2,
            ) = self.target_critic(
                next_states,
                next_actions,
            )

            target_q = torch.minimum(
                target_q1,
                target_q2,
            )

            target_q = (
                target_q
                - self.alpha.detach()
                * next_log_prob
            )

            target = (
                rewards
                + (
                    1.0 - dones
                )
                * self.gamma
                * target_q
            )

        # =================================================
        # Critic
        # =================================================

        (
            current_q1,
            current_q2,
        ) = self.critic(
            states,
            actions,
        )

        critic_loss = (
            F.mse_loss(
                current_q1,
                target,
            )
            +
            F.mse_loss(
                current_q2,
                target,
            )
        )

        if not torch.isfinite(
            critic_loss
        ):

            return None

        self.critic_optimizer.zero_grad(
            set_to_none=True
        )

        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            max_norm=1.0,
        )

        self.critic_optimizer.step()

        # =================================================
        # Actor
        # =================================================

        (
            new_actions,
            log_prob,
            _,
        ) = self.actor.sample(
            states
        )

        q1_new = (
            self.critic.q1_forward(
                states,
                new_actions,
            )
        )

        actor_loss = (
            self.alpha.detach()
            * log_prob
            - q1_new
        ).mean()

        if not torch.isfinite(
            actor_loss
        ):

            return None

        self.actor_optimizer.zero_grad(
            set_to_none=True
        )

        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            max_norm=1.0,
        )

        self.actor_optimizer.step()

        # =================================================
        # Alpha
        # =================================================

        alpha_loss = -(
            self.log_alpha
            * (
                log_prob
                + self.target_entropy
            ).detach()
        ).mean()

        self.alpha_optimizer.zero_grad(
            set_to_none=True
        )

        alpha_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [self.log_alpha],
            max_norm=1.0,
        )

        self.alpha_optimizer.step()

        # =================================================
        # Target update
        # =================================================

        self._soft_update()

        return {

            "critic_loss":
                float(
                    critic_loss.item()
                ),

            "actor_loss":
                float(
                    actor_loss.item()
                ),

            "alpha_loss":
                float(
                    alpha_loss.item()
                ),

            "alpha":
                float(
                    self.alpha.item()
                ),
        }

    # =====================================================
    # Soft target update
    # =====================================================

    def _soft_update(self):

        for (
            target_parameter,
            parameter,
        ) in zip(
            self.target_critic.parameters(),
            self.critic.parameters(),
        ):

            target_parameter.data.copy_(
                self.tau
                * parameter.data
                +
                (
                    1.0
                    - self.tau
                )
                * target_parameter.data
            )

    # =====================================================
    # Save
    # =====================================================

    def save(
        self,
        path: str,
    ):

        torch.save(
            {
                "actor":
                    self.actor.state_dict(),

                "critic":
                    self.critic.state_dict(),

                "target_critic":
                    self.target_critic.state_dict(),

                "log_alpha":
                    self.log_alpha.detach().cpu(),
            },
            path,
        )

    # =====================================================
    # Load
    # =====================================================

    def load(
        self,
        path: str,
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.actor.load_state_dict(
            checkpoint["actor"]
        )

        self.critic.load_state_dict(
            checkpoint["critic"]
        )

        self.target_critic.load_state_dict(
            checkpoint["target_critic"]
        )

        self.log_alpha.data.copy_(
            checkpoint[
                "log_alpha"
            ].to(
                self.device
            )
        )