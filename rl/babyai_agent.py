from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import astuple
from gym import Space
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from transformers import GPT2Config

import agent
from agent import NNBase
from babyai_env import Spaces
from utils import init


def get_size(space: Space):
    if isinstance(space, (Box, MultiDiscrete)):
        return int(np.prod(space.shape))
    if isinstance(space, Discrete):
        return 1
    raise TypeError()


class Lambda(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class Agent(agent.Agent):
    def __init__(self, observation_space, **kwargs):
        # spaces = Spaces(*observation_space.spaces)
        super().__init__(
            # obs_shape=spaces.image.shape,
            obs_shape=observation_space.shape,
            observation_space=observation_space,
            **kwargs,
        )

    def build_base(self, obs_shape, **kwargs):
        return Base(**kwargs)

    def forward(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        # try:
        # action[:] = float(input("go:"))
        # except ValueError:
        # pass

        return value, action, action_log_probs, rnn_hxs


class GRUEmbed(nn.Module):
    def __init__(self, num_embeddings: int, hidden_size: int, output_size: int):
        super().__init__()
        gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.embed = nn.Sequential(
            nn.Embedding(num_embeddings, hidden_size),
            gru,
        )
        self.projection = nn.Linear(hidden_size, output_size)

    def forward(self, x, **_):
        hidden = self.embed.forward(x)[1].squeeze(0)
        return self.projection(hidden)


class Base(NNBase):
    def __init__(
        self,
        embedding_size: str,
        hidden_size: int,
        observation_space: Box,
        recurrent: bool,
        second_layer: bool,
        # encoded: torch.Tensor,
    ):
        super().__init__(
            recurrent=recurrent,
            recurrent_input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.num_embeddings = int(observation_space.high.max())
        self.embedding_dim = hidden_size

        self.embed = self._build_embed()
        self.K = nn.Linear(hidden_size, hidden_size)
        self.Q = nn.Linear(hidden_size, hidden_size)

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.critic_linear = init_(nn.Linear(2, 1))

        self.train()

    def _build_embed(self):
        return nn.Sequential(
            nn.EmbeddingBag(
                num_embeddings=self.num_embeddings,
                embedding_dim=self.embedding_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def forward(self, inputs, rnn_hxs, masks):
        n = inputs.size(0)
        obs = inputs.reshape(n * 3, -1)
        embedded = self.embed(obs.long())
        embedded = embedded.reshape(n, 3, -1)
        lemma, choices = torch.split(embedded, [1, 2], dim=1)
        lemma = self.K(lemma)
        choices = self.Q(choices)
        x = (lemma * choices).sum(-1)
        return self.critic_linear(x), x, rnn_hxs
