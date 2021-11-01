from typing import Literal

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

import agent
from agent import NNBase
from distributions import FixedCategorical
from utils import get_gpt_size

GPTSize = Literal["small", "medium", "large", "xl"]
PRETRAINED = "pretrained"
RANDOMIZED = "randomized"
BASELINE = "baseline"
# noinspection PyTypeHints
Architecture = Literal[PRETRAINED, RANDOMIZED, BASELINE]


class Categorical(nn.Module):
    @staticmethod
    def forward(x):
        return FixedCategorical(logits=x)


class Agent(agent.Agent):
    def __init__(self, obs_shape, action_space, architecture: Architecture, **kwargs):
        if architecture == BASELINE:
            raise NotImplementedError()
        super().__init__(
            obs_shape,
            action_space,
            randomize_parameters=architecture == RANDOMIZED,
            **kwargs
        )
        self.dist = Categorical()

    def build_base(self, obs_shape, **kwargs):
        return Base(**kwargs)


def build_gpt(gpt_size: GPTSize, randomize_parameters: bool):
    gpt_size = get_gpt_size(gpt_size)
    return (
        GPT2Model(
            GPT2Config.from_pretrained(
                gpt_size,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
        )
        if randomize_parameters
        else GPT2Model.from_pretrained(
            gpt_size,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
    )


class GPTEmbed(nn.Module):
    def __init__(
        self,
        embedding_size: GPTSize,
        randomize_parameters: bool,
        train_wpe: bool,
        train_ln: bool,
    ):
        super().__init__()
        self.gpt = build_gpt(embedding_size, randomize_parameters)
        for name, p in self.gpt.named_parameters():
            requires_grad = (train_wpe and "wpe" in name) or (train_ln and "ln" in name)
            p.requires_grad_(requires_grad)

    def forward(self, x: torch.Tensor):
        return self.gpt.forward(x.long()).last_hidden_state[:, :, -1]


class Base(NNBase):
    def __init__(self, embedding_size: GPTSize, hidden_size: int, **kwargs):
        super().__init__(
            recurrent=False,
            recurrent_input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.embedding_size = GPT2Config.from_pretrained(
            get_gpt_size(embedding_size)
        ).n_embd
        self.K = nn.Linear(hidden_size, hidden_size)
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.emb = nn.Sequential(
            GPTEmbed(embedding_size=embedding_size, **kwargs),
            nn.Linear(self.embedding_size, hidden_size),
            nn.ReLU(),
        )
        self.critic_linear = nn.Linear(2 * hidden_size, 1)

    def forward(self, inputs, rnn_hxs, masks):
        embedded = self.emb(inputs)
        n_classes = inputs.size(1) - 1
        lemma, choices = torch.split(embedded, [1, n_classes], dim=1)
        lemma = self.K(lemma)
        choices = self.Q(choices)
        hidden = lemma * choices
        weights = hidden.sum(-1)
        return self.critic_linear(hidden.reshape(inputs.size(0), -1)), weights, rnn_hxs
