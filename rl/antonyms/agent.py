from typing import Callable, Literal, cast, get_args

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
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
        assert architecture in get_args(Architecture)
        super().__init__(obs_shape, action_space, architecture=architecture, **kwargs)
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
        data_path: str,
    ):
        super().__init__()
        self.gpt = build_gpt(embedding_size, randomize_parameters)
        for name, p in self.gpt.named_parameters():
            requires_grad = (train_wpe and "wpe" in name) or (train_ln and "ln" in name)
            p.requires_grad_(requires_grad)
        self.frozen_gpt = not (train_ln or train_wpe)
        if self.frozen_gpt:
            (train_inputs, _), (test_inputs, _) = torch.load(data_path)
            inputs = np.concatenate([train_inputs, test_inputs], axis=0)
            self.register_buffer("inputs", torch.tensor(inputs))
            outputs = [
                self._forward(batch)
                for batch in tqdm(
                    torch.split(self.inputs, 40, dim=0),
                    desc="Generating frozen embedding...",
                )
            ]
            self.register_buffer("outputs", torch.cat(outputs, dim=0))

    def _forward(self, x: torch.Tensor):
        return self.gpt.forward(x.long()).last_hidden_state[:, :, -1]

    def forward(self, x: torch.Tensor):
        if self.frozen_gpt:
            equals = self.inputs.unsqueeze(1) == x.unsqueeze(0)
            equals = cast(torch.Tensor, equals)
            matches = equals.all(-1).all(-1)
            inputs_indices, x_indices = matches.nonzero().T
            _, x_indices = torch.sort(x_indices)
            return self.outputs[inputs_indices[x_indices]]
        return self._forward(x)


class BaselineEmbed(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed = nn.EmbeddingBag(
            num_embeddings=vocab_size, embedding_dim=hidden_size
        )

    def forward(self, x: torch.Tensor):
        reshape = x.long().reshape(-1, x.size(-1))
        embedded = self.embed(reshape)
        return embedded.reshape(x.size(0), x.size(1), -1)


class Lambda(nn.Module):
    def f(self, f: Callable[[torch.Tensor], torch.Tensor]):
        self.f = f

    def forward(self, x):
        return self.f(x)


class Base(NNBase):
    def __init__(
        self,
        embedding_size: GPTSize,
        hidden_size: int,
        architecture: Architecture,
        **kwargs
    ):
        super().__init__(
            recurrent=False,
            recurrent_input_size=hidden_size,
            hidden_size=hidden_size,
        )
        config = GPT2Config.from_pretrained(get_gpt_size(embedding_size))
        self.K = nn.Linear(hidden_size, hidden_size)
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.emb = (
            BaselineEmbed(vocab_size=config.vocab_size, hidden_size=hidden_size)
            if architecture == BASELINE
            else nn.Sequential(
                GPTEmbed(
                    embedding_size=embedding_size,
                    randomize_parameters=architecture == RANDOMIZED,
                    **kwargs
                ),
                nn.Linear(config.n_embd, hidden_size),
            )
        )
        self.critic_linear = nn.Linear(2 * hidden_size, 1)

    def forward(self, inputs, rnn_hxs, masks):
        embedded = torch.relu(self.emb(inputs))
        n_classes = inputs.size(1) - 1
        lemma, choices = torch.split(embedded, [1, n_classes], dim=1)
        lemma = self.K(lemma)
        choices = self.Q(choices)
        hidden = lemma * choices
        weights = hidden.sum(-1)
        return self.critic_linear(hidden.reshape(inputs.size(0), -1)), weights, rnn_hxs
