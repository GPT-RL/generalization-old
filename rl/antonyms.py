import re
from typing import Any, Generator, List, Literal, NamedTuple, Type, cast

import gym
import numpy as np
import pandas as pd
import torch
from gym.spaces import Box, Discrete
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import GPT2Config, GPT2Tokenizer


GPTSize = Literal["small", "medium", "large", "xl"]


def get_huggingface_size(gpt_size: GPTSize):
    gpt_size = "" if gpt_size == "small" else f"-{gpt_size}"
    gpt_size = f"gpt2{gpt_size}"
    return gpt_size


def get_gpt_size(gpt_size: GPTSize) -> int:
    gpt_size = get_huggingface_size(gpt_size)
    return GPT2Config.from_pretrained(gpt_size).n_embd


def get_tokenizer(gpt_size: GPTSize) -> GPT2Tokenizer:
    return GPT2Tokenizer.from_pretrained(get_huggingface_size(gpt_size))


class TimeStep(NamedTuple):
    state: Any
    reward: float
    done: bool
    info: dict


class AntonymsEnv(gym.Env):
    def __init__(self, data: pd.DataFrame, rng: np.random.Generator):
        self.data = data
        self.iterator = None
        self.rng = rng
        assert data[ANTONYM].shape == data[LEMMA].shape == data[NON_ANTONYM].shape
        (d,) = data[LEMMA][0].shape
        high = np.stack([*data[LEMMA], *data[ANTONYM], *data[NON_ANTONYM]]).max(
            initial=0
        )
        self.original_observation_space = self.observation_space = Box(
            low=0, high=high, shape=(3 * d,)
        )
        self.action_space = Discrete(2)

    def step(self, action):
        return self.iterator.send(action)

    def reset(self):
        self.iterator = self.generator()
        s, *_ = next(self.iterator)
        return s

    def render(self, mode="human"):
        pass

    def generator(self) -> Generator[TimeStep, int, None]:
        row = self.data.sample(random_state=self.rng.bit_generator)
        row = row.iloc[0]
        lemma, antonym, non_antonym, target = row
        obs = np.concatenate(
            [
                lemma,
                antonym if target else non_antonym,
                non_antonym if target else antonym,
            ]
        )
        action = yield TimeStep(obs, 0, False, {})
        yield TimeStep(obs, int(action == target), True, {})


def shuffle(df: pd.DataFrame, **kwargs):
    return df.sample(frac=1, **kwargs).reset_index(drop=True)


ANTONYM = "antonyms"
NON_ANTONYM = "non_antonyms"
LEMMA = "lemma"
TARGET = "targets"
STRING = "string"
TOKEN = "token"


def explode_antonyms(data: pd.DataFrame):
    data[ANTONYM] = data.apply(
        func=lambda x: re.split("[;|]", x.antonyms),
        axis=1,
    )
    data = data.explode(ANTONYM)
    return data


def tokenize_data(data: pd.DataFrame, tokenizer: GPT2Tokenizer):
    concatenated = pd.concat([data[LEMMA], data[ANTONYM]])
    with tqdm(desc="Encoding data", total=len(concatenated)) as bar:

        def encode(s: str):
            bar.update(1)
            tensor = tokenizer.encode(s, return_tensors="pt")
            return cast(torch.Tensor, tensor).squeeze(0)

        encoded = list(map(encode, concatenated))
        padded = pad_sequence(encoded, padding_value=tokenizer.eos_token_id).T
        n_lemma = len(data[LEMMA])
        assert 2 * n_lemma == len(padded)
        token_lemmas = list(padded[:n_lemma])
        token_antonyms = list(padded[n_lemma:])
        return token_lemmas, token_antonyms


def isin(a: torch.Tensor, b: torch.Tensor):
    assert len(a.shape) == 2
    assert len(b.shape) == 2
    assert a.size(-1) == b.size(-1)
    equal = a.unsqueeze(1) == b.unsqueeze(0)
    equal = cast(torch.Tensor, equal)
    return equal.all(-1).any(1)


def check_disjoint(data: pd.DataFrame, eos: int):
    lemmas = torch.stack(list(data[LEMMA]))
    antonyms = torch.stack(list(data[ANTONYM]))
    non_vocab = antonyms.max() + 1
    lemmas_ = lemmas * (lemmas == eos) * non_vocab + lemmas * (lemmas != eos)
    intersecting = lemmas_.unsqueeze(1) == antonyms.unsqueeze(2)
    intersecting = cast(torch.Tensor, intersecting)
    intersecting = intersecting.any(2).any(1)
    disjoint = pd.Series(~intersecting.numpy())
    return disjoint


def split_data(data: pd.DataFrame, n_test: int):
    lemmas = torch.stack(list(data[LEMMA]))
    antonyms = torch.stack(list(data[ANTONYM]))
    vocab = torch.cat([lemmas, antonyms])
    vocab = torch.unique(vocab, dim=0)
    test_vocab = vocab[:n_test]

    lemma_in_test = isin(lemmas, test_vocab)
    antonym_in_test = isin(antonyms, test_vocab)
    is_train = ~lemma_in_test & ~antonym_in_test
    is_test = lemma_in_test & antonym_in_test
    return pd.Series(is_train.numpy()), pd.Series(is_test.numpy())


def get_non_antonyms(data: pd.DataFrame, rng: np.random.Generator):
    lemmas = torch.stack(list(data[LEMMA]))
    antonyms = torch.stack(list(data[ANTONYM]))
    choices = torch.tensor(rng.uniform(size=len(lemmas)))

    unique_lemmas, indices, counts = torch.unique(
        lemmas, return_inverse=True, return_counts=True, dim=0
    )
    valid_indices = indices.unsqueeze(-1) != indices.unsqueeze(0)
    valid_indices = cast(torch.Tensor, valid_indices)
    chosen_indices = choices * (valid_indices.sum(-1) - 1)
    chosen_indices = chosen_indices.round().long()

    def _get_non_antonyms():
        for i, (valid, choice) in enumerate(zip(valid_indices, chosen_indices)):
            yield (antonyms[valid])[choice]

    return pd.Series(list(_get_non_antonyms()))


class AntonymsQNet(nn.Module):
    def __init__(
        self,
        activation_fn: Type[nn.Module],
        net_arch: List[int],
        **kwargs,
    ):
        super().__init__()
        input_size, *net_arch, output_dim = net_arch
        embed = self._build_embed(embedding_dim=input_size, **kwargs)
        self.embed = nn.Sequential(embed, nn.ReLU())
        self.K = nn.Sequential(
            *create_mlp(input_size, output_dim, net_arch, activation_fn)
        )
        self.Q = nn.Sequential(
            *create_mlp(input_size, output_dim, net_arch, activation_fn)
        )

    @staticmethod
    def _build_embed(*args, embedding_dim: int, **kwargs):
        return nn.EmbeddingBag(*args, embedding_dim=embedding_dim, **kwargs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        n = obs.size(0)
        obs = obs.reshape(n * 3, -1)
        embedded = self.embed(obs.long())
        embedded = embedded.reshape(n, 3, -1)
        lemma, choices = torch.split(embedded, [1, 2], dim=1)
        lemma = self.K(lemma)
        choices = self.Q(choices)
        return (lemma * choices).sum(-1)


class AntonymsNetwork(QNetwork):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        normalize_images: bool,
        **kwargs,
    ):
        super().__init__(
            observation_space, action_space, features_extractor, features_dim
        )
        self.q_net = self._build_q_net(**kwargs)

    @staticmethod
    def _build_q_net(**kwargs):
        return AntonymsQNet(**kwargs)


class AntonymsPolicy(DQNPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        **kwargs,
    ):
        self.kwargs = kwargs
        super().__init__(observation_space, action_space, lr_schedule)

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        net_args.update(**self.kwargs)
        return self._build_network(**net_args)

    def _build_network(self, **net_args):
        return AntonymsNetwork(**net_args)
