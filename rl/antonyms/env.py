from dataclasses import dataclass
from typing import Generator, NamedTuple, Union

import gym
import numpy as np
import pandas as pd
import torch
from gym.spaces import Box, Discrete


class TimeStep(NamedTuple):
    s: np.ndarray
    r: float
    t: bool
    i: dict


def parse_data(data: pd.DataFrame):
    tokens = data.token
    lemmas = torch.tensor(list(tokens.lemma))
    choice0 = torch.tensor(list(tokens[0]))
    choice1 = torch.tensor(list(tokens[1]))
    return choice0, choice1, lemmas


@dataclass
class Env(gym.Env):
    inputs: torch.Tensor
    targets: torch.Tensor
    seed: int

    def __post_init__(self):
        self.iterator = None
        self.random = np.random.default_rng(seed=self.seed)
        self.observation_space = Box(
            low=self.inputs.min().item(),
            high=self.inputs.max().item(),
            shape=self.inputs[0].shape,
        )
        self.action_space = Discrete(2)

    def generator(self) -> Generator[TimeStep, Union[np.ndarray, int], None]:
        idx = self.random.choice(len(self.inputs))
        obs = self.inputs[idx]
        target = self.targets[idx]
        action = yield TimeStep(obs, 0, False, {})
        success = int(action) == int(target)
        yield TimeStep(obs, int(success), True, {})

    def step(self, action):
        return self.iterator.send(action)

    def reset(self):
        self.iterator = self.generator()
        return next(self.iterator).s

    def render(self, mode="human"):
        pass


NON_ANTONYM = "non-antonyms"
ANTONYM = "antonyms"
LEMMA = "lemma"
TARGET = "target"


def shuffle(df: pd.DataFrame, **kwargs):
    return df.sample(frac=1, **kwargs).reset_index(drop=True)
