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


@dataclass
class Env(gym.Env):
    inputs: np.ndarray
    targets: np.ndarray
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


def get_inputs_and_targets(data: pd.DataFrame, seed: int):
    antonym = data[ANTONYM].copy().reset_index(drop=True)
    data = shuffle(data, random_state=seed)  # shuffle data
    data[NON_ANTONYM] = antonym
    # permute choices (otherwise correct answer is always 0)
    input_columns = [ANTONYM, NON_ANTONYM]
    jj, ii = np.meshgrid(np.arange(2), np.arange(len(data)))
    jj = np.random.default_rng(seed).permuted(
        jj, axis=1
    )  # shuffle indices along y-axis
    permuted_inputs = data[input_columns].to_numpy()[
        ii, jj
    ]  # shuffle data using indices
    data[input_columns] = permuted_inputs

    inputs = np.stack(
        [torch.stack(list(data[col])).numpy() for col in [LEMMA, ANTONYM, NON_ANTONYM]],
        axis=1,
    )
    targets = jj[:, 0]
    return inputs, targets


NON_ANTONYM = "non-antonyms"
ANTONYM = "antonyms"
LEMMA = "lemma"
TARGET = "target"


def shuffle(df: pd.DataFrame, **kwargs):
    return df.sample(frac=1, **kwargs).reset_index(drop=True)
