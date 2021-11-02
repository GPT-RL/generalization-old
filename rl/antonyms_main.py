import functools
import re
import zipfile
from typing import cast

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.monitor import Monitor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import GPT2Tokenizer

import main
from antonyms.agent import Agent, Architecture, GPTSize, PRETRAINED
from antonyms.env import Env
from envs import VecPyTorch
from utils import get_gpt_size


class Args(main.Args):
    architecture: Architecture = PRETRAINED
    data_path: str = "antonyms.pkl"
    embedding_size: GPTSize = "medium"  # what size of pretrained GPT to use
    n_train: int = 4000
    train_ln: bool = False
    train_wpe: bool = False

    def configure(self) -> None:
        self.add_subparsers(dest="logger_args")
        main.configure_logger_args(self)


class ArgsType(main.ArgsType, Args):
    pass


NON_ANTONYM = "non-antonyms"
ANTONYM = "antonyms"
LEMMA = "lemma"
TARGET = "target"


def shuffle(df: pd.DataFrame, **kwargs):
    return df.sample(frac=1, **kwargs).reset_index(drop=True)


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
        [torch.stack(list(data[col])).numpy() for col in [LEMMA, *input_columns]],
        axis=1,
    )
    targets = jj[:, 0]
    return inputs, targets


class Trainer(main.Trainer):
    @classmethod
    def make_agent(cls, envs: VecPyTorch, args: ArgsType) -> Agent:
        obs_shape = envs.observation_space.shape
        action_space = envs.action_space
        return Agent(
            architecture=args.architecture,
            obs_shape=obs_shape,
            action_space=action_space,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            train_ln=args.train_ln,
            train_wpe=args.train_wpe,
        )

    @staticmethod
    def recurrent(args: Args):
        if "sequence" in args.env:
            assert args.recurrent
        return args.recurrent

    @staticmethod
    def num_evaluation_episodes():
        return 100

    # noinspection PyMethodOverriding
    @classmethod
    def make_vec_envs(
        cls,
        embedding_size: GPTSize,
        n_train: int,
        seed: int,
        test: bool,
        data_path: str,
        **kwargs,
    ):
        train_data, test_data = torch.load(data_path)

        inputs, targets = get_inputs_and_targets(
            test_data if test else train_data, seed
        )
        return super().make_vec_envs(
            seed=seed, inputs=inputs, targets=targets, test=test, **kwargs
        )

    @classmethod
    def make_env(cls, *args, **kwargs):
        def _thunk(
            inputs: torch.Tensor,
            seed: int,
            targets: torch.Tensor,
            allow_early_resets: bool,
            **_,
        ):
            env = Env(inputs=inputs, targets=targets, seed=seed)
            return Monitor(env, allow_early_resets=allow_early_resets)

        return functools.partial(_thunk, **kwargs)


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
