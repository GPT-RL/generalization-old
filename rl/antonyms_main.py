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
    data_path: str = "antonyms.zip"
    embedding_size: GPTSize = "medium"  # what size of pretrained GPT to use
    n_train: int = 9000
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


def explode_antonyms(data: pd.DataFrame):
    data[ANTONYM] = data.apply(
        func=lambda x: re.split("[;|]", x.antonyms),
        axis=1,
    )
    data = data.explode(ANTONYM)
    return data


def check_disjoint(lemmas: torch.Tensor, antonyms: torch.Tensor, eos: int):
    non_vocab = antonyms.max() + 1
    lemmas_ = lemmas * (lemmas == eos) * non_vocab + lemmas * (lemmas != eos)
    intersecting = lemmas_.unsqueeze(1) == antonyms.unsqueeze(2)
    intersecting = cast(torch.Tensor, intersecting)
    intersecting = intersecting.any(2).any(1)
    return ~intersecting


def isin(a: torch.Tensor, b: torch.Tensor):
    assert len(a.shape) == 2
    assert len(b.shape) == 2
    assert a.size(-1) == b.size(-1)
    equal = a.unsqueeze(1) == b.unsqueeze(0)
    equal = cast(torch.Tensor, equal)
    return equal.all(-1).any(1)


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


def get_datasets(data_path: str, seed: int, embedding_size: GPTSize, n_train: int):
    with zipfile.ZipFile(data_path) as zip_file:
        with zip_file.open("antonyms.csv") as file:
            data: pd.DataFrame = pd.read_csv(file)
    data = shuffle(data, random_state=seed)
    data = explode_antonyms(data)
    data = data.reset_index(drop=True)
    tokenizer = GPT2Tokenizer.from_pretrained(get_gpt_size(embedding_size))
    columns = [LEMMA, ANTONYM]
    with tqdm(
        desc="Encoding data", total=sum(len(data[col]) for col in columns)
    ) as bar:

        def encode(s: str):
            bar.update(1)
            return tuple(tokenizer.encode(s))

        for col in columns:
            data[col] = data[col].apply(encode)
    padded = pad_sequence(
        list(map(torch.tensor, [*data[LEMMA], *data[ANTONYM]])),
        padding_value=tokenizer.eos_token_id,
    ).T
    lemmas, antonyms = torch.split(padded, [len(data), len(data)])
    is_disjoint = check_disjoint(lemmas, antonyms, tokenizer.eos_token_id)
    lemmas = lemmas[is_disjoint]
    antonyms = antonyms[is_disjoint]
    data = data[pd.Series(is_disjoint.numpy())].reset_index(drop=True)
    vocab = padded.unique(dim=0)
    test_vocab = vocab[torch.randperm(len(vocab))][n_train:]
    lemma_is_in_test = isin(lemmas, test_vocab).numpy()
    antonym_is_in_test = isin(antonyms, test_vocab).numpy()
    add_to_test_data = lemma_is_in_test & antonym_is_in_test
    add_to_train_data = ~lemma_is_in_test & ~antonym_is_in_test
    data[LEMMA] = lemmas
    data[ANTONYM] = antonyms
    train_data = data[add_to_train_data].copy()
    test_data = data[add_to_test_data].copy()
    return test_data, train_data


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
        test_data, train_data = get_datasets(
            data_path=data_path,
            seed=seed,
            embedding_size=embedding_size,
            n_train=n_train,
        )

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
