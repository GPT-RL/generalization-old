import functools
from typing import cast

import pandas as pd
import torch
from stable_baselines3.common.monitor import Monitor

import main
from antonyms.agent import Agent, Architecture, GPTSize, PRETRAINED
from antonyms.env import Env, parse_data
from envs import VecPyTorch


class Args(main.Args):
    architecture: Architecture = PRETRAINED
    data_path: str = "antonyms.pkl"
    embedding_size: GPTSize = "medium"  # what size of pretrained GPT to use
    n_train: int = 4000
    n_layers: int = 0
    train_ln: bool = False
    train_wpe: bool = False

    def configure(self) -> None:
        self.add_subparsers(dest="logger_args")
        main.configure_logger_args(self)


class ArgsType(main.ArgsType, Args):
    pass


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
            data_path=args.data_path,
            n_layers=args.n_layers,
            device=cls.get_device(args.cuda),
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

        data = pd.read_pickle(data_path)
        data = data[data.test] if test else data[data.train]
        choice0, choice1, lemmas = parse_data(data)
        inputs = torch.stack([lemmas, choice0, choice1], dim=1)
        targets = torch.tensor(list(data.target))

        return super().make_vec_envs(
            seed=seed, inputs=inputs, targets=targets, test=test, **kwargs
        )

    @classmethod
    def make_env(cls, *args, **kwargs):
        def _thunk(
            inputs: torch.tensor,
            targets: torch.tensor,
            seed: int,
            allow_early_resets: bool,
            **_,
        ):
            env = Env(inputs=inputs, targets=targets, seed=seed)
            return Monitor(env, allow_early_resets=allow_early_resets)

        return functools.partial(_thunk, **kwargs)


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
