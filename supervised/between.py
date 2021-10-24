from __future__ import print_function

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Literal, Optional, cast, get_args

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from gql import gql
from numpy.random import Generator
from run_logger import HasuraLogger
from tap import Tap
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

from spec import spec

GPTSize = Literal["small", "medium", "large", "xl"]


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

    def forward(self, x, **_):
        return self.gpt.forward(x).last_hidden_state[:, -1]


class Net(nn.Module):
    def __init__(self, embedding_size: GPTSize, hidden_size: int, **kwargs):
        super(Net, self).__init__()
        self.embedding_size = GPT2Config.from_pretrained(
            get_gpt_size(embedding_size)
        ).n_embd
        self.net = nn.Sequential(
            GPTEmbed(embedding_size=embedding_size, **kwargs),
            nn.Linear(self.embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)


def get_gpt_size(gpt_size: GPTSize):
    gpt_size = "" if gpt_size == "small" else f"-{gpt_size}"
    gpt_size = f"gpt2{gpt_size}"
    return gpt_size


def shuffle(df: pd.DataFrame, **kwargs):
    return df.sample(frac=1, **kwargs).reset_index(drop=True)


ANTONYMS = "antonyms"
SYNONYMS = "synonyms"
LEMMA = "lemma"
TARGET = "target"
CATEGORY = "category"
WORD1 = "word1"
WORD2 = "word2"


@dataclass
class _Dataset(Dataset):
    inputs: torch.tensor
    targets: torch.tensor

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


RUN_OR_SWEEP = Literal["run", "sweep"]


class Run(Tap):
    name: str

    def configure(self) -> None:
        self.add_argument("name", type=str)  # positional


class Sweep(Tap):
    sweep_id: int = None


def configure_logger_args(args: Tap):
    args.add_subparser("run", Run)
    args.add_subparser("sweep", Sweep)


class Args(Tap):
    batch_size: int = 32
    config: Optional[str] = None  # If given, yaml config from which to load params
    data_path: str = "data.zip"
    dry_run: bool = False
    embedding_size: GPTSize = "small"
    epochs: int = 14
    gamma: float = 0.99
    graphql_endpoint: str = os.getenv("GRAPHQL_ENDPOINT")
    hidden_size: int = 512
    host_machine: str = os.getenv("HOST_MACHINE")
    load_id: int = None  # path to load parameters from if at all
    log_interval: int = 100
    log_level: str = "INFO"
    lr: float = 1.0
    max_integer: int = 20
    test_integer: int = 2
    no_cuda: bool = False
    randomize_parameters: bool = False
    save_model: bool = False
    seed: int = 1
    test_batch_size: int = 1000
    train_ln: bool = False
    train_wpe: bool = False

    def configure(self) -> None:
        self.add_subparsers(dest="logger_args")
        self.add_subparser("run", Run)
        self.add_subparser("sweep", Sweep)


class ArgsType(Args):
    logger_args: Optional[RUN_OR_SWEEP]


def get_save_path(run_id: Optional[int]):
    return (
        Path("/tmp/logs/checkpoint.pkl")
        if run_id is None
        else Path("/tmp/logs", str(run_id), "checkpoint.pkl")
    )


def train(args: Args, logger: HasuraLogger):
    # Training settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    data = np.arange(args.max_integer)
    data = np.expand_dims(data, axis=(1, 2))
    data = np.tile(data, (1, args.max_integer, args.max_integer))
    data = np.stack(
        [
            data.flatten(),
            data.swapaxes(0, 1).flatten(),
            data.swapaxes(0, 2).flatten(),
        ],
        axis=1,
    )

    c1 = data[:, 1] != data[:, 0]
    c2 = data[:, 2] != data[:, 1]
    c3 = data[:, 2] != data[:, 0]
    keep = c1 & c2 & c3
    data = data[keep]

    rng = np.random.default_rng()
    rng.shuffle(data, axis=0)

    argmin = np.argmin(data, axis=1)
    argmax = np.argmax(data, axis=1)
    extrema = np.stack([argmin, argmax], axis=1)

    indices = np.arange(3)
    indices = np.expand_dims(indices, 0)
    indices = np.tile(indices, (len(data), 1))

    extrema = np.expand_dims(extrema, 1)
    indices = np.expand_dims(indices, 2)
    not_extreme = np.all(indices != extrema, axis=2)
    targets = indices[not_extreme].flatten()

    def get_divisors():
        divisor = 1
        while divisor <= args.max_integer:
            yield divisor
            divisor *= 10

    is_test = np.dstack([(data % d) == args.test_integer for d in get_divisors()])
    is_test = is_test.any(axis=(1, 2))

    tokenizer = GPT2Tokenizer.from_pretrained(get_gpt_size(args.embedding_size))

    def tokenize():
        for n1, n2, n3 in tqdm(data, desc="Tokenizing data"):
            yield tokenizer.encode(f"{n1},{n2},{n3}", return_tensors="pt").squeeze(0)

    inputs = list(tokenize())
    inputs = pad_sequence(inputs, padding_value=tokenizer.eos_token_id).squeeze(0).T

    _is_test = torch.tensor(is_test)
    _targets = torch.tensor(targets, dtype=torch.long)
    train_dataset = _Dataset(inputs=inputs[~_is_test], targets=_targets[~_is_test])
    test_dataset = _Dataset(inputs=inputs[_is_test], targets=_targets[_is_test])

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = Net(
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        randomize_parameters=args.randomize_parameters,
        train_wpe=args.train_wpe,
        train_ln=args.train_ln,
    ).to(device)

    save_path = get_save_path(logger.run_id)
    if args.load_id is not None:
        load_path = get_save_path(args.load_id)
        logging.info(f"Loading checkpoint from {load_path}...")
        model.load_state_dict(torch.load(load_path))
    if args.save_model:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    start = time.time()
    frames = 0

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):

        model.eval()
        test_loss = 0
        correct = []
        with torch.no_grad():
            for data, target in test_loader:
                frames += len(data)
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += [pred.eq(target.view_as(pred)).squeeze(-1).float()]

        test_loss /= len(test_loader.dataset)
        test_accuracy = torch.cat(correct).mean()
        now = time.time()
        log = {
            EPOCH: epoch,
            TEST_LOSS: test_loss,
            TEST_ACCURACY: test_accuracy.item(),
            RUN_ID: logger.run_id,
            HOURS: (now - start) / 3600,
            FPS: frames / (now - start),
        }
        pprint(log)
        if logger.run_id is not None:
            logger.log(log)

        model.train()
        correct = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += [pred.eq(target.view_as(pred)).squeeze(-1).float()]
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                accuracy = torch.cat(correct).mean()
                log = {
                    EPOCH: epoch,
                    LOSS: loss.item(),
                    ACCURACY: accuracy.item(),
                    RUN_ID: logger.run_id,
                    HOURS: (time.time() - start) / 3600,
                }
                pprint(log)
                if logger.run_id is not None:
                    logger.log(log)

                if args.dry_run:
                    break

        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), str(save_path))


EXCLUDED = {
    "config",
    "name",
    "sync_envs",
    "render",
    "render_test",
    "subcommand",
    "sweep_id",
    "load_id",
    "logger_args",
}

FPS = "FPS"
GRADIENT_NORM = "gradient norm"
TIME = "time"
HOURS = "hours"
EPOCH = "epoch"
SAVE_COUNT = "save count"
LOSS = "loss"
TEST_LOSS = "test loss"
ACCURACY = "accuracy"
TEST_ACCURACY = "test accuracy"
RUN_ID = "run ID"


def update_args(args, parameters, check_hasattr=True):
    for k, v in parameters.items():
        if k not in EXCLUDED:
            if check_hasattr:
                assert hasattr(args, k), k
            setattr(args, k, v)


def main(args: ArgsType):
    logging.getLogger().setLevel(args.log_level)
    if args.config is not None:
        with Path(args.config).open() as f:
            config = yaml.load(f, yaml.FullLoader)
            args = args.from_dict(
                {k: v for k, v in config.items() if k not in EXCLUDED}
            )

    metadata = dict(reproducibility_info=args.get_reproducibility_info())
    if args.host_machine:
        metadata.update(host_machine=args.host_machine)
    if name := getattr(args, "name", None):
        metadata.update(name=name)

    logger: HasuraLogger
    with HasuraLogger(args.graphql_endpoint) as logger:
        valid = (*get_args(RUN_OR_SWEEP), None)
        assert args.logger_args in valid, f"{args.logger_args} is not in {valid}."

        if args.logger_args is not None:
            charts = [
                spec(x=x, y=y)
                for y in (
                    LOSS,
                    ACCURACY,
                    TEST_LOSS,
                    TEST_ACCURACY,
                )
                for x in (HOURS, EPOCH)
            ]
            sweep_id = getattr(args, "sweep_id", None)
            parameters = logger.create_run(
                metadata=metadata,
                sweep_id=sweep_id,
                charts=charts,
            )

            if parameters is not None:
                update_args(args, parameters)
            logger.update_metadata(
                dict(parameters=args.as_dict(), run_id=logger.run_id)
            )

        if args.load_id is not None:
            parameters = logger.execute(
                gql(
                    """
query GetParameters($id: Int!) {
  run_by_pk(id: $id) {
    metadata(path: "parameters")
  }
}"""
                ),
                variable_values=dict(id=args.load_id),
            )["run_by_pk"]["metadata"]
            update_args(args, parameters, check_hasattr=False)
        return train(args=args, logger=logger)


if __name__ == "__main__":
    main(cast(ArgsType, Args().parse_args()))
