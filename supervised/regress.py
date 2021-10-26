from __future__ import print_function

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Literal, Optional, cast, get_args

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from gql import gql
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


class Lambda(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


BASELINE = "baseline"
RANDOMIZED = "randomized"
PRETRAINED = "pretrained"
ARCHITECTURE = Literal[RANDOMIZED, PRETRAINED, BASELINE]


class GPTEmbed(nn.Module):
    def __init__(
        self,
        embedding_size: GPTSize,
        architecture: ARCHITECTURE,
        train_wpe: bool,
        train_ln: bool,
        inputs: torch.Tensor,
    ):
        super().__init__()
        gpt = build_gpt(embedding_size, architecture == RANDOMIZED)
        for name, p in gpt.named_parameters():
            requires_grad = (train_wpe and "wpe" in name) or (train_ln and "ln" in name)
            p.requires_grad_(requires_grad)

        gpt = nn.Sequential(
            Lambda(lambda x: x.long()),
            gpt,
            Lambda(lambda x: x.last_hidden_state[:, -1]),
        )
        if (train_ln or train_wpe) and (architecture in [RANDOMIZED, PRETRAINED]):
            self.net = gpt
        else:
            num_embeddings = int(inputs.max() + 1)
            dummy_tokens = torch.arange(num_embeddings).unsqueeze(-1)
            embeddings = gpt(dummy_tokens)
            self.net = nn.Sequential(
                Lambda(lambda x: x.long()),
                nn.Embedding(num_embeddings, int(embeddings.size(1)))
                if architecture == BASELINE
                else nn.Embedding.from_pretrained(embeddings),
                Lambda(lambda x: x[:, -1]),
            )

    def forward(self, x, **_):
        return self.net(x)


class Net(nn.Module):
    def __init__(
        self,
        embedding_size: GPTSize,
        hidden_size: int,
        max_int: int,
        n_layers: int,
        inputs,
        **kwargs,
    ):
        super(Net, self).__init__()
        self.max_int = max_int
        self.size_goal = inputs.size(-1)
        self.embedding_size = GPT2Config.from_pretrained(
            get_gpt_size(embedding_size)
        ).n_embd
        # gpt = GPTEmbed(embedding_size=embedding_size, inputs=inputs, **kwargs)
        linearity = nn.Linear(max_int, self.embedding_size, bias=False)
        nn.init.normal_(linearity.weight)

        self.embedding1 = nn.Sequential(
            linearity,
            nn.Linear(self.embedding_size, hidden_size),
        )
        self.embedding2 = nn.Sequential(
            # gpt,
            nn.Linear(self.size_goal, hidden_size)
        )
        self.net = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.ReLU(),
            *[
                nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
                for _ in range(n_layers)
            ],
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        x1, x2 = torch.split(x, [self.max_int, self.size_goal], dim=-1)
        x = self.embedding1(x1) * self.embedding2(x2)

        return self.net(x).squeeze(-1)


def get_gpt_size(gpt_size: GPTSize):
    gpt_size = "" if gpt_size == "small" else f"-{gpt_size}"
    gpt_size = f"gpt2{gpt_size}"
    return gpt_size


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
    discount: Optional[float] = 0.9
    dry_run: bool = False
    embedding_size: GPTSize = "small"
    epochs: int = 14
    gamma: float = 0.99
    graphql_endpoint: str = os.getenv("GRAPHQL_ENDPOINT")
    hidden_size: int = 512
    host_machine: str = os.getenv("HOST_MACHINE")
    load_id: int = None  # path to load parameters from if at all
    log_interval: int = 5
    log_level: str = "INFO"
    lr: float = 1.0
    max_integer: int = 20
    n_layers: int = 1
    no_cuda: bool = False
    architecture: ARCHITECTURE = PRETRAINED
    save_model: bool = False
    seed: int = 1
    temp: float = 35
    test_batch_size: int = 1000
    test_integer: int = 2
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

    obs = torch.tile(torch.eye(args.max_integer), (args.max_integer, 1))
    raw_goal = torch.arange(args.max_integer)
    goal = torch.repeat_interleave(raw_goal, args.max_integer)
    targets = goal.float()
    # if args.discount is not None:
    #     targets = args.discount ** targets
    is_train = [str(args.test_integer) not in str(g) for g in goal]
    is_train = torch.tensor(is_train)
    is_test = cast(torch.Tensor, goal == args.test_integer)
    test_code = 1
    train_code = 2
    dataset = cast(torch.Tensor, test_code * is_test + train_code * is_train)

    def to_binary(g):
        for n in tqdm(g, desc="Tokenizing data"):
            yield torch.tensor(list(map(int, "{0:b}".format(n.item())))).flip(-1)

    raw_binary = list(to_binary(raw_goal))
    raw_binary = pad_sequence(raw_binary, padding_value=0).T.flip(-1)
    binary = torch.repeat_interleave(raw_binary, args.max_integer, dim=0)

    data = torch.stack([targets, dataset], dim=1)
    data = torch.cat([obs, binary, data], dim=1)
    raw_inputs = data[:, : args.max_integer + raw_binary.size(-1)]
    raw_inputs = raw_inputs.to(device)
    raw_targets = targets.to(device)
    raw_dataset = dataset.to(device)
    raw_goal = torch.sort(goal.to(device)).values

    def repeat_data(in_dataset, batch_size):
        tiles = int(torch.ceil(batch_size / sum(in_dataset)))
        return torch.tile(data[in_dataset], (tiles, 1))

    data = torch.cat(
        [
            repeat_data(raw_dataset == train_code, args.batch_size),
            repeat_data(raw_dataset == test_code, args.test_batch_size),
        ],
        dim=0,
    )

    torch.manual_seed(args.seed)
    data = data[torch.randperm(len(data))]

    inputs, targets, dataset = torch.split(
        data, [args.max_integer + raw_binary.size(-1), 1, 1], dim=-1
    )
    dataset = dataset.flatten()

    train_dataset = _Dataset(
        inputs=inputs[dataset == train_code], targets=targets[dataset == train_code]
    )
    test_dataset = _Dataset(
        inputs=inputs[dataset == test_code], targets=targets[dataset == test_code]
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = Net(
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        architecture=args.architecture,
        train_wpe=args.train_wpe,
        train_ln=args.train_ln,
        max_int=args.max_integer,
        inputs=raw_binary,
        n_layers=args.n_layers,
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

    save_count = 0

    def get_metric(x: torch.Tensor):
        with torch.no_grad():
            return torch.mean(x.float()).item()

    def get_accuracy(is_dataset: torch.Tensor, raw_outputs: torch.Tensor):
        distances = torch.abs(raw_outputs - raw_targets.unsqueeze(-1))
        correct_target = raw_targets[distances.argmin(0)] == raw_targets
        return get_metric(correct_target[is_dataset])

    def get_goal_dependent_stuff(is_dataset: torch.Tensor, raw_outputs: torch.Tensor):
        def get_expected_return(values: torch.Tensor):
            v = F.pad(values, (1, 1), value=-float("inf"))

            triples = torch.stack([v[:-2], v[1:-1], v[2:]]).T
            pi = torch.softmax(cast(torch.Tensor, triples * args.temp), dim=-1)
            T = (
                torch.diag(pi[1:, 0], -1)
                + torch.diag(pi[:, 1])
                + torch.diag(pi[:-1, 2], 1)
            )
            return torch.matrix_power(T, 2 * args.max_integer).mean(0)

        unique_goals, goals_count = raw_goal[is_dataset].unique(return_counts=True)
        out = torch.split(raw_outputs[is_dataset], list(goals_count))
        tgt = torch.split(raw_targets[is_dataset], list(goals_count))

        expected, optimal, correct_max = torch.tensor(
            [
                (
                    get_expected_return(o)[g].item(),
                    get_expected_return(t)[g].item(),
                    o.argmax() == t.argmax(),
                )
                for g, o, t in zip(unique_goals, out, tgt)
            ]
        ).T

        return (
            get_metric(expected),
            get_metric(optimal - expected),
            get_metric(correct_max),
        )

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):

        log_epoch = epoch % args.log_interval == 0

        if log_epoch:
            test_loss = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += F.mse_loss(output.flatten(), target.flatten()).item()

            test_output = model(raw_inputs)
            test_return_, test_regret, test_correct_max = get_goal_dependent_stuff(
                raw_dataset == test_code, test_output
            )
            log = {
                EPOCH: epoch,
                TEST_LOSS: test_loss,
                TEST_ACCURACY: get_accuracy(raw_dataset == test_code, test_output),
                TEST_CORRECT_MAX: test_correct_max,
                TEST_EXPECTED_REGRET: test_regret,
                TEST_EXPECTED_RETURN: test_return_,
                RUN_ID: logger.run_id,
                HOURS: (time.time() - start) / 3600,
            }
            pprint(log)
            if logger.run_id is not None:
                logger.log(log)

        frames = 0
        tick = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            frames += len(data)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output.flatten(), target.flatten())
            loss.backward()
            optimizer.step()
            if batch_idx == 0 and log_epoch:
                raw_output = model(raw_inputs)
                return_, regret, correct_max = get_goal_dependent_stuff(
                    raw_dataset == train_code, raw_output
                )
                log = {
                    EPOCH: epoch,
                    LOSS: loss.item(),
                    RUN_ID: logger.run_id,
                    HOURS: (time.time() - start) / 3600,
                    ACCURACY: get_accuracy(raw_dataset == train_code, raw_output),
                    CORRECT_MAX: correct_max,
                    EXPECTED_REGRET: regret,
                    EXPECTED_RETURN: return_,
                    SAVE_COUNT: save_count,
                }
                pprint(log)
                if logger.run_id is not None:
                    logger.log(log)

                if args.dry_run:
                    break

        if log_epoch:
            now = time.time()
            log = {
                RUN_ID: logger.run_id,
                EPOCH: epoch,
                HOURS: (now - start) / 3600,
                FPS: frames / (now - tick),
            }
            pprint(log)
            if logger.run_id is not None:
                logger.log(log)
        scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), str(save_path))
            save_count += 1


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
EXPECTED_REGRET = "expected regret"
TEST_EXPECTED_REGRET = "test expected regret"
EXPECTED_RETURN = "expected return"
TEST_EXPECTED_RETURN = "test expected return"
CORRECT_MAX = "correct max"
TEST_CORRECT_MAX = "test correct max"
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
                spec(x=EPOCH, y=y, scale_type="log" if LOSS in y else "linear")
                for y in (
                    LOSS,
                    TEST_LOSS,
                    ACCURACY,
                    TEST_ACCURACY,
                    EXPECTED_REGRET,
                    TEST_EXPECTED_REGRET,
                    EXPECTED_RETURN,
                    TEST_EXPECTED_RETURN,
                    CORRECT_MAX,
                    TEST_CORRECT_MAX,
                    SAVE_COUNT,
                    FPS,
                )
            ] + [
                spec(x=HOURS, y=y, scale_type="log")
                for y in (
                    LOSS,
                    TEST_LOSS,
                )
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
