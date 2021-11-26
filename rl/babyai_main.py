import functools
import itertools
from typing import Generator, List, Literal, Union, cast

import gym
from babyai.levels.verifier import ObjDesc, PickupInstr
from gym.wrappers import TimeLimit
import torch
from stable_baselines3.common.monitor import Monitor
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer

import main
from babyai_agent import Agent
from babyai_env import (
    ActionInObsWrapper,
    DirectionWrapper,
    DirectionsEnv,
    FullyObsWrapper,
    GoAndFaceDirections,
    GoAndFaceEnv,
    GoToLocEnv,
    GoToObjEnv,
    GoToRowEnv,
    LinearEnv,
    MissionEnumeratorWrapper,
    NegationEnv,
    NegationObject,
    PickupEnv,
    PickupRedEnv,
    PickupSynonymWrapper,
    PlantAnimalWrapper,
    RGBImgObsWithDirectionWrapper,
    RenderColorPickupEnv,
    RolloutsWrapper,
    SequenceEnv,
    SequenceParaphrasesWrapper,
    ToggleEnv,
    ZeroOneRewardWrapper,
)
from descs import CardinalDirection, LocDesc, OrdinalDirection, RowDesc
from envs import RenderWrapper, VecPyTorch
from instrs import GoToLoc, GoToRow
from utils import get_gpt_size


class Args(main.Args):
    embedding_size: Literal[
        "small", "medium", "large", "xl"
    ] = "medium"  # what size of pretrained GPT to use
    env: str = "GoToLocal"  # env ID for gym
    go_and_face_synonyms: str = None
    negation_types: str = None
    negation_colors: str = None
    num_dists: int = 1
    room_size: int = 5
    scaled_reward: bool = False
    second_layer: bool = False
    strict: bool = True
    test_colors: str = None
    train_colors: str = None
    test_descriptors: str = None
    test_number: int = None
    test_wordings: str = None
    test_walls: str = "south,southeast"
    train_wordings: str = None

    def configure(self) -> None:
        self.add_subparsers(dest="logger_args")
        main.configure_logger_args(self)


class ArgsType(main.ArgsType, Args):
    pass


class Trainer(main.Trainer):
    @classmethod
    def make_agent(cls, envs: VecPyTorch, args: ArgsType) -> Agent:
        action_space = envs.action_space
        observation_space, *_ = envs.get_attr("original_observation_space")
        missions: List[str]
        missions, *_ = envs.get_attr("missions")
        tokenizer = GPT2Tokenizer.from_pretrained(get_gpt_size(args.embedding_size))
        encoded = [tokenizer.encode(m, return_tensors="pt") for m in missions]
        encoded = [torch.squeeze(m, 0) for m in encoded]
        encoded = pad_sequence(encoded, padding_value=tokenizer.eos_token_id).T
        return cls._make_agent(
            action_space=action_space,
            observation_space=observation_space,
            encoded=encoded,
            args=args,
        )

    @classmethod
    def _make_agent(
        cls,
        encoded: torch.Tensor,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Dict,
        args: ArgsType,
    ):
        return Agent(
            action_space=action_space,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            observation_space=observation_space,
            recurrent=cls.recurrent(args),
            second_layer=args.second_layer,
            encoded=encoded,
        )

    @staticmethod
    def recurrent(args: Args):
        if "sequence" in args.env:
            assert args.recurrent
        return args.recurrent

    @classmethod
    def make_env(cls, env, allow_early_resets, render: bool = False, *args, **kwargs):
        def _thunk(
            env_id: str,
            go_and_face_synonyms: str,
            negation_colors: str,
            negation_types: str,
            num_dists: int,
            room_size: int,
            scaled_reward: bool,
            seed: int,
            strict: bool,
            test: bool,
            test_colors: str,
            train_colors: str,
            test_descriptors: str,
            test_number: int,
            test_walls: str,
            test_wordings: str,
            train_wordings: str,
            **_,
        ):
            goal_objects = (
                [("ball", "green")]
                if test
                else [
                    ("box", "green"),
                    ("box", "yellow"),
                    ("ball", "yellow"),
                ]
            )
            _kwargs = dict(
                room_size=room_size, strict=strict, num_dists=num_dists, seed=seed
            )
            missions = None
            if env_id == "go-to-obj":
                _env = GoToObjEnv(*args, goal_objects=goal_objects, **_kwargs)
                longest_mission = "go to the red ball"
            elif env_id == "toggle":
                _env = ToggleEnv(*args, **_kwargs)
                longest_mission = "toggle the red ball"
            elif env_id == "pickup":
                _env = PickupEnv(
                    *args,
                    num_dists=1,
                    goal_objects=goal_objects,
                    **_kwargs,
                )
                longest_mission = "pick up the red ball"
            elif env_id == "pickup-synonyms":
                _env = PickupRedEnv(*args, goal_objects=goal_objects, **_kwargs)
                if test:
                    _env = PickupSynonymWrapper(_env)
                longest_mission = "pick-up the crimson phone"
            elif env_id == "plant-animal":
                objects = {*PlantAnimalWrapper.replacements.keys()}
                test_objects = {
                    PlantAnimalWrapper.purple_animal,
                    PlantAnimalWrapper.black_plant,
                }
                objects = sorted(test_objects if test else objects - test_objects)
                objects = [o.split() for o in objects]
                objects = [(t, c) for (c, t) in objects]
                _env = PickupEnv(
                    *args, room_objects=objects, goal_objects=objects, **_kwargs
                )
                _env = PlantAnimalWrapper(_env)

                def missions():
                    for _, vs in _env.replacements.items():
                        for v in vs:
                            yield f"pick up the {v}"

                missions = list(missions())
            elif env_id == "directions":
                test_directions = {CardinalDirection.north}
                directions = (
                    test_directions
                    if test
                    else {*OrdinalDirection, *CardinalDirection} - test_directions
                )
                _env = DirectionsEnv(*args, directions=directions, **_kwargs)
                longest_mission = "go to northwest corner"
            elif env_id.startswith("go-and-face"):
                del kwargs["strict"]

                def parse_test_walls() -> Generator[
                    Union[CardinalDirection, OrdinalDirection], None, None
                ]:
                    for wall in test_walls.split(","):
                        try:
                            yield CardinalDirection[wall]
                        except KeyError:
                            yield OrdinalDirection[wall]

                test_directions = {
                    GoAndFaceDirections(
                        room_direction=OrdinalDirection.southeast,
                        wall_direction=d1,
                        face_direction=d2,
                    )
                    for d1 in parse_test_walls()
                    for d2 in CardinalDirection
                }

                def get_directions():
                    for d1 in OrdinalDirection:
                        for d2 in [*OrdinalDirection, *CardinalDirection]:
                            for d3 in CardinalDirection:
                                yield GoAndFaceDirections(
                                    room_direction=d1,
                                    wall_direction=d2,
                                    face_direction=d3,
                                )

                directions = set(get_directions())
                directions = test_directions if test else directions - test_directions
                _kwargs.update({k: True for k in go_and_face_synonyms.split(",")})
                _env = GoAndFaceEnv(
                    *args,
                    directions=directions,
                    synonyms="synonyms" in env_id,
                    **kwargs,
                )
                longest_mission = (
                    "go to the southeast room, go to the west wall, and face east"
                )
            elif env_id == "sequence-paraphrases":
                _env = SequenceEnv(*args, num_rows=1, num_cols=1, **_kwargs)
                _env = SequenceParaphrasesWrapper(
                    _env,
                    test=test,
                    train_wordings=train_wordings.split(","),
                    test_wordings=test_wordings.split(","),
                )
                longest_mission = "go to (0, 0), having already gone to (0, 0)"
            elif env_id == "sequence":
                _env = SequenceEnv(*args, num_rows=1, num_cols=1, **_kwargs)
                longest_mission = "go to (0, 0), then go to (0, 0)"
            elif env_id == "negation":

                colors = negation_colors.split(",")
                types = negation_types.split(",")

                objects = {(ty, col) for ty in types for col in colors}
                goal_objects = (
                    {
                        NegationObject(positive=True, type=ty, color=col)
                        for (ty, col) in objects
                    }
                    | {NegationObject(positive=False, type=ty) for ty in types}
                    | {NegationObject(positive=False, color=col) for col in colors}
                )

                def get_test_goals():
                    assert isinstance(test_descriptors, str)
                    for s in test_descriptors.split(","):
                        if s in types:
                            yield NegationObject(type=s, positive=False)
                        elif s in colors:
                            yield NegationObject(color=s, positive=False)
                        else:
                            raise RuntimeError(f"{s} is not a valid test_descriptor.")

                test_goals = set(get_test_goals())
                goal_objects = test_goals if test else goal_objects - test_goals
                _env = NegationEnv(
                    *args, goal_objects=goal_objects, room_objects=objects, **_kwargs
                )
            elif env_id == "colors":
                test_colors = test_colors.split(",")
                train_colors = train_colors.split(",")
                ball = "ball"
                train_objects = sorted({(ball, col) for col in train_colors})
                test_objects = sorted({(ball, col) for col in test_colors})
                objects = test_objects if test else train_objects
                _env = RenderColorPickupEnv(
                    *args,
                    room_objects=objects,
                    goal_objects=objects,
                    **_kwargs,
                )
            elif env_id == "go-to-loc":

                def is_test(x: int, y: int):
                    return str(test_number) in f"{x}{y}"

                def is_train(x: int, y: int):
                    return not is_test(x, y)

                filter_fn = is_test if test else is_train

                floor_indices = list(range(1, room_size - 1))
                locs = list(
                    itertools.product(
                        floor_indices,
                        floor_indices,
                    )
                )
                del _kwargs["strict"]
                _kwargs.update(num_dists=0)
                _env = GoToLocEnv(
                    locs=[l for l in locs if filter_fn(*l)],
                    scaled_reward=scaled_reward,
                    **_kwargs,
                )
                missions = [
                    GoToLoc(LocDesc(_env.grid, *loc)).surface(None) for loc in locs
                ]
            elif env_id == "go-to-row":

                def is_test(y: int):
                    return str(test_number) in str(y)

                def is_train(y: int):
                    return not is_test(y)

                filter_fn = is_test if test else is_train

                rows = list(range(1, room_size - 1))
                del _kwargs["strict"]
                _kwargs.update(num_dists=0)
                _env = GoToRowEnv(
                    rows=[r for r in rows if filter_fn(r)],
                    scaled_reward=scaled_reward,
                    **_kwargs,
                )
                missions = [
                    GoToRow(RowDesc(_env.grid, row)).surface(None) for row in rows
                ]
            elif env_id == "linear":

                def is_test(y: int):
                    return test_number is not None and str(test_number) in str(y)

                def is_train(y: int):
                    return not is_test(y)

                filter_fn = is_test if test else is_train

                rows = list(range(room_size))
                del _kwargs["strict"]
                _kwargs.update(num_dists=0)
                _env = LinearEnv(
                    locations=[l for l in rows if filter_fn(l)],
                    seed=seed,
                    size=room_size,
                    scaled_reward=scaled_reward,
                )
                _env = TimeLimit(_env, max_episode_steps=room_size + 1)
                missions = [f"{target}" for target in rows]

            else:
                raise RuntimeError(f"{env_id} is not a valid env_id")

            _env = DirectionWrapper(_env)
            if env_id == "colors":
                _env = RGBImgObsWithDirectionWrapper(_env)
            elif env_id != "linear":
                _env = FullyObsWrapper(_env)

            _env = ActionInObsWrapper(_env)
            if not (env_id in ("go-to-loc", "go-to-row", "linear") and scaled_reward):
                _env = ZeroOneRewardWrapper(_env)
            _env = MissionEnumeratorWrapper(_env, missions=missions)
            _env = RolloutsWrapper(_env)

            _env = Monitor(_env, allow_early_resets=allow_early_resets)
            if render:
                _env = RenderWrapper(_env)

            return _env

        return functools.partial(_thunk, env_id=env, **kwargs)

    # @classmethod
    # def make_vec_envs(cls, num_frame_stack=None, **kwargs):
    #     tokenizer = GPT2Tokenizer.from_pretrained(
    #         get_gpt_size(kwargs["embedding_size"])
    #     )
    #     return super().make_vec_envs(tokenizer=tokenizer, **kwargs)


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
