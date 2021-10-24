import codecs
import os
import pickle
from dataclasses import dataclass, replace
from typing import List, Literal, NamedTuple

import numpy as np
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from gym_minigrid.minigrid import (
    Ball,
    Box,
    Door,
    Floor,
    Goal,
    Grid,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    Key,
    Lava,
    STATE_TO_IDX,
    Wall,
)
from gym_minigrid.window import Window
from tap import Tap
from transformers import GPT2Tokenizer

from babyai_agent import get_size
from babyai_env import Spaces
from babyai_main import Trainer
from utils import get_gpt_size

INDEX = 0
IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}


class Args(Tap):
    graphql_endpoint: str = os.getenv("GRAPHQL_ENDPOINT")
    step: int = 0
    id: int
    granularity: Literal["sweep", "run"] = "run"
    tile_size: int = 32
    test: bool = True


class InvalidGranularityError(RuntimeError):
    pass


class InvalidObjectError(RuntimeError):
    pass


class BatchTimeStep(NamedTuple):
    action: np.ndarray
    observation: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    info: List[dict]


@dataclass
class RawTimeStep:
    action: np.ndarray
    observation: np.ndarray
    reward: np.ndarray
    done: bool
    info: dict


class TimeStep(NamedTuple):
    action: str
    observation: Spaces
    reward: float
    done: bool
    info: dict


class Metadata(NamedTuple):
    embedding_size: str
    room_size: int
    num_dists: int
    strict: bool


def main(args: Args):
    transport = RequestsHTTPTransport(
        url=args.graphql_endpoint,
        # headers={"x-hasura-admin-secret": None},
    )
    client = Client(transport=transport)

    if args.granularity == "sweep":
        document = gql(
            """
query Query($id: Int) {
  run(where: {run: {sweep_id: {_eq: $id}}}) {
    embedding_size: metadata(path: "parameters.embedding_size")
    room_size: metadata(path: "parameters.room_size")
    num_dists: metadata(path: "parameters.num_dists")
    strict: metadata(path: "parameters.strict") 
  }
}
"""
        )
    elif args.granularity == "run":
        document = gql(
            """
query Query($id: Int) {
  run(where: {id: {_eq: $id}}) {
    embedding_size: metadata(path: "parameters.embedding_size")
    room_size: metadata(path: "parameters.room_size")
    num_dists: metadata(path: "parameters.num_dists")
    strict: metadata(path: "parameters.strict") 
  }
}
"""
        )
    else:
        raise InvalidGranularityError()
    metadata = client.execute(document, variable_values=dict(id=args.id))["run"][0]
    metadata = Metadata(**metadata)

    if args.granularity == "sweep":
        document = gql(
            """
query Query($id: Int) {
  run_blob(where: {run: {sweep_id: {_eq: $id}}}) {
    text
  }
}
"""
        )
    elif args.granularity == "run":
        document = gql(
            """
query Query($id: Int) {
  run_blob(where: {run_id: {_eq: $id}}) {
    text
  }
}
"""
        )
    else:
        raise InvalidGranularityError()
    blobs = client.execute(document, variable_values=dict(id=args.id))
    blobs = blobs["run_blob"]

    tokenizer = GPT2Tokenizer.from_pretrained(get_gpt_size(metadata.embedding_size))
    env = Trainer().make_env(
        env_id="",
        seed=0,
        rank=0,
        allow_early_resets=False,
        tokenizer=tokenizer,
        goal_objects=[("ball", "red")],
        room_size=metadata.room_size,
        num_dists=metadata.num_dists,
        strict=metadata.strict,
    )()
    IDX_TO_ACTIONS = {a.value: a.name for a in env.actions}
    observation_spaces = env.original_observation_space
    sections = np.cumsum([get_size(space) for space in observation_spaces])

    def get_time_steps():
        for dictionary in blobs:
            pickled = dictionary["text"]

            # https://stackoverflow.com/a/30469744/4176597
            data = pickle.loads(codecs.decode(pickled.encode(), "base64"))
            test = data["test"]
            training_step = data["step"]
            print("Training step:", training_step, "Test:", test)
            if test != args.test:
                continue
            if training_step < args.step:
                continue
            raw_time_steps = data["time_steps"]

            def get_raw_time_steps():
                for batch_step in raw_time_steps:
                    batch_step = BatchTimeStep(**batch_step)
                    yield [
                        RawTimeStep(*s) for s in zip(*batch_step)
                    ]  # Each TimeStep corresponds to a different process. All correspond to the same time-step.

            for raw_time_steps in [*zip(*get_raw_time_steps())]:
                # time_steps corresponds to a single process.
                time_step: RawTimeStep
                for time_step in raw_time_steps:
                    observation = Spaces(
                        *np.split(
                            time_step.observation,
                            sections[:-1],
                            axis=-1,
                        )
                    )
                    room_size = int((observation.image.size / 3) ** 0.5)
                    observation = replace(
                        observation,
                        image=observation.image.reshape(
                            (room_size, room_size, 3)
                        ).astype(np.uint8),
                        mission=tokenizer.decode(
                            observation.mission, skip_special_tokens=True
                        ),
                    )
                    yield TimeStep(
                        action=IDX_TO_ACTIONS[int(time_step.action)],
                        observation=observation,
                        reward=float(time_step.reward),
                        done=time_step.done,
                        info=time_step.info,
                    )

    time_steps = [*get_time_steps()]
    INDEX = len(time_steps) - 1

    def redraw():
        time_step: TimeStep = time_steps[INDEX]
        image = time_step.observation.image
        height, width, depth = image.shape
        grid = Grid(width, height)
        agent_pos = None
        agent_dir = None
        for i, row in enumerate(image):
            for j, (object_idx, color_idx, state_idx) in enumerate(row):
                obj = IDX_TO_OBJECT[object_idx]
                color = IDX_TO_COLOR[color_idx]
                state = IDX_TO_STATE.get(state_idx)
                if obj == "wall":
                    v = Wall(color)
                elif obj == "floor":
                    v = Floor(color)
                elif obj == "door":
                    v = Door(
                        color,
                        is_open=state == "open",
                        is_locked=state == "locked",
                    )
                elif obj == "key":
                    v = Key(color)
                elif obj == "ball":
                    v = Ball(color)
                elif obj == "box":
                    v = Box(color)
                elif obj == "goal":
                    v = Goal()
                elif obj == "lava":
                    v = Lava()
                elif obj == "agent":
                    agent_pos = np.array([i, j])
                    agent_dir = state_idx
                    v = None
                else:
                    v = None
                if v is not None:
                    grid.set(i, j, v)

        assert agent_pos is not None, image
        assert agent_dir is not None, image
        img = grid.render(
            args.tile_size,
            agent_pos=agent_pos,
            agent_dir=agent_dir,
        )
        window.show_img(img)
        # window.set_caption(
        #     f"mission: {time_step.observation.mission}, reward: {time_step.reward}, done: {time_step.done}, action: {time_step.action}"
        # )

        print(
            f"mission: {time_step.observation.mission}",
            f"reward: {time_step.reward}",
            f"done: {time_step.done}",
            f"action: {time_step.action}",
            sep="\n",
            end="\n\n",
        )

    def key_handler(event):
        global INDEX
        print("pressed", event.key)

        if event.key == "left" and INDEX > 0:
            INDEX = INDEX - 1
            redraw()
            return
        elif (
            event.key == "right" and INDEX < len(time_steps) - 1
        ):
            INDEX = INDEX + 1
            redraw()
            return
        elif event.key == "escape":
            window.close()
            return

    window = Window("gym_minigrid")
    window.reg_key_handler(key_handler)
    redraw()
    window.show(block=True)


if __name__ == "__main__":
    main(Args(explicit_bool=True).parse_args())
