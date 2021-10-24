import abc
import re
import typing
from dataclasses import dataclass
from enum import Enum, auto
from typing import List

import numpy as np
from babyai.levels.verifier import ObjDesc
from gym_minigrid.minigrid import WorldObj
from gym_minigrid.roomgrid import RoomGrid

TYPES = ["key", "ball", "box"]


class CardinalDirection(Enum):
    east = auto()
    south = auto()
    west = auto()
    north = auto()


class OrdinalDirection(Enum):
    northeast = auto()
    northwest = auto()
    southeast = auto()
    southwest = auto()


DIRECTION_SYNONYM = {
    OrdinalDirection.northeast: ["upper right", "top right"],
    OrdinalDirection.northwest: ["upper left", "top left"],
    OrdinalDirection.southeast: ["lower right", "bottom right"],
    OrdinalDirection.southwest: ["lower left", "bottom left"],
    CardinalDirection.east: ["right"],
    CardinalDirection.south: ["lower", "bottom"],
    CardinalDirection.west: ["left"],
    CardinalDirection.north: ["upper", "top"],
}

OPPOSITES = {
    OrdinalDirection.northeast: OrdinalDirection.southwest,
    OrdinalDirection.northwest: OrdinalDirection.southeast,
    CardinalDirection.east: CardinalDirection.west,
    CardinalDirection.south: CardinalDirection.north,
}


OPPOSITES.update({v: k for k, v in OPPOSITES.items()})


class Desc:
    def surface(self):
        """
        Generate a natural language representation of the object description
        """
        return repr(self)

    @staticmethod
    def find_matching_objs(
        *args, **kwargs
    ) -> typing.Tuple[List[WorldObj], List[WorldObj]]:
        return [], []


@dataclass
class RandomDesc(Desc):
    direction: typing.Union[OrdinalDirection, CardinalDirection]
    random: typing.Optional[np.random.RandomState]

    def __post_init__(self):
        self.choices_list = list(self.choices())
        self.repr = self.sample_repr()

    @abc.abstractmethod
    def choices(self) -> typing.Generator[str, None, None]:
        raise NotImplementedError

    def __repr__(self):
        return self.repr

    def sample_repr(self):
        if self.random is None:
            return self.choices_list[0]
        return self.choices_list[self.random.choice(len(self.choices_list))]


ADJACENCIES = {
    OrdinalDirection.northeast: [
        "to the right of the northwest",
        "above the southeast",
    ],
    OrdinalDirection.northwest: ["to the left of the northeast", "above the southwest"],
    OrdinalDirection.southeast: [
        "to the right of the southwest",
        "below the northeast",
    ],
    OrdinalDirection.southwest: [
        "to the left of the southeast",
        "below the northwest",
    ],
}


class NegativeObjDesc(ObjDesc):
    def surface(self, env):
        """
        Generate a natural language representation of the object description
        """

        self.find_matching_objs(env)
        assert len(self.obj_set) > 0, "no object matching description"

        def indefinite_article(noun: str):
            article = "an" if re.match(r"[aeiou].*", noun) else "a"
            return article + " " + noun

        s = "object"
        if self.type is not None and (self.color, self.loc) == (None, None):
            what_it_is_not = indefinite_article(self.type)
        elif self.color and (self.type, self.loc) == (None, None):
            what_it_is_not = self.color
        elif self.loc and (self.color, self.type) == (None, None):
            if self.loc == "front":
                what_it_is_not = " in front of you"
            elif self.loc == "behind":
                what_it_is_not = " behind you"
            else:
                what_it_is_not = " on your " + self.loc
        elif self.type and self.color and not self.loc:
            what_it_is_not = indefinite_article(self.color + " " + s)
        else:
            raise RuntimeError(
                f"Invalid specification: {self.type}, {self.color}, {self.loc}"
            )

        # Singular vs plural
        if len(self.obj_set) > 1:
            s = indefinite_article(s) + " that is not " + what_it_is_not
        else:
            s = "the " + s + " that is not " + what_it_is_not

        return s

    def find_matching_objs(self, env, use_location=True):
        """
        Find the set of objects matching the description and their positions.
        When use_location is False, we only update the positions of already tracked objects, without taking into account
        the location of the object. e.g. A ball that was on "your right" initially will still be tracked as being "on
        your right" when you move.
        """
        if self.obj_set and self.obj_poss:
            return self.obj_set, self.obj_poss
        _, negative_poss = super().find_matching_objs(env, use_location=use_location)
        negative_poss = set(negative_poss)
        self.obj_set = []
        self.obj_poss = []

        for i in range(env.grid.width):
            for j in range(env.grid.height):
                if (i, j) not in negative_poss:
                    cell = env.grid.get(i, j)
                    if cell and cell.type in TYPES:
                        self.obj_set.append(cell)
                        self.obj_poss.append((i, j))

        return self.obj_set, self.obj_poss


@dataclass
class RoomDesc(RandomDesc):
    direction: OrdinalDirection
    opposite_synonyms: bool = False
    opposites: bool = False
    adjacencies: bool = False
    synonyms: bool = False

    # For some reason this is necessary...
    def __repr__(self):
        return super().__repr__()

    def choices(self):
        yield f"the {self.direction.name} room"
        if self.adjacencies:
            for adjacent_to in ADJACENCIES[self.direction]:
                yield f"the room {adjacent_to} room"
        if self.synonyms:
            for synonym in DIRECTION_SYNONYM[self.direction]:
                yield f"the room in the {synonym}"
        opposite = OPPOSITES[self.direction]
        if self.opposites:
            yield f"the room opposite from the {opposite.name}"
        if self.opposite_synonyms:
            for synonym in DIRECTION_SYNONYM[opposite]:
                yield f"the room opposite from the {synonym}"
                yield f"the room furthest from the {synonym}"


@dataclass
class WallDesc(RandomDesc):
    direction: CardinalDirection
    opposite_synonyms: bool = False
    opposites: bool = False
    synonyms: bool = False

    def choices(self):
        yield f"the {self.direction.name} wall"
        if self.synonyms:
            for synonym in DIRECTION_SYNONYM[self.direction]:
                yield f"the {synonym} wall"
        opposite = OPPOSITES[self.direction]
        if self.opposites:
            yield f"the wall opposite from the {opposite.name}"
        if self.opposite_synonyms:
            for synonym in DIRECTION_SYNONYM[opposite]:
                yield f"the wall opposite from the {synonym} wall"
                # yield f"the wall furthest from the {synonym} wall"


@dataclass
class CornerDesc(RandomDesc):
    direction: OrdinalDirection
    opposite_synonyms: bool = False
    opposites: bool = False
    adjacencies: bool = False
    synonyms: bool = False

    def choices(self):
        yield f"the {self.direction.name} corner"
        if self.synonyms:
            for synonym in DIRECTION_SYNONYM[self.direction]:
                yield f"the {synonym} corner"
        if self.adjacencies:
            for adjacent_to in ADJACENCIES[self.direction]:
                yield f"the corner {adjacent_to} corner"
        opposite = OPPOSITES[self.direction]
        if self.opposites:
            yield f"the corner opposite from the {opposite.name}"
        if self.opposite_synonyms:
            for synonym in DIRECTION_SYNONYM[opposite]:
                yield f"the corner opposite from the {synonym}"
                # yield f"the corner furthest from the {synonym}"


@dataclass
class FaceDesc(RandomDesc):
    direction_synonyms = {
        CardinalDirection.east: "right",
        CardinalDirection.south: "down",
        CardinalDirection.west: "left",
        CardinalDirection.north: "up",
    }
    direction: CardinalDirection
    opposites: bool = False
    synonyms: bool = False

    def choices(self):
        yield f"face {self.direction.name}"
        if self.opposites:
            opposite = OPPOSITES[self.direction]
            yield f"face away from the {opposite.name}"
        if self.synonyms:
            yield f"look {self.direction_synonyms[self.direction]}"


class LocDesc(Desc):
    """
    Description of a set of objects in an environment
    """

    def __init__(self, grid: RoomGrid, x: int, y: int):
        assert 1 <= x < grid.width - 1
        assert 1 <= y < grid.height - 1
        self.i = x
        self.j = y

    def __repr__(self):
        return f"({self.i}, {self.j})"

    @property
    def array(self):
        return np.array([self.i, self.j])

    @staticmethod
    def find_matching_objs(*args, **kwargs):
        return [], []


class RowDesc(Desc):
    """
    Description of a set of objects in an environment
    """

    def __init__(self, grid: RoomGrid, y: int):
        assert 1 <= y < grid.width - 1
        self.y = y

    def __repr__(self):
        return f"row {self.y}"

    @staticmethod
    def find_matching_objs(*args, **kwargs):
        return [], []
