import abc
from abc import ABC

import numpy as np
from babyai.levels.verifier import ActionInstr, AndInstr
from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.roomgrid import Room, RoomGrid

from descs import (
    CardinalDirection,
    LocDesc,
    OrdinalDirection,
    RandomDesc,
    RoomDesc,
    RowDesc,
)


class GoToRoomInstr(ActionInstr):
    """
    Pick up an object matching a given description
    eg: pick up the grey ball
    """

    def __init__(self, desc: RoomDesc, strict=False):
        super().__init__()
        self.desc = desc
        self.strict = strict
        self.room_positions = {
            OrdinalDirection.northwest: (0, 0),
            OrdinalDirection.northeast: (1, 0),
            OrdinalDirection.southwest: (0, 1),
            OrdinalDirection.southeast: (1, 1),
        }

    def surface(self, env):
        return "go to " + self.desc.surface()

    def verify_action(self, action):
        self.env: RoomGrid
        goal_room: Room = self.env.get_room(*self.room_positions[self.desc.direction])
        return "success" if goal_room.pos_inside(*self.env.agent_pos) else "continue"


def get_limits(room: Room):
    (topX, topY), (sizeX, sizeY) = room.top, room.size
    minX = topX + 1
    maxX = topX + sizeX - 2
    minY = topY + 1
    maxY = topY + sizeY - 2
    return maxX, maxY, minX, minY


class RandomInstr(ActionInstr, ABC):
    def __init__(self, desc: RandomDesc):
        super().__init__()
        self.desc = desc
        self.random_desc_surface = self.sample_desc_surface()

    @abc.abstractmethod
    def surface(self, env) -> str:
        raise NotImplementedError

    def sample_desc_surface(self):
        return self.desc.sample_repr()

    def reset_verifier(self, env):
        """
        This gets called at the beginning of each episode.
        """
        super().reset_verifier(env)
        self.random_desc_surface = self.sample_desc_surface()


class GoToCornerInstr(RandomInstr):
    """
    Pick up an object matching a given description
    eg: pick up the grey ball
    """

    def __init__(self, *args, strict=False, **kwargs):
        self.strict = strict
        super().__init__(*args, **kwargs)

    def surface(self, env):
        return "go to " + self.random_desc_surface

    def verify_action(self, action):
        """
        This gets called each timestep.
        """
        self.env: RoomGrid
        x, y = self.env.agent_pos
        room: Room = self.env.room_from_pos(x, y)
        maxX, maxY, minX, minY = get_limits(room)
        direction = self.desc.direction
        if not action == self.env.actions.done:
            return "continue"

        def validate_direction(positional_direction: OrdinalDirection):
            # print(direction, positional_direction)
            if direction == positional_direction:
                return "success"
            elif self.strict:
                return "failure"
            else:
                return "continue"

        # print(direction)
        # print("x,y", x, y)
        # print("maxX", maxX)
        # print("minX", minX)
        # print("maxY", maxY)
        # print("minY", minY)

        if (x, y) == (minX, minY):
            return validate_direction(OrdinalDirection.northwest)
        if (x, y) == (maxX, minY):
            return validate_direction(OrdinalDirection.northeast)
        if (x, y) == (minX, maxY):
            return validate_direction(OrdinalDirection.southwest)
        if (x, y) == (maxX, maxY):
            return validate_direction(OrdinalDirection.southeast)

        return "continue"


class GoToWallInstr(RandomInstr):
    """
    Pick up an object matching a given description
    eg: pick up the grey ball
    """

    def __init__(self, *args, strict=False, **kwargs):
        self.strict = strict
        super().__init__(*args, **kwargs)

    def surface(self, env):
        return "go to " + self.random_desc_surface

    def verify_action(self, action):
        self.env: RoomGrid
        x, y = self.env.agent_pos
        room: Room = self.env.room_from_pos(x, y)
        maxX, maxY, minX, minY = get_limits(room)
        direction = self.desc.direction
        if not action == self.env.actions.done:
            return "continue"

        def validate_direction(positional_direction: CardinalDirection):
            # print(direction, positional_direction)
            if direction == positional_direction:
                return "success"
            elif self.strict:
                return "failure"
            else:
                return "continue"

        # print(direction)
        # print("x,y", x, y)
        # print("maxX", maxX)
        # print("minX", minX)
        # print("maxY", maxY)
        # print("minY", minY)

        validations = []

        if y == minY:
            validations.append(validate_direction(CardinalDirection.north))
        if y == maxY:
            validations.append(validate_direction(CardinalDirection.south))
        if x == minX:
            validations.append(validate_direction(CardinalDirection.west))
        if x == maxX:
            validations.append(validate_direction(CardinalDirection.east))

        for result in ["success", "failure"]:
            if result in validations:
                return result

        return "continue"


class FaceInstr(RandomInstr):
    """
    Pick up an object matching a given description
    eg: pick up the grey ball
    """

    def surface(self, env):
        return self.random_desc_surface

    def verify_action(self, action):
        self.env: MiniGridEnv
        return (
            "success"
            if self.env.agent_dir == [*CardinalDirection].index(self.desc.direction)
            else "continue"
        )


class ToggleInstr(ActionInstr):
    """
    Pick up an object matching a given description
    eg: pick up the grey ball
    """

    def __init__(self, obj_desc, strict=False):
        super().__init__()
        self.desc = obj_desc
        self.strict = strict

    def surface(self, env):
        return "toggle " + self.desc.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)
        self.desc.find_matching_objs(env)

    def verify_action(self, action):
        # Only verify when the pickup action is performed
        if action != self.env.actions.toggle:
            return "continue"

        # For each object position
        for pos in self.desc.obj_poss:
            # If the agent is next to (and facing) the object
            if np.array_equal(pos, self.env.front_pos):
                return "success"

        if self.strict:
            return "failure"  # not allowed to toggle except in front of correct object

        return "continue"


class GoToLoc(ActionInstr):
    """
    Pick up an object matching a given description
    eg: pick up the grey ball
    """

    def __init__(self, loc_desc: LocDesc):
        self.desc = loc_desc
        super().__init__()

    def surface(self, env):
        return "go to " + self.desc.surface()

    def reset_verifier(self, env: MiniGridEnv):
        super().reset_verifier(env)

    def verify_action(self):
        raise NotImplementedError

    def verify(self, action):
        if not action == self.env.actions.done:
            return "continue"
        if np.array_equal(self.env.agent_pos, self.desc.array):
            return "success"

        return "failure"


class GoToRow(ActionInstr):
    """
    Pick up an object matching a given description
    eg: pick up the grey ball
    """

    def __init__(self, loc_desc: RowDesc):
        self.desc = loc_desc
        super().__init__()

    def surface(self, env):
        return "go to " + self.desc.surface()

    def reset_verifier(self, env: MiniGridEnv):
        super().reset_verifier(env)

    def verify_action(self):
        raise NotImplementedError

    def verify(self, action):
        if not action == self.env.actions.done:
            return "continue"
        _, y = self.env.agent_pos
        if y == self.desc.y:
            return "success"

        return "failure"


class AndDoneInstr(AndInstr):
    def verify(self, action):
        if action is self.env.actions.done:
            a_done = self.instr_a.verify(action)
            b_done = self.instr_b.verify(action)
            return (
                "success"
                if (a_done == "success" and b_done == "success")
                else "failure"
            )
        return "continue"


class MultiAndInstr(ActionInstr):
    def __init__(self, *instrs: ActionInstr):
        self.instructions = *self.tail, self.head = instrs
        super().__init__()

    def surface(self, env):
        surfaces = [i.surface(env) for i in self.tail]
        return f"{', '.join(surfaces)}, and {self.head.surface(env)}"

    def reset_verifier(self, env: MiniGridEnv):
        for i in self.instructions:
            i.reset_verifier(env)
        super().reset_verifier(env)

    def verify_action(self, *args, **kwargs):
        verifications = [i.verify(*args, **kwargs) for i in self.instructions]
        # for i, v in zip(self.instructions, verifications):
        # print(i, v)

        if "failure" in verifications:
            return "failure"
        if "continue" in verifications:
            return "continue"
        return "success"
