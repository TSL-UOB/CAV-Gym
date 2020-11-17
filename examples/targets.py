import math
from enum import Enum


class TargetVelocity(Enum):
    MIN = 0
    MID = 1
    MAX = 2


class TargetOrientation(Enum):
    NORTH = math.pi * 0.5
    NORTH_EAST = math.pi * 0.25
    EAST = 0.0
    SOUTH_EAST = -(math.pi * 0.25)
    SOUTH = -(math.pi * 0.5)
    SOUTH_WEST = -(math.pi * 0.75)
    WEST = math.pi
    NORTH_WEST = math.pi * 0.75
