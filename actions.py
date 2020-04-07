from enum import Enum


class AccelerationAction(Enum):
    NEUTRAL = 0
    NORMAL_ACCELERATE = 1
    NORMAL_DECELERATE = 2
    HARD_ACCELERATE = 3
    HARD_DECELERATE = 4

    def __repr__(self):
        return self.name


class TurnAction(Enum):
    NEUTRAL = 0
    NORMAL_LEFT = 1
    NORMAL_RIGHT = 2
    HARD_LEFT = 3
    HARD_RIGHT = 4

    def __repr__(self):
        return self.name


class TrafficLightAction(Enum):
    NOOP = 0
    TURN_RED = 1
    TURN_AMBER = 2
    TURN_GREEN = 3

    def __repr__(self):
        return self.name
