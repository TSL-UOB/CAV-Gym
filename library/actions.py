from enum import Enum


class Action(Enum):
    def __repr__(self):
        return self.name


class TrafficLightAction(Action):
    NOOP = 0
    TURN_RED = 1
    TURN_AMBER = 2
    TURN_GREEN = 3
