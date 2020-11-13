from enum import Enum


class Action(Enum):
    def __repr__(self):
        return self.name


class TrafficLightAction(Action):
    NOOP = 0
    TURN_RED = 1
    TURN_AMBER = 2
    TURN_GREEN = 3


class TargetVelocity(Action):
    NOOP = 0
    STOP = 1
    SLOW = 2
    FAST = 3


class TargetOrientation(Action):
    NOOP = 0
    FORWARD_LEFT = 1
    LEFT = 2
    REAR_LEFT = 3
    REAR = 4
    REAR_RIGHT = 5
    RIGHT = 6
    FORWARD_RIGHT = 7
