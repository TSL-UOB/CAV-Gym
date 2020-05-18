from enum import Enum


class Observation(Enum):
    def __repr__(self):
        return self.name


class EmptyObservation(Observation):
    NONE = 0


class VelocityObservation(Observation):
    INACTIVE = 0
    ACTIVE = 1


class OrientationObservation(Observation):
    INACTIVE = 0
    ACTIVE = 1
