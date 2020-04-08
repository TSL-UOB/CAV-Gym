from enum import Enum


class Observation(Enum):
    NONE = 0

    def __repr__(self):
        return self.name
