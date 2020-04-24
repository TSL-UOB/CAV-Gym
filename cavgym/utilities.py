from dataclasses import dataclass


DEG2RAD = 0.017453292519943295
REACTION_TIME = 0.675


@dataclass
class Point:
    x: float
    y: float

    def __copy__(self):
        return Point(self.x, self.y)
