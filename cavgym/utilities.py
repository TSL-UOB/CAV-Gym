import math
from dataclasses import dataclass, astuple

from shapely.geometry import Polygon

DEG2RAD = 0.017453292519943295
REACTION_TIME = 0.675


@dataclass
class Point:
    x: float
    y: float

    def relative(self, anchor):
        relative_x = anchor.x + self.x
        relative_y = anchor.y + self.y
        return Point(x=relative_x, y=relative_y)

    def rotate(self, angle):  # Rotate point around (0, 0)
        rotated_x = (math.cos(angle) * self.x) - (math.sin(angle) * self.y)
        rotated_y = (math.sin(angle) * self.x) + (math.cos(angle) * self.y)
        return Point(x=rotated_x, y=rotated_y)

    def __copy__(self):
        return Point(self.x, self.y)

    def __iter__(self):
        yield from astuple(self)


@dataclass
class Bounds:
    rear: float
    left: float
    front: float
    right: float

    def __post_init__(self):
        self.coordinates = BoundingBox(
            rear_left=Point(self.rear, self.left),
            front_left=Point(self.front, self.left),
            front_right=Point(self.front, self.right),
            rear_right=Point(self.rear, self.right)
        )

    def __iter__(self):
        yield from astuple(self)


@dataclass
class BoundingBox:
    rear_left: Point
    front_left: Point
    front_right: Point
    rear_right: Point

    def __iter__(self):
        yield from astuple(self)

    def intersects(self, other):
        return Polygon(list(self)).intersects(Polygon(list(other)))


def make_bounding_box(anchor: Point, relative_bounds: Bounds, orientation):  # position is taken as (0, 0) in relative_bounds
    return BoundingBox(
        rear_left=relative_bounds.coordinates.rear_left.rotate(orientation).relative(anchor),
        front_left=relative_bounds.coordinates.front_left.rotate(orientation).relative(anchor),
        front_right=relative_bounds.coordinates.front_right.rotate(orientation).relative(anchor),
        rear_right=relative_bounds.coordinates.rear_right.rotate(orientation).relative(anchor)
    )
