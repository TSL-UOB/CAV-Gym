import math
from dataclasses import dataclass, astuple

from shapely.geometry import Polygon

DEG2RAD = 0.017453292519943295
RAD2DEG = 57.29577951308232


@dataclass
class Point:
    x: float
    y: float

    def distance(self, other):
        return math.sqrt(((other.x - self.x) ** 2) + ((other.y - self.y) ** 2))

    def translate(self, anchor):
        translated_x = anchor.x + self.x
        translated_y = anchor.y + self.y
        return Point(x=translated_x, y=translated_y)

    def rotate(self, angle):  # Rotate point around (0, 0)
        rotated_x = (math.cos(angle) * self.x) - (math.sin(angle) * self.y)
        rotated_y = (math.sin(angle) * self.x) + (math.cos(angle) * self.y)
        return Point(x=rotated_x, y=rotated_y)

    def enlarge(self, center, scale=100):
        dist_x = self.x - center.x
        dist_y = self.y - center.y
        return Point(center.x + (dist_x * scale), center.y + (dist_y * scale))

    def transform(self, angle, anchor):
        return self.rotate(angle).translate(anchor)

    def __copy__(self):
        return Point(self.x, self.y)

    def __iter__(self):
        yield from astuple(self)


class Shape:
    def __iter__(self):
        yield from astuple(self)

    def transform(self, orientation, position):
        raise NotImplementedError

    def intersects(self, other):
        return Polygon(list(self)).intersects(Polygon(list(other)))


@dataclass(frozen=True)
class ConvexQuadrilateral(Shape):
    rear_left: Point
    front_left: Point
    front_right: Point
    rear_right: Point

    def transform(self, orientation, position):
        return ConvexQuadrilateral(
            rear_left=self.rear_left.transform(orientation, position),
            front_left=self.front_left.transform(orientation, position),
            front_right=self.front_right.transform(orientation, position),
            rear_right=self.rear_right.transform(orientation, position)
        )

    def centre(self):
        return Point((self.front_left.x + self.rear_right.x) * 0.5, (self.front_left.y + self.rear_right.y) * 0.5)

    def front_centre(self):
        return Point((self.front_left.x + self.front_right.x) * 0.5, (self.front_left.y + self.front_right.y) * 0.5)

    def rear_centre(self):
        return Point((self.rear_left.x + self.rear_right.x) * 0.5, (self.rear_left.y + self.rear_right.y) * 0.5)

    def left_centre(self):
        return Point((self.rear_left.x + self.front_left.x) * 0.5, (self.rear_left.y + self.front_left.y) * 0.5)

    def right_centre(self):
        return Point((self.rear_right.x + self.front_right.x) * 0.5, (self.rear_right.y + self.front_right.y) * 0.5)

    def split_laterally(self, left_percentage=0.5):
        front_split = Point((self.front_left.x * (1 - left_percentage)) + (self.front_right.x * left_percentage), (self.front_left.y * (1 - left_percentage)) + (self.front_right.y * left_percentage))
        rear_split = Point((self.rear_left.x * (1 - left_percentage)) + (self.rear_right.x * left_percentage), (self.rear_left.y * (1 - left_percentage)) + (self.rear_right.y * left_percentage))
        left = ConvexQuadrilateral(
            rear_left=self.rear_left,
            front_left=self.front_left,
            front_right=front_split,
            rear_right=rear_split
        )
        right = ConvexQuadrilateral(
            rear_left=rear_split,
            front_left=front_split,
            front_right=self.front_right,
            rear_right=self.rear_right
        )
        return left, right

    def divide_laterally(self, segments):
        if segments == 1:
            yield self
        else:
            segment, remainder = self.split_laterally(left_percentage=1 / segments)
            yield segment
            yield from remainder.divide_laterally(segments - 1)

    def flip_laterally(self):
        return ConvexQuadrilateral(
            rear_left=self.rear_right,
            front_left=self.front_right,
            front_right=self.front_left,
            rear_right=self.rear_left
        )

    def split_longitudinally(self, rear_percentage=0.5):
        left_split = Point((self.rear_left.x * (1 - rear_percentage)) + (self.front_left.x * rear_percentage), (self.rear_left.y * (1 - rear_percentage)) + (self.front_left.y * rear_percentage))
        right_split = Point((self.rear_right.x * (1 - rear_percentage)) + (self.front_right.x * rear_percentage), (self.rear_right.y * (1 - rear_percentage)) + (self.front_right.y * rear_percentage))
        rear = ConvexQuadrilateral(
            rear_left=self.rear_left,
            front_left=left_split,
            front_right=right_split,
            rear_right=self.rear_right
        )
        front = ConvexQuadrilateral(
            rear_left=left_split,
            front_left=self.front_left,
            front_right=self.front_right,
            rear_right=right_split
        )
        return rear, front

    def flip_longitudinally(self):
        return ConvexQuadrilateral(
            rear_left=self.front_left,
            front_left=self.rear_left,
            front_right=self.rear_right,
            rear_right=self.front_right
        )

    def flip(self):
        return self.flip_longitudinally().flip_laterally()


def make_rectangle(length, width, anchor=Point(0, 0), rear_offset=0.5, left_offset=0.5):
    rear = anchor.x - (length * rear_offset)
    front = anchor.x + (length * (1 - rear_offset))
    left = anchor.y + (width * left_offset)
    right = anchor.y - (width * (1 - left_offset))
    return ConvexQuadrilateral(
        rear_left=Point(rear, left),
        front_left=Point(front, left),
        front_right=Point(front, right),
        rear_right=Point(rear, right)
    )


@dataclass(frozen=True)
class CircleSegment(Shape):
    rear: Point
    arc: list

    def __iter__(self):
        yield tuple(self.rear)
        for point in self.arc:
            yield tuple(point)

    def transform(self, orientation, position):
        return CircleSegment(
            rear=self.rear.transform(orientation, position),
            arc=[point.transform(orientation, position) for point in self.arc]
        )


def make_circle_segment(radius, angle, anchor=Point(0, 0), angle_left_offset=0.5):
    assert 0 < angle < DEG2RAD * 180
    assert 0 <= angle_left_offset <= 1

    def make_arc(start, end, resolution=30):
        resolution_angle = angle / resolution
        points = [start]
        for i in range(resolution - 1):
            point = start.rotate(resolution_angle * -(i + 1))
            points.append(point)
        points.append(end)
        assert len(points) == resolution + 1
        return points

    front_left = Point(anchor.x + radius, anchor.y).rotate(angle * angle_left_offset)
    front_right = Point(anchor.x + radius, anchor.y).rotate(-angle * (1 - angle_left_offset))
    arc = make_arc(front_left, front_right)
    assert arc[0].x == front_left.x and arc[0].y == front_left.y
    assert arc[-1].x == front_right.x and arc[-1].y == front_right.y
    return CircleSegment(
        rear=anchor,
        arc=arc
    )


@dataclass(frozen=True)
class Triangle(Shape):
    rear: Point
    front_left: Point
    front_right: Point

    def angle(self):
        angle = math.atan2(self.front_left.y - self.rear.y, self.front_left.x - self.rear.x) - math.atan2(self.front_right.y - self.rear.y, self.front_right.x - self.rear.x)
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle <= -math.pi:
            angle += 2 * math.pi
        return angle

    def normalise(self):  # front_left is on the left if angle is postive
        angle = self.angle()
        if angle >= 0:
            return self
        else:
            return Triangle(
                rear=self.rear,
                front_left=self.front_right,
                front_right=self.front_left
            )

    def transform(self, orientation, position):
        return Triangle(
            rear=self.rear.transform(orientation, position),
            front_left=self.front_left.transform(orientation, position),
            front_right=self.front_right.transform(orientation, position)
        )


def normalise_angle(radians):
    while radians <= -math.pi:
        radians += 2 * math.pi
    while radians > math.pi:
        radians -= 2 * math.pi
    return radians
