import math
from dataclasses import dataclass

from shapely.geometry import Polygon

DEG2RAD = 0.017453292519943295
RAD2DEG = 57.29577951308232


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def distance(self, other):
        return math.sqrt(((other.y - self.y) ** 2) + ((other.x - self.x) ** 2))

    def translate(self, anchor):
        return anchor + self

    def rotate(self, angle):  # Rotate point around (0, 0)
        if angle == 0:
            return self
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        rotated_x = (cos_angle * self.x) - (sin_angle * self.y)
        rotated_y = (sin_angle * self.x) + (cos_angle * self.y)
        return Point(x=rotated_x, y=rotated_y)

    def enlarge(self, center, scale=100):
        dist_x = self.x - center.x
        dist_y = self.y - center.y
        return Point(center.x + (dist_x * scale), center.y + (dist_y * scale))

    def transform(self, angle, anchor):
        if angle == 0:
            return self.translate(anchor)
        else:
            return self.rotate(angle).translate(anchor)

    def __copy__(self):
        return Point(self.x, self.y)

    def __iter__(self):  # massive performance improvement over astuple(self)
        yield self.x
        yield self.y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)


class Shape:
    def __iter__(self):
        raise NotImplementedError

    def translate(self, position):
        raise NotImplementedError

    def transform(self, orientation, position):
        raise NotImplementedError

    def intersects(self, other):
        return Polygon(list(self)).intersects(Polygon(list(other)))

    def contains(self, other):
        return Polygon(list(self)).contains(Polygon(list(other)))

    def mostly_intersects(self, other):
        self_polygon = Polygon(list(self))
        self_polygon_intersection = self_polygon.intersection(Polygon(list(other)))
        return self_polygon_intersection.area / self_polygon.area > 0.5


@dataclass(frozen=True)
class ConvexQuadrilateral(Shape):
    rear_left: Point
    front_left: Point
    front_right: Point
    rear_right: Point

    def __iter__(self):
        yield tuple(self.rear_left)
        yield tuple(self.front_left)
        yield tuple(self.front_right)
        yield tuple(self.rear_right)

    def translate(self, position):
        return ConvexQuadrilateral(
            rear_left=self.rear_left.translate(position),
            front_left=self.front_left.translate(position),
            front_right=self.front_right.translate(position),
            rear_right=self.rear_right.translate(position)
        )

    def transform(self, orientation, position):
        if orientation == 0:
            return self.translate(position)
        else:
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

    def longitudinal_line(self):
        return Line(self.rear_centre(), self.front_centre())


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

    def translate(self, position):
        return CircleSegment(
            rear=self.rear.translate(position),
            arc=[point.translate(position) for point in self.arc]
        )

    def transform(self, orientation, position):
        if orientation == 0:
            return self.translate(position)
        else:
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

    def __iter__(self):
        yield tuple(self.rear)
        yield tuple(self.front_left)
        yield tuple(self.front_right)

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

    def translate(self, position):
        return Triangle(
            rear=self.rear.translate(position),
            front_left=self.front_left.translate(position),
            front_right=self.front_right.translate(position)
        )

    def transform(self, orientation, position):
        if orientation == 0:
            return self.translate(orientation)
        else:
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


@dataclass(frozen=True)
class Line(Shape):
    start: Point
    end: Point

    def __iter__(self):
        yield tuple(self.start)
        yield tuple(self.end)

    def translate(self, position):
        return Line(
            start=self.start.translate(position),
            end=self.end.translate(position)
        )

    def transform(self, orientation, position):
        if orientation == 0:
            return self.translate(position)
        else:
            return Line(
                start=self.start.transform(orientation, position),
                end=self.end.transform(orientation, position)
            )

    def closest_point_from(self, point):  # https://stackoverflow.com/a/47198877
        dx, dy = self.end.x - self.start.x, self.end.y - self.start.y
        denominator = (dx * dx) + (dy * dy)
        a = (dy * (point.y - self.start.y) + dx * (point.x - self.start.x)) / denominator
        return Point(self.start.x + a * dx, self.start.y + a * dy)

    def orientation(self):
        delta_x = self.end.x - self.start.x
        delta_y = self.end.y - self.start.y
        return math.atan2(delta_y, delta_x)

    def random_point(self, np_random):
        deviation = np_random.uniform(0.0, 1.0)
        inverse_deviation = 1.0 - deviation
        x = (inverse_deviation * self.start.x) + (deviation * self.end.x)
        y = (inverse_deviation * self.start.y) + (deviation * self.end.y)
        return Point(x=x, y=y)


@dataclass(frozen=True)
class Circle(Shape):
    centre: Point
    radius: float

    def __iter__(self, resolution=30):
        for i in range(resolution):
            angle = 2 * math.pi * (i / resolution)
            yield tuple(self.circumference_point(angle))
        yield tuple(self.circumference_point(0))

    def translate(self, position):
        return Circle(
            centre=self.centre.translate(position),
            radius=self.radius
        )

    def transform(self, orientation, position):
        return self.translate(position)

    def circumference(self):
        return 2 * math.pi * self.radius

    def circumference_point(self, angle):
        x = self.radius * math.cos(angle) + self.centre.x
        y = self.radius * math.sin(angle) + self.centre.y
        return Point(x, y)

    def arc_from_length(self, start_angle, arc_length):
        arc_angle = (arc_length / self.circumference()) * (2 * math.pi)
        return self.arc_from_angle(start_angle, arc_angle)

    def arc_from_angle(self, start_angle, arc_angle):
        return Arc(circle=self, start_angle=start_angle, arc_angle=arc_angle)


@dataclass(frozen=True)
class Arc(Shape):
    circle: Circle
    start_angle: float
    arc_angle: float

    def __iter__(self, resolution=30):
        for i in range(resolution):
            angle = self.arc_angle * (i / resolution)
            yield tuple(self.circle.circumference_point(self.start_angle + angle))
        yield tuple(self.circle.circumference_point(self.start_angle + self.arc_angle))

    def translate(self, position):
        return Arc(circle=self.circle.translate(position), start_angle=self.start_angle, arc_angle=self.arc_angle)

    def transform(self, orientation, position):
        if orientation == 0:
            return self.translate(position)
        else:
            return Arc(circle=self.circle.translate(position), start_angle=self.start_angle + orientation, arc_angle=self.arc_angle)

    def arc_length(self):
        return abs(self.circle.radius * self.arc_angle)

    def start_point(self):
        return self.circle.circumference_point(self.start_angle)

    def end_point(self):
        return self.circle.circumference_point(self.start_angle + self.arc_angle)

    def split_longitudinally(self, rear_percentage=0.5):
        rear_arc_angle = self.arc_angle * rear_percentage
        rear = Arc(
            circle=self.circle,
            start_angle=self.start_angle,
            arc_angle=rear_arc_angle
        )
        front = Arc(
            circle=self.circle,
            start_angle=self.start_angle + rear_arc_angle,
            arc_angle=self.arc_angle - rear_arc_angle
        )
        return rear, front


@dataclass(frozen=True)
class Arrow(Shape):
    left_arc: Arc
    centre_arc: Arc
    right_arc: Arc

    def __iter__(self, resolution=30):
        for point in reversed(list(self.right_arc)):
            yield tuple(point)
        centre_points = list(self.centre_arc)
        yield tuple(centre_points[0])
        for point in self.left_arc:
            yield tuple(point)
        yield tuple(centre_points[-1])

    def translate(self, position):
        return Arrow(left_arc=self.left_arc.translate(position), centre_arc=self.centre_arc.translate(position), right_arc=self.right_arc.translate(position))

    def transform(self, orientation, position):
        if orientation == 0:
            return self.translate(position)
        else:
            return Arrow(left_arc=self.left_arc.transform(orientation, position), centre_arc=self.centre_arc.transform(orientation, position), right_arc=self.right_arc.transform(orientation, position))

    def split_longitudinally(self, rear_percentage=0.5):
        left_arc_rear, left_arc_front = self.left_arc.split_longitudinally(rear_percentage)
        centre_arc_rear, centre_arc_front = self.centre_arc.split_longitudinally(rear_percentage)
        right_arc_rear, right_arc_front = self.right_arc.split_longitudinally(rear_percentage)
        rear = Arrow(
            left_arc=left_arc_rear,
            centre_arc=centre_arc_rear,
            right_arc=right_arc_rear
        )
        front = Arrow(
            left_arc=left_arc_front,
            centre_arc=centre_arc_front,
            right_arc=right_arc_front
        )
        return rear, front


@dataclass(frozen=True)
class Zone(Shape):
    curve: Arrow
    straight: ConvexQuadrilateral

    def __iter__(self):
        if self.straight is None:
            yield from self.curve
        else:
            for point in self.curve.right_arc:
                yield tuple(point)
            yield tuple(self.straight.front_right)
            yield tuple(self.straight.front_left)
            for point in reversed(list(self.curve.left_arc)):
                yield tuple(point)
            centre_points = list(self.curve.centre_arc)
            yield tuple(centre_points[0])

    def translate(self, position):
        return Zone(curve=self.curve.translate(position), straight=self.straight.translate(position))

    def transform(self, orientation, position):
        if orientation == 0:
            return self.translate(position)
        else:
            return Zone(curve=self.curve.transform(orientation, position), straight=self.straight.transform(orientation, position))
