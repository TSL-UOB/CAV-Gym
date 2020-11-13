from abc import ABC
from copy import copy

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float

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


@dataclass(frozen=True)
class ConvexQuadrilateral:
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


def normalise_angle(radians):
    while radians <= -math.pi:
        radians += 2 * math.pi
    while radians > math.pi:
        radians -= 2 * math.pi
    return radians + 0.0 if radians == 0 else radians  # avoid -0


class Actor(ABC):
    def __init__(self, init_state, constants, **kwargs):
        super().__init__(**kwargs)  # important to pass on kwargs if class is used as superclass in multiple inheritance

        self.init_state = init_state
        self.constants = constants

        self.state = copy(self.init_state)

    def reset(self):
        self.state = copy(self.init_state)

    def bounding_box(self):
        raise NotImplementedError

    def step(self, action, time_resolution):
        raise NotImplementedError


@dataclass
class DynamicActorState:
    position: Point
    velocity: float
    orientation: float

    def __copy__(self):
        return DynamicActorState(copy(self.position), self.velocity, self.orientation)

    def __iter__(self):
        yield from self.position
        yield self.velocity
        yield self.orientation


@dataclass(frozen=True)
class DynamicActorConstants:
    length: float
    width: float
    wheelbase: float

    min_velocity: float
    max_velocity: float
    min_throttle: float
    max_throttle: float
    min_steering_angle: float
    max_steering_angle: float


class DynamicActor(Actor):
    def __init__(self, init_state, constants):
        super().__init__(init_state=init_state, constants=constants)

        self.shape = make_rectangle(self.constants.length, self.constants.width)

        self.wheelbase_offset_front = self.constants.wheelbase / 2.0
        self.wheelbase_offset_rear = self.constants.wheelbase / 2.0

        self.throttle = None
        self.steering_angle = None

    def reset(self):
        super().reset()

        self.throttle = None
        self.steering_angle = None

    def bounding_box(self):
        return self.shape.transform(self.state.orientation, self.state.position)

    def step(self, action, time_resolution):
        self.throttle, self.steering_angle = action

        """Simple vehicle dynamics: http://engineeringdotnet.blogspot.com/2010/04/simple-2d-car-physics-in-games.html"""
        # cos_orientation = math.cos(self.state.orientation)
        # sin_orientation = math.sin(self.state.orientation)

        # prod_velocity_time_resolution = self.state.velocity * time_resolution
        # sum_orientation_steering_angle = normalise_angle(self.state.orientation + self.steering_angle)

        front_wheel_calculate_position = Point(
            x=self.state.position.x + (self.wheelbase_offset_front * math.cos(self.state.orientation)),
            y=self.state.position.y + (self.wheelbase_offset_front * math.sin(self.state.orientation))
        )
        front_wheel_apply_steering_angle = Point(
            x=self.state.velocity * time_resolution * math.cos(self.state.orientation + self.steering_angle),
            y=self.state.velocity * time_resolution * math.sin(self.state.orientation + self.steering_angle)
        )
        front_wheel_position = front_wheel_calculate_position + front_wheel_apply_steering_angle

        back_wheel_calculate_position = Point(
            x=self.state.position.x - (self.wheelbase_offset_rear * math.cos(self.state.orientation)),
            y=self.state.position.y - (self.wheelbase_offset_rear * math.sin(self.state.orientation))
        )
        back_wheel_apply_steering_angle = Point(
            x=self.state.velocity * time_resolution * math.cos(self.state.orientation),
            y=self.state.velocity * time_resolution * math.sin(self.state.orientation)
        )
        back_wheel_position = back_wheel_calculate_position + back_wheel_apply_steering_angle

        self.state.position = Point(
            x=(front_wheel_position.x + back_wheel_position.x) / 2.0,
            y=(front_wheel_position.y + back_wheel_position.y) / 2.0
        )

        self.state.velocity = max(
            self.constants.min_velocity,
            min(
                self.constants.max_velocity,
                self.state.velocity + (self.throttle * time_resolution)
            )
        )

        self.state.orientation = normalise_angle(math.atan2(front_wheel_position.y - back_wheel_position.y, front_wheel_position.x - back_wheel_position.x))

    def action_translation(self, successor_state, time_resolution):
        throttle = successor_state.velocity - self.state.velocity

        def front_wheel_position_x(state):
            return state.position.x + (self.wheelbase_offset_front * math.cos(state.orientation))

        def front_wheel_position_y(state):
            return state.position.y + (self.wheelbase_offset_front * math.sin(state.orientation))

        assert self.state.velocity * time_resolution != 0

        steering_angle_x = math.acos(
            (front_wheel_position_x(successor_state) - front_wheel_position_x(self.state)) / (self.state.velocity * time_resolution)
        ) - self.state.orientation

        steering_angle_y = math.asin(
            (front_wheel_position_y(successor_state) - front_wheel_position_y(self.state)) / (self.state.velocity * time_resolution)
        ) - self.state.orientation

        # assert steering_angle_x == steering_angle_y

        return throttle, steering_angle_x
