import math
from copy import copy
from dataclasses import dataclass
from enum import Enum

from shapely import geometry, affinity

from cavgym import utilities
from cavgym.actions import AccelerationAction, TurnAction, TrafficLightAction
from cavgym.assets import Road


class Actor:
    def __init__(self, init_state, constants):
        self.init_state = init_state
        self.constants = constants

        self.state = copy(self.init_state)

    def reset(self):
        self.state = copy(self.init_state)

    def bounding_box(self):
        raise NotImplementedError

    def intersects(self, other):
        return self.bounding_box().intersects(other.bounding_box())

    def step_action(self, joint_action, index):
        raise NotImplementedError

    def step_dynamics(self, time_resolution):
        raise NotImplementedError


@dataclass
class DynamicActorState:
    position: utilities.Point
    velocity: float
    orientation: float
    acceleration: float
    angular_velocity: float

    def __copy__(self):
        return DynamicActorState(copy(self.position), self.velocity, self.orientation, self.acceleration, self.angular_velocity)


@dataclass(frozen=True)
class DynamicActorConstants:
    length: float
    width: float
    wheelbase: float

    min_velocity: float
    max_velocity: float

    normal_acceleration: float
    normal_deceleration: float
    hard_acceleration: float
    hard_deceleration: float
    normal_left_turn: float
    normal_right_turn: float
    hard_left_turn: float
    hard_right_turn: float


class DynamicActor(Actor):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)

        rear, front = -self.constants.length / 2.0, self.constants.length / 2.0
        right, left = -self.constants.width / 2.0, self.constants.width / 2.0
        self.bounds = rear, right, front, left

    def bounding_box(self):
        assert self.bounds is not None
        box = geometry.box(*self.bounds)
        rotated_box = affinity.rotate(box, self.state.orientation, use_radians=True)
        return affinity.translate(rotated_box, self.state.position.x, self.state.position.y)

    def stopping_bounds(self):
        assert self.bounds is not None
        rear, right, front, left = self.bounds
        normal_braking_distance = (self.state.velocity ** 2) / (2 * -self.constants.normal_deceleration)
        hard_braking_distance = (self.state.velocity ** 2) / (2 * -self.constants.hard_deceleration)
        # reaction_distance = self.state.velocity * utilities.REACTION_TIME
        front_hard_braking = front + hard_braking_distance
        return (front, right, front_hard_braking, left), (front_hard_braking, right, front + normal_braking_distance, left)

    def reaction_zone(self):
        _, reaction_bounds = self.stopping_bounds()
        box = geometry.box(*reaction_bounds)
        rotated_box = affinity.rotate(box, self.state.orientation, use_radians=True)
        return affinity.translate(rotated_box, self.state.position.x, self.state.position.y)

    def step_action(self, joint_action, index_self):
        acceleration_action_id, turn_action_id = joint_action[index_self]
        acceleration_action = AccelerationAction(acceleration_action_id)
        turn_action = TurnAction(turn_action_id)

        if acceleration_action is AccelerationAction.NEUTRAL:
            self.state.acceleration = 0
        elif acceleration_action is AccelerationAction.NORMAL_ACCELERATE:
            self.state.acceleration = self.constants.normal_acceleration
        elif acceleration_action is AccelerationAction.NORMAL_DECELERATE:
            self.state.acceleration = self.constants.normal_deceleration
        elif acceleration_action is AccelerationAction.HARD_ACCELERATE:
            self.state.acceleration = self.constants.hard_acceleration
        elif acceleration_action is AccelerationAction.HARD_DECELERATE:
            self.state.acceleration = self.constants.hard_deceleration

        if turn_action is TurnAction.NEUTRAL:
            self.state.angular_velocity = 0
        elif turn_action is TurnAction.NORMAL_LEFT:
            self.state.angular_velocity = self.constants.normal_left_turn
        elif turn_action is TurnAction.NORMAL_RIGHT:
            self.state.angular_velocity = self.constants.normal_right_turn
        elif turn_action is TurnAction.HARD_LEFT:
            self.state.angular_velocity = self.constants.hard_left_turn
        elif turn_action is TurnAction.HARD_RIGHT:
            self.state.angular_velocity = self.constants.hard_right_turn

    def step_dynamics(self, time_resolution):
        self.state.velocity = max(
            self.constants.min_velocity,
            min(
                self.constants.max_velocity,
                self.state.velocity + (self.state.acceleration * time_resolution)
            )
        )

        """Simple vehicle dynamics: http://engineeringdotnet.blogspot.com/2010/04/simple-2d-car-physics-in-games.html"""

        front_wheel_position = utilities.Point(
            self.state.position.x + ((self.constants.wheelbase / 2.0) * math.cos(self.state.orientation)),
            self.state.position.y + ((self.constants.wheelbase / 2.0) * math.sin(self.state.orientation))
        )
        front_wheel_position.x += self.state.velocity * time_resolution * math.cos(self.state.orientation + self.state.angular_velocity)
        front_wheel_position.y += self.state.velocity * time_resolution * math.sin(self.state.orientation + self.state.angular_velocity)

        back_wheel_position = utilities.Point(
            self.state.position.x - ((self.constants.wheelbase / 2.0) * math.cos(self.state.orientation)),
            self.state.position.y - ((self.constants.wheelbase / 2.0) * math.sin(self.state.orientation))
        )
        back_wheel_position.x += self.state.velocity * time_resolution * math.cos(self.state.orientation)
        back_wheel_position.y += self.state.velocity * time_resolution * math.sin(self.state.orientation)

        self.state.position.x = (front_wheel_position.x + back_wheel_position.x) / 2.0
        self.state.position.y = (front_wheel_position.y + back_wheel_position.y) / 2.0

        self.state.orientation = math.atan2(front_wheel_position.y - back_wheel_position.y, front_wheel_position.x - back_wheel_position.x)


class Pedestrian(DynamicActor):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)


class Car(DynamicActor):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)


class Bus(DynamicActor):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)


class Bicycle(DynamicActor):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)


class TrafficLightState(Enum):
    RED = 0
    AMBER = 1
    GREEN = 2

    def __copy__(self):
        return TrafficLightState(self)


@dataclass(frozen=True)
class PelicanCrossingConstants:
    road: Road
    width: float
    x_position: float


class PelicanCrossing(Actor):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)

        rear, front = -self.constants.width / 2.0, self.constants.width / 2.0
        right, left = -self.constants.road.width / 2.0, self.constants.road.width / 2.0
        self.bounds = rear, right, front, left

        box = geometry.box(*self.bounds)
        rotated_box = affinity.rotate(box, self.constants.road.constants.orientation, use_radians=True)
        self.static_bounding_box = affinity.translate(rotated_box, self.constants.x_position, self.constants.road.constants.position.y)

    def bounding_box(self):
        return self.static_bounding_box

    def step_action(self, joint_action, index):
        traffic_light_action = TrafficLightAction(joint_action[index])

        if traffic_light_action is TrafficLightAction.TURN_RED:
            self.state = TrafficLightState.RED
        elif traffic_light_action is TrafficLightAction.TURN_AMBER:
            self.state = TrafficLightState.AMBER
        elif traffic_light_action is TrafficLightAction.TURN_GREEN:
            self.state = TrafficLightState.GREEN

    def step_dynamics(self, time_resolution):
        pass
