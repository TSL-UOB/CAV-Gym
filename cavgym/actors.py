import math
from copy import copy
from dataclasses import dataclass
from enum import Enum

from cavgym import utilities
from cavgym.actions import AccelerationAction, TurnAction, TrafficLightAction
from cavgym.assets import Road, TrafficLight
from cavgym.utilities import Bounds, Point


class Actor:
    def __init__(self, init_state, constants):
        self.init_state = init_state
        self.constants = constants

        self.state = copy(self.init_state)

    def reset(self):
        self.state = copy(self.init_state)

    def bounding_box(self):
        raise NotImplementedError

    def step_action(self, joint_action, index):
        raise NotImplementedError

    def step_dynamics(self, time_resolution):
        raise NotImplementedError


@dataclass
class DynamicActorState:
    position: utilities.Point
    velocity: float
    orientation: float
    acceleration: int
    angular_velocity: float

    def __copy__(self):
        return DynamicActorState(copy(self.position), self.velocity, self.orientation, self.acceleration, self.angular_velocity)


@dataclass(frozen=True)
class DynamicActorConstants:
    length: int
    width: int
    wheelbase: int

    min_velocity: float
    max_velocity: float

    normal_acceleration: int
    normal_deceleration: int
    hard_acceleration: int
    hard_deceleration: int
    normal_left_turn: float
    normal_right_turn: float
    hard_left_turn: float
    hard_right_turn: float


class DynamicActor(Actor):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)

        self.relative_bounds = Bounds(rear=-self.constants.length / 2.0, left=self.constants.width / 2.0, front=self.constants.length / 2.0, right=-self.constants.width / 2.0)

    def bounding_box(self):
        return utilities.make_bounding_box(self.state.position, self.relative_bounds, self.state.orientation)

    def stopping_bounds(self):
        hard_braking_distance = (self.state.velocity ** 2) / (2 * -self.constants.hard_deceleration)
        reaction_distance = self.state.velocity * utilities.REACTION_TIME
        hard_braking_relative_bounds = Bounds(rear=self.relative_bounds.front, left=self.relative_bounds.left, front=self.relative_bounds.front + hard_braking_distance, right=self.relative_bounds.right)
        reaction_relative_bounds = Bounds(rear=self.relative_bounds.front + hard_braking_distance, left=self.relative_bounds.left, front=self.relative_bounds.front + hard_braking_distance + reaction_distance, right=self.relative_bounds.right)
        return hard_braking_relative_bounds, reaction_relative_bounds

    def stopping_zones(self):
        hard_braking_relative_bounds, reaction_relative_bounds = self.stopping_bounds()
        hard_braking_relative_bounding_box = utilities.make_bounding_box(self.state.position, hard_braking_relative_bounds, self.state.orientation)
        reaction_relative_bounding_box = utilities.make_bounding_box(self.state.position, reaction_relative_bounds, self.state.orientation)
        return hard_braking_relative_bounding_box, reaction_relative_bounding_box

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


class Vehicle(DynamicActor):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)

        light_offset = self.constants.width * 0.25

        self.indicator_relative_bounds = Bounds(
            rear=self.relative_bounds.rear + light_offset,
            left=self.relative_bounds.left,
            front=self.relative_bounds.front - light_offset,
            right=self.relative_bounds.right
        )

        self.longitudinal_relative_bounds = Bounds(
            rear=self.relative_bounds.rear,
            left=self.relative_bounds.left - light_offset,
            front=self.relative_bounds.front,
            right=self.relative_bounds.right + light_offset
        )

    def indicators(self):
        return utilities.make_bounding_box(self.state.position, self.indicator_relative_bounds, self.state.orientation)

    def longitudinal_lights(self):
        return utilities.make_bounding_box(self.state.position, self.longitudinal_relative_bounds, self.state.orientation)


class Car(Vehicle):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)

        roof_offset = self.constants.length * 0.25

        self.roof_relative_bounds = Bounds(
            rear=self.relative_bounds.rear + roof_offset,
            left=self.relative_bounds.left,
            front=self.relative_bounds.front - roof_offset,
            right=self.relative_bounds.right
        )

    def roof(self):
        return utilities.make_bounding_box(self.state.position, self.roof_relative_bounds, self.state.orientation)


class Bus(Vehicle):
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
    width: int
    x_position: int


class PelicanCrossing(Actor):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)

        relative_bounds = Bounds(
            rear=-self.constants.width / 2.0,
            left=self.constants.road.width / 2.0,
            front=self.constants.width / 2.0,
            right=-self.constants.road.width / 2.0
        )

        position = Point(self.constants.x_position, 0.0).rotate(self.constants.road.constants.orientation).relative(self.constants.road.constants.position)
        self.static_bounding_box = utilities.make_bounding_box(position, relative_bounds, self.constants.road.constants.orientation)

        outbound_intersection_relative_bounds = Bounds(rear=relative_bounds.rear, left=relative_bounds.left, front=relative_bounds.front, right=relative_bounds.left - self.constants.road.outbound.width)
        inbound_intersection_relative_bounds = Bounds(rear=relative_bounds.rear, left=relative_bounds.right + self.constants.road.inbound.width, front=relative_bounds.front, right=relative_bounds.right)
        self.outbound_intersection_bounding_box = utilities.make_bounding_box(position, outbound_intersection_relative_bounds, self.constants.road.constants.orientation)
        self.inbound_intersection_bounding_box = utilities.make_bounding_box(position, inbound_intersection_relative_bounds, self.constants.road.constants.orientation)

        outbound_traffic_light_position = utilities.Point(self.static_bounding_box.rear_left.x, self.static_bounding_box.rear_left.y + 20.0)
        inbound_traffic_light_position = utilities.Point(self.static_bounding_box.front_right.x, self.static_bounding_box.front_right.y - 20.0)
        self.outbound_traffic_light = TrafficLight(outbound_traffic_light_position, self.constants.road.constants.orientation)
        self.inbound_traffic_light = TrafficLight(inbound_traffic_light_position, self.constants.road.constants.orientation)

        self.outbound_spawn = Point(self.constants.x_position + (self.constants.width * 0.15), (self.constants.road.width / 2.0) + (self.constants.road.constants.lane_width / 2.0)).rotate(self.constants.road.constants.orientation).relative(self.constants.road.constants.position)
        self.inbound_spawn = Point(self.constants.x_position - (self.constants.width * 0.15), -(self.constants.road.width / 2.0) - (self.constants.road.constants.lane_width / 2.0)).rotate(self.constants.road.constants.orientation).relative(self.constants.road.constants.position)

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
