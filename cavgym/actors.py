import math
from copy import copy
from dataclasses import dataclass
from enum import Enum

from cavgym import geometry
from cavgym.actions import AccelerationAction, TurnAction, TrafficLightAction
from cavgym.assets import Road, Occlusion
from cavgym.geometry import Point


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
    position: geometry.Point
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

    view_distance: int
    view_angle: float

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


class DynamicActor(Actor, Occlusion):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)

        self.shape = geometry.make_rectangle(self.constants.length, self.constants.width)

        self.field_of_view_shape = geometry.make_circle_segment(self.constants.view_distance, self.constants.view_angle)

    def bounding_box(self):
        return self.shape.transform(self.state.orientation, self.state.position)

    def stopping_zones(self):
        hard_braking_distance = (self.state.velocity ** 2) / (2 * -self.constants.hard_deceleration)
        reaction_distance = self.state.velocity * geometry.REACTION_TIME
        hard_braking_zone = geometry.make_rectangle(hard_braking_distance, self.constants.width, rear_offset=0).transform(self.state.orientation, Point(self.constants.length * 0.5, 0).transform(self.state.orientation, self.state.position))
        reaction_zone = geometry.make_rectangle(reaction_distance, self.constants.width, rear_offset=0).transform(self.state.orientation, Point((self.constants.length * 0.5) + hard_braking_distance, 0).transform(self.state.orientation, self.state.position))
        return hard_braking_zone, reaction_zone

    def field_of_view(self):
        return self.field_of_view_shape.transform(self.state.orientation, self.state.position)

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

        front_wheel_position = geometry.Point(
            self.state.position.x + ((self.constants.wheelbase / 2.0) * math.cos(self.state.orientation)),
            self.state.position.y + ((self.constants.wheelbase / 2.0) * math.sin(self.state.orientation))
        )
        front_wheel_position.x += self.state.velocity * time_resolution * math.cos(self.state.orientation + self.state.angular_velocity)
        front_wheel_position.y += self.state.velocity * time_resolution * math.sin(self.state.orientation + self.state.angular_velocity)

        back_wheel_position = geometry.Point(
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

        self.indicators_shape = geometry.make_rectangle(self.constants.length * 0.8, self.constants.width)
        self.longitudinal_lights_shape = geometry.make_rectangle(self.constants.length, self.constants.width * 0.6)

    def indicators(self):
        return self.indicators_shape.transform(self.state.orientation, self.state.position)

    def longitudinal_lights(self):
        return self.longitudinal_lights_shape.transform(self.state.orientation, self.state.position)


class Car(Vehicle):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)

        self.roof_shape = geometry.make_rectangle(self.constants.length * 0.5, self.constants.width)

    def roof(self):
        return self.roof_shape.transform(self.state.orientation, self.state.position)


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
class TrafficLightConstants:
    width: int
    height: int
    position: Point
    orientation: float


class TrafficLight(Actor, Occlusion):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)

        self.static_bounding_box = geometry.make_rectangle(self.constants.width, self.constants.height).transform(self.constants.orientation, self.constants.position)

        def make_light(y):
            return Point(0.0, y).transform(self.constants.orientation, self.constants.position)

        self.red_light = make_light(self.constants.height * 0.25)
        self.amber_light = make_light(0.0)
        self.green_light = make_light(-self.constants.height * 0.25)

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


@dataclass(frozen=True)
class PelicanCrossingConstants:
    road: Road
    width: int
    x_position: int


class PelicanCrossing(Actor):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)

        position = Point(self.constants.x_position, 0.0).transform(self.constants.road.constants.orientation, self.constants.road.constants.position)

        self.static_bounding_box = geometry.make_rectangle(self.constants.width, self.constants.road.width).transform(self.constants.road.constants.orientation, position)

        self.outbound_intersection_bounding_box = geometry.make_rectangle(self.constants.width, self.constants.road.outbound.width).transform(self.constants.road.constants.orientation, Point(0, (self.constants.road.inbound.width - position.y) * 0.5).transform(0, position))
        self.inbound_intersection_bounding_box = geometry.make_rectangle(self.constants.width, self.constants.road.inbound.width).transform(self.constants.road.constants.orientation, Point(0, -(self.constants.road.outbound.width - position.y) * 0.5).transform(0, position))

        outbound_traffic_light_position = geometry.Point(self.static_bounding_box.rear_left.x, self.static_bounding_box.rear_left.y + 20.0)
        inbound_traffic_light_position = geometry.Point(self.static_bounding_box.front_right.x, self.static_bounding_box.front_right.y - 20.0)

        outbound_traffic_light_constants = TrafficLightConstants(
            width=10,
            height=20,
            position=outbound_traffic_light_position,
            orientation=self.constants.road.constants.orientation
        )
        inbound_traffic_light_constants = TrafficLightConstants(
            width=10,
            height=20,
            position=inbound_traffic_light_position,
            orientation=self.constants.road.constants.orientation
        )
        self.outbound_traffic_light = TrafficLight(init_state, outbound_traffic_light_constants)
        self.inbound_traffic_light = TrafficLight(init_state, inbound_traffic_light_constants)

        self.outbound_spawn = Point(self.constants.x_position + (self.constants.width * 0.15), (self.constants.road.width / 2.0) + (self.constants.road.constants.lane_width / 2.0)).transform(self.constants.road.constants.orientation, self.constants.road.constants.position)
        self.inbound_spawn = Point(self.constants.x_position - (self.constants.width * 0.15), -(self.constants.road.width / 2.0) - (self.constants.road.constants.lane_width / 2.0)).transform(self.constants.road.constants.orientation, self.constants.road.constants.position)

    def bounding_box(self):
        return self.static_bounding_box

    def step_action(self, joint_action, index):
        pelican_crossing_action = TrafficLightAction(joint_action[index])

        if pelican_crossing_action is TrafficLightAction.TURN_RED:
            self.state = TrafficLightState.RED
        elif pelican_crossing_action is TrafficLightAction.TURN_AMBER:
            self.state = TrafficLightState.AMBER
        elif pelican_crossing_action is TrafficLightAction.TURN_GREEN:
            self.state = TrafficLightState.GREEN

        self.outbound_traffic_light.step_action(joint_action, index)
        self.inbound_traffic_light.step_action(joint_action, index)

    def step_dynamics(self, time_resolution):
        pass
