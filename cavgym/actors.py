import math
from copy import copy
from dataclasses import dataclass
from enum import Enum

from gym.utils import seeding

from cavgym import geometry
from cavgym.actions import TrafficLightAction, OrientationAction, VelocityAction
from cavgym.assets import Road, Occlusion
from cavgym.geometry import Point


REACTION_TIME = 0.675


class Actor:
    def __init__(self, init_state, constants, **kwargs):
        super().__init__(**kwargs)  # important to pass on kwargs if class is used as superclass in multiple inheritance

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

    min_velocity: int
    max_velocity: int

    acceleration_rate: int
    deceleration_rate: int
    left_turn_rate: float
    right_turn_rate: float

    target_slow_velocity: int
    target_fast_velocity: int

    def __post_init__(self):
        assert self.min_velocity <= self.target_slow_velocity <= self.max_velocity
        assert self.min_velocity <= self.target_fast_velocity <= self.max_velocity


class DynamicActor(Actor, Occlusion):
    def __init__(self, init_state, constants):
        super().__init__(init_state=init_state, constants=constants)

        self.shape = geometry.make_rectangle(self.constants.length, self.constants.width)

        self.target_velocity = None
        self.target_orientation = None

    def reset(self):
        super().reset()

        self.target_velocity = None
        self.target_orientation = None

    def bounding_box(self):
        return self.shape.transform(self.state.orientation, self.state.position)

    def stopping_zones(self):
        braking_distance = (self.state.velocity ** 2) / (2 * -self.constants.deceleration_rate)
        reaction_distance = self.state.velocity * REACTION_TIME
        braking_zone = geometry.make_rectangle(braking_distance, self.constants.width, rear_offset=0).transform(self.state.orientation, Point(self.constants.length * 0.5, 0).transform(self.state.orientation, self.state.position))
        reaction_zone = geometry.make_rectangle(reaction_distance, self.constants.width, rear_offset=0).transform(self.state.orientation, Point((self.constants.length * 0.5) + braking_distance, 0).transform(self.state.orientation, self.state.position))
        return braking_zone, reaction_zone

    def line_anchor(self, road):
        closest = road.bounding_box().longitudinal_line().closest_point_from(self.state.position)
        return geometry.Line(self.state.position, closest)

    def line_anchor_relative_angle(self, road):
        angle = self.line_anchor(road).orientation()
        return geometry.normalise_angle(angle - self.state.orientation)

    def step_action(self, joint_action, index_self):
        velocity_action_id, orientation_action_id = joint_action[index_self]
        velocity_action = VelocityAction(velocity_action_id)
        orientation_action = OrientationAction(orientation_action_id)

        if velocity_action is VelocityAction.STOP:
            self.target_velocity = 0
        elif velocity_action is VelocityAction.SLOW:
            self.target_velocity = self.constants.target_slow_velocity
        elif velocity_action is VelocityAction.FAST:
            self.target_velocity = self.constants.target_fast_velocity

        if self.target_velocity is not None:
            difference = self.target_velocity - self.state.velocity
            if difference < 0:
                self.state.acceleration = self.constants.deceleration_rate
            elif difference > 0:
                self.state.acceleration = self.constants.acceleration_rate
            else:
                self.target_velocity = None

        if orientation_action is OrientationAction.FORWARD_LEFT:
            self.target_orientation = self.state.orientation + (geometry.DEG2RAD * 45)
        elif orientation_action is OrientationAction.LEFT:
            self.target_orientation = self.state.orientation + (geometry.DEG2RAD * 90)
        elif orientation_action is OrientationAction.REAR_LEFT:
            self.target_orientation = self.state.orientation + (geometry.DEG2RAD * 135)
        elif orientation_action is OrientationAction.REAR:
            self.target_orientation = self.state.orientation + (geometry.DEG2RAD * 180)
        elif orientation_action is OrientationAction.REAR_RIGHT:
            self.target_orientation = self.state.orientation - (geometry.DEG2RAD * 135)
        elif orientation_action is OrientationAction.RIGHT:
            self.target_orientation = self.state.orientation - (geometry.DEG2RAD * 90)
        elif orientation_action is OrientationAction.FORWARD_RIGHT:
            self.target_orientation = self.state.orientation - (geometry.DEG2RAD * 45)

        if self.target_orientation is not None:
            self.target_orientation = geometry.normalise_angle(self.target_orientation)
            assert -math.pi < self.target_orientation <= math.pi

        if orientation_action is not OrientationAction.NOOP and self.target_orientation is not None:
            difference = geometry.normalise_angle(self.target_orientation - self.state.orientation)
            if difference < 0:
                self.state.angular_velocity = self.constants.right_turn_rate
            elif difference > 0:
                self.state.angular_velocity = self.constants.left_turn_rate
            else:
                self.target_orientation = None

    def step_dynamics(self, time_resolution):
        self.state.velocity = max(
            self.constants.min_velocity,
            min(
                self.constants.max_velocity,
                self.state.velocity + (self.state.acceleration * time_resolution)
            )
        )

        """Simple vehicle dynamics: http://engineeringdotnet.blogspot.com/2010/04/simple-2d-car-physics-in-games.html"""

        front_wheel_calculate_position = geometry.Point(
            x=self.state.position.x + ((self.constants.wheelbase / 2.0) * math.cos(self.state.orientation)),
            y=self.state.position.y + ((self.constants.wheelbase / 2.0) * math.sin(self.state.orientation))
        )
        front_wheel_apply_angular_velocity = Point(
            x=self.state.velocity * time_resolution * math.cos(self.state.orientation + self.state.angular_velocity),
            y=self.state.velocity * time_resolution * math.sin(self.state.orientation + self.state.angular_velocity)
        )
        front_wheel_position = front_wheel_calculate_position + front_wheel_apply_angular_velocity

        back_wheel_calculate_position = geometry.Point(
            x=self.state.position.x - ((self.constants.wheelbase / 2.0) * math.cos(self.state.orientation)),
            y=self.state.position.y - ((self.constants.wheelbase / 2.0) * math.sin(self.state.orientation))
        )
        back_wheel_apply_angular_velocity = Point(
            x=self.state.velocity * time_resolution * math.cos(self.state.orientation),
            y=self.state.velocity * time_resolution * math.sin(self.state.orientation)
        )
        back_wheel_position = back_wheel_calculate_position + back_wheel_apply_angular_velocity

        self.state.position = Point(
            x=(front_wheel_position.x + back_wheel_position.x) / 2.0,
            y=(front_wheel_position.y + back_wheel_position.y) / 2.0
        )

        self.state.orientation = math.atan2(front_wheel_position.y - back_wheel_position.y, front_wheel_position.x - back_wheel_position.x)  # range is [-math.pi, math.pi] (min or max may be exclusive)
        assert -math.pi < self.state.orientation <= math.pi

        if self.target_velocity is not None and abs(self.state.velocity - self.target_velocity) < 2:
            self.state.velocity = self.target_velocity
            self.target_velocity = None
            self.state.acceleration = 0

        if self.target_orientation is not None and abs(geometry.normalise_angle(self.target_orientation - self.state.orientation)) < (geometry.DEG2RAD * 5):  # experimentally, pedestrians seem to require a +/- 5 degrees error bound
            self.state.orientation = self.target_orientation
            self.target_orientation = None
            self.state.angular_velocity = 0


class Pedestrian(DynamicActor):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)


@dataclass(frozen=True)
class SpawnPedestrianState:
    positions: list
    velocity: float
    orientations: list
    acceleration: int
    angular_velocity: float


class SpawnPedestrian(Pedestrian):
    def __init__(self, spawn_init_state, constants, np_random=seeding.np_random(None)[0]):
        self.spawn_init_state = spawn_init_state

        self.np_random = np_random

        super().__init__(self.spawn(), constants)

    def reset(self):
        self.init_state = self.spawn()
        super().reset()

    def spawn(self):
        position = self.np_random.choice(self.spawn_init_state.positions)
        orientation = self.np_random.choice(self.spawn_init_state.orientations)
        return DynamicActorState(
            position=position,
            velocity=self.spawn_init_state.velocity,
            orientation=orientation,
            acceleration=self.spawn_init_state.acceleration,
            angular_velocity=self.spawn_init_state.angular_velocity
        )


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
        super().__init__(init_state=init_state, constants=constants)

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
