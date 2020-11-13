from abc import ABC

import math
from copy import copy
from dataclasses import dataclass
from enum import Enum

# noinspection PyPackageRequirements
import numpy as np  # dependency of gym
from gym import spaces
from gym.utils import seeding

from library import geometry
from library.actions import TrafficLightAction
from library.assets import Road, Occlusion
from library.geometry import Point

REACTION_TIME = 0.675


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

    def observation_space(self):
        raise NotImplementedError

    def action_space(self):
        raise NotImplementedError


@dataclass
class DynamicActorState:
    position: geometry.Point
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
    track: float

    min_velocity: float
    max_velocity: float
    min_throttle: float
    max_throttle: float
    min_steering_angle: float
    max_steering_angle: float


class DynamicActor(Actor, Occlusion):
    def __init__(self, init_state, constants):
        super().__init__(init_state=init_state, constants=constants)

        self.shape = geometry.make_rectangle(self.constants.length, self.constants.width)

        self.wheelbase_offset = self.constants.wheelbase / 2.0

        self.wheels = geometry.make_rectangle(self.constants.wheelbase, self.constants.track)

        self.throttle = 0.0
        self.steering_angle = 0.0

    def observation_space(self):
        return spaces.Box(
            low=np.array([-math.inf, -math.inf, self.constants.min_velocity, -math.pi], dtype=np.float32),
            high=np.array([math.inf, math.inf, self.constants.max_velocity, math.pi], dtype=np.float32),
            dtype=np.float32
        )  # x, y, velocity, orientation

    def action_space(self):
        return spaces.Box(
            low=np.array([self.constants.min_throttle, self.constants.min_steering_angle], dtype=np.float32),
            high=np.array([self.constants.max_throttle, self.constants.max_steering_angle], dtype=np.float32),
            dtype=np.float32
        )  # throttle, steering_angle

    def reset(self):
        super().reset()

        self.throttle = 0.0
        self.steering_angle = 0.0

    def bounding_box(self):
        return self.shape.transform(self.state.orientation, self.state.position)

    def wheel_positions(self):
        return self.wheels.transform(self.state.orientation, self.state.position)

    def stopping_zones(self):
        return None, None
        #
        # braking_distance = (self.state.velocity ** 2) / (2 * -self.constants.min_throttle)
        # reaction_distance = self.state.velocity * REACTION_TIME
        # total_distance = braking_distance + reaction_distance
        #
        # if total_distance == 0:
        #     return None, None
        #
        # if self.steering_angle == 0:
        #     zone = geometry.make_rectangle(total_distance, self.constants.width, rear_offset=0).transform(self.state.orientation, Point(self.constants.length * 0.5, 0).transform(self.state.orientation, self.state.position))
        #     braking_zone, reaction_zone = zone.split_longitudinally(braking_distance / total_distance)
        #     return braking_zone, reaction_zone
        #
        # front_remaining_length = (self.constants.length - self.constants.wheelbase) / 2.0
        #
        # def find_radius(angle, opposite):
        #     opposite_to_adjacent_ratio = abs(math.tan(angle))
        #     adjacent = opposite / opposite_to_adjacent_ratio
        #     return adjacent
        #
        # radius = find_radius(self.steering_angle, self.constants.wheelbase)  # distance from centre of rotation to centre of rear axle
        #
        # counter_clockwise = self.steering_angle > 0
        #
        # rear_axle_centre_point = self.wheel_positions().rear_centre()
        # relative_rotation_angle = self.state.orientation + (math.pi / 2.0) if counter_clockwise else self.state.orientation - (math.pi / 2.0)
        # rotation_point = Point(radius, 0).transform(relative_rotation_angle, rear_axle_centre_point)
        #
        # circle_motion = geometry.Circle(centre=rotation_point, radius=radius)
        # current_angle = self.state.orientation - (math.pi / 2.0) if counter_clockwise else self.state.orientation + (math.pi / 2.0)
        #
        # radius_min = radius - (self.constants.width / 2.0)
        # radius_mid = math.sqrt((radius_min ** 2) + ((self.constants.wheelbase + front_remaining_length) ** 2))
        # radius_max = math.sqrt(((radius_min + self.constants.width) ** 2) + ((self.constants.wheelbase + front_remaining_length) ** 2))
        # radius_front_centre = math.sqrt((radius ** 2) + ((self.constants.wheelbase + front_remaining_length) ** 2))
        #
        # circle_min = geometry.Circle(centre=rotation_point, radius=radius_min)
        # circle_mid = geometry.Circle(centre=rotation_point, radius=radius_mid)
        # circle_max = geometry.Circle(centre=rotation_point, radius=radius_max)
        # circle_front_centre = geometry.Circle(centre=rotation_point, radius=radius_front_centre)
        #
        # mid_corner = self.bounding_box().front_left if counter_clockwise else self.bounding_box().front_right
        # max_corner = self.bounding_box().front_right if counter_clockwise else self.bounding_box().front_left
        #
        # angle_mid = geometry.Triangle(rear=rotation_point, front_left=mid_corner, front_right=rear_axle_centre_point).angle()
        # angle_max = geometry.Triangle(rear=rotation_point, front_left=max_corner, front_right=rear_axle_centre_point).angle()
        # angle_front_centre = geometry.Triangle(rear=rotation_point, front_left=self.bounding_box().front_centre(), front_right=rear_axle_centre_point).angle()
        #
        # target_orientation = math.degrees(90)  # test
        #
        # target_angle = geometry.normalise_angle(target_orientation - (math.pi / 2.0) if counter_clockwise else target_orientation + (math.pi / 2.0))
        # target_arc = circle_motion.arc_from_angle(current_angle, geometry.normalise_angle(target_angle - current_angle))
        # target_arc_length = target_arc.arc_length()
        #
        # zone_arc = circle_motion.arc_from_length(current_angle, min(total_distance, target_arc_length) if counter_clockwise else -min(total_distance, target_arc_length))
        #
        # zone_min_arc = circle_min.arc_from_angle(current_angle, zone_arc.arc_angle)
        # zone_mid_arc = circle_mid.arc_from_angle(geometry.normalise_angle(current_angle + angle_mid), zone_arc.arc_angle)
        # zone_max_arc = circle_max.arc_from_angle(geometry.normalise_angle(current_angle + angle_max), zone_arc.arc_angle)
        # zone_front_centre_arc = circle_front_centre.arc_from_angle(geometry.normalise_angle(current_angle + angle_front_centre), zone_arc.arc_angle)
        #
        # zone_arrow = geometry.Arrow(left_arc=zone_min_arc if counter_clockwise else zone_max_arc, centre_arc=zone_mid_arc, right_arc=zone_max_arc if counter_clockwise else zone_min_arc)
        # zone_arrow_length = zone_arc.arc_length()
        # zone_straight_length = total_distance - zone_arrow_length
        #
        # if zone_straight_length > 0:
        #     zone_straight_orientation = geometry.normalise_angle(target_angle + (math.pi / 2.0) if counter_clockwise else target_angle - (math.pi / 2.0))
        #     zone_straight = geometry.make_rectangle(zone_straight_length, self.constants.width, rear_offset=0).transform(zone_straight_orientation, zone_front_centre_arc.end_point())
        #
        #     if braking_distance < zone_arrow_length:  # braking zone entirely witin curve
        #         braking_zone_arrow, reaction_zone_arrow = zone_arrow.split_longitudinally(braking_distance / zone_arrow_length)
        #         braking_zone = braking_zone_arrow
        #         reaction_zone = geometry.Zone(curve=reaction_zone_arrow, straight=zone_straight)
        #     else:  # reaction zone entirely witin straight
        #         braking_zone_straight, reaction_zone_straight = zone_straight.split_longitudinally((braking_distance - zone_arrow_length) / zone_straight_length)
        #         braking_zone = geometry.Zone(curve=zone_arrow, straight=braking_zone_straight)
        #         reaction_zone = reaction_zone_straight
        # else:
        #     braking_zone, reaction_zone = zone_arrow.split_longitudinally(braking_distance / total_distance)
        #
        # return braking_zone, reaction_zone

    def line_anchor(self, road):
        closest = road.bounding_box().longitudinal_line().closest_point_from(self.state.position)
        return geometry.Line(self.state.position, closest)

    def line_anchor_relative_angle(self, road):
        angle = self.line_anchor(road).orientation()
        return geometry.normalise_angle(angle - self.state.orientation)

    def step(self, action, time_resolution):
        self.throttle, self.steering_angle = action

        successor_velocity = max(
            self.constants.min_velocity,
            min(
                self.constants.max_velocity,
                self.state.velocity + (self.throttle * time_resolution)
            )
        )

        distance_velocity = self.state.velocity * time_resolution
        cos_orientation = math.cos(self.state.orientation)
        sin_orientation = math.sin(self.state.orientation)

        if self.steering_angle == 0:
            self.state = DynamicActorState(
                position=Point(
                    x=self.state.position.x + distance_velocity * cos_orientation,
                    y=self.state.position.y + distance_velocity * sin_orientation
                ),
                velocity=successor_velocity,
                orientation=self.state.orientation
            )
        else:
            rear_position = Point(
                x=self.state.position.x - self.wheelbase_offset * cos_orientation,
                y=self.state.position.y - self.wheelbase_offset * sin_orientation
            )

            wheelbase_tan_steering_angle = self.constants.wheelbase / math.tan(self.steering_angle)

            centre_of_rotation = Point(
                x=rear_position.x - wheelbase_tan_steering_angle * sin_orientation,
                y=rear_position.y + wheelbase_tan_steering_angle * cos_orientation
            )

            diff_x = self.state.position.x - centre_of_rotation.x
            diff_y = self.state.position.y - centre_of_rotation.y

            theta = (-1 if self.steering_angle < 0 else 1) * (distance_velocity / math.sqrt(diff_x ** 2 + diff_y ** 2))

            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            orientation_theta = self.state.orientation + theta

            self.state = DynamicActorState(
                position=Point(
                    x=centre_of_rotation.x + diff_x * cos_theta - diff_y * sin_theta,
                    y=centre_of_rotation.y + diff_x * sin_theta + diff_y * cos_theta
                ),
                velocity=successor_velocity,
                orientation=math.atan2(math.sin(orientation_theta), math.cos(orientation_theta))
            )


class Pedestrian(DynamicActor):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)


@dataclass(frozen=True)
class SpawnPedestrianState:
    position_boxes: list
    velocity: float
    orientations: list


class SpawnPedestrian(Pedestrian):
    def __init__(self, spawn_init_state, constants, np_random=seeding.np_random(None)[0]):
        self.spawn_init_state = spawn_init_state

        self.np_random = np_random

        super().__init__(self.spawn(), constants)

    def reset(self):
        self.init_state = self.spawn()
        super().reset()

    def spawn(self):
        areas = [box.area() for box in self.spawn_init_state.position_boxes]
        total_area = sum(areas)
        chosen_box = self.np_random.choice(self.spawn_init_state.position_boxes, p=[area / total_area for area in areas])
        position = chosen_box.random_point(self.np_random)
        orientation = self.np_random.choice(self.spawn_init_state.orientations)
        return DynamicActorState(
            position=position,
            velocity=self.spawn_init_state.velocity,
            orientation=orientation
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

    def __iter__(self):
        yield self


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

    def observation_space(self):
        return [state for state in TrafficLightState]

    def action_space(self):
        return [action for action in TrafficLightAction]

    def bounding_box(self):
        return self.static_bounding_box

    def step(self, action, time_resolution):
        traffic_light_action = TrafficLightAction(action)

        if traffic_light_action is TrafficLightAction.TURN_RED:
            self.state = TrafficLightState.RED
        elif traffic_light_action is TrafficLightAction.TURN_AMBER:
            self.state = TrafficLightState.AMBER
        elif traffic_light_action is TrafficLightAction.TURN_GREEN:
            self.state = TrafficLightState.GREEN


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

        self.outbound_intersection_bounding_box = geometry.make_rectangle(self.constants.width, self.constants.road.outbound.width).transform(self.constants.road.constants.orientation, Point(0, (self.constants.road.inbound.width - position.y) * 0.5).translate(position))
        self.inbound_intersection_bounding_box = geometry.make_rectangle(self.constants.width, self.constants.road.inbound.width).transform(self.constants.road.constants.orientation, Point(0, -(self.constants.road.outbound.width - position.y) * 0.5).translate(position))

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

    def observation_space(self):
        return spaces.Discrete(TrafficLightState.__len__())

    def action_space(self):
        return spaces.Discrete(TrafficLightAction.__len__())

    def bounding_box(self):
        return self.static_bounding_box

    def step(self, action, time_resolution):
        pelican_crossing_action = TrafficLightAction(action)

        if pelican_crossing_action is TrafficLightAction.TURN_RED:
            self.state = TrafficLightState.RED
        elif pelican_crossing_action is TrafficLightAction.TURN_AMBER:
            self.state = TrafficLightState.AMBER
        elif pelican_crossing_action is TrafficLightAction.TURN_GREEN:
            self.state = TrafficLightState.GREEN

        self.outbound_traffic_light.step(action, time_resolution)
        self.inbound_traffic_light.step(action, time_resolution)
