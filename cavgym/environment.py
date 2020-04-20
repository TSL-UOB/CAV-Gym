import math
from copy import copy
from dataclasses import dataclass
from enum import Enum

import gym
from gym import spaces
from gym.utils import seeding
from shapely import geometry, affinity

from cavgym.actions import TrafficLightAction, AccelerationAction, TurnAction
from cavgym.rendering import RoadEnvViewer

DEG2RAD = 0.017453292519943295

REACTION_TIME = 0.675


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
class Point:
    x: float
    y: float

    def __copy__(self):
        return Point(self.x, self.y)


@dataclass
class DynamicActorState:
    position: Point
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

        front_wheel_position = Point(
            self.state.position.x + ((self.constants.wheelbase / 2.0) * math.cos(self.state.orientation)),
            self.state.position.y + ((self.constants.wheelbase / 2.0) * math.sin(self.state.orientation))
        )
        front_wheel_position.x += self.state.velocity * time_resolution * math.cos(self.state.orientation + self.state.angular_velocity)
        front_wheel_position.y += self.state.velocity * time_resolution * math.sin(self.state.orientation + self.state.angular_velocity)

        back_wheel_position = Point(
            self.state.position.x - ((self.constants.wheelbase / 2.0) * math.cos(self.state.orientation)),
            self.state.position.y - ((self.constants.wheelbase / 2.0) * math.sin(self.state.orientation))
        )
        back_wheel_position.x += self.state.velocity * time_resolution * math.cos(self.state.orientation)
        back_wheel_position.y += self.state.velocity * time_resolution * math.sin(self.state.orientation)

        self.state.position.x = (front_wheel_position.x + back_wheel_position.x) / 2.0
        self.state.position.y = (front_wheel_position.y + back_wheel_position.y) / 2.0

        self.state.orientation = math.atan2(front_wheel_position.y - back_wheel_position.y, front_wheel_position.x - back_wheel_position.x)


class Vehicle(DynamicActor):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)

    def stopping_bounds(self):
        assert self.bounds is not None
        rear, right, front, left = self.bounds
        braking_distance = (self.state.velocity ** 2) / (2 * -self.constants.hard_deceleration)
        reaction_distance = self.state.velocity * REACTION_TIME
        front_braking = front + braking_distance
        return (front, right, front_braking, left), (front_braking, right, front_braking + reaction_distance, left)

    def reaction_zone(self):
        _, reaction_bounds = self.stopping_bounds()
        box = geometry.box(*reaction_bounds)
        rotated_box = affinity.rotate(box, self.state.orientation, use_radians=True)
        return affinity.translate(rotated_box, self.state.position.x, self.state.position.y)


class Pedestrian(DynamicActor):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)


class TrafficLightState(Enum):
    RED = 0
    AMBER = 1
    GREEN = 2

    def __copy__(self):
        return TrafficLightState(self)


@dataclass(frozen=True)
class StaticActorConstants:
    height: float
    width: float

    position: Point
    orientation: float


class StaticActor(Actor):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)

        rear, front = -self.constants.width / 2.0, self.constants.width / 2.0
        right, left = -self.constants.height / 2.0, self.constants.height / 2.0
        self.bounds = rear, right, front, left

        box = geometry.box(*self.bounds)
        rotated_box = affinity.rotate(box, self.constants.orientation, use_radians=True)
        self.static_bounding_box = affinity.translate(rotated_box, self.constants.position.x, self.constants.position.y)

    def bounding_box(self):
        return self.static_bounding_box

    def step_action(self, joint_action, index):
        raise NotImplementedError

    def step_dynamics(self, time_resolution):
        raise NotImplementedError


class TrafficLight(StaticActor):
    def __init__(self, init_state, constants):
        super().__init__(init_state, constants)

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
class RoadConstants:
    length: float
    num_outbound_lanes: int
    num_inbound_lanes: int
    lane_width: float

    position: Point
    orientation: float


class Road:
    def __init__(self, constants):
        self.constants = constants

        self.num_lanes = self.constants.num_outbound_lanes + self.constants.num_inbound_lanes
        self.width = self.num_lanes * self.constants.lane_width

        rear, front = 0, self.constants.length
        right, left = -self.width / 2.0, self.width / 2.0
        self.bounds = rear, right, front, left

        left_inbound = right + (self.constants.num_inbound_lanes * self.constants.lane_width)
        self.inbound_bounds = rear, right, front, left_inbound
        self.outbound_bounds = rear, left_inbound, front, left_inbound + (self.constants.num_outbound_lanes * self.constants.lane_width)

        self.inbound_lanes_bounds = list(self._iter_lane_split(self.inbound_bounds, self.constants.num_inbound_lanes))
        self.outbound_lanes_bounds = list(self._iter_lane_split(self.outbound_bounds, self.constants.num_outbound_lanes))

        box = geometry.box(*self.bounds)
        rotated_box = affinity.rotate(box, self.constants.orientation, use_radians=True)
        self.static_bounding_box = affinity.translate(rotated_box, self.constants.position.x, self.constants.position.y)

    def _iter_lane_split(self, bounds, num_lanes):
        rear, right, front, _ = bounds
        lane_left = right
        for _ in range(num_lanes):
            lane_right = lane_left
            lane_left = lane_right + self.constants.lane_width
            yield rear, lane_right, front, lane_left


class RoadLayout:
    def __init__(self, main_road, minor_roads=None):
        self.main_road = main_road
        self.minor_roads = minor_roads


class MarkovGameEnv(gym.Env):
    r"""
    A multi-agent extension of the main OpenAI Gym class,
    which assumes that agents execute actions in each time step simulataneously (as in Markov games).
    """

    def step(self, joint_action):
        """Accepts a joint action and returns a tuple (joint_observation, joint_reward, done, info).

        Args:
            joint_action (list(object)): an action provided by each agent

        Returns:
            joint_observation (list(object)): each agent's observation of the current environment
            joint_reward (list(float)) : each agent's amount of reward returned after previous joint action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def reset(self):
        """Resets the state of the environment and returns an initial joint observation.

        Returns:
            observation (list(object)): the initial joint observation.
        """
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError


@dataclass(frozen=True)
class RoadEnvConstants:
    viewer_width: int
    viewer_height: int
    time_resolution: float
    road_layout: RoadLayout


class RoadEnv(MarkovGameEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, actors, constants, seed=None):
        self.np_random = None
        self.seed(seed)
        self.actors = actors
        self.constants = constants

        actor_spaces = list()
        for actor in self.actors:
            if isinstance(actor, Vehicle) or isinstance(actor, Pedestrian):
                actor_spaces.append(spaces.Tuple([spaces.Discrete(AccelerationAction.__len__()), spaces.Discrete(TurnAction.__len__())]))
            elif isinstance(actor, TrafficLight):
                actor_spaces.append(spaces.Discrete(TrafficLightAction.__len__()))
        self.action_space = spaces.Tuple(actor_spaces)
        self.observation_space = spaces.Tuple([spaces.Discrete(1) for _ in self.actors])

        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, joint_action):
        assert self.action_space.contains(joint_action), "%r (%s) invalid" % (joint_action, type(joint_action))

        for index, actor in enumerate(self.actors):
            actor.step_action(joint_action, index)

        for actor in self.actors:
            actor.step_dynamics(self.constants.time_resolution)

        joint_reward = [-1 if any(vehicle.intersects(other_vehicle) for other_vehicle in self.actors if vehicle is not other_vehicle) else 0 for vehicle in self.actors]

        return self.observation_space.sample(), joint_reward, any(reward < 0 for reward in joint_reward), None

    def reset(self):
        for vehicle in self.actors:
            vehicle.reset()
        return self.observation_space.sample()

    def render(self, mode='human'):
        if not self.viewer:
            self.viewer = RoadEnvViewer(self.constants.viewer_width, self.constants.viewer_height, self.constants.road_layout, self.actors)
        else:
            self.viewer.update(self.actors)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
