import math
from copy import copy
from dataclasses import dataclass
from enum import Enum

import gym
from gym import spaces
from gym.utils import seeding
from shapely import geometry, affinity

from rendering import RoadEnvViewer

DEG2RAD = 0.017453292519943295


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


@dataclass
class Point:
    x: float
    y: float

    def __copy__(self):
        return Point(self.x, self.y)


class Actor:
    def __init__(self, init_state):
        self.init_state = init_state
        self.state = copy(self.init_state)

        self.bounds = None

    def reset(self):
        raise NotImplementedError

    def bounding_box(self):
        raise NotImplementedError

    def intersects(self, other):
        return self.bounding_box().intersects(other.bounding_box())

    def step_action(self, joint_action, index):
        raise NotImplementedError

    def step_dynamics(self, time_resolution):
        raise NotImplementedError


@dataclass
class VehicleState:
    position: Point
    orientation: float
    diagonal_velocity: float
    throttle: float
    turn: float

    def __copy__(self):
        return VehicleState(copy(self.position), self.orientation, self.diagonal_velocity, self.throttle, self.turn)


@dataclass(frozen=True)
class VehicleConfig:
    length: float = 50.0
    width: float = 20.0
    wheel_base: float = 45.0

    min_diagonal_velocity: float = 0.0
    max_diagonal_velocity: float = 200.0

    speed_accelerate: float = 20.0
    speed_decelerate: float = -40.0
    speed_hard_accelerate: float = 60.0
    speed_hard_decelerate: float = -80.0
    radians_left: float = DEG2RAD * 15.0
    radians_right: float = DEG2RAD * -15.0
    radians_hard_left: float = DEG2RAD * 45.0
    radians_hard_right: float = DEG2RAD * -45.0


class ThrottleAction(Enum):
    NEUTRAL = 0
    ACCELERATE = 1
    DECELERATE = 2
    HARD_ACCELERATE = 3
    HARD_DECELERATE = 4

    def __repr__(self):
        return self.name


class TurnAction(Enum):
    NEUTRAL = 0
    LEFT = 1
    RIGHT = 2
    HARD_LEFT = 3
    HARD_RIGHT = 4

    def __repr__(self):
        return self.name


class Vehicle(Actor):
    def __init__(self, init_state, config=VehicleConfig()):
        super().__init__(init_state)
        self.config = config

        min_x, max_x = -self.config.length / 2.0, self.config.length / 2.0
        min_y, max_y = -self.config.width / 2.0, self.config.width / 2.0
        self.bounds = min_x, min_y, max_x, max_y

    def reset(self):
        self.state = copy(self.init_state)

    def bounding_box(self):
        assert self.bounds is not None
        box = geometry.box(*self.bounds)
        rotated_box = affinity.rotate(box, self.state.orientation, use_radians=True)
        return affinity.translate(rotated_box, self.state.position.x, self.state.position.y)

    def step_action(self, joint_action, index):
        throttle_action_id, turn_action_id = joint_action[index]
        throttle_action = ThrottleAction(throttle_action_id)
        turn_action = TurnAction(turn_action_id)

        if throttle_action is ThrottleAction.NEUTRAL:
            self.state.throttle = 0
        elif throttle_action is ThrottleAction.ACCELERATE:
            self.state.throttle = self.config.speed_accelerate
        elif throttle_action is ThrottleAction.DECELERATE:
            self.state.throttle = self.config.speed_decelerate
        elif throttle_action is ThrottleAction.HARD_ACCELERATE:
            self.state.throttle = self.config.speed_hard_accelerate
        elif throttle_action is ThrottleAction.HARD_DECELERATE:
            self.state.throttle = self.config.speed_hard_decelerate

        if turn_action is TurnAction.NEUTRAL:
            self.state.turn = 0
        elif turn_action is TurnAction.LEFT:
            self.state.turn = self.config.radians_left
        elif turn_action is TurnAction.RIGHT:
            self.state.turn = self.config.radians_right
        elif turn_action is TurnAction.HARD_LEFT:
            self.state.turn = self.config.radians_hard_left
        elif turn_action is TurnAction.HARD_RIGHT:
            self.state.turn = self.config.radians_hard_right

    def step_dynamics(self, time_resolution):
        self.state.diagonal_velocity = max(
            self.config.min_diagonal_velocity,
            min(
                self.config.max_diagonal_velocity,
                self.state.diagonal_velocity + (self.state.throttle * time_resolution)
            )
        )

        """Simple vehicle dynamics: http://engineeringdotnet.blogspot.com/2010/04/simple-2d-car-physics-in-games.html"""

        front_wheel_position = Point(
            self.state.position.x + ((self.config.wheel_base / 2.0) * math.cos(self.state.orientation)),
            self.state.position.y + ((self.config.wheel_base / 2.0) * math.sin(self.state.orientation))
        )
        front_wheel_position.x += self.state.diagonal_velocity * time_resolution * math.cos(self.state.orientation + self.state.turn)
        front_wheel_position.y += self.state.diagonal_velocity * time_resolution * math.sin(self.state.orientation + self.state.turn)

        back_wheel_position = Point(
            self.state.position.x - ((self.config.wheel_base / 2.0) * math.cos(self.state.orientation)),
            self.state.position.y - ((self.config.wheel_base / 2.0) * math.sin(self.state.orientation))
        )
        back_wheel_position.x += self.state.diagonal_velocity * time_resolution * math.cos(self.state.orientation)
        back_wheel_position.y += self.state.diagonal_velocity * time_resolution * math.sin(self.state.orientation)

        self.state.position.x = (front_wheel_position.x + back_wheel_position.x) / 2.0
        self.state.position.y = (front_wheel_position.y + back_wheel_position.y) / 2.0

        self.state.orientation = math.atan2(front_wheel_position.y - back_wheel_position.y, front_wheel_position.x - back_wheel_position.x)


class TrafficLightState(Enum):
    RED = 0
    AMBER = 1
    GREEN = 2

    def __copy__(self):
        return TrafficLightState(self)


@dataclass(frozen=True)
class TrafficLightConfig:
    position: Point
    height: float = 20.0
    width: float = 10.0
    orientation: float = 0.0


class TrafficLightAction(Enum):
    NOOP = 0
    TURN_RED = 1
    TURN_AMBER = 2
    TURN_GREEN = 3

    def __repr__(self):
        return self.name


class TrafficLight(Actor):
    def __init__(self, init_state, config):
        self.init_state = init_state
        self.config = config

        self.state = copy(self.init_state)

        min_x, max_x = -self.config.width / 2.0, self.config.width / 2.0
        min_y, max_y = -self.config.height / 2.0, self.config.height / 2.0
        self.bounds = min_x, min_y, max_x, max_y

        box = geometry.box(*self.bounds)
        rotated_box = affinity.rotate(box, self.config.orientation, use_radians=True)
        self.static_bounding_box = affinity.translate(rotated_box, self.config.position.x, self.config.position.y)

    def reset(self):
        self.state = copy(self.init_state)

    def bounding_box(self):
        return self.static_bounding_box

    def intersects(self, other):
        return self.bounding_box().intersects(other.bounding_box())

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


class Observation(Enum):
    NONE = 0

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class RoadEnvConfig:
    viewer_width: int = 1600
    viewer_height: int = 100
    time_resolution: float = 1.0 / 60.0


class RoadEnv(MarkovGameEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, actors, config=RoadEnvConfig(), seed=None):
        self.np_random = None
        self.seed(seed)
        self.actors = actors
        self.config = config

        actor_spaces = list()
        for actor in self.actors:
            if isinstance(actor, Vehicle):
                actor_spaces.append(spaces.Tuple([spaces.Discrete(ThrottleAction.__len__()), spaces.Discrete(TurnAction.__len__())]))
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
            actor.step_dynamics(self.config.time_resolution)

        joint_reward = [-1 if any(vehicle.intersects(other_vehicle) for other_vehicle in self.actors if vehicle is not other_vehicle) else 0 for vehicle in self.actors]

        return self.observation_space.sample(), joint_reward, any(reward < 0 for reward in joint_reward), None

    def reset(self):
        for vehicle in self.actors:
            vehicle.reset()
        return self.observation_space.sample()

    def render(self, mode='human'):
        if not self.viewer:
            self.viewer = RoadEnvViewer(self.config.viewer_width, self.config.viewer_height, self.actors)
        else:
            self.viewer.update(self.actors)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
