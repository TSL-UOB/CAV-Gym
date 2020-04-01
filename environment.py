import dataclasses
import math
from copy import copy
from enum import Enum

import numpy as np
from gym import Env, spaces
from gym.envs.classic_control.rendering import Line, LineStyle
from gym.utils import seeding
from shapely import geometry, affinity

DEG2RAD = 0.017453292519943295


class MarkovGameEnv(Env):
    r"""
    A multi-agent extension of the main OpenAI Gym class,
    which assumes that agents execute actions in each time step simulataneously (as in Markov games).
    """

    def step(self, joint_action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts a joint action and returns a tuple (joint_observation, joint_reward, done, info).

        Args:
            joint_action (dict(object)): an action provided by each agent

        Returns:
            joint_observation (dict(object)): each agent's observation of the current environment
            joint_reward (dict(float)) : each agent's amount of reward returned after previous joint action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def reset(self):
        """Resets the state of the environment and returns an initial joint observation.

        Returns:
            observation (dict(object)): the initial joint observation.
        """
        raise NotImplementedError

    def render(self, mode='human'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError


@dataclasses.dataclass
class Point:
    x: float
    y: float

    def __copy__(self):
        return Point(self.x, self.y)


@dataclasses.dataclass
class VehicleState:
    position: Point
    diagonal_velocity: float
    throttle: float
    orientation: float
    turn: float

    def __copy__(self):
        return VehicleState(copy(self.position), self.diagonal_velocity, self.throttle, self.orientation, self.turn)


@dataclasses.dataclass(frozen=True)
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


class Vehicle:
    def __init__(self, init_state, config=VehicleConfig()):
        self.init_state = init_state
        self.config = config

        self.state = copy(self.init_state)

    def reset(self):
        self.state = copy(self.init_state)

    def bounds(self):
        min_x, min_y = -self.config.length / 2.0, -self.config.width / 2.0
        max_x, max_y = self.config.length / 2.0, self.config.width / 2.0
        return min_x, min_y, max_x, max_y

    def boundary_points(self):
        min_x, min_y, max_x, max_y = self.bounds()
        return Point(max_x, min_y), Point(max_x, max_y), Point(min_x, max_y), Point(min_x, min_y)

    def left_indicator_points(self):
        min_x, _, max_x, max_y = self.bounds()
        offset = self.config.width * 0.25
        return Point(max_x - offset, max_y), Point(min_x + offset, max_y)

    def right_indicator_points(self):
        min_x, min_y, max_x, _ = self.bounds()
        offset = self.config.width * 0.25
        return Point(max_x - offset, min_y), Point(min_x + offset, min_y)

    def brake_light_points(self):
        min_x, min_y, _, max_y = self.bounds()
        offset = self.config.width * 0.25
        return Point(min_x, max_y - offset), Point(min_x, min_y + offset)

    def headlight_points(self):
        _, min_y, max_x, max_y = self.bounds()
        offset = self.config.width * 0.25
        return Point(max_x, max_y - offset), Point(max_x, min_y + offset)

    def intersects(self, other_vehicle):
        def contour(vehicle):
            min_x, min_y, max_x, max_y = self.bounds()
            box = geometry.box(min_x, min_y, max_x, max_y)
            rotated_box = affinity.rotate(box, vehicle.state.orientation, use_radians=True)
            return affinity.translate(rotated_box, vehicle.state.position.x, vehicle.state.position.y)
        return contour(self).intersects(contour(other_vehicle))


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


class Observation(Enum):
    NONE = 0

    def __repr__(self):
        return self.name


@dataclasses.dataclass(frozen=True)
class RoadEnvConfig:
    viewer_width: int = 1600
    viewer_height: int = 100
    time_resolution: float = 1.0 / 60.0


@dataclasses.dataclass(frozen=True)
class Lane:
    min_y: float
    max_y: float


class JointReward(list):
    def __iadd__(self, other):
        for i, value in enumerate(other):
            self[i] += value

    def __float__(self):
        return [float(value) for value in self]


class RoadEnv(MarkovGameEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, vehicles, config=RoadEnvConfig(), seed=None):
        self.np_random = None
        self.seed(seed)
        self.vehicles = vehicles
        self.config = config

        self.action_space = spaces.Tuple([spaces.Tuple([spaces.Discrete(ThrottleAction.__len__()), spaces.Discrete(TurnAction.__len__())]) for _ in self.vehicles])
        self.observation_space = spaces.Tuple([spaces.Discrete(1) for _ in self.vehicles])

        # self.action_space.seed(seed)
        # self.observation_space.seed(seed)

        # self.action_space = spaces.Tuple([spaces.MultiDiscrete([ThrottleAction.__len__(), TurnAction.__len__()]) for _ in self.vehicles])
        # self.observation_space = spaces.Tuple([spaces.Discrete(1) for _ in self.vehicles])
        #
        # for space in self.action_space:
        #     seed += 11
        #     space.seed(seed)
        #
        # for _ in range(10):
        #     print(self.action_space.sample())

        self.viewer = None
        self.vehicle_transforms = None
        self.left_indicator_transforms = None
        self.right_indicator_transforms = None
        self.brake_light_transforms = None
        self.headlight_transforms = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, joint_action):
        assert self.action_space.contains(joint_action), "%r (%s) invalid" % (joint_action, type(joint_action))

        for vehicle, action_ids in zip(self.vehicles, joint_action):
            throttle_action_id, turn_action_id = action_ids
            throttle_action = ThrottleAction(throttle_action_id)
            turn_action = TurnAction(turn_action_id)

            if throttle_action is ThrottleAction.NEUTRAL:
                vehicle.state.throttle = 0
            elif throttle_action is ThrottleAction.ACCELERATE:
                vehicle.state.throttle = vehicle.config.speed_accelerate
            elif throttle_action is ThrottleAction.DECELERATE:
                vehicle.state.throttle = vehicle.config.speed_decelerate
            elif throttle_action is ThrottleAction.HARD_ACCELERATE:
                vehicle.state.throttle = vehicle.config.speed_hard_accelerate
            elif throttle_action is ThrottleAction.HARD_DECELERATE:
                vehicle.state.throttle = vehicle.config.speed_hard_decelerate

            if turn_action is TurnAction.NEUTRAL:
                vehicle.state.turn = 0
            elif turn_action is TurnAction.LEFT:
                vehicle.state.turn = vehicle.config.radians_left
            elif turn_action is TurnAction.RIGHT:
                vehicle.state.turn = vehicle.config.radians_right
            elif turn_action is TurnAction.HARD_LEFT:
                vehicle.state.turn = vehicle.config.radians_hard_left
            elif turn_action is TurnAction.HARD_RIGHT:
                vehicle.state.turn = vehicle.config.radians_hard_right

            vehicle.state.diagonal_velocity = max(
                vehicle.config.min_diagonal_velocity,
                min(
                    vehicle.config.max_diagonal_velocity,
                    vehicle.state.diagonal_velocity + (vehicle.state.throttle * self.config.time_resolution)
                )
            )

            """Simple vehicle dynamics: http://engineeringdotnet.blogspot.com/2010/04/simple-2d-car-physics-in-games.html"""

            front_wheel_position = Point(
                vehicle.state.position.x + ((vehicle.config.wheel_base / 2.0) * math.cos(vehicle.state.orientation)),
                vehicle.state.position.y + ((vehicle.config.wheel_base / 2.0) * math.sin(vehicle.state.orientation))
            )
            front_wheel_position.x += vehicle.state.diagonal_velocity * self.config.time_resolution * math.cos(vehicle.state.orientation + vehicle.state.turn)
            front_wheel_position.y += vehicle.state.diagonal_velocity * self.config.time_resolution * math.sin(vehicle.state.orientation + vehicle.state.turn)

            back_wheel_position = Point(
                vehicle.state.position.x - ((vehicle.config.wheel_base / 2.0) * math.cos(vehicle.state.orientation)),
                vehicle.state.position.y - ((vehicle.config.wheel_base / 2.0) * math.sin(vehicle.state.orientation))
            )
            back_wheel_position.x += vehicle.state.diagonal_velocity * self.config.time_resolution * math.cos(vehicle.state.orientation)
            back_wheel_position.y += vehicle.state.diagonal_velocity * self.config.time_resolution * math.sin(vehicle.state.orientation)

            vehicle.state.position.x = (front_wheel_position.x + back_wheel_position.x) / 2.0
            vehicle.state.position.y = (front_wheel_position.y + back_wheel_position.y) / 2.0

            vehicle.state.orientation = math.atan2(front_wheel_position.y - back_wheel_position.y, front_wheel_position.x - back_wheel_position.x)

        joint_reward = [-1 if any(vehicle.intersects(other_vehicle) for other_vehicle in self.vehicles if vehicle is not other_vehicle) else 0 for vehicle in self.vehicles]

        return self.observation_space.sample(), JointReward(joint_reward), any(reward < 0 for reward in joint_reward), None

    def reset(self):
        for vehicle in self.vehicles:
            vehicle.reset()
        return self.observation_space.sample()

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        from gym.envs.classic_control.rendering import FilledPolygon
        if not self.viewer:
            self.viewer = rendering.Viewer(self.config.viewer_width, self.config.viewer_height)

            self.viewer.add_geom(Line((0, self.config.viewer_height * 0.1), (self.config.viewer_width, self.config.viewer_height * 0.1)))
            self.viewer.add_geom(Line((0, self.config.viewer_height * 0.9), (self.config.viewer_width, self.config.viewer_height * 0.9)))
            lane_line = Line((0, self.config.viewer_height * 0.5), (self.config.viewer_width, self.config.viewer_height * 0.5))
            lane_line.add_attr(LineStyle(0x0FF0))
            self.viewer.add_geom(lane_line)

            self.vehicle_transforms = [rendering.Transform(translation=(vehicle.state.position.x, vehicle.state.position.y), rotation=vehicle.state.orientation) for vehicle in self.vehicles]
            self.left_indicator_transforms = [tuple(rendering.Transform(translation=(point.x, point.y), scale=(0, 0)) for point in vehicle.left_indicator_points()) for vehicle in self.vehicles]
            self.right_indicator_transforms = [tuple(rendering.Transform(translation=(point.x, point.y), scale=(0, 0)) for point in vehicle.right_indicator_points()) for vehicle in self.vehicles]
            self.brake_light_transforms = [tuple(rendering.Transform(translation=(point.x, point.y), scale=(0, 0)) for point in vehicle.brake_light_points()) for vehicle in self.vehicles]
            self.headlight_transforms = [tuple(rendering.Transform(translation=(point.x, point.y), scale=(0, 0)) for point in vehicle.headlight_points()) for vehicle in self.vehicles]

            for vehicle, vehicle_transform, left_indicator_transforms, right_indicator_transforms, brake_light_transforms, headlight_transforms in zip(self.vehicles, self.vehicle_transforms, self.left_indicator_transforms, self.right_indicator_transforms, self.brake_light_transforms, self.headlight_transforms):
                vehicle_polygon = FilledPolygon([(point.x, point.y) for point in vehicle.boundary_points()])
                vehicle_polygon.add_attr(vehicle_transform)

                for indicator_transform in left_indicator_transforms + right_indicator_transforms:
                    indicator = rendering.make_circle(4)
                    indicator.set_color(1, 0.75, 0)
                    indicator.add_attr(indicator_transform)
                    indicator.add_attr(vehicle_transform)
                    self.viewer.add_geom(indicator)

                for brake_light_transform in brake_light_transforms:
                    brake_light = rendering.make_circle(4)
                    brake_light.set_color(1, 0, 0)
                    brake_light.add_attr(brake_light_transform)
                    brake_light.add_attr(vehicle_transform)
                    self.viewer.add_geom(brake_light)

                for headlight_transform in headlight_transforms:
                    headlight = rendering.make_circle(4)
                    headlight.set_color(1, 1, 0)
                    headlight.add_attr(headlight_transform)
                    headlight.add_attr(vehicle_transform)
                    self.viewer.add_geom(headlight)

                self.viewer.add_geom(vehicle_polygon)
        else:
            for vehicle, vehicle_transform, left_indicator_transforms, right_indicator_transforms, brake_light_transforms, headlight_transforms in zip(self.vehicles, self.vehicle_transforms, self.left_indicator_transforms, self.right_indicator_transforms, self.brake_light_transforms, self.headlight_transforms):
                vehicle_transform.set_translation(vehicle.state.position.x, vehicle.state.position.y)
                vehicle_transform.set_rotation(vehicle.state.orientation)

                if vehicle.state.turn > 0:
                    for indicator_transform in left_indicator_transforms:
                        if vehicle.state.turn == vehicle.config.radians_left:
                            indicator_transform.set_scale(0.75, 0.75)
                        else:
                            indicator_transform.set_scale(1, 1)
                    for indicator_transform in right_indicator_transforms:
                        indicator_transform.set_scale(0, 0)
                elif vehicle.state.turn < 0:
                    for indicator_transform in right_indicator_transforms:
                        if vehicle.state.turn == vehicle.config.radians_right:
                            indicator_transform.set_scale(0.75, 0.75)
                        else:
                            indicator_transform.set_scale(1, 1)
                    for indicator_transform in left_indicator_transforms:
                        indicator_transform.set_scale(0, 0)
                else:
                    for indicator_transform in right_indicator_transforms + left_indicator_transforms:
                        indicator_transform.set_scale(0, 0)

                if vehicle.state.throttle < 0:
                    for brake_light_transform in brake_light_transforms:
                        if vehicle.state.throttle == vehicle.config.speed_decelerate:
                            brake_light_transform.set_scale(0.75, 0.75)
                        else:
                            brake_light_transform.set_scale(1, 1)
                    for headlight_transform in headlight_transforms:
                        headlight_transform.set_scale(0, 0)
                elif vehicle.state.throttle > 0:
                    for headlight_transform in headlight_transforms:
                        if vehicle.state.throttle == vehicle.config.speed_accelerate:
                            headlight_transform.set_scale(0.75, 0.75)
                        else:
                            headlight_transform.set_scale(1, 1)
                    for brake_light_transform in brake_light_transforms:
                        brake_light_transform.set_scale(0, 0)
                else:
                    for transform in brake_light_transforms + headlight_transforms:
                        transform.set_scale(0, 0)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
