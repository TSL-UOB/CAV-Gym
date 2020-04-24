from dataclasses import dataclass

import gym
from gym import spaces
from gym.utils import seeding
from cavgym.actions import TrafficLightAction, AccelerationAction, TurnAction
from cavgym.actors import PelicanCrossing, Pedestrian, DynamicActor
from cavgym.rendering import RoadEnvViewer
from cavgym.assets import RoadMap


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
class CAVEnvConstants:
    viewer_width: int
    viewer_height: int
    time_resolution: float
    road_map: RoadMap


class CAVEnv(MarkovGameEnv):
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
            if isinstance(actor, DynamicActor):
                actor_spaces.append(spaces.Tuple([spaces.Discrete(AccelerationAction.__len__()), spaces.Discrete(TurnAction.__len__())]))
            elif isinstance(actor, PelicanCrossing):
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

        joint_reward = [-1 if any(vehicle.intersects(other_vehicle) for other_vehicle in self.actors if vehicle is not other_vehicle and not isinstance(vehicle, PelicanCrossing) and not isinstance(other_vehicle, PelicanCrossing)) else 0 for vehicle in self.actors]

        return self.observation_space.sample(), joint_reward, any(reward < 0 for reward in joint_reward), None

    def reset(self):
        for vehicle in self.actors:
            vehicle.reset()
        return self.observation_space.sample()

    def render(self, mode='human'):
        if not self.viewer:
            self.viewer = RoadEnvViewer(self.constants.viewer_width, self.constants.viewer_height, self.constants.road_map, self.actors)
        else:
            self.viewer.update(self.actors)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
