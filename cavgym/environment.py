from dataclasses import dataclass

import gym
from gym import spaces
from gym.utils import seeding

from cavgym.actions import TrafficLightAction, OrientationAction, VelocityAction
from cavgym.actors import PelicanCrossing, DynamicActor, TrafficLight
from cavgym.observations import OrientationObservation, EmptyObservation, VelocityObservation
from cavgym.rendering import RoadEnvViewer
from cavgym.assets import RoadMap, Occlusion


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

    def __init__(self, actors, constants, np_random=seeding.np_random(None)[0]):
        self.actors = actors
        self.constants = constants

        self.np_random = np_random

        def set_np_random(space):
            space.np_random = self.np_random
            if isinstance(space, spaces.Tuple):
                for subspace in space:
                    set_np_random(subspace)

        actor_spaces = list()
        for actor in self.actors:
            if isinstance(actor, DynamicActor):
                velocity_action_space = spaces.Discrete(VelocityAction.__len__())
                orientation_action_space = spaces.Discrete(OrientationAction.__len__())
                actor_space = spaces.Tuple([velocity_action_space, orientation_action_space])
            elif isinstance(actor, PelicanCrossing) or isinstance(actor, TrafficLight):
                actor_space = spaces.Discrete(TrafficLightAction.__len__())
            actor_spaces.append(actor_space)
        self.action_space = spaces.Tuple(actor_spaces)
        set_np_random(self.action_space)

        observation_spaces = list()
        for actor in self.actors:
            if isinstance(actor, DynamicActor):
                velocity_observation_space = spaces.Discrete(VelocityObservation.__len__())
                orientation_observation_space = spaces.Discrete(OrientationObservation.__len__())
                observation_space = spaces.Tuple([velocity_observation_space, orientation_observation_space])
            else:
                observation_space = spaces.Discrete(EmptyObservation.__len__())
            observation_spaces.append(observation_space)
        self.observation_space = spaces.Tuple(observation_spaces)
        set_np_random(self.observation_space)

        self.ego = self.actors[0]

        self.viewer = None

    def determine_collidable(self):
        collidable = [actor for actor in self.actors if isinstance(actor, Occlusion)]
        for actor in self.actors:
            if isinstance(actor, PelicanCrossing):
                collidable += [actor.outbound_traffic_light, actor.inbound_traffic_light]
        if self.constants.road_map.obstacle is not None:
            collidable.append(self.constants.road_map.obstacle)
        return collidable

    def step(self, joint_action):
        assert self.action_space.contains(joint_action), "%r (%s) invalid" % (joint_action, type(joint_action))

        for index, actor in enumerate(self.actors):
            actor.step_action(joint_action, index)

        for actor in self.actors:
            actor.step_dynamics(self.constants.time_resolution)

        joint_observation = list()
        for actor in self.actors:
            if isinstance(actor, DynamicActor):
                velocity_observation = VelocityObservation.ACTIVE if actor.target_velocity is not None else VelocityObservation.INACTIVE
                orientation_observation = OrientationObservation.ACTIVE if actor.target_orientation is not None else OrientationObservation.INACTIVE
                joint_observation.append(tuple([velocity_observation.value, orientation_observation.value]))
            else:
                joint_observation.append(EmptyObservation.NONE.value)

        joint_reward = [-1 if not isinstance(actor, PelicanCrossing) and any(actor.bounding_box().intersects(other.bounding_box()) for other in self.determine_collidable() if actor is not other) else 0 for actor in self.actors]

        return joint_observation, joint_reward, any(reward < 0 for reward in joint_reward), None

    def reset(self):
        for actor in self.actors:
            actor.reset()
        return list(self.observation_space.sample())

    def render(self, mode='human'):
        if not self.viewer:
            self.viewer = RoadEnvViewer(self.constants.viewer_width, self.constants.viewer_height, self.constants.road_map, self.actors, self.ego)
        else:
            self.viewer.update(self.actors, self.ego)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
