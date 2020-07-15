import logging
from dataclasses import dataclass

import gym
from gym import spaces
from gym.utils import seeding

from library import geometry
from library.actions import TrafficLightAction, OrientationAction, VelocityAction
from library.actors import PelicanCrossing, DynamicActor, TrafficLight, Pedestrian
from library.assets import RoadMap, Occlusion
from library.observations import OrientationObservation, EmptyObservation, VelocityObservation, RoadObservation, \
    DistanceObservation
from scenarios.constants import M2PX

console_logger = logging.getLogger("library.console.environment")


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
    road_map: RoadMap


@dataclass(frozen=True)
class EnvConfig:
    frequency: int = 60
    terminate_collision: bool = False
    terminate_offroad: bool = False
    terminate_zone: bool = False
    distance_threshold: float = M2PX * 22.5


class CAVEnv(MarkovGameEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, actors, constants, env_config=EnvConfig(), np_random=seeding.np_random(None)[0]):
        self.actors = actors
        self.constants = constants
        self.env_config = env_config
        self.np_random = np_random

        self.time_resolution = 1.0 / self.env_config.frequency

        console_logger.info(f"frequency={self.env_config.frequency}")

        def set_np_random(space):
            space.np_random = self.np_random
            if isinstance(space, spaces.Tuple):
                for subspace in space:
                    set_np_random(subspace)

        actor_spaces = list()
        for actor in self.actors:
            actor_space = None
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
                road_observation_space = spaces.Discrete(RoadObservation.__len__())
                distance_observation_space = spaces.Discrete(DistanceObservation.__len__())
                observation_space = spaces.Tuple([velocity_observation_space, orientation_observation_space, road_observation_space, distance_observation_space])
            else:
                observation_space = spaces.Discrete(EmptyObservation.__len__())
            observation_spaces.append(observation_space)
        self.observation_space = spaces.Tuple(observation_spaces)
        set_np_random(self.observation_space)

        self.episode_liveness = [0 for _ in self.actors]
        self.run_liveness = [0 for _ in self.actors]

        self.ego = self.actors[0]

        self.viewer = None

    def collidable_entities(self):
        entities = [actor for actor in self.actors if isinstance(actor, Occlusion)]
        for actor in self.actors:
            if isinstance(actor, PelicanCrossing):
                entities += [actor.outbound_traffic_light, actor.inbound_traffic_light]
        if self.constants.road_map.obstacle is not None:
            entities.append(self.constants.road_map.obstacle)
        return entities

    def observe(self):
        joint_observation = list()
        for actor in self.actors:
            if isinstance(actor, DynamicActor):
                velocity_observation = VelocityObservation.ACTIVE if actor.target_velocity is not None else VelocityObservation.INACTIVE
                orientation_observation = OrientationObservation.ACTIVE if actor.target_orientation is not None else OrientationObservation.INACTIVE

                def find_road_observation(road):
                    if actor.bounding_box().intersects(road.bounding_box()):
                        return RoadObservation.ON_ROAD
                    else:
                        relative_angle = actor.line_anchor_relative_angle(road)

                        if geometry.DEG2RAD * -157.5 <= relative_angle < geometry.DEG2RAD * -112.5:
                            return RoadObservation.ROAD_REAR_RIGHT
                        elif geometry.DEG2RAD * -112.5 <= relative_angle < geometry.DEG2RAD * -67.5:
                            return RoadObservation.ROAD_RIGHT
                        elif geometry.DEG2RAD * -67.5 <= relative_angle < geometry.DEG2RAD * -22.5:
                            return RoadObservation.ROAD_FRONT_RIGHT
                        elif geometry.DEG2RAD * -22.5 <= relative_angle < geometry.DEG2RAD * 22.5:
                            return RoadObservation.ROAD_FRONT
                        elif geometry.DEG2RAD * 22.5 <= relative_angle < geometry.DEG2RAD * 67.5:
                            return RoadObservation.ROAD_FRONT_LEFT
                        elif geometry.DEG2RAD * 67.5 <= relative_angle < geometry.DEG2RAD * 112.5:
                            return RoadObservation.ROAD_LEFT
                        elif geometry.DEG2RAD * 112.5 <= relative_angle < geometry.DEG2RAD * 157.5:
                            return RoadObservation.ROAD_REAR_LEFT
                        elif geometry.DEG2RAD * 157.5 <= relative_angle <= geometry.DEG2RAD * 180 or geometry.DEG2RAD * -180 < relative_angle < geometry.DEG2RAD * -157.5:
                            return RoadObservation.ROAD_REAR
                        else:
                            raise Exception("relative angle is not in the interval (-math.pi, math.pi]")

                road_observation = find_road_observation(self.constants.road_map.major_road)
                _, assertion_zone = self.ego.stopping_zones()
                distance_observation = DistanceObservation.SATISFIED if actor is not self.ego and assertion_zone is not None and assertion_zone.distance(actor.state.position) < self.env_config.distance_threshold else DistanceObservation.UNSATISFIED
                joint_observation.append(tuple([velocity_observation.value, orientation_observation.value, road_observation.value, distance_observation.value]))
            else:
                joint_observation.append(EmptyObservation.NONE.value)

        return joint_observation

    def step(self, joint_action):
        assert self.action_space.contains(joint_action), f"{joint_action} ({type(joint_action)}) invalid"

        for index, actor in enumerate(self.actors):
            actor.step_action(joint_action, index)

        for actor in self.actors:
            actor.step_dynamics(self.time_resolution)

        joint_observation = self.observe()

        for i, actor in enumerate(self.actors):
            if any(actor.bounding_box().mostly_intersects(road.bounding_box()) for road in self.constants.road_map.roads):
                self.episode_liveness[i] += 1
                self.run_liveness[i] += 1

        joint_reward = [0 for _ in self.actors]

        terminate = False

        if self.env_config.terminate_collision:
            collidable_entities = self.collidable_entities()
            collidable_bounding_boxes = [entity.bounding_box() for entity in collidable_entities]  # no need to recompute bounding boxes
            collided_entities = list()  # record collided entities so that they can be skipped in subsequent iterations

            def collision_detected(entity):
                if entity in collided_entities:
                    return True
                entity_bounding_box = collidable_bounding_boxes[collidable_entities.index(entity)]
                for other, other_bounding_box in zip(collidable_entities, collidable_bounding_boxes):
                    if entity is not other and other not in collided_entities and entity_bounding_box.intersects(other_bounding_box):
                        collided_entities.append(entity)
                        collided_entities.append(other)  # other can be skipped in future because we know it has collided with entity
                        return True
                else:
                    collidable_entities.remove(entity)
                    collidable_bounding_boxes.remove(entity_bounding_box)  # ensure that this list matches collidable_entities
                    return False

            collision_detections = [not isinstance(actor, PelicanCrossing) and collision_detected(actor) for actor in self.actors]
            collision_occurred = any(collision_detections)
            terminate = collision_occurred

        if not terminate and self.env_config.terminate_offroad:
            ego_on_road = any(self.ego.bounding_box().intersects(road.bounding_box()) for road in self.constants.road_map.roads)
            terminate = not ego_on_road

        def successful_test_pedestrian():
            for other_actor_index, other_actor in enumerate(self.actors):
                if other_actor is not self.ego and isinstance(other_actor, Pedestrian) and other_actor.bounding_box().intersects(self.ego.stopping_zones()[1]):
                    return other_actor_index
            return None

        pedestrian_in_reaction_zone = None
        if not terminate and self.env_config.terminate_zone:
            pedestrian_in_reaction_zone = successful_test_pedestrian()
            terminate = pedestrian_in_reaction_zone is not None

        info = {'pedestrian': pedestrian_in_reaction_zone}

        return joint_observation, joint_reward, terminate, info

    def reset(self):
        for actor in self.actors:
            actor.reset()
        self.episode_liveness = [0 for _ in self.actors]
        return self.observe()

    def render(self, mode='human'):
        from library.rendering import RoadEnvViewer  # lazy import of pyglet to allow headless mode on headless machines
        if not self.viewer:
            self.viewer = RoadEnvViewer(self.constants.viewer_width, self.constants.viewer_height, self.constants.road_map, self.actors, self.ego)
        else:
            self.viewer.update(self.actors, self.ego)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
