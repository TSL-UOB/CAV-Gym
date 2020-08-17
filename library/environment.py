import logging
from dataclasses import dataclass

import gym
from gym import spaces
from gym.utils import seeding

from config import Mode, Scenario
from library.actors import PelicanCrossing, Pedestrian
from library.assets import RoadMap, Occlusion

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


class CAVEnv(MarkovGameEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, actors, constants, env_config, np_random=seeding.np_random(None)[0]):
        self.actors = actors
        self.constants = constants
        self.env_config = env_config
        self.np_random = np_random

        self.frequency = 30 if self.env_config.mode_config.mode is Mode.RENDER and self.env_config.mode_config.record else 60
        self.time_resolution = 1.0 / self.frequency

        console_logger.info(f"frequency={self.frequency}")

        def set_np_random(space):
            space.np_random = self.np_random
            if isinstance(space, spaces.Tuple):
                for subspace in space:
                    set_np_random(subspace)

        self.action_space = spaces.Tuple([actor.action_space() for actor in self.actors])
        set_np_random(self.action_space)

        self.observation_space = spaces.Tuple([actor.observation_space() for actor in self.actors])
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

    def state(self):
        return [tuple(actor.state) for actor in self.actors]

    def info(self):
        actor_polygons = [actor.bounding_box() for actor in self.actors]

        road = self.constants.road_map.major_road
        road_polygon = road.bounding_box()

        def angle(entity, entity_polygon):
            return entity.line_anchor_relative_angle(road) if not entity_polygon.intersects(road_polygon) else None

        road_angles = [angle(actor, actor_polygons[i]) for i, actor in enumerate(self.actors)]

        return {'actor_polygons': actor_polygons, 'road_angles': road_angles}

    def step(self, joint_action):
        assert self.action_space.contains(joint_action), f"{joint_action} ({type(joint_action)}) invalid"

        for index, actor in enumerate(self.actors):
            actor.step_action(joint_action, index)

        for actor in self.actors:
            actor.step_dynamics(self.time_resolution)

        info = self.info()

        actor_polygons = info['actor_polygons']

        state = self.state()

        joint_reward = [-1 for _ in self.actors]

        for i, polygon in enumerate(actor_polygons):
            if any(polygon.mostly_intersects(road.bounding_box()) for road in self.constants.road_map.roads):  # on road
                self.episode_liveness[i] += 1
                self.run_liveness[i] += 1
                if i > 0:  # not ego
                    joint_reward[i] -= 5
            else:  # off road
                if i == 0:  # ego
                    joint_reward[i] -= 5

        terminate = False

        if self.env_config.collisions:
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

        if not terminate and self.env_config.offroad:
            ego_on_road = any(actor_polygons[0].intersects(road.bounding_box()) for road in self.constants.road_map.roads)
            terminate = not ego_on_road

        def ego_collision(entity_bounding_box):
            if entity_bounding_box.intersects(actor_polygons[0]):
                return True
            braking_zone = self.ego.stopping_zones()[0]
            if braking_zone is not None and entity_bounding_box.intersects(braking_zone):
                return True
            return False

        if not terminate:  # terminate early if pedestrian collides with ego or braking zone of ego (uninteresting tests)
            ego_collision = any(ego_collision(actor_polygons[i]) for i, actor in enumerate(self.actors) if actor is not self.ego and isinstance(actor, Pedestrian))
            terminate = ego_collision

        def check_pedestrian_in_reaction_zone():
            braking_zone = self.ego.stopping_zones()[1]
            if braking_zone is not None:
                for other_actor_index, other_actor in enumerate(self.actors):
                    if other_actor is not self.ego and isinstance(other_actor, Pedestrian) and actor_polygons[other_actor_index].intersects(braking_zone):
                        return other_actor_index
            return None

        pedestrian_in_reaction_zone = None
        if not terminate and self.env_config.scenario_config.scenario is Scenario.PEDESTRIANS and self.env_config.scenario_config.zone:
            pedestrian_in_reaction_zone = check_pedestrian_in_reaction_zone()
            terminate = pedestrian_in_reaction_zone is not None

        if terminate:
            winner = pedestrian_in_reaction_zone if pedestrian_in_reaction_zone else 0  # index of winner
            joint_reward[winner] += 100
            # noinspection PyTypeChecker
            info['winner'] = winner

        return state, joint_reward, terminate, info

    def reset(self):
        for actor in self.actors:
            actor.reset()
        self.episode_liveness = [0 for _ in self.actors]
        return self.state()

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
