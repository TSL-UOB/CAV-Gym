import logging
from dataclasses import dataclass

import gym
from gym import spaces
from gym.utils import seeding

from config import Mode, CollisionType
from library.bodies import PelicanCrossing, Pedestrian
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

    def __init__(self, bodies, constants, env_config, np_random=seeding.np_random(None)[0]):
        self.bodies = bodies
        self.constants = constants
        self.env_config = env_config
        self.np_random = np_random

        self.frequency = 60
        self.time_resolution = 1.0 / self.frequency

        console_logger.info(f"frequency={self.frequency}")

        def set_np_random(space):
            space.np_random = self.np_random
            if isinstance(space, spaces.Tuple):
                for subspace in space:
                    set_np_random(subspace)

        self.action_space = spaces.Tuple([body.action_space() for body in self.bodies])
        set_np_random(self.action_space)

        self.observation_space = spaces.Tuple([body.observation_space() for body in self.bodies])
        set_np_random(self.observation_space)

        self.episode_liveness = [0 for _ in self.bodies]
        self.run_liveness = [0 for _ in self.bodies]

        self.ego = self.bodies[0]

        self.ego_maintenance_velocity = self.ego.init_state.velocity
        self.ego_max_velocity_offset = max(abs(self.ego.constants.max_velocity - self.ego_maintenance_velocity), abs(self.ego.constants.min_velocity - self.ego_maintenance_velocity))

        self.current_timestep = 0

        self.viewer = None

    def collidable_entities(self):
        entities = [body for body in self.bodies if isinstance(body, Occlusion)]
        for body in self.bodies:
            if isinstance(body, PelicanCrossing):
                entities += [body.outbound_traffic_light, body.inbound_traffic_light]
        if self.constants.road_map.obstacle is not None:
            entities.append(self.constants.road_map.obstacle)
        return entities

    def state(self):
        return [list(body.state) for body in self.bodies]

    def info(self):
        body_polygons = [body.bounding_box() for body in self.bodies]

        road = self.constants.road_map.major_road
        road_polygon = road.bounding_box()

        def angle(entity, entity_polygon):
            return entity.line_anchor_relative_angle(road) if not entity_polygon.intersects(road_polygon) else None

        road_angles = [angle(body, body_polygons[i]) for i, body in enumerate(self.bodies)]

        return {'body_polygons': body_polygons, 'road_angles': road_angles}

    def step(self, joint_action):
        assert self.action_space.contains(joint_action), f"{joint_action} ({type(joint_action)}) invalid"

        for index, body in enumerate(self.bodies):
            body.step(joint_action[index], self.time_resolution)

        info = self.info()

        body_polygons = info['body_polygons']

        state = self.state()

        joint_reward = [0.0 for _ in self.bodies]

        ego_velocity_relative_offset = abs(self.ego.state.velocity - self.ego_maintenance_velocity) / self.ego_max_velocity_offset
        joint_reward[0] -= ego_velocity_relative_offset * self.env_config.cost_step

        for i, polygon in enumerate(body_polygons):
            if i > 0:
                percentage_intersects = max(polygon.percentage_intersects(road.bounding_box()) for road in self.constants.road_map.roads)
                joint_reward[i] -= percentage_intersects * self.env_config.cost_step
                if percentage_intersects > 0.5:  # pedestrian mostly on road
                    self.episode_liveness[i] += 1
                    self.run_liveness[i] += 1

        terminate = False
        win_ego = False
        win_tester = None

        if not terminate and all(x > self.constants.viewer_width for x, y in self.ego.bounding_box()):
            terminate = True
            win_ego = True

        if not terminate and self.env_config.terminate_collisions is CollisionType.ALL:
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

            collision_detections = [not isinstance(body, PelicanCrossing) and collision_detected(body) for body in self.bodies]
            collision_occurred = any(collision_detections)
            terminate = collision_occurred

        if not terminate and self.env_config.terminate_ego_offroad:
            ego_on_road = any(body_polygons[0].intersects(road.bounding_box()) for road in self.constants.road_map.roads)
            terminate = not ego_on_road

        def ego_collision(entity_bounding_box):
            if entity_bounding_box.intersects(body_polygons[0]):
                return True
            braking_zone = self.ego.stopping_zones()[0]
            if braking_zone is not None and entity_bounding_box.intersects(braking_zone):
                return True
            return False

        if not terminate and self.env_config.terminate_collisions is CollisionType.EGO:  # terminate early if pedestrian collides with ego or braking zone of ego (uninteresting tests)
            ego_collision = any(ego_collision(body_polygons[i]) for i, body in enumerate(self.bodies) if body is not self.ego and isinstance(body, Pedestrian))
            terminate = ego_collision

        def check_pedestrian_in_reaction_zone():
            reaction_zone = self.ego.stopping_zones()[1]
            if reaction_zone is not None:
                for other_body_index, other_body in enumerate(self.bodies):
                    if other_body is not self.ego and isinstance(other_body, Pedestrian) and body_polygons[other_body_index].intersects(reaction_zone):
                        return other_body_index
            return None

        if not terminate and self.env_config.terminate_ego_zones:
            pedestrian_in_reaction_zone = check_pedestrian_in_reaction_zone()
            terminate = pedestrian_in_reaction_zone is not None
            win_tester = pedestrian_in_reaction_zone

        if terminate or self.current_timestep == self.env_config.max_timesteps - 1:
            assert not win_ego or win_tester is None
            joint_reward[0] += self.env_config.reward_win if win_ego else -self.env_config.reward_win if win_tester is not None else self.env_config.reward_draw
            for i, _ in enumerate(joint_reward):
                if i > 0:
                    joint_reward[i] += -self.env_config.reward_win if win_ego else self.env_config.reward_draw if win_tester is None else self.env_config.reward_win if win_tester == i else self.env_config.reward_draw
            # noinspection PyTypeChecker
            if win_ego:
                info['winner'] = 0
            elif win_tester is not None:
                info['winner'] = win_tester
        else:
            assert not win_ego and win_tester is None

        self.current_timestep += 1
        return state, joint_reward, terminate, info

    def reset(self):
        for body in self.bodies:
            body.reset()
        self.episode_liveness = [0 for _ in self.bodies]
        return self.state()

    def render(self, mode='human'):
        from library.rendering import RoadEnvViewer  # lazy import of pyglet to allow headless mode on headless machines
        if not self.viewer:
            self.viewer = RoadEnvViewer(self.constants.viewer_width, self.constants.viewer_height, self.constants.road_map, self.bodies, self.ego)
        else:
            self.viewer.update(self.bodies, self.ego)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
