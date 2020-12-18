import json
import pathlib
from abc import ABC, abstractmethod
from argparse import ArgumentParser, FileType
from dataclasses import dataclass, asdict
from enum import Enum
from json import JSONEncoder, load
from typing import Optional, Union

import gym
from enforce_typing import enforce_types
from gym.utils import seeding

from examples.agents.dynamic_body import KeyboardAgent
from examples.agents.ego import QLearningEgoAgent
from examples.agents.pedestrian import RandomConstrainedAgent, ProximityAgent, ElectionAgent, QLearningAgent
from examples.agents.template import RandomAgent, NoopAgent
from library.actions import TrafficLightAction
from library.bodies import DynamicBody, Pedestrian, TrafficLight, PelicanCrossing
from reporting import Verbosity, get_console, pretty_str_list


class Scenario(Enum):
    BUS_STOP = "bus-stop"
    CROSSROADS = "crossroads"
    PEDESTRIANS = "pedestrians"
    PELICAN_CROSSING = "pelican-crossing"

    def __str__(self):
        return self.value


class AgentType(Enum):
    NOOP = "noop"
    KEYBOARD = "keyboard"
    RANDOM = "random"
    RANDOM_CONSTRAINED = "random-constrained"
    PROXIMITY = "proximity"
    ELECTION = "election"
    Q_LEARNING = "q-learning"

    def __str__(self):
        return self.value


class Mode(Enum):
    HEADLESS = "headless"
    RENDER = "render"

    def __str__(self):
        return self.value


class ScenarioConfig(ABC):
    @property
    @abstractmethod
    def scenario(self):
        raise NotImplementedError


class AgentConfig(ABC):
    @property
    @abstractmethod
    def agent(self):
        raise NotImplementedError


class ModeConfig(ABC):
    @property
    @abstractmethod
    def mode(self):
        raise NotImplementedError


@enforce_types
@dataclass(frozen=True)
class BusStopConfig(ScenarioConfig):
    scenario = Scenario.BUS_STOP


@enforce_types
@dataclass(frozen=True)
class CrossroadsConfig(ScenarioConfig):
    scenario = Scenario.CROSSROADS


@enforce_types
@dataclass(frozen=True)
class PedestriansConfig(ScenarioConfig):
    num_pedestrians: int
    outbound_pavement: float
    inbound_pavement: float

    scenario = Scenario.PEDESTRIANS

    def __post_init__(self):
        if self.num_pedestrians < 0:
            raise ValueError("num_pedestrians must be >= 0")
        if self.outbound_pavement < 0 or self.outbound_pavement > 1:
            raise ValueError("outbound_pavement must be in [0,1]")
        if self.inbound_pavement < 0 or self.inbound_pavement > 1:
            raise ValueError("inbound_pavement must be in [0,1]")


@enforce_types
@dataclass(frozen=True)
class PelicanCrossingConfig(ScenarioConfig):
    scenario = Scenario.PELICAN_CROSSING


@enforce_types
@dataclass(frozen=True)
class NoopConfig(AgentConfig):
    agent = AgentType.NOOP


@enforce_types
@dataclass(frozen=True)
class KeyboardConfig(AgentConfig):
    agent = AgentType.KEYBOARD


@enforce_types
@dataclass(frozen=True)
class RandomConfig(AgentConfig):
    epsilon: float

    agent = AgentType.RANDOM

    def __post_init__(self):
        if self.epsilon < 0 or self.epsilon > 1:
            raise ValueError("epsilon must be in [0, 1]")


@enforce_types
@dataclass(frozen=True)
class RandomConstrainedConfig(AgentConfig):
    epsilon: float

    agent = AgentType.RANDOM_CONSTRAINED

    def __post_init__(self):
        if self.epsilon < 0 or self.epsilon > 1:
            raise ValueError("epsilon must be in [0, 1]")


@enforce_types
@dataclass(frozen=True)
class ProximityConfig(AgentConfig):
    threshold: float

    agent = AgentType.PROXIMITY

    def __post_init__(self):
        if self.threshold <= 0:
            raise ValueError("threshold must be >= 0")


@enforce_types
@dataclass(frozen=True)
class ElectionConfig(AgentConfig):
    threshold: float

    agent = AgentType.ELECTION

    def __post_init__(self):
        if self.threshold <= 0:
            raise ValueError("threshold must be >= 0")


@enforce_types
@dataclass(frozen=True)
class FeatureConfig:
    distance_x: bool
    distance_y: bool
    distance: bool
    relative_angle: bool
    heading: bool
    on_road: bool
    inverse_distance: bool


@enforce_types
@dataclass(frozen=True)
class QLearningConfig(AgentConfig):
    alpha: float
    gamma: float
    epsilon: float
    features: FeatureConfig
    log: Optional[str]

    agent = AgentType.Q_LEARNING

    def __post_init__(self):
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("alpha must be in [0, 1]")
        if self.epsilon < 0 or self.epsilon > 1:
            raise ValueError("epsilon must be in [0, 1]")
        if self.gamma < 0 or self.gamma > 1:
            raise ValueError("gamma must be in [0, 1]")


@enforce_types
@dataclass(frozen=True)
class HeadlessConfig(ModeConfig):
    mode = Mode.HEADLESS


@enforce_types
@dataclass(frozen=True)
class RenderConfig(ModeConfig):
    episode_condition: int
    video_dir: Optional[str]

    mode = Mode.RENDER

    def __post_init__(self):
        if self.episode_condition < 1:
            raise ValueError("episode_condition must be >= 1")


class CollisionType(Enum):
    NONE = "none"
    EGO = "ego"
    ALL = "all"


@enforce_types
@dataclass(frozen=True)
class Config:
    verbosity: Verbosity
    episode_log: Optional[str]
    run_log: Optional[str]
    seed: Optional[int]
    episodes: int
    max_timesteps: int
    terminate_collisions: CollisionType
    terminate_ego_zones: bool
    terminate_ego_offroad: bool
    reward_win: float
    reward_draw: float
    cost_step: float
    scenario_config: Union[BusStopConfig, CrossroadsConfig, PedestriansConfig, PelicanCrossingConfig]
    ego_config: Union[NoopConfig, KeyboardConfig, RandomConfig, QLearningConfig]
    tester_config: Union[NoopConfig, RandomConfig, RandomConstrainedConfig, ProximityConfig, ElectionConfig, QLearningConfig]
    mode_config: Union[HeadlessConfig, RenderConfig]

    def __post_init__(self):
        if self.episodes <= 0:
            raise ValueError("seed must be > 0")
        if self.seed and self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.max_timesteps <= 0:
            raise ValueError("seed must be > 0")

    def setup(self):
        console = get_console(self.verbosity)

        np_random, np_seed = seeding.np_random(self.seed)
        console.info(f"seed={np_seed}")

        if self.scenario_config.scenario is Scenario.PELICAN_CROSSING:
            env = gym.make('PelicanCrossing-v0', env_config=self, np_random=np_random)
        elif self.scenario_config.scenario is Scenario.BUS_STOP:
            env = gym.make('BusStop-v0', env_config=self, np_random=np_random)
        elif self.scenario_config.scenario is Scenario.CROSSROADS:
            env = gym.make('Crossroads-v0', env_config=self, np_random=np_random)
        elif self.scenario_config.scenario is Scenario.PEDESTRIANS:
            env = gym.make('Pedestrians-v0', env_config=self, num_pedestrians=self.scenario_config.num_pedestrians, outbound_percentage=self.scenario_config.outbound_pavement, inbound_percentage=self.scenario_config.inbound_pavement, np_random=np_random)
        else:
            raise NotImplementedError

        console.info(f"bodies={pretty_str_list(body.__class__.__name__ for body in env.bodies)}")

        keyboard_agent = None
        if self.ego_config.agent is AgentType.NOOP:
            agent = NoopAgent(
                index=0,
                noop_action=env.bodies[0].noop_action
            )
        elif self.ego_config.agent is AgentType.KEYBOARD:
            assert self.mode_config.mode is Mode.RENDER, "keyboard agents only work in render mode"
            keyboard_agent = KeyboardAgent(
                index=0,
                body=env.bodies[0],
                time_resolution=env.time_resolution
            )
            agent = keyboard_agent
        elif self.ego_config.agent is AgentType.RANDOM:
            agent = RandomAgent(
                index=0,
                noop_action=env.bodies[0].noop_action,
                epsilon=self.ego_config.epsilon
            )
        elif self.ego_config.agent is AgentType.Q_LEARNING:
            agent = keyboard_agent if keyboard_agent is not None else QLearningEgoAgent(
                index=0,
                np_random=np_random,
                q_learning_config=self.ego_config,
                body=env.bodies[0],
                time_resolution=env.time_resolution,
                width=env.constants.viewer_width,
                height=env.constants.viewer_height,
                num_velocity_targets=5,
                num_opponents=len(env.bodies)-1
            )
        # elif self.ego_config.agent is AgentType.FRENET:
        #     oubound_lane = env.constants.road_map.major_road.outbound.lanes[0].static_bounding_box
        #     # inbound_lane = env.constants.road_map.major_road.inbound.lanes[0].static_bounding_box
        #     # inbound_lane_end, inbound_lane_start = inbound_lane.split_longitudinally()
        #     agent = FrenetAgent(
        #         index=0,
        #         body=env.bodies[0],
        #         time_resolution=env.time_resolution,
        #         lane_width=env.constants.road_map.major_road.constants.lane_width,
        #         waypoints=[
        #             oubound_lane.rear_centre(),
        #             # inbound_lane_start.centre(),
        #             # oubound_lane.centre(),
        #             # inbound_lane_end.centre(),
        #             oubound_lane.front_centre()
        #         ]
        #     )
        else:
            raise NotImplementedError

        agents = [agent]
        for i, body in enumerate(env.bodies[1:], start=1):
            if isinstance(body, DynamicBody):
                if self.tester_config.agent is AgentType.NOOP:
                    agent = NoopAgent(
                        index=i,
                        noop_action=body.noop_action
                    )
                elif self.tester_config.agent is AgentType.RANDOM:
                    agent = RandomAgent(
                        index=i,
                        noop_action=body.noop_action,
                        epsilon=self.tester_config.epsilon,
                        np_random=np_random
                    )
                elif self.tester_config.agent is AgentType.RANDOM_CONSTRAINED and isinstance(body, Pedestrian):
                    agent = RandomConstrainedAgent(
                        index=i,
                        body=body,
                        time_resolution=env.time_resolution,
                        road_centre=env.constants.road_map.major_road.bounding_box().longitudinal_line(),
                        epsilon=self.tester_config.epsilon,
                        np_random=np_random
                    )
                elif self.tester_config.agent is AgentType.PROXIMITY and isinstance(body, Pedestrian):
                    agent = ProximityAgent(
                        index=i,
                        body=body,
                        time_resolution=env.time_resolution,
                        road_centre=env.constants.road_map.major_road.bounding_box().longitudinal_line(),
                        distance_threshold=self.tester_config.threshold
                    )
                elif self.tester_config.agent is AgentType.ELECTION and isinstance(body, Pedestrian):
                    agent = ElectionAgent(
                        index=i,
                        body=body,
                        time_resolution=env.time_resolution,
                        road_centre=env.constants.road_map.major_road.bounding_box().longitudinal_line(),
                        distance_threshold=self.tester_config.threshold
                    )
                elif self.tester_config.agent is AgentType.Q_LEARNING and isinstance(body, Pedestrian):
                    agent = QLearningAgent(
                        index=i,
                        body=body,
                        ego_constants=env.bodies[0].constants,
                        road_polgon=env.constants.road_map.major_road.static_bounding_box,
                        time_resolution=env.time_resolution,
                        width=env.constants.viewer_width,
                        height=env.constants.viewer_height,
                        np_random=np_random,
                        q_learning_config=self.tester_config
                    )
                else:
                    raise NotImplementedError
            elif isinstance(body, TrafficLight) or isinstance(body, PelicanCrossing):
                if self.tester_config.agent is AgentType.NOOP:
                    agent = NoopAgent(
                        index=i,
                        noop_action=TrafficLightAction.NOOP.value
                    )
                elif self.tester_config.agent is AgentType.RANDOM:
                    agent = RandomAgent(
                        index=i,
                        noop_action=TrafficLightAction.NOOP.value,
                        epsilon=self.tester_config.epsilon,
                        np_random=np_random
                    )
            agents.append(agent)

        console.info(f"agents={pretty_str_list(agent.__class__.__name__ for agent in agents)}")
        console.info(f"ego=({env.bodies[0].__class__.__name__}, {agents[0].__class__.__name__})")

        return np_seed, env, agents, keyboard_agent

    def write_json(self, path):
        path_obj = pathlib.Path(path)
        directory_obj = path_obj.parent
        directory_obj.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(self, json_file, ensure_ascii=False, indent=2, cls=ConfigJSONEncoder)


class ConfigJSONEncoder(JSONEncoder):  # serialise Config to data
    def default(self, obj):
        if isinstance(obj, Config):
            obj_dict = asdict(obj)
            obj_dict["verbosity"] = obj.verbosity.value
            obj_dict["terminate_collisions"] = obj.terminate_collisions.value
            obj_dict["scenario_config"] = {"option": str(obj.scenario_config.scenario), **asdict(obj.scenario_config)}
            obj_dict["ego_config"] = {"option": str(obj.ego_config.agent), **asdict(obj.ego_config)}
            obj_dict["tester_config"] = {"option": str(obj.tester_config.agent), **asdict(obj.tester_config)}
            obj_dict["mode_config"] = {"option": str(obj.mode_config.mode), **asdict(obj.mode_config)}
            return obj_dict
        else:
            return super().default(obj)


def make_scenario_config(data):  # deserialise data to ScenarioConfig
    option = data.pop("option")
    if option == Scenario.BUS_STOP.value:
        if data:
            raise ValueError(f"unexpected parameters {data}")
        return BusStopConfig()
    elif option == Scenario.CROSSROADS.value:
        if data:
            raise ValueError(f"unexpected parameters {data}")
        return CrossroadsConfig()
    elif option == Scenario.PEDESTRIANS.value:
        return PedestriansConfig(**data)
    elif option == Scenario.PELICAN_CROSSING.value:
        if data:
            raise ValueError(f"unexpected parameters {data}")
        return PelicanCrossingConfig()
    else:
        raise NotImplementedError


def make_q_learning_config(data):
    feature_config_data = data.pop("feature_config")
    feature_config = FeatureConfig(**feature_config_data)
    return QLearningConfig(**data, features=feature_config)


def make_ego_config(data):  # deserialise data to AgentConfig
    option = data.pop("option")
    if option == AgentType.NOOP.value:
        return NoopConfig()
    elif option == AgentType.KEYBOARD.value:
        return KeyboardConfig()
    elif option == AgentType.RANDOM.value:
        return RandomConfig(**data)
    elif option == AgentType.Q_LEARNING.value:
        return make_q_learning_config(data)
    else:
        raise NotImplementedError


def make_tester_config(data):  # deserialise data to AgentConfig
    option = data.pop("option")
    if option == AgentType.NOOP.value:
        return NoopConfig()
    elif option == AgentType.RANDOM.value:
        return RandomConfig(**data)
    elif option == AgentType.RANDOM_CONSTRAINED.value:
        return RandomConstrainedConfig(**data)
    elif option == AgentType.PROXIMITY.value:
        return ProximityConfig(**data)
    elif option == AgentType.ELECTION.value:
        return ElectionConfig(**data)
    elif option == AgentType.Q_LEARNING.value:
        return make_q_learning_config(data)
    else:
        raise NotImplementedError


def make_mode_config(data):  # deserialise data to ModeConfig
    option = data.pop("option")
    if option == Mode.HEADLESS.value:
        if data:
            raise ValueError(f"unexpected parameters {data}")
        return HeadlessConfig()
    elif option == Mode.RENDER.value:
        return RenderConfig(**data)
    else:
        raise NotImplementedError


def make_config(data):
    data["verbosity"] = Verbosity(str(data["verbosity"]))
    data["terminate_collisions"] = CollisionType(str(data["terminate_collisions"]))
    data["scenario_config"] = make_scenario_config(data["scenario_config"])
    data["ego_config"] = make_ego_config(data["ego_config"])
    data["tester_config"] = make_tester_config(data["tester_config"])
    data["mode_config"] = make_mode_config(data["mode_config"])
    return Config(**data)


class ConfigParser(ArgumentParser):
    def __init__(self):
        super().__init__()

        self.add_argument("input", metavar="INPUT", nargs="?", type=FileType("r"), default="-", help="read config from %(metavar)s file, or from stdin if no file is provided")

    def parse_config(self):
        args = self.parse_args()
        data = load(args.input)
        return make_config(data)
