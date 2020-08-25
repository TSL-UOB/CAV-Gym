from abc import ABC, abstractmethod
from argparse import ArgumentParser, FileType
from dataclasses import dataclass, asdict
from enum import Enum
from json import JSONEncoder, load
from typing import Optional, Union

import gym
from enforce_typing import enforce_types
from gym.utils import seeding

from library.actors import DynamicActor, Pedestrian, TrafficLight, PelicanCrossing
from examples.agents import RandomPedestrianAgent, RandomConstrainedPedestrianAgent, ElectionPedestrianAgent, \
    QLearningAgent, RandomVehicleAgent, RandomTrafficLightAgent, NoopAgent, ProximityPedestrianAgent, KeyboardAgent
from reporting import Verbosity


class Scenario(Enum):
    BUS_STOP = "bus-stop"
    CROSSROADS = "crossroads"
    PEDESTRIANS = "pedestrians"
    PELICAN_CROSSING = "pelican-crossing"

    def __str__(self):
        return self.value


class AgentType(Enum):
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
    actors: int

    scenario = Scenario.PEDESTRIANS

    def __post_init__(self):
        if self.actors < 0:
            raise ValueError("actors must be >= 0")


@enforce_types
@dataclass(frozen=True)
class PelicanCrossingConfig(ScenarioConfig):
    scenario = Scenario.PELICAN_CROSSING


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
    on_road: bool
    facing: bool
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
    keyboard: bool
    record: Optional[str]

    mode = Mode.RENDER


@enforce_types
@dataclass(frozen=True)
class Config:
    verbosity: Verbosity
    episode_log: Optional[str]
    run_log: Optional[str]
    seed: Optional[int]
    episodes: int
    max_timesteps: int
    collisions: bool
    offroad: bool
    zone: bool
    scenario_config: Union[BusStopConfig, CrossroadsConfig, PedestriansConfig, PelicanCrossingConfig]
    agent_config: Union[RandomConfig, RandomConstrainedConfig, ProximityConfig, ElectionConfig, QLearningConfig]
    mode_config: Union[HeadlessConfig, RenderConfig]

    def __post_init__(self):
        if self.episodes <= 0:
            raise ValueError("seed must be > 0")
        if self.seed and self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.max_timesteps <= 0:
            raise ValueError("seed must be > 0")

    def setup(self):
        np_random, np_seed = seeding.np_random(self.seed)

        if self.scenario_config.scenario is Scenario.PELICAN_CROSSING:
            env = gym.make('PelicanCrossing-v0', env_config=self, np_random=np_random)
        elif self.scenario_config.scenario is Scenario.BUS_STOP:
            env = gym.make('BusStop-v0', env_config=self, np_random=np_random)
        elif self.scenario_config.scenario is Scenario.CROSSROADS:
            env = gym.make('Crossroads-v0', env_config=self, np_random=np_random)
        elif self.scenario_config.scenario is Scenario.PEDESTRIANS:
            env = gym.make('Pedestrians-v0', env_config=self, num_pedestrians=self.scenario_config.actors, np_random=np_random)
        else:
            raise NotImplementedError

        keyboard_agent = KeyboardAgent(index=0) if self.mode_config.mode is Mode.RENDER and self.mode_config.keyboard else None
        agent = keyboard_agent if keyboard_agent is not None else NoopAgent(index=0)
        agents = [agent]
        for i, actor in enumerate(env.actors[1:], start=1):
            if isinstance(actor, DynamicActor):
                if isinstance(actor, Pedestrian):
                    if self.agent_config.agent is AgentType.RANDOM:
                        agent = RandomPedestrianAgent(
                            index=i,
                            epsilon=self.agent_config.epsilon,
                            np_random=np_random
                        )
                    elif self.agent_config.agent is AgentType.RANDOM_CONSTRAINED:
                        agent = RandomConstrainedPedestrianAgent(
                            index=i,
                            epsilon=self.agent_config.epsilon,
                            road=env.constants.road_map.major_road,
                            np_random=np_random
                        )
                    elif self.agent_config.agent is AgentType.PROXIMITY:
                        agent = ProximityPedestrianAgent(
                            index=i,
                            road=env.constants.road_map.major_road,
                            distance_threshold=self.agent_config.threshold
                        )
                    elif self.agent_config.agent is AgentType.ELECTION:
                        agent = ElectionPedestrianAgent(
                            index=i,
                            road=env.constants.road_map.major_road,
                            distance_threshold=self.agent_config.threshold
                        )
                    elif self.agent_config.agent is AgentType.Q_LEARNING:
                        agent = QLearningAgent(
                            index=i,
                            ego_constants=env.actors[0].constants,
                            self_constants=env.actors[1].constants,
                            road_polgon=env.constants.road_map.major_road.static_bounding_box,
                            time_resolution=env.time_resolution,
                            width=env.constants.viewer_width,
                            height=env.constants.viewer_height,
                            np_random=np_random,
                            q_learning_config=self.agent_config
                        )
                    else:
                        raise NotImplementedError
                else:
                    agent = RandomVehicleAgent(index=i, np_random=np_random)
            elif isinstance(actor, TrafficLight) or isinstance(actor, PelicanCrossing):
                agent = RandomTrafficLightAgent(np_random=np_random)
            agents.append(agent)

        return np_seed, env, agents, keyboard_agent


class ConfigJSONEncoder(JSONEncoder):  # serialise Config to data
    def default(self, obj):
        if isinstance(obj, Config):
            obj_dict = asdict(obj)
            obj_dict["verbosity"] = obj.verbosity.value
            obj_dict["scenario_config"] = {"option": str(obj.scenario_config.scenario), **asdict(obj.scenario_config)}
            obj_dict["agent_config"] = {"option": str(obj.agent_config.agent), **asdict(obj.agent_config)}
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


def make_agent_config(data):  # deserialise data to AgentConfig
    option = data.pop("option")
    if option == AgentType.RANDOM.value:
        return RandomConfig(**data)
    elif option == AgentType.RANDOM_CONSTRAINED.value:
        return RandomConstrainedConfig(**data)
    elif option == AgentType.PROXIMITY.value:
        return ProximityConfig(**data)
    elif option == AgentType.ELECTION.value:
        return ElectionConfig(**data)
    elif option == AgentType.Q_LEARNING.value:
        feature_config_data = data.pop("feature_config")
        feature_config = FeatureConfig(**feature_config_data)
        return QLearningConfig(**data, features=feature_config)
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
    data["scenario_config"] = make_scenario_config(data["scenario_config"])
    data["agent_config"] = make_agent_config(data["agent_config"])
    data["mode_config"] = make_mode_config(data["mode_config"])
    return Config(**data)


class ConfigParser(ArgumentParser):
    def __init__(self):
        super().__init__()

        self.add_argument("input", metavar="INPUT", nargs="?", type=FileType("r"), default="-", help="read config from %(metavar)s file, or standard input if no %(metavar)s")

    def parse_config(self):
        args = self.parse_args()
        data = load(args.input)
        return make_config(data)
