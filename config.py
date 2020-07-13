from argparse import ArgumentParser, ArgumentTypeError
from dataclasses import dataclass
from enum import Enum

import gym
from gym.utils import seeding

import reporting
from library.actors import DynamicActor, Pedestrian, TrafficLight, PelicanCrossing
from library.environment import EnvConfig
from reporting import Verbosity, pretty_str_set
from scenarios.agents import KeyboardAgent, RandomPedestrianAgent, RandomConstrainedPedestrianAgent, \
    ProximityPedestrianAgent, ElectionPedestrianAgent, RandomVehicleAgent, RandomTrafficLightAgent, NoopVehicleAgent


class RenderMode(Enum):
    HEADLESS = 0
    SCREEN = 1
    VIDEO = 2


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

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class RunConfig:
    episodes: int = 1
    max_timesteps: int = 1000
    render_mode: RenderMode = RenderMode.SCREEN
    keyboard_agent: KeyboardAgent = None
    logging_dir: str = None
    record_dir: str = None
    verbosity: Verbosity = Verbosity.INFO
    election: bool = False


@dataclass(frozen=True)
class Config:
    run: RunConfig = RunConfig()
    env: EnvConfig = EnvConfig()
    scenario: Scenario = Scenario.PEDESTRIANS
    actors: int = 3
    agent_type: AgentType = AgentType.RANDOM
    seed: int = None

    def __post_init__(self):
        pass


class ConfigParser(ArgumentParser):
    def __init__(self):
        super().__init__()

        def non_negative_int(value):
            ivalue = int(value)
            if ivalue < 0:
                raise ArgumentTypeError(f"invalid non-negative int value: {value}")
            return ivalue

        def positive_int(value):
            ivalue = int(value)
            if ivalue < 1:
                raise ArgumentTypeError(f"invalid positive int value: {value}")
            return ivalue

        self.add_argument("-c", "--collisions", help="terminate when any collision occurs", action="store_true")
        self.add_argument("-e", "--episodes", type=positive_int, default=RunConfig.episodes, metavar="N", help="set number of episodes as %(metavar)s (default: %(default)s)")
        self.add_argument("-l", "--log", metavar="DIR", help="enable logging of run to %(metavar)s")
        self.add_argument("-o", "--offroad", help="terminate when ego does not intersect road", action="store_true")
        self.add_argument("-s", "--seed", type=non_negative_int, metavar="N", help="set random seed as %(metavar)s")
        self.add_argument("-t", "--timesteps", type=positive_int, default=RunConfig.max_timesteps, metavar="N", help="set max timesteps per episode as %(metavar)s (default: %(default)s)")
        self.add_argument("-v", "--verbosity", type=Verbosity, choices=[choice for choice in Verbosity], default=RunConfig.verbosity, metavar="LEVEL", help=f"set verbosity as %(metavar)s (choices: {pretty_str_set([choice for choice in Verbosity])}, default: {RunConfig.verbosity})")

        self.set_defaults(keyboard_agent=False, record=None, number=None, agent=Config.agent_type, zone=False)

        scenario_subparsers = self.add_subparsers(dest="scenario", required=True, metavar="SCENARIO", parser_class=ArgumentParser)

        scenario_subparsers.add_parser(Scenario.BUS_STOP.value, help="three-lane one-way major road with three cars, a cyclist, a bus, and a bus stop")
        scenario_subparsers.add_parser(Scenario.CROSSROADS.value, help="two-lane two-way crossroads road with two cars and a pedestrian")
        pedestrians_subparser = scenario_subparsers.add_parser(Scenario.PEDESTRIANS.value, help="two-lane one-way major road with a car and a variable number of pedestrians")
        scenario_subparsers.add_parser(Scenario.PELICAN_CROSSING.value, help="two-lane two-way major road with two cars, two pedestrians, and a pelican crossing")

        pedestrians_subparser.add_argument("-a", "--agent", type=AgentType, choices=[choice for choice in AgentType], default=Config.agent_type, metavar="TYPE", help=f"set agent type as %(metavar)s (choices: {pretty_str_set([choice for choice in AgentType])}, default: {Config.agent_type})")
        pedestrians_subparser.add_argument("-n", "--number", type=positive_int, default=Config.actors, metavar="N", help="set number of actors as %(metavar)s (default: %(default)s)")
        pedestrians_subparser.add_argument("-z", "--zone", help="terminate when pedestrian intersects assertion zone", action="store_true")

        for scenario, scenario_subparser in scenario_subparsers.choices.items():
            mode_subparsers = scenario_subparser.add_subparsers(dest="mode", required=True, metavar="MODE")

            mode_subparsers.add_parser("headless", help="run without rendering")  # headless mode has no additional options

            render_subparser = mode_subparsers.add_parser("render", help="run while rendering to screen")

            render_subparser.add_argument("-k", "--keyboard-agent", help="enable keyboard-control of ego actor", action="store_true")
            render_subparser.add_argument("-r", "--record", metavar="DIR", help="enable video recording of run to %(metavar)s")

    def parse_config(self):
        args = self.parse_args()
        render_mode = RenderMode.HEADLESS if args.mode == "headless" else RenderMode.VIDEO if args.record else RenderMode.SCREEN
        agent_type = args.agent
        return Config(
            run=RunConfig(
                episodes=args.episodes,
                max_timesteps=args.timesteps,
                render_mode=render_mode,
                keyboard_agent=KeyboardAgent() if args.keyboard_agent else None,
                logging_dir=args.log,
                record_dir=args.record,
                verbosity=args.verbosity,
                election=agent_type is AgentType.ELECTION
            ),
            env=EnvConfig(
                frequency=30 if render_mode is RenderMode.VIDEO else 60,  # frequency appears to be locked by Gym rendering
                terminate_collision=args.collisions,
                terminate_offroad=args.offroad,
                terminate_zone=args.zone,
                distance_threshold=EnvConfig.distance_threshold
            ),
            scenario=Scenario(args.scenario),
            actors=args.number,
            agent_type=agent_type,
            seed=args.seed
        )


def setup(config=Config()):
    np_random, np_seed = seeding.np_random(config.seed)
    reporting.get_console(config.run.verbosity).info(f"seed={np_seed}")

    if config.scenario is Scenario.PELICAN_CROSSING:
        env = gym.make('PelicanCrossing-v0', env_config=config.env, np_random=np_random)
    elif config.scenario is Scenario.BUS_STOP:
        env = gym.make('BusStop-v0', env_config=config.env, np_random=np_random)
    elif config.scenario is Scenario.CROSSROADS:
        env = gym.make('Crossroads-v0', env_config=config.env, np_random=np_random)
    elif config.scenario is Scenario.PEDESTRIANS:
        env = gym.make('Pedestrians-v0', num_pedestrians=config.actors, env_config=config.env, np_random=np_random)
    else:
        raise NotImplementedError

    agent = config.run.keyboard_agent if config.run.keyboard_agent is not None else NoopVehicleAgent()
    agents = [agent]
    for actor in env.actors[1:]:
        if isinstance(actor, DynamicActor):
            if isinstance(actor, Pedestrian):
                delay = env.env_config.frequency * 1  # one second delay before reorientation
                if config.agent_type is AgentType.RANDOM:
                    agents.append(RandomPedestrianAgent(np_random=np_random))
                elif config.agent_type is AgentType.RANDOM_CONSTRAINED:
                    agents.append(RandomConstrainedPedestrianAgent(delay=delay, np_random=np_random))
                elif config.agent_type is AgentType.PROXIMITY:
                    agents.append(ProximityPedestrianAgent(delay=delay))
                elif config.agent_type is AgentType.ELECTION:
                    agents.append(ElectionPedestrianAgent(delay=delay))
                else:
                    print(config.agent_type)
                    raise NotImplementedError
            else:
                agents.append(RandomVehicleAgent(np_random=np_random))
        elif isinstance(actor, TrafficLight) or isinstance(actor, PelicanCrossing):
            agents.append(RandomTrafficLightAgent(np_random=np_random))

    return env, agents, config.run
