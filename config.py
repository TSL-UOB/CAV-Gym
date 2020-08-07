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
    ProximityPedestrianAgent, ElectionPedestrianAgent, RandomVehicleAgent, RandomTrafficLightAgent, NoopAgent, \
    ApproximateQLearningAgent
from scenarios.constants import M2PX


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
    Q_LEARNING = "q-learning"

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
class AgentConfig:
    alpha: float = 0.1
    threshold: float = M2PX * 22.5
    epsilon: float = 0.01
    gamma: float = 0.99
    type: AgentType = AgentType.RANDOM


@dataclass(frozen=True)
class Config:
    run: RunConfig = RunConfig()
    env: EnvConfig = EnvConfig()
    scenario: Scenario = Scenario.PEDESTRIANS
    actors: int = 3
    seed: int = None
    agent: AgentConfig = AgentConfig()

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

        def zero_one_float(value):
            fvalue = float(value)
            if fvalue < 0 or fvalue > 1:
                raise ArgumentTypeError(f"invalid [0, 1] value: {value}")
            return fvalue

        def non_negative_float(value):
            fvalue = float(value)
            if fvalue <= 0:
                raise ArgumentTypeError(f"invalid non-negative float value: {value}")
            return fvalue

        self.add_argument("-c", "--collisions", help="terminate when any collision occurs", action="store_true")
        self.add_argument("-e", "--episodes", type=positive_int, default=RunConfig.episodes, metavar="N", help="set number of episodes as %(metavar)s (default: %(default)s)")
        self.add_argument("-l", "--log", metavar="DIR", help="enable logging of run to %(metavar)s")
        self.add_argument("-o", "--offroad", help="terminate when ego does not intersect road", action="store_true")
        self.add_argument("-s", "--seed", type=non_negative_int, metavar="N", help="set random seed as %(metavar)s")
        self.add_argument("-t", "--timesteps", type=positive_int, default=RunConfig.max_timesteps, metavar="N", help="set max timesteps per episode as %(metavar)s (default: %(default)s)")
        self.add_argument("-v", "--verbosity", type=Verbosity, choices=[choice for choice in Verbosity], default=RunConfig.verbosity, metavar="LEVEL", help=f"set verbosity as %(metavar)s (choices: {pretty_str_set([choice for choice in Verbosity])}, default: {RunConfig.verbosity})")

        self.set_defaults(keyboard_agent=False, record=None, number=None, agent=AgentConfig.type, zone=False, alpha=AgentConfig.alpha, threshold=AgentConfig.threshold, epsilon=AgentConfig.epsilon, gamma=AgentConfig.gamma)

        def add_mode_options(parser):
            mode_subparsers = parser.add_subparsers(dest="mode", required=True, metavar="MODE")

            mode_subparsers.add_parser("headless", help="run without rendering")  # headless mode has no additional options
            render_subparser = mode_subparsers.add_parser("render", help="run while rendering to screen")

            render_subparser.add_argument("-k", "--keyboard-agent", help="enable keyboard-control of ego actor", action="store_true")
            render_subparser.add_argument("-r", "--record", metavar="DIR", help="enable video recording of run to %(metavar)s")

        def add_agent_options(parser):
            agent_subparsers = parser.add_subparsers(dest="agent", required=True, metavar="AGENT")

            random_subparser = agent_subparsers.add_parser(AgentType.RANDOM.value, help=f"randomly choose action")
            random_constrained_subparser = agent_subparsers.add_parser(AgentType.RANDOM_CONSTRAINED.value, help=f"randomly choose crossing behaviour")
            proximity_subparser = agent_subparsers.add_parser(AgentType.PROXIMITY.value, help=f"choose crossing behaviour based on proximity to ego")
            election_subparser = agent_subparsers.add_parser(AgentType.ELECTION.value, help=f"vote for crossing behaviour based on proximity to ego")
            q_learning_subparser = agent_subparsers.add_parser(AgentType.Q_LEARNING.value, help=f"choose action based on Q learning")

            q_learning_subparser.add_argument("--alpha", type=zero_one_float, default=AgentConfig.alpha, metavar="N", help="set learning rate as %(metavar)s (default: %(default)s)")
            for agent_subparser in [proximity_subparser, election_subparser]:
                agent_subparser.add_argument("--threshold", type=non_negative_float, default=AgentConfig.threshold, metavar="N", help="set distance threshold as %(metavar)s (default: %(default)s)")
            for agent_subparser in [random_subparser, random_constrained_subparser, q_learning_subparser]:
                agent_subparser.add_argument("--epsilon", type=zero_one_float, default=AgentConfig.epsilon, metavar="N", help="set exploration probability as %(metavar)s (default: %(default)s)")
            q_learning_subparser.add_argument("--gamma", type=zero_one_float, default=AgentConfig.gamma, metavar="N", help="set discount factor as %(metavar)s (default: %(default)s)")

            for _, agent_subparser in agent_subparsers.choices.items():
                add_mode_options(agent_subparser)

        def add_scenario_subparser(parser):
            scenario_subparsers = parser.add_subparsers(dest="scenario", required=True, metavar="SCENARIO", parser_class=ArgumentParser)

            bus_stop_subparser = scenario_subparsers.add_parser(Scenario.BUS_STOP.value, help="three-lane one-way major road with three cars, a cyclist, a bus, and a bus stop")
            crossroads_subparser = scenario_subparsers.add_parser(Scenario.CROSSROADS.value, help="two-lane two-way crossroads road with two cars and a pedestrian")
            pedestrians_subparser = scenario_subparsers.add_parser(Scenario.PEDESTRIANS.value, help="two-lane one-way major road with a car and a variable number of pedestrians")
            pelican_crossing_subparser = scenario_subparsers.add_parser(Scenario.PELICAN_CROSSING.value, help="two-lane two-way major road with two cars, two pedestrians, and a pelican crossing")

            pedestrians_subparser.add_argument("-n", "--number", type=positive_int, default=Config.actors, metavar="N", help="set number of actors as %(metavar)s (default: %(default)s)")
            pedestrians_subparser.add_argument("-z", "--zone", help="terminate when pedestrian intersects assertion zone", action="store_true")

            add_agent_options(pedestrians_subparser)
            for scenario_subparser in [bus_stop_subparser, crossroads_subparser, pelican_crossing_subparser]:
                add_mode_options(scenario_subparser)

        add_scenario_subparser(self)

    def parse_config(self):
        args = self.parse_args()
        render_mode = RenderMode.HEADLESS if args.mode == "headless" else RenderMode.VIDEO if args.record else RenderMode.SCREEN
        agent_type = AgentType(args.agent)
        return Config(
            run=RunConfig(
                episodes=args.episodes,
                max_timesteps=args.timesteps,
                render_mode=render_mode,
                keyboard_agent=KeyboardAgent(index=0) if args.keyboard_agent else None,
                logging_dir=args.log,
                record_dir=args.record,
                verbosity=args.verbosity,
                election=agent_type is AgentType.ELECTION
            ),
            env=EnvConfig(
                frequency=30 if render_mode is RenderMode.VIDEO else 60,  # frequency appears to be locked by Gym rendering
                terminate_collision=args.collisions,
                terminate_offroad=args.offroad,
                terminate_zone=args.zone
            ),
            scenario=Scenario(args.scenario),
            actors=args.number,
            seed=args.seed,
            agent=AgentConfig(
                alpha=args.alpha,
                threshold=args.threshold,
                epsilon=args.epsilon,
                gamma=args.gamma,
                type=agent_type
            )
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

    agent = config.run.keyboard_agent if config.run.keyboard_agent is not None else NoopAgent(index=0)
    agents = [agent]
    for i, actor in enumerate(env.actors[1:], start=1):
        if isinstance(actor, DynamicActor):
            if isinstance(actor, Pedestrian):
                if config.agent.type is AgentType.RANDOM:
                    agents.append(RandomPedestrianAgent(index=i, epsilon=config.agent.epsilon, np_random=np_random))
                elif config.agent.type is AgentType.RANDOM_CONSTRAINED:
                    agents.append(RandomConstrainedPedestrianAgent(index=i, epsilon=config.agent.epsilon, road=env.constants.road_map.major_road, np_random=np_random))
                elif config.agent.type is AgentType.PROXIMITY:
                    agents.append(ProximityPedestrianAgent(index=i, road=env.constants.road_map.major_road, distance_threshold=config.agent.threshold))
                elif config.agent.type is AgentType.ELECTION:
                    agents.append(ElectionPedestrianAgent(index=i, road=env.constants.road_map.major_road, distance_threshold=config.agent.threshold))
                elif config.agent.type is AgentType.Q_LEARNING:
                    agents.append(ApproximateQLearningAgent(index=i, alpha=config.agent.alpha, epsilon=config.agent.epsilon, gamma=config.agent.gamma, ego_constants=env.actors[0].constants, self_constants=env.actors[1].constants, road_polgon=env.constants.road_map.major_road.static_bounding_box, time_resolution=env.time_resolution, width=env.constants.viewer_width, height=env.constants.viewer_height, np_random=np_random))
                else:
                    raise NotImplementedError
            else:
                agents.append(RandomVehicleAgent(index=i, np_random=np_random))
        elif isinstance(actor, TrafficLight) or isinstance(actor, PelicanCrossing):
            agents.append(RandomTrafficLightAgent(np_random=np_random))

    return env, agents, config.run
