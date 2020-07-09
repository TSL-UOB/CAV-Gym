import argparse
import logging
import pathlib
import sys
import timeit
from dataclasses import dataclass
from enum import Enum

import gym
from gym import wrappers
from gym.utils import seeding
from scipy import stats

import scenarios  # noqa
from scenarios.agents import RandomTrafficLightAgent, RandomVehicleAgent, KeyboardAgent, RandomConstrainedPedestrianAgent, \
    NoopVehicleAgent, RandomPedestrianAgent, ProximityPedestrianAgent, ElectionPedestrianAgent
import utilities
from library.actions import OrientationAction
from library.actors import DynamicActor, TrafficLight, PelicanCrossing, Pedestrian
from library.environment import EnvConfig
from library.observations import OrientationObservation


class CustomLogRecord(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.code = f"{self.filename}:{self.funcName}"  # new variable used to align existing variables as group


logging.setLogRecordFactory(CustomLogRecord)
console_formatter = logging.Formatter("%(levelname)-7s %(relativeCreated)-7d %(code)-27s %(message)s")
file_formatter = logging.Formatter("%(message)s")


def set_console_logger(logger, destination, event_filter):
    handler = logging.StreamHandler(destination)
    handler.addFilter(event_filter)
    handler.setFormatter(console_formatter)
    logger.addHandler(handler)


console_logger = logging.getLogger("library.console")

set_console_logger(console_logger, sys.stdout, lambda record: record.levelno <= logging.INFO)  # redirect INFO events and below to stdout (to avoid duplicate events)
set_console_logger(console_logger, sys.stderr, lambda record: record.levelno > logging.INFO)  # redirect WARNING events and above to stderr (to avoid duplicate events)


def setup_file_logger_output(name, directory, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(f"{directory}/{filename}")
    handler.setFormatter(file_formatter)
    logger.addHandler(handler)
    return logger


file_episodes_logger = None
file_run_logger = None


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


class Verbosity(Enum):
    INFO = "info"
    DEBUG = "debug"
    SILENT = "silent"

    def __str__(self):
        return self.value

    def logging_level(self):
        if self is Verbosity.DEBUG:
            return logging.DEBUG
        elif self is Verbosity.SILENT:
            return logging.WARNING
        else:
            return logging.INFO


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


class ConfigParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()

        def non_negative_int(value):
            ivalue = int(value)
            if ivalue < 0:
                raise argparse.ArgumentTypeError(f"invalid non-negative int value: {value}")
            return ivalue

        def positive_int(value):
            ivalue = int(value)
            if ivalue < 1:
                raise argparse.ArgumentTypeError(f"invalid positive int value: {value}")
            return ivalue

        self.add_argument("-c", "--collisions", help="terminate when any collision occurs", action="store_true")
        self.add_argument("-e", "--episodes", type=positive_int, default=RunConfig.episodes, metavar="N", help="set number of episodes as %(metavar)s (default: %(default)s)")
        self.add_argument("-l", "--log", metavar="DIR", help="enable logging of run to %(metavar)s")
        self.add_argument("-o", "--offroad", help="terminate when ego does not intersect road", action="store_true")
        self.add_argument("-s", "--seed", type=non_negative_int, metavar="N", help="set random seed as %(metavar)s")
        self.add_argument("-t", "--timesteps", type=positive_int, default=RunConfig.max_timesteps, metavar="N", help="set max timesteps per episode as %(metavar)s (default: %(default)s)")
        self.add_argument("-v", "--verbosity", type=Verbosity, choices=[choice for choice in Verbosity], default=RunConfig.verbosity, metavar="LEVEL", help=f"set verbosity as %(metavar)s (choices: {utilities.pretty_str_set([choice for choice in Verbosity])}, default: {RunConfig.verbosity})")

        self.set_defaults(keyboard_agent=False, record=None, number=None, agent=Config.agent_type, zone=False)

        scenario_subparsers = self.add_subparsers(dest="scenario", required=True, metavar="SCENARIO", parser_class=argparse.ArgumentParser)

        scenario_subparsers.add_parser(Scenario.BUS_STOP.value, help="three-lane one-way major road with three cars, a cyclist, a bus, and a bus stop")
        scenario_subparsers.add_parser(Scenario.CROSSROADS.value, help="two-lane two-way crossroads road with two cars and a pedestrian")
        pedestrians_subparser = scenario_subparsers.add_parser(Scenario.PEDESTRIANS.value, help="two-lane one-way major road with a car and a variable number of pedestrians")
        scenario_subparsers.add_parser(Scenario.PELICAN_CROSSING.value, help="two-lane two-way major road with two cars, two pedestrians, and a pelican crossing")

        pedestrians_subparser.add_argument("-a", "--agent", type=AgentType, choices=[choice for choice in AgentType], default=Config.agent_type, metavar="TYPE", help=f"set agent type as %(metavar)s (choices: {utilities.pretty_str_set([choice for choice in AgentType])}, default: {Config.agent_type})")
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
    console_logger.setLevel(config.run.verbosity.logging_level())

    np_random, np_seed = seeding.np_random(config.seed)
    console_logger.info(f"seed={np_seed}")

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


def run(env, agents, run_config=RunConfig()):
    console_logger.setLevel(run_config.verbosity.logging_level())

    assert len(env.actors) == len(agents), "each actor must be assigned an agent and vice versa"

    console_logger.info(f"actors={utilities.pretty_str_list(actor.__class__.__name__ for actor in env.actors)}")
    console_logger.info(f"agents={utilities.pretty_str_list(agent.__class__.__name__ for agent in agents)}")
    console_logger.info(f"ego=({env.actors[0].__class__.__name__}, {agents[0].__class__.__name__})")

    if run_config.keyboard_agent is not None or run_config.record_dir is not None:
        assert run_config.render_mode is not RenderMode.HEADLESS, "keyboard agents and recordings do not work in headless mode"

    if run_config.logging_dir is not None:
        global file_episodes_logger
        global file_run_logger
        file_episodes_logger = setup_file_logger_output("library.file.episodes", run_config.logging_dir, "episodes.log")
        file_run_logger = setup_file_logger_output("library.file.run", run_config.logging_dir, "run.log")

    if run_config.record_dir is not None:
        from library import mods  # lazy import of pyglet to allow headless mode on headless machines
        env = wrappers.Monitor(env, run_config.record_dir, video_callable=lambda episode_id: True, force=True)  # save all episodes instead of default behaviour (episodes 1, 8, 27, 64, ...)
        env.stats_recorder = mods.make_joint_stats_recorder(env, len(agents))  # workaround to avoid bugs due to existence of joint rewards

    if run_config.render_mode is not RenderMode.HEADLESS and run_config.keyboard_agent is not None:
        env.render()  # must render before key_press can be assigned
        env.unwrapped.viewer.window.on_key_press = run_config.keyboard_agent.key_press

    def measure_time(start_time):
        return (timeit.default_timer() - start_time) * 1000

    def determine_simulation_speed(time_ms, timesteps):
        return (timesteps * env.time_resolution * 1000) / time_ms

    def analyse_episode(start_time, timesteps, env_info):
        runtime = measure_time(start_time)
        simulation_speed = determine_simulation_speed(runtime, timesteps)
        assert 1 <= timesteps <= run_config.max_timesteps
        completed = timesteps == run_config.max_timesteps
        assert 'pedestrian' in env_info
        pedestrian_index = env_info['pedestrian']
        successful = pedestrian_index is not None
        score = env.episode_liveness[pedestrian_index] * -5 if successful else 0
        return runtime, simulation_speed, completed, successful, score

    def report_episode(index, timesteps, runtime, simulation_speed, completed, successful, score):
        episode_status = "completed" if completed else "terminated"
        test_status = f"successful test with score {score}" if successful else "unsuccessful test"
        console_logger.info(f"episode {index} {episode_status} after {timesteps} timestep(s) in {utilities.pretty_float(runtime, decimal_places=0)} ms ({utilities.pretty_float(simulation_speed)}:1 real-time), {test_status}")
        if file_episodes_logger is not None:
            file_episodes_logger.info(f"{episode},{timesteps},{runtime},{1 if successful else 0},{score}")

    def active_joint_action(joint_action_vote, electorate, active):
        assert active in electorate
        for i in electorate:
            if i != active:  # only the winner gets to cross
                agents[i].reset()  # tell the agent their orientation action will not be executed
                velocity_action_id, _ = joint_action_vote[i]  # allow the agent's velocity action to be executed
                joint_action_vote[i] = velocity_action_id, OrientationAction.NOOP.value  # reset the agent's orientation action
        return joint_action_vote

    active_election_agent = None

    def election(previous_joint_observation, joint_action_vote):
        nonlocal active_election_agent

        electorate = [i for i, agent in enumerate(agents) if isinstance(agent, ElectionPedestrianAgent)]

        if active_election_agent and not agents[active_election_agent].crossing_action:
            _, orientation_observation_id, _, _ = previous_joint_observation[active_election_agent]
            orientation_observation = OrientationObservation(orientation_observation_id)
            if orientation_observation is OrientationObservation.INACTIVE:
                previous_active_election_agent = active_election_agent
                active_election_agent = None  # previous winner has finished crossing
                return active_joint_action(joint_action_vote, electorate, previous_active_election_agent)  # need to execute final action

        if active_election_agent:
            return active_joint_action(joint_action_vote, electorate, active_election_agent)

        winner = None

        voters = []
        for i in electorate:
            _, orientation_action_id = joint_action_vote[i]
            orientation_action = OrientationAction(orientation_action_id)
            if orientation_action is not OrientationAction.NOOP:  # agent has voted to cross
                voters.append(i)

        if voters:
            joint_winners = []
            winning_distance = float("inf")
            for i in voters:
                distance = env.actors[i].state.position.distance(env.ego.state.position)
                if distance < winning_distance:
                    joint_winners = [i]
                    winning_distance = distance
                elif distance == winning_distance:
                    joint_winners.append(i)
            winner = env.np_random.choice(joint_winners) if joint_winners else None

        if winner:
            active_election_agent = winner  # no elections show take place until the winner finishes crossing
            return active_joint_action(joint_action_vote, electorate, active_election_agent)
        else:
            return joint_action_vote

    episode_data = list()

    def analyse_run(start_time):
        runtime = measure_time(start_time)

        # row = (episode, timesteps, runtime, success, score)
        timesteps = sum([row[1] for row in episode_data])

        simulation_speed = determine_simulation_speed(runtime, timesteps)

        successful_test_data = [row for row in episode_data if row[3]]
        successful_test_data_timesteps = [row[1] for row in successful_test_data]
        successful_test_data_runtime = [row[2] for row in successful_test_data]
        successful_test_data_score = [row[4] for row in successful_test_data]

        successful_tests = len(successful_test_data)
        assert successful_tests == len(successful_test_data_timesteps) == len(successful_test_data_runtime) == len(successful_test_data_score)

        def confidence_interval(data, alpha=0.05):  # 95% confidence interval
            data_length = len(data)
            if data_length <= 1:  # variance (and thus data_sem) requires at least 2 data points
                nan = float("nan")
                return nan, nan
            else:
                data_mean = sum(data) / data_length
                # data_sem = statistics.stdev(data) / math.sqrt(data_length)  # use data_sem = statistics.pstdev(data) if data is population
                # data_sem = np.std(data, ddof=1) / math.sqrt(data_length)  # use data_sem = np.std(data) / math.sqrt(data_length) if data is population
                data_sem = stats.sem(data)  # use data_sem = scipy.stats.sem(data, ddof=0) if data is population
                data_df = data_length - 1
                # data_error = data_sem * stats.t.isf(alpha / 2, data_df)  # equivalent to data_error = data_sem * -stats.t.ppf(alpha / 2, data_df)
                # return data_mean - data_error, data_mean + data_error
                return stats.t.interval(1 - alpha, data_df, loc=data_mean, scale=data_sem)

        successful_tests_timesteps_ci = confidence_interval(successful_test_data_timesteps)
        successful_tests_runtime_ci = confidence_interval(successful_test_data_runtime)
        successful_tests_score_ci = confidence_interval(successful_test_data_score)

        return timesteps, runtime, simulation_speed, successful_tests, successful_tests_timesteps_ci, successful_tests_runtime_ci, successful_tests_score_ci

    def report_run(timesteps, runtime, simulation_speed, successful_tests, successful_tests_timesteps_ci, successful_tests_runtime_ci, successful_tests_score_ci):
        test_status = f"{successful_tests} successful test(s) with timestep confidence {utilities.pretty_float_tuple(successful_tests_timesteps_ci, decimal_places=0)}, runtime confidence {utilities.pretty_float_tuple(successful_tests_runtime_ci, decimal_places=0)}, and score confidence {utilities.pretty_float_tuple(successful_tests_score_ci, decimal_places=0)}" if successful_tests > 0 else "no successful test(s)"
        console_logger.info(f"run completed after {run_config.episodes} episode(s) and {timesteps} timestep(s) in {utilities.pretty_float(runtime, decimal_places=0)} ms ({utilities.pretty_float(simulation_speed)}:1 real-time), {test_status}")
        if file_run_logger is not None:
            file_run_logger.info(f"{run_config.episodes},{timesteps},{runtime},{successful_tests},{successful_tests_timesteps_ci},{successful_tests_runtime_ci},{successful_tests_score_ci}")

    run_start_time = timeit.default_timer()
    episode = None
    for previous_episode in range(run_config.episodes):  # initially previous_episode=0
        episode_start_time = timeit.default_timer()
        episode = previous_episode + 1

        joint_observation = env.reset()
        info = None
        console_logger.debug(f"observation={joint_observation}")

        for agent in agents:
            agent.reset()

        timestep = None
        for previous_timestep in range(run_config.max_timesteps):  # initially previous_timestep=0
            timestep = previous_timestep + 1

            if run_config.render_mode is not RenderMode.HEADLESS:
                env.render()

            joint_action = [agent.choose_action(previous_observation, action_space) for agent, previous_observation, action_space in zip(agents, joint_observation, env.action_space)]
            if run_config.election:
                joint_action = election(joint_observation, joint_action)
            joint_observation, joint_reward, done, info = env.step(joint_action)

            console_logger.debug(f"timestep={timestep}")
            console_logger.debug(f"action={joint_action}")
            console_logger.debug(f"observation={joint_observation}")
            console_logger.debug(f"reward={joint_reward}")
            console_logger.debug(f"done={done}")
            console_logger.debug(f"info={info}")

            if done:
                break
        else:
            if run_config.record_dir is not None:
                env.stats_recorder.done = True  # need to manually tell the monitor that the episode is over if it runs to completion (not sure why)

        episode_runtime, episode_simulation_speed, episode_completed, episode_successful, episode_score = analyse_episode(episode_start_time, timestep, info)
        episode_data.append((episode, timestep, episode_runtime, episode_successful, episode_score))
        report_episode(episode, timestep, episode_runtime, episode_simulation_speed, episode_completed, episode_successful, episode_score)
    else:
        run_results = analyse_run(run_start_time)
        report_run(*run_results)

    env.close()


if __name__ == '__main__':
    cli_args = ConfigParser()
    cli_config = cli_args.parse_config()
    cli_env, cli_agents, cli_run_config = setup(cli_config)
    run(cli_env, cli_agents, run_config=cli_run_config)
    # import cProfile
    # cProfile.run("run_simulation(cli_env, cli_agents, core_config=cli_run_config)", sort="tottime")  # reminder: there is a significant performance hit when using cProfile
