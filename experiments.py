from argparse import ArgumentParser, ArgumentTypeError
from multiprocessing import Pool

from reporting import Verbosity
from config import Config, PedestriansConfig, HeadlessConfig, QLearningConfig, FeatureConfig, AgentType, RandomConfig, \
    RandomConstrainedConfig, ProximityConfig, ElectionConfig
from simulation import Simulation


def make_agent_config(agent_type, log_dir, **kwargs):
    if agent_type is AgentType.RANDOM:
        return RandomConfig(**kwargs)
    elif agent_type is AgentType.RANDOM_CONSTRAINED:
        return RandomConstrainedConfig(**kwargs)
    elif agent_type is AgentType.PROXIMITY:
        return ProximityConfig(**kwargs)
    elif agent_type is AgentType.ELECTION:
        return ElectionConfig(**kwargs)
    elif agent_type is AgentType.Q_LEARNING:
        return QLearningConfig(
            alpha=0.18,
            gamma=0.87,
            epsilon=0.0005,
            features=FeatureConfig(
                distance_x=True,
                distance_y=True,
                distance=True,
                on_road=False,
                facing=False,
                inverse_distance=True
            ),
            log=f"{log_dir}/qlearning.log"
        )
    else:
        raise NotImplementedError


def make_config(agent_type, epsilon):
    log_dir = f"logs/agent_type={agent_type}/epsilon={epsilon}"
    return log_dir, Config(
        verbosity=Verbosity.SILENT,
        episode_log=f"{log_dir}/episode.log",
        run_log=f"{log_dir}/run.log",
        seed=0,
        episodes=1000,
        max_timesteps=1000,
        collisions=False,
        offroad=True,
        zone=True,
        living_cost=1.0,
        road_cost=5.0,
        win_reward=6000.0,
        scenario_config=PedestriansConfig(
            actors=1,
            outbound_pavement=1.0,
            inbound_pavement=1.0
        ),
        agent_config=make_agent_config(agent_type, log_dir, epsilon=epsilon),
        mode_config=HeadlessConfig()
    )


def run(agent_type, epsilon):
    print(f"starting: agent_type={agent_type}, epsilon={epsilon}")

    log_dir, config = make_config(agent_type, epsilon)
    config.write_json(f"{log_dir}/config.json")

    np_seed, env, agents, keyboard_agent = config.setup()

    simulation = Simulation(env, agents, config=config, keyboard_agent=keyboard_agent)
    simulation.run()

    print(f"finished: agent_type={agent_type}, epsilon={epsilon}")


class PoolParser(ArgumentParser):
    def __init__(self):
        super().__init__()

        def positive_int(value):
            ivalue = int(value)
            if ivalue < 1:
                raise ArgumentTypeError(f"invalid positive int value: {value}")
            return ivalue

        self.add_argument("-p", "--processes", type=positive_int, default=10, metavar="N", help="set number of processes as %(metavar)s (default: %(default)s)")

    def parse_pool(self):
        args = self.parse_args()
        return Pool(args.processes)


if __name__ == '__main__':
    agent_types = [AgentType.RANDOM_CONSTRAINED]
    epsilons = [i / 100 for i in range(1, 100)]

    parameters = [(agent_type, epsilon) for agent_type in agent_types for epsilon in epsilons]

    parser = PoolParser()
    pool = parser.parse_pool()
    pool.starmap(run, parameters)
