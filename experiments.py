from argparse import ArgumentParser, ArgumentTypeError
from multiprocessing import Pool

from reporting import Verbosity
from config import Config, PedestriansConfig, HeadlessConfig, QLearningConfig, FeatureConfig
from simulation import Simulation


def make_config(outbound_pavement, inbound_pavement):
    log_dir = f"logs/outbound_pavement={outbound_pavement}/inbound_pavement={inbound_pavement}"
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
            outbound_pavement=outbound_pavement,
            inbound_pavement=inbound_pavement
        ),
        agent_config=QLearningConfig(
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
        ),
        mode_config=HeadlessConfig()
    )


def run(outbound_pavement, inbound_pavement):
    print(f"starting: outbound_pavement={outbound_pavement}, inbound_pavement={inbound_pavement}")

    log_dir, config = make_config(outbound_pavement, inbound_pavement)
    config.write_json(f"{log_dir}/config.json")

    np_seed, env, agents, keyboard_agent = config.setup()

    simulation = Simulation(env, agents, config=config, keyboard_agent=keyboard_agent)
    simulation.run()

    print(f"finished: outbound_pavement={outbound_pavement}, inbound_pavement={inbound_pavement}")


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
    outbound_pavements = [1.0, 0.75, 0.5, 0.25, 0.0]
    inbound_pavements = [1.0, 0.75, 0.5, 0.25, 0.0]

    parameters = [(outbound_pavement, inbound_pavement) for outbound_pavement in outbound_pavements for inbound_pavement in inbound_pavements if outbound_pavement > 0 or inbound_pavement > 0]

    parser = PoolParser()
    pool = parser.parse_pool()
    pool.starmap(run, parameters)
