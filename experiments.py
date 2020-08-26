from argparse import ArgumentParser, ArgumentTypeError
from multiprocessing import Pool

from reporting import Verbosity
from config import Config, PedestriansConfig, HeadlessConfig, QLearningConfig, FeatureConfig
from simulation import Simulation


def make_config(alpha, gamma, epsilon):
    log_dir = f"logs/alpha={alpha}/gamma={gamma}/epsilon={epsilon}"
    return log_dir, Config(
        verbosity=Verbosity.SILENT,
        episode_log=f"{log_dir}/episode.log",
        run_log=f"{log_dir}/run.log",
        seed=0,
        episodes=100,
        max_timesteps=1000,
        collisions=False,
        offroad=True,
        zone=True,
        living_cost=1.0,
        road_cost=5.0,
        win_reward=6000.0,
        scenario_config=PedestriansConfig(actors=1),
        agent_config=QLearningConfig(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
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


def run(alpha, gamma, epsilon):
    print(f"starting: alpha={alpha}, gamma={gamma}, epsilon={epsilon}")

    log_dir, config = make_config(alpha, gamma, epsilon)
    config.write_json(f"{log_dir}/config.json")

    np_seed, env, agents, keyboard_agent = config.setup()

    simulation = Simulation(env, agents, config=config, keyboard_agent=keyboard_agent)
    simulation.run()

    print(f"finished: alpha={alpha}, gamma={gamma}, epsilon={epsilon}")


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
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    gammas = [0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

    parameters = [(alpha, gamma, epsilon) for alpha in alphas for gamma in gammas for epsilon in epsilons]

    parser = PoolParser()
    pool = parser.parse_pool()
    pool.starmap(run, parameters)
