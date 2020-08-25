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


if __name__ == '__main__':
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    parameters = [(alpha, gamma, epsilon) for alpha in alphas for gamma in gammas for epsilon in epsilons]

    pool = Pool(10)
    pool.starmap(run, parameters)
