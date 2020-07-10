from config import ConfigParser, setup
from simulation import Simulation

if __name__ == '__main__':
    cli_args = ConfigParser()
    cli_config = cli_args.parse_config()
    cli_env, cli_agents, cli_run_config = setup(cli_config)
    simulation = Simulation(cli_env, cli_agents, run_config=cli_run_config)
    simulation.run()
    # import cProfile
    # cProfile.run("run_simulation(cli_env, cli_agents, core_config=cli_run_config)", sort="tottime")  # reminder: there is a significant performance hit when using cProfile
