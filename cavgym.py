from config import ConfigParser
from simulation import Simulation


def profile(statement, write=True):
    if write:
        write_profile(statement)
    else:
        stdout_profile(statement)


def stdout_profile(statement, sort="tottime"):
    import cProfile
    cProfile.run(statement, sort=sort)


def write_profile(statement, directory="logs", stats_file="profile.pstats", dot_file="profile.dot", output_file="profile.pdf"):
    import cProfile
    import pathlib
    # noinspection PyPackageRequirements
    import gprof2dot
    import subprocess
    stats_path = f"{directory}/{stats_file}"
    dot_path = f"{directory}/{dot_file}"
    output_path = f"{directory}/{output_file}"
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    cProfile.run(statement, stats_path)
    gprof2dot.main(["-f", "pstats", stats_path, "-o", dot_path])
    subprocess.run(["dot", "-Tpdf", dot_path, "-o", output_path])
    subprocess.run(["rm", stats_path, dot_path])


if __name__ == '__main__':
    parser = ConfigParser()
    config = parser.parse_config()
    np_seed, env, agents, keyboard_agent = config.setup()

    simulation = Simulation(env, agents, config=config, keyboard_agent=keyboard_agent)
    simulation.run()
    # profile("simulation.run()")  # reminder: there is a significant performance hit when using cProfile
