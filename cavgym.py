import timeit

from gym import wrappers

import reporting
import scenarios  # noqa
import utilities
from config import RunConfig, ConfigParser, setup, RenderMode
from scenarios.election import Election


def run(env, agents, run_config=RunConfig()):
    reporting.console_logger.setLevel(run_config.verbosity.logging_level())

    assert len(env.actors) == len(agents), "each actor must be assigned an agent and vice versa"

    reporting.console_logger.info(f"actors={utilities.pretty_str_list(actor.__class__.__name__ for actor in env.actors)}")
    reporting.console_logger.info(f"agents={utilities.pretty_str_list(agent.__class__.__name__ for agent in agents)}")
    reporting.console_logger.info(f"ego=({env.actors[0].__class__.__name__}, {agents[0].__class__.__name__})")

    if run_config.keyboard_agent is not None or run_config.record_dir is not None:
        assert run_config.render_mode is not RenderMode.HEADLESS, "keyboard agents and recordings do not work in headless mode"

    if run_config.logging_dir is not None:
        reporting.file_episodes_logger = reporting.setup_file_logger_output("library.file.episodes", run_config.logging_dir, "episodes.log")
        reporting.file_run_logger = reporting.setup_file_logger_output("library.file.run", run_config.logging_dir, "run.log")

    if run_config.record_dir is not None:
        from library import mods  # lazy import of pyglet to allow headless mode on headless machines
        env = wrappers.Monitor(env, run_config.record_dir, video_callable=lambda episode_id: True, force=True)  # save all episodes instead of default behaviour (episodes 1, 8, 27, 64, ...)
        env.stats_recorder = mods.make_joint_stats_recorder(env, len(agents))  # workaround to avoid bugs due to existence of joint rewards

    if run_config.render_mode is not RenderMode.HEADLESS and run_config.keyboard_agent is not None:
        env.render()  # must render before key_press can be assigned
        env.unwrapped.viewer.window.on_key_press = run_config.keyboard_agent.key_press

    if run_config.election:
        election = Election(env, agents)
    else:
        election = None

    run_start_time = timeit.default_timer()
    for previous_episode in range(run_config.episodes):  # initially previous_episode=0
        episode_start_time = timeit.default_timer()
        episode = previous_episode + 1

        joint_observation = env.reset()
        info = None
        reporting.console_logger.debug(f"observation={joint_observation}")

        for agent in agents:
            agent.reset()

        timestep = None
        for previous_timestep in range(run_config.max_timesteps):  # initially previous_timestep=0
            timestep = previous_timestep + 1

            if run_config.render_mode is not RenderMode.HEADLESS:
                env.render()

            joint_action = [agent.choose_action(previous_observation, action_space) for agent, previous_observation, action_space in zip(agents, joint_observation, env.action_space)]
            if election:
                joint_action = election.result(joint_observation, joint_action)

            joint_observation, joint_reward, done, info = env.step(joint_action)

            reporting.console_logger.debug(f"timestep={timestep}")
            reporting.console_logger.debug(f"action={joint_action}")
            reporting.console_logger.debug(f"observation={joint_observation}")
            reporting.console_logger.debug(f"reward={joint_reward}")
            reporting.console_logger.debug(f"done={done}")
            reporting.console_logger.debug(f"info={info}")

            if done:
                break
        else:
            if run_config.record_dir is not None:
                env.stats_recorder.done = True  # need to manually tell the monitor that the episode is over if it runs to completion (not sure why)

        episode_end_time = timeit.default_timer()
        episode_results = reporting.analyse_episode(episode, episode_start_time, episode_end_time, timestep, info, run_config, env)
        reporting.episode_data.append(episode_results)
        episode_results.log()
    else:
        run_end_time = timeit.default_timer()
        run_results = reporting.analyse_run(run_start_time, run_end_time, run_config, env)
        run_results.log()

    env.close()


if __name__ == '__main__':
    cli_args = ConfigParser()
    cli_config = cli_args.parse_config()
    cli_env, cli_agents, cli_run_config = setup(cli_config)
    run(cli_env, cli_agents, run_config=cli_run_config)
    # import cProfile
    # cProfile.run("run_simulation(cli_env, cli_agents, core_config=cli_run_config)", sort="tottime")  # reminder: there is a significant performance hit when using cProfile
