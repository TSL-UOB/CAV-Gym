import timeit

from gym import wrappers

import reporting
from config import RunConfig, RenderMode
from reporting import MyLogger
from scenarios.election import Election
from utilities import pretty_str_list


class Simulation:
    def __init__(self, env, agents, run_config=RunConfig()):
        assert len(env.actors) == len(agents), "each actor must be assigned an agent and vice versa"
        self.env = env
        self.agents = agents
        self.run_config = run_config

        if self.run_config.keyboard_agent is not None or self.run_config.record_dir is not None:
            assert self.run_config.render_mode is not RenderMode.HEADLESS, "keyboard agents and recordings do not work in headless mode"

        if self.run_config.record_dir is not None:
            from library import mods  # lazy import of pyglet to allow headless mode on headless machines
            self.env = wrappers.Monitor(self.env, self.run_config.record_dir, video_callable=lambda episode_id: True, force=True)  # save all episodes instead of default behaviour (episodes 1, 8, 27, 64, ...)
            self.env.stats_recorder = mods.make_joint_stats_recorder(self.env, len(agents))  # workaround to avoid bugs due to existence of joint rewards

        if self.run_config.election:
            self.election = Election(env, agents)
        else:
            self.election = None

        self.logger = MyLogger(self.run_config.verbosity)

        if self.run_config.logging_dir is not None:
            self.logger.set_file_loggers(self.run_config.logging_dir)

    def conditional_render(self):
        if self.run_config.render_mode is not RenderMode.HEADLESS:
            self.env.render()
            if self.run_config.keyboard_agent is not None and self.env.unwrapped.viewer.window.on_key_press is not self.run_config.keyboard_agent.key_press:  # must render before key_press can be assigned
                self.env.unwrapped.viewer.window.on_key_press = self.run_config.keyboard_agent.key_press

    def run(self):
        self.logger.console.info(f"actors={pretty_str_list(actor.__class__.__name__ for actor in self.env.actors)}")
        self.logger.console.info(f"agents={pretty_str_list(agent.__class__.__name__ for agent in self.agents)}")
        self.logger.console.info(f"ego=({self.env.actors[0].__class__.__name__}, {self.agents[0].__class__.__name__})")

        run_start_time = timeit.default_timer()
        for previous_episode in range(self.run_config.episodes):  # initially previous_episode=0
            episode_start_time = timeit.default_timer()
            episode = previous_episode + 1

            joint_observation = self.env.reset()
            info = None

            self.logger.console.debug(f"observation={joint_observation}")

            for agent in self.agents:
                agent.reset()

            self.conditional_render()

            timestep = None
            for previous_timestep in range(self.run_config.max_timesteps):  # initially previous_timestep=0
                timestep = previous_timestep + 1

                joint_action = [agent.choose_action(previous_observation, action_space) for agent, previous_observation, action_space in zip(self.agents, joint_observation, self.env.action_space)]

                if self.election:
                    joint_action = self.election.result(joint_observation, joint_action)

                joint_observation, joint_reward, done, info = self.env.step(joint_action)

                self.logger.console.debug(f"timestep={timestep}")
                self.logger.console.debug(f"action={joint_action}")
                self.logger.console.debug(f"observation={joint_observation}")
                self.logger.console.debug(f"reward={joint_reward}")
                self.logger.console.debug(f"done={done}")
                self.logger.console.debug(f"info={info}")

                self.conditional_render()

                if done:
                    break
            else:
                if self.run_config.record_dir is not None:
                    self.env.stats_recorder.done = True  # need to manually tell the monitor that the episode is over if it runs to completion (not sure why)

            episode_end_time = timeit.default_timer()
            episode_results = reporting.analyse_episode(episode, episode_start_time, episode_end_time, timestep, info, self.run_config, self.env)
            reporting.episode_data.append(episode_results)
            self.logger.console.info(episode_results.console_message())
            if self.logger.episode_file:
                self.logger.episode_file.info(episode_results.file_message())
        else:
            run_end_time = timeit.default_timer()
            run_results = reporting.analyse_run(run_start_time, run_end_time, self.run_config, self.env)
            self.logger.console.info(run_results.console_message())
            if self.logger.run_file:
                self.logger.run_file.info(run_results.file_message())

        self.env.close()
