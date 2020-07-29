import timeit

from gym import wrappers

import reporting
from config import RunConfig, RenderMode
from scenarios.election import Election


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

        self.console = reporting.get_console(self.run_config.verbosity)

        self.episode_file = None
        self.run_file = None
        if self.run_config.logging_dir is not None:
            self.episode_file, self.run_file = reporting.get_file_loggers(self.run_config.logging_dir)

    def conditional_render(self):
        if self.run_config.render_mode is not RenderMode.HEADLESS:
            self.env.render()
            if self.run_config.keyboard_agent is not None and self.env.unwrapped.viewer.window.on_key_press is not self.run_config.keyboard_agent.key_press:  # must render before key_press can be assigned
                self.env.unwrapped.viewer.window.on_key_press = self.run_config.keyboard_agent.key_press

    def run(self):
        self.console.info(f"actors={reporting.pretty_str_list(actor.__class__.__name__ for actor in self.env.actors)}")
        self.console.info(f"agents={reporting.pretty_str_list(agent.__class__.__name__ for agent in self.agents)}")
        self.console.info(f"ego=({self.env.actors[0].__class__.__name__}, {self.agents[0].__class__.__name__})")

        run_start_time = timeit.default_timer()
        for previous_episode in range(self.run_config.episodes):  # initially previous_episode=0
            episode_start_time = timeit.default_timer()
            episode = previous_episode + 1

            joint_observation = self.env.reset()
            info = None

            self.console.debug(f"observation={joint_observation}")

            for agent in self.agents:
                agent.reset()

            self.conditional_render()

            timestep = None
            for previous_timestep in range(self.run_config.max_timesteps):  # initially previous_timestep=0
                timestep = previous_timestep + 1

                joint_action = [agent.choose_action(previous_observation, action_space) for agent, previous_observation, action_space in zip(self.agents, joint_observation, self.env.action_space)]

                if self.election:
                    joint_action = self.election.result(joint_observation, joint_action)

                previous_joint_observation = joint_observation
                joint_observation, joint_reward, done, info = self.env.step(joint_action)

                self.console.debug(f"timestep={timestep}")
                self.console.debug(f"action={joint_action}")
                self.console.debug(f"observation={joint_observation}")
                self.console.debug(f"reward={joint_reward}")
                self.console.debug(f"done={done}")
                self.console.debug(f"info={info}")

                for agent, previous_observation, action, observation, reward in zip(self.agents, previous_joint_observation, joint_action, joint_observation, joint_reward):
                    agent.process_feedback(previous_observation, action, observation, reward)

                self.conditional_render()

                if done:
                    break
            else:
                if self.run_config.record_dir is not None:
                    self.env.stats_recorder.done = True  # need to manually tell the monitor that the episode is over if it runs to completion (not sure why)

            episode_end_time = timeit.default_timer()
            episode_results = reporting.analyse_episode(episode, episode_start_time, episode_end_time, timestep, info, self.run_config, self.env)
            reporting.episode_data.append(episode_results)
            self.console.info(episode_results.console_message())
            if self.episode_file:
                self.episode_file.info(episode_results.file_message())
        else:
            run_end_time = timeit.default_timer()
            run_results = reporting.analyse_run(run_start_time, run_end_time, self.run_config, self.env)
            self.console.info(run_results.console_message())
            if self.run_file:
                self.run_file.info(run_results.file_message())

        self.env.close()
