import timeit

from gym import wrappers

import reporting
from config import Mode, AgentType
from examples.election import Election


class Simulation:
    def __init__(self, env, agents, config, keyboard_agent):
        assert len(env.bodies) == len(agents), "each body must be assigned an agent and vice versa"
        self.env = env
        self.agents = agents
        self.config = config
        self.keyboard_agent = keyboard_agent

        if self.keyboard_agent is not None:
            assert self.config.mode_config.mode is Mode.RENDER, "keyboard agents only work in render mode"

        if self.config.mode_config.mode is Mode.RENDER and self.config.mode_config.video_dir is not None:
            from library import mods  # lazy import of pyglet to allow headless mode on headless machines
            self.env = wrappers.Monitor(self.env, self.config.mode_config.video_dir, video_callable=lambda episode_id: True, force=True)  # save all episodes instead of default behaviour (episodes 1, 8, 27, 64, ...)
            self.env.stats_recorder = mods.make_joint_stats_recorder(self.env, len(agents))  # workaround to avoid bugs due to existence of joint rewards

        if self.config.tester_config.agent is AgentType.ELECTION:
            self.election = Election(env, agents)
        else:
            self.election = None

        self.console = reporting.get_console(self.config.verbosity)

        self.episode_file = None
        if self.config.episode_log is not None:
            self.episode_file = reporting.get_episode_file_logger(self.config.episode_log)

        self.run_file = None
        if self.config.run_log is not None:
            self.run_file = reporting.get_run_file_logger(self.config.run_log)

    def should_render(self, episode):
        return self.config.mode_config.mode is Mode.RENDER and episode % self.config.mode_config.episode_condition == 0

    def run(self):
        episode_data = list()
        run_start_time = timeit.default_timer()
        for episode in range(1, self.config.episodes+1):
            episode_start_time = timeit.default_timer()

            state = self.env.reset()
            info = self.env.info()

            self.console.debug(f"state={state}")

            for agent in self.agents:
                agent.reset()

            if self.should_render(episode):
                self.env.render()
                if self.keyboard_agent is not None and self.env.unwrapped.viewer.window.on_key_press is not self.keyboard_agent.key_press:  # must render before key_press can be assigned
                    self.env.unwrapped.viewer.window.on_key_press = self.keyboard_agent.key_press

            final_timestep = self.config.max_timesteps
            for timestep in range(1, self.config.max_timesteps+1):
                joint_action = [agent.choose_action(state, action_space, info) for agent, action_space in zip(self.agents, self.env.action_space)]

                if self.election:
                    joint_action = self.election.result(state, joint_action)

                previous_state = state
                state, joint_reward, done, info = self.env.step(joint_action)

                self.console.debug(f"timestep={timestep}")
                self.console.debug(f"action={joint_action}")
                self.console.debug(f"state={state}")
                self.console.debug(f"reward={joint_reward}")
                self.console.debug(f"done={done}")
                self.console.debug(f"info={info}")

                for agent, action, reward in zip(self.agents, joint_action, joint_reward):
                    agent.process_feedback(previous_state, action, state, reward)

                if self.should_render(episode):
                    self.env.render()

                if done:
                    final_timestep = timestep
                    break
            else:
                if self.config.mode_config.mode is Mode.RENDER and self.config.mode_config.video_dir is not None:
                    self.env.stats_recorder.done = True  # need to manually tell the monitor that the episode is over if it runs to completion (not sure why)

            if self.should_render(episode) and not self.should_render(episode+1):
                self.env.close()  # closes viewer rather than environment

            episode_end_time = timeit.default_timer()
            episode_results = reporting.analyse_episode(episode, episode_start_time, episode_end_time, final_timestep, info, self.config, self.env)
            episode_data.append(episode_results)
            self.console.info(episode_results.console_message())
            if self.episode_file:
                self.episode_file.info(episode_results.file_message())
        else:
            run_end_time = timeit.default_timer()
            run_results = reporting.analyse_run(episode_data, run_start_time, run_end_time, self.config, self.env)
            self.console.info(run_results.console_message())
            if self.run_file:
                self.run_file.info(run_results.file_message())

        self.env.close()  # closes viewer rather than environment
