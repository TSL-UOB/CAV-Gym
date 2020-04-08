from pathlib import Path

import gym
from gym import wrappers

from agents import RandomDynamicActorAgent, RandomTrafficLightAgent, HumanDynamicActorAgent
from cavgym import mods


def run_pelican_crossing(save_video=False):
    env = gym.make('PelicanCrossing-v0')

    human_agent = HumanDynamicActorAgent()
    agents = [human_agent, RandomDynamicActorAgent(), RandomTrafficLightAgent(), RandomTrafficLightAgent(), RandomDynamicActorAgent()]

    if save_video:
        env = wrappers.Monitor(env, f"{Path.home()}/CAV-Gym/Videos/", video_callable=lambda episode_id: True, force=True)  # save all episodes instead of default behaviour (episodes 1, 8, 27, 64, ...)
        env.stats_recorder = mods.make_joint_stats_recorder(env, len(agents))  # workaround to avoid bugs due to existence of joint rewards

    env.render()  # must render before key_press can be assigned
    env.unwrapped.viewer.window.on_key_press = human_agent.key_press

    for episode in range(1):
        joint_observation = env.reset()
        done = False

        for agent in agents:
            agent.reset()

        for timestep in range(1000):
            env.render()

            if done:
                print(f"episode done after {timestep} timestep(s)")
                break

            previous_joint_observation = joint_observation
            joint_action = [agent.choose_action(previous_observation, action_space) for agent, previous_observation, action_space in zip(agents, previous_joint_observation, env.action_space)]
            joint_observation, joint_reward, done, info = env.step(joint_action)

            print(f"{timestep}: {previous_joint_observation} {joint_action} -> {joint_observation} {joint_reward} {done} {info}")
        else:
            if save_video:
                env.stats_recorder.done = True  # need to manually tell the monitor that the episode is over (not sure why)

    env.close()


if __name__ == '__main__':
    run_pelican_crossing(save_video=False)
