import argparse
from enum import Enum

import gym
from gym import wrappers

from agents import RandomDynamicActorAgent, RandomTrafficLightAgent, HumanDynamicActorAgent
from cavgym import mods


class Scenario(Enum):
    BUS_STOP = "bus-stop"
    CROSSROADS = "crossroads"
    PEDESTRIANS = "pedestrians"
    PELICAN_CROSSING = "pelican-crossing"

    def __str__(self):
        return self.value


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=Scenario, choices=[value for value in Scenario], nargs="?", default=Scenario.PELICAN_CROSSING, help="choose scenario to run (default: %(default)s)")
    parser.add_argument("-d", "--debug", help="print debug information", action="store_true")
    mutually_exclusive = parser.add_mutually_exclusive_group()
    mutually_exclusive.add_argument("-n", "--no-render", help="run without rendering", action="store_true")
    mutually_exclusive.add_argument("-r", "--record", metavar="DIR", help="record video of run to directory %(metavar)s")
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1")

    args = parser.parse_args()
    return args.scenario, not args.no_render, args.record, args.debug


def run(scenario, render=True, record_dir=None, debug=False):
    human_agent = HumanDynamicActorAgent() if render else None
    agent = human_agent if human_agent is not None else RandomDynamicActorAgent()
    if scenario is Scenario.PELICAN_CROSSING:
        env = gym.make('PelicanCrossing-v0')
        agents = [agent, RandomDynamicActorAgent(), RandomTrafficLightAgent(), RandomDynamicActorAgent(), RandomDynamicActorAgent()]
        run_simulation(env, agents, render=render, human_agent=human_agent, record_dir=record_dir, debug=debug)
    elif scenario is Scenario.BUS_STOP:
        env = gym.make('BusStop-v0')
        agents = [agent, RandomDynamicActorAgent(), RandomDynamicActorAgent(), RandomDynamicActorAgent(), RandomDynamicActorAgent()]
        run_simulation(env, agents, render=render, human_agent=human_agent, record_dir=record_dir, debug=debug)
    elif scenario is Scenario.CROSSROADS:
        env = gym.make('Crossroads-v0')
        agents = [agent, RandomDynamicActorAgent(), RandomDynamicActorAgent()]
        run_simulation(env, agents, render=render, human_agent=human_agent, record_dir=record_dir, debug=debug)
    elif scenario is Scenario.PEDESTRIANS:
        env = gym.make('Pedestrians-v0')
        agents = [agent, RandomDynamicActorAgent(), RandomDynamicActorAgent(), RandomDynamicActorAgent()]
        run_simulation(env, agents, render=render, human_agent=human_agent, record_dir=record_dir, debug=debug)
    else:
        print(f"{scenario} scenario is not yet implemented")


def run_simulation(env, agents, render=True, human_agent=None, record_dir=None, debug=False):
    assert len(env.actors) == len(agents), "each actor must be assigned an agent and vice versa"

    if human_agent is not None or record_dir is not None:
        assert render, "human agents and recordings only work in render mode"

    if record_dir is not None:
        env = wrappers.Monitor(env, record_dir, video_callable=lambda episode_id: True, force=True)  # save all episodes instead of default behaviour (episodes 1, 8, 27, 64, ...)
        env.stats_recorder = mods.make_joint_stats_recorder(env, len(agents))  # workaround to avoid bugs due to existence of joint rewards

    if render and human_agent is not None:
        env.render()  # must render before key_press can be assigned
        env.unwrapped.viewer.window.on_key_press = human_agent.key_press

    for episode in range(1):
        joint_observation = env.reset()
        done = False

        for agent in agents:
            agent.reset()

        for timestep in range(1000):
            if render:
                env.render()

            if done:
                print(f"episode {episode+1} terminated after {timestep} timestep(s)")
                break

            previous_joint_observation = joint_observation
            joint_action = [agent.choose_action(previous_observation, action_space) for agent, previous_observation, action_space in zip(agents, previous_joint_observation, env.action_space)]
            joint_observation, joint_reward, done, info = env.step(joint_action)

            if debug:
                print(f"{timestep}: {previous_joint_observation} {joint_action} -> {joint_observation} {joint_reward} {done} {info}")
        else:
            print(f"episode {episode+1} completed")
            if record_dir is not None:
                env.stats_recorder.done = True  # need to manually tell the monitor that the episode is over (not sure why)

    env.close()


if __name__ == '__main__':
    arg_scenario, arg_render, arg_record_dir, arg_debug = parse_arguments()
    run(arg_scenario, render=arg_render, record_dir=arg_record_dir, debug=arg_debug)
