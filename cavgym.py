import argparse

import gym
from gym import wrappers
from gym.utils import seeding

from cavgym.agents import RandomTrafficLightAgent, RandomVehicleAgent, RandomPedestrianAgent, KeyboardAgent
from cavgym import mods, Scenario
from cavgym.actors import DynamicActor, TrafficLight, PelicanCrossing, Pedestrian


def parse_arguments():
    def non_negative_int(value):
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError(f"invalid non-negative int value: {value}")
        return ivalue

    def positive_int(value):
        ivalue = int(value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError(f"invalid poisitve int value: {value}")
        return ivalue

    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=Scenario, choices=[value for value in Scenario], nargs="?", default=Scenario.PELICAN_CROSSING, help="choose scenario to run (default: %(default)s)")
    parser.add_argument("-d", "--debug", help="print debug information", action="store_true")
    parser.add_argument("-e", "--episodes", type=positive_int, default=1, metavar="N", help="number of episodes (default: %(default)s)")
    parser.add_argument("-k", "--keyboard-agent", help="run with keyboard-controlled agent", action="store_true")
    mutually_exclusive = parser.add_mutually_exclusive_group()
    mutually_exclusive.add_argument("-n", "--no-render", help="run without rendering", action="store_true")
    mutually_exclusive.add_argument("-r", "--record", metavar="DIR", help="save video of run to directory")
    parser.add_argument("-s", "--seed", type=non_negative_int, metavar="N", help="enable fixed random seed")
    parser.add_argument("-t", "--timesteps", type=positive_int, default=1000, metavar="N", help="max number of timesteps per episode (default: %(default)s)")
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1")

    args = parser.parse_args()
    return args.scenario, args.episodes, args.timesteps, not args.no_render, args.keyboard_agent, args.record, args.debug, args.seed


def run(scenario, episodes=1, max_timesteps=1000, render=True, keyboard_agent=None, record_dir=None, debug=False, seed=None):
    np_random, _ = seeding.np_random(seed)
    if scenario is Scenario.PELICAN_CROSSING:
        env = gym.make('PelicanCrossing-v0', np_random=np_random)
    elif scenario is Scenario.BUS_STOP:
        env = gym.make('BusStop-v0', np_random=np_random)
    elif scenario is Scenario.CROSSROADS:
        env = gym.make('Crossroads-v0', np_random=np_random)
    elif scenario is Scenario.PEDESTRIANS:
        env = gym.make('Pedestrians-v0', np_random=np_random)
    else:
        raise NotImplementedError
    agent = keyboard_agent if keyboard_agent is not None else RandomVehicleAgent(np_random=np_random)
    agents = [agent]
    for actor in env.actors[1:]:
        if isinstance(actor, DynamicActor):
            if isinstance(actor, Pedestrian):
                agents.append(RandomPedestrianAgent(np_random=np_random))
            else:
                agents.append(RandomVehicleAgent(np_random=np_random))
        elif isinstance(actor, TrafficLight) or isinstance(actor, PelicanCrossing):
            agents.append(RandomTrafficLightAgent(np_random=np_random))
    run_simulation(env, agents, episodes=episodes, max_timesteps=max_timesteps, render=render, keyboard_agent=keyboard_agent, record_dir=record_dir, debug=debug)


def run_simulation(env, agents, episodes=1, max_timesteps=1000, render=True, keyboard_agent=None, record_dir=None, debug=False):
    assert len(env.actors) == len(agents), "each actor must be assigned an agent and vice versa"

    if keyboard_agent is not None or record_dir is not None:
        assert render, "keyboard agents and recordings only work in render mode"

    if record_dir is not None:
        env = wrappers.Monitor(env, record_dir, video_callable=lambda episode_id: True, force=True)  # save all episodes instead of default behaviour (episodes 1, 8, 27, 64, ...)
        env.stats_recorder = mods.make_joint_stats_recorder(env, len(agents))  # workaround to avoid bugs due to existence of joint rewards

    if render and keyboard_agent is not None:
        env.render()  # must render before key_press can be assigned
        env.unwrapped.viewer.window.on_key_press = keyboard_agent.key_press

    for episode in range(episodes):
        joint_observation = env.reset()
        done = False

        for agent in agents:
            agent.reset()

        for timestep in range(max_timesteps):
            if render:
                env.render()

            if done:
                print(f"episode {episode+1} terminated after {timestep} timestep(s)")
                break

            previous_joint_observation = joint_observation
            joint_action = [agent.choose_action(previous_observation, action_space) for agent, previous_observation, action_space in zip(agents, previous_joint_observation, env.action_space)]
            joint_observation, joint_reward, done, info = env.step(joint_action)

            if debug:
                print(f"{timestep+1}: {previous_joint_observation} {joint_action} -> {joint_observation} {joint_reward} {done} {info}")
        else:
            print(f"episode {episode+1} completed")
            if record_dir is not None:
                env.stats_recorder.done = True  # need to manually tell the monitor that the episode is over (not sure why)

    env.close()


if __name__ == '__main__':
    arg_scenario, arg_episodes, arg_timesteps, arg_render, arg_keyboard_agent, arg_record_dir, arg_debug, arg_seed = parse_arguments()
    run(arg_scenario, episodes=arg_episodes, max_timesteps=arg_timesteps, render=arg_render, keyboard_agent=KeyboardAgent() if arg_keyboard_agent else None, record_dir=arg_record_dir, debug=arg_debug, seed=arg_seed)
    # run(Scenario.PEDESTRIANS, episodes=10, max_timesteps=1200, render=True, keyboard_agent=None, record_dir=None, debug=False, seed=0)
