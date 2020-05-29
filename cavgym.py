import argparse
import logging
import sys
import timeit

import gym
import numpy as np
from gym import wrappers
from gym.utils import seeding

from cavgym.agents import RandomTrafficLightAgent, RandomVehicleAgent, KeyboardAgent, RandomConstrainedPedestrianAgent
from cavgym import mods, Scenario
from cavgym.actors import DynamicActor, TrafficLight, PelicanCrossing, Pedestrian
from cavgym.environment import RenderMode

root = logging.getLogger()
root.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.addFilter(lambda record: record.levelno > logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
for handler in [stdout_handler, stderr_handler]:
    handler.setFormatter(formatter)
    root.addHandler(handler)

logger = logging.getLogger(__name__)  # Add this line to any file that uses logging


def parse_arguments():
    def non_negative_int(value):
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError(f"invalid non-negative int value: {value}")
        return ivalue

    def positive_int(value):
        ivalue = int(value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError(f"invalid positive int value: {value}")
        return ivalue

    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=Scenario, choices=[value for value in Scenario], help="choose scenario to run")
    parser.add_argument("-d", "--debug", help="print debug information", action="store_true")
    parser.add_argument("-e", "--episodes", type=positive_int, default=1, metavar="N", help="number of episodes (default: %(default)s)")
    parser.add_argument("-s", "--seed", type=non_negative_int, metavar="N", help="enable fixed random seed")
    parser.add_argument("-t", "--timesteps", type=positive_int, default=1000, metavar="N", help="max number of timesteps per episode (default: %(default)s)")
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1")

    parser.set_defaults(keyboard_agent=False, record=None)

    subparsers = parser.add_subparsers(dest="mode", required=True, help="choose mode to run scenario")

    subparsers.add_parser("headless")  # headless mode has no additional options

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("-k", "--keyboard-agent", help="enable keyboard-controlled agent", action="store_true")
    render_parser.add_argument("-r", "--record", metavar="DIR", help="save video of run to directory")

    args = parser.parse_args()
    return args.scenario, args.episodes, args.timesteps, True if args.mode == "render" else False, args.keyboard_agent, args.record, args.debug, args.seed


def run(scenario, episodes=1, max_timesteps=1000, render=True, keyboard_agent=None, record_dir=None, debug=False, seed=None):
    if debug:
        root.setLevel(logging.DEBUG)
    mode = RenderMode.VIDEO if render and record_dir is not None else RenderMode.SCREEN if render else RenderMode.NONE
    np_random, np_seed = seeding.np_random(seed)
    logger.info(f"seed={np_seed}")
    if scenario is Scenario.PELICAN_CROSSING:
        env = gym.make('PelicanCrossing-v0', mode=mode, np_random=np_random)
    elif scenario is Scenario.BUS_STOP:
        env = gym.make('BusStop-v0', mode=mode, np_random=np_random)
    elif scenario is Scenario.CROSSROADS:
        env = gym.make('Crossroads-v0', mode=mode, np_random=np_random)
    elif scenario is Scenario.PEDESTRIANS:
        env = gym.make('Pedestrians-v0', num_pedestrians=3, mode=mode, np_random=np_random)
    else:
        raise NotImplementedError
    agent = keyboard_agent if keyboard_agent is not None else RandomVehicleAgent(np_random=np_random)
    agents = [agent]
    for actor in env.actors[1:]:
        if isinstance(actor, DynamicActor):
            if isinstance(actor, Pedestrian):
                agents.append(RandomConstrainedPedestrianAgent(frequency=env.frequency, np_random=np_random))
            else:
                agents.append(RandomVehicleAgent(np_random=np_random))
        elif isinstance(actor, TrafficLight) or isinstance(actor, PelicanCrossing):
            agents.append(RandomTrafficLightAgent(np_random=np_random))
    run_simulation(env, agents, episodes=episodes, max_timesteps=max_timesteps, render=render, keyboard_agent=keyboard_agent, record_dir=record_dir)


def run_simulation(env, agents, episodes=1, max_timesteps=1000, render=True, keyboard_agent=None, record_dir=None):
    assert len(env.actors) == len(agents), "each actor must be assigned an agent and vice versa"

    logger.info(f"agents={[(agent.__class__.__name__, actor.__class__.__name__) for agent, actor in zip(agents, env.actors)]}")

    if keyboard_agent is not None or record_dir is not None:
        assert render, "keyboard agents and recordings only work in render mode"

    if record_dir is not None:
        env = wrappers.Monitor(env, record_dir, video_callable=lambda episode_id: True, force=True)  # save all episodes instead of default behaviour (episodes 1, 8, 27, 64, ...)
        env.stats_recorder = mods.make_joint_stats_recorder(env, len(agents))  # workaround to avoid bugs due to existence of joint rewards

    if render and keyboard_agent is not None:
        env.render()  # must render before key_press can be assigned
        env.unwrapped.viewer.window.on_key_press = keyboard_agent.key_press

    def measure_time(start_time, timesteps):
        time_ms = (timeit.default_timer() - start_time) * 1000
        real_time_ratio = (timesteps * env.time_resolution * 1000) / time_ms
        return time_ms, real_time_ratio

    def pretty_float(float_value, decimal_places=2):
        return f"{round(float_value, decimal_places):g}"

    def pretty_float_list(float_list, **kwargs):
        content = ", ".join([pretty_float(float_value, **kwargs) for float_value in float_list])
        return f"[{content}]"

    total_timesteps = 0
    run_start_time = timeit.default_timer()
    for previous_episode in range(episodes):  # initially previous_episode=0
        episode_start_time = timeit.default_timer()
        episode = previous_episode + 1

        joint_observation = env.reset()
        logger.debug(f"init: observation={joint_observation}")

        for agent in agents:
            agent.reset()

        for previous_timestep in range(max_timesteps):  # initially previous_timestep=0
            timestep = previous_timestep + 1

            if render:
                env.render()

            joint_action = [agent.choose_action(previous_observation, action_space) for agent, previous_observation, action_space in zip(agents, joint_observation, env.action_space)]
            joint_observation, joint_reward, done, info = env.step(joint_action)

            logger.debug(f"timestep={timestep}: action={joint_action} observation={joint_observation} reward={joint_reward} done={done} info={info}")

            if done:
                total_timesteps += timestep
                episode_time, episode_ratio = measure_time(episode_start_time, timestep)
                logging.info(f"episode={episode}: terminated after {timestep} timestep(s) taking {pretty_float(episode_time)} ms ({pretty_float(episode_ratio)}:1 real-time)")
                logger.info(f"liveness={pretty_float_list(env.episode_liveness / timestep)}")
                break
        else:
            total_timesteps += timestep
            episode_time, episode_ratio = measure_time(episode_start_time, timestep)
            logger.info(f"episode={episode}: completed after {timestep} timestep(s) taking {pretty_float(episode_time)} ms ({pretty_float(episode_ratio)}:1 real-time)")
            logger.info(f"liveness={pretty_float_list(env.episode_liveness / timestep)}")
            if record_dir is not None:
                env.stats_recorder.done = True  # need to manually tell the monitor that the episode is over (not sure why)
    else:
        run_time, run_ratio = measure_time(run_start_time, total_timesteps)
        logger.info(f"completed after {episode} episode(s) totalling {total_timesteps} timestep(s) taking {pretty_float(run_time)} ms ({pretty_float(run_ratio)}:1 real-time)")
        logger.info(f"liveness={pretty_float_list(env.run_liveness / total_timesteps)}")

    env.close()


if __name__ == '__main__':
    arg_scenario, arg_episodes, arg_timesteps, arg_render, arg_keyboard_agent, arg_record_dir, arg_debug, arg_seed = parse_arguments()
    run(arg_scenario, episodes=arg_episodes, max_timesteps=arg_timesteps, render=arg_render, keyboard_agent=KeyboardAgent() if arg_keyboard_agent else None, record_dir=arg_record_dir, debug=arg_debug, seed=arg_seed)
    # cProfile.run("run(Scenario.PEDESTRIANS, episodes=60, max_timesteps=3600, render=False, keyboard_agent=None, record_dir=None, debug=False, seed=0)", sort="tottime")  # reminder: there is a significant performance hit when using cProfile
