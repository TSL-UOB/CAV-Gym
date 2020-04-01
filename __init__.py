from gym import wrappers

from agents import RandomAgent, HumanAgent
from environment import RoadEnv, Vehicle, VehicleState, Point, ThrottleAction, TurnAction, Observation, RoadEnvConfig, \
    DEG2RAD
from utilities import make_joint_stats_recorder

SEED = None


vehicles = [Vehicle(VehicleState(Point(0, 70), 100, 0, 0, 0)), Vehicle(VehicleState(Point(1600, 30), 100, 0, DEG2RAD * 180, 0))]
human_agent = HumanAgent()
agents = [human_agent, RandomAgent(seed=SEED)]

save_video = True
env = RoadEnv(vehicles, seed=SEED)
if save_video:
    env = wrappers.Monitor(env, "/Users/kevin/Downloads/gym_gif/", video_callable=lambda episode_id: True, force=True)  # save all episodes instead of default behaviour (episodes 1, 8, 27, 64, ...)
    env.stats_recorder = make_joint_stats_recorder(env, len(agents))  # workaround to avoid bugs due to existence of joint rewards
env.render()
env.unwrapped.viewer.window.on_key_press = human_agent.key_press

for episode in range(1):
    done = False
    joint_observation = env.reset()
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

        pretty_previous_joint_observation = [Observation(observation_id) for observation_id in previous_joint_observation]
        pretty_joint_action = [(ThrottleAction(throttle_action_id), TurnAction(turn_action_id)) for throttle_action_id, turn_action_id in joint_action]
        pretty_joint_observation = [Observation(observation_id) for observation_id in joint_observation]
        print(f"{timestep}: {pretty_previous_joint_observation} {pretty_joint_action} -> {pretty_joint_observation} {joint_reward} {done} {info}")
    else:
        if save_video:
            env.stats_recorder.done = True  # Need to manually tell the monitor that the episode is over (not sure why)

env.close()
