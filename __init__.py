from gym import wrappers

from agents import RandomVehicleAgent, HumanVehicleAgent, RandomTrafficLightAgent
from environment import RoadEnv, Vehicle, VehicleState, Point, ThrottleAction, TurnAction, Observation, DEG2RAD, \
    TrafficLight, TrafficLightConfig, TrafficLightState
from utilities import make_joint_stats_recorder

SEED = None


actors = [
    Vehicle(VehicleState(Point(0, 70), 0, 100, 0, 0)),
    Vehicle(VehicleState(Point(1600, 30), DEG2RAD * 180, 100, 0, 0)),
    TrafficLight(TrafficLightState.GREEN, TrafficLightConfig(Point(800, 90)))
]
human_agent = HumanVehicleAgent()
agents = [human_agent, RandomVehicleAgent(seed=SEED), RandomTrafficLightAgent(seed=SEED)]

save_video = True
env = RoadEnv(actors, seed=SEED)
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

        print(f"{timestep}: {previous_joint_observation} {joint_action} -> {joint_observation} {joint_reward} {done} {info}")
    else:
        if save_video:
            env.stats_recorder.done = True  # Need to manually tell the monitor that the episode is over (not sure why)

env.close()
