from pathlib import Path

from gym import wrappers

from agents import RandomTrafficLightAgent, HumanDynamicActorAgent, RandomDynamicActorAgent
from environment import RoadEnv, Point, DEG2RAD, TrafficLight, TrafficLightState, Pedestrian, DynamicActorConstants, \
    DynamicActorState, Vehicle, StaticActorConstants, RoadEnvConstants
from mods import make_joint_stats_recorder

env_constants = RoadEnvConstants(
    viewer_width=1600,
    viewer_height=160,
    time_resolution=1.0 / 60.0
)

vehicle_constants = DynamicActorConstants(
    length=50.0,
    width=20.0,
    wheelbase=45.0,
    min_velocity=0.0,
    max_velocity=200.0,
    normal_acceleration=20.0,
    normal_deceleration=-40.0,
    hard_acceleration=60.0,
    hard_deceleration=-80.0,
    normal_left_turn=DEG2RAD * 15.0,
    normal_right_turn=DEG2RAD * -15.0,
    hard_left_turn=DEG2RAD * 45.0,
    hard_right_turn=DEG2RAD * -45.0
)

pedestrian_constants = DynamicActorConstants(
    length=10.0,
    width=15.0,
    wheelbase=9.0,
    min_velocity=0.0,
    max_velocity=40.0,
    normal_acceleration=100.0,
    normal_deceleration=-100.0,
    hard_acceleration=200.0,
    hard_deceleration=-200.0,
    normal_left_turn=DEG2RAD * 30.0,
    normal_right_turn=DEG2RAD * -30.0,
    hard_left_turn=DEG2RAD * 90.0,
    hard_right_turn=DEG2RAD * -90.0
)

actors = [
    Vehicle(
        init_state=DynamicActorState(
            position=Point(0.0, 100.0),
            velocity=100.0,
            orientation=0.0,
            acceleration=0.0,
            angular_velocity=0.0
        ),
        constants=vehicle_constants
    ),
    Vehicle(
        init_state=DynamicActorState(
            position=Point(1600.0, 60.0),
            velocity=100.0,
            orientation=DEG2RAD * 180.0,
            acceleration=0.0,
            angular_velocity=0.0
        ),
        constants=vehicle_constants
    ),
    TrafficLight(
        init_state=TrafficLightState.GREEN,
        constants=StaticActorConstants(
            height=20.0,
            width=10.0,
            position=Point(780.0, 130.0),
            orientation=0.0
        )
    ),
    TrafficLight(
        init_state=TrafficLightState.GREEN,
        constants=StaticActorConstants(
            height=20.0,
            width=10.0,
            position=Point(820.0, 30.0),
            orientation=0.0
        )
    ),
    Pedestrian(
        init_state=DynamicActorState(
            position=Point(800.0, 30.0),
            velocity=0.0,
            orientation=DEG2RAD * 90.0,
            acceleration=0.0,
            angular_velocity=0.0
        ),
        constants=pedestrian_constants
    )
]

human_agent = HumanDynamicActorAgent()
agents = [human_agent, RandomDynamicActorAgent(), RandomTrafficLightAgent(), RandomTrafficLightAgent(), RandomDynamicActorAgent()]

save_video = True
env = RoadEnv(actors=actors, constants=env_constants)
if save_video:
    env = wrappers.Monitor(env, f"{Path.home()}/CAV-Gym/Videos/", video_callable=lambda episode_id: True, force=True)  # save all episodes instead of default behaviour (episodes 1, 8, 27, 64, ...)
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
