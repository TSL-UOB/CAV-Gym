from library import geometry
from library.actors import DynamicActorConstants

M2PX = 16  # pixels per metre

car_constants = DynamicActorConstants(
    length=M2PX * 4.5,
    width=M2PX * 1.75,
    wheelbase=M2PX * 3,
    track=M2PX * 1.75,
    min_velocity=0,
    max_velocity=M2PX * 9,
    throttle_up_rate=M2PX * 9,
    throttle_down_rate=M2PX * -9,
    steer_left_angle=geometry.DEG2RAD * 35,
    steer_right_angle=geometry.DEG2RAD * -35,
    target_slow_velocity=M2PX * 4.5,
    target_fast_velocity=M2PX * 9
)

pedestrian_constants = DynamicActorConstants(
    length=M2PX * 0.65625,
    width=M2PX * 0.875,
    wheelbase=M2PX * 0.328125,
    track=M2PX * 0.875,
    min_velocity=0,
    max_velocity=M2PX * 1.4,
    throttle_up_rate=M2PX * 1.4,
    throttle_down_rate=M2PX * -1.4,
    steer_left_angle=geometry.DEG2RAD * 90,
    steer_right_angle=geometry.DEG2RAD * -90,
    target_slow_velocity=M2PX * 0.7,
    target_fast_velocity=M2PX * 1.4
)

bus_constants = DynamicActorConstants(
    length=M2PX * 12,
    width=M2PX * 2.55,
    wheelbase=M2PX * 16.875,
    track=M2PX * 2.55,
    min_velocity=0,
    max_velocity=M2PX * 6.75,
    throttle_up_rate=M2PX * 6.75,
    throttle_down_rate=-M2PX * 6.75,
    steer_left_angle=geometry.DEG2RAD * 30,
    steer_right_angle=geometry.DEG2RAD * -30,
    target_slow_velocity=M2PX * 3.375,
    target_fast_velocity=M2PX * 6.75
)

bicycle_constants = DynamicActorConstants(
    length=M2PX * 2.25,
    width=M2PX * 0.875,
    wheelbase=M2PX * 2.025,
    track=M2PX * 0.875,
    min_velocity=0,
    max_velocity=M2PX * 4.5,
    throttle_up_rate=M2PX * 4.5,
    throttle_down_rate=-M2PX * 4.5,
    steer_left_angle=geometry.DEG2RAD * 90,
    steer_right_angle=geometry.DEG2RAD * -90,
    target_slow_velocity=M2PX * 2.25,
    target_fast_velocity=M2PX * 4.5
)
