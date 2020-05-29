from cavgym import geometry
from cavgym.actors import DynamicActorConstants

M2PX = 16  # pixels per metre

car_constants = DynamicActorConstants(
    length=M2PX * 4.5,
    width=M2PX * 1.75,
    wheelbase=M2PX * 4.05,
    min_velocity=0,
    max_velocity=M2PX * 9,
    acceleration_rate=M2PX * 9,
    deceleration_rate=M2PX * -9,
    left_turn_rate=geometry.DEG2RAD * 35,
    right_turn_rate=geometry.DEG2RAD * -35,
    target_slow_velocity=M2PX * 4.5,
    target_fast_velocity=M2PX * 9
)

pedestrian_constants = DynamicActorConstants(
    length=M2PX * 0.65625,
    width=M2PX * 0.875,
    wheelbase=M2PX * 0.328125,
    min_velocity=0,
    max_velocity=M2PX * 1.4,
    acceleration_rate=M2PX * 1.4,
    deceleration_rate=M2PX * -1.4,
    left_turn_rate=geometry.DEG2RAD * 90,
    right_turn_rate=geometry.DEG2RAD * -90,
    target_slow_velocity=M2PX * 0.7,
    target_fast_velocity=M2PX * 1.4
)

bus_constants = DynamicActorConstants(
    length=M2PX * 12,
    width=M2PX * 2.55,
    wheelbase=M2PX * 16.875,
    min_velocity=0,
    max_velocity=M2PX * 6.75,
    acceleration_rate=M2PX * 6.75,
    deceleration_rate=-M2PX * 6.75,
    left_turn_rate=geometry.DEG2RAD * 30,
    right_turn_rate=geometry.DEG2RAD * -30,
    target_slow_velocity=M2PX * 3.375,
    target_fast_velocity=M2PX * 6.75
)

bicycle_constants = DynamicActorConstants(
    length=M2PX * 2.25,
    width=M2PX * 0.875,
    wheelbase=M2PX * 2.025,
    min_velocity=0,
    max_velocity=M2PX * 4.5,
    acceleration_rate=M2PX * 4.5,
    deceleration_rate=-M2PX * 4.5,
    left_turn_rate=geometry.DEG2RAD * 90,
    right_turn_rate=geometry.DEG2RAD * -90,
    target_slow_velocity=M2PX * 2.25,
    target_fast_velocity=M2PX * 4.5
)
