from cavgym import geometry
from cavgym.actors import DynamicActorConstants

car_constants = DynamicActorConstants(
    length=50,
    width=20,
    wheelbase=45,
    min_velocity=0,
    max_velocity=200,
    acceleration_rate=60,
    deceleration_rate=-80,
    left_turn_rate=geometry.DEG2RAD * 45,
    right_turn_rate=geometry.DEG2RAD * -45,
    target_slow_velocity=50,
    target_fast_velocity=100
)

pedestrian_constants = DynamicActorConstants(
    length=10,
    width=15,
    wheelbase=9,
    min_velocity=0,
    max_velocity=40,
    acceleration_rate=200,
    deceleration_rate=-200,
    left_turn_rate=geometry.DEG2RAD * 90,
    right_turn_rate=geometry.DEG2RAD * -90,
    target_slow_velocity=20,
    target_fast_velocity=40
)

bus_constants = DynamicActorConstants(
    length=100,
    width=25,
    wheelbase=90,
    min_velocity=0,
    max_velocity=150,
    acceleration_rate=45,
    deceleration_rate=-60,
    left_turn_rate=geometry.DEG2RAD * 45,
    right_turn_rate=geometry.DEG2RAD * -45,
    target_slow_velocity=40,
    target_fast_velocity=80
)

bicycle_constants = DynamicActorConstants(
    length=30,
    width=10,
    wheelbase=18,
    min_velocity=0,
    max_velocity=100,
    acceleration_rate=30,
    deceleration_rate=-40,
    left_turn_rate=geometry.DEG2RAD * 90,
    right_turn_rate=geometry.DEG2RAD * -90,
    target_slow_velocity=30,
    target_fast_velocity=60
)
