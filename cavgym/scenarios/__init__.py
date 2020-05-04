from cavgym import geometry
from cavgym.actors import DynamicActorConstants

car_constants = DynamicActorConstants(
    length=50,
    width=20,
    wheelbase=45,
    view_distance=100,
    view_angle=geometry.DEG2RAD * 90.0,
    min_velocity=0.0,
    max_velocity=200.0,
    normal_acceleration=20,
    normal_deceleration=-40,
    hard_acceleration=60,
    hard_deceleration=-80,
    normal_left_turn=geometry.DEG2RAD * 15.0,
    normal_right_turn=geometry.DEG2RAD * -15.0,
    hard_left_turn=geometry.DEG2RAD * 45.0,
    hard_right_turn=geometry.DEG2RAD * -45.0
)

pedestrian_constants = DynamicActorConstants(
    length=10,
    width=15,
    wheelbase=9,
    view_distance=100,
    view_angle=geometry.DEG2RAD * 90.0,
    min_velocity=0.0,
    max_velocity=40.0,
    normal_acceleration=100,
    normal_deceleration=-100,
    hard_acceleration=200,
    hard_deceleration=-200,
    normal_left_turn=geometry.DEG2RAD * 30.0,
    normal_right_turn=geometry.DEG2RAD * -30.0,
    hard_left_turn=geometry.DEG2RAD * 90.0,
    hard_right_turn=geometry.DEG2RAD * -90.0
)

bus_constants = DynamicActorConstants(
    length=100,
    width=25,
    wheelbase=90,
    view_distance=100,
    view_angle=geometry.DEG2RAD * 90.0,
    min_velocity=0.0,
    max_velocity=150.0,
    normal_acceleration=15,
    normal_deceleration=-30,
    hard_acceleration=45,
    hard_deceleration=-60,
    normal_left_turn=geometry.DEG2RAD * 15.0,
    normal_right_turn=geometry.DEG2RAD * -15.0,
    hard_left_turn=geometry.DEG2RAD * 45.0,
    hard_right_turn=geometry.DEG2RAD * -45.0
)

bicycle_constants = DynamicActorConstants(
    length=30,
    width=10,
    wheelbase=18,
    view_distance=100,
    view_angle=geometry.DEG2RAD * 90.0,
    min_velocity=0.0,
    max_velocity=100.0,
    normal_acceleration=10,
    normal_deceleration=-20,
    hard_acceleration=30,
    hard_deceleration=-40,
    normal_left_turn=geometry.DEG2RAD * 30.0,
    normal_right_turn=geometry.DEG2RAD * -30.0,
    hard_left_turn=geometry.DEG2RAD * 90.0,
    hard_right_turn=geometry.DEG2RAD * -90.0
)
