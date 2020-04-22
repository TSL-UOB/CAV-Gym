from cavgym import utilities
from cavgym.actors import DynamicActorConstants

car_constants = DynamicActorConstants(
    length=50.0,
    width=20.0,
    wheelbase=45.0,
    min_velocity=0.0,
    max_velocity=200.0,
    normal_acceleration=20.0,
    normal_deceleration=-40.0,
    hard_acceleration=60.0,
    hard_deceleration=-80.0,
    normal_left_turn=utilities.DEG2RAD * 15.0,
    normal_right_turn=utilities.DEG2RAD * -15.0,
    hard_left_turn=utilities.DEG2RAD * 45.0,
    hard_right_turn=utilities.DEG2RAD * -45.0
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
    normal_left_turn=utilities.DEG2RAD * 30.0,
    normal_right_turn=utilities.DEG2RAD * -30.0,
    hard_left_turn=utilities.DEG2RAD * 90.0,
    hard_right_turn=utilities.DEG2RAD * -90.0
)

bus_constants = DynamicActorConstants(
    length=100.0,
    width=25.0,
    wheelbase=90.0,
    min_velocity=0.0,
    max_velocity=150.0,
    normal_acceleration=15.0,
    normal_deceleration=-30.0,
    hard_acceleration=45.0,
    hard_deceleration=-60.0,
    normal_left_turn=utilities.DEG2RAD * 15.0,
    normal_right_turn=utilities.DEG2RAD * -15.0,
    hard_left_turn=utilities.DEG2RAD * 45.0,
    hard_right_turn=utilities.DEG2RAD * -45.0
)

bicycle_constants = DynamicActorConstants(
    length=30.0,
    width=10.0,
    wheelbase=18.0,
    min_velocity=0.0,
    max_velocity=100.0,
    normal_acceleration=10.0,
    normal_deceleration=-20.0,
    hard_acceleration=30.0,
    hard_deceleration=-40.0,
    normal_left_turn=utilities.DEG2RAD * 30.0,
    normal_right_turn=utilities.DEG2RAD * -30.0,
    hard_left_turn=utilities.DEG2RAD * 90.0,
    hard_right_turn=utilities.DEG2RAD * -90.0
)
