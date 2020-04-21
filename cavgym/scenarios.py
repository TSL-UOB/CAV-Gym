from cavgym import utilities
from cavgym.environment import DynamicActorConstants, RoadEnvConstants, RoadEnv, Vehicle, DynamicActorState, TrafficLightState, Pedestrian, PelicanCrossing, RoadLayout, Road, RoadConstants, PelicanCrossingConstants

road_layout = RoadLayout(
    main_road=Road(
        constants=RoadConstants(
            length=1600.0,
            num_outbound_lanes=3,
            num_inbound_lanes=2,
            lane_width=40.0,
            position=utilities.Point(0.0, 0.0),
            orientation=0.0
        )
    )
)

env_constants = RoadEnvConstants(
    viewer_width=int(road_layout.main_road.constants.length),
    viewer_height=int(road_layout.main_road.constants.lane_width * (road_layout.main_road.constants.num_outbound_lanes + road_layout.main_road.constants.num_inbound_lanes + 2)),
    time_resolution=1.0 / 60.0,
    road_layout=road_layout
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


class PelicanCrossingEnv(RoadEnv):
    def __init__(self):
        super().__init__(
            actors=[
                Vehicle(
                    init_state=DynamicActorState(
                        position=utilities.Point(0.0, road_layout.main_road.outbound_lanes_bounds[0][1] + (road_layout.main_road.constants.lane_width / 2.0)),
                        velocity=100.0,
                        orientation=0.0,
                        acceleration=0.0,
                        angular_velocity=0.0
                    ),
                    constants=vehicle_constants
                ),
                Vehicle(
                    init_state=DynamicActorState(
                        position=utilities.Point(env_constants.viewer_width, road_layout.main_road.inbound_lanes_bounds[-1][3] - (road_layout.main_road.constants.lane_width / 2.0)),
                        velocity=100.0,
                        orientation=utilities.DEG2RAD * 180.0,
                        acceleration=0.0,
                        angular_velocity=0.0
                    ),
                    constants=vehicle_constants
                ),
                PelicanCrossing(
                    init_state=TrafficLightState.GREEN,
                    constants=PelicanCrossingConstants(
                        road=road_layout.main_road,
                        width=road_layout.main_road.constants.lane_width * 1.5,
                        x_position=road_layout.main_road.constants.length / 2.0
                    )
                ),
                Pedestrian(
                    init_state=DynamicActorState(
                        position=utilities.Point(env_constants.viewer_width / 2.0, -(env_constants.viewer_height / 2.0) + (road_layout.main_road.constants.lane_width * 0.75)),
                        velocity=0.0,
                        orientation=utilities.DEG2RAD * 90.0,
                        acceleration=0.0,
                        angular_velocity=0.0
                    ),
                    constants=pedestrian_constants
                )
            ],
            constants=env_constants
        )
