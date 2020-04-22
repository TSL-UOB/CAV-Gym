from cavgym import utilities
from cavgym.actors import DynamicActorState, Car
from cavgym.environment import CAVEnvConstants, RoadMap, CAVEnv
from cavgym.assets import RoadConstants, Road
from cavgym.scenarios import car_constants


road_layout = RoadMap(
    main_road=Road(
        constants=RoadConstants(
            length=1600.0,
            num_outbound_lanes=1,
            num_inbound_lanes=1,
            lane_width=40.0,
            position=utilities.Point(0.0, 0.0),
            orientation=0.0
        )
    )
)

env_constants = CAVEnvConstants(
    viewer_width=int(road_layout.main_road.constants.length),
    viewer_height=int(road_layout.main_road.constants.lane_width * (road_layout.main_road.constants.num_outbound_lanes + road_layout.main_road.constants.num_inbound_lanes + 2)),
    time_resolution=1.0 / 60.0,
    road_layout=road_layout
)

actors = [
    Car(
        init_state=DynamicActorState(
            position=utilities.Point(0.0, road_layout.main_road.outbound_lanes_bounds[0][1] + (road_layout.main_road.constants.lane_width / 2.0)),
            velocity=100.0,
            orientation=0.0,
            acceleration=0.0,
            angular_velocity=0.0
        ),
        constants=car_constants
    ),
    Car(
        init_state=DynamicActorState(
            position=utilities.Point(env_constants.viewer_width, road_layout.main_road.inbound_lanes_bounds[-1][3] - (road_layout.main_road.constants.lane_width / 2.0)),
            velocity=100.0,
            orientation=utilities.DEG2RAD * 180.0,
            acceleration=0.0,
            angular_velocity=0.0
        ),
        constants=car_constants
    )
]


class CrossroadsEnv(CAVEnv):
    def __init__(self):
        super().__init__(actors=actors, constants=env_constants)