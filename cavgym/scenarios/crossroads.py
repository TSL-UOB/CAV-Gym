from cavgym import utilities
from cavgym.actors import DynamicActorState, Car
from cavgym.environment import CAVEnvConstants, RoadMap, CAVEnv
from cavgym.assets import RoadConstants, Road
from cavgym.scenarios import car_constants


road_map = RoadMap(
    major_road=Road(
        constants=RoadConstants(
            length=1600.0,
            num_outbound_lanes=1,
            num_inbound_lanes=1,
            lane_width=40.0,
            position=utilities.Point(0.0, 0.0),
            orientation=utilities.DEG2RAD * 0.0
        )
    )
)

env_constants = CAVEnvConstants(
    viewer_width=int(road_map.major_road.constants.length),
    viewer_height=int(road_map.major_road.constants.lane_width * (road_map.major_road.constants.num_outbound_lanes + road_map.major_road.constants.num_inbound_lanes + 2)),
    time_resolution=1.0 / 60.0,
    road_map=road_map
)

actors = [
    Car(
        init_state=DynamicActorState(
            position=road_map.major_road.outbound.lane_spawns[0],
            velocity=100.0,
            orientation=road_map.major_road.outbound_orientation,
            acceleration=0.0,
            angular_velocity=0.0
        ),
        constants=car_constants
    ),
    Car(
        init_state=DynamicActorState(
            position=road_map.major_road.inbound.lane_spawns[0],
            velocity=100.0,
            orientation=road_map.major_road.inbound_orientation,
            acceleration=0.0,
            angular_velocity=0.0
        ),
        constants=car_constants
    )
]


class CrossroadsEnv(CAVEnv):
    def __init__(self):
        super().__init__(actors=actors, constants=env_constants)
