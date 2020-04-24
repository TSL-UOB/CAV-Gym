from cavgym import utilities
from cavgym.actors import DynamicActorState, Bus, Car, Bicycle
from cavgym.environment import CAVEnvConstants, RoadMap, CAVEnv
from cavgym.assets import Road, RoadConstants
from cavgym.scenarios import car_constants, bicycle_constants, bus_constants

road_layout = RoadMap(
    major_road=Road(
        constants=RoadConstants(
            length=1600.0,
            num_outbound_lanes=3,
            num_inbound_lanes=0,
            lane_width=40.0,
            position=utilities.Point(0.0, 0.0),
            orientation=0.0
        )
    )
)

env_constants = CAVEnvConstants(
    viewer_width=int(road_layout.major_road.constants.length),
    viewer_height=int(road_layout.major_road.constants.lane_width * (road_layout.major_road.constants.num_outbound_lanes + road_layout.major_road.constants.num_inbound_lanes + 2)),
    time_resolution=1.0 / 60.0,
    road_layout=road_layout
)

actors = [
    Bus(
        init_state=DynamicActorState(
            position=utilities.Point(0.0, road_layout.major_road.outbound_lanes_bounds[2][1] + (road_layout.major_road.constants.lane_width / 2.0)),
            velocity=100.0,
            orientation=0.0,
            acceleration=0.0,
            angular_velocity=0.0
        ),
        constants=bus_constants
    ),
    Car(
        init_state=DynamicActorState(
            position=utilities.Point(0.0, road_layout.major_road.outbound_lanes_bounds[1][1] + (road_layout.major_road.constants.lane_width / 2.0)),
            velocity=100.0,
            orientation=0.0,
            acceleration=0.0,
            angular_velocity=0.0
        ),
        constants=car_constants
    ),
    Bicycle(
        init_state=DynamicActorState(
            position=utilities.Point(0.0, road_layout.major_road.outbound_lanes_bounds[0][1] + (road_layout.major_road.constants.lane_width / 2.0)),
            velocity=100.0,
            orientation=0.0,
            acceleration=0.0,
            angular_velocity=0.0
        ),
        constants=bicycle_constants
    )
]


class BusStopEnv(CAVEnv):
    def __init__(self):
        super().__init__(actors=actors, constants=env_constants)
