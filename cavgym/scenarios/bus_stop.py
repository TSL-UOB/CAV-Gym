from cavgym import utilities
from cavgym.actors import DynamicActorState, Bus, Car, Bicycle
from cavgym.environment import CAVEnvConstants, RoadMap, CAVEnv
from cavgym.assets import Road, RoadConstants, BusStop, BusStopConstants
from cavgym.scenarios import car_constants, bicycle_constants, bus_constants

road_map = RoadMap(
    major_road=Road(
        constants=RoadConstants(
            length=1600,
            num_outbound_lanes=3,
            num_inbound_lanes=0,
            lane_width=40,
            position=utilities.Point(0.0, 0.0),
            orientation=utilities.DEG2RAD * 0.0
        )
    )
)

road_map.major_road.outbound.set_bus_stop(
    BusStop(
        BusStopConstants(
            road_direction=road_map.major_road.outbound,
            x_position=1200
        )
    )
)

env_constants = CAVEnvConstants(
    viewer_width=road_map.major_road.constants.length,
    viewer_height=road_map.major_road.constants.lane_width * (road_map.major_road.constants.num_outbound_lanes + road_map.major_road.constants.num_inbound_lanes + 2),
    time_resolution=1.0 / 60.0,
    road_map=road_map
)

actors = [
    Bus(
        init_state=DynamicActorState(
            position=utilities.Point(400, 0).rotate(road_map.major_road.outbound_orientation).relative(road_map.major_road.outbound.lane_spawns[0]),
            velocity=100.0,
            orientation=road_map.major_road.outbound_orientation,
            acceleration=0,
            angular_velocity=0.0
        ),
        constants=bus_constants
    ),
    Car(
        init_state=DynamicActorState(
            position=utilities.Point(200, 0).rotate(road_map.major_road.outbound_orientation).relative(road_map.major_road.outbound.lane_spawns[0]),
            velocity=100.0,
            orientation=road_map.major_road.outbound_orientation,
            acceleration=0,
            angular_velocity=0.0
        ),
        constants=car_constants
    ),
    Car(
        init_state=DynamicActorState(
            position=road_map.major_road.outbound.lane_spawns[0],
            velocity=100.0,
            orientation=road_map.major_road.outbound_orientation,
            acceleration=0,
            angular_velocity=0.0
        ),
        constants=car_constants
    ),
    Car(
        init_state=DynamicActorState(
            position=road_map.major_road.outbound.lane_spawns[1],
            velocity=100.0,
            orientation=road_map.major_road.outbound_orientation,
            acceleration=0,
            angular_velocity=0.0
        ),
        constants=car_constants
    ),
    Bicycle(
        init_state=DynamicActorState(
            position=road_map.major_road.outbound.lane_spawns[2],
            velocity=100.0,
            orientation=road_map.major_road.outbound_orientation,
            acceleration=0,
            angular_velocity=0.0
        ),
        constants=bicycle_constants
    )
]


class BusStopEnv(CAVEnv):
    def __init__(self):
        super().__init__(actors=actors, constants=env_constants)
