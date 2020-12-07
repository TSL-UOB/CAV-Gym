from library import geometry
from library.bodies import DynamicBodyState, Bus, Car, Bicycle
from library.environment import CAVEnvConstants, RoadMap, CAVEnv
from library.assets import Road, RoadConstants, BusStop, BusStopConstants
from examples.constants import car_constants, bicycle_constants, bus_constants, M2PX

road_map = RoadMap(
    major_road=Road(
        constants=RoadConstants(
            length=M2PX * 99,
            num_outbound_lanes=3,
            num_inbound_lanes=0,
            lane_width=M2PX * 3.65,
            position=geometry.Point(0.0, 0.0),
            orientation=0.0
        )
    )
)

road_map.major_road.outbound.set_bus_stop(
    BusStop(
        BusStopConstants(
            road_direction=road_map.major_road.outbound,
            x_position=M2PX * 99 * 0.75,
            length=bus_constants.length * 1.25
        )
    )
)

env_constants = CAVEnvConstants(
    viewer_width=road_map.major_road.constants.length,
    viewer_height=road_map.major_road.width + ((M2PX * 3) * 2),
    road_map=road_map
)

bodies = [
    Car(
        init_state=DynamicBodyState(
            position=geometry.Point(200, 0).rotate(road_map.major_road.outbound.orientation).translate(road_map.major_road.outbound.lanes[0].spawn),
            velocity=car_constants.max_velocity / 2.0,
            orientation=road_map.major_road.outbound.orientation
        ),
        constants=car_constants
    ),
    Bus(
        init_state=DynamicBodyState(
            position=geometry.Point(400, 0).rotate(road_map.major_road.outbound.orientation).translate(road_map.major_road.outbound.lanes[0].spawn),
            velocity=car_constants.max_velocity / 2.0,
            orientation=road_map.major_road.outbound.orientation
        ),
        constants=bus_constants
    ),
    Car(
        init_state=DynamicBodyState(
            position=road_map.major_road.outbound.lanes[0].spawn,
            velocity=car_constants.max_velocity / 2.0,
            orientation=road_map.major_road.outbound.orientation
        ),
        constants=car_constants
    ),
    Car(
        init_state=DynamicBodyState(
            position=road_map.major_road.outbound.lanes[1].spawn,
            velocity=car_constants.max_velocity / 2.0,
            orientation=road_map.major_road.outbound.orientation
        ),
        constants=car_constants
    ),
    Bicycle(
        init_state=DynamicBodyState(
            position=road_map.major_road.outbound.lanes[2].spawn,
            velocity=bicycle_constants.max_velocity / 2.0,
            orientation=road_map.major_road.outbound.orientation
        ),
        constants=bicycle_constants
    )
]


class BusStopEnv(CAVEnv):
    def __init__(self, **kwargs):
        super().__init__(bodies=bodies, constants=env_constants, **kwargs)
