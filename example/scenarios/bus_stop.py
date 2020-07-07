from libcavgym import geometry
from libcavgym.actors import DynamicActorState, Bus, Car, Bicycle
from libcavgym.environment import CAVEnvConstants, RoadMap, CAVEnv
from libcavgym.assets import Road, RoadConstants, BusStop, BusStopConstants
from example.constants import car_constants, bicycle_constants, bus_constants, M2PX

road_map = RoadMap(
    major_road=Road(
        constants=RoadConstants(
            length=M2PX * 99,
            num_outbound_lanes=3,
            num_inbound_lanes=0,
            lane_width=M2PX * 3.65,
            position=geometry.Point(0.0, 0.0),
            orientation=geometry.DEG2RAD * 0.0
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

actors = [
    Car(
        init_state=DynamicActorState(
            position=geometry.Point(200, 0).rotate(road_map.major_road.outbound.orientation).translate(road_map.major_road.outbound.lanes[0].spawn),
            velocity=car_constants.target_fast_velocity,
            orientation=road_map.major_road.outbound.orientation,
            acceleration=0.0,
            angular_velocity=0.0
        ),
        constants=car_constants
    ),
    Bus(
        init_state=DynamicActorState(
            position=geometry.Point(400, 0).rotate(road_map.major_road.outbound.orientation).translate(road_map.major_road.outbound.lanes[0].spawn),
            velocity=bus_constants.target_fast_velocity,
            orientation=road_map.major_road.outbound.orientation,
            acceleration=0.0,
            angular_velocity=0.0
        ),
        constants=bus_constants
    ),
    Car(
        init_state=DynamicActorState(
            position=road_map.major_road.outbound.lanes[0].spawn,
            velocity=car_constants.target_fast_velocity,
            orientation=road_map.major_road.outbound.orientation,
            acceleration=0.0,
            angular_velocity=0.0
        ),
        constants=car_constants
    ),
    Car(
        init_state=DynamicActorState(
            position=road_map.major_road.outbound.lanes[1].spawn,
            velocity=car_constants.target_fast_velocity,
            orientation=road_map.major_road.outbound.orientation,
            acceleration=0.0,
            angular_velocity=0.0
        ),
        constants=car_constants
    ),
    Bicycle(
        init_state=DynamicActorState(
            position=road_map.major_road.outbound.lanes[2].spawn,
            velocity=bicycle_constants.target_fast_velocity,
            orientation=road_map.major_road.outbound.orientation,
            acceleration=0.0,
            angular_velocity=0.0
        ),
        constants=bicycle_constants
    )
]


class BusStopEnv(CAVEnv):
    def __init__(self, **kwargs):
        super().__init__(actors=actors, constants=env_constants, **kwargs)
