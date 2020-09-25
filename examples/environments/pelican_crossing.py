from library import geometry
from library.actors import DynamicActorState, TrafficLightState, PelicanCrossingConstants, Car, Pedestrian
from library.environment import CAVEnvConstants, RoadMap, CAVEnv, PelicanCrossing
from library.assets import Road, RoadConstants, Obstacle, ObstacleConstants
from examples.constants import car_constants, pedestrian_constants, M2PX

road_map = RoadMap(
    major_road=Road(
        constants=RoadConstants(
            length=M2PX * 99,
            num_outbound_lanes=1,
            num_inbound_lanes=1,
            lane_width=M2PX * 3.65,
            position=geometry.Point(0.0, 0.0),
            orientation=geometry.DEG2RAD * 0.0
        )
    )
)

env_constants = CAVEnvConstants(
    viewer_width=road_map.major_road.constants.length,
    viewer_height=road_map.major_road.width + ((M2PX * 3) * 2),
    road_map=road_map
)

pelican_crossing = PelicanCrossing(
    init_state=TrafficLightState.GREEN,
    constants=PelicanCrossingConstants(
        road=road_map.major_road,
        width=road_map.major_road.constants.lane_width * 1.5,
        x_position=road_map.major_road.constants.length * 0.5
    )
)

road_map.set_obstacle(
    Obstacle(
        ObstacleConstants(
            width=M2PX * 3,
            height=M2PX * 1.5,
            position=geometry.Point(-20, -20).transform(pelican_crossing.constants.road.constants.orientation, pelican_crossing.static_bounding_box.rear_right),
            orientation=pelican_crossing.constants.road.constants.orientation
        )
    )
)

actors = [
    Car(
        init_state=DynamicActorState(
            position=road_map.major_road.outbound.lanes[0].spawn,
            velocity=car_constants.target_fast_velocity,
            orientation=road_map.major_road.outbound.orientation,
            throttle=0.0,
            steering_angle=0.0,
            target_velocity=None,
            target_orientation=None
        ),
        constants=car_constants
    ),
    Car(
        init_state=DynamicActorState(
            position=road_map.major_road.inbound.lanes[0].spawn,
            velocity=0.0,
            orientation=road_map.major_road.inbound.orientation,
            throttle=0.0,
            steering_angle=0.0,
            target_velocity=None,
            target_orientation=None
        ),
        constants=car_constants
    ),
    pelican_crossing,
    Pedestrian(
        init_state=DynamicActorState(
            position=pelican_crossing.inbound_spawn,
            velocity=0.0,
            orientation=road_map.major_road.outbound.orientation + (geometry.DEG2RAD * 90.0),
            throttle=0.0,
            steering_angle=0.0,
            target_velocity=None,
            target_orientation=None
        ),
        constants=pedestrian_constants
    ),
    Pedestrian(
        init_state=DynamicActorState(
            position=pelican_crossing.outbound_spawn,
            velocity=0.0,
            orientation=road_map.major_road.outbound.orientation + (geometry.DEG2RAD * 270.0),
            throttle=0.0,
            steering_angle=0.0,
            target_velocity=None,
            target_orientation=None
        ),
        constants=pedestrian_constants
    )
]


class PelicanCrossingEnv(CAVEnv):
    def __init__(self, **kwargs):
        super().__init__(actors=actors, constants=env_constants, **kwargs)
