from cavgym import utilities
from cavgym.actors import DynamicActorState, TrafficLightState, PelicanCrossingConstants, Car, Pedestrian
from cavgym.environment import CAVEnvConstants, RoadMap, CAVEnv, PelicanCrossing
from cavgym.assets import Road, RoadConstants, Obstacle, ObstacleConstants
from cavgym.scenarios import car_constants, pedestrian_constants


road_map = RoadMap(
    major_road=Road(
        constants=RoadConstants(
            length=1600,
            num_outbound_lanes=1,
            num_inbound_lanes=1,
            lane_width=40,
            position=utilities.Point(0.0, 0.0),
            orientation=utilities.DEG2RAD * 0.0
        )
    )
)

env_constants = CAVEnvConstants(
    viewer_width=road_map.major_road.constants.length,
    viewer_height=road_map.major_road.constants.lane_width * (road_map.major_road.constants.num_outbound_lanes + road_map.major_road.constants.num_inbound_lanes + 2),
    time_resolution=1.0 / 60.0,
    road_map=road_map
)

pelican_crossing = PelicanCrossing(
    init_state=TrafficLightState.GREEN,
    constants=PelicanCrossingConstants(
        road=road_map.major_road,
        width=road_map.major_road.constants.lane_width * 1.5,
        x_position=road_map.major_road.constants.length / 2.0
    )
)

road_map.set_obstacle(
    Obstacle(
        ObstacleConstants(
            width=40,
            height=20,
            position=utilities.Point(-20, -20).rotate(pelican_crossing.constants.road.constants.orientation).relative(pelican_crossing.static_bounding_box.rear_right),
            orientation=pelican_crossing.constants.road.constants.orientation
        )
    )
)

actors = [
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
            position=road_map.major_road.inbound.lane_spawns[0],
            velocity=100.0,
            orientation=road_map.major_road.inbound_orientation,
            acceleration=0,
            angular_velocity=0.0
        ),
        constants=car_constants
    ),
    pelican_crossing,
    Pedestrian(
        init_state=DynamicActorState(
            position=pelican_crossing.inbound_spawn,
            velocity=0.0,
            orientation=road_map.major_road.outbound_orientation + (utilities.DEG2RAD * 90.0),
            acceleration=0,
            angular_velocity=0.0
        ),
        constants=pedestrian_constants
    ),
    Pedestrian(
        init_state=DynamicActorState(
            position=pelican_crossing.outbound_spawn,
            velocity=0.0,
            orientation=road_map.major_road.outbound_orientation + (utilities.DEG2RAD * 270.0),
            acceleration=0,
            angular_velocity=0.0
        ),
        constants=pedestrian_constants
    )
]


class PelicanCrossingEnv(CAVEnv):
    def __init__(self):
        super().__init__(actors=actors, constants=env_constants)
