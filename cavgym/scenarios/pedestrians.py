from cavgym import geometry
from cavgym.actors import DynamicActorState, Car, Pedestrian
from cavgym.environment import CAVEnvConstants, RoadMap, CAVEnv
from cavgym.assets import Road, RoadConstants
from cavgym.scenarios import car_constants, pedestrian_constants


road_map = RoadMap(
    major_road=Road(
        constants=RoadConstants(
            length=1600,
            num_outbound_lanes=1,
            num_inbound_lanes=1,
            lane_width=40,
            position=geometry.Point(0.0, 0.0),
            orientation=geometry.DEG2RAD * 0.0
        )
    )
)

env_constants = CAVEnvConstants(
    viewer_width=road_map.major_road.constants.length,
    viewer_height=road_map.major_road.width + (road_map.major_road.constants.lane_width * 2),
    time_resolution=1.0 / 60.0,
    road_map=road_map
)

actors = [
    Car(
        init_state=DynamicActorState(
            position=road_map.major_road.outbound.lanes[0].spawn,
            velocity=100.0,
            orientation=road_map.major_road.outbound.orientation,
            acceleration=0,
            angular_velocity=0.0
        ),
        constants=car_constants
    ),
    Pedestrian(
        init_state=DynamicActorState(
            position=geometry.Point(0, road_map.major_road.constants.lane_width * 0.5).translate(road_map.major_road.outbound.bounding_box().left_centre()),
            velocity=0.0,
            orientation=road_map.major_road.outbound.orientation,
            acceleration=0,
            angular_velocity=0.0
        ),
        constants=pedestrian_constants
    ),
    Pedestrian(
        init_state=DynamicActorState(
            position=geometry.Point(-road_map.major_road.constants.lane_width, -road_map.major_road.constants.lane_width * 0.5).translate(road_map.major_road.inbound.bounding_box().left_centre()),
            velocity=0.0,
            orientation=road_map.major_road.inbound.orientation,
            acceleration=0,
            angular_velocity=0.0
        ),
        constants=pedestrian_constants
    ),
    Pedestrian(
        init_state=DynamicActorState(
            position=geometry.Point(road_map.major_road.constants.lane_width, -road_map.major_road.constants.lane_width * 0.5).translate(road_map.major_road.inbound.bounding_box().left_centre()),
            velocity=0.0,
            orientation=road_map.major_road.outbound.orientation,
            acceleration=0,
            angular_velocity=0.0
        ),
        constants=pedestrian_constants
    )
]


class PedestriansEnv(CAVEnv):
    def __init__(self):
        super().__init__(actors=actors, constants=env_constants)
