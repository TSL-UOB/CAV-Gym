from library import geometry
from library.actors import DynamicActorState, Car, Pedestrian
from library.environment import CAVEnvConstants, RoadMap, CAVEnv
from library.assets import RoadConstants, Road
from scenarios.constants import car_constants, pedestrian_constants, M2PX

major_road = Road(
    constants=RoadConstants(
        length=M2PX * 99,
        num_outbound_lanes=1,
        num_inbound_lanes=1,
        lane_width=M2PX * 3.65,
        position=geometry.Point(0.0, 0.0),
        orientation=geometry.DEG2RAD * 0.0
    )
)

road_map = RoadMap(
    major_road=major_road,
    minor_roads=[
        Road(
            constants=RoadConstants(
                length=M2PX * 24.75,
                num_outbound_lanes=1,
                num_inbound_lanes=1,
                lane_width=M2PX * 3.65,
                position=major_road.spawn_position_inbound((major_road.constants.length * 0.5) - (major_road.constants.lane_width * 0.25)),
                orientation=major_road.spawn_orientation(geometry.DEG2RAD * 270.0)
            )
        ),
        Road(
            constants=RoadConstants(
                length=M2PX * 24.75,
                num_outbound_lanes=1,
                num_inbound_lanes=1,
                lane_width=M2PX * 3.65,
                position=major_road.spawn_position_outbound((major_road.constants.length * 0.5) + (major_road.constants.lane_width * 0.25)),
                orientation=major_road.spawn_orientation(geometry.DEG2RAD * 90.0)
            )
        )
    ]
)

env_constants = CAVEnvConstants(
    viewer_width=road_map.major_road.constants.length,
    viewer_height=sum(minor_road.constants.length for minor_road in road_map.minor_roads) - major_road.width,
    road_map=road_map
)

actors = [
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
            position=road_map.major_road.inbound.lanes[0].spawn,
            velocity=car_constants.target_fast_velocity,
            orientation=road_map.major_road.inbound.orientation,
            acceleration=0.0,
            angular_velocity=0.0
        ),
        constants=car_constants
    ),
    Pedestrian(
        init_state=DynamicActorState(
            position=geometry.Point(160, -20).translate(road_map.intersection_bounding_boxes[0].front_left),
            velocity=0.0,
            orientation=road_map.major_road.inbound.orientation,
            acceleration=0.0,
            angular_velocity=0.0
        ),
        constants=pedestrian_constants
    )
]


class CrossroadsEnv(CAVEnv):
    def __init__(self, **kwargs):
        super().__init__(actors=actors, constants=env_constants, **kwargs)
