import math

from library import geometry
from library.bodies import DynamicBodyState, Car, Pedestrian
from library.environment import CAVEnvConstants, RoadMap, CAVEnv
from library.assets import RoadConstants, Road
from examples.constants import car_constants, pedestrian_constants, M2PX

major_road = Road(
    constants=RoadConstants(
        length=M2PX * 99,
        num_outbound_lanes=1,
        num_inbound_lanes=1,
        lane_width=M2PX * 3.65,
        position=geometry.Point(0.0, 0.0),
        orientation=0.0
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
                orientation=major_road.spawn_orientation(math.radians(270.0))
            )
        ),
        Road(
            constants=RoadConstants(
                length=M2PX * 24.75,
                num_outbound_lanes=1,
                num_inbound_lanes=1,
                lane_width=M2PX * 3.65,
                position=major_road.spawn_position_outbound((major_road.constants.length * 0.5) + (major_road.constants.lane_width * 0.25)),
                orientation=major_road.spawn_orientation(math.radians(90.0))
            )
        )
    ]
)

env_constants = CAVEnvConstants(
    viewer_width=road_map.major_road.constants.length,
    viewer_height=sum(minor_road.constants.length for minor_road in road_map.minor_roads) - major_road.width,
    road_map=road_map
)

bodies = [
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
            position=road_map.major_road.inbound.lanes[0].spawn,
            velocity=car_constants.max_velocity / 2.0,
            orientation=road_map.major_road.inbound.orientation
        ),
        constants=car_constants
    ),
    Pedestrian(
        init_state=DynamicBodyState(
            position=geometry.Point(160, -20).translate(road_map.intersection_bounding_boxes[0].front_left),
            velocity=0.0,
            orientation=road_map.major_road.inbound.orientation
        ),
        constants=pedestrian_constants
    )
]


class CrossroadsEnv(CAVEnv):
    def __init__(self, **kwargs):
        super().__init__(bodies=bodies, constants=env_constants, **kwargs)
