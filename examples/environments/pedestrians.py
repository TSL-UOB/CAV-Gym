import math

from gym.utils import seeding

from library import geometry
from library.actors import DynamicActorState, Car, SpawnPedestrian, SpawnPedestrianState
from library.environment import CAVEnvConstants, RoadMap, CAVEnv
from library.assets import Road, RoadConstants
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

pavement_width = (M2PX * 3)

env_constants = CAVEnvConstants(
    viewer_width=int(road_map.major_road.constants.length),
    viewer_height=int(road_map.major_road.width + (pavement_width * 2)),
    road_map=road_map
)

bounding_box = road_map.major_road.bounding_box()
outbound_pavement = geometry.make_rectangle(road_map.major_road.constants.length, pavement_width, rear_offset=0).transform(road_map.major_road.constants.orientation, geometry.Point(0, pavement_width / 2).translate(bounding_box.rear_left))
inbound_pavement = geometry.make_rectangle(road_map.major_road.constants.length, pavement_width, rear_offset=0).transform(road_map.major_road.constants.orientation, geometry.Point(0, -(pavement_width / 2)).translate(bounding_box.rear_right))
pedestrian_diameter = math.sqrt(pedestrian_constants.length ** 2 + pedestrian_constants.width ** 2)
x_scale = 1 - (pedestrian_diameter / road_map.major_road.constants.length)
y_scale = 1 - (pedestrian_diameter / pavement_width)
spawn_position_boxes = [
    outbound_pavement.rescale(x_scale=x_scale, y_scale=y_scale),
    inbound_pavement.rescale(x_scale=x_scale, y_scale=y_scale)
]
spawn_orientations = [road_map.major_road.outbound.orientation, road_map.major_road.inbound.orientation]


class PedestriansEnv(CAVEnv):
    def __init__(self, num_pedestrians=3, np_random=seeding.np_random(None)[0], **kwargs):
        def spawn_pedestrian():
            return SpawnPedestrian(
                spawn_init_state=SpawnPedestrianState(
                    position_boxes=spawn_position_boxes,
                    velocity=0.0,
                    orientations=spawn_orientations,
                    acceleration=0.0,
                    angular_velocity=0.0,
                    target_velocity=None,
                    target_orientation=None
                ),
                constants=pedestrian_constants,
                np_random=np_random
            )

        actors = [
            Car(
                init_state=DynamicActorState(
                    position=road_map.major_road.outbound.lanes[0].spawn,
                    velocity=car_constants.target_fast_velocity,
                    orientation=road_map.major_road.outbound.orientation,
                    acceleration=0.0,
                    angular_velocity=0.0,
                    target_velocity=None,
                    target_orientation=None
                ),
                constants=car_constants
            )
        ]
        actors += [spawn_pedestrian() for _ in range(num_pedestrians)]

        super().__init__(actors=actors, constants=env_constants, np_random=np_random, **kwargs)
