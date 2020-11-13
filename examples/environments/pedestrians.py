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
# spawn_position_boxes = [
#     outbound_pavement.rescale(x_scale=x_scale, y_scale=y_scale),
#     inbound_pavement.rescale(x_scale=x_scale, y_scale=y_scale)
# ]
spawn_orientations = [road_map.major_road.outbound.orientation, road_map.major_road.inbound.orientation]


def make_spawn_position_boxes(outbound_percentage, inbound_percentage):
    assert 0 <= outbound_percentage <= 1
    assert 0 <= inbound_percentage <= 1
    assert outbound_percentage > 0 or inbound_percentage > 0

    def make_spawn_position_box(pavement, percentage):
        if percentage == 0:
            return None  # nothing
        else:
            rescaled_pavement = pavement.rescale(x_scale=x_scale, y_scale=y_scale)
            if percentage == 1:
                return rescaled_pavement  # everything
            else:
                return rescaled_pavement.split_longitudinally(1 - percentage)[1]  # part

    outbound_box = make_spawn_position_box(outbound_pavement, outbound_percentage)
    inbound_box = make_spawn_position_box(inbound_pavement, inbound_percentage)

    boxes = list()
    if outbound_box:
        boxes.append(outbound_box)
    if inbound_box:
        boxes.append(inbound_box)
    return boxes


class PedestriansEnv(CAVEnv):
    def __init__(self, num_pedestrians, outbound_percentage, inbound_percentage, np_random=seeding.np_random(None)[0], **kwargs):
        def spawn_pedestrian():
            return SpawnPedestrian(
                spawn_init_state=SpawnPedestrianState(
                    position_boxes=make_spawn_position_boxes(outbound_percentage, inbound_percentage),
                    velocity=0.0,
                    orientations=spawn_orientations
                ),
                constants=pedestrian_constants,
                np_random=np_random
            )

        actors = [
            Car(
                init_state=DynamicActorState(
                    position=road_map.major_road.outbound.lanes[0].spawn,
                    velocity=car_constants.max_velocity,
                    orientation=road_map.major_road.outbound.orientation
                ),
                constants=car_constants
            )
        ]
        actors += [spawn_pedestrian() for _ in range(num_pedestrians)]

        super().__init__(actors=actors, constants=env_constants, np_random=np_random, **kwargs)
