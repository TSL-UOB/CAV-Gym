from gym.utils import seeding

from libcavgym import geometry
from libcavgym.actors import DynamicActorState, Car, SpawnPedestrian, SpawnPedestrianState
from libcavgym.environment import CAVEnvConstants, RoadMap, CAVEnv
from libcavgym.assets import Road, RoadConstants
from example.constants import car_constants, pedestrian_constants, M2PX

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

bounding_box = road_map.major_road.bounding_box()
spawn_offset = M2PX * 1.5
spawn_position_lines = [
    geometry.Line(start=geometry.Point(0, spawn_offset).translate(bounding_box.rear_left), end=geometry.Point(0, spawn_offset).translate(bounding_box.front_left)),
    geometry.Line(start=geometry.Point(0, -spawn_offset).translate(bounding_box.rear_right), end=geometry.Point(0, -spawn_offset).translate(bounding_box.front_right))
]
spawn_orientations = [road_map.major_road.outbound.orientation, road_map.major_road.inbound.orientation]


class PedestriansEnv(CAVEnv):
    def __init__(self, num_pedestrians=3, np_random=seeding.np_random(None)[0], **kwargs):
        def spawn_pedestrian():
            return SpawnPedestrian(
                spawn_init_state=SpawnPedestrianState(
                    position_lines=spawn_position_lines,
                    velocity=0.0,
                    orientations=spawn_orientations,
                    acceleration=0.0,
                    angular_velocity=0.0
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
                    angular_velocity=0.0
                ),
                constants=car_constants
            )
        ]
        actors += [spawn_pedestrian() for _ in range(num_pedestrians)]

        super().__init__(actors=actors, constants=env_constants, np_random=np_random, **kwargs)
