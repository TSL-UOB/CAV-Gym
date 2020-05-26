from gym.utils import seeding

from cavgym import geometry
from cavgym.actors import DynamicActorState, Car, SpawnPedestrian, SpawnPedestrianState
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

bounding_box = road_map.major_road.bounding_box()
spawn_offset = road_map.major_road.constants.lane_width * 0.5
spawn_position_lines = [
    geometry.Line(start=geometry.Point(0, spawn_offset).transform(0, bounding_box.rear_left), end=geometry.Point(0, spawn_offset).transform(0, bounding_box.front_left)),
    geometry.Line(start=geometry.Point(0, -spawn_offset).transform(0, bounding_box.rear_right), end=geometry.Point(0, -spawn_offset).transform(0, bounding_box.front_right))
]
spawn_orientations = [road_map.major_road.outbound.orientation, road_map.major_road.inbound.orientation]


class PedestriansEnv(CAVEnv):
    def __init__(self, num_pedestrians=3, np_random=seeding.np_random(None)[0]):
        def spawn_pedestrian():
            return SpawnPedestrian(
                spawn_init_state=SpawnPedestrianState(
                    position_lines=spawn_position_lines,
                    velocity=0.0,
                    orientations=spawn_orientations,
                    acceleration=0,
                    angular_velocity=0.0
                ),
                constants=pedestrian_constants,
                np_random=np_random
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
            )
        ]
        actors += [spawn_pedestrian() for _ in range(num_pedestrians)]

        super().__init__(actors=actors, constants=env_constants, np_random=np_random)
