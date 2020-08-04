from enum import Enum


class Observation(Enum):
    def __repr__(self):
        return self.name


class RoadObservation(Observation):
    ON_ROAD = 0
    ROAD_FRONT = 1
    ROAD_FRONT_LEFT = 2
    ROAD_LEFT = 3
    ROAD_REAR_LEFT = 4
    ROAD_REAR = 5
    ROAD_REAR_RIGHT = 6
    ROAD_RIGHT = 7
    ROAD_FRONT_RIGHT = 8
