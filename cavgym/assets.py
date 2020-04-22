from dataclasses import dataclass

from shapely import geometry, affinity

from cavgym import utilities


@dataclass(frozen=True)
class RoadConstants:
    length: float
    num_outbound_lanes: int
    num_inbound_lanes: int
    lane_width: float

    position: utilities.Point
    orientation: float


class Road:
    def __init__(self, constants):
        self.constants = constants

        self.num_lanes = self.constants.num_outbound_lanes + self.constants.num_inbound_lanes
        self.width = self.num_lanes * self.constants.lane_width

        rear, front = 0, self.constants.length
        right, left = -self.width / 2.0, self.width / 2.0
        self.bounds = rear, right, front, left

        left_inbound = right + (self.constants.num_inbound_lanes * self.constants.lane_width)
        self.inbound_bounds = rear, right, front, left_inbound
        self.outbound_bounds = rear, left_inbound, front, left_inbound + (self.constants.num_outbound_lanes * self.constants.lane_width)

        self.inbound_lanes_bounds = list(self._iter_lane_split(self.inbound_bounds, self.constants.num_inbound_lanes))
        self.outbound_lanes_bounds = list(self._iter_lane_split(self.outbound_bounds, self.constants.num_outbound_lanes))

        box = geometry.box(*self.bounds)
        rotated_box = affinity.rotate(box, self.constants.orientation, use_radians=True)
        self.static_bounding_box = affinity.translate(rotated_box, self.constants.position.x, self.constants.position.y)

    def _iter_lane_split(self, bounds, num_lanes):
        rear, right, front, _ = bounds
        lane_left = right
        for _ in range(num_lanes):
            lane_right = lane_left
            lane_left = lane_right + self.constants.lane_width
            yield rear, lane_right, front, lane_left


class RoadMap:
    def __init__(self, main_road, minor_roads=None):
        self.main_road = main_road
        self.minor_roads = minor_roads
