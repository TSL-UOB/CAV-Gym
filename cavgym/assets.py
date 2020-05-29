import itertools
from dataclasses import dataclass

from cavgym import geometry
from cavgym.geometry import Point


class Occlusion:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # important to pass on kwargs if class is used as superclass in multiple inheritance

    def bounding_box(self):
        raise NotImplementedError

    def occlusion_zone(self, observation_point):
        points = [Point(x, y) for x, y in self.bounding_box()]
        max_angle = 0
        max_triangle = None
        for pair in itertools.combinations(points, 2):
            p1, p2 = pair
            triangle = geometry.Triangle(rear=observation_point, front_left=p1, front_right=p2)
            angle = triangle.angle()
            if abs(angle) > abs(max_angle):
                max_angle = angle
                max_triangle = triangle
        anchor_triangle = max_triangle.normalise()

        enlarged = geometry.Triangle(
            rear=anchor_triangle.rear,
            front_left=anchor_triangle.front_left.enlarge(anchor_triangle.rear),
            front_right=anchor_triangle.front_right.enlarge(anchor_triangle.rear)
        )

        return geometry.ConvexQuadrilateral(
            rear_left=anchor_triangle.front_left,
            front_left=enlarged.front_left,
            front_right=enlarged.front_right,
            rear_right=anchor_triangle.front_right
        )


class Lane:
    def __init__(self, bounding_box):
        self.static_bounding_box = bounding_box

        self.spawn = self.static_bounding_box.rear_centre()

    def bounding_box(self):
        return self.static_bounding_box


class Direction:
    def __init__(self, bounding_box, num_lanes, lane_width, orientation):
        self.static_bounding_box = bounding_box
        self.num_lanes = num_lanes
        self.lane_width = lane_width
        self.orientation = orientation

        self.width = self.lane_width * self.num_lanes

        self.lanes = [Lane(bounding_box) for bounding_box in self.static_bounding_box.divide_laterally(self.num_lanes)] if self.num_lanes > 0 else list()

        self.bus_stop = None

    def bounding_box(self):
        return self.static_bounding_box

    def set_bus_stop(self, bus_stop):
        self.bus_stop = bus_stop


@dataclass(frozen=True)
class RoadConstants:
    length: int
    num_outbound_lanes: int
    num_inbound_lanes: int
    lane_width: int

    position: geometry.Point
    orientation: float


class Road:
    def __init__(self, constants):
        self.constants = constants

        self.num_lanes = self.constants.num_outbound_lanes + self.constants.num_inbound_lanes
        self.width = self.num_lanes * self.constants.lane_width

        self.static_bounding_box = geometry.make_rectangle(self.constants.length, self.width, rear_offset=0).transform(self.constants.orientation, self.constants.position)

        left, right = self.static_bounding_box.split_laterally(left_percentage=self.constants.num_outbound_lanes / self.num_lanes)
        self.outbound = Direction(left, self.constants.num_outbound_lanes, self.constants.lane_width, self.constants.orientation)
        self.inbound = Direction(right.flip(), self.constants.num_inbound_lanes, self.constants.lane_width, self.constants.orientation + (geometry.DEG2RAD * 180.0))

    def spawn_position(self, relative_position):
        return relative_position.rotate(self.constants.orientation).translate(self.constants.position)

    def spawn_position_outbound(self, relative_x):
        return Point(relative_x, 0).rotate(self.constants.orientation).translate(self.static_bounding_box.rear_right)

    def spawn_position_inbound(self, relative_x):
        return Point(relative_x, 0).rotate(self.constants.orientation).translate(self.static_bounding_box.rear_left)

    def spawn_orientation(self, relative_orientation):
        return self.constants.orientation + relative_orientation

    def bounding_box(self):
        return self.static_bounding_box


class RoadMap:
    def __init__(self, major_road, minor_roads=None):
        self.major_road = major_road
        self.minor_roads = minor_roads

        self.roads = [self.major_road] + (self.minor_roads if self.minor_roads is not None else list())

        if self.minor_roads is not None:
            road_partitions = [minor_road.static_bounding_box.split_longitudinally(rear_percentage=self.major_road.width / minor_road.constants.length) for minor_road in self.minor_roads]

            self.intersection_bounding_boxes = [intersection for intersection, _ in road_partitions]
            self.difference_bounding_boxes = [remainder for _, remainder in road_partitions]

            outbound_partitions = [minor_road.outbound.static_bounding_box.split_longitudinally(rear_percentage=self.major_road.width / minor_road.constants.length) for minor_road in self.minor_roads]

            self.outbound_intersection_bounding_boxes = [intersection for intersection, _ in outbound_partitions]
            self.outbound_difference_bounding_boxes = [remainder for _, remainder in outbound_partitions]

            inbound_partitions = [minor_road.inbound.static_bounding_box.split_longitudinally(rear_percentage=1 - (self.major_road.width / minor_road.constants.length)) for minor_road in self.minor_roads]

            self.inbound_intersection_bounding_boxes = [intersection for _, intersection in inbound_partitions]
            self.inbound_difference_bounding_boxes = [remainder for remainder, _ in inbound_partitions]

        self.obstacle = None

    def set_obstacle(self, obstacle):
        self.obstacle = obstacle


@dataclass(frozen=True)
class ObstacleConstants:
    width: int
    height: int
    position: geometry.Point
    orientation: float


class Obstacle(Occlusion):
    def __init__(self, constants):
        self.constants = constants

        self.static_bounding_box = geometry.make_rectangle(self.constants.width, self.constants.height).transform(self.constants.orientation, self.constants.position)

    def bounding_box(self):
        return self.static_bounding_box


@dataclass(frozen=True)
class BusStopConstants:
    road_direction: Direction
    x_position: int


class BusStop:
    def __init__(self, constants):
        self.constants = constants

        self.static_bounding_box = geometry.make_rectangle(self.constants.road_direction.lane_width * 3, self.constants.road_direction.lane_width * 0.75, left_offset=0).translate(Point(self.constants.x_position, 0).translate(self.constants.road_direction.static_bounding_box.rear_left))
