from dataclasses import dataclass

from cavgym import geometry
from cavgym.geometry import Point


class TrafficLight:
    def __init__(self, position, orientation, width=10.0, height=20.0):
        self.position = position
        self.orientation = orientation
        self.width = width
        self.height = height

        self.static_bounding_box = geometry.make_rectangle(self.width, self.height).transform(self.orientation, self.position)

        def make_light(y):
            return Point(0.0, y).transform(self.orientation, self.position)

        self.red_light = make_light(self.height * 0.25)
        self.amber_light = make_light(0.0)
        self.green_light = make_light(-self.height * 0.25)


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


# class Outbound(Direction):
#     def __init__(self, bounding_box, num_lanes, lane_width):
#         super().__init__(bounding_box, num_lanes, lane_width)
#
#         self.lane_spawns = [lane.static_bounding_box.rear_centre() for lane in self.lanes]
#
#
# class Inbound(Direction):
#     def __init__(self, bounding_box, num_lanes, lane_width):
#         super().__init__(bounding_box, num_lanes, lane_width)
#
#         # self.lanes.reverse()
#
#         self.lane_spawns = [lane.static_bounding_box.rear_centre() for lane in self.lanes]


@dataclass(frozen=True)
class ObstacleConstants:
    width: int
    height: int
    position: geometry.Point
    orientation: float


class Obstacle:
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

        self.static_bounding_box = geometry.make_rectangle(self.constants.road_direction.lane_width * 3, self.constants.road_direction.lane_width * 0.75, left_offset=0).transform(0, Point(self.constants.x_position, 0).transform(0, self.constants.road_direction.static_bounding_box.rear_left))


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

        # self.outbound_orientation = self.constants.orientation
        # self.inbound_orientation = self.outbound_orientation + (geometry.DEG2RAD * 180.0)

    def spawn_position(self, relative_position):
        return relative_position.rotate(self.constants.orientation).relative(self.constants.position)

    def spawn_position_outbound(self, relative_x):
        # return self.spawn_position(Point(relative_x, self.outbound.width))
        return Point(relative_x, 0).rotate(self.constants.orientation).relative(self.static_bounding_box.rear_right)

    def spawn_position_inbound(self, relative_x):
        # return self.spawn_position(Point(relative_x, -self.inbound.width))
        return Point(relative_x, 0).rotate(self.constants.orientation).relative(self.static_bounding_box.rear_left)

    def spawn_orientation(self, relative_orientation):
        return self.constants.orientation + relative_orientation

    def bounding_box(self):
        return self.static_bounding_box


class RoadMap:
    def __init__(self, major_road, minor_roads=None):
        self.major_road = major_road
        self.minor_roads = minor_roads

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

            # def make_outbound_intersection(direction):
            #     return geometry.make_rectangle(self.major_road.width, direction.width, rear_offset=0).transform(direction.orientation, direction.static_bounding_box.rear_centre())
            #
            # def make_inbound_intersection(direction):
            #     return geometry.make_rectangle(self.major_road.width, direction.width, rear_offset=1).transform(direction.orientation, direction.static_bounding_box.front_centre()).flip_longitudinally()
            #
            # self.outbound_intersection_bounding_boxes = [make_outbound_intersection(minor_road.outbound) for minor_road in self.minor_roads]
            # self.inbound_intersection_bounding_boxes = [make_inbound_intersection(minor_road.inbound) for minor_road in self.minor_roads]
            #
            # def make_direction_difference(minor_road, direction):
            #     return geometry.make_rectangle(minor_road.constants.length - major_road.width, direction.width, rear_offset=0).transform(minor_road.constants.orientation, Point(major_road.width, 0).transform(minor_road.constants.orientation, direction.static_bounding_box.rear_centre()))
            #
            # self.outbound_difference_bounding_boxes = [make_direction_difference(minor_road, minor_road.outbound) for minor_road in self.minor_roads]
            # self.inbound_difference_bounding_boxes = [make_direction_difference(minor_road, minor_road.inbound) for minor_road in self.minor_roads]

        self.obstacle = None

    def set_obstacle(self, obstacle):
        self.obstacle = obstacle
