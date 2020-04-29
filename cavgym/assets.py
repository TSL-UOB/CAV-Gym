from dataclasses import dataclass

from cavgym import utilities
from cavgym.utilities import Point, Bounds


class TrafficLight:
    def __init__(self, position, orientation, width=10.0, height=20.0):
        self.position = position
        self.orientation = orientation
        self.width = width
        self.height = height

        relative_bounds = Bounds(
            rear=-self.width / 2.0,
            left=self.height / 2.0,
            front=self.width / 2.0,
            right=-self.height / 2.0
        )

        self.static_bounding_box = utilities.make_bounding_box(self.position, relative_bounds, self.orientation)

        def make_light(y):
            return Point(0.0, y).rotate(self.orientation).relative(self.position)

        self.red_light = make_light(self.height * 0.25)
        self.amber_light = make_light(0.0)
        self.green_light = make_light(-self.height * 0.25)


@dataclass(frozen=True)
class RoadConstants:
    length: int
    num_outbound_lanes: int
    num_inbound_lanes: int
    lane_width: int

    position: utilities.Point
    orientation: float


class Lane:
    def __init__(self, position, relative_bounds, orientation):
        self.static_bounding_box = utilities.make_bounding_box(position, relative_bounds, orientation)

    def bounding_box(self):
        return self.static_bounding_box


class Direction:
    def __init__(self, position, relative_bounds, orientation, num_lanes, lane_width):
        self.position = position
        self.relative_bounds = relative_bounds
        self.orientation = orientation
        self.num_lanes = num_lanes
        self.lane_width = lane_width

        self.width = self.num_lanes * self.lane_width

        self.static_bounding_box = utilities.make_bounding_box(position, relative_bounds, orientation)
        self.lanes = [Lane(position, lane_relative_bounds, orientation) for lane_relative_bounds in self._iter_lanes_relative_bounds()]
        self.lane_spawns = list(self._iter_lane_spawns())

    def bounding_box(self):
        return self.static_bounding_box

    def _iter_lanes_relative_bounds(self):
        raise NotImplementedError

    def _iter_lane_spawns(self):
        raise NotImplementedError


class Outbound(Direction):
    def __init__(self, position, relative_bounds, orientation, num_lanes, lane_width):
        super().__init__(position, relative_bounds, orientation, num_lanes, lane_width)

    def _iter_lanes_relative_bounds(self):
        left, right = self.relative_bounds.left, self.relative_bounds.right
        lane_left = left
        for _ in range(self.num_lanes):
            lane_right = lane_left - self.lane_width
            yield Bounds(rear=self.relative_bounds.rear, right=lane_right, front=self.relative_bounds.front, left=lane_left)
            lane_left = lane_right

    def _iter_lane_spawns(self):
        for lane in self.lanes:
            coordinates = lane.bounding_box()
            spawn = Point(x=coordinates.rear_left.x, y=coordinates.rear_left.y - (self.lane_width / 2.0))
            yield spawn


class Inbound(Direction):
    def __init__(self, position, relative_bounds, orientation, num_lanes, lane_width):
        super().__init__(position, relative_bounds, orientation, num_lanes, lane_width)

    def _iter_lanes_relative_bounds(self):
        left, right = self.relative_bounds.left, self.relative_bounds.right
        lane_right = right
        for _ in range(self.num_lanes):
            lane_left = lane_right + self.lane_width
            yield Bounds(rear=self.relative_bounds.rear, right=lane_right, front=self.relative_bounds.front, left=lane_left)
            lane_right = lane_left

    def _iter_lane_spawns(self):
        for lane in self.lanes:
            coordinates = lane.bounding_box()
            spawn = Point(x=coordinates.front_right.x, y=coordinates.front_right.y + (self.lane_width / 2.0))
            yield spawn


class Road:
    def __init__(self, constants):
        self.constants = constants

        self.num_lanes = self.constants.num_outbound_lanes + self.constants.num_inbound_lanes
        self.width = self.num_lanes * self.constants.lane_width

        self.relative_bounds = Bounds(rear=0.0, left=self.width / 2.0, front=self.constants.length, right=-self.width / 2.0)

        direction_split = self.relative_bounds.left - (self.constants.num_outbound_lanes * self.constants.lane_width)
        self.outbound = Outbound(self.constants.position, Bounds(rear=self.relative_bounds.rear, left=self.relative_bounds.left, front=self.relative_bounds.front, right=direction_split), self.constants.orientation, self.constants.num_outbound_lanes, self.constants.lane_width)
        self.inbound = Inbound(self.constants.position, Bounds(rear=self.relative_bounds.rear, left=direction_split, front=self.relative_bounds.front, right=self.relative_bounds.right), self.constants.orientation, self.constants.num_inbound_lanes, self.constants.lane_width)

        self.static_bounding_box = utilities.make_bounding_box(self.constants.position, self.relative_bounds, self.constants.orientation)

        self.outbound_orientation = self.constants.orientation
        self.inbound_orientation = self.outbound_orientation + (utilities.DEG2RAD * 180.0)

    def spawn_position(self, relative_position):
        return relative_position.rotate(self.constants.orientation).relative(self.constants.position)

    def spawn_position_outbound(self, relative_x):
        return self.spawn_position(Point(relative_x, self.outbound.width))

    def spawn_position_inbound(self, relative_x):
        return self.spawn_position(Point(relative_x, -self.inbound.width))

    def spawn_orientation(self, relative_orientation):
        return self.constants.orientation + relative_orientation

    def bounding_box(self):
        return self.static_bounding_box


class RoadMap:
    def __init__(self, major_road, minor_roads=None):
        self.major_road = major_road
        self.minor_roads = minor_roads

        if self.minor_roads is not None:
            def make_intersection(position, relative_bounds, orientation):
                intersection_relative_bounds = Bounds(
                    rear=relative_bounds.rear,
                    left=relative_bounds.left,
                    front=relative_bounds.rear + self.major_road.width,
                    right=relative_bounds.right
                )
                return utilities.make_bounding_box(position, intersection_relative_bounds, orientation)

            def make_road_intersection(minor_road):
                return make_intersection(minor_road.constants.position, minor_road.relative_bounds, minor_road.constants.orientation)

            self.intersection_bounding_boxes = [make_road_intersection(minor_road) for minor_road in self.minor_roads]

            def make_direction_intersection(direction):
                return make_intersection(direction.position, direction.relative_bounds, direction.orientation)

            self.outbound_intersection_bounding_boxes = [make_direction_intersection(minor_road.outbound) for minor_road in self.minor_roads]
            self.inbound_intersection_bounding_boxes = [make_direction_intersection(minor_road.inbound) for minor_road in self.minor_roads]

            def make_difference(position, relative_bounds, orientation):
                difference_relative_bounds = Bounds(
                    rear=relative_bounds.rear + self.major_road.width,
                    left=relative_bounds.left,
                    front=relative_bounds.front,
                    right=relative_bounds.right
                )
                return utilities.make_bounding_box(position, difference_relative_bounds, orientation)

            def make_road_difference(minor_road):
                return make_difference(minor_road.constants.position, minor_road.relative_bounds, minor_road.constants.orientation)

            self.difference_bounding_boxes = [make_road_difference(minor_road) for minor_road in self.minor_roads]

            def make_direction_difference(direction):
                return make_difference(direction.position, direction.relative_bounds, direction.orientation)

            self.outbound_difference_bounding_boxes = [make_direction_difference(minor_road.outbound) for minor_road in self.minor_roads]
            self.inbound_difference_bounding_boxes = [make_direction_difference(minor_road.inbound) for minor_road in self.minor_roads]
