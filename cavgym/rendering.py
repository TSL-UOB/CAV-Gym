from enum import Enum

from gym.envs.classic_control import rendering

from cavgym import mods, utilities
from cavgym.actors import TrafficLightState, Car, Bus, Bicycle, Pedestrian, PelicanCrossing


class BulbState(Enum):
    OFF = 0
    DIM = 1
    FULL = 2


class ActorView:
    def update(self, actor):
        raise NotImplementedError


def make_rectangle_coordinates(bounds):
    rear, right, front, left = bounds
    return [(front, right), (front, left), (rear, left), (rear, right)]


def make_rectangle(bounds, rgb=None, transform=None):
    coordinates = make_rectangle_coordinates(bounds)
    rectangle = rendering.make_polygon(coordinates)
    return attributed_shape(rectangle, rgb=rgb, transform=transform)


def make_circle(diameter, rgb=None, transform=None):
    circle = rendering.make_circle(diameter)
    return attributed_shape(circle, rgb=rgb, transform=transform)


def make_compound(elements, rgb=None, transform=None):
    compound = rendering.Compound(elements)
    return attributed_shape(compound, rgb=rgb, transform=transform)


def attributed_shape(shape, rgb=None, transform=None):
    if rgb is not None:
        shape.set_color(*rgb)
    if transform is not None:
        shape.add_attr(transform)
    return shape


class DynamicActorView(ActorView):
    def __init__(self, dynamic_actor):
        self.transform = rendering.Transform(translation=(dynamic_actor.state.position.x, dynamic_actor.state.position.y), rotation=dynamic_actor.state.orientation)

        hard_braking_bounds, normal_braking_bounds = dynamic_actor.stopping_bounds()
        self.hard_braking = make_rectangle(hard_braking_bounds, rgb=(0.9, 0.9, 0.9), transform=self.transform)
        self.normal_braking = make_rectangle(normal_braking_bounds, rgb=(0.95, 0.95, 0.95), transform=self.transform)

        self.body = make_rectangle(dynamic_actor.bounds, transform=self.transform)

    def update(self, dynamic_actor):
        self.transform.set_translation(dynamic_actor.state.position.x, dynamic_actor.state.position.y)
        self.transform.set_rotation(dynamic_actor.state.orientation)

        hard_braking_bounds, normal_braking_bounds = dynamic_actor.stopping_bounds()
        self.hard_braking.v = make_rectangle_coordinates(hard_braking_bounds)
        self.normal_braking.v = make_rectangle_coordinates(normal_braking_bounds)


class VehicleView(DynamicActorView):
    def __init__(self, vehicle):
        super().__init__(vehicle)

        rear, right, front, left = vehicle.bounds
        light_offset = vehicle.constants.width * 0.25

        self.scale = {
            BulbState.OFF: (0, 0),
            BulbState.DIM: (0.75, 0.75),
            BulbState.FULL: (1, 1)
        }

        def make_light(x, y):
            light_transform = rendering.Transform(translation=(x, y), scale=self.scale[BulbState.OFF])
            light = make_circle(vehicle.constants.width * 0.2, transform=light_transform)
            return light, light_transform

        def make_indicators(y):
            front_indicator, front_indicator_transform = make_light(front - light_offset, y)
            rear_indicator, rear_indicator_transform = make_light(rear + light_offset, y)
            indicators = make_compound([front_indicator, rear_indicator], rgb=(1, 0.75, 0), transform=self.transform)
            return indicators, (front_indicator_transform, rear_indicator_transform)

        self.right_indicators, self.right_indicator_transforms = make_indicators(right)
        self.left_indicators, self.left_indicator_transforms = make_indicators(left)

        def make_longitudinal_lights(x, rgb):
            longitudinal_light_left, longitudinal_light_left_transform = make_light(x, left - light_offset)
            longitudinal_light_right, longitudinal_light_right_transform = make_light(x, right + light_offset)
            longitudinal_lights = make_compound([longitudinal_light_left, longitudinal_light_right], rgb=rgb, transform=self.transform)
            return longitudinal_lights, (longitudinal_light_left_transform, longitudinal_light_right_transform)

        self.brake_lights, self.brake_light_transforms = make_longitudinal_lights(rear, (1, 0, 0))
        self.headlights, self.headlight_transforms = make_longitudinal_lights(front, (1, 1, 0))

    def set_lights(self, light_transforms, state):
        for light_transform in light_transforms:
            light_transform.set_scale(*self.scale[state])

    def set_right_indicators(self, state):
        self.set_lights(self.right_indicator_transforms, state)

    def set_left_indicators(self, state):
        self.set_lights(self.left_indicator_transforms, state)

    def set_brake_lights(self, state):
        self.set_lights(self.brake_light_transforms, state)

    def set_headlights(self, state):
        self.set_lights(self.headlight_transforms, state)

    def update(self, vehicle):
        super().update(vehicle)

        left_indicator_state = BulbState.OFF
        right_indicator_state = BulbState.OFF
        if vehicle.state.angular_velocity > 0:
            left_indicator_state = BulbState.DIM if vehicle.state.angular_velocity == vehicle.constants.normal_left_turn else BulbState.FULL
        elif vehicle.state.angular_velocity < 0:
            right_indicator_state = BulbState.DIM if vehicle.state.angular_velocity == vehicle.constants.normal_right_turn else BulbState.FULL
        self.set_left_indicators(left_indicator_state)
        self.set_right_indicators(right_indicator_state)

        brake_lights_state = BulbState.OFF
        headlights_state = BulbState.OFF
        if vehicle.state.acceleration < 0:
            brake_lights_state = BulbState.DIM if vehicle.state.acceleration == vehicle.constants.normal_deceleration else BulbState.FULL
        elif vehicle.state.acceleration > 0:
            headlights_state = BulbState.DIM if vehicle.state.acceleration == vehicle.constants.normal_acceleration else BulbState.FULL
        self.set_brake_lights(brake_lights_state)
        self.set_headlights(headlights_state)


class CarView(VehicleView):
    def __init__(self, car):
        super().__init__(car)

        rear, right, front, left = car.bounds
        roof_offset = car.constants.length * 0.25
        roof_bounds = rear + roof_offset, right, front - roof_offset, left

        self.roof = make_rectangle(roof_bounds, rgb=(0.5, 0.5, 0.5), transform=self.transform)


class BicycleView(DynamicActorView):
    def __init__(self, bicycle):
        super().__init__(bicycle)

        self.head = make_circle(bicycle.constants.width * 0.3, rgb=(0.5, 0.5, 0.5), transform=self.transform)


class PedestrianView(DynamicActorView):
    def __init__(self, pedestrian):
        super().__init__(pedestrian)

        self.head = make_circle(pedestrian.constants.length * 0.4, rgb=(0.5, 0.5, 0.5), transform=self.transform)


class TrafficLightView(ActorView):
    def __init__(self, width, height, position):
        self.transform = rendering.Transform(translation=(position.x, position.y), rotation=0.0)

        rear, front = -width / 2.0, width / 2.0
        right, left = -height / 2.0, height / 2.0
        traffic_light_bounds = rear, right, front, left

        self.body = make_rectangle(traffic_light_bounds, transform=self.transform)

        def make_light(y, rgb):
            light_transform = rendering.Transform(translation=(0, y))
            light = make_circle(width * 0.25, rgb=rgb, transform=light_transform)
            light.add_attr(self.transform)
            return light, light_transform

        self.red_light, self.red_light_transform = make_light(height * 0.25, (1, 0, 0))
        self.amber_light, self.amber_light_transform = make_light(0, (1, 0.75, 0))
        self.green_light, self.green_light_transform = make_light(-height * 0.25, (0, 1, 0))

    def set_light(self, light_transform):
        for other_light_transform in [self.red_light_transform, self.amber_light_transform, self.green_light_transform]:
            if other_light_transform is not light_transform:
                other_light_transform.set_scale(0, 0)
        light_transform.set_scale(1, 1)

    def set_red_light(self):
        self.set_light(self.red_light_transform)

    def set_amber_light(self):
        self.set_light(self.amber_light_transform)

    def set_green_light(self):
        self.set_light(self.green_light_transform)

    def update(self, traffic_light):
        if traffic_light.state is TrafficLightState.RED:
            self.set_red_light()
        elif traffic_light.state is TrafficLightState.AMBER:
            self.set_amber_light()
        elif traffic_light.state is TrafficLightState.GREEN:
            self.set_green_light()

    def items(self):
        yield from [self.body, self.red_light, self.amber_light, self.green_light]


class PelicanCrossingView(ActorView):
    def __init__(self, pelican_crossing):
        self.transform = rendering.Transform(translation=(pelican_crossing.constants.x_position, 0.0), rotation=0.0)

        rear, right, front, left = pelican_crossing.bounds

        self.area = make_rectangle(pelican_crossing.bounds, rgb=(1, 1, 1), transform=self.transform)

        _, outbound_right, _, outbound_left = pelican_crossing.constants.road.outbound_bounds
        self.outbound_marking = make_compound([rendering.make_polyline([(rear + i, outbound_left), (rear + i, outbound_right)]) for i in range(2)], transform=self.transform)

        _, inbound_right, _, inbound_left = pelican_crossing.constants.road.inbound_bounds
        self.inbound_marking = make_compound([rendering.make_polyline([(front - i, inbound_left), (front - i, inbound_right)]) for i in range(2)], transform=self.transform)

        inner_rear = rear + (pelican_crossing.constants.width * 0.15)
        inner_front = front - (pelican_crossing.constants.width * 0.15)
        self.inner = rendering.make_polygon([(inner_front, right), (inner_front, left), (inner_rear, left), (inner_rear, right)], filled=False)
        self.inner.add_attr(mods.FactoredLineStyle(0x0F0F, 1))
        self.inner.add_attr(self.transform)

        def make_traffic_light_view(x, y):
            traffic_light_view = TrafficLightView(10.0, 20.0, utilities.Point(x, y))
            for item in traffic_light_view.items():
                item.add_attr(self.transform)
            traffic_light_view.set_green_light()
            return traffic_light_view

        self.outbound_traffic_light_view = make_traffic_light_view(-pelican_crossing.constants.width / 2.0, left + 20.0)
        self.inbound_traffic_light_view = make_traffic_light_view(pelican_crossing.constants.width / 2.0, right - 20.0)

    def update(self, pelican_crossing):
        for traffic_light_view in [self.outbound_traffic_light_view, self.inbound_traffic_light_view]:
            traffic_light_view.update(pelican_crossing)

    def items(self):
        yield from [self.area, self.outbound_marking, self.inbound_marking, self.inner]
        yield from self.outbound_traffic_light_view.items()
        yield from self.inbound_traffic_light_view.items()


class RoadView:
    def __init__(self, road):
        rear, right, front, left = road.bounds
        self.edge_markings = rendering.make_polygon([(front, right), (front, left), (rear, left), (rear, right)], filled=False)
        self.transform = rendering.Transform(translation=(road.constants.position.x, road.constants.position.y), rotation=road.constants.orientation)
        self.edge_markings.add_attr(self.transform)

        def make_centre_line():
            inbound_rear, _, inbound_front, inbound_left = road.inbound_bounds
            centre_line = rendering.make_polyline([(inbound_rear, inbound_left), (inbound_front, inbound_left)])
            centre_line.add_attr(mods.FactoredLineStyle(0x00FF, 2))
            centre_line.add_attr(self.transform)
            return centre_line

        self.centre_markings = make_centre_line()

        def iter_lane_lines(lanes):
            for lane_bounds in lanes[:-1]:
                lane_rear, _, lane_front, lane_left = lane_bounds
                yield (lane_rear, lane_left), (lane_front, lane_left)

        def make_lane_line(line):
            lane_line = rendering.make_polyline(line)
            lane_line.add_attr(mods.FactoredLineStyle(0x00FF, 2))
            return lane_line

        self.lane_markings = make_compound([make_lane_line(line) for line in iter_lane_lines(road.outbound_lanes_bounds)] + [make_lane_line(line) for line in iter_lane_lines(road.inbound_lanes_bounds)], transform=self.transform)

        self.pelican_crossing_view = None

    def set_pelican_crossing(self, pelican_crossing):
        self.pelican_crossing_view = PelicanCrossingView(pelican_crossing)
        for item in self.pelican_crossing_view.items():
            item.add_attr(self.transform)

    def markings(self):
        yield from [self.centre_markings, self.lane_markings, self.edge_markings]


class RoadMapView:
    def __init__(self, road_map):
        self.major_road_view = RoadView(road_map.major_road)
        self.minor_road_views = [RoadView(minor_road) for minor_road in road_map.minor_roads] if road_map.minor_roads is not None else list()

    def markings(self):
        yield from self.major_road_view.markings()
        for minor_road_view in self.minor_road_views:
            yield from minor_road_view.markings()


class RoadEnvViewer(rendering.Viewer):
    def __init__(self, width, height, road_map, actors):
        super().__init__(width, height)
        self.transform.set_translation(0.0, self.height / 2.0)  # Specify that (0, 0) should be centre-left of viewer (default is bottom-left)

        self.road_map_view = RoadMapView(road_map)

        for marking in self.road_map_view.markings():
            self.add_geom(marking)

        self.actor_views = list()
        for actor in actors:
            if isinstance(actor, Car):
                car_view = CarView(actor)
                self.actor_views.append(car_view)
            elif isinstance(actor, Bus):
                bus_view = VehicleView(actor)
                self.actor_views.append(bus_view)
            elif isinstance(actor, Bicycle):
                bicycle_view = BicycleView(actor)
                self.actor_views.append(bicycle_view)
            elif isinstance(actor, Pedestrian):
                pedestrian_view = PedestrianView(actor)
                self.actor_views.append(pedestrian_view)
            elif isinstance(actor, PelicanCrossing):
                self.road_map_view.major_road_view.set_pelican_crossing(actor)
                self.actor_views.append(self.road_map_view.major_road_view.pelican_crossing_view)

        if self.road_map_view.major_road_view.pelican_crossing_view is not None:
            for item in self.road_map_view.major_road_view.pelican_crossing_view.items():
                self.add_geom(item)
            self.add_geom(self.road_map_view.major_road_view.edge_markings)  # Need to redraw on top of pelican crossing

        self.dynamic_actor_views = [actor_view for actor_view in self.actor_views if isinstance(actor_view, DynamicActorView)]

        for dynamic_actor_view in self.dynamic_actor_views:
            self.add_geom(dynamic_actor_view.hard_braking)
            self.add_geom(dynamic_actor_view.normal_braking)

        self.vehicle_views = [dynamic_actor_view for dynamic_actor_view in self.dynamic_actor_views if isinstance(dynamic_actor_view, VehicleView)]

        for vehicle_view in self.vehicle_views:
            self.add_geom(vehicle_view.right_indicators)
            self.add_geom(vehicle_view.left_indicators)

        for vehicle_view in self.vehicle_views:
            self.add_geom(vehicle_view.brake_lights)

        for vehicle_view in self.vehicle_views:
            self.add_geom(vehicle_view.headlights)

        for dynamic_actor_view in self.dynamic_actor_views:
            self.add_geom(dynamic_actor_view.body)

        self.car_views = [vehicle_view for vehicle_view in self.vehicle_views if isinstance(vehicle_view, CarView)]

        for car_view in self.car_views:
            self.add_geom(car_view.roof)

        self.bicycle_views = [dynamic_actor_view for dynamic_actor_view in self.dynamic_actor_views if isinstance(dynamic_actor_view, BicycleView)]

        for bicycle_view in self.bicycle_views:
            self.add_geom(bicycle_view.head)

        self.pedestrian_views = [dynamic_actor_view for dynamic_actor_view in self.dynamic_actor_views if isinstance(dynamic_actor_view, PedestrianView)]

        for pedestrian_view in self.pedestrian_views:
            self.add_geom(pedestrian_view.head)

    def update(self, actors):
        for actor_view, actor in zip(self.actor_views, actors):
            actor_view.update(actor)
