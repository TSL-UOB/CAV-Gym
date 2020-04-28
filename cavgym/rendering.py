from enum import Enum

from gym.envs.classic_control import rendering

from cavgym import mods
from cavgym.actors import TrafficLightState, Car, Bus, Bicycle, Pedestrian, PelicanCrossing
from cavgym.utilities import Point


class BulbState(Enum):
    OFF = 0
    DIM = 1
    FULL = 2


class ActorView:
    def update(self, actor):
        raise NotImplementedError


class DynamicActorView(ActorView):
    def __init__(self, dynamic_actor):
        self.body = rendering.make_polygon(list(dynamic_actor.bounding_box()))

        hard_braking_relative_bounding_box, reaction_relative_bounding_box = dynamic_actor.stopping_zones()
        self.hard_braking = rendering.make_polygon(list(hard_braking_relative_bounding_box))
        self.hard_braking.set_color(0.9, 0.9, 0.9)
        self.reaction = rendering.make_polygon(list(reaction_relative_bounding_box))
        self.reaction.set_color(0.95, 0.95, 0.95)

    def update(self, dynamic_actor):
        self.body.v = list(dynamic_actor.bounding_box())

        hard_braking_relative_bounding_box, reaction_relative_bounding_box = dynamic_actor.stopping_zones()
        self.hard_braking.v = list(hard_braking_relative_bounding_box)
        self.reaction.v = list(reaction_relative_bounding_box)

    def geoms(self):
        yield from [self.hard_braking, self.reaction, self.body]


class VehicleView(DynamicActorView):
    def __init__(self, vehicle):
        super().__init__(vehicle)

        self.scale = {
            BulbState.OFF: 0.0,
            BulbState.DIM: vehicle.constants.width * 0.15,
            BulbState.FULL: vehicle.constants.width * 0.2
        }

        indicator_bounding_box = vehicle.indicators()
        self.left_indicators = self.make_lights(indicator_bounding_box.rear_left, indicator_bounding_box.front_left, BulbState.FULL)
        self.left_indicators.set_color(1, 0.75, 0)
        self.right_indicators = self.make_lights(indicator_bounding_box.rear_right, indicator_bounding_box.front_right, BulbState.FULL)
        self.right_indicators.set_color(1, 0.75, 0)

        longitudinal_bounding_box = vehicle.longitudinal_lights()
        self.brake_lights = self.make_lights(longitudinal_bounding_box.rear_left, longitudinal_bounding_box.rear_right, BulbState.FULL)
        self.brake_lights.set_color(1, 0, 0)
        self.headlights = self.make_lights(longitudinal_bounding_box.front_left, longitudinal_bounding_box.front_right, BulbState.FULL)
        self.headlights.set_color(1, 1, 0)

    def make_lights(self, position1, position2, state):
        return rendering.Compound([
            mods.make_circle(*position1, self.scale[state]),
            mods.make_circle(*position2, self.scale[state])
        ])

    def update(self, vehicle):
        super().update(vehicle)

        left_indicator_state = BulbState.OFF
        right_indicator_state = BulbState.OFF
        if vehicle.state.angular_velocity > 0:
            left_indicator_state = BulbState.DIM if vehicle.state.angular_velocity == vehicle.constants.normal_left_turn else BulbState.FULL
        elif vehicle.state.angular_velocity < 0:
            right_indicator_state = BulbState.DIM if vehicle.state.angular_velocity == vehicle.constants.normal_right_turn else BulbState.FULL

        indicator_bounding_box = vehicle.indicators()
        self.left_indicators.gs = self.make_lights(indicator_bounding_box.rear_left, indicator_bounding_box.front_left, left_indicator_state).gs
        self.right_indicators.gs = self.make_lights(indicator_bounding_box.rear_right, indicator_bounding_box.front_right, right_indicator_state).gs

        brake_lights_state = BulbState.OFF
        headlights_state = BulbState.OFF
        if vehicle.state.acceleration < 0:
            brake_lights_state = BulbState.DIM if vehicle.state.acceleration == vehicle.constants.normal_deceleration else BulbState.FULL
        elif vehicle.state.acceleration > 0:
            headlights_state = BulbState.DIM if vehicle.state.acceleration == vehicle.constants.normal_acceleration else BulbState.FULL

        longitudinal_bounding_box = vehicle.longitudinal_lights()
        self.brake_lights.gs = self.make_lights(longitudinal_bounding_box.rear_left, longitudinal_bounding_box.rear_right, brake_lights_state).gs
        self.headlights.gs = self.make_lights(longitudinal_bounding_box.front_left, longitudinal_bounding_box.front_right, headlights_state).gs


class CarView(VehicleView):
    def __init__(self, car):
        super().__init__(car)

        self.roof = rendering.make_polygon(list(car.roof()))
        self.roof.set_color(0.5, 0.5, 0.5)

    def update(self, car):
        super().update(car)

        self.roof.v = list(car.roof())


class BicycleView(DynamicActorView):
    def __init__(self, bicycle):
        super().__init__(bicycle)

        self.head = self.make_head(bicycle)

    def make_head(self, bicycle):
        head = mods.make_circle(*bicycle.state.position, bicycle.constants.width * 0.3)
        head.set_color(0.5, 0.5, 0.5)
        return head

    def update(self, bicycle):
        super().update(bicycle)

        self.head.v = self.make_head(bicycle).v


class PedestrianView(DynamicActorView):
    def __init__(self, pedestrian):
        super().__init__(pedestrian)

        self.head = self.make_head(pedestrian)

    def make_head(self, pedestrian):
        head = mods.make_circle(*pedestrian.state.position, pedestrian.constants.length * 0.4)
        head.set_color(0.5, 0.5, 0.5)
        return head

    def update(self, pedestrian):
        super().update(pedestrian)

        self.head.v = self.make_head(pedestrian).v


class TrafficLightView(ActorView):
    def __init__(self, traffic_light):
        self.body = rendering.make_polygon(list(traffic_light.static_bounding_box))

        self.red_light = mods.make_circle(*traffic_light.red_light, traffic_light.width * 0.25)
        self.amber_light = mods.make_circle(*traffic_light.amber_light, traffic_light.width * 0.25)
        self.green_light = mods.make_circle(*traffic_light.green_light, traffic_light.width * 0.25)

        self.set_green_light()

    def set_red_light(self):
        self.red_light.set_color(1, 0, 0)
        for light in [self.amber_light, self.green_light]:
            light.set_color(0, 0, 0)

    def set_amber_light(self):
        self.amber_light.set_color(1, 0.75, 0)
        for light in [self.red_light, self.green_light]:
            light.set_color(0, 0, 0)

    def set_green_light(self):
        self.green_light.set_color(0, 1, 0)
        for light in [self.red_light, self.amber_light]:
            light.set_color(0, 0, 0)

    def update(self, traffic_light):
        if traffic_light.state is TrafficLightState.RED:
            self.set_red_light()
        elif traffic_light.state is TrafficLightState.AMBER:
            self.set_amber_light()
        elif traffic_light.state is TrafficLightState.GREEN:
            self.set_green_light()

    def geoms(self):
        yield from [self.body, self.red_light, self.amber_light, self.green_light]


class PelicanCrossingView(ActorView):
    def __init__(self, pelican_crossing):
        coordinates = pelican_crossing.bounding_box()
        self.area = rendering.make_polygon(list(coordinates))
        self.area.set_color(1, 1, 1)

        self.markings = rendering.Compound([
            rendering.make_polyline([tuple(pelican_crossing.outbound_intersection_bounding_box.rear_left), tuple(pelican_crossing.outbound_intersection_bounding_box.rear_right)]),
            rendering.make_polyline([tuple(pelican_crossing.inbound_intersection_bounding_box.front_left), tuple(pelican_crossing.inbound_intersection_bounding_box.front_right)]),
        ])

        offset_rear_right = Point(coordinates.rear_right.x + (pelican_crossing.constants.width * 0.15), coordinates.rear_right.y)
        offset_rear_left = Point(coordinates.rear_left.x + (pelican_crossing.constants.width * 0.15), coordinates.rear_left.y)
        offset_front_right = Point(coordinates.front_right.x - (pelican_crossing.constants.width * 0.15), coordinates.front_right.y)
        offset_front_left = Point(coordinates.front_left.x - (pelican_crossing.constants.width * 0.15), coordinates.front_left.y)
        self.inner = rendering.Compound([
            rendering.make_polyline([tuple(offset_rear_right), tuple(offset_rear_left)]),
            rendering.make_polyline([tuple(offset_front_right), tuple(offset_front_left)])
        ])
        self.inner.add_attr(mods.FactoredLineStyle(0x0F0F, 1))

        self.outbound_traffic_light_view = TrafficLightView(pelican_crossing.outbound_traffic_light)
        self.inbound_traffic_light_view = TrafficLightView(pelican_crossing.inbound_traffic_light)

    def update(self, pelican_crossing):
        for traffic_light_view in [self.outbound_traffic_light_view, self.inbound_traffic_light_view]:
            traffic_light_view.update(pelican_crossing)

    def geoms(self):
        yield from [self.area, self.markings, self.inner]
        yield from self.outbound_traffic_light_view.geoms()
        yield from self.inbound_traffic_light_view.geoms()


class RoadView:
    def __init__(self, road):
        coordinates = road.bounding_box()
        self.edge_markings = rendering.Compound([
            rendering.make_polyline([tuple(coordinates.rear_left), tuple(coordinates.front_left)]),
            rendering.make_polyline([tuple(coordinates.rear_right), tuple(coordinates.front_right)])
        ])

        outbound_coordinates = road.outbound.bounding_box()
        self.centre_markings = rendering.make_polyline([tuple(outbound_coordinates.rear_right), tuple(outbound_coordinates.front_right)])
        self.centre_markings.add_attr(mods.FactoredLineStyle(0x00FF, 2))

        lane_lines = list()
        for lane in road.outbound.lanes[:-1] + road.inbound.lanes[1:]:
            lane_coordinates = lane.bounding_box()
            lane_line = rendering.make_polyline([tuple(lane_coordinates.rear_right), tuple(lane_coordinates.front_right)])
            lane_lines.append(lane_line)
        self.lane_markings = rendering.Compound(lane_lines)
        self.lane_markings.add_attr(mods.FactoredLineStyle(0x00FF, 2))

        self.pelican_crossing_view = None

    def set_pelican_crossing(self, pelican_crossing):
        self.pelican_crossing_view = PelicanCrossingView(pelican_crossing)

    def geoms(self):
        yield from [self.centre_markings, self.lane_markings, self.edge_markings]


class RoadMapView:
    def __init__(self, road_map):
        self.major_road_view = RoadView(road_map.major_road)
        self.minor_road_views = [RoadView(minor_road) for minor_road in road_map.minor_roads] if road_map.minor_roads is not None else list()

    def geoms(self):
        yield from self.major_road_view.geoms()
        for minor_road_view in self.minor_road_views:
            yield from minor_road_view.geoms()


class RoadEnvViewer(rendering.Viewer):
    def __init__(self, width, height, road_map, actors):
        super().__init__(width, height)
        self.transform.set_translation(0.0, self.height / 2.0)  # Specify that (0, 0) should be centre-left of viewer (default is bottom-left)

        self.road_map_view = RoadMapView(road_map)

        for geom in self.road_map_view.geoms():
            self.add_geom(geom)

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
            for geom in self.road_map_view.major_road_view.pelican_crossing_view.geoms():
                self.add_geom(geom)
            self.add_geom(self.road_map_view.major_road_view.edge_markings)  # Need to redraw on top of pelican crossing

        self.dynamic_actor_views = [actor_view for actor_view in self.actor_views if isinstance(actor_view, DynamicActorView)]

        for dynamic_actor_view in self.dynamic_actor_views:
            for geom in dynamic_actor_view.geoms():
                self.add_geom(geom)

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
