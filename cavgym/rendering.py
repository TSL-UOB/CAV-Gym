from enum import Enum

from gym.envs.classic_control import rendering

from cavgym import mods
from cavgym.actors import TrafficLightState, Car, Bus, Bicycle, Pedestrian, PelicanCrossing
from cavgym.geometry import Point


class RGB(Enum):
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 1)
    GREEN = (0, 1, 0)
    RED = (1, 0, 0)
    CYAN = (0, 1, 1)
    MAGENTA = (1, 0, 1)
    YELLOW = (1, 1, 0)
    WHITE = (1, 1, 1)


class BulbState(Enum):
    OFF = 0
    DIM = 1
    FULL = 2


class OcclusionView:
    def __init__(self, occlusion, ego, **kwargs):
        self.occlusion_zone = None

        if occlusion is not ego:
            self.occlusion_zone = rendering.make_polygon(list(occlusion.occlusion_zone(ego.state.position)), filled=False)
            self.occlusion_zone.set_color(*RGB.RED.value)

        super().__init__(**kwargs)  # important to pass on kwargs if class is used as superclass in multiple inheritance

    def update_occlusion_zone(self, occlusion, ego):
        if occlusion is not ego:
            self.occlusion_zone.v = list(occlusion.occlusion_zone(ego.state.position))


class ActorView:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # important to pass on kwargs if class is used as superclass in multiple inheritance

    def update(self, actor, ego):
        raise NotImplementedError


class DynamicActorView(ActorView, OcclusionView):
    def __init__(self, dynamic_actor, ego, road):
        super().__init__(occlusion=dynamic_actor, ego=ego)

        self.body = rendering.make_polygon(list(dynamic_actor.bounding_box()))
        if dynamic_actor is ego:
            self.body.set_color(*RGB.RED.value)

        braking_relative_bounding_box, reaction_relative_bounding_box = dynamic_actor.stopping_zones()
        self.braking = rendering.make_polygon(list(braking_relative_bounding_box), filled=False)
        self.braking.set_color(*RGB.GREEN.value)
        self.reaction = rendering.make_polygon(list(reaction_relative_bounding_box), filled=False)
        self.reaction.set_color(*RGB.BLUE.value)

        self.focal_road = road
        if dynamic_actor.bounding_box().intersects(self.focal_road.bounding_box()):
            self.road_angle = rendering.make_polyline(list())
        else:
            self.road_angle = rendering.make_polyline(list(dynamic_actor.line_anchor(self.focal_road)))
        self.road_angle.set_color(*RGB.MAGENTA.value)

    def update(self, dynamic_actor, ego):
        self.body.v = list(dynamic_actor.bounding_box())

        braking_relative_bounding_box, reaction_relative_bounding_box = dynamic_actor.stopping_zones()
        self.braking.v = list(braking_relative_bounding_box)
        self.reaction.v = list(reaction_relative_bounding_box)

        self.update_occlusion_zone(dynamic_actor, ego)

        if dynamic_actor.bounding_box().intersects(self.focal_road.bounding_box()):
            self.road_angle.v = list()
        else:
            self.road_angle.v = list(dynamic_actor.line_anchor(self.focal_road))

    def geoms(self):
        yield from [self.body, self.braking, self.reaction, self.road_angle]
        if self.occlusion_zone is not None:
            yield self.occlusion_zone


class VehicleView(DynamicActorView):
    def __init__(self, vehicle, ego, road):
        super().__init__(vehicle, ego, road)

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
        self.brake_lights.set_color(*RGB.RED.value)
        self.headlights = self.make_lights(longitudinal_bounding_box.front_left, longitudinal_bounding_box.front_right, BulbState.FULL)
        self.headlights.set_color(*RGB.YELLOW.value)

    def make_lights(self, position1, position2, state):
        return rendering.Compound([
            mods.make_circle(*position1, self.scale[state]),
            mods.make_circle(*position2, self.scale[state])
        ])

    def update(self, vehicle, ego):
        super().update(vehicle, ego)

        left_indicator_state = BulbState.OFF
        right_indicator_state = BulbState.OFF
        if vehicle.state.angular_velocity > 0:
            left_indicator_state = BulbState.FULL
        elif vehicle.state.angular_velocity < 0:
            right_indicator_state = BulbState.FULL

        indicator_bounding_box = vehicle.indicators()
        self.left_indicators.gs = self.make_lights(indicator_bounding_box.rear_left, indicator_bounding_box.front_left, left_indicator_state).gs
        self.right_indicators.gs = self.make_lights(indicator_bounding_box.rear_right, indicator_bounding_box.front_right, right_indicator_state).gs

        brake_lights_state = BulbState.OFF
        headlights_state = BulbState.OFF
        if vehicle.state.acceleration < 0:
            brake_lights_state = BulbState.FULL
        elif vehicle.state.acceleration > 0:
            headlights_state = BulbState.FULL

        longitudinal_bounding_box = vehicle.longitudinal_lights()
        self.brake_lights.gs = self.make_lights(longitudinal_bounding_box.rear_left, longitudinal_bounding_box.rear_right, brake_lights_state).gs
        self.headlights.gs = self.make_lights(longitudinal_bounding_box.front_left, longitudinal_bounding_box.front_right, headlights_state).gs


class CarView(VehicleView):
    def __init__(self, car, ego, road):
        super().__init__(car, ego, road)

        self.roof = rendering.make_polygon(list(car.roof()))
        if car is ego:
            self.roof.set_color(0.5, 0, 0)
        else:
            self.roof.set_color(0.5, 0.5, 0.5)

    def update(self, car, ego):
        super().update(car, ego)

        self.roof.v = list(car.roof())


class BicycleView(DynamicActorView):
    def __init__(self, bicycle, ego, road):
        super().__init__(bicycle, ego, road)

        self.head = self.make_head(bicycle)

    def make_head(self, bicycle):
        head = mods.make_circle(*bicycle.state.position, bicycle.constants.width * 0.3)
        head.set_color(0.5, 0.5, 0.5)
        return head

    def update(self, bicycle, ego):
        super().update(bicycle, ego)

        self.head.v = self.make_head(bicycle).v


class PedestrianView(DynamicActorView):
    def __init__(self, pedestrian, ego, road):
        super().__init__(pedestrian, ego, road)

        self.head = self.make_head(pedestrian)

    def make_head(self, pedestrian):
        head = mods.make_circle(*pedestrian.state.position, pedestrian.constants.length * 0.4)
        head.set_color(0.5, 0.5, 0.5)
        return head

    def update(self, pedestrian, ego):
        super().update(pedestrian, ego)

        self.head.v = self.make_head(pedestrian).v


class TrafficLightView(ActorView, OcclusionView):
    def __init__(self, traffic_light, ego):
        super().__init__(occlusion=traffic_light, ego=ego)

        self.body = rendering.make_polygon(list(traffic_light.static_bounding_box))

        self.red_light = mods.make_circle(*traffic_light.red_light, traffic_light.constants.width * 0.25)
        self.amber_light = mods.make_circle(*traffic_light.amber_light, traffic_light.constants.width * 0.25)
        self.green_light = mods.make_circle(*traffic_light.green_light, traffic_light.constants.width * 0.25)

        self.set_green_light()

    def set_red_light(self):
        self.red_light.set_color(*RGB.RED.value)
        for light in [self.amber_light, self.green_light]:
            light.set_color(*RGB.BLACK.value)

    def set_amber_light(self):
        self.amber_light.set_color(1, 0.75, 0)
        for light in [self.red_light, self.green_light]:
            light.set_color(*RGB.BLACK.value)

    def set_green_light(self):
        self.green_light.set_color(*RGB.GREEN.value)
        for light in [self.red_light, self.amber_light]:
            light.set_color(*RGB.BLACK.value)

    def update(self, traffic_light, ego):
        if traffic_light.state is TrafficLightState.RED:
            self.set_red_light()
        elif traffic_light.state is TrafficLightState.AMBER:
            self.set_amber_light()
        elif traffic_light.state is TrafficLightState.GREEN:
            self.set_green_light()

        self.update_occlusion_zone(traffic_light, ego)

    def geoms(self):
        yield from [self.body, self.red_light, self.amber_light, self.green_light, self.occlusion_zone]


class PelicanCrossingView(ActorView):
    def __init__(self, pelican_crossing, ego):
        coordinates = pelican_crossing.bounding_box()
        self.area = rendering.make_polygon(list(coordinates))
        self.area.set_color(*RGB.WHITE.value)

        self.markings = rendering.Compound([
            rendering.make_polyline([tuple(pelican_crossing.outbound_intersection_bounding_box.rear_left), tuple(pelican_crossing.outbound_intersection_bounding_box.rear_right)]),
            rendering.make_polyline([tuple(pelican_crossing.inbound_intersection_bounding_box.front_left), tuple(pelican_crossing.inbound_intersection_bounding_box.front_right)]),
            rendering.make_polyline([tuple(pelican_crossing.static_bounding_box.rear_left), tuple(pelican_crossing.static_bounding_box.front_left)]),
            rendering.make_polyline([tuple(pelican_crossing.static_bounding_box.rear_right), tuple(pelican_crossing.static_bounding_box.front_right)])
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

        self.outbound_traffic_light_view = TrafficLightView(pelican_crossing.outbound_traffic_light, ego)
        self.inbound_traffic_light_view = TrafficLightView(pelican_crossing.inbound_traffic_light, ego)

    def update(self, pelican_crossing, ego):
        for traffic_light, traffic_light_view in zip([pelican_crossing.outbound_traffic_light, pelican_crossing.inbound_traffic_light], [self.outbound_traffic_light_view, self.inbound_traffic_light_view]):
            traffic_light_view.update(traffic_light, ego)

    def geoms(self):
        yield from [self.area, self.markings, self.inner]
        yield from self.outbound_traffic_light_view.geoms()
        yield from self.inbound_traffic_light_view.geoms()


class RoadView:
    def __init__(self, road):
        self.area = rendering.make_polygon(list(road.static_bounding_box))
        self.area.set_color(*RGB.WHITE.value)

        coordinates = road.bounding_box()
        self.edge_markings = rendering.Compound([
            rendering.make_polyline([tuple(coordinates.rear_left), tuple(coordinates.front_left)]),
            rendering.make_polyline([tuple(coordinates.rear_right), tuple(coordinates.front_right)])
        ])

        outbound_coordinates = road.outbound.bounding_box()
        self.centre_markings = rendering.make_polyline([tuple(outbound_coordinates.rear_right), tuple(outbound_coordinates.front_right)])
        self.centre_markings.add_attr(mods.FactoredLineStyle(0x00FF, 2))

        lane_lines = list()
        for lane in road.outbound.lanes[:-1] + road.inbound.lanes[:-1]:
            lane_coordinates = lane.bounding_box()
            lane_line = rendering.make_polyline([tuple(lane_coordinates.rear_right), tuple(lane_coordinates.front_right)])
            lane_lines.append(lane_line)
        self.lane_markings = rendering.Compound(lane_lines)
        self.lane_markings.add_attr(mods.FactoredLineStyle(0x00FF, 2))

        self.pelican_crossing_view = None

        self.bus_stop_views = list()
        for direction in [road.outbound, road.inbound]:
            if direction.bus_stop is not None:
                markings = rendering.make_polygon(list(direction.bus_stop.static_bounding_box), filled=False)
                self.bus_stop_views.append(markings)

    def set_pelican_crossing(self, pelican_crossing, ego):
        self.pelican_crossing_view = PelicanCrossingView(pelican_crossing, ego)

    def geoms(self):
        yield from [self.area, self.centre_markings, self.lane_markings, self.edge_markings]
        if self.bus_stop_views is not None:
            yield from self.bus_stop_views


class ObstacleView(OcclusionView):
    def __init__(self, obstacle, ego):
        super().__init__(obstacle, ego)

        self.body = rendering.make_polygon(list(obstacle.static_bounding_box))

    def geoms(self):
        yield from [self.body, self.occlusion_zone]

    def update(self, obstacle, ego):
        self.update_occlusion_zone(obstacle, ego)


class RoadMapView:
    def __init__(self, road_map, actor):
        self.major_road_view = RoadView(road_map.major_road)
        self.minor_road_views = [RoadView(minor_road) for minor_road in road_map.minor_roads] if road_map.minor_roads is not None else list()

        if self.minor_road_views:
            self.clear_intersections = rendering.Compound([rendering.make_polyline([tuple(bounding_box.front_left), tuple(bounding_box.front_right)]) for bounding_box in road_map.intersection_bounding_boxes])
            self.clear_intersections.set_color(*RGB.WHITE.value)

            self.intersection_markings = rendering.Compound([rendering.make_polyline([tuple(bounding_box.rear_left), tuple(bounding_box.rear_right)]) for bounding_box in road_map.inbound_intersection_bounding_boxes])
            self.intersection_markings.add_attr(mods.FactoredLineStyle(0x0F0F, 2))

        self.obstacle_view = None
        if road_map.obstacle is not None:
            self.obstacle_view = ObstacleView(road_map.obstacle, actor)

    def geoms(self):
        for minor_road_view in self.minor_road_views:
            yield from minor_road_view.geoms()
        yield from self.major_road_view.geoms()
        if self.minor_road_views:
            yield from [self.clear_intersections, self.intersection_markings]
        if self.obstacle_view:
            yield from self.obstacle_view.geoms()


class RoadEnvViewer(rendering.Viewer):
    def __init__(self, width, height, road_map, actors, ego):
        super().__init__(width, height)
        self.road_map = road_map

        self.transform.set_translation(0.0, self.height / 2.0)  # Specify that (0, 0) should be centre-left of viewer (default is bottom-left)

        self.road_map_view = RoadMapView(road_map, ego)

        for geom in self.road_map_view.geoms():
            self.add_geom(geom)

        self.actor_views = list()
        for actor in actors:
            if isinstance(actor, Car):
                car_view = CarView(actor, ego, road_map.major_road)
                self.actor_views.append(car_view)
            elif isinstance(actor, Bus):
                bus_view = VehicleView(actor, ego, road_map.major_road)
                self.actor_views.append(bus_view)
            elif isinstance(actor, Bicycle):
                bicycle_view = BicycleView(actor, ego, road_map.major_road)
                self.actor_views.append(bicycle_view)
            elif isinstance(actor, Pedestrian):
                pedestrian_view = PedestrianView(actor, ego, road_map.major_road)
                self.actor_views.append(pedestrian_view)
            elif isinstance(actor, PelicanCrossing):
                self.road_map_view.major_road_view.set_pelican_crossing(actor, ego)
                self.actor_views.append(self.road_map_view.major_road_view.pelican_crossing_view)

        if self.road_map_view.major_road_view.pelican_crossing_view is not None:
            for geom in self.road_map_view.major_road_view.pelican_crossing_view.geoms():
                self.add_geom(geom)

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

    def update(self, actors, ego):
        for actor_view, actor in zip(self.actor_views, actors):
            actor_view.update(actor, ego)

        if self.road_map_view.obstacle_view is not None:
            self.road_map_view.obstacle_view.update(self.road_map.obstacle, ego)
