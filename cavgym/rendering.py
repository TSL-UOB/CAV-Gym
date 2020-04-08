from enum import Enum

from gym.envs.classic_control import rendering

from cavgym import environment


class LightState(Enum):
    OFF = 0
    DIM = 1
    FULL = 2


class ActorView:
    def update(self, actor):
        raise NotImplementedError


class VehicleView(ActorView):
    def __init__(self, vehicle):
        rear, right, front, left = vehicle.bounds
        light_offset = vehicle.constants.width * 0.25
        roof_offset = vehicle.constants.length * 0.25

        self.scale = {
            LightState.OFF: (0, 0),
            LightState.DIM: (0.75, 0.75),
            LightState.FULL: (1, 1)
        }

        self.body = rendering.make_polygon([(front, right), (front, left), (rear, left), (rear, right)])
        self.transform = rendering.Transform(translation=(vehicle.state.position.x, vehicle.state.position.y), rotation=vehicle.state.orientation)
        self.body.add_attr(self.transform)

        self.roof = rendering.make_polygon([(front - roof_offset, right), (front - roof_offset, left), (rear + roof_offset, left), (rear + roof_offset, right)])
        self.roof.set_color(0.5, 0.5, 0.5)
        self.roof.add_attr(self.transform)

        front_right_indicator = rendering.make_circle(vehicle.constants.width * 0.2)
        rear_right_indicator = rendering.make_circle(vehicle.constants.width * 0.2)
        self.right_indicators = rendering.Compound([front_right_indicator, rear_right_indicator])
        self.right_indicators.set_color(1, 0.75, 0)
        self.front_right_indicator_transform = rendering.Transform(translation=(front - light_offset, right), scale=self.scale[LightState.OFF])
        self.rear_right_indicator_transform = rendering.Transform(translation=(rear + light_offset, right), scale=self.scale[LightState.OFF])
        front_right_indicator.add_attr(self.front_right_indicator_transform)
        rear_right_indicator.add_attr(self.rear_right_indicator_transform)
        self.right_indicators.add_attr(self.transform)

        front_left_indicator = rendering.make_circle(vehicle.constants.width * 0.2)
        rear_left_indicator = rendering.make_circle(vehicle.constants.width * 0.2)
        self.left_indicators = rendering.Compound([front_left_indicator, rear_left_indicator])
        self.left_indicators.set_color(1, 0.75, 0)
        self.front_left_indicator_transform = rendering.Transform(translation=(front - light_offset, left), scale=self.scale[LightState.OFF])
        self.rear_left_indicator_transform = rendering.Transform(translation=(rear + light_offset, left), scale=self.scale[LightState.OFF])
        front_left_indicator.add_attr(self.front_left_indicator_transform)
        rear_left_indicator.add_attr(self.rear_left_indicator_transform)
        self.left_indicators.add_attr(self.transform)

        brake_light_left = rendering.make_circle(vehicle.constants.width * 0.2)
        brake_light_right = rendering.make_circle(vehicle.constants.width * 0.2)
        self.brake_lights = rendering.Compound([brake_light_left, brake_light_right])
        self.brake_lights.set_color(1, 0, 0)
        self.brake_light_left_transform = rendering.Transform(translation=(rear, left - light_offset), scale=self.scale[LightState.OFF])
        self.brake_light_right_transform = rendering.Transform(translation=(rear, right + light_offset), scale=self.scale[LightState.OFF])
        brake_light_left.add_attr(self.brake_light_left_transform)
        brake_light_right.add_attr(self.brake_light_right_transform)
        self.brake_lights.add_attr(self.transform)

        headlight_left = rendering.make_circle(vehicle.constants.width * 0.2)
        headlight_right = rendering.make_circle(vehicle.constants.width * 0.2)
        self.headlights = rendering.Compound([headlight_left, headlight_right])
        self.headlights.set_color(1, 1, 0)
        self.headlight_left_transform = rendering.Transform(translation=(front, left - light_offset), scale=self.scale[LightState.OFF])
        self.headlight_right_transform = rendering.Transform(translation=(front, right + light_offset), scale=self.scale[LightState.OFF])
        headlight_left.add_attr(self.headlight_left_transform)
        headlight_right.add_attr(self.headlight_right_transform)
        self.headlights.add_attr(self.transform)

    def set_right_indicators(self, state):
        self.front_right_indicator_transform.set_scale(*self.scale[state])
        self.rear_right_indicator_transform.set_scale(*self.scale[state])

    def set_left_indicators(self, state):
        self.front_left_indicator_transform.set_scale(*self.scale[state])
        self.rear_left_indicator_transform.set_scale(*self.scale[state])

    def set_brake_lights(self, state):
        self.brake_light_left_transform.set_scale(*self.scale[state])
        self.brake_light_right_transform.set_scale(*self.scale[state])

    def set_headlights(self, state):
        self.headlight_left_transform.set_scale(*self.scale[state])
        self.headlight_right_transform.set_scale(*self.scale[state])

    def update(self, vehicle):
        self.transform.set_translation(vehicle.state.position.x, vehicle.state.position.y)
        self.transform.set_rotation(vehicle.state.orientation)

        if vehicle.state.angular_velocity > 0:
            if vehicle.state.angular_velocity == vehicle.constants.normal_left_turn:
                self.set_left_indicators(LightState.DIM)
            else:
                self.set_left_indicators(LightState.FULL)
            self.set_right_indicators(LightState.OFF)
        elif vehicle.state.angular_velocity < 0:
            if vehicle.state.angular_velocity == vehicle.constants.normal_right_turn:
                self.set_right_indicators(LightState.DIM)
            else:
                self.set_right_indicators(LightState.FULL)
            self.set_left_indicators(LightState.OFF)
        else:
            self.set_left_indicators(LightState.OFF)
            self.set_right_indicators(LightState.OFF)

        if vehicle.state.acceleration < 0:
            if vehicle.state.acceleration == vehicle.constants.normal_deceleration:
                self.set_brake_lights(LightState.DIM)
            else:
                self.set_brake_lights(LightState.FULL)
            self.set_headlights(LightState.OFF)
        elif vehicle.state.acceleration > 0:
            if vehicle.state.acceleration == vehicle.constants.normal_acceleration:
                self.set_headlights(LightState.DIM)
            else:
                self.set_headlights(LightState.FULL)
            self.set_brake_lights(LightState.OFF)
        else:
            self.set_brake_lights(LightState.OFF)
            self.set_headlights(LightState.OFF)


class PedestrianView(ActorView):
    def __init__(self, pedestrian):
        rear, right, front, left = pedestrian.bounds

        self.body = rendering.make_polygon([(front, right), (front, left), (rear, left), (rear, right)])
        self.transform = rendering.Transform(translation=(pedestrian.state.position.x, pedestrian.state.position.y), rotation=pedestrian.state.orientation)
        self.body.add_attr(self.transform)

        self.head = rendering.make_circle(pedestrian.constants.length * 0.3)
        self.head.set_color(0.5, 0.5, 0.5)
        self.head.add_attr(self.transform)

    def update(self, pedestrian):
        self.transform.set_translation(pedestrian.state.position.x, pedestrian.state.position.y)
        self.transform.set_rotation(pedestrian.state.orientation)


class TrafficLightView(ActorView):
    def __init__(self, traffic_light):
        rear, right, front, left = traffic_light.bounds

        self.body = rendering.make_polygon([(front, right), (front, left), (rear, left), (rear, right)])
        self.transform = rendering.Transform(translation=(traffic_light.constants.position.x, traffic_light.constants.position.y), rotation=traffic_light.constants.orientation)
        self.body.add_attr(self.transform)

        self.red_light = rendering.make_circle(traffic_light.constants.width * 0.25)
        self.red_light.set_color(1, 0, 0)
        self.red_light_transform = rendering.Transform(translation=(0, traffic_light.constants.height * 0.25))
        self.red_light.add_attr(self.red_light_transform)
        self.red_light.add_attr(self.transform)

        self.amber_light = rendering.make_circle(traffic_light.constants.width * 0.25)
        self.amber_light.set_color(1, 0.75, 0)
        self.amber_light_transform = rendering.Transform(translation=(0, 0))
        self.amber_light.add_attr(self.amber_light_transform)
        self.amber_light.add_attr(self.transform)

        self.green_light = rendering.make_circle(traffic_light.constants.width * 0.25)
        self.green_light.set_color(0, 1, 0)
        self.green_light_transform = rendering.Transform(translation=(0, -traffic_light.constants.height * 0.25))
        self.green_light.add_attr(self.green_light_transform)
        self.green_light.add_attr(self.transform)

    def set_red_light(self):
        self.red_light_transform.set_scale(1, 1)
        self.amber_light_transform.set_scale(0, 0)
        self.green_light_transform.set_scale(0, 0)

    def set_amber_light(self):
        self.red_light_transform.set_scale(0, 0)
        self.amber_light_transform.set_scale(1, 1)
        self.green_light_transform.set_scale(0, 0)

    def set_green_light(self):
        self.red_light_transform.set_scale(0, 0)
        self.amber_light_transform.set_scale(0, 0)
        self.green_light_transform.set_scale(1, 1)

    def update(self, traffic_light):
        if traffic_light.state is environment.TrafficLightState.RED:
            self.set_red_light()
        elif traffic_light.state is environment.TrafficLightState.AMBER:
            self.set_amber_light()
        elif traffic_light.state is environment.TrafficLightState.GREEN:
            self.set_green_light()


class RoadView:
    def __init__(self, width, y_values):
        lines = [rendering.make_polyline([(0, y_value), (width, y_value)]) for y_value in y_values]
        for line in lines[1:-1]:
            line.add_attr(rendering.LineStyle(0x0FF0))
        self.markings = rendering.Compound(lines)


class RoadEnvViewer(rendering.Viewer):
    def __init__(self, width, height, actors):
        super().__init__(width, height)

        self.road_view = RoadView(self.width, [self.height * 0.25, self.height * 0.5, self.height * 0.75])
        self.add_geom(self.road_view.markings)

        self.actor_views = list()
        self.vehicle_views = list()
        self.traffic_light_views = list()
        self.pedestrian_views = list()
        for actor in actors:
            if isinstance(actor, environment.Vehicle):
                vehicle_view = VehicleView(actor)
                self.actor_views.append(vehicle_view)
                self.vehicle_views.append(vehicle_view)
            elif isinstance(actor, environment.TrafficLight):
                traffic_light_view = TrafficLightView(actor)
                self.actor_views.append(traffic_light_view)
                self.traffic_light_views.append(traffic_light_view)
            elif isinstance(actor, environment.Pedestrian):
                pedestrian_view = PedestrianView(actor)
                self.actor_views.append(pedestrian_view)
                self.pedestrian_views.append(pedestrian_view)

        for traffic_light_view in self.traffic_light_views:
            self.add_geom(traffic_light_view.body)
            self.add_geom(traffic_light_view.red_light)
            self.add_geom(traffic_light_view.amber_light)
            self.add_geom(traffic_light_view.green_light)

        for vehicle_view in self.vehicle_views:
            self.add_geom(vehicle_view.right_indicators)
            self.add_geom(vehicle_view.left_indicators)

        for vehicle_view in self.vehicle_views:
            self.add_geom(vehicle_view.brake_lights)

        for vehicle_view in self.vehicle_views:
            self.add_geom(vehicle_view.headlights)

        # Add bodies/roofs last to ensure they are not occluded by lights of other vehicles
        for vehicle_view in self.vehicle_views:
            self.add_geom(vehicle_view.body)
            self.add_geom(vehicle_view.roof)

        for pedestrian_view in self.pedestrian_views:
            self.add_geom(pedestrian_view.body)
            self.add_geom(pedestrian_view.head)

    def update(self, actors):
        for actor_view, actor in zip(self.actor_views, actors):
            actor_view.update(actor)
