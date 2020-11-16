from abc import ABC

import math

from gym.utils import seeding

import reporting
from library import geometry
from library.actions import TrafficLightAction
from library.bodies import Car, Pedestrian, DynamicBodyState
from library.geometry import Point
from examples.constants import M2PX, car_constants


class Agent(ABC):
    def reset(self):
        raise NotImplementedError

    def choose_action(self, state, action_space, info=None):
        raise NotImplementedError

    def process_feedback(self, previous_state, action, state, reward):
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, epsilon=0.1, np_random=seeding.np_random(None)[0], **kwargs):
        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.np_random = np_random

    def epsilon_valid(self):
        return self.np_random.uniform(0.0, 1.0) < self.epsilon

    def reset(self):
        raise NotImplementedError

    def choose_action(self, state, action_space, info=None):
        raise NotImplementedError

    def process_feedback(self, previous_state, action, state, reward):
        raise NotImplementedError


class RandomTrafficLightAgent(RandomAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self):
        pass

    def choose_action(self, state, action_space, info=None):
        if self.epsilon_valid():
            action = self.np_random.choice([value for value in TrafficLightAction])
        else:
            action = TrafficLightAction.NOOP

        return action.value

    def process_feedback(self, previous_state, action, state, reward):
        pass


class DynamicBodyAgent(Agent):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)

        self.index = index

    def reset(self):
        raise NotImplementedError

    def active_velocity(self, state):
        return None

    def active_orientation(self, state):
        return None

    def choose_action(self, state, action_space, info=None):
        raise NotImplementedError

    def process_feedback(self, previous_state, action, state, reward):
        raise NotImplementedError


class NoopAgent(DynamicBodyAgent):
    def reset(self):
        pass

    def choose_action(self, state, action_space, info=None):
        return [0.0, 0.0]

    def process_feedback(self, previous_state, action, state, reward):
        pass


key_target_velocity = {
    65365: float(car_constants.max_velocity),  # page up
    65366: float((car_constants.max_velocity - car_constants.min_velocity) / 2),  # page down
    65367: float(car_constants.min_velocity),  # end
}

key_target_orientation = {
    65361: math.pi,  # arrow left
    65362: math.pi * 0.5,  # arrow up
    65363: 0.0,  # arrow right
    65364: -(math.pi * 0.5),  # arrow down

    65457: -(math.pi * 0.75),  # numpad 1
    65458: -(math.pi * 0.5),  # numpad 2
    65459: -(math.pi * 0.25),  # numpad 3
    65460: math.pi,  # numpad 4
    65462: 0.0,  # numpad 6
    65463: math.pi * 0.75,  # numpad 7
    65464: math.pi * 0.5,  # numpad 8
    65465: math.pi * 0.25  # numpad 9
}


class KeyboardAgent(DynamicBodyAgent):
    def __init__(self, body, time_resolution, **kwargs):
        super().__init__(**kwargs)

        self.body = body
        self.time_resolution = time_resolution

    def reset(self):
        self.body.target_velocity = None
        self.body.target_orientation = None

    def choose_action(self, state, action_space, info=None):
        self_state = state[self.index]
        self_velocity = self_state[2]
        self_orientation = self_state[3]

        throttle_action = 0.0
        if self.body.target_velocity is not None:
            diff = (self.body.target_velocity - self_velocity) / self.time_resolution

            if diff < self.body.constants.min_throttle:
                throttle_action = self.body.constants.min_throttle
            elif diff > self.body.constants.max_throttle:
                throttle_action = self.body.constants.max_throttle
            else:
                throttle_action = diff

        steering_action = 0.0
        if self_velocity != 0 and self.body.target_orientation is not None:
            turn_angle = math.atan2(math.sin(self.body.target_orientation - self_orientation), math.cos(self.body.target_orientation - self_orientation))

            def calc_T(v, e):
                return (-1 if e < 0 else 1) * 2 * self.time_resolution * v / math.sqrt(self.body.constants.wheelbase**2 * (1 + 4 / math.tan(e)**2))

            constant_steering_action = self.body.constants.min_steering_angle if turn_angle < 0 else self.body.constants.max_steering_angle
            max_turn_angle = calc_T(self_velocity, constant_steering_action)

            def e(T):
                return (-1 if T < 0 else 1) * math.atan(2 * self.body.constants.wheelbase * math.sqrt(T**2 / (4 * self_velocity**2 * self.time_resolution**2 - self.body.constants.wheelbase**2 * T**2)))

            if turn_angle / max_turn_angle > 1:
                steering_action = e(max_turn_angle)
            else:
                steering_action = e(turn_angle)

        return [throttle_action, steering_action]

    def process_feedback(self, previous_state, action, state, reward):
        self_state = state[self.index]
        self_velocity = self_state[2]
        self_orientation = self_state[3]

        error = 0.000000000000001
        if self.body.target_velocity is not None:
            if abs(self.body.target_velocity - self_velocity) < error:
                self.body.target_velocity = None
        if self.body.target_orientation is not None:
            if abs(math.atan2(math.sin(self.body.target_orientation - self_orientation), math.cos(self.body.target_orientation - self_orientation))) < error:
                self.body.target_orientation = None

    def key_press(self, key, _mod):
        if key in key_target_velocity:
            self.body.target_velocity = key_target_velocity[key]
        elif key in key_target_orientation:
            self.body.target_orientation = key_target_orientation[key]


class RandomVehicleAgent(RandomAgent, DynamicBodyAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.action = [0.0, 0.0]

    def reset(self):
        self.action = [0.0, 0.0]

    def choose_action(self, state, action_space, info=None):
        if self.epsilon_valid():
            self.action = list(action_space.sample())
        return self.action

    def process_feedback(self, previous_state, action, state, reward):
        pass


class RandomPedestrianAgent(RandomAgent, DynamicBodyAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.action = [0.0, 0.0]

    def reset(self):
        self.action = [0.0, 0.0]

    def choose_action(self, state, action_space, info=None):
        if self.epsilon_valid():
            self.action = list(action_space.sample())
        return self.action

    def process_feedback(self, previous_state, action, state, reward):
        pass


class RandomConstrainedPedestrianAgent(RandomPedestrianAgent):  # (RoadCrossingPedestrianAgent, RandomPedestrianAgent):  # base class order is important so that RoadCrossingPedestrianAgent.reset() is called rather than RandomPedestrianAgent.reset()
    def __init__(self, body, time_resolution, road, **kwargs):
        super().__init__(**kwargs)

        self.body = body
        self.time_resolution = time_resolution
        self.road_centre = road.bounding_box().longitudinal_line()

        self.initial_distance = None
        self.waypoint = None
        self.target_orientation = None
        self.prior_orientation = None

    def reset(self):
        self.initial_distance = None
        self.waypoint = None
        self.target_orientation = None
        self.prior_orientation = None

    def choose_action(self, state, action_space, info=None):
        self_state = DynamicBodyState(
            position=Point(x=float(state[self.index][0]), y=float(state[self.index][1])),
            velocity=float(state[self.index][2]),
            orientation=float(state[self.index][3])
        )

        if self.waypoint is None and self.target_orientation is None and self.epsilon_valid():
            closest_point = self.road_centre.closest_point_from(self_state.position)
            relative_angle = math.atan2(closest_point.y - self_state.position.y, closest_point.x - self_state.position.x)
            if self.initial_distance is None:
                self.initial_distance = self_state.position.distance(closest_point)
            self.waypoint = Point(
                x=closest_point.x + self.initial_distance * math.cos(relative_angle),
                y=closest_point.y + self.initial_distance * math.sin(relative_angle),
            )
            self.target_orientation = math.atan2(self.waypoint.y - self_state.position.y, self.waypoint.x - self_state.position.x)
            self.prior_orientation = self_state.orientation

        if self.target_orientation is None or self_state.velocity == 0:
            steering_action = 0.0
        else:
            turn_angle = math.atan2(math.sin(self.target_orientation - self_state.orientation), math.cos(self.target_orientation - self_state.orientation))

            def calc_T(v, e):
                return (-1 if e < 0 else 1) * 2 * self.time_resolution * v / math.sqrt(self.body.constants.wheelbase**2 * (1 + 4 / math.tan(e)**2))

            constant_steering_action = self.body.constants.min_steering_angle if turn_angle < 0 else self.body.constants.max_steering_angle
            max_turn_angle = calc_T(self_state.velocity, constant_steering_action)

            def e(T):
                return (-1 if T < 0 else 1) * math.atan(2 * self.body.constants.wheelbase * math.sqrt(T**2 / (4 * self_state.velocity**2 * self.time_resolution**2 - self.body.constants.wheelbase**2 * T**2)))

            if turn_angle / max_turn_angle > 1:
                steering_action = e(max_turn_angle)
            else:
                steering_action = e(turn_angle)

        return [0.0, steering_action]

    def process_feedback(self, previous_state, action, state, reward):
        self_state = DynamicBodyState(
            position=Point(x=float(state[self.index][0]), y=float(state[self.index][1])),
            velocity=float(state[self.index][2]),
            orientation=float(state[self.index][3])
        )

        if self.waypoint is not None:
            distance = self_state.position.distance(self.waypoint)
            if distance < 1:
                self.waypoint = None
                self.target_orientation = self.prior_orientation
                self.prior_orientation = None
        if self.target_orientation is not None:
            if abs(math.atan2(math.sin(self.target_orientation - self_state.orientation), math.cos(self.target_orientation - self_state.orientation))) < 0.000000000000001:
                self.target_orientation = None


class ProximityPedestrianAgent(DynamicBodyAgent):  # (RoadCrossingPedestrianAgent):
    def __init__(self, body, time_resolution, road, distance_threshold, **kwargs):
        super().__init__(**kwargs)

        self.body = body
        self.time_resolution = time_resolution
        self.road_centre = road.bounding_box().longitudinal_line()
        self.distance_threshold = distance_threshold

        self.initial_distance = None
        self.waypoint = None
        self.target_orientation = None
        self.prior_orientation = None

    def reset(self):
        self.initial_distance = None
        self.waypoint = None
        self.target_orientation = None
        self.prior_orientation = None

    def choose_action(self, state, action_space, info=None):
        self_state = DynamicBodyState(
            position=Point(x=float(state[self.index][0]), y=float(state[self.index][1])),
            velocity=float(state[self.index][2]),
            orientation=float(state[self.index][3])
        )
        ego_position = Point(x=state[0][0], y=state[0][1])

        if self.waypoint is None and self.target_orientation is None and self_state.position.distance(ego_position) < self.distance_threshold:
            closest_point = self.road_centre.closest_point_from(self_state.position)
            relative_angle = math.atan2(closest_point.y - self_state.position.y, closest_point.x - self_state.position.x)
            if self.initial_distance is None:
                self.initial_distance = self_state.position.distance(closest_point)
            self.waypoint = Point(
                x=closest_point.x + self.initial_distance * math.cos(relative_angle),
                y=closest_point.y + self.initial_distance * math.sin(relative_angle),
            )
            self.target_orientation = math.atan2(self.waypoint.y - self_state.position.y, self.waypoint.x - self_state.position.x)
            self.prior_orientation = self_state.orientation

        if self.target_orientation is None or self_state.velocity == 0:
            steering_action = 0.0
        else:
            turn_angle = math.atan2(math.sin(self.target_orientation - self_state.orientation), math.cos(self.target_orientation - self_state.orientation))

            def calc_T(v, e):
                return (-1 if e < 0 else 1) * 2 * self.time_resolution * v / math.sqrt(self.body.constants.wheelbase**2 * (1 + 4 / math.tan(e)**2))

            constant_steering_action = self.body.constants.min_steering_angle if turn_angle < 0 else self.body.constants.max_steering_angle
            max_turn_angle = calc_T(self_state.velocity, constant_steering_action)

            def e(T):
                return (-1 if T < 0 else 1) * math.atan(2 * self.body.constants.wheelbase * math.sqrt(T**2 / (4 * self_state.velocity**2 * self.time_resolution**2 - self.body.constants.wheelbase**2 * T**2)))

            if turn_angle / max_turn_angle > 1:
                steering_action = e(max_turn_angle)
            else:
                steering_action = e(turn_angle)

        return [0.0, steering_action]

    def process_feedback(self, previous_state, action, state, reward):
        self_state = DynamicBodyState(
            position=Point(x=float(state[self.index][0]), y=float(state[self.index][1])),
            velocity=float(state[self.index][2]),
            orientation=float(state[self.index][3])
        )

        if self.waypoint is not None:
            distance = self_state.position.distance(self.waypoint)
            if distance < 1:
                self.waypoint = None
                self.target_orientation = self.prior_orientation
                self.prior_orientation = None
        if self.target_orientation is not None:
            if abs(math.atan2(math.sin(self.target_orientation - self_state.orientation), math.cos(self.target_orientation - self_state.orientation))) < 0.000000000000001:
                self.target_orientation = None


class ElectionPedestrianAgent(DynamicBodyAgent):  # (ProximityPedestrianAgent):
    def __init__(self, body, time_resolution, road, distance_threshold, **kwargs):
        super().__init__(**kwargs)

        self.body = body
        self.time_resolution = time_resolution
        self.road_centre = road.bounding_box().longitudinal_line()
        self.distance_threshold = distance_threshold

        self.initial_distance = None
        self.waypoint = None
        self.target_orientation = None
        self.prior_orientation = None
        self.crossing = False

    def reset(self):
        self.initial_distance = None
        self.waypoint = None
        self.target_orientation = None
        self.prior_orientation = None
        self.voting = False
        self.crossing = False

    def choose_action(self, state, action_space, info=None):
        self_state = DynamicBodyState(
            position=Point(x=float(state[self.index][0]), y=float(state[self.index][1])),
            velocity=float(state[self.index][2]),
            orientation=float(state[self.index][3])
        )
        ego_position = Point(x=state[0][0], y=state[0][1])

        if self.waypoint is None and self.target_orientation is None and self_state.position.distance(ego_position) < self.distance_threshold:
            closest_point = self.road_centre.closest_point_from(self_state.position)
            relative_angle = math.atan2(closest_point.y - self_state.position.y, closest_point.x - self_state.position.x)
            if self.initial_distance is None:
                self.initial_distance = self_state.position.distance(closest_point)
            self.waypoint = Point(
                x=closest_point.x + self.initial_distance * math.cos(relative_angle),
                y=closest_point.y + self.initial_distance * math.sin(relative_angle),
            )
            self.target_orientation = math.atan2(self.waypoint.y - self_state.position.y, self.waypoint.x - self_state.position.x)
            self.prior_orientation = self_state.orientation
            self.voting = True
            self.crossing = True
        else:
            self.voting = False

        if self.target_orientation is None or self_state.velocity == 0:
            steering_action = 0.0
        else:
            turn_angle = math.atan2(math.sin(self.target_orientation - self_state.orientation), math.cos(self.target_orientation - self_state.orientation))

            def calc_T(v, e):
                return (-1 if e < 0 else 1) * 2 * self.time_resolution * v / math.sqrt(self.body.constants.wheelbase**2 * (1 + 4 / math.tan(e)**2))

            constant_steering_action = self.body.constants.min_steering_angle if turn_angle < 0 else self.body.constants.max_steering_angle
            max_turn_angle = calc_T(self_state.velocity, constant_steering_action)

            def e(T):
                return (-1 if T < 0 else 1) * math.atan(2 * self.body.constants.wheelbase * math.sqrt(T**2 / (4 * self_state.velocity**2 * self.time_resolution**2 - self.body.constants.wheelbase**2 * T**2)))

            if turn_angle / max_turn_angle > 1:
                steering_action = e(max_turn_angle)
            else:
                steering_action = e(turn_angle)

        return [0.0, steering_action]

    def process_feedback(self, previous_state, action, state, reward):
        self_state = DynamicBodyState(
            position=Point(x=float(state[self.index][0]), y=float(state[self.index][1])),
            velocity=float(state[self.index][2]),
            orientation=float(state[self.index][3])
        )

        if self.waypoint is not None:
            distance = self_state.position.distance(self.waypoint)
            if distance < 1:
                self.waypoint = None
                self.target_orientation = self.prior_orientation
                self.prior_orientation = None
        if self.target_orientation is not None:
            if abs(math.atan2(math.sin(self.target_orientation - self_state.orientation), math.cos(self.target_orientation - self_state.orientation))) < 0.000000000000001:
                self.target_orientation = None
        if self.waypoint is None and self.target_orientation is None:
            self.crossing = False


class QLearningAgent(RandomPedestrianAgent):
    # self.alpha is learning rate (should decrease over time)
    # self.gamma is discount fbody (should be fixed over time?)
    # self.epsilon is exploration probability (should decrease over time)
    def __init__(self, body, ego_constants, road_polgon, time_resolution, width, height, q_learning_config, **kwargs):
        super().__init__(epsilon=q_learning_config.epsilon, **kwargs)  # self.epsilon is exploration probability (should decrease over time)

        self.body = body
        self.ego_constants = ego_constants
        # self.self_constants = self_constants
        self.road_polgon = road_polgon
        self.time_resolution = time_resolution
        self.alpha = q_learning_config.alpha  # learning rate (should decrease over time)
        self.gamma = q_learning_config.gamma  # discount fbody (should be fixed over time?)
        self.feature_config = q_learning_config.features

        self.log_file = None
        if q_learning_config.log is not None:
            self.log_file = reporting.get_agent_file_logger(q_learning_config.log)

        self.feature_bounds = dict()
        if self.feature_config.distance_x:
            self.feature_bounds["distance_x"] = (0, width)
        if self.feature_config.distance_y:
            self.feature_bounds["distance_y"] = (0, height)
        if self.feature_config.distance:
            self.feature_bounds["distance"] = (1, math.sqrt((width ** 2) + (height ** 2)))
        if self.feature_config.on_road:
            self.feature_bounds["on_road"] = (0, 1)
        if self.feature_config.facing:
            self.feature_bounds["facing"] = (0, math.pi)
        if self.feature_config.inverse_distance:
            self.feature_bounds["inverse_distance"] = (0, 1)
            x_mid = M2PX * 16  # 0 < x_mid < self.x_max
            y_mid = 0.5  # 0 < y_mid < 1
            self.x_max = self.feature_bounds["distance"][1] if "distance" in self.feature_bounds else math.sqrt((width ** 2) + (height ** 2))
            self.n = math.log(1 - y_mid) / math.log(x_mid / self.x_max)

        self.feature_weights = {feature: 0.0 for feature in self.feature_bounds.keys()}
        if self.log_file:
            self.enabled_features = sorted(self.feature_bounds.keys())
            self.log_file.info(f"{','.join(map(str, self.enabled_features))}")

        target_velocities = [float(self.body.constants.max_velocity), float((self.body.constants.max_velocity - self.body.constants.min_velocity) / 2)]
        target_orientations = [math.pi, math.pi * 0.5, 0.0, -(math.pi * 0.5)]
        self.available_targets = [(target_velocity, target_orientation) for target_velocity in target_velocities for target_orientation in target_orientations]
        self.target_velocity = None
        self.target_orientation = None

    def reset(self):
        self.target_velocity = None
        self.target_orientation = None
        if self.log_file:
            self.log_file.info(f"{','.join(map(str, [self.feature_weights[feature] for feature in self.enabled_features]))}")

    def features(self, state, target):  # question: what does it mean for a feature to depend on an action and/or what does a Q value mean if it does not depend on an action?
        def make_body_state(data):
            # assert len(data) == 8
            return DynamicBodyState(
                position=Point(data[0], data[1]),
                velocity=data[2],
                orientation=data[3]
            )

        ego_state = make_body_state(state[0])
        self_state = make_body_state(state[self.index])

        ego_body = Car(ego_state, self.ego_constants)
        self_body = Pedestrian(self_state, self.body.constants)

        target_velocity, target_orientation = target

        if target_velocity is None:
            throttle_action = 0.0
        else:
            diff = (target_velocity - self_state.velocity) / self.time_resolution
            throttle_action = min(self.body.constants.max_throttle, max(self.body.constants.min_throttle, diff))

        if target_orientation is None or self_state.velocity == 0:
            steering_action = 0.0
        else:
            turn_angle = math.atan2(math.sin(target_orientation - self_state.orientation), math.cos(target_orientation - self_state.orientation))

            def calc_T(v, e):
                return (-1 if e < 0 else 1) * 2 * self.time_resolution * v / math.sqrt(self.body.constants.wheelbase**2 * (1 + 4 / math.tan(e)**2))

            constant_steering_action = self.body.constants.min_steering_angle if turn_angle < 0 else self.body.constants.max_steering_angle
            max_turn_angle = calc_T(self_state.velocity, constant_steering_action)

            def e(T):
                return (-1 if T < 0 else 1) * math.atan(2 * self.body.constants.wheelbase * math.sqrt(T**2 / (4 * self_state.velocity**2 * self.time_resolution**2 - self.body.constants.wheelbase**2 * T**2)))

            if turn_angle / max_turn_angle > 1:
                steering_action = e(max_turn_angle)
            else:
                steering_action = e(turn_angle)

        action = [throttle_action, steering_action]
        joint_action = [(0.0, 0.0), action]

        spawn_bodies = [ego_body, self_body]
        for i, spawn_body in enumerate(spawn_bodies):
            spawn_body.step(joint_action[i], self.time_resolution)

        ego_position = ego_body.state.position
        self_position = self_body.state.position

        def normalise(value, min_bound, max_bound):
            if value < min_bound:
                return 0
            elif value > max_bound:
                return 1
            else:
                return (value - min_bound) / (max_bound - min_bound)

        unnormalised_values = dict()
        if self.feature_config.distance_x:
            unnormalised_values["distance_x"] = self_position.distance_x(ego_position)
        if self.feature_config.distance_y:
            unnormalised_values["distance_y"] = self_position.distance_y(ego_position)
        if self.feature_config.distance:
            unnormalised_values["distance"] = self_position.distance(ego_position)
        if self.feature_config.on_road:
            unnormalised_values["on_road"] = 1 if self_body.bounding_box().intersects(self.road_polgon) else 0
        if self.feature_config.facing:
            unnormalised_values["facing"] = abs(geometry.Line(start=ego_position, end=self_position).orientation() - ego_body.state.orientation)
        if self.feature_config.inverse_distance:
            x = unnormalised_values["distance"] if "distance" in unnormalised_values else self_position.distance(ego_position)
            unnormalised_values["inverse_distance"] = 1 - (x / self.x_max) ** self.n  # thanks to Ram Varadarajan

        normalised_values = {feature: normalise(feature_value, *self.feature_bounds[feature]) for feature, feature_value in unnormalised_values.items()}
        return normalised_values

    def q_value(self, state, target):  # if features do not depend on target, then the Q value will not either
        feature_values = self.features(state, target)
        q_value = sum(feature_value * self.feature_weights[feature] for feature, feature_value in feature_values.items())
        # assert math.isfinite(q_value)
        # print(reporting.pretty_float_list(list(feature_values.values())), reporting.pretty_float_list(list(self.feature_weights.values())), reporting.pretty_float(q_value))
        return q_value

    def choose_action(self, state, action_space, info=None):
        self_state = DynamicBodyState(
            position=Point(x=float(state[self.index][0]), y=float(state[self.index][1])),
            velocity=float(state[self.index][2]),
            orientation=float(state[self.index][3])
        )

        if self.target_velocity is None and self.target_orientation is None:
            if self.epsilon_valid():
                self.target_velocity, self.target_orientation = self.np_random.choice(self.available_targets)
            else:
                best_targets = list()  # there may be multiple targets with max Q value
                max_q_value = -math.inf
                for target in self.available_targets:
                    q_value = self.q_value(state, target)
                    if q_value > max_q_value:
                        best_targets = [target]
                        max_q_value = q_value
                    elif q_value == max_q_value:
                        best_targets.append(target)
                assert best_targets, "no best target(s) found"
                self.target_velocity, self.target_orientation = best_targets[0] if len(best_targets) == 1 else best_targets[self.np_random.choice(range(len(best_targets)))]

        if self.target_velocity is None:
            throttle_action = 0.0
        else:
            diff = (self.target_velocity - self_state.velocity) / self.time_resolution
            min_throttle = self.body.constants.min_throttle  # action_space.low[0]
            max_throttle = self.body.constants.max_throttle  # action_space.high[0]
            throttle_action = min(max_throttle, max(min_throttle, diff))

        if self.target_orientation is None or self_state.velocity == 0:
            steering_action = 0.0
        else:
            turn_angle = math.atan2(math.sin(self.target_orientation - self_state.orientation), math.cos(self.target_orientation - self_state.orientation))

            def calc_T(v, e):
                return (-1 if e < 0 else 1) * 2 * self.time_resolution * v / math.sqrt(self.body.constants.wheelbase**2 * (1 + 4 / math.tan(e)**2))

            constant_steering_action = self.body.constants.min_steering_angle if turn_angle < 0 else self.body.constants.max_steering_angle
            max_turn_angle = calc_T(self_state.velocity, constant_steering_action)

            def e(T):
                return (-1 if T < 0 else 1) * math.atan(2 * self.body.constants.wheelbase * math.sqrt(T**2 / (4 * self_state.velocity**2 * self.time_resolution**2 - self.body.constants.wheelbase**2 * T**2)))

            if turn_angle / max_turn_angle > 1:
                steering_action = e(max_turn_angle)
            else:
                steering_action = e(turn_angle)

        error = 0.000000000000001  # account for minor (precision) error only: large error indicates issue with logic
        if throttle_action < action_space.low[0] or throttle_action > action_space.high[0]:
            assert abs(action_space.low[0] - throttle_action) < error or abs(action_space.high[0] - throttle_action) < error
        if steering_action < action_space.low[1] or steering_action > action_space.high[1]:
            assert abs(action_space.low[0] - throttle_action) < error or abs(action_space.high[0] - throttle_action) < error
        throttle_action = min(self.body.constants.max_throttle, max(self.body.constants.min_throttle, throttle_action))
        steering_action = min(self.body.constants.max_steering_angle, max(self.body.constants.min_steering_angle, steering_action))

        return [throttle_action, steering_action]

    def process_feedback(self, previous_state, action, state, reward):  # agent executed action in previous_state, and then arrived in state where it received reward
        self_state = DynamicBodyState(
            position=Point(x=float(state[self.index][0]), y=float(state[self.index][1])),
            velocity=float(state[self.index][2]),
            orientation=float(state[self.index][3])
        )

        # q_value = self.q_value(previous_state, action)
        # new_q_value = reward + self.gamma * max(self.q_value(state, action_prime) for action_prime in self.available_actions)
        # q_value_gain = new_q_value - q_value
        # for feature, feature_value in self.features(previous_state, action).items():
        #     self.feature_weights[feature] = self.feature_weights[feature] + self.alpha * q_value_gain * feature_value

        target = self.target_velocity, self.target_orientation
        q_value = self.q_value(previous_state, target)
        difference = (reward + self.gamma * max(self.q_value(state, target_prime) for target_prime in self.available_targets)) - q_value
        for feature, feature_value in self.features(previous_state, target).items():
            self.feature_weights[feature] = self.feature_weights[feature] + self.alpha * difference * feature_value

        # print(reporting.pretty_float_list(list(self.feature_weights.values())))

        error = 0.000000000000001
        if self.target_velocity is not None:
            if abs(self.target_velocity - self_state.velocity) < error:
                self.target_velocity = None
        if self.target_orientation is not None:
            if abs(math.atan2(math.sin(self.target_orientation - self_state.orientation), math.cos(self.target_orientation - self_state.orientation))) < error:
                self.target_orientation = None


class DecayedQLearningAgent(QLearningAgent):
    def __init__(self, decay_length=1000, alpha_start=1, alpha_end=0.01, epsilon_start=0.2, epsilon_end=0, **kwargs):
        assert decay_length > 0

        super().__init__(alpha=alpha_start, epsilon=epsilon_start, **kwargs)

        self.decay_length = decay_length
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end

        self.decay_updates = 0

    def reset(self):
        decay = max((self.decay_length - self.decay_updates) / self.decay_length, 0)

        def decayed_value(start, end):
            return ((start - end) * decay) + end

        self.alpha = decayed_value(self.alpha_start, self.alpha_end)
        self.epsilon = decayed_value(self.epsilon_start, self.epsilon_end)
        self.decay_updates += 1

        # print(f"alpha={self.alpha}, epsilon={self.epsilon}, gamma={self.gamma}")
