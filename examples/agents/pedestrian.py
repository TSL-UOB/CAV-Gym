import math

import reporting
from examples.agents.dynamic_body import make_body_state, make_steering_action, TARGET_ERROR, make_throttle_action, \
    TargetAgent
from examples.agents.template import RandomAgent, NoopAgent
from examples.constants import M2PX
from examples.targets import TargetOrientation, TargetVelocity
from library import geometry
from library.bodies import Pedestrian, Car
from library.geometry import Point


class CrossingAgent(NoopAgent):
    def __init__(self, body, time_resolution, road_centre, **kwargs):
        super().__init__(noop_action=body.noop_action, **kwargs)

        self.body = body
        self.time_resolution = time_resolution
        self.road_centre = road_centre

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
        raise NotImplementedError

    def process_feedback(self, previous_state, action, state, reward):
        body_state = make_body_state(state, self.index)

        if self.waypoint is not None:
            distance = body_state.position.distance(self.waypoint)
            if distance < 1:
                self.waypoint = None
                self.target_orientation = self.prior_orientation
                self.prior_orientation = None
        if self.target_orientation is not None:
            diff = self.target_orientation - body_state.orientation
            if abs(math.atan2(math.sin(diff), math.cos(diff))) < TARGET_ERROR:
                self.target_orientation = None

    def choose_crossing_action(self, state, condition):
        body_state = make_body_state(state, self.index)

        if self.waypoint is None and self.target_orientation is None and condition:
            closest_point = self.road_centre.closest_point_from(body_state.position)
            relative_angle = math.atan2(closest_point.y - body_state.position.y, closest_point.x - body_state.position.x)
            if self.initial_distance is None:
                self.initial_distance = body_state.position.distance(closest_point)
            self.waypoint = Point(
                x=closest_point.x + self.initial_distance * math.cos(relative_angle),
                y=closest_point.y + self.initial_distance * math.sin(relative_angle),
            )
            self.target_orientation = math.atan2(self.waypoint.y - body_state.position.y, self.waypoint.x - body_state.position.x)
            self.prior_orientation = body_state.orientation
            crossing_initiated = True
        else:
            crossing_initiated = False

        steering_action = make_steering_action(body_state, self.body.constants, self.time_resolution, self.target_orientation, self.noop_action)
        return [self.noop_action[0], steering_action], crossing_initiated


class RandomConstrainedAgent(CrossingAgent, RandomAgent):  # (RoadCrossingPedestrianAgent, RandomPedestrianAgent):  # base class order is important so that RoadCrossingPedestrianAgent.reset() is called rather than RandomPedestrianAgent.reset()
    def choose_action(self, state, action_space, info=None):
        action, _ = self.choose_crossing_action(state, self.epsilon_valid())
        return action


class ProximityAgent(CrossingAgent):
    def __init__(self, distance_threshold, **kwargs):
        super().__init__(**kwargs)

        self.distance_threshold = distance_threshold

    def choose_action(self, state, action_space, info=None):
        action, _ = self.choose_crossing_action(state, self.proximity_trigger(state))
        return action

    def proximity_trigger(self, state):
        body_position = Point(x=state[self.index][0], y=state[self.index][1])
        ego_position = Point(x=state[0][0], y=state[0][1])
        return body_position.distance(ego_position) < self.distance_threshold


class ElectionAgent(ProximityAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.voting = False
        self.crossing = False

    def reset(self):
        super().reset()
        self.voting = False
        self.crossing = False

    def choose_action(self, state, action_space, info=None):
        action, self.voting = self.choose_crossing_action(state, self.proximity_trigger(state))
        if self.voting:
            self.crossing = True
        return action

    def process_feedback(self, previous_state, action, state, reward):
        super().process_feedback(previous_state, action, state, reward)

        if self.waypoint is None and self.target_orientation is None:
            self.crossing = False


class QLearningAgent(TargetAgent, RandomAgent):
    # self.alpha is learning rate (should decrease over time)
    # self.gamma is discount fbody (should be fixed over time?)
    # self.epsilon is exploration probability (should decrease over time)
    def __init__(self, ego_constants, road_polgon, width, height, q_learning_config, **kwargs):
        super().__init__(epsilon=q_learning_config.epsilon, **kwargs)  # self.epsilon is exploration probability (should decrease over time)

        self.ego_constants = ego_constants
        self.road_polgon = road_polgon
        self.alpha = q_learning_config.alpha  # learning rate (should decrease over time)
        self.gamma = q_learning_config.gamma  # discount fbody (should be fixed over time?)
        self.feature_config = q_learning_config.features

        self.log_file = None
        if q_learning_config.log is not None:
            self.log_file = reporting.get_agent_file_logger(q_learning_config.log)

        self.feature_bounds = dict()
        if self.feature_config.distance_x:
            self.feature_bounds["distance_x"] = (0.0, width)
        if self.feature_config.distance_y:
            self.feature_bounds["distance_y"] = (0.0, height)
        if self.feature_config.distance:
            self.feature_bounds["distance"] = (0.0, math.sqrt((width ** 2) + (height ** 2)))
        if self.feature_config.on_road:
            self.feature_bounds["on_road"] = (0.0, 1.0)
        if self.feature_config.relative_angle:
            self.feature_bounds["relative_angle"] = (0.0, math.pi)
        if self.feature_config.inverse_distance:
            self.feature_bounds["inverse_distance"] = (0.0, 1.0)
            x_mid = M2PX * 16  # 0 < x_mid < self.x_max
            y_mid = 0.5  # 0 < y_mid < 1
            self.x_max = self.feature_bounds["distance"][1] if "distance" in self.feature_bounds else math.sqrt((width ** 2) + (height ** 2))
            self.n = math.log(1 - y_mid) / math.log(x_mid / self.x_max)

        self.feature_weights = {feature: 0.0 for feature in self.feature_bounds.keys()}
        if self.log_file:
            self.enabled_features = sorted(self.feature_bounds.keys())
            self.log_file.info(f"{','.join(map(str, self.enabled_features))}")

        target_orientations = [TargetOrientation.NORTH, TargetOrientation.EAST, TargetOrientation.SOUTH, TargetOrientation.WEST]
        self.available_targets = [(self.target_velocity_mapping[target_velocity], target_orientation.value) for target_velocity in TargetVelocity for target_orientation in target_orientations]

    def reset(self):
        super().reset()
        if self.log_file:
            self.log_file.info(f"{','.join(map(str, [self.feature_weights[feature] for feature in self.enabled_features]))}")

    def features(self, state, target):  # question: what does it mean for a feature to depend on an action and/or what does a Q value mean if it does not depend on an action?
        ego_state = make_body_state(state, 0)
        self_state = make_body_state(state, self.index)

        ego_body = Car(ego_state, self.ego_constants)
        self_body = Pedestrian(self_state, self.body.constants)

        target_velocity, target_orientation = target

        throttle_action = make_throttle_action(self_state, self.body.constants, self.time_resolution, target_velocity, self.noop_action)
        steering_action = make_steering_action(self_state, self.body.constants, self.time_resolution, target_orientation, self.noop_action)

        action = [throttle_action, steering_action]
        joint_action = [ego_body.noop_action, action]

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
        if self.feature_config.relative_angle:
            unnormalised_values["relative_angle"] = abs(geometry.normalise_angle(geometry.Line(start=self_state.position, end=ego_position.position).orientation() - self_state.orientation))
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
        if self.body.target_velocity is None and self.body.target_orientation is None:
            if self.epsilon_valid():
                self.body.target_velocity, self.body.target_orientation = self.np_random.choice(self.available_targets)
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
                self.body.target_velocity, self.body.target_orientation = best_targets[0] if len(best_targets) == 1 else best_targets[self.np_random.choice(range(len(best_targets)))]

        return super().choose_action(state, action_space, info=info)

    def process_feedback(self, previous_state, action, state, reward):  # agent executed action in previous_state, and then arrived in state where it received reward
        target = self.body.target_velocity, self.body.target_orientation
        q_value = self.q_value(previous_state, target)
        difference = (reward + self.gamma * max(self.q_value(state, target_prime) for target_prime in self.available_targets)) - q_value
        for feature, feature_value in self.features(previous_state, target).items():
            self.feature_weights[feature] = self.feature_weights[feature] + self.alpha * difference * feature_value

        # print(reporting.pretty_float_list(list(self.feature_weights.values())))

        super().process_feedback(previous_state, action, state, reward)
