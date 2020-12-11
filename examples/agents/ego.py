import math

import numpy as np

import reporting
from examples.agents.dynamic_body import make_body_state, make_throttle_action
from examples.agents.template import RandomAgent
from library import geometry
from library.bodies import DynamicBodyState
from library.geometry import Point

TARGET_ERROR = 0.000000000000001
ACTION_ERROR = 0.000000000000001


class QLearningEgoAgent(RandomAgent):
    def __init__(self, q_learning_config, body, time_resolution, num_opponents, num_velocity_targets, width, height, **kwargs):
        super().__init__(noop_action=body.noop_action, epsilon=q_learning_config.epsilon, **kwargs)

        self.alpha = q_learning_config.alpha  # learning rate (should decrease over time)
        self.gamma = q_learning_config.gamma  # discount fbody (should be fixed over time?)
        self.feature_config = q_learning_config.features

        self.body = body
        self.time_resolution = time_resolution

        self.opponent_indexes = list(range(1, num_opponents + 1))

        self.target_velocities = list(np.linspace(start=self.body.constants.min_velocity, stop=self.body.constants.max_velocity, num=num_velocity_targets, endpoint=True))

        self.log_file = None
        if q_learning_config.log is not None:
            self.log_file = reporting.get_agent_file_logger(q_learning_config.log)

        self.feature_bounds = dict()
        if self.feature_config.distance_x:
            self.feature_bounds["distance_x"] = (-float(width), float(width))
        if self.feature_config.distance_y:
            self.feature_bounds["distance_y"] = (-float(height), float(height))
        if self.feature_config.distance:
            self.feature_bounds["distance"] = (0.0, math.sqrt((width ** 2) + (height ** 2)))
        if self.feature_config.relative_angle:
            self.feature_bounds["relative_angle"] = (0.0, math.pi)

        self.feature_weights = {index: {feature: 0.0 for feature in self.feature_bounds.keys()} for index in self.opponent_indexes}

        if self.log_file:
            self.enabled_features = {index: sorted(self.feature_bounds.keys()) for index in self.opponent_indexes}
            labels = [f"{feature}{index}" for index, features in self.enabled_features.items() for feature in features]
            self.log_file.info(f"{','.join(map(str, labels))}")

    def reset(self):
        self.body.target_velocity = None

    def choose_action(self, state, action_space, info=None):
        body_state = make_body_state(state, self.index)

        if self.body.target_velocity is None:
            if self.epsilon_valid():
                self.body.target_velocity = self.np_random.choice(self.target_velocities)
            else:
                best_targets = list()  # there may be multiple targets with max Q value
                max_q_value = -math.inf
                for target_velocity in self.target_velocities:
                    q_value = self.q_value(state, target_velocity)
                    if q_value > max_q_value:
                        best_targets = [target_velocity]
                        max_q_value = q_value
                    elif q_value == max_q_value:
                        best_targets.append(target_velocity)
                assert best_targets, "no best target(s) found"
                self.body.target_velocity = best_targets[0] if len(best_targets) == 1 else self.np_random.choice(best_targets)

            if abs(self.body.target_velocity - body_state.velocity) < TARGET_ERROR:
                self.body.target_velocity = None
                return self.noop_action

        throttle_action = make_throttle_action(body_state, self.body.constants, self.time_resolution, self.body.target_velocity, self.noop_action)

        # account for minor (precision) error only: large error indicates issue with logic
        if throttle_action < action_space.low[0] or throttle_action > action_space.high[0]:
            assert abs(action_space.low[0] - throttle_action) < ACTION_ERROR or abs(action_space.high[0] - throttle_action) < ACTION_ERROR
            throttle_action = min(self.body.constants.max_throttle, max(self.body.constants.min_throttle, throttle_action))

        return [throttle_action, self.noop_action[1]]

    def process_feedback(self, previous_state, action, state, reward):
        difference = (reward + self.gamma * max(self.q_value(state, target_velocity) for target_velocity in self.target_velocities)) - self.q_value(previous_state, self.body.target_velocity)
        for index, opponent_features in self.features(previous_state, self.body.target_velocity).items():
            for feature, feature_value in opponent_features.items():
                self.feature_weights[index][feature] = self.feature_weights[index][feature] + self.alpha * difference * feature_value

        if self.log_file:
            weights = [self.feature_weights[index][feature] for index, features in self.enabled_features.items() for feature in features]
            self.log_file.info(f"{','.join(map(str, weights))}")

        body_state = make_body_state(state, self.index)

        if self.body.target_velocity is not None:
            if abs(self.body.target_velocity - body_state.velocity) < TARGET_ERROR:
                self.body.target_velocity = None

    def q_value(self, state, target_velocity):
        feature_values = self.features(state, target_velocity)
        q_value = sum(feature_value * self.feature_weights[index][feature] for index, opponent_feature_values in feature_values.items() for feature, feature_value in opponent_feature_values.items())
        return q_value

    def features(self, state, target_velocity):
        # print()
        # sleep(0.05)
        return {index: self.features_opponent(state, target_velocity, index) for index in self.opponent_indexes}

    def features_opponent(self, state, target_velocity, opponent_index):
        self_state = make_body_state(state, self.index)
        opponent_state = make_body_state(state, opponent_index)

        def one_step_lookahead(body_state, throttle):  # one-step lookahead with no steering
            distance_velocity = body_state.velocity * self.time_resolution
            return DynamicBodyState(
                position=Point(
                    x=body_state.position.x + distance_velocity * math.cos(body_state.orientation),
                    y=body_state.position.y + distance_velocity * math.sin(body_state.orientation)
                ),
                velocity=max(self.body.constants.min_velocity, min(self.body.constants.max_velocity, body_state.velocity + (throttle * self.time_resolution))),
                orientation=body_state.orientation
            )

        def n_step_lookahead(body_state, throttle, n=2):
            next_body_state = body_state
            for _ in range(n):
                next_body_state = one_step_lookahead(next_body_state, throttle)
            return next_body_state

        throttle_action = make_throttle_action(self_state, self.body.constants, self.time_resolution, target_velocity, self.noop_action)

        self_state = n_step_lookahead(self_state, throttle_action)
        opponent_state = n_step_lookahead(opponent_state, 0.0)

        def normalise(value, min_bound, max_bound):
            if value < min_bound:
                return 0
            elif value > max_bound:
                return 1
            else:
                return (value - min_bound) / (max_bound - min_bound)

        unnormalised_values = dict()
        if self.feature_config.distance_x:
            unnormalised_values["distance_x"] = self_state.position.distance_x(opponent_state.position)
        if self.feature_config.distance_y:
            unnormalised_values["distance_y"] = self_state.position.distance_y(opponent_state.position)
        if self.feature_config.distance:
            unnormalised_values["distance"] = self_state.position.distance(opponent_state.position)
        if self.feature_config.relative_angle:
            unnormalised_values["relative_angle"] = abs(geometry.normalise_angle(geometry.Line(start=self_state.position, end=opponent_state.position).orientation() - self_state.orientation))

        # print(f"R{opponent_index}={unnormalised_values['distance']}, A{opponent_index}={math.degrees(unnormalised_values['relative_angle'])}", end=", ")

        normalised_values = {feature: normalise(feature_value, *self.feature_bounds[feature]) for feature, feature_value in unnormalised_values.items()}
        # print(f"R{opponent_index}={normalised_values['distance']}, A{opponent_index}={normalised_values['relative_angle']}", end=", ")
        return normalised_values
