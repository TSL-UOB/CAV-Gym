import math

import numpy as np

from examples.agents.dynamic_body import make_body_state, make_throttle_action
from examples.agents.template import RandomAgent
from library import geometry
from library.bodies import DynamicBodyState
from library.geometry import Point

TARGET_ERROR = 0.000000000000001
ACTION_ERROR = 0.000000000000001


class QLearningEgoAgent(RandomAgent):
    def __init__(self, q_learning_config, body, time_resolution, num_velocity_targets, width, height, **kwargs):
        super().__init__(epsilon=q_learning_config.epsilon, **kwargs)

        self.alpha = q_learning_config.alpha  # learning rate (should decrease over time)
        self.gamma = q_learning_config.gamma  # discount fbody (should be fixed over time?)
        self.feature_config = q_learning_config.features

        self.body = body
        self.time_resolution = time_resolution

        self.target_velocities = list(np.linspace(start=self.body.constants.min_velocity, stop=self.body.constants.max_velocity, num=num_velocity_targets, endpoint=True))

        self.feature_bounds = dict()
        if self.feature_config.distance_x:
            self.feature_bounds["distance_x"] = (0, width)
        if self.feature_config.distance_y:
            self.feature_bounds["distance_y"] = (0, height)
        if self.feature_config.distance:
            self.feature_bounds["distance"] = (0, math.sqrt((width ** 2) + (height ** 2)))
        if self.feature_config.facing:
            self.feature_bounds["facing"] = (0, math.pi)

        self.feature_weights = {feature: 0.0 for feature in self.feature_bounds.keys()}

    def reset(self):
        self.body.target_velocity = None

    def choose_action(self, state, action_space, info=None):
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

        body_state = make_body_state(state, self.index)
        throttle_action = make_throttle_action(body_state, self.body.constants, self.time_resolution, self.body.target_velocity, self.noop_action)

        # account for minor (precision) error only: large error indicates issue with logic
        if throttle_action < action_space.low[0] or throttle_action > action_space.high[0]:
            assert abs(action_space.low[0] - throttle_action) < ACTION_ERROR or abs(action_space.high[0] - throttle_action) < ACTION_ERROR
            throttle_action = min(self.body.constants.max_throttle, max(self.body.constants.min_throttle, throttle_action))

        return [throttle_action, self.noop_action[1]]

    def process_feedback(self, previous_state, action, state, reward):
        difference = (reward + self.gamma * max(self.q_value(state, target_velocity) for target_velocity in self.target_velocities)) - self.q_value(previous_state, self.body.target_velocity)
        for feature, feature_value in self.features(previous_state, self.body.target_velocity).items():
            self.feature_weights[feature] = self.feature_weights[feature] + self.alpha * difference * feature_value

        # print(self.feature_weights)

        body_state = make_body_state(state, self.index)

        if self.body.target_velocity is not None:
            if abs(self.body.target_velocity - body_state.velocity) < TARGET_ERROR:
                self.body.target_velocity = None

    def q_value(self, state, target_velocity):
        feature_values = self.features(state, target_velocity)
        q_value = sum(feature_value * self.feature_weights[feature] for feature, feature_value in feature_values.items())
        return q_value

    def features(self, state, target_velocity):
        self_state = make_body_state(state, self.index)
        opponent_state = make_body_state(state, self.index + 1)

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
        if self.feature_config.facing:
            unnormalised_values["facing"] = abs(geometry.Line(start=self_state.position, end=opponent_state.position).orientation() - self_state.orientation)

        normalised_values = {feature: normalise(feature_value, *self.feature_bounds[feature]) for feature, feature_value in unnormalised_values.items()}
        return normalised_values
