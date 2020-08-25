import math

from gym.utils import seeding

import reporting
from library import geometry
from library.actions import OrientationAction, VelocityAction, TrafficLightAction
from library.actors import Car, Pedestrian, DynamicActorState
from library.geometry import Point
from library.observations import RoadObservation
from examples.constants import M2PX


class Agent:
    def reset(self):
        raise NotImplementedError

    def choose_action(self, state, info=None):
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

    def choose_action(self, state, info=None):
        raise NotImplementedError

    def process_feedback(self, previous_state, action, state, reward):
        raise NotImplementedError


class RandomTrafficLightAgent(RandomAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self):
        pass

    def choose_action(self, observation, info=None):
        if self.epsilon_valid():
            action = self.np_random.choice([value for value in TrafficLightAction])
        else:
            action = TrafficLightAction.NOOP

        return action.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass


class DynamicActorAgent(Agent):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)

        self.index = index

    def reset(self):
        raise NotImplementedError

    def active_velocity(self, state):
        target_velocity = state[self.index][6]
        return target_velocity is not None

    def active_orientation(self, state):
        target_orientation = state[self.index][7]
        return target_orientation is not None

    def choose_action(self, state, info=None):
        raise NotImplementedError

    def process_feedback(self, previous_observation, action, observation, reward):
        raise NotImplementedError


class NoopAgent(DynamicActorAgent):
    def reset(self):
        pass

    def choose_action(self, state, info=None):
        return VelocityAction.NOOP.value, OrientationAction.NOOP.value

    def process_feedback(self, previous_state, action, state, reward):
        pass


key_velocity_actions = {
    65365: VelocityAction.FAST,  # page up
    65366: VelocityAction.SLOW,  # page down
    65367: VelocityAction.STOP,  # end
}

key_orientation_actions = {
    65361: OrientationAction.LEFT,  # arrow left
    65363: OrientationAction.RIGHT,  # arrow right
    65364: OrientationAction.REAR,  # arrow down

    65457: OrientationAction.REAR_LEFT,  # numpad 1
    65458: OrientationAction.REAR,  # numpad 2
    65459: OrientationAction.REAR_RIGHT,  # numpad 3
    65460: OrientationAction.LEFT,  # numpad 4
    65462: OrientationAction.RIGHT,  # numpad 6
    65463: OrientationAction.FORWARD_LEFT,  # numpad 7
    65465: OrientationAction.FORWARD_RIGHT  # numpad 9
}


class KeyboardAgent(DynamicActorAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pending_velocity_action = VelocityAction.NOOP
        self.pending_orientation_action = OrientationAction.NOOP

    def reset(self):
        self.pending_velocity_action = VelocityAction.NOOP
        self.pending_orientation_action = OrientationAction.NOOP

    def choose_action(self, state, info=None):
        if not self.active_velocity(state):
            velocity_action = self.pending_velocity_action
            self.pending_velocity_action = VelocityAction.NOOP
        else:
            velocity_action = VelocityAction.NOOP

        if not self.active_orientation(state):
            orientation_action = self.pending_orientation_action
            self.pending_orientation_action = OrientationAction.NOOP
        else:
            orientation_action = OrientationAction.NOOP

        return velocity_action.value, orientation_action.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass

    def key_press(self, key, _mod):
        if key in key_velocity_actions:
            self.pending_velocity_action = key_velocity_actions[key]
        if key in key_orientation_actions:
            self.pending_orientation_action = key_orientation_actions[key]


class RandomVehicleAgent(RandomAgent, DynamicActorAgent):
    def reset(self):
        pass

    def choose_action(self, state, info=None):
        if not self.active_velocity(state) and self.epsilon_valid():
            velocity_action = self.np_random.choice([value for value in VelocityAction if value is not VelocityAction.STOP])
        else:
            velocity_action = VelocityAction.NOOP

        return velocity_action.value, OrientationAction.NOOP.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass


class RandomPedestrianAgent(RandomAgent, DynamicActorAgent):
    def reset(self):
        pass

    def choose_random_velocity_action(self, state, condition):
        if not self.active_velocity(state) and condition:
            return self.np_random.choice([action for action in VelocityAction])
        else:
            return VelocityAction.NOOP

    def choose_random_orientation_action(self, state, condition):
        if not self.active_orientation(state) and condition:
            return self.np_random.choice([action for action in OrientationAction])
        else:
            return OrientationAction.NOOP

    def choose_action(self, state, info=None):
        epsilon_valid = self.epsilon_valid()

        velocity_action = self.choose_random_velocity_action(state, epsilon_valid)
        orientation_action = self.choose_random_orientation_action(state, epsilon_valid)

        return velocity_action.value, orientation_action.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass


cross_road_action = {
    RoadObservation.ROAD_FRONT: OrientationAction.NOOP,
    RoadObservation.ROAD_FRONT_LEFT: OrientationAction.FORWARD_LEFT,
    RoadObservation.ROAD_LEFT: OrientationAction.LEFT,
    RoadObservation.ROAD_REAR_LEFT: OrientationAction.REAR_LEFT,
    RoadObservation.ROAD_REAR: OrientationAction.REAR,
    RoadObservation.ROAD_REAR_RIGHT: OrientationAction.REAR_RIGHT,
    RoadObservation.ROAD_RIGHT: OrientationAction.RIGHT,
    RoadObservation.ROAD_FRONT_RIGHT: OrientationAction.FORWARD_RIGHT
}

end_cross_road_action = {
    OrientationAction.NOOP: OrientationAction.LEFT,
    OrientationAction.FORWARD_LEFT: OrientationAction.RIGHT,
    OrientationAction.LEFT: OrientationAction.RIGHT,
    OrientationAction.REAR_LEFT: OrientationAction.RIGHT,
    OrientationAction.REAR: OrientationAction.LEFT,
    OrientationAction.REAR_RIGHT: OrientationAction.LEFT,
    OrientationAction.RIGHT: OrientationAction.LEFT,
    OrientationAction.FORWARD_RIGHT: OrientationAction.LEFT
}


class RoadCrossingPedestrianAgent(DynamicActorAgent):
    def __init__(self, road, **kwargs):
        super().__init__(**kwargs)

        self.road = road

        self.delay = 15  # defined by experimentation

        self.crossing_action = None
        self.entered_road = False
        self.pavement_count = 0
        self.reorient_count = 0

    def reset(self):
        self.crossing_action = None
        self.entered_road = False
        self.pavement_count = 0
        self.reorient_count = 0

    def road_observation(self, info):
        actor_polygons = info['actor_polygons']
        self_polygon = actor_polygons[self.index]
        road_polygon = self.road.bounding_box()
        if self_polygon.intersects(road_polygon):
            return RoadObservation.ON_ROAD
        else:
            road_angles = info['road_angles']
            self_angle = road_angles[self.index]
            if geometry.DEG2RAD * -157.5 <= self_angle < geometry.DEG2RAD * -112.5:
                return RoadObservation.ROAD_REAR_RIGHT
            elif geometry.DEG2RAD * -112.5 <= self_angle < geometry.DEG2RAD * -67.5:
                return RoadObservation.ROAD_RIGHT
            elif geometry.DEG2RAD * -67.5 <= self_angle < geometry.DEG2RAD * -22.5:
                return RoadObservation.ROAD_FRONT_RIGHT
            elif geometry.DEG2RAD * -22.5 <= self_angle < geometry.DEG2RAD * 22.5:
                return RoadObservation.ROAD_FRONT
            elif geometry.DEG2RAD * 22.5 <= self_angle < geometry.DEG2RAD * 67.5:
                return RoadObservation.ROAD_FRONT_LEFT
            elif geometry.DEG2RAD * 67.5 <= self_angle < geometry.DEG2RAD * 112.5:
                return RoadObservation.ROAD_LEFT
            elif geometry.DEG2RAD * 112.5 <= self_angle < geometry.DEG2RAD * 157.5:
                return RoadObservation.ROAD_REAR_LEFT
            elif geometry.DEG2RAD * 157.5 <= self_angle <= geometry.DEG2RAD * 180 or geometry.DEG2RAD * -180 < self_angle < geometry.DEG2RAD * -157.5:
                return RoadObservation.ROAD_REAR
            else:
                raise Exception("relative angle is not in the interval (-math.pi, math.pi]")

    def choose_crossing_orientation_action(self, state, condition, info):
        assert info
        assert 'actor_polygons' in info
        assert 'road_angles' in info

        active_orientation = self.active_orientation(state)
        road_observation = self.road_observation(info)
        on_road = road_observation is RoadObservation.ON_ROAD

        orientation_action = OrientationAction.NOOP
        if self.crossing_action is not None and not active_orientation and not self.entered_road and not on_road:  # crossing, not turning, and on origin pavement
            self.pavement_count += 1
        elif self.crossing_action is not None and not active_orientation and not self.entered_road and on_road:  # crossing and on road
            self.entered_road = True
        elif self.crossing_action is not None and self.entered_road and not on_road:  # crossing and on destination pavement
            self.reorient_count += 1
            if self.reorient_count >= self.pavement_count + self.delay:  # crossed on destination pavement for at least as long as origin pavement
                orientation_action = end_cross_road_action[self.crossing_action]
                self.reset()
        elif self.crossing_action is None and not active_orientation and not on_road and condition:  # not crossing and decided to cross
            self.crossing_action = cross_road_action[road_observation]
            orientation_action = self.crossing_action

        return orientation_action

    def choose_action(self, observation, info=None):
        raise NotImplementedError

    def process_feedback(self, previous_observation, action, observation, reward):
        raise NotImplementedError


class RandomConstrainedPedestrianAgent(RoadCrossingPedestrianAgent, RandomPedestrianAgent):  # base class order is important so that RoadCrossingPedestrianAgent.reset() is called rather than RandomPedestrianAgent.reset()
    def choose_action(self, state, info=None):
        assert info and 'actor_polygons' in info

        epsilon_valid = self.epsilon_valid()

        velocity_action = self.choose_random_velocity_action(state, epsilon_valid)
        orientation_action = self.choose_crossing_orientation_action(state, epsilon_valid, info)

        return velocity_action.value, orientation_action.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass


class ProximityPedestrianAgent(RoadCrossingPedestrianAgent):
    def __init__(self, distance_threshold, **kwargs):
        super().__init__(**kwargs)

        self.distance_threshold = distance_threshold

        self.moving = False

    def reset(self):
        super().reset()
        self.moving = False

    def proximity_satisfied(self, state):
        self_state = state[self.index]
        self_position = Point(self_state[0], self_state[1])
        ego_state = state[0]
        ego_position = Point(ego_state[0], ego_state[1])
        return self_position.distance(ego_position) < self.distance_threshold

    def choose_action(self, state, info=None):
        assert info and 'actor_polygons' in info

        if not self.moving and not self.active_velocity(state):
            velocity_action = VelocityAction.FAST
            self.moving = True
        else:
            velocity_action = VelocityAction.NOOP

        orientation_action = self.choose_crossing_orientation_action(state, self.proximity_satisfied(state), info)

        return velocity_action.value, orientation_action.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass


class ElectionPedestrianAgent(ProximityPedestrianAgent):
    pass


class QLearningAgent(RandomPedestrianAgent):
    # self.alpha is learning rate (should decrease over time)
    # self.gamma is discount factor (should be fixed over time?)
    # self.epsilon is exploration probability (should decrease over time)
    def __init__(self, ego_constants, self_constants, road_polgon, time_resolution, width, height, q_learning_config, **kwargs):
        super().__init__(epsilon=q_learning_config.epsilon, **kwargs)  # self.epsilon is exploration probability (should decrease over time)

        self.ego_constants = ego_constants
        self.self_constants = self_constants
        self.road_polgon = road_polgon
        self.time_resolution = time_resolution
        self.alpha = q_learning_config.alpha  # learning rate (should decrease over time)
        self.gamma = q_learning_config.gamma  # discount factor (should be fixed over time?)
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

        self.available_actions = [(velocity_action.value, orientation_action.value) for velocity_action in VelocityAction for orientation_action in OrientationAction]

    def reset(self):
        if self.log_file:
            self.log_file.info(f"{','.join(map(str, [self.feature_weights[feature] for feature in self.enabled_features]))}")

    def features(self, state, action):  # question: what does it mean for a feature to depend on an action and/or what does a Q value mean if it does not depend on an action?
        def make_actor_state(data):
            # assert len(data) == 8
            return DynamicActorState(
                position=Point(data[0], data[1]),
                velocity=data[2],
                orientation=data[3],
                acceleration=data[4],
                angular_velocity=data[5],
                target_velocity=data[6],
                target_orientation=data[7]
            )

        ego_state = make_actor_state(state[0])
        self_state = make_actor_state(state[self.index])

        ego_actor = Car(ego_state, self.ego_constants)
        self_actor = Pedestrian(self_state, self.self_constants)

        joint_action = [(VelocityAction.NOOP.value, OrientationAction.NOOP.value), action]

        spawn_actors = [ego_actor, self_actor]
        for i, spawn_actor in enumerate(spawn_actors):
            spawn_actor.step_action(joint_action, i)

        for i, spawn_actor in enumerate(spawn_actors):
            spawn_actor.step_dynamics(self.time_resolution)

        ego_position = ego_actor.state.position
        self_position = self_actor.state.position

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
            unnormalised_values["on_road"] = 1 if self_actor.bounding_box().intersects(self.road_polgon) else 0
        if self.feature_config.facing:
            unnormalised_values["facing"] = abs(geometry.Line(start=ego_position, end=self_position).orientation() - ego_actor.state.orientation)
        if self.feature_config.inverse_distance:
            x = unnormalised_values["distance"] if "distance" in unnormalised_values else self_position.distance(ego_position)
            unnormalised_values["inverse_distance"] = 1 - (x / self.x_max) ** self.n  # thanks to Ram Varadarajan

        normalised_values = {feature: normalise(feature_value, *self.feature_bounds[feature]) for feature, feature_value in unnormalised_values.items()}
        # print(pretty_float_list(unnormalised_values), pretty_float_list(normalised_values))
        return normalised_values

    def q_value(self, state, action):  # if features do not depend on action, then the Q value will not either
        feature_values = self.features(state, action)
        # assert len(feature_values) == self.num_featues
        q_value = sum(feature_value * self.feature_weights[feature] for feature, feature_value in feature_values.items())
        # assert math.isfinite(q_value)
        # print(pretty_float_list(feature_values), pretty_float_list(self.feature_weights), pretty_float(q_value))
        return q_value

    def choose_action(self, state, info=None):
        if self.epsilon_valid():
            velocity_action = self.choose_random_velocity_action(state, True)
            orientation_action = self.choose_random_orientation_action(state, True)
            return velocity_action.value, orientation_action.value
        else:
            best_actions = list()  # there may be multiple actions with max Q value
            max_q_value = -math.inf
            for action in self.available_actions:
                q_value = self.q_value(state, action)
                if q_value > max_q_value:
                    best_actions = [action]
                    max_q_value = q_value
                elif q_value == max_q_value:
                    best_actions.append(action)
            assert best_actions, "no best action(s) found"
            action = best_actions[0] if len(best_actions) == 1 else best_actions[self.np_random.choice(range(len(best_actions)))]
            return action

    def process_feedback(self, previous_state, action, state, reward):  # agent executed action in previous_state, and then arrived in state where it received reward
        q_value = self.q_value(previous_state, action)
        new_q_value = reward + self.gamma * max(self.q_value(state, action_prime) for action_prime in self.available_actions)
        q_value_gain = new_q_value - q_value
        for feature, feature_value in self.features(previous_state, action).items():
            self.feature_weights[feature] = self.feature_weights[feature] + self.alpha * q_value_gain * feature_value


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

        print(f"alpha={self.alpha}, epsilon={self.epsilon}, gamma={self.gamma}")
