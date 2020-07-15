from gym.utils import seeding

from library.actions import OrientationAction, VelocityAction
from library.environment import TrafficLightAction
from library.observations import OrientationObservation, VelocityObservation, RoadObservation, DistanceObservation


class Agent:
    def reset(self):
        raise NotImplementedError

    def choose_action(self, observation, action_space):
        raise NotImplementedError

    def process_feedback(self, previous_observation, action, observation, reward):
        raise NotImplementedError


class KeyboardAgent(Agent):
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

    def __init__(self):
        self.pending_velocity_action = VelocityAction.NOOP
        self.pending_orientation_action = OrientationAction.NOOP

    def reset(self):
        self.pending_velocity_action = VelocityAction.NOOP
        self.pending_orientation_action = OrientationAction.NOOP

    def choose_action(self, observation, action_space):
        velocity_observation_id, orientation_observation_id, _, _ = observation

        velocity_observation = VelocityObservation(velocity_observation_id)
        if velocity_observation is not VelocityObservation.ACTIVE:
            velocity_action = self.pending_velocity_action
            self.pending_velocity_action = VelocityAction.NOOP
        else:
            velocity_action = VelocityAction.NOOP

        orientation_observation = OrientationObservation(orientation_observation_id)
        if orientation_observation is not OrientationObservation.ACTIVE:
            orientation_action = self.pending_orientation_action
            self.pending_orientation_action = OrientationAction.NOOP
        else:
            orientation_action = OrientationAction.NOOP

        return velocity_action.value, orientation_action.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass

    def key_press(self, key, mod):
        if key in self.key_velocity_actions:
            self.pending_velocity_action = self.key_velocity_actions[key]
        if key in self.key_orientation_actions:
            self.pending_orientation_action = self.key_orientation_actions[key]


class RandomAgent(Agent):
    def __init__(self, epsilon=0.1, np_random=seeding.np_random(None)[0], **kwargs):
        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.np_random = np_random

    def epsilon_valid(self):
        return self.np_random.uniform(0.0, 1.0) < self.epsilon

    def reset(self):
        raise NotImplementedError

    def choose_action(self, observation, action_space):
        raise NotImplementedError

    def process_feedback(self, previous_observation, action, observation, reward):
        raise NotImplementedError


class NoopVehicleAgent(Agent):
    def reset(self):
        pass

    def choose_action(self, observation, action_space):
        return VelocityAction.NOOP.value, OrientationAction.NOOP.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass


class RandomVehicleAgent(RandomAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self):
        pass

    def choose_action(self, observation, action_space):
        velocity_observation_id, _, _, _ = observation

        epsilon_valid = self.epsilon_valid()

        velocity_observation = VelocityObservation(velocity_observation_id)
        if velocity_observation is not VelocityObservation.ACTIVE and epsilon_valid:
            velocity_action = self.np_random.choice([value for value in VelocityAction if value is not VelocityAction.STOP])
        else:
            velocity_action = VelocityAction.NOOP

        return velocity_action.value, OrientationAction.NOOP.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass


class RandomPedestrianAgent(RandomAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self):
        pass

    def choose_random_velocity_action(self, velocity_observation, condition):
        if velocity_observation is not VelocityObservation.ACTIVE and condition:
            # return self.np_random.choice([action for action in VelocityAction if action is not VelocityAction.STOP])
            return self.np_random.choice([action for action in VelocityAction])
        else:
            return VelocityAction.NOOP

    def choose_random_orientation_action(self, orientation_observation, condition):
        if orientation_observation is not OrientationObservation.ACTIVE and condition:
            return self.np_random.choice([action for action in OrientationAction])
        else:
            return OrientationAction.NOOP

    def choose_action(self, observation, action_space):
        velocity_observation_id, orientation_observation_id, _, _ = observation

        velocity_observation = VelocityObservation(velocity_observation_id)
        orientation_observation = OrientationObservation(orientation_observation_id)

        epsilon_valid = self.epsilon_valid()

        velocity_action = self.choose_random_velocity_action(velocity_observation, epsilon_valid)
        orientation_action = self.choose_random_orientation_action(orientation_observation, epsilon_valid)

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


class RoadCrossingPedestrianAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def choose_crossing_orientation_action(self, orientation_observation, road_observation, condition):
        orientation_action = OrientationAction.NOOP

        if self.crossing_action is not None and orientation_observation is not OrientationObservation.ACTIVE and not self.entered_road and road_observation is not RoadObservation.ON_ROAD:  # crossing, not turning, and on origin pavement
            self.pavement_count += 1
        elif self.crossing_action is not None and orientation_observation is not OrientationObservation.ACTIVE and not self.entered_road and road_observation is RoadObservation.ON_ROAD:  # crossing and on road
            self.entered_road = True
        elif self.crossing_action is not None and self.entered_road and road_observation is not RoadObservation.ON_ROAD:  # crossing and on destination pavement
            self.reorient_count += 1
            if self.reorient_count >= self.pavement_count + self.delay:  # crossed on destination pavement for at least as long as origin pavement
                orientation_action = end_cross_road_action[self.crossing_action]
                self.reset()
        elif self.crossing_action is None and orientation_observation is not OrientationObservation.ACTIVE and road_observation is not RoadObservation.ON_ROAD and condition:  # not crossing and decided to cross
            self.crossing_action = cross_road_action[road_observation]
            orientation_action = self.crossing_action

        return orientation_action

    def choose_action(self, observation, action_space):
        raise NotImplementedError

    def process_feedback(self, previous_observation, action, observation, reward):
        raise NotImplementedError


class RandomConstrainedPedestrianAgent(RoadCrossingPedestrianAgent, RandomPedestrianAgent):  # base class order is important so that RoadCrossingPedestrianAgent.reset() is called rather than RandomPedestrianAgent.reset()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def choose_action(self, observation, action_space):
        velocity_observation_id, orientation_observation_id, road_observation_id, _ = observation

        velocity_observation = VelocityObservation(velocity_observation_id)
        orientation_observation = OrientationObservation(orientation_observation_id)
        road_observation = RoadObservation(road_observation_id)

        epsilon_valid = self.epsilon_valid()

        velocity_action = self.choose_random_velocity_action(velocity_observation, epsilon_valid)
        orientation_action = self.choose_crossing_orientation_action(orientation_observation, road_observation, epsilon_valid)

        return velocity_action.value, orientation_action.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass


class ProximityPedestrianAgent(RoadCrossingPedestrianAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.moving = False

    def reset(self):
        super().reset()
        self.moving = False

    def choose_action(self, observation, action_space):
        velocity_observation_id, orientation_observation_id, road_observation_id, distance_observation_id = observation

        velocity_observation = VelocityObservation(velocity_observation_id)
        orientation_observation = OrientationObservation(orientation_observation_id)
        road_observation = RoadObservation(road_observation_id)
        distance_observation = DistanceObservation(distance_observation_id)

        if not self.moving and velocity_observation is not VelocityObservation.ACTIVE:
            velocity_action = VelocityAction.FAST
            self.moving = True
        else:
            velocity_action = VelocityAction.NOOP

        orientation_action = self.choose_crossing_orientation_action(orientation_observation, road_observation, distance_observation is DistanceObservation.SATISFIED)

        return velocity_action.value, orientation_action.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass


class ElectionPedestrianAgent(ProximityPedestrianAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RandomTrafficLightAgent(RandomAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self):
        pass

    def choose_action(self, observation, action_space):
        epsilon_valid = self.epsilon_valid()

        if epsilon_valid:
            action_id = action_space.sample()
            action = TrafficLightAction(action_id)
        else:
            action = TrafficLightAction.NOOP

        return action.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass
