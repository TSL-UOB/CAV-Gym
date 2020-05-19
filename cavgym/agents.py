from gym.utils import seeding

from cavgym.actions import OrientationAction, VelocityAction
from cavgym.environment import TrafficLightAction
from cavgym.observations import OrientationObservation, VelocityObservation


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
        velocity_observation_id, orientation_observation_id = observation

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
    def __init__(self, epsilon=0.1, np_random=seeding.np_random(None)[0]):
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


class RandomVehicleAgent(RandomAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self):
        pass

    def choose_action(self, observation, action_space):
        velocity_observation_id, _ = observation

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

    def choose_action(self, observation, action_space):
        velocity_observation_id, orientation_observation_id = observation

        epsilon_valid = self.epsilon_valid()

        velocity_observation = VelocityObservation(velocity_observation_id)
        if velocity_observation is not VelocityObservation.ACTIVE and epsilon_valid:
            velocity_action_id, _ = action_space.sample()
            velocity_action = VelocityAction(velocity_action_id)
        else:
            velocity_action = VelocityAction.NOOP

        orientation_observation = OrientationObservation(orientation_observation_id)
        if orientation_observation is not OrientationObservation.ACTIVE and epsilon_valid:
            _, orientation_action_id = action_space.sample()
            orientation_action = OrientationAction(orientation_action_id)
        else:
            orientation_action = OrientationAction.NOOP

        return velocity_action.value, orientation_action.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass


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
