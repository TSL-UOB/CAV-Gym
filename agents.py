from gym.utils import seeding

from environment import TurnAction, TrafficLightAction, AccelerationAction


class Agent:
    def reset(self):
        raise NotImplementedError

    def choose_action(self, observation, action_space):
        raise NotImplementedError

    def process_feedback(self, previous_observation, action, observation, reward):
        raise NotImplementedError


class HumanDynamicActorAgent(Agent):
    key_acceleration_actions = {
        32: {
            False: AccelerationAction.NEUTRAL,
            True: AccelerationAction.NEUTRAL
        },
        65362: {
            False: AccelerationAction.NORMAL_ACCELERATE,
            True: AccelerationAction.HARD_ACCELERATE
        },
        65364: {
            False: AccelerationAction.NORMAL_DECELERATE,
            True: AccelerationAction.HARD_DECELERATE
        }
    }
    key_turn_actions = {
        32: {
            False: TurnAction.NEUTRAL,
            True: TurnAction.NEUTRAL
        },
        65361: {
            False: TurnAction.NORMAL_LEFT,
            True: TurnAction.HARD_LEFT
        },
        65363: {
            False: TurnAction.NORMAL_RIGHT,
            True: TurnAction.HARD_RIGHT
        }
    }

    def __init__(self):
        self.acceleration_action = AccelerationAction.NEUTRAL
        self.turn_action = TurnAction.NEUTRAL

    def reset(self):
        self.acceleration_action = AccelerationAction.NEUTRAL
        self.turn_action = TurnAction.NEUTRAL

    def choose_action(self, observation, action_space):
        return self.acceleration_action.value, self.turn_action.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass

    def key_press(self, key, mod):
        if key in self.key_acceleration_actions:
            self.acceleration_action = self.key_acceleration_actions[key][mod == 644]
        if key in self.key_turn_actions:
            self.turn_action = self.key_turn_actions[key][mod == 644]


class RandomAgent(Agent):
    def __init__(self, epsilon=0.1, seed=None):
        self.np_random = None
        self.seed(seed)
        self.epsilon = epsilon

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        raise NotImplementedError

    def choose_action(self, observation, action_space):
        raise NotImplementedError

    def process_feedback(self, previous_observation, action, observation, reward):
        raise NotImplementedError


class RandomDynamicActorAgent(RandomAgent):
    def __init__(self, epsilon=0.1, seed=None):
        super().__init__(epsilon, seed)

        self.acceleration_action = AccelerationAction.NEUTRAL
        self.turn_action = TurnAction.NEUTRAL

    def reset(self):
        self.acceleration_action = AccelerationAction.NEUTRAL
        self.turn_action = TurnAction.NEUTRAL

    def choose_action(self, observation, action_space):
        if self.np_random.uniform(0.0, 1.0) < self.epsilon:
            acceleration_action_id, _ = action_space.sample()
            self.acceleration_action = AccelerationAction(acceleration_action_id)
        return self.acceleration_action.value, TurnAction.NEUTRAL.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass


class RandomTrafficLightAgent(RandomAgent):
    def __init__(self, epsilon=0.1, seed=None):
        super().__init__(epsilon, seed)

        self.action = TrafficLightAction.NOOP

    def reset(self):
        self.action = TrafficLightAction.NOOP

    def choose_action(self, observation, action_space):
        if self.np_random.uniform(0.0, 1.0) < self.epsilon:
            action_id = action_space.sample()
            self.action = TrafficLightAction(action_id)
        else:
            self.action = TrafficLightAction.NOOP
        return self.action.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass
