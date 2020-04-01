from gym.utils import seeding

from environment import ThrottleAction, TurnAction


class Agent:
    def reset(self):
        raise NotImplementedError()

    def choose_action(self, observation, action_space):
        raise NotImplementedError()

    def process_feedback(self, previous_observation, action, observation, reward):
        raise NotImplementedError()


class RandomAgent(Agent):
    def __init__(self, epsilon=0.1, seed=None):
        self.np_random = None
        self.seed(seed)
        self.epsilon = epsilon
        self.throttle_action = ThrottleAction.NEUTRAL
        self.turn_action = TurnAction.NEUTRAL

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.throttle_action = ThrottleAction.NEUTRAL
        self.turn_action = TurnAction.NEUTRAL

    def choose_action(self, observation, action_space):
        if self.np_random.uniform(0.0, 1.0) < self.epsilon:
            throttle_action_id, turn_action_id = action_space.sample()
            self.throttle_action = ThrottleAction(throttle_action_id)
            # self.turn_action = TurnAction(turn_action_id)
        return self.throttle_action.value, self.turn_action.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass


class HumanAgent(Agent):
    key_throttle_actions = {
        32: {
            False: ThrottleAction.NEUTRAL,
            True: ThrottleAction.NEUTRAL
        },
        65362: {
            False: ThrottleAction.ACCELERATE,
            True: ThrottleAction.HARD_ACCELERATE
        },
        65364: {
            False: ThrottleAction.DECELERATE,
            True: ThrottleAction.HARD_DECELERATE
        }
    }
    key_turn_actions = {
        32: {
            False: TurnAction.NEUTRAL,
            True: TurnAction.NEUTRAL
        },
        65361: {
            False: TurnAction.LEFT,
            True: TurnAction.HARD_LEFT
        },
        65363: {
            False: TurnAction.RIGHT,
            True: TurnAction.HARD_RIGHT
        }
    }

    def __init__(self):
        self.throttle_action = ThrottleAction.NEUTRAL
        self.turn_action = TurnAction.NEUTRAL

    def reset(self):
        self.throttle_action = ThrottleAction.NEUTRAL
        self.turn_action = TurnAction.NEUTRAL

    def choose_action(self, observation, action_space):
        return self.throttle_action.value, self.turn_action.value

    def process_feedback(self, previous_observation, action, observation, reward):
        pass

    def key_press(self, key, mod):
        if key in self.key_throttle_actions:
            self.throttle_action = self.key_throttle_actions[key][mod == 644]
        if key in self.key_turn_actions:
            self.turn_action = self.key_turn_actions[key][mod == 644]
