import math

from examples.agents.template import NoopAgent
from examples.targets import TargetVelocity, TargetOrientation
from library.bodies import DynamicBodyState
from library.geometry import Point

TARGET_ERROR = 0.000000000000001
ACTION_ERROR = 0.000000000000001


def make_body_state(env_state, index):
    body_state = env_state[index]
    assert len(body_state) == 4
    return DynamicBodyState(
        position=Point(
            x=float(body_state[0]),
            y=float(body_state[1])
        ),
        velocity=float(body_state[2]),
        orientation=float(body_state[3])
    )


def make_throttle_action(body_state, body_constants, time_resolution, target_velocity, noop_action):
    if target_velocity is None:
        throttle_action = noop_action[0]
    else:
        diff = (target_velocity - body_state.velocity) / time_resolution
        throttle_action = min(body_constants.max_throttle, max(body_constants.min_throttle, diff))
    return throttle_action


def make_steering_action(body_state, body_constants, time_resolution, target_orientation, noop_action):
    if body_state.velocity == 0 or target_orientation is None:
        steering_action = noop_action[1]
    else:
        target_turn_angle = math.atan2(math.sin(target_orientation - body_state.orientation), math.cos(target_orientation - body_state.orientation))

        constant_steering_action = body_constants.min_steering_angle if target_turn_angle < 0 else body_constants.max_steering_angle
        max_turn_angle = (-1 if constant_steering_action < 0 else 1) * 2 * time_resolution * body_state.velocity / math.sqrt(body_constants.wheelbase**2 * (1 + 4 / math.tan(constant_steering_action)**2))

        def calc_steering_angle(turn_angle):
            return (-1 if turn_angle < 0 else 1) * math.atan(2 * body_constants.wheelbase * math.sqrt(turn_angle**2 / (4 * body_state.velocity**2 * time_resolution**2 - body_constants.wheelbase**2 * turn_angle**2)))

        if target_turn_angle / max_turn_angle > 1:
            steering_action = calc_steering_angle(max_turn_angle)
        else:
            steering_action = calc_steering_angle(target_turn_angle)
    return steering_action


class TargetAgent(NoopAgent):
    def __init__(self, body, time_resolution, **kwargs):
        super().__init__(**kwargs)

        self.body = body
        self.time_resolution = time_resolution

        self.target_velocity_mapping = {
            TargetVelocity.MIN: self.body.constants.min_velocity,
            TargetVelocity.MID: (self.body.constants.min_velocity + self.body.constants.max_velocity) / 2,
            TargetVelocity.MAX: self.body.constants.max_velocity
        }

    def reset(self):
        self.body.target_velocity = None
        self.body.target_orientation = None

    def choose_action(self, state, action_space, info=None):
        body_state = make_body_state(state, self.index)
        throttle_action = make_throttle_action(body_state, self.body.constants, self.time_resolution, self.body.target_velocity, self.noop_action)
        steering_action = make_steering_action(body_state, self.body.constants, self.time_resolution, self.body.target_orientation, self.noop_action)

        # account for minor (precision) error only: large error indicates issue with logic
        if throttle_action < action_space.low[0] or throttle_action > action_space.high[0]:
            assert abs(action_space.low[0] - throttle_action) < ACTION_ERROR or abs(action_space.high[0] - throttle_action) < ACTION_ERROR
            throttle_action = min(self.body.constants.max_throttle, max(self.body.constants.min_throttle, throttle_action))
        if steering_action < action_space.low[1] or steering_action > action_space.high[1]:
            assert abs(action_space.low[0] - throttle_action) < ACTION_ERROR or abs(action_space.high[0] - throttle_action) < ACTION_ERROR
            steering_action = min(self.body.constants.max_steering_angle, max(self.body.constants.min_steering_angle, steering_action))

        return [throttle_action, steering_action]

    def process_feedback(self, previous_state, action, state, reward):
        body_state = make_body_state(state, self.index)

        if self.body.target_velocity is not None:
            if abs(self.body.target_velocity - body_state.velocity) < TARGET_ERROR:
                self.body.target_velocity = None
        if self.body.target_orientation is not None:
            diff = self.body.target_orientation - body_state.orientation
            if abs(math.atan2(math.sin(diff), math.cos(diff))) < TARGET_ERROR:
                self.body.target_orientation = None


key_target_velocity = {
    65365: TargetVelocity.MAX,  # page up
    65366: TargetVelocity.MID,  # page down
    65367: TargetVelocity.MIN,  # end
}

key_target_orientation = {
    65361: TargetOrientation.WEST,  # arrow left
    65362: TargetOrientation.NORTH,  # arrow up
    65363: TargetOrientation.EAST,  # arrow right
    65364: TargetOrientation.SOUTH,  # arrow down

    65457: TargetOrientation.SOUTH_WEST,  # numpad 1
    65458: TargetOrientation.SOUTH,  # numpad 2
    65459: TargetOrientation.SOUTH_EAST,  # numpad 3
    65460: TargetOrientation.WEST,  # numpad 4
    65462: TargetOrientation.EAST,  # numpad 6
    65463: TargetOrientation.NORTH_WEST,  # numpad 7
    65464: TargetOrientation.NORTH,  # numpad 8
    65465: TargetOrientation.NORTH_EAST  # numpad 9
}


class KeyboardAgent(TargetAgent):
    def key_press(self, key, _mod):
        if key in key_target_velocity:
            self.body.target_velocity = self.target_velocity_mapping[key_target_velocity[key]]
        elif key in key_target_orientation:
            self.body.target_orientation = key_target_orientation[key].value
