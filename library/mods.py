import json
import math
import os
import time

from gym import error
from gym.envs.classic_control.rendering import LineStyle, glEnable, glLineStipple, GL_LINE_STIPPLE, FilledPolygon, PolyLine
from gym.utils import atomic_write
from gym.utils.json_utils import json_encode_np


class JointStatsRecorder(object):
    """
    This is a modified version Gym's StatsRecorder class to support joint rewards.
    """
    def __init__(self, directory, file_prefix, agents, autoreset=False, env_id=None):
        self.autoreset = autoreset
        self.env_id = env_id

        self.initial_reset_timestamp = None
        self.directory = directory
        self.file_prefix = file_prefix
        self.agents = agents
        self.episode_lengths = []
        self.episode_rewards = []
        self.episode_types = [] # experimental addition
        self._type = 't'
        self.timestamps = []
        self.steps = None
        self.total_steps = 0
        self.rewards = None

        self.done = None
        self.closed = False

        filename = '{}.stats.json'.format(self.file_prefix)
        self.path = os.path.join(self.directory, filename)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        if type not in ['t', 'e']:
            raise error.Error('Invalid episode type {}: must be t for training or e for evaluation', type)
        self._type = type

    def before_step(self, action):
        assert not self.closed

        if self.done:
            raise error.ResetNeeded("Trying to step environment which is currently done. While the monitor is active for {}, you cannot step beyond the end of an episode. Call 'scenarios.reset()' to start the next episode.".format(self.env_id))
        elif self.steps is None:
            raise error.ResetNeeded("Trying to step an environment before reset. While the monitor is active for {}, you must call 'scenarios.reset()' before taking an initial step.".format(self.env_id))

    def after_step(self, observation, reward, done, info):
        self.steps += 1
        self.total_steps += 1
        self.rewards = [previous_value + value for previous_value, value in zip(self.rewards, reward)]
        self.done = done

        if done:
            self.save_complete()

        if done:
            if self.autoreset:
                self.before_reset()
                self.after_reset(observation)

    def before_reset(self):
        assert not self.closed

        if self.done is not None and not self.done and self.steps > 0:
            raise error.Error("Tried to reset environment which is not done. While the monitor is active for {}, you cannot call reset() unless the episode is over.".format(self.env_id))

        self.done = False
        if self.initial_reset_timestamp is None:
            self.initial_reset_timestamp = time.time()

    def after_reset(self, observation):
        self.steps = 0
        self.rewards = [0 for _ in range(self.agents)]
        # We write the type at the beginning of the episode. If a user
        # changes the type, it's more natural for it to apply next
        # time the user calls reset().
        self.episode_types.append(self._type)

    def save_complete(self):
        if self.steps is not None:
            self.episode_lengths.append(self.steps)
            self.episode_rewards.append([float(reward) for reward in self.rewards])
            self.timestamps.append(time.time())

    def close(self):
        self.flush()
        self.closed = True

    def flush(self):
        if self.closed:
            return

        with atomic_write.atomic_write(self.path) as f:
            json.dump({
                'initial_reset_timestamp': self.initial_reset_timestamp,
                'timestamps': self.timestamps,
                'episode_lengths': self.episode_lengths,
                'episode_rewards': self.episode_rewards,
                'episode_types': self.episode_types,
            }, f, default=json_encode_np)


def make_joint_stats_recorder(env, agents):
    return JointStatsRecorder(env.directory, '{}.episode_batch.{}'.format(env.file_prefix, env.file_infix), agents, autoreset=env.env_semantics_autoreset, env_id='(unknown)' if env.env.spec is None else env.env.spec.id)


class FactoredLineStyle(LineStyle):
    def __init__(self, style, factor):
        super().__init__(style)
        self.factor = factor

    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(self.factor, self.style)


def make_circle(x, y, radius, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2*math.pi*i / res
        points.append((x + math.cos(ang)*radius, y + math.sin(ang)*radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)