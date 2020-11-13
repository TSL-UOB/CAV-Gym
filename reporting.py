import logging
import math
import pathlib
import sys
from dataclasses import dataclass
from enum import Enum


class Verbosity(Enum):  # cannot be included in config.py due to circular imports
    INFO = "info"
    DEBUG = "debug"
    SILENT = "silent"

    def __str__(self):
        return self.value


def pretty_str_iter(str_iter):
    return f"{', '.join(map(str, str_iter))}"


# def pretty_str_tuple(str_tuple):
#     return f"({pretty_str_iter(str_tuple)})"


def pretty_str_list(str_list):
    return f"[{pretty_str_iter(str_list)}]"


def pretty_str_set(str_set):
    return f"{{{pretty_str_iter(str_set)}}}"


def pretty_float(float_value, decimal_places=2):
    return f"{round(float_value, decimal_places):g}"


def pretty_float_iter(float_iter, **kwargs):
    return ", ".join(pretty_float(float_value, **kwargs) for float_value in float_iter)


# def pretty_float_tuple(float_list, **kwargs):
#     return f"({pretty_float_iter(float_list, **kwargs)})"


def pretty_float_list(float_list, **kwargs):
    return f"[{pretty_float_iter(float_list, **kwargs)}]"


class CustomLogRecord(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.code = f"{self.filename}:{self.funcName}"  # new variable used to align existing variables as group


logging.setLogRecordFactory(CustomLogRecord)
console_formatter = logging.Formatter("%(levelname)-7s %(relativeCreated)-7d %(code)-27s %(message)s")
file_formatter = logging.Formatter("%(message)s")


def make_console_handler(destination, event_filter):
    handler = logging.StreamHandler(destination)
    handler.addFilter(event_filter)
    handler.setFormatter(console_formatter)
    return handler


def get_console(verbosity):
    console = logging.getLogger("cavgym.console")

    if verbosity is Verbosity.DEBUG:
        console.setLevel(logging.DEBUG)
    elif verbosity is Verbosity.INFO:
        console.setLevel(logging.INFO)
    elif verbosity is Verbosity.SILENT:
        console.setLevel(logging.WARNING)
    else:
        raise NotImplementedError

    if not console.handlers:
        info_below_handler = make_console_handler(sys.stdout, lambda record: record.levelno <= logging.INFO)  # redirect INFO events and below to stdout (to avoid duplicate events)
        warning_above_handler = make_console_handler(sys.stderr, lambda record: record.levelno > logging.INFO)  # redirect WARNING events and above to stderr (to avoid duplicate events)

        console.addHandler(info_below_handler)
        console.addHandler(warning_above_handler)

    return console


def make_file_logger(path):
    logger = logging.getLogger(path)
    if not logger.handlers:  # logger has already been configured
        path_obj = pathlib.Path(path)
        directory_obj = path_obj.parent
        directory_obj.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(f"{path_obj}")
        handler.setFormatter(file_formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def get_episode_file_logger(path):
    return make_file_logger(path)


def get_run_file_logger(path):
    return make_file_logger(path)


def get_agent_file_logger(path):
    return make_file_logger(path)


class LogMessage:
    def console_message(self):
        raise NotImplementedError

    def file_message(self):
        raise NotImplementedError


@dataclass(frozen=True)
class TimeResults:
    timesteps: int
    start_time: float  # fractional seconds
    end_time: float  # fractional seconds
    resolution: float  # 1 / frequency

    def __post_init__(self):
        assert self.timesteps > 0
        assert self.start_time < self.end_time

    def runtime(self):
        return (self.end_time - self.start_time) * 1000  # fractional milliseconds

    def simulation_speed(self):
        return (self.timesteps * self.resolution * 1000) / self.runtime()  # fractional milliseconds


@dataclass(frozen=True)
class EpisodeResults(LogMessage):
    index: int
    time: TimeResults
    completed: bool
    interesting: bool
    score: int

    def __post_init__(self):
        assert self.interesting or math.isnan(self.score)

    def console_message(self):
        episode_status = "completed" if self.completed else "terminated"
        test_status = f"interesting test with score {self.score}" if self.interesting else "uninteresting test"
        return f"episode {self.index} {episode_status} after {self.time.timesteps} timestep(s) in {pretty_float(self.time.runtime(), decimal_places=0)} ms (*{pretty_float(self.time.simulation_speed())} real-time), {test_status}"

    def file_message(self):
        return f"{self.index},{self.time.timesteps},{self.time.runtime()},{1 if self.interesting else 0},{self.score}"


@dataclass(frozen=True)
class Interval:
    value: float
    error: float

    def __iter__(self):
        yield self.value
        yield self.error

    def pretty(self, **kwargs):
        return f"{pretty_float(self.value, **kwargs)} ± {pretty_float(self.error, **kwargs)}"


def confidence_interval(data, alpha=0.05):  # 95% confidence interval
    data_length = len(data)
    if data_length <= 1:  # variance (and thus data_sem) requires at least 2 data points
        nan = float("nan")
        return Interval(value=nan, error=nan)
    else:
        # noinspection PyPackageRequirements
        from scipy import stats  # dependency of gym
        data_mean = sum(data) / data_length
        # data_sem = statistics.stdev(data) / math.sqrt(data_length)  # use data_sem = statistics.pstdev(data) if data is population
        # data_sem = np.std(data, ddof=1) / math.sqrt(data_length)  # use data_sem = np.std(data) / math.sqrt(data_length) if data is population
        data_sem = stats.sem(data)  # use data_sem = scipy.stats.sem(data, ddof=0) if data is population
        data_df = data_length - 1
        data_error = data_sem * stats.t.isf(alpha / 2, data_df)  # equivalent to data_error = data_sem * -stats.t.ppf(alpha / 2, data_df)
        # return data_mean - data_error, data_mean + data_error
        # data_ci_min, data_ci_max = stats.t.interval(1 - alpha, data_df, loc=data_mean, scale=data_sem)  # the interval itself
        return Interval(value=data_mean, error=data_error)


@dataclass(frozen=True)
class InterestingTestResults:
    count: int
    total_timesteps: int
    total_runtime: float
    total_score: int
    confidence_timesteps: Interval
    confidence_runtime: Interval
    confidence_score: Interval

    def __post_init__(self):
        assert self.count > 0 or (math.isnan(self.total_timesteps) and math.isnan(self.total_runtime) and math.isnan(self.total_score))


@dataclass(frozen=True)
class RunResults(LogMessage):
    episodes: int
    time: TimeResults
    interesting_tests: InterestingTestResults

    def __post_init__(self):
        assert self.episodes > 0
        assert 0 <= self.interesting_tests.count <= self.episodes
        if self.interesting_tests.count > 0:
            assert 0 <= self.interesting_tests.total_timesteps <= self.episodes * self.time.timesteps

    def console_message(self):
        test_status = f"{self.interesting_tests.count} interesting test(s) with {self.interesting_tests.confidence_timesteps.pretty(decimal_places=0)} timestep(s), {self.interesting_tests.confidence_runtime.pretty(decimal_places=0)} ms runtime, and {self.interesting_tests.confidence_score.pretty(decimal_places=0)} score" if self.interesting_tests.count > 0 else "no interesting test(s)"
        return f"run completed after {self.episodes} episode(s) and {self.time.timesteps} timestep(s) in {pretty_float(self.time.runtime(), decimal_places=0)} ms (*{pretty_float(self.time.simulation_speed())} real-time), {test_status}"

    def file_message(self):
        return f"{self.episodes},{self.time.timesteps},{self.time.runtime()},{self.interesting_tests.count},{self.interesting_tests.confidence_timesteps.value},{self.interesting_tests.confidence_timesteps.error},{self.interesting_tests.confidence_runtime.value},{self.interesting_tests.confidence_runtime.error},{self.interesting_tests.confidence_score.value},{self.interesting_tests.confidence_score.error}"


def analyse_episode(index, start_time, end_time, timesteps, env_info, run_config, env):  # can run_config and env be removed?
    assert 1 <= timesteps <= run_config.max_timesteps
    completed = timesteps == run_config.max_timesteps
    interesting = 'winner' in env_info and env_info['winner'] > 0  # otherwise, draw or ego wons
    score = -sum(env.episode_liveness[1:]) if interesting else float("nan")
    return EpisodeResults(
        index=index,
        time=TimeResults(
            timesteps=timesteps,
            start_time=start_time,
            end_time=end_time,
            resolution=env.time_resolution
        ),
        completed=completed,
        interesting=interesting,
        score=score
    )


def analyse_run(episode_data, start_time, end_time, run_config, env):  # can run_config and env be removed?
    interesting_test_data = [row for row in episode_data if row.interesting]
    interesting_test_data_timesteps = [row.time.timesteps for row in interesting_test_data]
    interesting_test_data_runtime = [row.time.runtime() for row in interesting_test_data]
    interesting_test_data_score = [row.score for row in interesting_test_data]
    nan = float("nan")
    return RunResults(
        episodes=run_config.episodes,
        time=TimeResults(
            timesteps=sum([row.time.timesteps for row in episode_data]),
            start_time=start_time,
            end_time=end_time,
            resolution=env.time_resolution
        ),
        interesting_tests=InterestingTestResults(
            count=len(interesting_test_data),
            total_timesteps=sum(interesting_test_data_timesteps) if interesting_test_data else nan,
            total_runtime=sum(interesting_test_data_runtime) if interesting_test_data else nan,
            total_score=sum(interesting_test_data_score) if interesting_test_data else nan,
            confidence_timesteps=confidence_interval(interesting_test_data_timesteps),
            confidence_runtime=confidence_interval(interesting_test_data_runtime),
            confidence_score=confidence_interval(interesting_test_data_score),
        )
    )
