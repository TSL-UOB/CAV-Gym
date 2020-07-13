import logging
import math
import pathlib
import sys
from dataclasses import dataclass
from enum import Enum


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


# def pretty_float_iter(float_iter, **kwargs):
#     return ", ".join(pretty_float(float_value, **kwargs) for float_value in float_iter)
#
#
# def pretty_float_tuple(float_list, **kwargs):
#     return f"({pretty_float_iter(float_list, **kwargs)})"
#
#
# def pretty_float_list(float_list, **kwargs):
#     return f"[{pretty_float_iter(float_list, **kwargs)}]"


class CustomLogRecord(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.code = f"{self.filename}:{self.funcName}"  # new variable used to align existing variables as group


logging.setLogRecordFactory(CustomLogRecord)
console_formatter = logging.Formatter("%(levelname)-7s %(relativeCreated)-7d %(code)-27s %(message)s")
file_formatter = logging.Formatter("%(message)s")


class Verbosity(Enum):
    INFO = "info"
    DEBUG = "debug"
    SILENT = "silent"

    def __str__(self):
        return self.value


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


def make_file_logger(name, directory, filename):
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.handlers = list()
    else:
        logger.setLevel(logging.INFO)
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(f"{directory}/{filename}")
        handler.setFormatter(file_formatter)
        logger.addHandler(handler)
    return logger


def get_file_loggers(path):
    episode_file = make_file_logger("cavgym.file.episodes", path, "episodes.log")
    run_file = make_file_logger("cavgym.file.run", path, "run.log")
    return episode_file, run_file


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
    successful: bool
    score: int

    def __post_init__(self):
        assert self.successful or not self.score

    def console_message(self):
        episode_status = "completed" if self.completed else "terminated"
        test_status = f"successful test with score {self.score}" if self.successful else "unsuccessful test"
        return f"episode {self.index} {episode_status} after {self.time.timesteps} timestep(s) in {pretty_float(self.time.runtime(), decimal_places=0)} ms ({pretty_float(self.time.simulation_speed())}:1 real-time), {test_status}"

    def file_message(self):
        return f"{self.index},{self.time.timesteps},{self.time.runtime()},{1 if self.successful else 0},{self.score}"


@dataclass(frozen=True)
class Interval:
    min: float
    max: float

    def __iter__(self):
        yield self.min
        yield self.max

    def pretty(self, **kwargs):
        return f"({pretty_float(self.min, **kwargs)}, {pretty_float(self.max, **kwargs)})"


def confidence_interval(data, alpha=0.05):  # 95% confidence interval
    data_length = len(data)
    if data_length <= 1:  # variance (and thus data_sem) requires at least 2 data points
        nan = float("nan")
        return Interval(min=nan, max=nan)
    else:
        from scipy import stats
        data_mean = sum(data) / data_length
        # data_sem = statistics.stdev(data) / math.sqrt(data_length)  # use data_sem = statistics.pstdev(data) if data is population
        # data_sem = np.std(data, ddof=1) / math.sqrt(data_length)  # use data_sem = np.std(data) / math.sqrt(data_length) if data is population
        data_sem = stats.sem(data)  # use data_sem = scipy.stats.sem(data, ddof=0) if data is population
        data_df = data_length - 1
        # data_error = data_sem * stats.t.isf(alpha / 2, data_df)  # equivalent to data_error = data_sem * -stats.t.ppf(alpha / 2, data_df)
        # return data_mean - data_error, data_mean + data_error
        data_ci_min, data_ci_max = stats.t.interval(1 - alpha, data_df, loc=data_mean, scale=data_sem)
        return Interval(min=data_ci_min, max=data_ci_max)


@dataclass(frozen=True)
class SuccessfulTestResults:
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
    successful_tests: SuccessfulTestResults

    def __post_init__(self):
        assert self.episodes > 0
        assert 0 <= self.successful_tests.count <= self.episodes
        if self.successful_tests.count > 0:
            assert 0 <= self.successful_tests.total_timesteps <= self.episodes * self.time.timesteps

    def console_message(self):
        test_status = f"{self.successful_tests.count} successful test(s) with timestep confidence {self.successful_tests.confidence_timesteps.pretty(decimal_places=0)}, runtime confidence {self.successful_tests.confidence_runtime.pretty(decimal_places=0)}, and score confidence {self.successful_tests.confidence_score.pretty(decimal_places=0)}" if self.successful_tests.count > 0 else "no successful test(s)"
        return f"run completed after {self.episodes} episode(s) and {self.time.timesteps} timestep(s) in {pretty_float(self.time.runtime(), decimal_places=0)} ms ({pretty_float(self.time.simulation_speed())}:1 real-time), {test_status}"

    def file_message(self):
        return f"{self.episodes},{self.time.timesteps},{self.time.runtime()},{self.successful_tests.count},{self.successful_tests.confidence_timesteps.min},{self.successful_tests.confidence_timesteps.max},{self.successful_tests.confidence_runtime.min},{self.successful_tests.confidence_runtime.max},{self.successful_tests.confidence_score.min},{self.successful_tests.confidence_score.max}"


def analyse_episode(index, start_time, end_time, timesteps, env_info, run_config, env):  # can run_config and env be removed?
    assert 1 <= timesteps <= run_config.max_timesteps
    completed = timesteps == run_config.max_timesteps
    assert 'pedestrian' in env_info
    pedestrian_index = env_info['pedestrian']
    successful = pedestrian_index is not None
    score = env.episode_liveness[pedestrian_index] * -5 if successful else None
    return EpisodeResults(
        index=index,
        time=TimeResults(
            timesteps=timesteps,
            start_time=start_time,
            end_time=end_time,
            resolution=env.time_resolution
        ),
        completed=completed,
        successful=successful,
        score=score
    )


episode_data = list()


def analyse_run(start_time, end_time, run_config, env):  # can run_config and env be removed?
    successful_test_data = [row for row in episode_data if row.successful]
    successful_test_data_timesteps = [row.time.timesteps for row in successful_test_data]
    successful_test_data_runtime = [row.time.runtime() for row in successful_test_data]
    successful_test_data_score = [row.score for row in successful_test_data]
    nan = float("nan")
    return RunResults(
        episodes=run_config.episodes,
        time=TimeResults(
            timesteps=sum([row.time.timesteps for row in episode_data]),
            start_time=start_time,
            end_time=end_time,
            resolution=env.time_resolution
        ),
        successful_tests=SuccessfulTestResults(
            count=len(successful_test_data),
            total_timesteps=sum(successful_test_data_timesteps) if successful_test_data else nan,
            total_runtime=sum(successful_test_data_runtime) if successful_test_data else nan,
            total_score=sum(successful_test_data_score) if successful_test_data else nan,
            confidence_timesteps=confidence_interval(successful_test_data_timesteps),
            confidence_runtime=confidence_interval(successful_test_data_runtime),
            confidence_score=confidence_interval(successful_test_data_score),
        )
    )
