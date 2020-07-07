def pretty_str_iter(str_iter):
    return f"{', '.join(map(str, str_iter))}"


def pretty_str_tuple(str_tuple):
    return f"({pretty_str_iter(str_tuple)})"


def pretty_str_list(str_list):
    return f"[{pretty_str_iter(str_list)}]"


def pretty_str_set(str_set):
    return f"{{{pretty_str_iter(str_set)}}}"


def pretty_float(float_value, decimal_places=2):
    return f"{round(float_value, decimal_places):g}"


def pretty_float_iter(float_iter, **kwargs):
    return ", ".join(pretty_float(float_value, **kwargs) for float_value in float_iter)


def pretty_float_tuple(float_list, **kwargs):
    return f"({pretty_float_iter(float_list, **kwargs)})"


def pretty_float_list(float_list, **kwargs):
    return f"[{pretty_float_iter(float_list, **kwargs)}]"
