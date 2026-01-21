import functools
import time
from enum import StrEnum
from typing import Any

import polars as pl


class StratificationAlgorithms(StrEnum):
    minimal = "minimal"
    hierarchical = "hierarchical"


def log_function_call(logger):
    def log_function_call_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args_repr = [polars_repr(a) for a in args]
            kwargs_repr = [f"{k}={polars_repr(v)}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)

            logger.info(f"Calling {func.__name__}({signature})")
            start_time = time.time()

            result = func(*args, **kwargs)

            end_time = time.time()
            minutes, seconds = divmod(end_time - start_time, 60)

            logger.info(
                f"Finished {func.__name__} in {int(minutes)}m {seconds:.2f}s. "
                f"Results: {polars_repr(result)}"
            )
            return result

        return wrapper

    return log_function_call_decorator


def polars_repr(x: Any):  # because polars otherwise has quite verbose repr
    if isinstance(x, pl.DataFrame):
        return f"<polars.DataFrame: columns={x.columns}, shape={x.shape}>"
    elif isinstance(x, pl.Series):
        return f"<polars.Series: name='{x.name}', shape={x.shape}>"
    elif isinstance(x, tuple) or isinstance(x, list):
        return ", ".join(map(polars_repr, x))
    else:
        return repr(x)
