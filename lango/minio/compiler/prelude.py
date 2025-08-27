# NOTE: do not remove any imports, they are used in the generated code
import math
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

# Runtime support functions
F = TypeVar("F", bound=Callable[..., Any])


def curry(func: F) -> F:
    @wraps(func)
    def curried(*args: Any) -> Any:
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more_args: curried(*args + more_args)

    return curried  # type: ignore


def minio_show(value: Any) -> str:
    match value:
        case bool():
            return "True" if value else "False"
        case str():
            return f'"{value}"'
        case list():
            elements = [minio_show(x) for x in value]
            return "[" + ",".join(elements) + "]"
        case float():
            if value == float("inf"):
                return "Infinity"
            if value == float("-inf"):
                return "-Infinity"
            return str(value)
        case _:
            return str(value)


def minio_put_str(s: Any) -> None:
    match s:
        case str():
            s = s.encode().decode("unicode_escape")
    print(s, end="")


def minio_error(message: str) -> Any:
    raise RuntimeError(f"Runtime error: {message}")
