# NOTE: do not remove any imports, they are used in the generated code
import math
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

# Runtime support functions


def minio_show(value: Any) -> str:
    match value:
        case ("char", char_value):
            return f"'{char_value}'"
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
        case ("char", char_value):
            print(char_value, end="")
        case str():
            s = s.encode().decode("unicode_escape")
            print(s, end="")
        case _:
            print(s, end="")


def minio_error(message: str) -> Any:
    raise RuntimeError(f"Runtime error: {message}")


# Built-in mathematical functions
def minio_mod(x: int, y: int) -> int:
    if y == 0:
        raise RuntimeError("Runtime error: divide by zero")
    return x % y


def minio_elem(value: Any, lst: List[Any]) -> bool:
    return value in lst


def minio_map(func: Callable[[Any], Any], lst: list) -> list:
    """Built-in map function - applies function to each element of list."""
    return [func(item) for item in lst]


# Aliases for built-in functions so they can be called by their original names
show = minio_show
putStr = minio_put_str
error = minio_error
mod = minio_mod
elem = minio_elem
map = minio_map
