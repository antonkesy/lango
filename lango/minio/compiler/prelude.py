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


def minio_putStr(s: Any) -> None:
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
