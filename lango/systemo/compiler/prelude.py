# NOTE: do not remove any imports, they are used in the generated code
import math
import sys
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

# Runtime support functions
F = TypeVar("F", bound=Callable[..., Any])

# Mathematical constants used in systemo
NaN = float("nan")
Infinity = float("inf")


def curry(func: F) -> F:
    @wraps(func)
    def curried(*args: Any) -> Any:
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more_args: curried(*args + more_args)

    return curried  # type: ignore


def systemo_error(message: str) -> Any:
    raise RuntimeError(f"Runtime error: {message}")


# Primitive integer functions
def primIntAdd(x: int, y: int) -> int:
    return x + y


def primIntSub(x: int, y: int) -> int:
    return x - y


def primIntMul(x: int, y: int) -> int:
    return x * y


def primIntDiv(x: int, y: int) -> float:
    if y == 0:
        if x == 0:
            return float("nan")  # NaN
        return float("inf") if x > 0 else float("-inf")  # Infinity
    return x / y


def primIntPow(x: int, y: int) -> int:
    return x**y


def primIntNeg(x: int) -> int:
    return -x


def primIntLt(x: int, y: int) -> bool:
    return x < y


def primIntLe(x: int, y: int) -> bool:
    return x <= y


def primIntGt(x: int, y: int) -> bool:
    return x > y


def primIntGe(x: int, y: int) -> bool:
    return x >= y


def primIntEq(x: int, y: int) -> bool:
    return x == y


def primIntShow(x: int) -> str:
    return str(x)


# Primitive float functions
def primFloatAdd(x: float, y: float) -> float:
    return x + y


def primFloatSub(x: float, y: float) -> float:
    return x - y


def primFloatMul(x: float, y: float) -> float:
    return x * y


def primFloatDiv(x: float, y: float) -> float:
    if y == 0.0:
        if x == 0.0:
            return float("nan")  # NaN
        return float("inf") if x > 0 else float("-inf")  # Infinity
    return x / y


def primFloatPow(x: float, y: float) -> float:
    return x**y


def primFloatNeg(x: float) -> float:
    return -x


def primFloatLt(x: float, y: float) -> bool:
    return x < y


def primFloatLe(x: float, y: float) -> bool:
    return x <= y


def primFloatGt(x: float, y: float) -> bool:
    return x > y


def primFloatGe(x: float, y: float) -> bool:
    return x >= y


def primFloatEq(x: float, y: float) -> bool:
    return x == y


def primFloatShow(x: float) -> str:
    if x != x:  # NaN check (NaN != NaN is True)
        return "NaN"
    elif x == float("inf"):
        return "Infinity"
    elif x == float("-inf"):
        return "-Infinity"
    else:
        return str(x)


# Primitive string functions
def primStringConcat(x: str, y: str) -> str:
    return x + y


def primCharShow(x: tuple) -> str:
    if isinstance(x, tuple) and len(x) == 2 and x[0] == "char":
        return f"'{x[1]}'"
    return str(x)


def primPutStr(x: str) -> None:
    print(x, end="")


# Primitive list functions
def primListConcat(x: list, y: list) -> list:
    return x + y


def primListShow(lst: list) -> str:
    def show_element(elem):
        if isinstance(elem, str):
            return f'"{elem}"'
        elif isinstance(elem, bool):
            return str(elem)
        elif isinstance(elem, tuple) and len(elem) == 2 and elem[0] == "char":
            return f"'{elem[1]}'"
        else:
            return str(elem)

    elements = [show_element(elem) for elem in lst]
    return f"[{','.join(elements)}]"


# Built-in show function that handles type dispatch
def systemo_show(x: Any) -> str:
    if isinstance(x, bool):
        return str(x)
    elif isinstance(x, int):
        return primIntShow(x)
    elif isinstance(x, float):
        return primFloatShow(x)
    elif isinstance(x, str):
        return f'"{x}"'
    elif isinstance(x, tuple) and len(x) == 2 and x[0] == "char":
        return primCharShow(x)
    elif isinstance(x, list):
        return primListShow(x)
    else:
        return str(x)


# Built-in putStr function
def systemo_put_str(x: str) -> None:
    primPutStr(x)
