# Runtime support for System O programs.
#
# This module is imported by the interpreter and its source text is embedded
# verbatim into the Python code produced by the compilers, so it must not
# import anything from the lango package.
from typing import Any, Callable


class Char:
    """A character; kept distinct from ``str`` (the String type)."""

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Char) and other.value == self.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"Char({self.value!r})"


class Con:
    """A value ``k v_1 ... v_n`` of a user-defined datatype."""

    __slots__ = ("name", "tycon", "args")

    def __init__(self, name: str, tycon: str, args: tuple) -> None:
        self.name = name
        self.tycon = tycon
        self.args = args

    def __repr__(self) -> str:
        return " ".join([self.name] + [repr(arg) for arg in self.args])


def curry(arity: int, function: Callable[..., Any]) -> Any:
    """Turn an n-ary Python function into a curried function value."""
    if arity == 0:
        return function()

    def collect(collected: tuple) -> Any:
        if len(collected) == arity:
            return function(*collected)
        return lambda argument: collect(collected + (argument,))

    return collect(())


def constructor(name: str, tycon: str, arity: int) -> Any:
    return curry(arity, lambda *args: Con(name, tycon, args))


def pattern_match_failure(function_name: str) -> Any:
    raise RuntimeError(f"Non-exhaustive patterns in function {function_name}")


def undef(*args: Any) -> Any:
    """Dictionary for an ambiguous constraint (e.g. ``[] == []``).

    By coherence the program never applies it (Section 1 of the paper)."""
    raise RuntimeError("ambiguous overloaded identifier was applied")


def type_constructor_of(value: Any) -> str:
    """The outermost type constructor of a runtime value (dynamic semantics)."""
    if isinstance(value, bool):
        return "Bool"
    if isinstance(value, int):
        return "Int"
    if isinstance(value, float):
        return "Float"
    if isinstance(value, str):
        return "String"
    if isinstance(value, Char):
        return "Char"
    if isinstance(value, list):
        return "List"
    if isinstance(value, tuple):
        return f"Tuple{len(value)}"
    if isinstance(value, Con):
        return value.tycon
    if value is None:
        return "()"
    if callable(value):
        return "->"
    raise RuntimeError(f"Value of unknown type: {value!r}")


# --- Primitives -----------------------------------------------------------

NaN = float("nan")
Infinity = float("inf")


def primError(message: str) -> Any:
    raise RuntimeError(f"Runtime error: {message}")


def primPutStr(text: str) -> None:
    print(text, end="")


def primIntAdd(x: int) -> Callable[[int], int]:
    return lambda y: x + y


def primIntSub(x: int) -> Callable[[int], int]:
    return lambda y: x - y


def primIntMul(x: int) -> Callable[[int], int]:
    return lambda y: x * y


def primIntDiv(x: int) -> Callable[[int], float]:
    return lambda y: x / y


def primIntPow(x: int) -> Callable[[int], int]:
    return lambda y: x**y


def primIntNeg(x: int) -> int:
    return -x


def primIntLt(x: int) -> Callable[[int], bool]:
    return lambda y: x < y


def primIntLe(x: int) -> Callable[[int], bool]:
    return lambda y: x <= y


def primIntGt(x: int) -> Callable[[int], bool]:
    return lambda y: x > y


def primIntGe(x: int) -> Callable[[int], bool]:
    return lambda y: x >= y


def primIntEq(x: int) -> Callable[[int], bool]:
    return lambda y: x == y


def primIntShow(x: int) -> str:
    return str(x)


def primFloatAdd(x: float) -> Callable[[float], float]:
    return lambda y: x + y


def primFloatSub(x: float) -> Callable[[float], float]:
    return lambda y: x - y


def primFloatMul(x: float) -> Callable[[float], float]:
    return lambda y: x * y


def primFloatDiv(x: float) -> Callable[[float], float]:
    return lambda y: x / y


def primFloatPow(x: float) -> Callable[[float], float]:
    return lambda y: x**y


def primFloatNeg(x: float) -> float:
    return -x


def primFloatLt(x: float) -> Callable[[float], bool]:
    return lambda y: x < y


def primFloatLe(x: float) -> Callable[[float], bool]:
    return lambda y: x <= y


def primFloatGt(x: float) -> Callable[[float], bool]:
    return lambda y: x > y


def primFloatGe(x: float) -> Callable[[float], bool]:
    return lambda y: x >= y


def primFloatEq(x: float) -> Callable[[float], bool]:
    return lambda y: x == y


def primFloatShow(x: float) -> str:
    if x != x:
        return "NaN"
    if x == Infinity:
        return "Infinity"
    if x == -Infinity:
        return "-Infinity"
    return str(x)


def primBoolAnd(x: bool) -> Callable[[bool], bool]:
    return lambda y: x and y


def primBoolOr(x: bool) -> Callable[[bool], bool]:
    return lambda y: x or y


def primBoolEq(x: bool) -> Callable[[bool], bool]:
    return lambda y: x == y


def primStringConcat(x: str) -> Callable[[str], str]:
    return lambda y: x + y


def primStringShow(x: str) -> str:
    return f'"{x}"'


def primCharShow(x: Char) -> str:
    return f"'{x.value}'"


def primListConcat(x: list) -> Callable[[list], list]:
    return lambda y: x + y
