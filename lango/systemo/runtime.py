"""Runtime support for System O programs.

This module is imported by the interpreter and its source text is embedded
verbatim into the Python code produced by the compilers, so it must not
import anything from the lango package.
"""

from dataclasses import dataclass
from operator import add, eq, ge, gt, le, lt, mul, neg, pow, sub, truediv
from typing import Any, Callable


@dataclass(frozen=True, slots=True, repr=False)
class Char:
    """A character; kept distinct from ``str`` (the String type)."""

    value: str

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
        return " ".join([self.name, *map(repr, self.args)])


def curry(arity: int, function: Callable[..., Any]) -> Any:
    """Turn an n-ary Python function (n >= 1) into a curried function value."""

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
    match value:
        case bool():  # before int: bool is a subclass of int
            return "Bool"
        case int():
            return "Int"
        case float():
            return "Float"
        case str():
            return "String"
        case Char():
            return "Char"
        case list():
            return "List"
        case tuple():
            return f"Tuple{len(value)}"
        case Con():
            return value.tycon
        case None:
            return "()"
        case _ if callable(value):
            return "->"
    raise RuntimeError(f"Value of unknown type: {value!r}")


# --- Primitives -----------------------------------------------------------

NaN = float("nan")
Infinity = float("inf")


def binary(
    operation: Callable[[Any, Any], Any],
) -> Callable[[Any], Callable[[Any], Any]]:
    """A curried binary primitive."""
    return lambda x: lambda y: operation(x, y)


def primError(message: str) -> Any:
    raise RuntimeError(f"Runtime error: {message}")


def primPutStr(text: str) -> None:
    print(text, end="")


primIntAdd = binary(add)
primIntSub = binary(sub)
primIntMul = binary(mul)
primIntDiv = binary(truediv)
primIntPow = binary(pow)
primIntNeg = neg
primIntLt = binary(lt)
primIntLe = binary(le)
primIntGt = binary(gt)
primIntGe = binary(ge)
primIntEq = binary(eq)
primIntShow = str

primFloatAdd = binary(add)
primFloatSub = binary(sub)
primFloatMul = binary(mul)
primFloatDiv = binary(truediv)
primFloatPow = binary(pow)
primFloatNeg = neg
primFloatLt = binary(lt)
primFloatLe = binary(le)
primFloatGt = binary(gt)
primFloatGe = binary(ge)
primFloatEq = binary(eq)


def primFloatShow(x: float) -> str:
    if x != x:
        return "NaN"
    if x == Infinity:
        return "Infinity"
    if x == -Infinity:
        return "-Infinity"
    return str(x)


primBoolAnd = binary(lambda x, y: x and y)
primBoolOr = binary(lambda x, y: x or y)
primBoolEq = binary(eq)

primStringConcat = binary(add)


def primStringShow(x: str) -> str:
    return f'"{x}"'


def primCharShow(x: Char) -> str:
    return f"'{x.value}'"


primListConcat = binary(add)
