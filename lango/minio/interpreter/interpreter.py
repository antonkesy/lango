"""A tree-walking interpreter for MiniO.

Runtime values: Python ``int``/``float``/``bool``/``str``, characters as
``("char", c)``, lists, tuples and constructor values as dictionaries
``{"_constructor": name, "field_0": ..., ...}`` (positional) or
``{"_constructor": name, <field name>: ..., ...}`` (record syntax).
"""

from collections.abc import Callable, Mapping, Sequence
from operator import eq, ge, gt, le, lt, ne
from typing import Any

from lango.minio.typechecker.typecheck import type_check
from lango.shared.ast.nodes import (
    AddOperation,
    AndOperation,
    BinaryOperation,
    BoolLiteral,
    CharLiteral,
    ConcatOperation,
    ConsPattern,
    Constructor,
    ConstructorExpression,
    ConstructorPattern,
    DataConstructor,
    DataDeclaration,
    DivOperation,
    DoBlock,
    EqualOperation,
    Expression,
    FloatLiteral,
    FunctionApplication,
    FunctionDefinition,
    GreaterEqualOperation,
    GreaterThanOperation,
    GroupedExpression,
    IfElse,
    IndexOperation,
    IntLiteral,
    LessEqualOperation,
    LessThanOperation,
    LetStatement,
    ListLiteral,
    ListPattern,
    LiteralPattern,
    MulOperation,
    NegativeFloat,
    NegativeFloatPattern,
    NegativeInt,
    NegativeIntPattern,
    NegOperation,
    NotEqualOperation,
    NotOperation,
    OrOperation,
    Pattern,
    PowFloatOperation,
    PowIntOperation,
    Program,
    Statement,
    StringLiteral,
    SubOperation,
    TupleLiteral,
    TuplePattern,
    Variable,
    VariablePattern,
    is_expression,
)
from lango.shared.run_result import RunResult, run_program

type Value = Any
type Scope = Mapping[str, Value]
type Record = dict[str, Value]

CONSTRUCTOR_KEY = "_constructor"


def interpret(ast: Program, collect_stdout: bool = False) -> RunResult:
    type_check(ast)
    interpreter = Interpreter(ast)
    if "main" not in interpreter.functions:
        raise RuntimeError("No main function defined")

    def run() -> None:
        result = interpreter.function_value("main")
        if collect_stdout:
            return
        if callable(result):
            print("[main] is a function")
        elif result is not None:
            print(f"{result}\n")

    return run_program(run, collect_stdout)


# --- Built-in functions ------------------------------------------------------------


def _print(value: Value) -> None:
    match value:
        case ("char", char):
            print(char, end="")
        case str():
            print(value.encode().decode("unicode_escape"), end="")
        case _:
            print(value, end="")


def _put_str(arg: Value) -> Value:
    """``putStr``; applied to a function (``putStr show``) it prints that
    function's results."""
    if callable(arg):
        return lambda value: _print(arg(value))
    _print(arg)
    return None


def _show(value: Value) -> str:
    match value:
        case ("char", char):
            return f"'{char}'"
        case str():
            return f'"{value}"'
        case list():
            return "[" + ",".join(_show(item) for item in value) + "]"
        case float() if value == float("inf"):
            return "Infinity"
        case float() if value == float("-inf"):
            return "-Infinity"
        case dict() if CONSTRUCTOR_KEY in value:
            fields: list[str] = []
            while f"field_{len(fields)}" in value:
                fields.append(_show(value[f"field_{len(fields)}"]))
            name = value[CONSTRUCTOR_KEY]
            return f"{name}({', '.join(fields)})" if fields else name
        case _:
            return str(value)


def _error(message: str) -> Value:
    raise RuntimeError(f"Runtime error: {message}")


def _concat(left: Value, right: Value) -> Value:
    if isinstance(left, list) and isinstance(right, list):
        return left + right
    return str(left) + str(right)


def _div(left: Value, right: Value) -> Value:
    return float("inf") if right == 0 else left / right  # like Haskell


BUILTINS: dict[str, Value] = {
    "putStr": _put_str,
    "show": _show,
    "error": _error,
    "mod": lambda x: lambda y: x % y,
}

BINARY_OPERATORS: dict[type, Callable[[Value, Value], Value]] = {
    AddOperation: lambda a, b: a + b,
    SubOperation: lambda a, b: a - b,
    MulOperation: lambda a, b: a * b,
    DivOperation: _div,
    PowIntOperation: lambda a, b: int(a**b),
    PowFloatOperation: lambda a, b: float(a**b),
    EqualOperation: eq,
    NotEqualOperation: ne,
    LessThanOperation: lt,
    LessEqualOperation: le,
    GreaterThanOperation: gt,
    GreaterEqualOperation: ge,
    ConcatOperation: _concat,
}


def curry(arity: int, function: Callable[..., Value]) -> Value:
    def collect(collected: tuple) -> Value:
        if len(collected) >= arity:
            return function(*collected)
        return lambda argument: collect(collected + (argument,))

    return collect(())


# --- Interpreter -----------------------------------------------------------------


class Interpreter:
    def __init__(self, program: Program) -> None:
        self.functions: dict[str, list[FunctionDefinition]] = {}
        self.constructors: dict[str, DataConstructor] = {}
        for stmt in program.statements:
            match stmt:
                case FunctionDefinition(function_name=name):
                    self.functions.setdefault(name, []).append(stmt)
                case DataDeclaration(constructors=constructors):
                    for constructor in constructors:
                        self.constructors[constructor.name] = constructor

    # --- names -----------------------------------------------------------------

    def lookup(self, name: str, scope: Scope) -> Value:
        if name in scope:
            return scope[name]
        if name in self.functions:
            return self.function_value(name)
        if name in BUILTINS:
            return BUILTINS[name]
        if name in self.constructors:
            return self.constructor_value(name)
        raise RuntimeError(f"Unknown variable: {name}")

    def function_value(self, name: str) -> Value:
        clauses = self.functions[name]
        for clause in clauses:
            if not clause.patterns:
                return self.eval(clause.body, {})
        return self.apply_clauses(clauses, ())

    def apply_clauses(self, clauses: list[FunctionDefinition], args: tuple) -> Value:
        """The function value of ``clauses`` applied to ``args`` so far."""

        def apply(*more: Value) -> Value:
            arguments = args + more
            for clause in clauses:
                if len(clause.patterns) == len(arguments):
                    bindings: Record = {}
                    if all(
                        self.match(pattern, argument, bindings)
                        for pattern, argument in zip(clause.patterns, arguments)
                    ):
                        return self.eval(clause.body, bindings)
            if len(arguments) < max(len(clause.patterns) for clause in clauses):
                return self.apply_clauses(clauses, arguments)
            raise RuntimeError(
                f"No matching pattern found for function call with "
                f"{len(arguments)} arguments",
            )

        return apply

    def constructor_value(self, name: str) -> Value:
        arity = constructor_arity(self.constructors[name])
        if arity == 0:
            return {CONSTRUCTOR_KEY: name}
        return curry(
            arity,
            lambda *args: {
                CONSTRUCTOR_KEY: name,
                **{f"field_{i}": arg for i, arg in enumerate(args)},
            },
        )

    # --- expressions -------------------------------------------------------------

    def eval(self, node: Expression, scope: Scope) -> Value:
        match node:
            case (
                IntLiteral(value=value)
                | FloatLiteral(value=value)
                | StringLiteral(value=value)
                | BoolLiteral(value=value)
                | NegativeInt(value=value)
                | NegativeFloat(value=value)
            ):
                return value
            case CharLiteral(value=value):
                return ("char", value)
            case ListLiteral(elements=elements):
                return [self.eval(element, scope) for element in elements]
            case TupleLiteral(elements=elements):
                return tuple(self.eval(element, scope) for element in elements)
            case Variable(name=name) | Constructor(name=name):
                return self.lookup(name, scope)
            case AndOperation(left=left, right=right):
                return self.eval(left, scope) and self.eval(right, scope)
            case OrOperation(left=left, right=right):
                return self.eval(left, scope) or self.eval(right, scope)
            case BinaryOperation(left=left, right=right):
                operator = BINARY_OPERATORS[type(node)]
                return operator(self.eval(left, scope), self.eval(right, scope))
            case NotOperation(operand=operand):
                return not self.eval(operand, scope)
            case NegOperation(operand=operand):
                return -self.eval(operand, scope)
            case IndexOperation(list_expr=list_expr, index_expr=index_expr):
                return self.index(
                    self.eval(list_expr, scope),
                    self.eval(index_expr, scope),
                )
            case IfElse(condition=condition, then_expr=then_expr, else_expr=else_expr):
                branch = then_expr if self.eval(condition, scope) else else_expr
                return self.eval(branch, scope)
            case DoBlock(statements=statements):
                return self.eval_block(statements, scope)
            case FunctionApplication(function=function, argument=argument):
                return self.eval(function, scope)(self.eval(argument, scope))
            case ConstructorExpression(constructor_name=name, fields=fields):
                return {
                    CONSTRUCTOR_KEY: name,
                    **{f.field_name: self.eval(f.value, scope) for f in fields},
                }
            case GroupedExpression(expression=expression):
                return self.eval(expression, scope)
        raise NotImplementedError(f"Unhandled expression type: {type(node).__name__}")

    @staticmethod
    def index(values: Value, index: Value) -> Value:
        if not isinstance(values, list):
            raise RuntimeError(f"Cannot index non-list value: {type(values)}")
        if not isinstance(index, int):
            raise RuntimeError(f"List index must be an integer, got: {type(index)}")
        if not 0 <= index < len(values):
            raise RuntimeError(
                f"List index {index} out of bounds for list of length {len(values)}",
            )
        return values[index]

    def eval_block(self, statements: Sequence[Statement], scope: Scope) -> Value:
        result: Value = None
        for stmt in statements:
            match stmt:
                case LetStatement(variable=name, value=value):
                    scope = {**scope, name: self.eval(value, scope)}
                case _ if is_expression(stmt):
                    result = self.eval(stmt, scope)
        return result

    # --- patterns ----------------------------------------------------------------

    def match(self, pattern: Pattern, value: Value, bindings: Record) -> bool:
        """Does ``value`` match ``pattern``?  The variables it binds are added
        to ``bindings``."""
        match pattern:
            case VariablePattern(name=name):
                bindings[name] = value
                return True
            case (
                LiteralPattern(value=literal)
                | NegativeIntPattern(value=literal)
                | NegativeFloatPattern(value=literal)
            ):
                return bool(literal == value)
            case ConstructorPattern(constructor=name, patterns=subpatterns):
                if not (isinstance(value, dict) and value.get(CONSTRUCTOR_KEY) == name):
                    return False
                return self.match_all(
                    subpatterns,
                    self.constructor_fields(value),
                    bindings,
                )
            case ConsPattern(head=head, tail=tail):
                return (
                    isinstance(value, list)
                    and len(value) > 0
                    and self.match(head, value[0], bindings)
                    and self.match(tail, value[1:], bindings)
                )
            case ListPattern(patterns=subpatterns):
                return isinstance(value, list) and self.match_all(
                    subpatterns,
                    value,
                    bindings,
                )
            case TuplePattern(patterns=subpatterns):
                return isinstance(value, tuple) and self.match_all(
                    subpatterns,
                    value,
                    bindings,
                )
        raise NotImplementedError(f"Unhandled pattern type: {type(pattern).__name__}")

    def match_all(
        self,
        patterns: Sequence[Pattern],
        values: Sequence[Value],
        bindings: Record,
    ) -> bool:
        return len(patterns) == len(values) and all(
            self.match(pattern, value, bindings)
            for pattern, value in zip(patterns, values)
        )

    def constructor_fields(self, value: Record) -> list[Value]:
        """The field values of a constructor value in declaration order."""
        constructor = self.constructors.get(value[CONSTRUCTOR_KEY])
        if constructor is not None and constructor.record_constructor is not None:
            names = [f.name for f in constructor.record_constructor.fields]
            if all(name in value for name in names):
                return [value[name] for name in names]
        fields: list[Value] = []
        while f"field_{len(fields)}" in value:
            fields.append(value[f"field_{len(fields)}"])
        return fields


def constructor_arity(constructor: DataConstructor) -> int:
    if constructor.record_constructor is not None:
        return len(constructor.record_constructor.fields)
    return len(constructor.type_atoms or [])
