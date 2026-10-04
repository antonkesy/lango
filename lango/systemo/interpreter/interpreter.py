"""An interpreter following the untyped dynamic semantics of System O
(Section 3 of the paper).

Overloaded identifiers denote functions that dispatch on the outermost
type constructor of their first argument (``extend(T, [[e]], rho(o))`` in
Figure 3).  No type information is needed at run time; the program is type
checked first only to reject ill-typed programs.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from lango.shared.ast.nodes import (
    BoolLiteral,
    CharLiteral,
    ConsPattern,
    Constructor,
    ConstructorExpression,
    ConstructorPattern,
    DoBlock,
    Expression,
    FloatLiteral,
    FunctionApplication,
    FunctionDefinition,
    GroupedExpression,
    IfElse,
    IntLiteral,
    LetStatement,
    ListLiteral,
    ListPattern,
    LiteralPattern,
    NegativeFloat,
    NegativeFloatPattern,
    NegativeInt,
    NegativeIntPattern,
    Pattern,
    Program,
    Statement,
    StringLiteral,
    TupleLiteral,
    TuplePattern,
    Variable,
    VariablePattern,
    is_expression,
)
from lango.shared.run_result import RunResult, run_program
from lango.systemo import runtime
from lango.systemo.typechecker.infer import (
    FunctionDecl,
    InstanceInfo,
    TypedProgram,
    infer_program,
)
from lango.systemo.typechecker.primitives import CONSTANTS, PRIMITIVES

Value = Any


@dataclass(frozen=True)
class Scope:
    """A persistent environment: closures capture the scope they were defined in."""

    variables: dict[str, Value]
    parent: "Scope | None" = None

    def lookup(self, name: str) -> Value:
        scope: Scope | None = self
        while scope is not None:
            if name in scope.variables:
                return scope.variables[name]
            scope = scope.parent
        raise RuntimeError(f"Unknown variable: {name}")

    def child(self, variables: dict[str, Value]) -> "Scope":
        return Scope(variables, self)


class Overloaded:
    """The meaning of an overloaded identifier: dispatch on the outermost type
    constructor of the first argument (``extend(T, f, g)`` of Figure 3).

    All instances of a program are collected in one table, i.e. the program
    is read as ``inst o_1 ... in inst o_n ... in e`` (the form of programs in
    the paper); the type checker guarantees that every dispatch that happens
    was resolved against an instance in scope."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.instances: dict[str, Callable[[Value], Value]] = {}

    def __call__(self, value: Value) -> Value:
        tycon = runtime.type_constructor_of(value)
        instance = self.instances.get(tycon)
        if instance is None:
            raise RuntimeError(
                f"No instance of '{self.name}' for a value of type {tycon}",
            )
        return instance(value)


class Interpreter:
    def __init__(self, typed: TypedProgram) -> None:
        self.typed = typed

    # --- programs -------------------------------------------------------------

    def run(self) -> Value:
        scope = self.initial_scope()
        overloaded = {name: Overloaded(name) for name in self.typed.overloaded}
        scope.variables.update(overloaded)
        for decl in self.typed.decls:
            match decl:
                case FunctionDecl(name=name, clauses=clauses, arity=arity):
                    # let u = e in ... (recursive: the scope contains u)
                    scope = scope.child({})
                    scope.variables[name] = self.make_function(
                        name,
                        clauses,
                        arity,
                        scope,
                    )
                case InstanceInfo(name=name, tycon=tycon, clauses=clauses, arity=arity):
                    overloaded[name].instances[tycon] = self.make_function(
                        name,
                        clauses,
                        arity,
                        scope,
                    )
        return scope.lookup("main")

    def initial_scope(self) -> Scope:
        variables: dict[str, Value] = {
            name: getattr(runtime, name) for name in (*PRIMITIVES, *CONSTANTS)
        }
        for con in self.typed.constructors.values():
            variables[con.name] = runtime.constructor(
                con.name,
                con.tycon,
                len(con.field_types),
            )
        return Scope(variables)

    def make_function(
        self,
        name: str,
        clauses: Sequence[FunctionDefinition],
        arity: int,
        scope: Scope,
    ) -> Value:
        if arity == 0:
            return self.eval(clauses[0].body, scope)

        def apply(*args: Value) -> Value:
            for clause in clauses:
                bindings: dict[str, Value] = {}
                if all(
                    self.match(pattern, arg, bindings)
                    for pattern, arg in zip(clause.patterns, args)
                ):
                    return self.eval(clause.body, scope.child(bindings))
            return runtime.pattern_match_failure(name)

        return runtime.curry(arity, apply)

    # --- expressions ---------------------------------------------------------

    def eval(self, expr: Expression, scope: Scope) -> Value:
        match expr:
            case IntLiteral(value=value) | NegativeInt(value=value):
                return value
            case FloatLiteral(value=value) | NegativeFloat(value=value):
                return value
            case StringLiteral(value=value):
                return value
            case CharLiteral(value=value):
                return runtime.Char(value)
            case BoolLiteral(value=value):
                return value
            case ListLiteral(elements=elements):
                return [self.eval(e, scope) for e in elements]
            case TupleLiteral(elements=elements):
                return tuple(self.eval(e, scope) for e in elements)
            case Variable(name=name) | Constructor(name=name):
                return scope.lookup(name)
            case FunctionApplication(function=function, argument=argument):
                return self.eval(function, scope)(self.eval(argument, scope))
            case IfElse(condition=condition, then_expr=then_expr, else_expr=else_expr):
                if self.eval(condition, scope):
                    return self.eval(then_expr, scope)
                return self.eval(else_expr, scope)
            case GroupedExpression(expression=inner):
                return self.eval(inner, scope)
            case DoBlock(statements=statements):
                return self.eval_block(statements, scope)
            case ConstructorExpression(constructor_name=name, fields=fields):
                info = self.typed.constructors[name]
                values = {f.field_name: self.eval(f.value, scope) for f in fields}
                return runtime.Con(
                    name,
                    info.tycon,
                    tuple(values[field] for field in info.field_names),
                )
            case _:
                raise RuntimeError(f"Cannot evaluate {type(expr).__name__}")

    def eval_block(self, statements: Sequence[Statement], scope: Scope) -> Value:
        result: Value = None
        for stmt in statements:
            match stmt:
                case LetStatement(variable=name, value=value):
                    # let u = e in ... : e is evaluated in the enclosing scope
                    scope = scope.child({name: self.eval(value, scope)})
                    result = None
                case _:
                    assert is_expression(stmt)
                    result = self.eval(stmt, scope)
        return result

    # --- patterns ------------------------------------------------------------

    def match(self, pattern: Pattern, value: Value, bindings: dict[str, Value]) -> bool:
        match pattern:
            case VariablePattern(name=name):
                bindings[name] = value
                return True
            case LiteralPattern(value=literal):
                return self.literal_value(literal) == value
            case NegativeIntPattern(value=literal) | NegativeFloatPattern(
                value=literal,
            ):
                return literal == value
            case ConstructorPattern(constructor=name, patterns=subpatterns):
                return (
                    isinstance(value, runtime.Con)
                    and value.name == name
                    and all(
                        self.match(sub, arg, bindings)
                        for sub, arg in zip(subpatterns, value.args)
                    )
                )
            case ConsPattern(head=head, tail=tail):
                return (
                    isinstance(value, list)
                    and len(value) > 0
                    and self.match(head, value[0], bindings)
                    and self.match(tail, value[1:], bindings)
                )
            case ListPattern(patterns=subpatterns):
                return (
                    isinstance(value, list)
                    and len(value) == len(subpatterns)
                    and all(
                        self.match(sub, v, bindings)
                        for sub, v in zip(subpatterns, value)
                    )
                )
            case TuplePattern(patterns=subpatterns):
                return (
                    isinstance(value, tuple)
                    and len(value) == len(subpatterns)
                    and all(
                        self.match(sub, v, bindings)
                        for sub, v in zip(subpatterns, value)
                    )
                )
            case _:
                raise RuntimeError(f"Cannot match {type(pattern).__name__}")

    def literal_value(self, literal: Expression) -> Value:
        match literal:
            case CharLiteral(value=value):
                return runtime.Char(value)
            case (
                IntLiteral(value=value)
                | FloatLiteral(value=value)
                | StringLiteral(value=value)
                | BoolLiteral(value=value)
            ):
                return value
        raise RuntimeError(f"Not a literal: {type(literal).__name__}")


def interpret(ast: Program, collect_stdout: bool = False) -> RunResult:
    interpreter = Interpreter(infer_program(ast))
    return run_program(interpreter.run, collect_stdout)
