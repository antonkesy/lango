"""Compilation of MiniO to Python.

Every MiniO function ``f`` becomes a Python function ``minio_f`` taking all
its parameters at once; applications with fewer arguments become lambdas.
Constructors become classes (positional fields ``arg_i``, record fields in
``fields``), characters are ``("char", c)`` tuples and the runtime support
(``minio_show``, ``minio_putStr``, ``minio_error``) is the prelude in
``prelude.py``, which is embedded verbatim.
"""

from collections.abc import Sequence
from pathlib import Path

from lango.minio.common import (
    constructor_arity,
    constructors_of,
    field_names,
    functions_of,
    unapply,
)
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

PRELUDE = Path(__file__).with_name("prelude.py")
BUILTIN_ARITY = {"show": 1, "putStr": 1, "error": 1}

BINARY_OPERATORS = {
    AddOperation: "+",
    SubOperation: "-",
    MulOperation: "*",
    EqualOperation: "==",
    NotEqualOperation: "!=",
    LessThanOperation: "<",
    LessEqualOperation: "<=",
    GreaterThanOperation: ">",
    GreaterEqualOperation: ">=",
    ConcatOperation: "+",
    AndOperation: "and",
    OrOperation: "or",
}

# variables bound by the patterns of the clause being compiled
type Scope = frozenset[str]


def mangle(name: str) -> str:
    return f"minio_{name}"


def indent(lines: Sequence[str], depth: int = 1) -> list[str]:
    return ["    " * depth + line for line in lines]


def literal(value: object) -> str:
    match value:
        case bool():
            return str(value)
        case str():
            return f'"{value}"'
        case int() | float() if value < 0:
            return f"({value})"
        case _:
            return str(value)


class MinioCompiler:
    def __init__(self, program: Program) -> None:
        self.constructors = constructors_of(program)
        self.functions = functions_of(program)
        self.arity = BUILTIN_ARITY | {
            name: max(len(clause.patterns) for clause in clauses)
            for name, clauses in self.functions.items()
        }

    # --- program -------------------------------------------------------------------

    def compile(self) -> str:
        lines = PRELUDE.read_text().splitlines()
        for name, constructor in self.constructors.items():
            lines.extend(self.compile_constructor(constructor))
        for name, clauses in self.functions.items():
            lines.extend(self.compile_function(name, clauses))
            lines.append("")
        if "main" in self.functions:
            lines.extend(["", "if __name__ == '__main__':", f"    {mangle('main')}()"])
        return "\n".join(lines)

    def compile_constructor(self, constructor: DataConstructor) -> list[str]:
        """A class whose instances carry the fields of the constructor."""
        arity = constructor_arity(constructor)
        params = [f"arg_{i}" for i in range(arity)]
        names = field_names(constructor)
        if names is not None:
            stores = ["self.fields = {"]
            stores += [f"    '{name}': {param}," for name, param in zip(names, params)]
            stores.append("}")
        elif params:
            stores = [f"self.{param} = {param}" for param in params]
        else:
            stores = ["pass"]
        return [
            f"class {constructor.name}:",
            f"    def __init__(self{''.join(', ' + p for p in params)}) -> None:",
            *indent(stores, 2),
            "",
        ]

    # --- functions -----------------------------------------------------------------

    def compile_function(
        self,
        name: str,
        clauses: Sequence[FunctionDefinition],
    ) -> list[str]:
        """``def minio_f(arg_0, ..., arg_n)``: the clauses are tried in order."""
        params = [f"arg_{i}" for i in range(self.arity[name])]
        lines = [f"def {mangle(name)}({', '.join(params)}):"]
        exhaustive = False
        for clause in clauses:
            conditions: list[str] = []
            bindings: list[tuple[str, str]] = []
            for pattern, param in zip(clause.patterns, params):
                self.compile_pattern(pattern, param, conditions, bindings)
            scope = frozenset(variable for variable, _ in bindings)
            body = [f"{variable} = {value}" for variable, value in bindings]
            body += self.compile_body(clause.body, scope)
            if conditions:
                lines.append(f"    if {' and '.join(conditions)}:")
                lines.extend(indent(body, 2))
            else:
                lines.extend(indent(body))
                exhaustive = True
                break  # later clauses are unreachable
        if not exhaustive:
            lines.append(
                f"    raise ValueError('No matching pattern for {mangle(name)}')",
            )
        return lines

    def compile_body(self, body: Expression, scope: Scope) -> list[str]:
        """The statements returning the value of a clause body."""
        if not isinstance(body, DoBlock):
            return [f"return {self.compile_expression(body, scope)}"]
        lines: list[str] = []
        result = "None"
        for stmt in body.statements:
            match stmt:
                case LetStatement(variable=variable, value=value):
                    lines.append(
                        f"{mangle(variable)} = {self.compile_expression(value, scope)}",
                    )
                    result = "None"
                case _:
                    assert is_expression(stmt)
                    result = self.compile_expression(stmt, scope)
                    lines.append(result)
        if lines and result != "None":
            lines[-1] = f"return {result}"
        else:
            lines.append("return None")
        return lines

    def compile_pattern(
        self,
        pattern: Pattern,
        subject: str,
        conditions: list[str],
        bindings: list[tuple[str, str]],
    ) -> None:
        """Conditions under which ``subject`` matches, and the variables bound."""
        match pattern:
            case VariablePattern(name=name):
                bindings.append((name, subject))
            case (
                LiteralPattern(value=value)
                | NegativeIntPattern(value=value)
                | NegativeFloatPattern(value=value)
            ):
                conditions.append(f"{subject} == {literal(value)}")
            case ConstructorPattern(constructor=name, patterns=subpatterns):
                conditions.append(f"isinstance({subject}, {name})")
                names = field_names(self.constructors[name])
                for index, sub in enumerate(subpatterns):
                    field = (
                        f"{subject}.fields['{names[index]}']"
                        if names is not None
                        else f"{subject}.arg_{index}"
                    )
                    self.compile_pattern(sub, field, conditions, bindings)
            case ConsPattern(head=head, tail=tail):
                conditions.append(f"len({subject}) > 0")
                self.compile_pattern(head, f"{subject}[0]", conditions, bindings)
                self.compile_pattern(tail, f"{subject}[1:]", conditions, bindings)
            case ListPattern(patterns=subpatterns) | TuplePattern(patterns=subpatterns):
                conditions.append(f"len({subject}) == {len(subpatterns)}")
                for index, sub in enumerate(subpatterns):
                    self.compile_pattern(
                        sub,
                        f"{subject}[{index}]",
                        conditions,
                        bindings,
                    )
            case _:
                raise ValueError(f"Unhandled pattern {type(pattern).__name__}")

    # --- expressions ---------------------------------------------------------------

    def compile_expression(self, expr: Expression, scope: Scope) -> str:
        match expr:
            case (
                IntLiteral(value=value)
                | FloatLiteral(value=value)
                | BoolLiteral(value=value)
            ):
                return literal(value)
            case NegativeInt(value=value) | NegativeFloat(value=value):
                return literal(value)
            case StringLiteral(value=value):
                return f'"{value}"'
            case CharLiteral(value=value):
                return f"('char', '{value}')"  # a tuple, to tell chars from strings
            case ListLiteral(elements=elements):
                return f"[{', '.join(self.compile_expression(e, scope) for e in elements)}]"
            case TupleLiteral(elements=elements):
                items = [self.compile_expression(e, scope) for e in elements]
                return f"({''.join(item + ', ' for item in items)})"
            case Variable(name=name):
                return self.compile_variable(name, scope)
            case Constructor(name=name):
                return (
                    f"{name}()"
                    if constructor_arity(self.constructors[name]) == 0
                    else name
                )
            case DivOperation(left=left, right=right):
                l, r = self.compile_expression(left, scope), self.compile_expression(
                    right,
                    scope,
                )
                return f"({l} / {r}) if {r} != 0 else math.inf"  # like Haskell
            case PowIntOperation(left=left, right=right):
                l, r = self.compile_expression(left, scope), self.compile_expression(
                    right,
                    scope,
                )
                return f"int(({l} ** {r}))"
            case PowFloatOperation(left=left, right=right):
                l, r = self.compile_expression(left, scope), self.compile_expression(
                    right,
                    scope,
                )
                return f"float(({l} ** {r}))"
            case BinaryOperation(left=left, right=right):
                l, r = self.compile_expression(left, scope), self.compile_expression(
                    right,
                    scope,
                )
                return f"({l} {BINARY_OPERATORS[type(expr)]} {r})"
            case NotOperation(operand=operand):
                return f"(not {self.compile_expression(operand, scope)})"
            case NegOperation(operand=operand):
                return f"(-{self.compile_expression(operand, scope)})"
            case IndexOperation(list_expr=list_expr, index_expr=index_expr):
                l, i = self.compile_expression(
                    list_expr,
                    scope,
                ), self.compile_expression(index_expr, scope)
                return f"({l}[{i}])"
            case IfElse(condition=condition, then_expr=then_expr, else_expr=else_expr):
                c = self.compile_expression(condition, scope)
                t = self.compile_expression(then_expr, scope)
                e = self.compile_expression(else_expr, scope)
                return f"({t} if {c} else {e})"
            case FunctionApplication():
                return self.compile_application(expr, scope)
            case ConstructorExpression(constructor_name=name, fields=fields):
                given = {
                    f.field_name: self.compile_expression(f.value, scope)
                    for f in fields
                }
                names = field_names(self.constructors[name]) or list(given)
                return (
                    f"{name}({', '.join(given.get(field, 'None') for field in names)})"
                )
            case DoBlock(statements=statements):
                return self.compile_block(statements, scope)
            case GroupedExpression(expression=inner):
                return f"({self.compile_expression(inner, scope)})"
        raise ValueError(f"Unhandled expression {type(expr).__name__}")

    def compile_variable(self, name: str, scope: Scope) -> str:
        if name in scope:
            return name
        if self.arity.get(name) == 0:
            return f"{mangle(name)}()"  # a nullary function is its value
        return mangle(name)

    def compile_application(self, expr: FunctionApplication, scope: Scope) -> str:
        """Saturated calls pass all arguments at once; partial applications
        become lambdas and surplus arguments are applied to the result."""
        head, arguments = unapply(expr)
        args = [self.compile_expression(a, scope) for a in arguments]
        match head:
            case Constructor(name=name):
                return f"{name}({', '.join(args)})"
            case Variable(name=name) if name not in scope and name in self.arity:
                function = mangle(name)
                arity = self.arity[name]
                if len(args) < arity:
                    missing = [f"__arg_{i}" for i in range(arity - len(args))]
                    return f"lambda {', '.join(missing)}: {function}({', '.join(args + missing)})"
                call = f"{function}({', '.join(args[:arity])})"
                return call + "".join(f"({a})" for a in args[arity:])
            case _:
                return self.compile_expression(head, scope) + "".join(
                    f"({a})" for a in args
                )

    def compile_block(self, statements: Sequence[Statement], scope: Scope) -> str:
        """A ``do`` block in expression position: the statements are evaluated
        left to right by an ``or`` chain (they evaluate to ``None``); ``let``
        binds a global."""
        parts: list[str] = []
        for stmt in statements:
            match stmt:
                case LetStatement(variable=variable, value=value):
                    compiled = self.compile_expression(value, scope)
                    if len(statements) == 1:
                        return f"(lambda: {compiled})()"
                    parts.append(
                        f"globals().update({{'{mangle(variable)}': {compiled}}})",
                    )
                case _:
                    assert is_expression(stmt)
                    parts.append(self.compile_expression(stmt, scope))
        if len(parts) == 1:
            return parts[0]
        return f"({' or '.join(parts)})"


def compile_program(program: Program) -> str:
    return MinioCompiler(program).compile()
