"""Helpers shared by the MiniO interpreter and compilers."""

from lango.shared.ast.nodes import (
    BinaryOperation,
    ConsPattern,
    ConstructorExpression,
    ConstructorPattern,
    DataConstructor,
    DataDeclaration,
    DoBlock,
    Expression,
    FunctionApplication,
    FunctionDefinition,
    GroupedExpression,
    IfElse,
    IndexOperation,
    LetStatement,
    ListLiteral,
    ListPattern,
    Pattern,
    Program,
    Statement,
    TupleLiteral,
    TuplePattern,
    UnaryOperation,
    Variable,
    VariablePattern,
)


def constructor_arity(constructor: DataConstructor) -> int:
    if constructor.record_constructor is not None:
        return len(constructor.record_constructor.fields)
    return len(constructor.type_atoms or [])


def field_names(constructor: DataConstructor) -> list[str] | None:
    """The field names of a record constructor, ``None`` for a positional one."""
    if constructor.record_constructor is None:
        return None
    return [field.name for field in constructor.record_constructor.fields]


def constructors_of(program: Program) -> dict[str, DataConstructor]:
    return {
        constructor.name: constructor
        for stmt in program.statements
        if isinstance(stmt, DataDeclaration)
        for constructor in stmt.constructors
    }


def functions_of(program: Program) -> dict[str, list[FunctionDefinition]]:
    """The clauses of every function, grouped by name in order of first definition."""
    groups: dict[str, list[FunctionDefinition]] = {}
    for stmt in program.statements:
        if isinstance(stmt, FunctionDefinition):
            groups.setdefault(stmt.function_name, []).append(stmt)
    return groups


def pattern_variables(pattern: Pattern) -> set[str]:
    match pattern:
        case VariablePattern(name=name):
            return {name}
        case (
            ConstructorPattern(patterns=subpatterns)
            | TuplePattern(
                patterns=subpatterns,
            )
            | ListPattern(patterns=subpatterns)
        ):
            return set().union(*(pattern_variables(sub) for sub in subpatterns))
        case ConsPattern(head=head, tail=tail):
            return pattern_variables(head) | pattern_variables(tail)
        case _:
            return set()


def unapply(expr: Expression) -> tuple[Expression, list[Expression]]:
    """``f a_1 ... a_n`` as ``(f, [a_1, ..., a_n])`` (grouping around ``f`` is dropped)."""
    arguments: list[Expression] = []
    while True:
        match expr:
            case FunctionApplication(function=function, argument=argument):
                arguments.insert(0, argument)
                expr = function
            case GroupedExpression(expression=inner):
                expr = inner
            case _:
                return expr, arguments


def referenced_variables(expr: Expression | Statement) -> set[str]:
    """The variables an expression refers to (``let``-bound names included)."""
    match expr:
        case Variable(name=name):
            return {name}
        case BinaryOperation(left=left, right=right):
            return referenced_variables(left) | referenced_variables(right)
        case UnaryOperation(operand=operand) | GroupedExpression(expression=operand):
            return referenced_variables(operand)
        case LetStatement(value=value):
            return referenced_variables(value)
        case IndexOperation(list_expr=first, index_expr=second) | FunctionApplication(
            function=first,
            argument=second,
        ):
            return referenced_variables(first) | referenced_variables(second)
        case IfElse(condition=condition, then_expr=then_expr, else_expr=else_expr):
            return (
                referenced_variables(condition)
                | referenced_variables(then_expr)
                | referenced_variables(else_expr)
            )
        case ListLiteral(elements=elements) | TupleLiteral(elements=elements):
            return set().union(*(referenced_variables(e) for e in elements))
        case DoBlock(statements=statements):
            return set().union(*(referenced_variables(stmt) for stmt in statements))
        case ConstructorExpression(fields=fields):
            return set().union(*(referenced_variables(f.value) for f in fields))
        case _:
            return set()
