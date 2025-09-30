"""Desugaring of the surface syntax into the core terms of System O.

After this pass the program only contains the term forms of the paper
(variables, applications, ``let`` via ``do`` blocks, ``if``, literals and
constructors): infix operator chains are re-associated according to the
``infixl``/``infixr``/``infix`` declarations and turned into ordinary
applications of the (overloaded) operator functions.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from lango.shared.ast.nodes import (
    Associativity,
    ConstructorExpression,
    DataDeclaration,
    DoBlock,
    Expression,
    FieldAssignment,
    FunctionApplication,
    FunctionDefinition,
    GroupedExpression,
    IfElse,
    InstanceDeclaration,
    LetStatement,
    ListLiteral,
    PrecedenceDeclaration,
    Program,
    Statement,
    SymbolicOperation,
    TupleLiteral,
    Variable,
)

# Haskell's default fixity for operators without a declaration.
DEFAULT_FIXITY = (9, Associativity.LEFT)


class DesugarError(Exception):
    pass


@dataclass
class Fixities:
    table: Dict[str, Tuple[int, Associativity]]

    def of(self, operator: str) -> Tuple[int, Associativity]:
        return self.table.get(operator, DEFAULT_FIXITY)


def desugar_program(program: Program) -> Program:
    fixities = Fixities(
        {
            stmt.operator: (stmt.precedence, stmt.associativity)
            for stmt in program.statements
            if isinstance(stmt, PrecedenceDeclaration)
        },
    )
    statements: List[Statement] = []
    for stmt in program.statements:
        match stmt:
            case PrecedenceDeclaration():
                continue
            case FunctionDefinition():
                statements.append(_desugar_function(stmt, fixities))
            case InstanceDeclaration(
                instance_name=name,
                type_signature=sig,
                clauses=cs,
            ):
                statements.append(
                    InstanceDeclaration(
                        name,
                        sig,
                        [_desugar_function(c, fixities) for c in cs],
                    ),
                )
            case DataDeclaration():
                statements.append(stmt)
            case _:
                raise DesugarError(f"Unexpected top-level statement: {stmt}")
    return Program(statements)


def _desugar_function(
    func: FunctionDefinition,
    fixities: Fixities,
) -> FunctionDefinition:
    return FunctionDefinition(
        func.function_name,
        func.patterns,
        _desugar_expr(func.body, fixities),
    )


def _desugar_expr(expr: Expression, fixities: Fixities) -> Expression:
    match expr:
        case SymbolicOperation():
            operands, operators = _flatten_chain(expr)
            operands = [_desugar_expr(e, fixities) for e in operands]
            return _reassociate(operands, operators, fixities)
        case IfElse(condition=c, then_expr=t, else_expr=e):
            return IfElse(
                _desugar_expr(c, fixities),
                _desugar_expr(t, fixities),
                _desugar_expr(e, fixities),
            )
        case FunctionApplication(function=f, argument=a):
            return FunctionApplication(
                _desugar_expr(f, fixities),
                _desugar_expr(a, fixities),
            )
        case DoBlock(statements=stmts):
            return DoBlock([_desugar_statement(s, fixities) for s in stmts])
        case GroupedExpression(expression=inner):
            return GroupedExpression(_desugar_expr(inner, fixities))
        case ListLiteral(elements=elements):
            return ListLiteral([_desugar_expr(e, fixities) for e in elements])
        case TupleLiteral(elements=elements):
            return TupleLiteral([_desugar_expr(e, fixities) for e in elements])
        case ConstructorExpression(constructor_name=name, fields=fields):
            return ConstructorExpression(
                name,
                [
                    FieldAssignment(f.field_name, _desugar_expr(f.value, fixities))
                    for f in fields
                ],
            )
        case _:
            return expr


def _desugar_statement(stmt: Statement, fixities: Fixities) -> Statement:
    match stmt:
        case LetStatement(variable=name, value=value):
            return LetStatement(name, _desugar_expr(value, fixities))
        case _:
            return _desugar_expr(stmt, fixities)  # type: ignore[arg-type]


def _flatten_chain(expr: Expression) -> Tuple[List[Expression], List[str]]:
    """The parser produces right-nested chains ``a op1 (b op2 (c op3 d))``."""
    operands: List[Expression] = []
    operators: List[str] = []
    current = expr
    while isinstance(current, SymbolicOperation):
        left, right = current.operands
        operands.append(left)
        operators.append(current.operator)
        current = right
    operands.append(current)
    return operands, operators


def _reassociate(
    operands: List[Expression],
    operators: List[str],
    fixities: Fixities,
) -> Expression:
    """Precedence climbing over a flat operator chain."""

    def binary(op: str, left: Expression, right: Expression) -> Expression:
        return FunctionApplication(FunctionApplication(Variable(op), left), right)

    def parse(index: int, min_prec: int) -> Tuple[Expression, int]:
        left = operands[index]
        while index < len(operators):
            op = operators[index]
            prec, assoc = fixities.of(op)
            if prec < min_prec:
                break
            next_min = prec + 1 if assoc != Associativity.RIGHT else prec
            right, next_index = parse(index + 1, next_min)
            if assoc == Associativity.NONE and next_index < len(operators):
                next_prec, _ = fixities.of(operators[next_index])
                if next_prec == prec:
                    raise DesugarError(
                        f"Non-associative operator '{op}' used in a chain",
                    )
            left = binary(op, left, right)
            index = next_index
        return left, index

    result, consumed = parse(0, 0)
    if consumed != len(operators):
        raise DesugarError("Could not resolve operator precedence")
    return result
