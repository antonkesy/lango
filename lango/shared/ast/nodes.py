"""The abstract syntax shared by MiniO and System O.

MiniO has one node class per operator; System O parses every operator as a
``SymbolicOperation`` (operators are ordinary overloaded functions there).
The type checkers annotate expression nodes in place through ``ty``.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeGuard

from lango.shared.typechecker.lango_types import Type


@dataclass(slots=True)
class ASTNode:
    # filled in by the type checker
    ty: Type | None = field(default=None, kw_only=True, compare=False)


# --- Literals and names --------------------------------------------------------


@dataclass(slots=True)
class IntLiteral(ASTNode):
    value: int


@dataclass(slots=True)
class FloatLiteral(ASTNode):
    value: float


@dataclass(slots=True)
class StringLiteral(ASTNode):
    value: str


@dataclass(slots=True)
class CharLiteral(ASTNode):
    value: str


@dataclass(slots=True)
class BoolLiteral(ASTNode):
    value: bool


@dataclass(slots=True)
class NegativeInt(ASTNode):
    value: int


@dataclass(slots=True)
class NegativeFloat(ASTNode):
    value: float


@dataclass(slots=True)
class ListLiteral(ASTNode):
    elements: list["Expression"]


@dataclass(slots=True)
class TupleLiteral(ASTNode):
    elements: list["Expression"]


@dataclass(slots=True)
class Variable(ASTNode):
    name: str


@dataclass(slots=True)
class Constructor(ASTNode):
    name: str


# --- MiniO operators -----------------------------------------------------------


@dataclass(slots=True)
class BinaryOperation(ASTNode):
    left: "Expression"
    right: "Expression"


@dataclass(slots=True)
class UnaryOperation(ASTNode):
    operand: "Expression"


@dataclass(slots=True)
class AddOperation(BinaryOperation):
    pass


@dataclass(slots=True)
class SubOperation(BinaryOperation):
    pass


@dataclass(slots=True)
class MulOperation(BinaryOperation):
    pass


@dataclass(slots=True)
class DivOperation(BinaryOperation):
    pass


@dataclass(slots=True)
class PowIntOperation(BinaryOperation):
    pass


@dataclass(slots=True)
class PowFloatOperation(BinaryOperation):
    pass


@dataclass(slots=True)
class EqualOperation(BinaryOperation):
    pass


@dataclass(slots=True)
class NotEqualOperation(BinaryOperation):
    pass


@dataclass(slots=True)
class LessThanOperation(BinaryOperation):
    pass


@dataclass(slots=True)
class LessEqualOperation(BinaryOperation):
    pass


@dataclass(slots=True)
class GreaterThanOperation(BinaryOperation):
    pass


@dataclass(slots=True)
class GreaterEqualOperation(BinaryOperation):
    pass


@dataclass(slots=True)
class AndOperation(BinaryOperation):
    pass


@dataclass(slots=True)
class OrOperation(BinaryOperation):
    pass


@dataclass(slots=True)
class ConcatOperation(BinaryOperation):
    pass


@dataclass(slots=True)
class NotOperation(UnaryOperation):
    pass


@dataclass(slots=True)
class NegOperation(UnaryOperation):
    pass


@dataclass(slots=True)
class IndexOperation(ASTNode):
    list_expr: "Expression"
    index_expr: "Expression"


# --- System O operators --------------------------------------------------------


class Associativity(Enum):
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"


@dataclass(slots=True)
class SymbolicOperation(ASTNode):
    operator: str
    operands: list["Expression"]  # 1 for unary, 2 for binary


@dataclass(slots=True)
class PrecedenceDeclaration(ASTNode):
    precedence: int
    associativity: Associativity
    operator: str


# --- Compound expressions ------------------------------------------------------


@dataclass(slots=True)
class IfElse(ASTNode):
    condition: "Expression"
    then_expr: "Expression"
    else_expr: "Expression"


@dataclass(slots=True)
class DoBlock(ASTNode):
    statements: list["Statement"]


@dataclass(slots=True)
class LetStatement(ASTNode):
    variable: str
    value: "Expression"


@dataclass(slots=True)
class FunctionApplication(ASTNode):
    function: "Expression"
    argument: "Expression"


@dataclass(slots=True)
class FieldAssignment(ASTNode):
    field_name: str
    value: "Expression"


@dataclass(slots=True)
class ConstructorExpression(ASTNode):
    constructor_name: str
    fields: list[FieldAssignment]


@dataclass(slots=True)
class GroupedExpression(ASTNode):
    expression: "Expression"


# --- Type expressions ----------------------------------------------------------


@dataclass(slots=True)
class TypeConstructor(ASTNode):
    name: str


@dataclass(slots=True)
class TypeVariable(ASTNode):
    name: str


@dataclass(slots=True)
class ArrowType(ASTNode):
    from_type: "TypeExpression"
    to_type: "TypeExpression"


@dataclass(slots=True)
class TypeApplication(ASTNode):
    constructor: "TypeExpression"
    argument: "TypeExpression"


@dataclass(slots=True)
class GroupedType(ASTNode):
    type_expr: "TypeExpression"


@dataclass(slots=True)
class ListType(ASTNode):
    element_type: "TypeExpression"


@dataclass(slots=True)
class TupleType(ASTNode):
    element_types: list["TypeExpression"]


@dataclass(slots=True)
class TypeConstraint(ASTNode):
    """A constraint ``o :: a -> t`` on a quantified type variable (System O)."""

    name: str
    type_expr: "TypeExpression"


@dataclass(slots=True)
class ConstrainedType(ASTNode):
    """A type scheme ``(o1 :: t1, ...) => t`` as written in an instance declaration."""

    constraints: list[TypeConstraint]
    type_expr: "TypeExpression"


# --- Patterns ------------------------------------------------------------------


@dataclass(slots=True)
class ConstructorPattern(ASTNode):
    constructor: str
    patterns: list["Pattern"]


@dataclass(slots=True)
class ConsPattern(ASTNode):
    head: "Pattern"
    tail: "Pattern"


@dataclass(slots=True)
class VariablePattern(ASTNode):
    name: str


@dataclass(slots=True)
class LiteralPattern(ASTNode):
    # MiniO: the plain value; System O: the literal node (its class gives the type)
    value: Any


@dataclass(slots=True)
class NegativeIntPattern(ASTNode):
    value: int


@dataclass(slots=True)
class NegativeFloatPattern(ASTNode):
    value: float


@dataclass(slots=True)
class TuplePattern(ASTNode):
    patterns: list["Pattern"]


@dataclass(slots=True)
class ListPattern(ASTNode):
    patterns: list["Pattern"]


# --- Declarations --------------------------------------------------------------


@dataclass(slots=True)
class TypeParameter(ASTNode):
    name: str


@dataclass(slots=True)
class Field(ASTNode):
    name: str
    field_type: "TypeExpression"


@dataclass(slots=True)
class RecordConstructor(ASTNode):
    fields: list[Field]


@dataclass(slots=True)
class DataConstructor(ASTNode):
    name: str
    record_constructor: RecordConstructor | None = None
    type_atoms: list["TypeExpression"] | None = None


@dataclass(slots=True)
class DataDeclaration(ASTNode):
    type_name: str
    type_params: list[TypeParameter]
    constructors: list[DataConstructor]


@dataclass(slots=True)
class FunctionDefinition(ASTNode):
    function_name: str
    patterns: list["Pattern"]
    body: "Expression"


@dataclass(slots=True)
class InstanceDeclaration(ASTNode):
    """``inst o :: sigma { clauses }`` -- overloads ``o`` at the type scheme ``sigma``."""

    instance_name: str
    type_signature: "TypeExpression | ConstrainedType"
    clauses: list[FunctionDefinition]


@dataclass(slots=True)
class Program(ASTNode):
    statements: list["Statement"]


# --- Node families -------------------------------------------------------------

type Expression = (
    IntLiteral
    | FloatLiteral
    | StringLiteral
    | CharLiteral
    | BoolLiteral
    | NegativeInt
    | NegativeFloat
    | ListLiteral
    | TupleLiteral
    | Variable
    | Constructor
    | BinaryOperation
    | UnaryOperation
    | IndexOperation
    | SymbolicOperation
    | IfElse
    | DoBlock
    | FunctionApplication
    | ConstructorExpression
    | GroupedExpression
)

type TypeExpression = (
    TypeConstructor
    | TypeVariable
    | ArrowType
    | TypeApplication
    | GroupedType
    | ListType
    | TupleType
)

type Pattern = (
    ConstructorPattern
    | ConsPattern
    | TuplePattern
    | VariablePattern
    | LiteralPattern
    | ListPattern
    | NegativeIntPattern
    | NegativeFloatPattern
)

type Statement = (
    DataDeclaration
    | FunctionDefinition
    | InstanceDeclaration
    | PrecedenceDeclaration
    | LetStatement
    | Expression
)

EXPRESSION_NODES = (
    IntLiteral,
    FloatLiteral,
    StringLiteral,
    CharLiteral,
    BoolLiteral,
    NegativeInt,
    NegativeFloat,
    ListLiteral,
    TupleLiteral,
    Variable,
    Constructor,
    BinaryOperation,
    UnaryOperation,
    IndexOperation,
    SymbolicOperation,
    IfElse,
    DoBlock,
    FunctionApplication,
    ConstructorExpression,
    GroupedExpression,
)


def is_expression(stmt: Any) -> TypeGuard[Expression]:
    return isinstance(stmt, EXPRESSION_NODES)
