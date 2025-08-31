from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Union

from lango.shared.typechecker.lango_types import Type


class Associativity(Enum):
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"


@dataclass
class ASTNode(ABC):
    pass


# Literals
@dataclass
class IntLiteral(ASTNode):
    value: int


@dataclass
class FloatLiteral(ASTNode):
    value: float


@dataclass
class StringLiteral(ASTNode):
    value: str


@dataclass
class CharLiteral(ASTNode):
    value: str


@dataclass
class BoolLiteral(ASTNode):
    value: bool


@dataclass
class ListLiteral(ASTNode):
    elements: List["Expression"]
    ty: Optional[Type] = None


@dataclass
class TupleLiteral(ASTNode):
    elements: List["Expression"]
    ty: Optional[Type] = None


# Variables and Identifiers
@dataclass
class Variable(ASTNode):
    name: str
    ty: Optional[Type] = None


@dataclass
class Constructor(ASTNode):
    name: str
    ty: Optional[Type] = None


# Generic Symbolic Operations
@dataclass
class SymbolicOperation(ASTNode):
    """Generic symbolic operation node for all operators (binary, unary prefix)."""

    operator: str  # The operator symbol, e.g., "+", "(@)", "(?)", etc.
    operands: List["Expression"]  # List of operands (1 for unary, 2 for binary)
    ty: Optional[Type] = None


# Control Flow
@dataclass
class IfElse(ASTNode):
    condition: "Expression"
    then_expr: "Expression"
    else_expr: "Expression"
    ty: Optional[Type] = None


@dataclass
class DoBlock(ASTNode):
    statements: List["Statement"]
    ty: Optional[Type] = None


@dataclass
class LetStatement(ASTNode):
    variable: str
    value: "Expression"
    ty: Optional[Type] = None


# Function Application
@dataclass
class FunctionApplication(ASTNode):
    function: "Expression"
    argument: "Expression"
    ty: Optional[Type] = None


# Constructor Expressions
@dataclass
class FieldAssignment(ASTNode):
    field_name: str
    value: "Expression"
    ty: Optional[Type] = None


@dataclass
class ConstructorExpression(ASTNode):
    constructor_name: str
    fields: List[FieldAssignment]
    ty: Optional[Type] = None


# Grouping
@dataclass
class GroupedExpression(ASTNode):
    expression: "Expression"
    ty: Optional[Type] = None


# Negative numbers
@dataclass
class NegativeInt(ASTNode):
    value: int
    ty: Optional[Type] = None


@dataclass
class NegativeFloat(ASTNode):
    value: float
    ty: Optional[Type] = None


# Type System
@dataclass
class TypeConstructor(ASTNode):
    name: str
    ty: Optional[Type] = None


@dataclass
class TypeVariable(ASTNode):
    name: str
    ty: Optional[Type] = None


@dataclass
class ArrowType(ASTNode):
    from_type: "TypeExpression"
    to_type: "TypeExpression"
    ty: Optional[Type] = None


@dataclass
class TypeApplication(ASTNode):
    constructor: "TypeExpression"
    argument: "TypeExpression"
    ty: Optional[Type] = None


@dataclass
class GroupedType(ASTNode):
    type_expr: "TypeExpression"
    ty: Optional[Type] = None


@dataclass
class ListType(ASTNode):
    element_type: "TypeExpression"
    ty: Optional[Type] = None


@dataclass
class TupleType(ASTNode):
    element_types: List["TypeExpression"]
    ty: Optional[Type] = None


# Patterns
@dataclass
class ConstructorPattern(ASTNode):
    constructor: str
    patterns: List["Pattern"]
    ty: Optional[Type] = None


@dataclass
class ConsPattern(ASTNode):
    head: "Pattern"
    tail: "Pattern"
    ty: Optional[Type] = None


@dataclass
class TuplePattern(ASTNode):
    patterns: List["Pattern"]
    ty: Optional[Type] = None


@dataclass
class VariablePattern(ASTNode):
    name: str
    ty: Optional[Type] = None


@dataclass
class LiteralPattern(ASTNode):
    value: Any
    ty: Optional[Type] = None


@dataclass
class ListPattern(ASTNode):
    patterns: List["Pattern"]
    ty: Optional[Type] = None


@dataclass
class NegativeIntPattern(ASTNode):
    value: int
    ty: Optional[Type] = None


@dataclass
class NegativeFloatPattern(ASTNode):
    value: float
    ty: Optional[Type] = None


# Top-level Declarations
@dataclass
class TypeParameter(ASTNode):
    name: str
    ty: Optional[Type] = None


@dataclass
class Field(ASTNode):
    name: str
    field_type: "TypeExpression"
    ty: Optional[Type] = None


@dataclass
class RecordConstructor(ASTNode):
    fields: List[Field]
    ty: Optional[Type] = None


@dataclass
class DataConstructor(ASTNode):
    name: str
    # Either record_constructor or list of type atoms
    record_constructor: Optional[RecordConstructor] = None
    type_atoms: Optional[List["TypeExpression"]] = None
    ty: Optional[Type] = None


@dataclass
class DataDeclaration(ASTNode):
    type_name: str
    type_params: List[TypeParameter]
    constructors: List[DataConstructor]
    ty: Optional[Type] = None


@dataclass
class FunctionDefinition(ASTNode):
    function_name: str
    patterns: List["Pattern"]
    body: "Expression"
    ty: Optional[Type] = None


@dataclass
class InstanceDeclaration(ASTNode):
    instance_name: str
    type_signature: "TypeExpression"
    function_definition: FunctionDefinition
    ty: Optional[Type] = None


@dataclass
class PrecedenceDeclaration(ASTNode):
    precedence: int
    associativity: Associativity
    operator: str
    ty: Optional[Type] = None


@dataclass
class Program(ASTNode):
    statements: List["Statement"]
    ty: Optional[Type] = None


type Expression = Union[
    IntLiteral,
    FloatLiteral,
    StringLiteral,
    CharLiteral,
    BoolLiteral,
    ListLiteral,
    TupleLiteral,
    Variable,
    Constructor,
    SymbolicOperation,
    IfElse,
    DoBlock,
    FunctionApplication,
    ConstructorExpression,
    GroupedExpression,
    NegativeInt,
    NegativeFloat,
]

type TypeExpression = Union[
    TypeConstructor,
    TypeVariable,
    ArrowType,
    TypeApplication,
    GroupedType,
    ListType,
    TupleType,
]

type Pattern = Union[
    ConstructorPattern,
    ConsPattern,
    TuplePattern,
    VariablePattern,
    LiteralPattern,
    ListPattern,
    NegativeIntPattern,
    NegativeFloatPattern,
]

type Statement = Union[
    DataDeclaration,
    FunctionDefinition,
    InstanceDeclaration,
    PrecedenceDeclaration,
    LetStatement,
    Expression,
]


def is_expression(stmt: Any) -> bool:
    match stmt:
        case (
            IntLiteral()
            | FloatLiteral()
            | StringLiteral()
            | CharLiteral()
            | BoolLiteral()
            | ListLiteral()
            | Variable()
            | Constructor()
            | SymbolicOperation()
            | FunctionApplication()
            | ConstructorExpression()
            | DoBlock()
            | GroupedExpression()
            | IfElse()
        ):
            return True
        case _:
            return False
