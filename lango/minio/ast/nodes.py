from abc import ABC
from dataclasses import dataclass
from typing import Any, List, Optional, Union

from lango.shared.typechecker.lango_types import Type


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


# Arithmetic Operations
@dataclass
class AddOperation(ASTNode):
    left: "Expression"
    right: "Expression"
    ty: Optional[Type] = None


@dataclass
class SubOperation(ASTNode):
    left: "Expression"
    right: "Expression"
    ty: Optional[Type] = None


@dataclass
class MulOperation(ASTNode):
    left: "Expression"
    right: "Expression"
    ty: Optional[Type] = None


@dataclass
class DivOperation(ASTNode):
    left: "Expression"
    right: "Expression"
    ty: Optional[Type] = None


@dataclass
class PowIntOperation(ASTNode):
    left: "Expression"
    right: "Expression"
    ty: Optional[Type] = None


@dataclass
class PowFloatOperation(ASTNode):
    left: "Expression"
    right: "Expression"
    ty: Optional[Type] = None


# Comparison Operations
@dataclass
class EqualOperation(ASTNode):
    left: "Expression"
    right: "Expression"
    ty: Optional[Type] = None


@dataclass
class NotEqualOperation(ASTNode):
    left: "Expression"
    right: "Expression"
    ty: Optional[Type] = None


@dataclass
class LessThanOperation(ASTNode):
    left: "Expression"
    right: "Expression"
    ty: Optional[Type] = None


@dataclass
class LessEqualOperation(ASTNode):
    left: "Expression"
    right: "Expression"
    ty: Optional[Type] = None


@dataclass
class GreaterThanOperation(ASTNode):
    left: "Expression"
    right: "Expression"
    ty: Optional[Type] = None


@dataclass
class GreaterEqualOperation(ASTNode):
    left: "Expression"
    right: "Expression"
    ty: Optional[Type] = None


# Logical Operations
@dataclass
class AndOperation(ASTNode):
    left: "Expression"
    right: "Expression"
    ty: Optional[Type] = None


@dataclass
class OrOperation(ASTNode):
    left: "Expression"
    right: "Expression"
    ty: Optional[Type] = None


@dataclass
class NotOperation(ASTNode):
    operand: "Expression"
    ty: Optional[Type] = None


@dataclass
class NegOperation(ASTNode):
    operand: "Expression"
    ty: Optional[Type] = None


# String/List Operations
@dataclass
class ConcatOperation(ASTNode):
    left: "Expression"
    right: "Expression"
    ty: Optional[Type] = None


@dataclass
class IndexOperation(ASTNode):
    list_expr: "Expression"
    index_expr: "Expression"
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
class VariablePattern(ASTNode):
    name: str
    ty: Optional[Type] = None


@dataclass
class LiteralPattern(ASTNode):
    value: Any
    ty: Optional[Type] = None


@dataclass
class NegativeIntPattern(ASTNode):
    value: int
    ty: Optional[Type] = None


@dataclass
class NegativeFloatPattern(ASTNode):
    value: float
    ty: Optional[Type] = None


@dataclass
class TuplePattern(ASTNode):
    patterns: List["Pattern"]
    ty: Optional[Type] = None


@dataclass
class ListPattern(ASTNode):
    patterns: List["Pattern"]
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
class Program(ASTNode):
    statements: List["Statement"]
    ty: Optional[Type] = None


type Expression = Union[
    IntLiteral,
    FloatLiteral,
    StringLiteral,
    BoolLiteral,
    ListLiteral,
    Variable,
    Constructor,
    AddOperation,
    SubOperation,
    MulOperation,
    DivOperation,
    PowIntOperation,
    PowFloatOperation,
    EqualOperation,
    NotEqualOperation,
    LessThanOperation,
    LessEqualOperation,
    GreaterThanOperation,
    GreaterEqualOperation,
    AndOperation,
    OrOperation,
    NotOperation,
    ConcatOperation,
    IndexOperation,
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
]

type Pattern = Union[
    ConstructorPattern,
    ConsPattern,
    VariablePattern,
    LiteralPattern,
    NegativeIntPattern,
    NegativeFloatPattern,
    TuplePattern,
    ListPattern,
]

type Statement = Union[
    DataDeclaration,
    FunctionDefinition,
    LetStatement,
    Expression,
]
