from abc import ABC
from dataclasses import dataclass
from typing import Any, List, Optional, Union


# Base AST Node
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
class BoolLiteral(ASTNode):
    value: bool


@dataclass
class ListLiteral(ASTNode):
    elements: List["Expression"]


# Variables and Identifiers
@dataclass
class Variable(ASTNode):
    name: str


@dataclass
class Constructor(ASTNode):
    name: str


# Arithmetic Operations
@dataclass
class AddOperation(ASTNode):
    left: "Expression"
    right: "Expression"


@dataclass
class SubOperation(ASTNode):
    left: "Expression"
    right: "Expression"


@dataclass
class MulOperation(ASTNode):
    left: "Expression"
    right: "Expression"


@dataclass
class DivOperation(ASTNode):
    left: "Expression"
    right: "Expression"


@dataclass
class PowIntOperation(ASTNode):
    left: "Expression"
    right: "Expression"


@dataclass
class PowFloatOperation(ASTNode):
    left: "Expression"
    right: "Expression"


# Comparison Operations
@dataclass
class EqualOperation(ASTNode):
    left: "Expression"
    right: "Expression"


@dataclass
class NotEqualOperation(ASTNode):
    left: "Expression"
    right: "Expression"


@dataclass
class LessThanOperation(ASTNode):
    left: "Expression"
    right: "Expression"


@dataclass
class LessEqualOperation(ASTNode):
    left: "Expression"
    right: "Expression"


@dataclass
class GreaterThanOperation(ASTNode):
    left: "Expression"
    right: "Expression"


@dataclass
class GreaterEqualOperation(ASTNode):
    left: "Expression"
    right: "Expression"


# Logical Operations
@dataclass
class AndOperation(ASTNode):
    left: "Expression"
    right: "Expression"


@dataclass
class OrOperation(ASTNode):
    left: "Expression"
    right: "Expression"


@dataclass
class NotOperation(ASTNode):
    operand: "Expression"


# String/List Operations
@dataclass
class ConcatOperation(ASTNode):
    left: "Expression"
    right: "Expression"


@dataclass
class IndexOperation(ASTNode):
    list_expr: "Expression"
    index_expr: "Expression"


# Control Flow
@dataclass
class IfElse(ASTNode):
    condition: "Expression"
    then_expr: "Expression"
    else_expr: "Expression"


@dataclass
class DoBlock(ASTNode):
    statements: List["Statement"]


@dataclass
class LetStatement(ASTNode):
    variable: str
    value: "Expression"


# Function Application
@dataclass
class FunctionApplication(ASTNode):
    function: "Expression"
    argument: "Expression"


# Constructor Expressions
@dataclass
class FieldAssignment(ASTNode):
    field_name: str
    value: "Expression"


@dataclass
class ConstructorExpression(ASTNode):
    constructor_name: str
    fields: List[FieldAssignment]


# Grouping
@dataclass
class GroupedExpression(ASTNode):
    expression: "Expression"


# Negative numbers
@dataclass
class NegativeInt(ASTNode):
    value: int


@dataclass
class NegativeFloat(ASTNode):
    value: float


# Type System
@dataclass
class TypeConstructor(ASTNode):
    name: str


@dataclass
class TypeVariable(ASTNode):
    name: str


@dataclass
class ArrowType(ASTNode):
    from_type: "TypeExpression"
    to_type: "TypeExpression"


@dataclass
class TypeApplication(ASTNode):
    constructor: "TypeExpression"
    argument: "TypeExpression"


@dataclass
class GroupedType(ASTNode):
    type_expr: "TypeExpression"


# Patterns
@dataclass
class ConstructorPattern(ASTNode):
    constructor: str
    patterns: List["Pattern"]


@dataclass
class ConsPattern(ASTNode):
    head: "Pattern"
    tail: "Pattern"


@dataclass
class VariablePattern(ASTNode):
    name: str


@dataclass
class LiteralPattern(ASTNode):
    value: Any


@dataclass
class NegativeIntPattern(ASTNode):
    value: int


@dataclass
class NegativeFloatPattern(ASTNode):
    value: float


# Top-level Declarations
@dataclass
class TypeParameter(ASTNode):
    """Type parameter in data declaration."""

    name: str


@dataclass
class Field(ASTNode):
    name: str
    field_type: "TypeExpression"


@dataclass
class RecordConstructor(ASTNode):
    fields: List[Field]


@dataclass
class DataConstructor(ASTNode):
    name: str
    # Either record_constructor or list of type atoms
    record_constructor: Optional[RecordConstructor] = None
    type_atoms: Optional[List["TypeExpression"]] = None


@dataclass
class DataDeclaration(ASTNode):
    type_name: str
    type_params: List[TypeParameter]
    constructors: List[DataConstructor]


@dataclass
class FunctionSignature(ASTNode):
    function_name: str
    type_signature: "TypeExpression"


@dataclass
class FunctionDefinition(ASTNode):
    function_name: str
    patterns: List["Pattern"]
    body: "Expression"


@dataclass
class Program(ASTNode):
    statements: List["Statement"]


# Union types for different categories
Expression = Union[
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

TypeExpression = Union[
    TypeConstructor,
    TypeVariable,
    ArrowType,
    TypeApplication,
    GroupedType,
]

Pattern = Union[
    ConstructorPattern,
    ConsPattern,
    VariablePattern,
    LiteralPattern,
    NegativeIntPattern,
    NegativeFloatPattern,
]

Statement = Union[
    DataDeclaration,
    FunctionSignature,
    FunctionDefinition,
    LetStatement,
    Expression,  # Expressions can be statements too
]
