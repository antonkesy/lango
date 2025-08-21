"""
Abstract Syntax Tree (AST) node definitions for the Minio language.

This module defines custom data types to represent the parsed structure
of Minio programs, replacing raw Lark Tree and Token objects with
structured, type-safe representations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Union


# Base AST Node
@dataclass
class ASTNode(ABC):
    """Base class for all AST nodes."""

    pass


# Literals
@dataclass
class IntLiteral(ASTNode):
    """Integer literal like 42."""

    value: int


@dataclass
class FloatLiteral(ASTNode):
    """Float literal like 3.14."""

    value: float


@dataclass
class StringLiteral(ASTNode):
    """String literal like "hello"."""

    value: str


@dataclass
class BoolLiteral(ASTNode):
    """Boolean literal (True or False)."""

    value: bool


@dataclass
class ListLiteral(ASTNode):
    """List literal like [1, 2, 3]."""

    elements: List["Expression"]


# Variables and Identifiers
@dataclass
class Variable(ASTNode):
    """Variable reference like 'x' or 'myVar'."""

    name: str


@dataclass
class Constructor(ASTNode):
    """Constructor reference like 'Just' or 'Nothing'."""

    name: str


# Arithmetic Operations
@dataclass
class AddOperation(ASTNode):
    """Addition operation: left + right."""

    left: "Expression"
    right: "Expression"


@dataclass
class SubOperation(ASTNode):
    """Subtraction operation: left - right."""

    left: "Expression"
    right: "Expression"


@dataclass
class MulOperation(ASTNode):
    """Multiplication operation: left * right."""

    left: "Expression"
    right: "Expression"


@dataclass
class DivOperation(ASTNode):
    """Division operation: left / right."""

    left: "Expression"
    right: "Expression"


@dataclass
class PowIntOperation(ASTNode):
    """Integer power operation: left ^ right."""

    left: "Expression"
    right: "Expression"


@dataclass
class PowFloatOperation(ASTNode):
    """Float power operation: left ^^ right."""

    left: "Expression"
    right: "Expression"


# Comparison Operations
@dataclass
class EqualOperation(ASTNode):
    """Equality comparison: left == right."""

    left: "Expression"
    right: "Expression"


@dataclass
class NotEqualOperation(ASTNode):
    """Not equal comparison: left /= right."""

    left: "Expression"
    right: "Expression"


@dataclass
class LessThanOperation(ASTNode):
    """Less than comparison: left < right."""

    left: "Expression"
    right: "Expression"


@dataclass
class LessEqualOperation(ASTNode):
    """Less than or equal comparison: left <= right."""

    left: "Expression"
    right: "Expression"


@dataclass
class GreaterThanOperation(ASTNode):
    """Greater than comparison: left > right."""

    left: "Expression"
    right: "Expression"


@dataclass
class GreaterEqualOperation(ASTNode):
    """Greater than or equal comparison: left >= right."""

    left: "Expression"
    right: "Expression"


# Logical Operations
@dataclass
class AndOperation(ASTNode):
    """Logical AND operation: left && right."""

    left: "Expression"
    right: "Expression"


@dataclass
class OrOperation(ASTNode):
    """Logical OR operation: left || right."""

    left: "Expression"
    right: "Expression"


@dataclass
class NotOperation(ASTNode):
    """Logical NOT operation: not expr."""

    operand: "Expression"


# String/List Operations
@dataclass
class ConcatOperation(ASTNode):
    """Concatenation operation: left ++ right."""

    left: "Expression"
    right: "Expression"


@dataclass
class IndexOperation(ASTNode):
    """List indexing operation: list !! index."""

    list_expr: "Expression"
    index_expr: "Expression"


# Control Flow
@dataclass
class IfElse(ASTNode):
    """If-then-else expression."""

    condition: "Expression"
    then_expr: "Expression"
    else_expr: "Expression"


@dataclass
class DoBlock(ASTNode):
    """Do block with statements."""

    statements: List["Statement"]


@dataclass
class LetStatement(ASTNode):
    """Let statement: let var = expr."""

    variable: str
    value: "Expression"


# Function Application
@dataclass
class FunctionApplication(ASTNode):
    """Function application: func arg."""

    function: "Expression"
    argument: "Expression"


# Constructor Expressions
@dataclass
class FieldAssignment(ASTNode):
    """Field assignment in record constructor: field = value."""

    field_name: str
    value: "Expression"


@dataclass
class ConstructorExpression(ASTNode):
    """Record constructor expression: Constructor { field1 = val1, ... }."""

    constructor_name: str
    fields: List[FieldAssignment]


# Grouping
@dataclass
class GroupedExpression(ASTNode):
    """Grouped expression: (expr)."""

    expression: "Expression"


# Negative numbers
@dataclass
class NegativeInt(ASTNode):
    """Negative integer literal: (-42)."""

    value: int


@dataclass
class NegativeFloat(ASTNode):
    """Negative float literal: (-3.14)."""

    value: float


# Type System
@dataclass
class TypeConstructor(ASTNode):
    """Type constructor like Int, String, Bool."""

    name: str


@dataclass
class TypeVariable(ASTNode):
    """Type variable like 'a' or 'b'."""

    name: str


@dataclass
class ArrowType(ASTNode):
    """Function type: from_type -> to_type."""

    from_type: "TypeExpression"
    to_type: "TypeExpression"


@dataclass
class TypeApplication(ASTNode):
    """Type application: List Int, Maybe String."""

    constructor: "TypeExpression"
    argument: "TypeExpression"


@dataclass
class GroupedType(ASTNode):
    """Grouped type expression: (Type)."""

    type_expr: "TypeExpression"


# Patterns
@dataclass
class ConstructorPattern(ASTNode):
    """Constructor pattern: (Constructor pattern1 pattern2 ...)."""

    constructor: str
    patterns: List["Pattern"]


@dataclass
class ConsPattern(ASTNode):
    """Cons pattern: (head : tail)."""

    head: "Pattern"
    tail: "Pattern"


@dataclass
class VariablePattern(ASTNode):
    """Variable pattern in function definition."""

    name: str


@dataclass
class LiteralPattern(ASTNode):
    """Literal pattern (int, float, string, bool)."""

    value: Any


@dataclass
class NegativeIntPattern(ASTNode):
    """Negative integer pattern: (-42)."""

    value: int


@dataclass
class NegativeFloatPattern(ASTNode):
    """Negative float pattern: (-3.14)."""

    value: float


# Top-level Declarations
@dataclass
class TypeParameter(ASTNode):
    """Type parameter in data declaration."""

    name: str


@dataclass
class Field(ASTNode):
    """Field in record constructor: field :: Type."""

    name: str
    field_type: "TypeExpression"


@dataclass
class RecordConstructor(ASTNode):
    """Record constructor definition: { field1 :: Type1, ... }."""

    fields: List[Field]


@dataclass
class DataConstructor(ASTNode):
    """Data constructor definition."""

    name: str
    # Either record_constructor or list of type atoms
    record_constructor: Optional[RecordConstructor] = None
    type_atoms: Optional[List["TypeExpression"]] = None


@dataclass
class DataDeclaration(ASTNode):
    """Data type declaration: data Type param* = Constructor1 | Constructor2 | ..."""

    type_name: str
    type_params: List[TypeParameter]
    constructors: List[DataConstructor]


@dataclass
class FunctionSignature(ASTNode):
    """Function type signature: func :: Type -> Type -> Type."""

    function_name: str
    type_signature: "TypeExpression"


@dataclass
class FunctionDefinition(ASTNode):
    """Function definition: func pattern1 pattern2 = expr."""

    function_name: str
    patterns: List["Pattern"]
    body: "Expression"


@dataclass
class Program(ASTNode):
    """Root program node containing all top-level statements."""

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
