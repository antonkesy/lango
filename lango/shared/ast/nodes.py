from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, TypeGuard, Union

from lango.shared.typechecker.lango_types import Type


@dataclass
class ASTNode(ABC):
    pass


@dataclass
class IntLiteral(ASTNode):
    value: int
    ty: Optional[Type] = None


@dataclass
class FloatLiteral(ASTNode):
    value: float
    ty: Optional[Type] = None


@dataclass
class StringLiteral(ASTNode):
    value: str
    ty: Optional[Type] = None


@dataclass
class CharLiteral(ASTNode):
    value: str
    ty: Optional[Type] = None


@dataclass
class BoolLiteral(ASTNode):
    value: bool
    ty: Optional[Type] = None


@dataclass
class ListLiteral(ASTNode):
    elements: List["Expression"]
    ty: Optional[Type] = None


@dataclass
class TupleLiteral(ASTNode):
    elements: List["Expression"]
    ty: Optional[Type] = None


@dataclass
class Variable(ASTNode):
    name: str
    ty: Optional[Type] = None


@dataclass
class Constructor(ASTNode):
    name: str
    ty: Optional[Type] = None


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


@dataclass
class FunctionApplication(ASTNode):
    function: "Expression"
    argument: "Expression"
    ty: Optional[Type] = None


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


@dataclass
class GroupedExpression(ASTNode):
    expression: "Expression"
    ty: Optional[Type] = None


@dataclass
class NegativeInt(ASTNode):
    value: int
    ty: Optional[Type] = None


@dataclass
class NegativeFloat(ASTNode):
    value: float
    ty: Optional[Type] = None


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


def is_minio_expression(stmt: Any) -> TypeGuard["Expression"]:
    match stmt:
        case (
            IntLiteral()
            | FloatLiteral()
            | StringLiteral()
            | CharLiteral()
            | BoolLiteral()
            | ListLiteral()
            | TupleLiteral()
            | NegativeInt()
            | NegativeFloat()
            | Variable()
            | Constructor()
            | AddOperation()
            | SubOperation()
            | MulOperation()
            | DivOperation()
            | PowIntOperation()
            | PowFloatOperation()
            | EqualOperation()
            | NotEqualOperation()
            | LessThanOperation()
            | LessEqualOperation()
            | GreaterThanOperation()
            | GreaterEqualOperation()
            | ConcatOperation()
            | AndOperation()
            | OrOperation()
            | NotOperation()
            | NegOperation()
            | IndexOperation()
            | IfElse()
            | FunctionApplication()
            | ConstructorExpression()
            | DoBlock()
            | GroupedExpression()
        ):
            return True
        case _:
            return False


# SystemO
class Associativity(Enum):
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"


@dataclass
class SymbolicOperation(ASTNode):
    operator: str
    operands: List["Expression"]  # List of operands (1 for unary, 2 for binary)
    ty: Optional[Type] = None


@dataclass
class ListType(ASTNode):
    element_type: "TypeExpression"
    ty: Optional[Type] = None


@dataclass
class TypeConstraint(ASTNode):
    """A constraint ``o :: a -> t`` on a quantified type variable (System O)."""

    name: str
    type_expr: "TypeExpression"
    ty: Optional[Type] = None


@dataclass
class ConstrainedType(ASTNode):
    """A type scheme ``(o1 :: t1, ...) => t`` as written in an instance declaration."""

    constraints: List[TypeConstraint]
    type_expr: "TypeExpression"
    ty: Optional[Type] = None


@dataclass
class InstanceDeclaration(ASTNode):
    """``inst o :: sigma { clauses }`` -- overloads ``o`` at the type scheme ``sigma``."""

    instance_name: str
    type_signature: Union["TypeExpression", ConstrainedType]
    clauses: List[FunctionDefinition]
    ty: Optional[Type] = None


@dataclass
class PrecedenceDeclaration(ASTNode):
    precedence: int
    associativity: Associativity
    operator: str
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
