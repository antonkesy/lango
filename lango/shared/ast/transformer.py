"""Parse tree -> AST: the rules both languages share (see ``lango/shared/lango.lark``).

The language specific transformers add the rules of their own grammar and
override the hooks ``literal_text`` and ``literal_pattern``.
"""

from typing import Any

from lark import Token, Transformer, Tree

from lango.shared.ast.nodes import (
    ArrowType,
    BoolLiteral,
    CharLiteral,
    ConsPattern,
    Constructor,
    ConstructorExpression,
    ConstructorPattern,
    DataConstructor,
    DataDeclaration,
    DoBlock,
    Expression,
    Field,
    FieldAssignment,
    FloatLiteral,
    FunctionApplication,
    FunctionDefinition,
    GroupedExpression,
    GroupedType,
    IfElse,
    IntLiteral,
    ListLiteral,
    ListPattern,
    LiteralPattern,
    NegativeFloat,
    NegativeFloatPattern,
    NegativeInt,
    NegativeIntPattern,
    Pattern,
    Program,
    RecordConstructor,
    Statement,
    StringLiteral,
    TupleLiteral,
    TuplePattern,
    TupleType,
    TypeApplication,
    TypeConstructor,
    TypeParameter,
    TypeVariable,
    Variable,
    VariablePattern,
)


def text(item: Any) -> str:
    """The source text of a token (or the string of an already transformed item)."""
    match item:
        case Token(value=value):
            return str(value)
        case _:
            return str(item)


class SharedTransformer(Transformer):
    # --- hooks -----------------------------------------------------------------
    def literal_text(self, quoted: str) -> str:
        """The value of a string or character literal without its quotes."""
        return quoted[1:-1]

    # --- Literals ----------------------------------------------------------------
    def int(self, items: list[Any]) -> IntLiteral:
        return IntLiteral(int(text(items[0])))

    def float(self, items: list[Any]) -> FloatLiteral:
        return FloatLiteral(float(text(items[0])))

    def neg_int(self, items: list[Any]) -> NegativeInt:
        return NegativeInt(-int(text(items[0])))

    def neg_float(self, items: list[Any]) -> NegativeFloat:
        return NegativeFloat(-float(text(items[0])))

    def string(self, items: list[Any]) -> StringLiteral:
        return StringLiteral(self.literal_text(text(items[0])))

    def char(self, items: list[Any]) -> CharLiteral:
        return CharLiteral(self.literal_text(text(items[0])))

    def true(self, items: list[Any]) -> BoolLiteral:
        return BoolLiteral(True)

    def false(self, items: list[Any]) -> BoolLiteral:
        return BoolLiteral(False)

    def list(self, items: list[Any]) -> ListLiteral:
        return ListLiteral([item for item in items if item is not None])

    def tuple_literal(self, items: list[Any]) -> TupleLiteral:
        return TupleLiteral(list(items))

    # --- Names -------------------------------------------------------------------
    def var(self, items: list[Any]) -> Variable:
        return Variable(text(items[0]))

    def constructor(self, items: list[Any]) -> Constructor | DataConstructor:
        # Both constructor references (in expressions) and constructor
        # definitions (in data declarations) reduce through this rule.
        name = text(items[0])
        match items[1:]:
            case []:
                return Constructor(name)
            case [RecordConstructor() as record]:
                return DataConstructor(name, record_constructor=record)
            case type_atoms:
                return DataConstructor(name, type_atoms=list(type_atoms))

    # --- Control flow ------------------------------------------------------------
    def if_else(self, items: list[Any]) -> IfElse:
        return IfElse(items[0], items[1], items[2])

    def do_block(self, items: list[Any]) -> DoBlock:
        return DoBlock(items[0])

    def stmt_list(self, items: list[Any]) -> list[Statement]:
        flattened: list[Statement] = []
        for item in items:
            match item:
                case list() as lets:
                    flattened.extend(lets)
                case stmt:
                    flattened.append(stmt)
        return flattened

    # --- Applications and constructors -------------------------------------------
    def app(self, items: list[Any]) -> FunctionApplication:
        return FunctionApplication(items[0], items[1])

    def field_assign(self, items: list[Any]) -> FieldAssignment:
        return FieldAssignment(text(items[0]), items[1])

    def constructor_expr(self, items: list[Any]) -> ConstructorExpression:
        return ConstructorExpression(text(items[0]), items[1:])

    def grouped(self, items: list[Any]) -> GroupedExpression:
        return GroupedExpression(items[0])

    # --- Types -------------------------------------------------------------------
    def type_constructor(self, items: list[Any]) -> TypeConstructor:
        return TypeConstructor(text(items[0]))

    def type_var(self, items: list[Any]) -> TypeVariable:
        return TypeVariable(text(items[0]))

    def arrow_type(self, items: list[Any]) -> ArrowType:
        return ArrowType(items[0], items[1])

    def type_application(self, items: list[Any]) -> TypeApplication:
        return TypeApplication(items[0], items[1])

    def grouped_type(self, items: list[Any]) -> GroupedType:
        return GroupedType(items[0])

    def tuple_type(self, items: list[Any]) -> TupleType:
        return TupleType(list(items))

    # --- Patterns ----------------------------------------------------------------
    def var_pattern(self, items: list[Any]) -> VariablePattern:
        return VariablePattern(text(items[0]))

    def literal_pattern(self, items: list[Any]) -> LiteralPattern:
        # The literal node itself is kept so that the type of the literal
        # (Int, Float, String, Char, Bool) is known to the type checker.
        return LiteralPattern(items[0])

    def neg_int_pattern(self, items: list[Any]) -> NegativeIntPattern:
        return NegativeIntPattern(-int(text(items[0])))

    def neg_float_pattern(self, items: list[Any]) -> NegativeFloatPattern:
        return NegativeFloatPattern(-float(text(items[0])))

    def constructor_pattern(self, items: list[Any]) -> ConstructorPattern:
        return ConstructorPattern(text(items[0]), items[1:])

    def constructor_pattern_bare(self, items: list[Any]) -> ConstructorPattern:
        return ConstructorPattern(text(items[0]), [])

    def cons_pattern(self, items: list[Any]) -> ConsPattern:
        head, _colon, tail = items
        return ConsPattern(head, tail)

    def tuple_pattern(self, items: list[Any]) -> TuplePattern:
        return TuplePattern(list(items))

    def list_pattern(self, items: list[Any]) -> ListPattern:
        return ListPattern([item for item in items if item is not None])

    # --- Top-level declarations --------------------------------------------------
    def type_param(self, items: list[Any]) -> TypeParameter:
        return TypeParameter(text(items[0]))

    def field(self, items: list[Any]) -> Field:
        return Field(text(items[0]), items[1])

    def record_constructor(self, items: list[Any]) -> RecordConstructor:
        return RecordConstructor([item for item in items if item is not None])

    def data_decl(self, items: list[Any]) -> DataDeclaration:
        type_name = text(items[0])
        type_params: list[TypeParameter] = []
        constructors: list[DataConstructor] = []
        for item in items[1:]:
            match item:
                case TypeParameter():
                    type_params.append(item)
                case DataConstructor():
                    constructors.append(item)
                case Constructor(name=name):
                    constructors.append(DataConstructor(name, type_atoms=[]))
        return DataDeclaration(type_name, type_params, constructors)

    def func_def(self, items: list[Any]) -> FunctionDefinition:
        name = text(items[0])
        patterns: list[Pattern] = items[1:-1]
        body: Expression = items[-1]
        return FunctionDefinition(name, patterns, body)

    # --- Root --------------------------------------------------------------------
    def start(self, items: list[Any]) -> Program:
        statements: list[Statement] = []
        for item in items:
            match item:
                case None | Tree():
                    # empty prelude/postlude comment groups
                    continue
                case stmt:
                    statements.append(stmt)
        return Program(statements)
