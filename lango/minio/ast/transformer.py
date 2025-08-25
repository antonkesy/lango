"""
AST Transformer for converting Lark parse trees to custom AST nodes.

This module provides a transformer that converts raw Lark Tree and Token
objects into structured AST node classes defined in ast_nodes.py.
"""

from typing import Any, List, Union

from lark import Token, Transformer, Tree

from lango.minio.ast.nodes import (
    AddOperation,
    AndOperation,
    ArrowType,
    BoolLiteral,
    ConcatOperation,
    ConsPattern,
    Constructor,
    ConstructorExpression,
    ConstructorPattern,
    DataConstructor,
    DataDeclaration,
    DivOperation,
    DoBlock,
    EqualOperation,
    Expression,
    Field,
    FieldAssignment,
    FloatLiteral,
    FunctionApplication,
    FunctionDefinition,
    GreaterEqualOperation,
    GreaterThanOperation,
    GroupedExpression,
    GroupedType,
    IfElse,
    IndexOperation,
    IntLiteral,
    LessEqualOperation,
    LessThanOperation,
    LetStatement,
    ListLiteral,
    LiteralPattern,
    MulOperation,
    NegativeFloat,
    NegativeFloatPattern,
    NegativeInt,
    NegativeIntPattern,
    NotEqualOperation,
    NotOperation,
    OrOperation,
    Pattern,
    PowFloatOperation,
    PowIntOperation,
    Program,
    RecordConstructor,
    Statement,
    StringLiteral,
    SubOperation,
    TypeApplication,
    TypeConstructor,
    TypeParameter,
    TypeVariable,
    Variable,
    VariablePattern,
)


class ASTTransformer(Transformer):
    """Transformer that converts Lark parse trees to custom AST nodes."""

    def __init__(self) -> None:
        super().__init__()
        # Add methods for keywords that can't be used as method names
        setattr(self, "and", self._and)
        setattr(self, "or", self._or)
        setattr(self, "not", self._not)

    def _and(self, items: List[Any]) -> AndOperation:
        """Transform logical AND operation."""
        return AndOperation(items[0], items[2])

    def _or(self, items: List[Any]) -> OrOperation:
        """Transform logical OR operation."""
        return OrOperation(items[0], items[2])

    def _not(self, items: List[Any]) -> NotOperation:
        """Transform logical NOT operation."""
        return NotOperation(items[1])

    # Literals
    def int(self, items: List[Any]) -> IntLiteral:
        """Transform integer literal."""
        value = items[0]
        match value:
            case Token():
                return IntLiteral(int(value.value))
            case _:
                return IntLiteral(int(value))

    def float(self, items: List[Any]) -> FloatLiteral:
        """Transform float literal."""
        value = items[0]
        match value:
            case Token():
                return FloatLiteral(float(value.value))
            case _:
                return FloatLiteral(float(value))

    def neg_int(self, items: List[Any]) -> NegativeInt:
        """Transform negative integer literal."""
        value = items[1]
        match value:
            case Token():
                return NegativeInt(-int(value.value))
            case _:
                return NegativeInt(-int(value))

    def neg_float(self, items: List[Any]) -> NegativeFloat:
        """Transform negative float literal."""
        value = items[1]
        match value:
            case Token():
                return NegativeFloat(-float(value.value))
            case _:
                return NegativeFloat(-float(value))

    def string(self, items: List[Any]) -> StringLiteral:
        """Transform string literal."""
        value = items[0]
        match value:
            case Token():
                # Remove quotes
                return StringLiteral(value.value[1:-1])
            case _:
                # Already processed, remove quotes if still present
                text = str(value)
                if text.startswith('"') and text.endswith('"'):
                    return StringLiteral(text[1:-1])
                return StringLiteral(text)

    def true(self, items: List[Any]) -> BoolLiteral:
        """Transform True literal."""
        return BoolLiteral(True)

    def false(self, items: List[Any]) -> BoolLiteral:
        """Transform False literal."""
        return BoolLiteral(False)

    def list(self, items: List[Any]) -> ListLiteral:
        """Transform list literal."""
        # Filter out None items from empty list grammar
        elements = [item for item in items if item is not None]
        return ListLiteral(elements)

    # Variables and Identifiers
    def var(self, items: List[Any]) -> Variable:
        """Transform variable reference."""
        match items[0]:
            case Token(value=value):
                return Variable(value)
            case value:
                return Variable(str(value))

    def constructor(self, items: List[Any]) -> Union[Constructor, DataConstructor]:
        """Transform constructor - either reference or definition based on context."""
        # For constructor references (UIDENT -> constructor), we get just a token
        # For constructor definitions, we get UIDENT plus optional type atoms or record

        match items:
            case [Token(value=value)]:
                # This is a constructor reference
                return Constructor(value)
            case [token, *rest]:
                # This is a data constructor definition
                match token:
                    case Token(value=name):
                        pass
                    case value:
                        name = str(value)

                match rest:
                    case [RecordConstructor() as record_constructor]:
                        # Record constructor
                        return DataConstructor(
                            name,
                            record_constructor=record_constructor,
                        )
                    case type_atoms:
                        # Positional constructor
                        return DataConstructor(name, type_atoms=type_atoms)
            case _:
                # Default case - empty list should never happen in valid input
                raise ValueError(f"Invalid constructor items: {items}")

    # Arithmetic Operations
    def add(self, items: List[Any]) -> AddOperation:
        """Transform addition operation."""
        return AddOperation(items[0], items[2])

    def sub(self, items: List[Any]) -> SubOperation:
        """Transform subtraction operation."""
        return SubOperation(items[0], items[2])

    def mul(self, items: List[Any]) -> MulOperation:
        """Transform multiplication operation."""
        return MulOperation(items[0], items[2])

    def div(self, items: List[Any]) -> DivOperation:
        """Transform division operation."""
        return DivOperation(items[0], items[2])

    def pow_int(self, items: List[Any]) -> PowIntOperation:
        """Transform integer power operation."""
        return PowIntOperation(items[0], items[2])

    def pow_float(self, items: List[Any]) -> PowFloatOperation:
        """Transform float power operation."""
        return PowFloatOperation(items[0], items[2])

    # Comparison Operations
    def eq(self, items: List[Any]) -> EqualOperation:
        """Transform equality comparison."""
        return EqualOperation(items[0], items[2])

    def neq(self, items: List[Any]) -> NotEqualOperation:
        """Transform not equal comparison."""
        return NotEqualOperation(items[0], items[2])

    def lt(self, items: List[Any]) -> LessThanOperation:
        """Transform less than comparison."""
        return LessThanOperation(items[0], items[2])

    def lteq(self, items: List[Any]) -> LessEqualOperation:
        """Transform less than or equal comparison."""
        return LessEqualOperation(items[0], items[2])

    def gt(self, items: List[Any]) -> GreaterThanOperation:
        """Transform greater than comparison."""
        return GreaterThanOperation(items[0], items[2])

    def gteq(self, items: List[Any]) -> GreaterEqualOperation:
        """Transform greater than or equal comparison."""
        return GreaterEqualOperation(items[0], items[2])

    # String/List Operations
    def concat(self, items: List[Any]) -> ConcatOperation:
        """Transform concatenation operation."""
        return ConcatOperation(items[0], items[1])

    def index(self, items: List[Any]) -> IndexOperation:
        """Transform list indexing operation."""
        return IndexOperation(items[0], items[1])

    # Control Flow
    def if_else(self, items: List[Any]) -> IfElse:
        """Transform if-then-else expression."""
        return IfElse(items[0], items[1], items[2])

    def do_block(self, items: List[Any]) -> DoBlock:
        """Transform do block."""
        return DoBlock(items[0])

    def stmt_list(self, items: List[Any]) -> List[Statement]:
        """Transform statement list, flattening any list of statements."""
        flattened = []
        for item in items:
            match item:
                case list() as stmt_list:
                    # Flatten list of statements (e.g., from multiple let bindings)
                    flattened.extend(stmt_list)
                case stmt:
                    flattened.append(stmt)
        return flattened

    def do_stmt(
        self,
        items: List[Any],
    ) -> Union[LetStatement, Expression, List[LetStatement]]:
        """Transform do statement (either let or expression)."""
        if len(items) >= 2 and len(items) % 2 == 0:
            # Let block with multiple assignments
            # Create a LetStatement for each variable=expr pair
            statements = []
            for i in range(0, len(items), 2):
                match items[i]:
                    case Token(value=var_name):
                        pass
                    case value:
                        var_name = str(value)
                statements.append(LetStatement(var_name, items[i + 1]))

            # Return all let statements
            return statements
        else:
            # Just an expression
            return items[0]

    def let(self, items: List[Any]) -> LetStatement:
        """Transform let statement."""
        match items[0]:
            case Token(value=var_name):
                pass
            case value:
                var_name = str(value)
        return LetStatement(var_name, items[1])

    # Function Application
    def app(self, items: List[Any]) -> FunctionApplication:
        """Transform function application."""
        return FunctionApplication(items[0], items[1])

    # Constructor Expressions
    def field_assign(self, items: List[Any]) -> FieldAssignment:
        """Transform field assignment."""
        match items[0]:
            case Token(value=field_name):
                pass
            case value:
                field_name = str(value)
        return FieldAssignment(field_name, items[1])

    def constructor_expr(self, items: List[Any]) -> ConstructorExpression:
        """Transform constructor expression."""
        match items[0]:
            case Token(value=constructor_name):
                pass
            case value:
                constructor_name = str(value)
        fields = items[1:]  # Remaining items are field assignments
        return ConstructorExpression(constructor_name, fields)

    # Grouping
    def grouped(self, items: List[Any]) -> GroupedExpression:
        """Transform grouped expression."""
        return GroupedExpression(items[0])

    # Type System
    def type_constructor(self, items: List[Any]) -> TypeConstructor:
        """Transform type constructor."""
        match items[0]:
            case Token(value=name):
                pass
            case value:
                name = str(value)
        return TypeConstructor(name)

    def type_var(self, items: List[Any]) -> TypeVariable:
        """Transform type variable."""
        match items[0]:
            case Token(value=name):
                pass
            case value:
                name = str(value)
        return TypeVariable(name)

    def arrow_type(self, items: List[Any]) -> ArrowType:
        """Transform arrow type."""
        return ArrowType(items[0], items[1])

    def type_application(self, items: List[Any]) -> TypeApplication:
        """Transform type application."""
        return TypeApplication(items[0], items[1])

    def grouped_type(self, items: List[Any]) -> GroupedType:
        """Transform grouped type."""
        return GroupedType(items[0])

    # Patterns
    def constructor_pattern(self, items: List[Any]) -> ConstructorPattern:
        """Transform constructor pattern."""
        match items[0]:
            case Token(value=constructor_name):
                pass
            case value:
                constructor_name = str(value)
        raw_patterns = items[1:]

        # Convert tokens and expressions to patterns
        converted_patterns: List[Pattern] = []
        for pattern in raw_patterns:
            match pattern:
                case Token(type=token_type, value=token_value):
                    if token_type == "ID":
                        converted_patterns.append(VariablePattern(token_value))
                    else:
                        converted_patterns.append(LiteralPattern(token_value))
                case (
                    IntLiteral(value=value)
                    | FloatLiteral(value=value)
                    | StringLiteral(value=value)
                    | BoolLiteral(value=value)
                ):
                    # Convert literals to literal patterns
                    converted_patterns.append(LiteralPattern(value))
                case _:
                    converted_patterns.append(pattern)

        return ConstructorPattern(constructor_name, converted_patterns)

    def constructor_pattern_bare(self, items: List[Any]) -> ConstructorPattern:
        """Transform bare constructor pattern (nullary constructor used as pattern)."""
        match items[0]:
            case Token(value=constructor_name):
                pass
            case value:
                constructor_name = str(value)
        # Bare constructor pattern has no sub-patterns
        return ConstructorPattern(constructor_name, [])

    def cons_pattern(self, items: List[Any]) -> ConsPattern:
        """Transform cons pattern."""
        # items = [head_pattern, COLON_token, tail_pattern]
        head = items[0]
        tail = items[2]  # Skip the COLON token at items[1]

        # Convert tokens to patterns
        match head:
            case Token(type=token_type, value=token_value):
                if token_type == "ID":
                    head = VariablePattern(token_value)
                else:
                    head = LiteralPattern(token_value)
            case (
                IntLiteral(value=value)
                | FloatLiteral(value=value)
                | StringLiteral(value=value)
                | BoolLiteral(value=value)
            ):
                head = LiteralPattern(value)

        match tail:
            case Token(type=token_type, value=token_value):
                if token_type == "ID":
                    tail = VariablePattern(token_value)
                else:
                    tail = LiteralPattern(token_value)
            case (
                IntLiteral(value=value)
                | FloatLiteral(value=value)
                | StringLiteral(value=value)
                | BoolLiteral(value=value)
            ):
                tail = LiteralPattern(value)

        return ConsPattern(head, tail)

    # Patterns for literals (used in function definitions) - simplified approach
    # Remove __default_token__ to avoid conflicts

    # Top-level Declarations
    def type_param(self, items: List[Any]) -> TypeParameter:
        """Transform type parameter."""
        match items[0]:
            case Token(value=name):
                pass
            case value:
                name = str(value)
        return TypeParameter(name)

    def field(self, items: List[Any]) -> Field:
        """Transform field in record constructor."""
        match items[0]:
            case Token(value=field_name):
                pass
            case value:
                field_name = str(value)
        return Field(field_name, items[1])

    def record_constructor(self, items: List[Any]) -> RecordConstructor:
        """Transform record constructor."""
        fields = [item for item in items if item is not None]
        return RecordConstructor(fields)

    def data_decl(self, items: List[Any]) -> DataDeclaration:
        """Transform data declaration."""
        match items[0]:
            case Token(value=type_name):
                pass
            case value:
                type_name = str(value)

        # Separate type params and constructors
        type_params = []
        constructors = []

        for item in items[1:]:
            match item:
                case TypeParameter() as type_param:
                    type_params.append(type_param)
                case DataConstructor() as data_constructor:
                    constructors.append(data_constructor)
                case Constructor() as constructor:
                    # Convert Constructor to DataConstructor for nullary constructors
                    constructors.append(
                        DataConstructor(constructor.name, type_atoms=[]),
                    )

        return DataDeclaration(type_name, type_params, constructors)

    def func_def(self, items: List[Any]) -> FunctionDefinition:
        """Transform function definition."""
        match items[0]:
            case Token(value=func_name):
                pass
            case value:
                func_name = str(value)
        patterns = items[1:-1]  # Everything between name and expression
        body = items[-1]

        # Convert tokens and expressions to patterns if needed
        converted_patterns: List[Pattern] = []
        for pattern in patterns:
            match pattern:
                case Token(type=token_type, value=token_value):
                    if token_type == "ID":
                        converted_patterns.append(VariablePattern(token_value))
                    else:
                        converted_patterns.append(LiteralPattern(token_value))
                case (
                    IntLiteral(value=v)
                    | FloatLiteral(value=v)
                    | StringLiteral(value=v)
                    | BoolLiteral(value=v)
                ):
                    converted_patterns.append(LiteralPattern(v))
                case ListLiteral(elements):
                    converted_patterns.append(LiteralPattern(elements))
                case _:
                    converted_patterns.append(pattern)

        return FunctionDefinition(func_name, converted_patterns, body)

    # Root
    def start(self, items: List[Any]) -> Program:
        """Transform the root program."""
        statements = []
        for item in items:
            match item:
                case None:
                    continue
                case list() as stmt_list:
                    statements.extend(stmt_list)
                case stmt:
                    statements.append(stmt)

        return Program(statements)


def transform_parse_tree(tree: Tree) -> Program:
    """
    Transform a Lark parse tree into a custom AST.
    """
    transformer = ASTTransformer()
    return transformer.transform(tree)
