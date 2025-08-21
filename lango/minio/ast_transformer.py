"""
AST Transformer for converting Lark parse trees to custom AST nodes.

This module provides a transformer that converts raw Lark Tree and Token
objects into our structured AST node classes defined in ast_nodes.py.
"""

from typing import Any, List, Optional, Union

from lark import Token, Transformer, Tree

from .ast_nodes import (
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
    FunctionSignature,
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

    def __init__(self):
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
        if isinstance(value, Token):
            return IntLiteral(int(value.value))
        else:
            return IntLiteral(int(value))

    def float(self, items: List[Any]) -> FloatLiteral:
        """Transform float literal."""
        value = items[0]
        if isinstance(value, Token):
            return FloatLiteral(float(value.value))
        else:
            return FloatLiteral(float(value))

    def neg_int(self, items: List[Any]) -> NegativeInt:
        """Transform negative integer literal."""
        value = items[1]
        if isinstance(value, Token):
            return NegativeInt(-int(value.value))
        else:
            return NegativeInt(-int(value))

    def neg_float(self, items: List[Any]) -> NegativeFloat:
        """Transform negative float literal."""
        value = items[1]
        if isinstance(value, Token):
            return NegativeFloat(-float(value.value))
        else:
            return NegativeFloat(-float(value))

    def string(self, items: List[Any]) -> StringLiteral:
        """Transform string literal."""
        value = items[0]
        if isinstance(value, Token):
            # Remove quotes
            return StringLiteral(value.value[1:-1])
        else:
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
        return Variable(
            items[0].value if isinstance(items[0], Token) else str(items[0]),
        )

    def constructor(self, items: List[Any]) -> Union[Constructor, DataConstructor]:
        """Transform constructor - either reference or definition based on context."""
        # For constructor references (UIDENT -> constructor), we get just a token
        # For constructor definitions, we get UIDENT plus optional type atoms or record

        if len(items) == 1 and isinstance(items[0], Token):
            # This is a constructor reference
            return Constructor(items[0].value)
        else:
            # This is a data constructor definition
            name = items[0].value if isinstance(items[0], Token) else str(items[0])

            if len(items) > 1 and isinstance(items[1], RecordConstructor):
                # Record constructor
                return DataConstructor(name, record_constructor=items[1])
            else:
                # Positional constructor
                type_atoms = items[1:] if len(items) > 1 else []
                return DataConstructor(name, type_atoms=type_atoms)

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
            if isinstance(item, list):
                # Flatten list of statements (e.g., from multiple let bindings)
                flattened.extend(item)
            else:
                flattened.append(item)
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
                var_name = (
                    items[i].value if isinstance(items[i], Token) else str(items[i])
                )
                statements.append(LetStatement(var_name, items[i + 1]))

            # Return all let statements
            return statements
        else:
            # Just an expression
            return items[0]

    def let(self, items: List[Any]) -> LetStatement:
        """Transform let statement."""
        var_name = items[0].value if isinstance(items[0], Token) else str(items[0])
        return LetStatement(var_name, items[1])

    # Function Application
    def app(self, items: List[Any]) -> FunctionApplication:
        """Transform function application."""
        return FunctionApplication(items[0], items[1])

    # Constructor Expressions
    def field_assign(self, items: List[Any]) -> FieldAssignment:
        """Transform field assignment."""
        field_name = items[0].value if isinstance(items[0], Token) else str(items[0])
        return FieldAssignment(field_name, items[1])

    def constructor_expr(self, items: List[Any]) -> ConstructorExpression:
        """Transform constructor expression."""
        constructor_name = (
            items[0].value if isinstance(items[0], Token) else str(items[0])
        )
        fields = items[1:]  # Remaining items are field assignments
        return ConstructorExpression(constructor_name, fields)

    # Grouping
    def grouped(self, items: List[Any]) -> GroupedExpression:
        """Transform grouped expression."""
        return GroupedExpression(items[0])

    # Type System
    def type_constructor(self, items: List[Any]) -> TypeConstructor:
        """Transform type constructor."""
        name = items[0].value if isinstance(items[0], Token) else str(items[0])
        return TypeConstructor(name)

    def type_var(self, items: List[Any]) -> TypeVariable:
        """Transform type variable."""
        name = items[0].value if isinstance(items[0], Token) else str(items[0])
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
        constructor_name = (
            items[0].value if isinstance(items[0], Token) else str(items[0])
        )
        raw_patterns = items[1:]

        # Convert tokens and expressions to patterns
        converted_patterns = []
        for pattern in raw_patterns:
            if isinstance(pattern, Token):
                if pattern.type == "ID":
                    converted_patterns.append(VariablePattern(pattern.value))
                else:
                    converted_patterns.append(LiteralPattern(pattern.value))
            elif isinstance(
                pattern,
                (IntLiteral, FloatLiteral, StringLiteral, BoolLiteral),
            ):
                # Convert literals to literal patterns
                converted_patterns.append(LiteralPattern(pattern.value))
            else:
                converted_patterns.append(pattern)

        return ConstructorPattern(constructor_name, converted_patterns)

    def cons_pattern(self, items: List[Any]) -> ConsPattern:
        """Transform cons pattern."""
        # items = [head_pattern, COLON_token, tail_pattern]
        head = items[0]
        tail = items[2]  # Skip the COLON token at items[1]

        # Convert tokens to patterns
        if isinstance(head, Token):
            if head.type == "ID":
                head = VariablePattern(head.value)
            else:
                head = LiteralPattern(head.value)
        elif isinstance(head, (IntLiteral, FloatLiteral, StringLiteral, BoolLiteral)):
            head = LiteralPattern(head.value)

        if isinstance(tail, Token):
            if tail.type == "ID":
                tail = VariablePattern(tail.value)
            else:
                tail = LiteralPattern(tail.value)
        elif isinstance(tail, (IntLiteral, FloatLiteral, StringLiteral, BoolLiteral)):
            tail = LiteralPattern(tail.value)

        return ConsPattern(head, tail)

    # Patterns for literals (used in function definitions) - simplified approach
    # Remove __default_token__ to avoid conflicts

    # Top-level Declarations
    def type_param(self, items: List[Any]) -> TypeParameter:
        """Transform type parameter."""
        name = items[0].value if isinstance(items[0], Token) else str(items[0])
        return TypeParameter(name)

    def field(self, items: List[Any]) -> Field:
        """Transform field in record constructor."""
        field_name = items[0].value if isinstance(items[0], Token) else str(items[0])
        return Field(field_name, items[1])

    def record_constructor(self, items: List[Any]) -> RecordConstructor:
        """Transform record constructor."""
        fields = [item for item in items if item is not None]
        return RecordConstructor(fields)

    def data_decl(self, items: List[Any]) -> DataDeclaration:
        """Transform data declaration."""
        type_name = items[0].value if isinstance(items[0], Token) else str(items[0])

        # Separate type params and constructors
        type_params = []
        constructors = []

        for item in items[1:]:
            if isinstance(item, TypeParameter):
                type_params.append(item)
            elif isinstance(item, DataConstructor):
                constructors.append(item)

        return DataDeclaration(type_name, type_params, constructors)

    def func_sig(self, items: List[Any]) -> FunctionSignature:
        """Transform function signature."""
        func_name = items[0].value if isinstance(items[0], Token) else str(items[0])

        # Build the type signature from the type expressions
        if len(items) == 2:
            type_sig = items[1]
        else:
            # Multiple types connected with arrows
            type_sig = items[1]
            for i in range(2, len(items)):
                type_sig = ArrowType(type_sig, items[i])

        return FunctionSignature(func_name, type_sig)

    def func_def(self, items: List[Any]) -> FunctionDefinition:
        """Transform function definition."""
        func_name = items[0].value if isinstance(items[0], Token) else str(items[0])
        patterns = items[1:-1]  # Everything between name and expression
        body = items[-1]

        # Convert tokens and expressions to patterns if needed
        converted_patterns = []
        for pattern in patterns:
            if isinstance(pattern, Token):
                if pattern.type == "ID":
                    converted_patterns.append(VariablePattern(pattern.value))
                else:
                    converted_patterns.append(LiteralPattern(pattern.value))
            elif isinstance(
                pattern,
                (IntLiteral, FloatLiteral, StringLiteral, BoolLiteral),
            ):
                # Convert literals to literal patterns
                converted_patterns.append(LiteralPattern(pattern.value))
            else:
                converted_patterns.append(pattern)

        return FunctionDefinition(func_name, converted_patterns, body)

    # Root
    def start(self, items: List[Any]) -> Program:
        """Transform the root program."""
        # Filter out prelude/postlude comments and None items
        statements = []
        for item in items:
            if item is not None and not isinstance(item, list):
                statements.append(item)
            elif isinstance(item, list):
                statements.extend(item)

        return Program(statements)

    def prelude(self, items: List[Any]) -> None:
        """Transform prelude (comments) - ignore."""
        return None

    def postlude(self, items: List[Any]) -> None:
        """Transform postlude (comments) - ignore."""
        return None


def transform_parse_tree(tree: Tree) -> Program:
    """
    Transform a Lark parse tree into a custom AST.
    """
    transformer = ASTTransformer()
    return transformer.transform(tree)
