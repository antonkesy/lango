from typing import Any, List, Union

from lark import Token, Transformer, Tree

from lango.systemo.ast.nodes import (
    ArrowType,
    BoolLiteral,
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
    InstanceDeclaration,
    IntLiteral,
    LetStatement,
    ListLiteral,
    LiteralPattern,
    NegativeFloat,
    NegativeInt,
    Pattern,
    Program,
    RecordConstructor,
    Statement,
    StringLiteral,
    SymbolicOperation,
    TypeApplication,
    TypeConstructor,
    TypeParameter,
    TypeVariable,
    Variable,
    VariablePattern,
)


class ASTTransformer(Transformer):
    def __init__(self) -> None:
        super().__init__()

    # Literals
    def int(self, items: List[Any]) -> IntLiteral:
        value = items[0]
        match value:
            case Token():
                return IntLiteral(int(value.value))
            case _:
                return IntLiteral(int(value))

    def float(self, items: List[Any]) -> FloatLiteral:
        value = items[0]
        match value:
            case Token():
                return FloatLiteral(float(value.value))
            case _:
                return FloatLiteral(float(value))

    def neg_int(self, items: List[Any]) -> NegativeInt:
        value = items[1]
        match value:
            case Token():
                return NegativeInt(-int(value.value))
            case _:
                return NegativeInt(-int(value))

    def neg_float(self, items: List[Any]) -> NegativeFloat:
        value = items[1]
        match value:
            case Token():
                return NegativeFloat(-float(value.value))
            case _:
                return NegativeFloat(-float(value))

    def string(self, items: List[Any]) -> StringLiteral:
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
        return BoolLiteral(True)

    def false(self, items: List[Any]) -> BoolLiteral:
        return BoolLiteral(False)

    def list(self, items: List[Any]) -> ListLiteral:
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

    def operator_name(self, items: List[Any]) -> str:
        """Transform operator name (either ID or symbolic operator in parens)."""
        match items[0]:
            case Token(value=value):
                return value
            case value:
                return str(value)

    def symbolic_operator(self, items: List[Any]) -> str:
        """Transform symbolic operator token."""
        if not items:
            return ""  # Handle empty symbolic_operator rules
        match items[0]:
            case Token(value=value):
                return f"({value})"  # Wrap in parentheses for symbolic operators
            case value:
                return f"({str(value)})"

    def cons_op(self, items: List[Any]) -> FunctionApplication:
        """Transform binary : operator into function application."""
        # items[0] is the left operand, items[1] is the COLON token, items[2] is the right operand
        colon_var = Variable("(:)")  # Reference to the (:) function
        left_operand = items[0]
        right_operand = items[2]
        # Create nested function application: ((:) left) right
        partial_app = FunctionApplication(colon_var, left_operand)
        return FunctionApplication(partial_app, right_operand)

    def constructor(self, items: List[Any]) -> Union[Constructor, DataConstructor]:
        """Transform constructor - either reference or definition based on context."""
        # For constructor references (UIDENT -> constructor), we get just a token
        # For constructor definitions, we get UIDENT plus optional type atoms or record

        match items:
            case [Token(value=value)]:
                return Constructor(value)
            case [token, *rest]:
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

    # Generic Operator Transformations  
    def __default__(self, data: str, children: List[Any], meta=None) -> Any:
        """Handle generic operator transformations."""
        # Check if this is an operator rule
        if data.startswith(("infix_op_", "prefix_op_", "postfix_op_")):
            # Instead of trying to reverse-engineer the operator, let's just use 
            # the rule name directly as a unique identifier for the operation
            # The actual operator symbol doesn't matter for the AST representation
            return SymbolicOperation(data, children)

        # If not a recognized operator, fall back to default behavior
        return super().__default__(data, children, meta)    # Control Flow
    def if_else(self, items: List[Any]) -> IfElse:
        return IfElse(items[0], items[1], items[2])

    def do_block(self, items: List[Any]) -> DoBlock:
        return DoBlock(items[0])

    def stmt_list(self, items: List[Any]) -> List[Statement]:
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
        match items[0]:
            case Token(value=var_name):
                pass
            case value:
                var_name = str(value)
        return LetStatement(var_name, items[1])

    # Function Application
    def app(self, items: List[Any]) -> FunctionApplication:
        return FunctionApplication(items[0], items[1])

    # Constructor Expressions
    def field_assign(self, items: List[Any]) -> FieldAssignment:
        match items[0]:
            case Token(value=field_name):
                pass
            case value:
                field_name = str(value)
        return FieldAssignment(field_name, items[1])

    def constructor_expr(self, items: List[Any]) -> ConstructorExpression:
        match items[0]:
            case Token(value=constructor_name):
                pass
            case value:
                constructor_name = str(value)
        fields = items[1:]  # Remaining items are field assignments
        return ConstructorExpression(constructor_name, fields)

    # Grouping
    def grouped(self, items: List[Any]) -> GroupedExpression:
        return GroupedExpression(items[0])

    # Type System
    def type_constructor(self, items: List[Any]) -> TypeConstructor:
        match items[0]:
            case Token(value=name):
                pass
            case value:
                name = str(value)
        return TypeConstructor(name)

    def type_var(self, items: List[Any]) -> TypeVariable:
        match items[0]:
            case Token(value=name):
                pass
            case value:
                name = str(value)
        return TypeVariable(name)

    def arrow_type(self, items: List[Any]) -> ArrowType:
        return ArrowType(items[0], items[1])

    def type_application(self, items: List[Any]) -> TypeApplication:
        return TypeApplication(items[0], items[1])

    def grouped_type(self, items: List[Any]) -> GroupedType:
        return GroupedType(items[0])

    # Patterns
    def constructor_pattern(self, items: List[Any]) -> ConstructorPattern:
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
        match items[0]:
            case Token(value=constructor_name):
                pass
            case value:
                constructor_name = str(value)
        # Bare constructor pattern has no sub-patterns
        return ConstructorPattern(constructor_name, [])

    def operator_pattern(self, items: List[Any]) -> VariablePattern:
        """Transform operator pattern into variable pattern."""
        # items[0] should be the operator name string
        operator_name = items[0]
        return VariablePattern(operator_name)

    def symbolic_operator_pattern(self, items: List[Any]) -> VariablePattern:
        """Transform symbolic operator pattern into variable pattern."""
        # items[0] should be the symbolic operator string
        symbolic_op = items[0]
        # Wrap in parentheses to match the operator name format
        if isinstance(symbolic_op, str):
            operator_name = f"({symbolic_op})"
        else:
            operator_name = f"({str(symbolic_op)})"
        return VariablePattern(operator_name)

    def cons_pattern(self, items: List[Any]) -> ConsPattern:
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

    # Top-level Declarations
    def type_param(self, items: List[Any]) -> TypeParameter:
        match items[0]:
            case Token(value=name):
                pass
            case value:
                name = str(value)
        return TypeParameter(name)

    def field(self, items: List[Any]) -> Field:
        match items[0]:
            case Token(value=field_name):
                pass
            case value:
                field_name = str(value)
        return Field(field_name, items[1])

    def record_constructor(self, items: List[Any]) -> RecordConstructor:
        fields = [item for item in items if item is not None]
        return RecordConstructor(fields)

    def data_decl(self, items: List[Any]) -> DataDeclaration:
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

    def inst_decl(self, items: List[Any]) -> InstanceDeclaration:
        # items should be: [instance_name, type_signature, function_definition]
        match items[0]:
            case Token(value=instance_name):
                pass
            case value:
                instance_name = str(value)

        type_signature = items[1]  # This should be a TypeExpression
        function_definition = items[2]  # This should be a FunctionDefinition

        return InstanceDeclaration(instance_name, type_signature, function_definition)

    # Root
    def start(self, items: List[Any]) -> Program:
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
    transformer = ASTTransformer()
    return transformer.transform(tree)
