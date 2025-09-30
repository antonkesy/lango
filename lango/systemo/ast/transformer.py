from typing import Any, List, Union

from lark import Token, Transformer, Tree

from lango.shared.ast.nodes import (
    ArrowType,
    Associativity,
    BoolLiteral,
    CharLiteral,
    ConsPattern,
    ConstrainedType,
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
    ListPattern,
    ListType,
    LiteralPattern,
    NegativeFloat,
    NegativeFloatPattern,
    NegativeInt,
    NegativeIntPattern,
    Pattern,
    PrecedenceDeclaration,
    Program,
    RecordConstructor,
    Statement,
    StringLiteral,
    SymbolicOperation,
    TupleLiteral,
    TuplePattern,
    TupleType,
    TypeApplication,
    TypeConstraint,
    TypeConstructor,
    TypeParameter,
    TypeVariable,
    Variable,
    VariablePattern,
)

# Prefix minus is syntactic sugar for an application of the overloaded
# function ``negate`` (System O overloads functions, not syntax).
NEGATE = "negate"


def _text(item: Any) -> str:
    match item:
        case Token(value=value):
            return str(value)
        case _:
            return str(item)


def _unescape(text: str) -> str:
    return text.encode("utf-8").decode("unicode_escape")


class ASTTransformer(Transformer):
    # --- Literals -----------------------------------------------------------
    def int(self, items: List[Any]) -> IntLiteral:
        return IntLiteral(int(_text(items[0])))

    def float(self, items: List[Any]) -> FloatLiteral:
        return FloatLiteral(float(_text(items[0])))

    def neg_int(self, items: List[Any]) -> NegativeInt:
        return NegativeInt(-int(_text(items[0])))

    def neg_float(self, items: List[Any]) -> NegativeFloat:
        return NegativeFloat(-float(_text(items[0])))

    def string(self, items: List[Any]) -> StringLiteral:
        return StringLiteral(_unescape(_text(items[0])[1:-1]))

    def char(self, items: List[Any]) -> CharLiteral:
        return CharLiteral(_unescape(_text(items[0])[1:-1]))

    def true(self, items: List[Any]) -> BoolLiteral:
        return BoolLiteral(True)

    def false(self, items: List[Any]) -> BoolLiteral:
        return BoolLiteral(False)

    def list(self, items: List[Any]) -> ListLiteral:
        return ListLiteral([item for item in items if item is not None])

    def tuple_literal(self, items: List[Any]) -> TupleLiteral:
        return TupleLiteral(list(items))

    # --- Names --------------------------------------------------------------
    def var(self, items: List[Any]) -> Variable:
        return Variable(_text(items[0]))

    def symbolic_var(self, items: List[Any]) -> Variable:
        return Variable(_text(items[0]))

    def operator_name(self, items: List[Any]) -> str:
        return _text(items[0])

    def inst_operator_name(self, items: List[Any]) -> str:
        return _text(items[0])

    def symbolic_operator(self, items: List[Any]) -> str:
        return _text(items[0])

    def constructor(self, items: List[Any]) -> Union[Constructor, DataConstructor]:
        # Both constructor references (in expressions) and constructor
        # definitions (in data declarations) reduce through this rule.
        name = _text(items[0])
        match items[1:]:
            case []:
                return Constructor(name)
            case [RecordConstructor() as record]:
                return DataConstructor(name, record_constructor=record)
            case type_atoms:
                return DataConstructor(name, type_atoms=list(type_atoms))

    # --- Operators ----------------------------------------------------------
    def infix_op(self, items: List[Any]) -> SymbolicOperation:
        left, operator, right = items
        return SymbolicOperation(operator, [left, right])

    def unary_minus(self, items: List[Any]) -> FunctionApplication:
        return FunctionApplication(Variable(NEGATE), items[0])

    # --- Control flow -------------------------------------------------------
    def if_else(self, items: List[Any]) -> IfElse:
        return IfElse(items[0], items[1], items[2])

    def if_else_extended(self, items: List[Any]) -> IfElse:
        condition, then_expr, elsif_clauses, else_expr = items
        result = else_expr
        for elsif_cond, elsif_then in reversed(elsif_clauses):
            result = IfElse(elsif_cond, elsif_then, result)
        return IfElse(condition, then_expr, result)

    def elsif_clauses(self, items: List[Any]) -> List[tuple]:
        return [(items[i], items[i + 1]) for i in range(0, len(items), 2)]

    def do_block(self, items: List[Any]) -> DoBlock:
        return DoBlock(items[0])

    def stmt_list(self, items: List[Any]) -> List[Statement]:
        flattened: List[Statement] = []
        for item in items:
            match item:
                case list() as lets:
                    flattened.extend(lets)
                case stmt:
                    flattened.append(stmt)
        return flattened

    def do_stmt(self, items: List[Any]) -> Any:
        return items[0]

    def assignment_list(self, items: List[Any]) -> List[LetStatement]:
        return items

    def assignment(self, items: List[Any]) -> LetStatement:
        return LetStatement(_text(items[0]), items[1])

    # --- Applications and constructors --------------------------------------
    def app(self, items: List[Any]) -> FunctionApplication:
        return FunctionApplication(items[0], items[1])

    def field_assign(self, items: List[Any]) -> FieldAssignment:
        return FieldAssignment(_text(items[0]), items[1])

    def constructor_expr(self, items: List[Any]) -> ConstructorExpression:
        return ConstructorExpression(_text(items[0]), items[1:])

    def grouped(self, items: List[Any]) -> GroupedExpression:
        return GroupedExpression(items[0])

    # --- Types --------------------------------------------------------------
    def type_constructor(self, items: List[Any]) -> TypeConstructor:
        return TypeConstructor(_text(items[0]))

    def type_var(self, items: List[Any]) -> TypeVariable:
        return TypeVariable(_text(items[0]))

    def arrow_type(self, items: List[Any]) -> ArrowType:
        return ArrowType(items[0], items[1])

    def type_application(self, items: List[Any]) -> TypeApplication:
        return TypeApplication(items[0], items[1])

    def grouped_type(self, items: List[Any]) -> GroupedType:
        return GroupedType(items[0])

    def list_type(self, items: List[Any]) -> ListType:
        return ListType(items[0])

    def tuple_type(self, items: List[Any]) -> TupleType:
        return TupleType(list(items))

    def constraint(self, items: List[Any]) -> TypeConstraint:
        return TypeConstraint(items[0], items[1])

    def constraints(self, items: List[Any]) -> List[TypeConstraint]:
        return items

    def constrained_type(self, items: List[Any]) -> ConstrainedType:
        return ConstrainedType(items[0], items[1])

    def type_scheme(self, items: List[Any]) -> ConstrainedType:
        return ConstrainedType([], items[0])

    # --- Patterns -----------------------------------------------------------
    def var_pattern(self, items: List[Any]) -> VariablePattern:
        return VariablePattern(_text(items[0]))

    def literal_pattern(self, items: List[Any]) -> LiteralPattern:
        # The literal node itself is kept so that the type of the literal
        # (Int, Float, String, Char, Bool) is known to the type checker.
        return LiteralPattern(items[0])

    def neg_int_pattern(self, items: List[Any]) -> NegativeIntPattern:
        return NegativeIntPattern(-int(_text(items[0])))

    def neg_float_pattern(self, items: List[Any]) -> NegativeFloatPattern:
        return NegativeFloatPattern(-float(_text(items[0])))

    def constructor_pattern(self, items: List[Any]) -> ConstructorPattern:
        return ConstructorPattern(_text(items[0]), items[1:])

    def constructor_pattern_bare(self, items: List[Any]) -> ConstructorPattern:
        return ConstructorPattern(_text(items[0]), [])

    def cons_pattern(self, items: List[Any]) -> ConsPattern:
        head, _colon, tail = items
        return ConsPattern(head, tail)

    def tuple_pattern(self, items: List[Any]) -> TuplePattern:
        return TuplePattern(list(items))

    def list_pattern(self, items: List[Any]) -> ListPattern:
        return ListPattern([item for item in items if item is not None])

    # --- Top-level declarations ---------------------------------------------
    def type_param(self, items: List[Any]) -> TypeParameter:
        return TypeParameter(_text(items[0]))

    def field(self, items: List[Any]) -> Field:
        return Field(_text(items[0]), items[1])

    def record_constructor(self, items: List[Any]) -> RecordConstructor:
        return RecordConstructor([item for item in items if item is not None])

    def data_decl(self, items: List[Any]) -> DataDeclaration:
        type_name = _text(items[0])
        type_params: List[TypeParameter] = []
        constructors: List[DataConstructor] = []
        for item in items[1:]:
            match item:
                case TypeParameter():
                    type_params.append(item)
                case DataConstructor():
                    constructors.append(item)
                case Constructor(name=name):
                    constructors.append(DataConstructor(name, type_atoms=[]))
        return DataDeclaration(type_name, type_params, constructors)

    def _function_definition(self, items: List[Any]) -> FunctionDefinition:
        name = _text(items[0])
        patterns: List[Pattern] = items[1:-1]
        body: Expression = items[-1]
        return FunctionDefinition(name, patterns, body)

    def func_def(self, items: List[Any]) -> FunctionDefinition:
        return self._function_definition(items)

    def inst_func_def(self, items: List[Any]) -> FunctionDefinition:
        return self._function_definition(items)

    def inst_decl(self, items: List[Any]) -> InstanceDeclaration:
        name = _text(items[0])
        scheme = items[1]
        if not isinstance(scheme, ConstrainedType):
            scheme = ConstrainedType([], scheme)
        clauses: List[FunctionDefinition] = items[2:]
        for clause in clauses:
            if clause.function_name != name:
                raise ValueError(
                    f"Instance declaration for '{name}' defines '{clause.function_name}'",
                )
        return InstanceDeclaration(name, scheme, clauses)

    def _precedence(
        self,
        items: List[Any],
        associativity: Associativity,
    ) -> PrecedenceDeclaration:
        return PrecedenceDeclaration(
            int(_text(items[0])),
            associativity,
            _text(items[1]),
        )

    def infixl_decl(self, items: List[Any]) -> PrecedenceDeclaration:
        return self._precedence(items, Associativity.LEFT)

    def infixr_decl(self, items: List[Any]) -> PrecedenceDeclaration:
        return self._precedence(items, Associativity.RIGHT)

    def infix_decl(self, items: List[Any]) -> PrecedenceDeclaration:
        return self._precedence(items, Associativity.NONE)

    # --- Root ---------------------------------------------------------------
    def start(self, items: List[Any]) -> Program:
        statements: List[Statement] = []
        for item in items:
            match item:
                case None | Tree():
                    # empty prelude/postlude comment groups
                    continue
                case stmt:
                    statements.append(stmt)
        return Program(statements)


def transform_parse_tree(tree: Tree) -> Program:
    return ASTTransformer().transform(tree)
