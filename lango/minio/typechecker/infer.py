"""Hindley/Milner type inference for MiniO (substitution passing).

MiniO has no overloading: ``show``, ``putStr``, ``error`` and ``==`` are
built-in polymorphic functions, lists are ``List a`` and the arithmetic
operators work on ``Int`` or ``Float`` (defaulting to ``Int``).  Every
function name is bound to a monomorphic type variable before the functions
are checked in source order, so a function may refer to itself.
"""

from collections.abc import ItemsView, Sequence
from dataclasses import dataclass, field

from lango.shared.ast.nodes import (
    AddOperation,
    AndOperation,
    ArrowType,
    BinaryOperation,
    BoolLiteral,
    CharLiteral,
    ConcatOperation,
    ConsPattern,
    Constructor,
    ConstructorExpression,
    ConstructorPattern,
    DataDeclaration,
    DivOperation,
    DoBlock,
    EqualOperation,
    Expression,
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
    ListPattern,
    ListType,
    LiteralPattern,
    MulOperation,
    NegativeFloat,
    NegativeFloatPattern,
    NegativeInt,
    NegativeIntPattern,
    NegOperation,
    NotEqualOperation,
    NotOperation,
    OrOperation,
    Pattern,
    PowFloatOperation,
    PowIntOperation,
    Program,
    Statement,
    StringLiteral,
    SubOperation,
    TupleLiteral,
    TuplePattern,
)
from lango.shared.ast.nodes import TupleType as ASTTupleType
from lango.shared.ast.nodes import (
    TypeApplication,
    TypeConstructor,
    TypeExpression,
    TypeVariable,
    Variable,
    VariablePattern,
    is_expression,
)
from lango.shared.typechecker.errors import TypeInferenceError
from lango.shared.typechecker.lango_types import (
    BOOL_TYPE,
    CHAR_TYPE,
    FLOAT_TYPE,
    INT_TYPE,
    PRIMITIVE_TYPES,
    STRING_TYPE,
    UNIT_TYPE,
    DataType,
    FreshVarGenerator,
    FunctionType,
    TupleType,
    Type,
    TypeApp,
    TypeCon,
    TypeScheme,
    TypeSubstitution,
    TypeVar,
    function,
    generalize,
    unfold_function,
)
from lango.shared.typechecker.unify import UnificationError, unify_one

type InferenceResult = tuple[Type, TypeSubstitution]

LIST = TypeCon("List")
IO = TypeCon("IO")


def list_of(element: Type) -> Type:
    return TypeApp(LIST, element)


def mono(t: Type) -> TypeScheme:
    return TypeScheme(set(), t)


@dataclass(frozen=True)
class TypeEnvironment:
    bindings: dict[str, TypeScheme] = field(default_factory=dict)

    def lookup(self, name: str) -> TypeScheme | None:
        return self.bindings.get(name)

    def extend(self, name: str, scheme: TypeScheme) -> "TypeEnvironment":
        return TypeEnvironment({**self.bindings, name: scheme})

    def extend_many(self, bindings: dict[str, TypeScheme]) -> "TypeEnvironment":
        return TypeEnvironment({**self.bindings, **bindings})

    def apply_substitution(self, subst: TypeSubstitution) -> "TypeEnvironment":
        return TypeEnvironment(
            {name: scheme.substitute(subst) for name, scheme in self.bindings.items()},
        )

    def free_type_vars(self) -> set[str]:
        return set().union(*(scheme.free_vars() for scheme in self.bindings.values()))

    def items(self) -> ItemsView[str, TypeScheme]:
        return self.bindings.items()

    def __contains__(self, name: str) -> bool:
        return name in self.bindings

    def __getitem__(self, name: str) -> TypeScheme:
        return self.bindings[name]


def builtin_env() -> TypeEnvironment:
    a = TypeVar("a")
    return TypeEnvironment(
        {
            "putStr": mono(function(STRING_TYPE, TypeApp(IO, UNIT_TYPE))),
            "show": TypeScheme({"a"}, function(a, STRING_TYPE)),
            "error": TypeScheme({"a"}, function(STRING_TYPE, a)),
            "==": TypeScheme({"a"}, function(a, a, BOOL_TYPE)),
        },
    )


class TypeInferrer:
    def __init__(self) -> None:
        self.fresh_var_gen = FreshVarGenerator()
        # record constructor -> its field names in declaration order
        self.record_fields: dict[str, list[str]] = {}

    def fresh(self) -> TypeVar:
        return self.fresh_var_gen.fresh_var()

    @staticmethod
    def unify(t1: Type, t2: Type, what: str) -> TypeSubstitution:
        try:
            return unify_one(t1, t2)
        except UnificationError as e:
            raise TypeInferenceError(f"{what}: {e}") from e

    # --- declarations -----------------------------------------------------------

    def parse_type_expr(self, node: TypeExpression) -> Type:
        match node:
            case TypeConstructor(name=name):
                return PRIMITIVE_TYPES.get(name) or DataType(name)
            case TypeVariable(name=name):
                return TypeVar(name)
            case ArrowType(from_type=from_type, to_type=to_type):
                return FunctionType(
                    self.parse_type_expr(from_type),
                    self.parse_type_expr(to_type),
                )
            case TypeApplication(constructor=constructor, argument=argument):
                head = self.parse_type_expr(constructor)
                arg = self.parse_type_expr(argument)
                match head:
                    case DataType(name=name, type_args=args):
                        return DataType(name, (*args, arg))
                    case _:
                        return TypeApp(head, arg)
            case ListType(element_type=element_type):
                return list_of(self.parse_type_expr(element_type))
            case GroupedType(type_expr=type_expr):
                return self.parse_type_expr(type_expr)
            case ASTTupleType(element_types=element_types):
                return TupleType(tuple(self.parse_type_expr(e) for e in element_types))
        raise TypeInferenceError(f"Cannot parse type expression: {type(node).__name__}")

    def infer_data_decl(self, node: DataDeclaration) -> dict[str, TypeScheme]:
        """The type schemes of the constructors of ``data T a1 .. an = ...``."""
        params = [param.name for param in node.type_params]
        result = DataType(node.type_name, tuple(TypeVar(param) for param in params))
        schemes: dict[str, TypeScheme] = {}
        for constructor in node.constructors:
            if constructor.record_constructor is not None:
                fields = constructor.record_constructor.fields
                self.record_fields[constructor.name] = [f.name for f in fields]
                field_types = [self.parse_type_expr(f.field_type) for f in fields]
            else:
                field_types = [
                    self.parse_type_expr(atom) for atom in constructor.type_atoms or []
                ]
            schemes[constructor.name] = TypeScheme(
                set(params),
                function(*field_types, result),
            )
        return schemes

    # --- expressions ------------------------------------------------------------

    def infer_expr(self, expr: Expression, env: TypeEnvironment) -> InferenceResult:
        t, subst = self._infer_expr(expr, env)
        expr.ty = t
        return t, subst

    def _infer_expr(self, expr: Expression, env: TypeEnvironment) -> InferenceResult:
        match expr:
            case IntLiteral() | NegativeInt():
                return INT_TYPE, TypeSubstitution()
            case FloatLiteral() | NegativeFloat():
                return FLOAT_TYPE, TypeSubstitution()
            case StringLiteral():
                return STRING_TYPE, TypeSubstitution()
            case CharLiteral():
                return CHAR_TYPE, TypeSubstitution()
            case BoolLiteral():
                return BOOL_TYPE, TypeSubstitution()
            case ListLiteral(elements=elements):
                element = self.fresh()
                subst = TypeSubstitution()
                for e in elements:
                    t, subst = self.infer_next(e, env, subst)
                    subst = self.unify(
                        element.apply_substitution(subst),
                        t,
                        "List elements have incompatible types",
                    ).compose(subst)
                return list_of(element.apply_substitution(subst)), subst
            case TupleLiteral(elements=elements):
                types: list[Type] = []
                subst = TypeSubstitution()
                for e in elements:
                    t, subst = self.infer_next(e, env, subst)
                    types.append(t)
                return (
                    TupleType(tuple(t.apply_substitution(subst) for t in types)),
                    subst,
                )
            case Variable(name=name) | Constructor(name=name):
                scheme = env.lookup(name)
                if scheme is None:
                    raise TypeInferenceError(f"Unknown variable: {name}")
                return scheme.instantiate(self.fresh_var_gen), TypeSubstitution()
            case (
                AddOperation()
                | SubOperation()
                | MulOperation()
                | DivOperation()
                | PowIntOperation()
                | PowFloatOperation()
            ):
                return self.infer_numeric(expr, env)
            case (
                EqualOperation()
                | NotEqualOperation()
                | LessThanOperation()
                | LessEqualOperation()
                | GreaterThanOperation()
                | GreaterEqualOperation()
            ):
                left, right, subst = self.infer_operands(expr, env)
                subst = self.unify(
                    left, right, "Comparison requires operands of same type"
                ).compose(subst)
                return BOOL_TYPE, subst
            case AndOperation() | OrOperation():
                left, right, subst = self.infer_operands(expr, env)
                for operand in (left, right):
                    subst = self.unify(
                        operand.apply_substitution(subst),
                        BOOL_TYPE,
                        "Logical operation requires Bool operands",
                    ).compose(subst)
                return BOOL_TYPE, subst
            case ConcatOperation():
                left, right, subst = self.infer_operands(expr, env)
                subst = self.unify(
                    left, right, "Concatenation operands must have same type"
                ).compose(subst)
                return left.apply_substitution(subst), subst
            case NotOperation(operand=operand):
                t, subst = self.infer_expr(operand, env)
                subst = self.unify(
                    t, BOOL_TYPE, "'not' requires a Bool operand"
                ).compose(subst)
                return BOOL_TYPE, subst
            case NegOperation(operand=operand):
                t, subst = self.infer_expr(operand, env)
                return self.numeric_type(
                    t, subst, "Negation requires a numeric operand"
                )
            case IndexOperation(list_expr=list_expr, index_expr=index_expr):
                list_type, subst = self.infer_expr(list_expr, env)
                index_type, subst = self.infer_next(index_expr, env, subst)
                subst = self.unify(
                    index_type, INT_TYPE, "List index must be Int"
                ).compose(subst)
                element = self.fresh()
                subst = self.unify(
                    list_type.apply_substitution(subst),
                    list_of(element),
                    "Index operation requires a List",
                ).compose(subst)
                return element.apply_substitution(subst), subst
            case IfElse(condition=condition, then_expr=then_expr, else_expr=else_expr):
                condition_type, subst = self.infer_expr(condition, env)
                subst = self.unify(
                    condition_type, BOOL_TYPE, "If condition must be Bool"
                ).compose(subst)
                then_type, subst = self.infer_next(then_expr, env, subst)
                else_type, subst = self.infer_next(else_expr, env, subst)
                subst = self.unify(
                    then_type.apply_substitution(subst),
                    else_type,
                    "If branches have incompatible types",
                ).compose(subst)
                return then_type.apply_substitution(subst), subst
            case FunctionApplication(function=f, argument=argument):
                function_type, subst = self.infer_expr(f, env)
                argument_type, subst = self.infer_next(argument, env, subst)
                result = self.fresh()
                subst = self.unify(
                    function_type.apply_substitution(subst),
                    FunctionType(argument_type, result),
                    "Function application type mismatch",
                ).compose(subst)
                return result.apply_substitution(subst), subst
            case GroupedExpression(expression=inner):
                return self.infer_expr(inner, env)
            case DoBlock(statements=statements):
                return self.infer_block(statements, env)
            case ConstructorExpression(constructor_name=name, fields=fields):
                return self.infer_record(name, fields, env)
        raise TypeInferenceError(f"Unhandled expression type: {type(expr).__name__}")

    def infer_next(
        self,
        expr: Expression,
        env: TypeEnvironment,
        subst: TypeSubstitution,
    ) -> InferenceResult:
        """Infer ``expr`` after the substitution ``subst`` has been found."""
        t, more = self.infer_expr(expr, env.apply_substitution(subst))
        return t, more.compose(subst)

    def infer_operands(
        self,
        expr: BinaryOperation,
        env: TypeEnvironment,
    ) -> tuple[Type, Type, TypeSubstitution]:
        left, subst = self.infer_expr(expr.left, env)
        right, subst = self.infer_next(expr.right, env, subst)
        return left.apply_substitution(subst), right.apply_substitution(subst), subst

    def infer_numeric(
        self, expr: BinaryOperation, env: TypeEnvironment
    ) -> InferenceResult:
        left, right, subst = self.infer_operands(expr, env)
        subst = self.unify(
            left,
            right,
            "Binary numeric operation requires operands of same type",
        ).compose(subst)
        return self.numeric_type(
            left.apply_substitution(subst),
            subst,
            "Numeric operation requires Int or Float",
        )

    def numeric_type(
        self,
        t: Type,
        subst: TypeSubstitution,
        what: str,
    ) -> InferenceResult:
        """``t`` as a numeric type: ``Int`` or ``Float``, defaulting to ``Int``."""
        if t in (INT_TYPE, FLOAT_TYPE):
            return t, subst
        for candidate in (INT_TYPE, FLOAT_TYPE):
            try:
                return candidate, unify_one(t, candidate).compose(subst)
            except UnificationError:
                continue
        raise TypeInferenceError(f"{what}, got {t}")

    def infer_block(
        self,
        statements: Sequence[Statement],
        env: TypeEnvironment,
    ) -> InferenceResult:
        result: Type = UNIT_TYPE
        subst = TypeSubstitution()
        for stmt in statements:
            match stmt:
                case LetStatement(variable=name, value=value):
                    t, subst = self.infer_next(value, env, subst)
                    scheme = generalize(
                        env.apply_substitution(subst).free_type_vars(),
                        t.apply_substitution(subst),
                    )
                    env = env.extend(name, scheme)
                    result = UNIT_TYPE
                case _:
                    assert is_expression(stmt)
                    result, subst = self.infer_next(stmt, env, subst)
        return result.apply_substitution(subst), subst

    def infer_record(
        self,
        name: str,
        fields: Sequence,
        env: TypeEnvironment,
    ) -> InferenceResult:
        """``C { f_1 = e_1, ... }``: every declared field exactly once."""
        scheme = env.lookup(name)
        field_names = self.record_fields.get(name)
        if scheme is None or field_names is None:
            raise TypeInferenceError(f"Unknown record constructor: {name}")
        params, result = unfold_function(scheme.instantiate(self.fresh_var_gen))
        declared = dict(zip(field_names, params))
        given = [f.field_name for f in fields]
        if sorted(given) != sorted(field_names):
            raise TypeInferenceError(
                f"Constructor {name} expects fields {field_names}, got {given}",
            )
        subst = TypeSubstitution()
        for f in fields:
            t, subst = self.infer_next(f.value, env, subst)
            subst = self.unify(
                t,
                declared[f.field_name].apply_substitution(subst),
                f"Field {f.field_name} of {name} has the wrong type",
            ).compose(subst)
        return result.apply_substitution(subst), subst

    # --- patterns ---------------------------------------------------------------

    def infer_pattern(
        self,
        pattern: Pattern,
        pattern_type: Type,
        env: TypeEnvironment,
    ) -> tuple[TypeEnvironment, TypeSubstitution]:
        """Bind the variables of ``pattern``, which matches values of ``pattern_type``."""
        match pattern:
            case VariablePattern(name=name):
                return env.extend(name, mono(pattern_type)), TypeSubstitution()
            case LiteralPattern(value=value):
                return env, self.unify(
                    pattern_type,
                    self.literal_type(value),
                    "Literal pattern has the wrong type",
                )
            case NegativeIntPattern():
                return env, self.unify(
                    pattern_type, INT_TYPE, "Pattern has the wrong type"
                )
            case NegativeFloatPattern():
                return env, self.unify(
                    pattern_type, FLOAT_TYPE, "Pattern has the wrong type"
                )
            case ConstructorPattern(constructor=name, patterns=subpatterns):
                scheme = env.lookup(name)
                if scheme is None:
                    raise TypeInferenceError(f"Unknown constructor in pattern: {name}")
                params, result = unfold_function(scheme.instantiate(self.fresh_var_gen))
                if len(params) != len(subpatterns):
                    raise TypeInferenceError(
                        f"Constructor {name} expects {len(params)} arguments, "
                        f"got {len(subpatterns)}",
                    )
                subst = self.unify(
                    pattern_type, result, "Constructor pattern type mismatch"
                )
                return self.infer_subpatterns(subpatterns, params, env, subst)
            case ConsPattern(head=head, tail=tail):
                element = self.fresh()
                list_type = list_of(element)
                subst = self.unify(
                    pattern_type, list_type, "Cons pattern requires a List"
                )
                return self.infer_subpatterns(
                    [head, tail], [element, list_type], env, subst
                )
            case ListPattern(patterns=subpatterns):
                element = self.fresh()
                subst = self.unify(
                    pattern_type, list_of(element), "List pattern requires a List"
                )
                return self.infer_subpatterns(
                    subpatterns,
                    [element] * len(subpatterns),
                    env,
                    subst,
                )
            case TuplePattern(patterns=subpatterns):
                elements: list[Type] = [self.fresh() for _ in subpatterns]
                env, subst = self.infer_subpatterns(
                    subpatterns,
                    elements,
                    env,
                    TypeSubstitution(),
                )
                tuple_type = TupleType(
                    tuple(e.apply_substitution(subst) for e in elements)
                )
                subst = self.unify(
                    pattern_type, tuple_type, "Tuple pattern type mismatch"
                ).compose(subst)
                return env, subst
        raise TypeInferenceError(f"Unhandled pattern: {type(pattern).__name__}")

    def infer_subpatterns(
        self,
        patterns: Sequence[Pattern],
        types: Sequence[Type],
        env: TypeEnvironment,
        subst: TypeSubstitution,
    ) -> tuple[TypeEnvironment, TypeSubstitution]:
        for pattern, t in zip(patterns, types):
            env, more = self.infer_pattern(
                pattern,
                t.apply_substitution(subst),
                env.apply_substitution(subst),
            )
            subst = more.compose(subst)
        return env, subst

    @staticmethod
    def literal_type(value: object) -> Type:
        match value:
            case bool():  # before int: bool is a subclass of int
                return BOOL_TYPE
            case int():
                return INT_TYPE
            case float():
                return FLOAT_TYPE
            case str():
                return STRING_TYPE
        raise TypeInferenceError(f"Unsupported literal pattern: {value!r}")

    # --- functions and programs -------------------------------------------------

    def infer_function_group(
        self,
        clauses: Sequence[FunctionDefinition],
        env: TypeEnvironment,
        outer: TypeEnvironment,
    ) -> TypeScheme:
        """The clauses ``f p_1 ... p_n = e`` of one function share one type.

        ``env`` binds the functions of the program monomorphically (for the
        recursive uses); the result is generalised with respect to ``outer``,
        the environment without those provisional bindings."""
        name = clauses[0].function_name
        arity = len(clauses[0].patterns)
        if any(len(clause.patterns) != arity for clause in clauses):
            raise TypeInferenceError(
                f"Function {name} has clauses with different arities"
            )
        param_types = [self.fresh() for _ in range(arity)]
        result_type = self.fresh()
        subst = TypeSubstitution()
        for clause in clauses:
            clause_env, subst = self.infer_subpatterns(
                clause.patterns,
                param_types,
                env,
                subst,
            )
            body_type, subst = self.infer_next(clause.body, clause_env, subst)
            subst = self.unify(
                result_type.apply_substitution(subst),
                body_type,
                f"Function {name} clauses have incompatible return types",
            ).compose(subst)
        function_type = function(*param_types, result_type).apply_substitution(subst)
        for clause in clauses:
            clause.ty = function_type
        return generalize(
            outer.apply_substitution(subst).free_type_vars(), function_type
        )

    def infer_program(self, ast: Program) -> TypeEnvironment:
        env = builtin_env()
        groups: dict[str, list[FunctionDefinition]] = {}
        for stmt in ast.statements:
            match stmt:
                case DataDeclaration():
                    env = env.extend_many(self.infer_data_decl(stmt))
                case FunctionDefinition(function_name=name):
                    groups.setdefault(name, []).append(stmt)
        # every function is bound beforehand (monomorphically), so bodies may
        # refer to functions defined later or to themselves
        provisional = {name: mono(self.fresh()) for name in groups}
        for name, clauses in groups.items():
            try:
                scheme = self.infer_function_group(
                    clauses,
                    env.extend_many(provisional),
                    env,
                )
            except TypeInferenceError as e:
                raise TypeInferenceError(
                    f"Failed to infer type for function {name}: {e}",
                ) from e
            env = env.extend(name, scheme)
            del provisional[name]
        return env


def type_check_ast(ast: Program) -> TypeEnvironment:
    return TypeInferrer().infer_program(ast)
