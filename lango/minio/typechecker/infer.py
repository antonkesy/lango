"""
AST-based type inference engine using the Hindley-Milner algorithm.
"""

from typing import Dict, ItemsView, List, Optional, Set, Tuple

from lango.minio.ast.nodes import (
    AddOperation,
    AndOperation,
    ArrowType,
    ASTNode,
    BoolLiteral,
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
    LiteralPattern,
    MulOperation,
    NegativeFloat,
    NegativeInt,
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
    TypeApplication,
    TypeConstructor,
    TypeExpression,
    TypeVariable,
    Variable,
    VariablePattern,
)
from lango.minio.typechecker.minio_types import (
    BOOL_TYPE,
    FLOAT_TYPE,
    INT_TYPE,
    STRING_TYPE,
    UNIT_TYPE,
    DataType,
    FreshVarGenerator,
    FunctionType,
    Type,
    TypeApp,
    TypeCon,
    TypeScheme,
    TypeSubstitution,
    TypeVar,
    generalize,
)
from lango.minio.typechecker.unify import UnificationError, unify_one

TypeBindings = Dict[str, TypeScheme]
InferenceResult = Tuple[Type, TypeSubstitution]


class TypeEnvironment:
    """Type environment mapping identifiers to type schemes."""

    def __init__(self, bindings: Optional[TypeBindings] = None) -> None:
        """Initialize the type environment with optional bindings."""
        self.bindings: TypeBindings = bindings or {}

    def lookup(self, name: str) -> Optional[TypeScheme]:
        """Look up a name in the environment and return its type scheme."""
        return self.bindings.get(name)

    def extend(self, name: str, scheme: TypeScheme) -> "TypeEnvironment":
        """Return a new environment with an additional binding."""
        new_bindings = self.bindings.copy()
        new_bindings[name] = scheme
        return TypeEnvironment(new_bindings)

    def extend_many(self, new_bindings: Dict[str, TypeScheme]) -> "TypeEnvironment":
        """Return a new environment with multiple additional bindings."""
        combined_bindings = self.bindings.copy()
        combined_bindings.update(new_bindings)
        return TypeEnvironment(combined_bindings)

    def apply_substitution(self, subst: TypeSubstitution) -> "TypeEnvironment":
        """Apply a type substitution to all bindings in the environment."""
        new_bindings = {}
        for name, scheme in self.bindings.items():
            # Apply substitution to the scheme's type, keeping quantified vars
            applied_type = subst.apply(scheme.type)
            new_bindings[name] = TypeScheme(scheme.quantified_vars, applied_type)
        return TypeEnvironment(new_bindings)

    def free_type_vars(self) -> Set[str]:
        """Return all free type variables in this environment."""
        free_vars = set()
        for scheme in self.bindings.values():
            free_vars.update(scheme.free_vars())
        return free_vars

    def items(self) -> ItemsView[str, TypeScheme]:
        """Return items for iteration."""
        return self.bindings.items()

    def __contains__(self, name: str) -> bool:
        """Check if a name is in the environment."""
        return name in self.bindings

    def __getitem__(self, name: str) -> TypeScheme:
        """Get a type scheme by name."""
        return self.bindings[name]


class TypeInferenceError(Exception):
    """Exception raised during type inference."""

    def __init__(self, message: str, node: Optional[ASTNode] = None) -> None:
        self.message = message
        self.node = node
        super().__init__(message)


class TypeInferrer:
    """AST-based Hindley-Milner type inference engine."""

    def __init__(self) -> None:
        """Initialize the type inferrer."""
        self.fresh_var_gen = FreshVarGenerator()
        self.data_types: Dict[str, List[str]] = {}  # type_name -> [constructor_names]
        self.data_constructors: Dict[str, Tuple[str, List[Type]]] = (
            {}
        )  # constructor -> (type_name, field_types)

    def fresh_type_var(self) -> TypeVar:
        """Generate a fresh type variable."""
        var_name = self.fresh_var_gen.fresh()
        return TypeVar(var_name)

    def infer_data_decl(self, node: DataDeclaration) -> TypeEnvironment:
        """Process a data type declaration and return updated environment."""
        type_name = node.type_name
        type_params = [param.name for param in node.type_params]

        constructor_names = []
        constructor_types: Dict[str, TypeScheme] = {}

        for constructor in node.constructors:
            ctor_name = constructor.name
            constructor_names.append(ctor_name)

            # Create type parameter variables
            type_param_vars: List[Type] = [TypeVar(param) for param in type_params]
            result_type = DataType(type_name, type_param_vars)

            if constructor.record_constructor:
                # Record constructor
                field_types = []
                for field in constructor.record_constructor.fields:
                    field_type = self.parse_type_expr(field.field_type)
                    field_types.append(field_type)

                # For record constructors, create a function that takes all fields
                ctor_type: Type = result_type
                for field_type in reversed(field_types):
                    ctor_type = FunctionType(field_type, ctor_type)

                # Generalize over the type parameters
                bound_vars = set(type_params)
                ctor_scheme = TypeScheme(bound_vars, ctor_type)
                constructor_types[ctor_name] = ctor_scheme
                self.data_constructors[ctor_name] = (type_name, field_types)

            else:
                # Positional constructor
                if not constructor.type_atoms:
                    # Nullary constructor
                    bound_vars = set(type_params)
                    ctor_scheme = TypeScheme(bound_vars, result_type)
                    constructor_types[ctor_name] = ctor_scheme
                else:
                    # Constructor with arguments
                    field_types = []
                    for type_expr in constructor.type_atoms:
                        field_type = self.parse_type_expr(type_expr)
                        field_types.append(field_type)

                    # Create function type: field1 -> field2 -> ... -> DataType
                    positional_ctor_type: Type = result_type
                    for field_type in reversed(field_types):
                        positional_ctor_type = FunctionType(
                            field_type,
                            positional_ctor_type,
                        )

                    # Generalize over the type parameters
                    bound_vars = set(type_params)
                    ctor_scheme = TypeScheme(bound_vars, positional_ctor_type)
                    constructor_types[ctor_name] = ctor_scheme

                self.data_constructors[ctor_name] = (type_name, [])

        self.data_types[type_name] = constructor_names

        # Return environment extended with constructor types
        env = TypeEnvironment()
        for name, scheme in constructor_types.items():
            env = env.extend(name, scheme)

        return env

    def parse_type_expr(self, node: TypeExpression) -> Type:
        """Parse a type expression from the AST."""
        match node:
            case TypeConstructor(name=type_name):
                match type_name:
                    case "Int":
                        return INT_TYPE
                    case "String":
                        return STRING_TYPE
                    case "Float":
                        return FLOAT_TYPE
                    case "Bool":
                        return BOOL_TYPE
                    case _:
                        return DataType(type_name, [])
            case TypeVariable(name=name):
                return TypeVar(name)
            case ArrowType(from_type=from_type, to_type=to_type):
                from_type_parsed = self.parse_type_expr(from_type)
                to_type_parsed = self.parse_type_expr(to_type)
                return FunctionType(from_type_parsed, to_type_parsed)
            case TypeApplication(constructor=constructor, argument=argument):
                constructor_type = self.parse_type_expr(constructor)
                argument_type = self.parse_type_expr(argument)

                match constructor_type:
                    case DataType(name=name, type_args=type_args):
                        # Apply type argument to data type
                        new_args = type_args + [argument_type]
                        return DataType(name, new_args)
                    case _:
                        return TypeApp(constructor_type, argument_type)
            case GroupedType(type_expr=type_expr):
                return self.parse_type_expr(type_expr)
            case _:
                raise TypeInferenceError(
                    f"Cannot parse type expression: {type(node).__name__}",
                )

    def infer_expr(self, expr: Expression, env: TypeEnvironment) -> InferenceResult:
        """Infer the type of an expression."""
        result = self._infer_expr_internal(expr, env)
        # Set type on AST node if possible
        try:
            expr.ty = result[0]  # type: ignore
        except AttributeError:
            pass  # Some AST nodes don't have ty attribute
        return result

    def _infer_expr_internal(
        self,
        expr: Expression,
        env: TypeEnvironment,
    ) -> InferenceResult:
        """Internal expression inference without side effects."""
        match expr:
            # Literals
            case IntLiteral() | NegativeInt():
                return INT_TYPE, TypeSubstitution()
            case FloatLiteral() | NegativeFloat():
                return FLOAT_TYPE, TypeSubstitution()
            case StringLiteral():
                return STRING_TYPE, TypeSubstitution()
            case BoolLiteral():
                return BOOL_TYPE, TypeSubstitution()
            case ListLiteral(elements=elements):
                return self._infer_list_literal(elements, env)

            # Variables and constructors
            case Variable(name=name) | Constructor(name=name):
                return self._infer_identifier(name, env)

            # Binary operations
            case (
                AddOperation()
                | SubOperation()
                | MulOperation()
                | DivOperation()
                | PowIntOperation()
                | PowFloatOperation()
            ) as op:
                return self._infer_binary_op(op.left, op.right, env, "numeric")

            case (
                EqualOperation()
                | NotEqualOperation()
                | LessThanOperation()
                | LessEqualOperation()
                | GreaterThanOperation()
                | GreaterEqualOperation()
            ) as op:
                return self._infer_binary_op(op.left, op.right, env, "comparison")

            case (AndOperation() | OrOperation()) as op:
                return self._infer_binary_op(op.left, op.right, env, "logical")

            # Unary operations
            case NotOperation(operand=operand):
                return self._infer_unary_op(
                    operand,
                    env,
                    BOOL_TYPE,
                    "NOT operation requires Bool operand",
                )

            # Special operations
            case ConcatOperation(left=left, right=right):
                return self._infer_binary_op(left, right, env, "concat")
            case IndexOperation(list_expr=list_expr, index_expr=index_expr):
                return self._infer_index_operation(list_expr, index_expr, env)

            # Control flow
            case IfElse(condition=cond, then_expr=then_expr, else_expr=else_expr):
                return self._infer_if_else(cond, then_expr, else_expr, env)

            # Function application
            case FunctionApplication(function=func_expr, argument=arg_expr):
                return self._infer_function_application(func_expr, arg_expr, env)

            # Grouping and blocks
            case GroupedExpression(expression=inner_expr):
                return self._infer_expr_internal(inner_expr, env)
            case DoBlock(statements=stmts):
                return self.infer_do_block(stmts, env)
            case ConstructorExpression() as ctor_expr:
                return self.infer_constructor_expr(ctor_expr, env)

            case _:
                raise TypeInferenceError(
                    f"Unhandled expression type: {type(expr).__name__}",
                )

    def _infer_list_literal(
        self,
        elements: List[Expression],
        env: TypeEnvironment,
    ) -> InferenceResult:
        """Infer type of list literal."""
        if not elements:
            element_type = self.fresh_type_var()
            return TypeApp(TypeCon("List"), element_type), TypeSubstitution()

        # Infer first element type and unify with rest
        first_type, subst = self.infer_expr(elements[0], env)
        current_subst = subst

        for element in elements[1:]:
            elem_type, elem_subst = self.infer_expr(
                element,
                env.apply_substitution(current_subst),
            )
            current_subst = current_subst.compose(elem_subst)

            try:
                unify_subst = unify_one(
                    first_type.apply_substitution(current_subst),
                    elem_type,
                )
                current_subst = current_subst.compose(unify_subst)
                first_type = first_type.apply_substitution(unify_subst)
            except UnificationError as e:
                raise TypeInferenceError(f"List elements have incompatible types: {e}")

        list_type = TypeApp(
            TypeCon("List"),
            first_type.apply_substitution(current_subst),
        )
        return list_type, current_subst

    def _infer_identifier(self, name: str, env: TypeEnvironment) -> InferenceResult:
        """Infer type of variable or constructor."""
        scheme = env.lookup(name)
        if scheme is None:
            raise TypeInferenceError(f"Unknown identifier: {name}")
        return scheme.instantiate(self.fresh_var_gen), TypeSubstitution()

    def _infer_binary_op(
        self,
        left: Expression,
        right: Expression,
        env: TypeEnvironment,
        op_type: str,
    ) -> InferenceResult:
        """Unified binary operation inference."""
        left_type, left_subst = self.infer_expr(left, env)
        right_type, right_subst = self.infer_expr(
            right,
            env.apply_substitution(left_subst),
        )
        combined_subst = left_subst.compose(right_subst)

        match op_type:
            case "numeric":
                return self._infer_numeric_op(left_type, right_type, combined_subst)
            case "comparison":
                return self._infer_comparison_op(left_type, right_type, combined_subst)
            case "logical":
                return self._infer_logical_op(left_type, right_type, combined_subst)
            case "concat":
                return self._infer_concat_op(left_type, right_type, combined_subst)
            case _:
                raise TypeInferenceError(f"Unknown binary operation type: {op_type}")

    def _infer_numeric_op(
        self,
        left_type: Type,
        right_type: Type,
        subst: TypeSubstitution,
    ) -> InferenceResult:
        """Infer numeric binary operation."""
        try:
            unify_subst = unify_one(
                left_type.apply_substitution(subst),
                right_type.apply_substitution(subst),
            )
            final_subst = subst.compose(unify_subst)
            unified_type = left_type.apply_substitution(final_subst)

            if unified_type in (INT_TYPE, FLOAT_TYPE):
                return unified_type, final_subst

            # Try Int, then Float
            for target_type in (INT_TYPE, FLOAT_TYPE):
                try:
                    type_unify = unify_one(unified_type, target_type)
                    return target_type, final_subst.compose(type_unify)
                except UnificationError:
                    continue

            raise TypeInferenceError(
                f"Numeric operation requires Int or Float, got {unified_type}",
            )
        except UnificationError:
            raise TypeInferenceError(
                "Binary numeric operation requires operands of same type",
            )

    def _infer_comparison_op(
        self,
        left_type: Type,
        right_type: Type,
        subst: TypeSubstitution,
    ) -> InferenceResult:
        """Infer comparison binary operation."""
        try:
            unify_subst = unify_one(
                left_type.apply_substitution(subst),
                right_type.apply_substitution(subst),
            )
            return BOOL_TYPE, subst.compose(unify_subst)
        except UnificationError:
            raise TypeInferenceError("Comparison requires operands of same type")

    def _infer_logical_op(
        self,
        left_type: Type,
        right_type: Type,
        subst: TypeSubstitution,
    ) -> InferenceResult:
        """Infer logical binary operation."""
        try:
            left_bool = unify_one(left_type, BOOL_TYPE)
            right_bool = unify_one(right_type, BOOL_TYPE)
            return BOOL_TYPE, subst.compose(left_bool).compose(right_bool)
        except UnificationError:
            raise TypeInferenceError("Logical operation requires Bool operands")

    def _infer_concat_op(
        self,
        left_type: Type,
        right_type: Type,
        subst: TypeSubstitution,
    ) -> InferenceResult:
        """Infer concatenation operation."""
        try:
            unify_subst = unify_one(
                left_type.apply_substitution(subst),
                right_type.apply_substitution(subst),
            )
            final_subst = subst.compose(unify_subst)
            return left_type.apply_substitution(final_subst), final_subst
        except UnificationError:
            raise TypeInferenceError("Concatenation operands must have same type")

    def _infer_unary_op(
        self,
        operand: Expression,
        env: TypeEnvironment,
        expected_type: Type,
        error_msg: str,
    ) -> InferenceResult:
        """Infer unary operation."""
        operand_type, subst = self.infer_expr(operand, env)
        try:
            unify_subst = unify_one(operand_type, expected_type)
            return expected_type, subst.compose(unify_subst)
        except UnificationError:
            raise TypeInferenceError(f"{error_msg}, got {operand_type}")

    def _infer_index_operation(
        self,
        list_expr: Expression,
        index_expr: Expression,
        env: TypeEnvironment,
    ) -> InferenceResult:
        """Infer list indexing operation."""
        list_type, list_subst = self.infer_expr(list_expr, env)
        index_type, index_subst = self.infer_expr(
            index_expr,
            env.apply_substitution(list_subst),
        )
        combined_subst = list_subst.compose(index_subst)

        # Index must be Int
        try:
            int_unify = unify_one(index_type, INT_TYPE)
            subst_with_int = combined_subst.compose(int_unify)
        except UnificationError:
            raise TypeInferenceError(f"List index must be Int, got {index_type}")

        # List must be List[T]
        element_type = self.fresh_type_var()
        expected_list_type = TypeApp(TypeCon("List"), element_type)

        try:
            list_unify = unify_one(
                list_type.apply_substitution(subst_with_int),
                expected_list_type,
            )
            final_subst = subst_with_int.compose(list_unify)
            return element_type.apply_substitution(final_subst), final_subst
        except UnificationError:
            raise TypeInferenceError(
                f"Index operation requires a List, got {list_type}",
            )

    def _infer_if_else(
        self,
        cond: Expression,
        then_expr: Expression,
        else_expr: Expression,
        env: TypeEnvironment,
    ) -> InferenceResult:
        """Infer if-else expression."""
        cond_type, cond_subst = self.infer_expr(cond, env)

        # Condition must be Bool
        try:
            bool_unify = unify_one(cond_type, BOOL_TYPE)
            subst_after_cond = cond_subst.compose(bool_unify)
        except UnificationError:
            raise TypeInferenceError(f"If condition must be Bool, got {cond_type}")

        # Infer branches
        then_type, then_subst = self.infer_expr(
            then_expr,
            env.apply_substitution(subst_after_cond),
        )
        subst_after_then = subst_after_cond.compose(then_subst)

        else_type, else_subst = self.infer_expr(
            else_expr,
            env.apply_substitution(subst_after_then),
        )
        subst_after_else = subst_after_then.compose(else_subst)

        # Branches must have same type
        try:
            branch_unify = unify_one(
                then_type.apply_substitution(subst_after_else),
                else_type,
            )
            final_subst = subst_after_else.compose(branch_unify)
            return then_type.apply_substitution(final_subst), final_subst
        except UnificationError:
            raise TypeInferenceError(
                f"If branches have incompatible types: {then_type} vs {else_type}",
            )

    def _infer_function_application(
        self,
        func_expr: Expression,
        arg_expr: Expression,
        env: TypeEnvironment,
    ) -> InferenceResult:
        """Infer function application."""
        func_type, func_subst = self.infer_expr(func_expr, env)
        arg_type, arg_subst = self.infer_expr(
            arg_expr,
            env.apply_substitution(func_subst),
        )
        combined_subst = func_subst.compose(arg_subst)

        # Create fresh return type
        return_type = self.fresh_type_var()
        expected_func_type = FunctionType(arg_type, return_type)

        try:
            func_unify = unify_one(
                func_type.apply_substitution(combined_subst),
                expected_func_type,
            )
            final_subst = combined_subst.compose(func_unify)
            return return_type.apply_substitution(final_subst), final_subst
        except UnificationError:
            raise TypeInferenceError("Function application type mismatch")

    def infer_function(
        self,
        func_def: FunctionDefinition,
        env: TypeEnvironment,
    ) -> Tuple[TypeScheme, TypeEnvironment]:
        """Infer the type of a function definition."""
        # For now, handle simple functions without pattern matching
        if len(func_def.patterns) == 0:
            # Nullary function
            body_type, body_subst = self.infer_expr(func_def.body, env)
            final_type = body_type.apply_substitution(body_subst)
            scheme = generalize(
                env.apply_substitution(body_subst).free_type_vars(),
                final_type,
            )
            # Set the AST node's type
            func_def.ty = final_type
            return scheme, env.extend(func_def.function_name, scheme)

        # Function with parameters - create function type
        param_types = [self.fresh_type_var() for _ in func_def.patterns]

        # Extend environment with pattern bindings
        extended_env = env
        current_subst = TypeSubstitution()

        for pattern, param_type in zip(func_def.patterns, param_types):
            pattern_env, pattern_subst = self.infer_pattern(
                pattern,
                param_type.apply_substitution(current_subst),
                extended_env.apply_substitution(current_subst),
            )
            extended_env = pattern_env
            current_subst = current_subst.compose(pattern_subst)

        # Infer body type
        body_type, body_subst = self.infer_expr(
            func_def.body,
            extended_env.apply_substitution(current_subst),
        )
        final_subst = current_subst.compose(body_subst)

        # Create function type
        func_type = body_type.apply_substitution(final_subst)
        for param_type in reversed(param_types):
            func_type = FunctionType(
                param_type.apply_substitution(final_subst),
                func_type,
            )

        # Generalize
        scheme = generalize(
            env.apply_substitution(final_subst).free_type_vars(),
            func_type,
        )
        # Set the AST node's type
        func_def.ty = func_type
        return scheme, env.extend(func_def.function_name, scheme)

    def infer_function_group(
        self,
        func_defs: List[FunctionDefinition],
        env: TypeEnvironment,
    ) -> Tuple[TypeScheme, TypeEnvironment]:
        """Infer the type of multiple function definitions with the same name."""
        if not func_defs:
            raise TypeInferenceError("Empty function group")

        function_name = func_defs[0].function_name

        # All function clauses must have the same arity
        first_arity = len(func_defs[0].patterns)
        for func_def in func_defs[1:]:
            if len(func_def.patterns) != first_arity:
                raise TypeInferenceError(
                    f"Function {function_name} has clauses with different arities",
                )

        if first_arity == 0:
            # Nullary functions - all clauses should return the same type
            clause_types = []
            final_subst = TypeSubstitution()

            for func_def in func_defs:
                body_type, body_subst = self.infer_expr(
                    func_def.body,
                    env.apply_substitution(final_subst),
                )
                final_subst = final_subst.compose(body_subst)
                clause_types.append(body_type.apply_substitution(final_subst))

            # Unify all clause return types
            unified_type = clause_types[0]
            for clause_type in clause_types[1:]:
                try:
                    unify_subst = unify_one(
                        unified_type.apply_substitution(final_subst),
                        clause_type,
                    )
                    final_subst = final_subst.compose(unify_subst)
                    unified_type = unified_type.apply_substitution(unify_subst)
                except UnificationError as e:
                    raise TypeInferenceError(
                        f"Function {function_name} clauses have incompatible return types: {e}",
                    )

            final_type = unified_type.apply_substitution(final_subst)
            scheme = generalize(
                env.apply_substitution(final_subst).free_type_vars(),
                final_type,
            )

            # Set types on all function definitions
            for func_def in func_defs:
                func_def.ty = final_type

            return scheme, env.extend(function_name, scheme)

        # Functions with parameters - create shared parameter types
        param_types = [self.fresh_type_var() for _ in range(first_arity)]
        clause_return_types = []
        final_subst = TypeSubstitution()

        for func_def in func_defs:
            # Extend environment with pattern bindings for this clause
            clause_env = env
            clause_subst = final_subst

            for pattern, param_type in zip(func_def.patterns, param_types):
                pattern_env, pattern_subst = self.infer_pattern(
                    pattern,
                    param_type.apply_substitution(clause_subst),
                    clause_env.apply_substitution(clause_subst),
                )
                clause_env = pattern_env
                clause_subst = clause_subst.compose(pattern_subst)

            # Infer body type for this clause
            body_type, body_subst = self.infer_expr(
                func_def.body,
                clause_env.apply_substitution(clause_subst),
            )
            clause_subst = clause_subst.compose(body_subst)
            final_subst = final_subst.compose(clause_subst)

            clause_return_types.append(body_type.apply_substitution(final_subst))

        # Unify all clause return types
        unified_return_type = clause_return_types[0]
        for clause_return_type in clause_return_types[1:]:
            try:
                unify_subst = unify_one(
                    unified_return_type.apply_substitution(final_subst),
                    clause_return_type.apply_substitution(final_subst),
                )
                final_subst = final_subst.compose(unify_subst)
                unified_return_type = unified_return_type.apply_substitution(
                    unify_subst,
                )
            except UnificationError as e:
                raise TypeInferenceError(
                    f"Function {function_name} clauses have incompatible return types: {e}",
                )

        # Create function type
        func_type = unified_return_type.apply_substitution(final_subst)
        for param_type in reversed(param_types):
            func_type = FunctionType(
                param_type.apply_substitution(final_subst),
                func_type,
            )

        # Generalize
        scheme = generalize(
            env.apply_substitution(final_subst).free_type_vars(),
            func_type,
        )

        # Set types on all function definitions
        for func_def in func_defs:
            func_def.ty = func_type

        return scheme, env.extend(function_name, scheme)

    def infer_do_block(
        self,
        statements: List["Statement"],
        env: TypeEnvironment,
    ) -> InferenceResult:
        """Infer type of do block - return type of last statement."""
        if not statements:
            return UNIT_TYPE, TypeSubstitution()

        current_env = env
        current_subst = TypeSubstitution()

        # Process all statements except the last
        for stmt in statements[:-1]:
            current_env, stmt_subst = self._process_statement(stmt, current_env)
            current_subst = current_subst.compose(stmt_subst)

        # Process the last statement and return its type
        return self._process_final_statement(
            statements[-1],
            current_env.apply_substitution(current_subst),
            current_subst,
        )

    def _process_statement(
        self,
        stmt: "Statement",
        env: TypeEnvironment,
    ) -> Tuple[TypeEnvironment, TypeSubstitution]:
        """Process a statement and return updated environment and substitution."""
        match stmt:
            case LetStatement(variable=var, value=value):
                value_type, subst = self.infer_expr(value, env)
                var_scheme = generalize(
                    env.apply_substitution(subst).free_type_vars(),
                    value_type,
                )
                return env.extend(var, var_scheme), subst
            case _:
                # For expression statements, just type check and return original env
                if self._is_expression_statement(stmt):
                    _, subst = self.infer_expr(stmt, env)  # type: ignore
                    return env, subst
                return env, TypeSubstitution()

    def _process_final_statement(
        self,
        stmt: "Statement",
        env: TypeEnvironment,
        current_subst: TypeSubstitution,
    ) -> InferenceResult:
        """Process final statement and return its type."""
        match stmt:
            case LetStatement(variable=var, value=value):
                value_type, subst = self.infer_expr(value, env)
                return UNIT_TYPE, current_subst.compose(
                    subst,
                )  # Let statements don't return values
            case _:
                if self._is_expression_statement(stmt):
                    return self.infer_expr(stmt, env)  # type: ignore
                return UNIT_TYPE, current_subst

    def _is_expression_statement(self, stmt: "Statement") -> bool:
        """Check if statement is actually an expression."""
        expression_types = (
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
        )
        return isinstance(stmt, expression_types)

    def infer_constructor_expr(
        self,
        expr: ConstructorExpression,
        env: TypeEnvironment,
    ) -> InferenceResult:
        """Infer type of constructor expression."""
        # Look up constructor in environment
        constructor_name = expr.constructor_name
        if constructor_name not in env:
            raise TypeInferenceError(f"Unknown constructor: {constructor_name}")

        constructor_scheme = env[constructor_name]
        constructor_type = constructor_scheme.instantiate(self.fresh_var_gen)

        # Constructor type should be a function type from field types to result type
        # For now, assume simple case and return the result type
        # This is a simplification - full implementation would check field types
        current_subst = TypeSubstitution()

        # Infer types of field expressions
        for field in expr.fields:
            field_type, field_subst = self.infer_expr(
                field.value,
                env.apply_substitution(current_subst),
            )
            current_subst = current_subst.compose(field_subst)

        # Return constructor result type (simplified)
        match constructor_type:
            case FunctionType():
                # Walk through function type to get final return type
                result_type: Type = constructor_type
                while True:
                    match result_type:
                        case FunctionType():
                            result_type = result_type.result
                        case _:
                            break
                return result_type, current_subst
            case _:
                return constructor_type, current_subst

    def infer_pattern(
        self,
        pattern: Pattern,
        pattern_type: Type,
        env: TypeEnvironment,
    ) -> Tuple[TypeEnvironment, TypeSubstitution]:
        """Infer pattern and return extended environment and substitution."""
        match pattern:
            case VariablePattern(name=name):
                # Variable patterns bind the variable to the pattern type
                param_scheme = TypeScheme(set(), pattern_type)
                return env.extend(name, param_scheme), TypeSubstitution()

            case ConstructorPattern(constructor=constructor, patterns=patterns):
                # Constructor patterns need to unify with constructor type
                current_subst = TypeSubstitution()
                extended_env = env

                # Look up constructor type
                if constructor not in env:
                    raise TypeInferenceError(
                        f"Unknown constructor in pattern: {constructor}",
                    )

                constructor_scheme = env[constructor]
                constructor_type = constructor_scheme.instantiate(self.fresh_var_gen)

                # Unify pattern type with constructor result type
                match constructor_type:
                    case FunctionType():
                        # Walk through function type to get result type
                        result_type: Type = constructor_type
                        param_types = []
                        while True:
                            match result_type:
                                case FunctionType():
                                    param_types.append(result_type.param)
                                    result_type = result_type.result
                                case _:
                                    break

                        # Unify pattern type with constructor result type
                        unify_subst = unify_one(pattern_type, result_type)
                        current_subst = current_subst.compose(unify_subst)

                        # Infer sub-patterns with their corresponding parameter types
                        if len(patterns) != len(param_types):
                            raise TypeInferenceError(
                                f"Constructor {constructor} expects {len(param_types)} arguments, got {len(patterns)}",
                            )

                        for subpattern, param_type in zip(
                            patterns,
                            param_types,
                        ):
                            sub_env, sub_subst = self.infer_pattern(
                                subpattern,
                                param_type.apply_substitution(current_subst),
                                extended_env.apply_substitution(current_subst),
                            )
                            extended_env = sub_env
                            current_subst = current_subst.compose(sub_subst)
                    case _:
                        # Constructor with no parameters
                        unify_subst = unify_one(pattern_type, constructor_type)
                        current_subst = current_subst.compose(unify_subst)

                return extended_env, current_subst

            case ConsPattern(head=head, tail=tail):
                # Cons pattern (head : tail) - both head and tail must be compatible
                current_subst = TypeSubstitution()

                # Pattern type should be List of some type
                elem_type = self.fresh_type_var()
                list_type = TypeApp(TypeCon("List"), elem_type)

                # Unify pattern type with list type
                unify_subst = unify_one(pattern_type, list_type)
                current_subst = current_subst.compose(unify_subst)

                # Infer head pattern with element type
                head_env, head_subst = self.infer_pattern(
                    head,
                    elem_type.apply_substitution(current_subst),
                    env.apply_substitution(current_subst),
                )
                current_subst = current_subst.compose(head_subst)

                # Infer tail pattern with list type
                tail_env, tail_subst = self.infer_pattern(
                    tail,
                    list_type.apply_substitution(current_subst),
                    head_env.apply_substitution(current_subst),
                )
                current_subst = current_subst.compose(tail_subst)

                return tail_env, current_subst

            case ListLiteral(elements=elements):
                # List literal pattern [] or [p1, p2, ...]
                elem_type = self.fresh_type_var()
                list_type = TypeApp(TypeCon("List"), elem_type)

                # Unify pattern type with list type
                unify_subst = unify_one(pattern_type, list_type)
                current_subst = unify_subst
                extended_env = env

                # For now, only handle empty lists properly
                # TODO: Handle non-empty list patterns when elements are patterns, not expressions
                if len(elements) > 0:
                    # This is a limitation - we can't properly handle non-empty list patterns yet
                    # because elements are typed as expressions, not patterns
                    pass

                return extended_env, current_subst

            case LiteralPattern(value=value):
                # Literal patterns constrain the pattern type to the literal's type
                literal_type = None
                if isinstance(value, bool):
                    literal_type = BOOL_TYPE
                elif isinstance(value, int):
                    literal_type = INT_TYPE
                elif isinstance(value, float):
                    literal_type = FLOAT_TYPE
                elif isinstance(value, str):
                    literal_type = STRING_TYPE
                elif isinstance(value, list) and len(value) == 0:
                    # Empty list pattern [] - constrain to List of some type
                    elem_type = self.fresh_type_var()
                    literal_type = TypeApp(TypeCon("List"), elem_type)
                else:
                    raise TypeInferenceError(
                        f"Unsupported literal pattern type: {type(value)} with value: {value}",
                    )

                # Unify pattern type with literal type
                unify_subst = unify_one(pattern_type, literal_type)
                return env, unify_subst

            case _:
                # Other pattern types (literals, etc.)
                return env, TypeSubstitution()

    def infer_program(self, ast: Program) -> TypeEnvironment:
        """Infer types for an entire program."""
        env = TypeEnvironment()

        # Add built-in functions
        # putStr :: String -> IO ()
        putstr_type = FunctionType(TypeCon("String"), TypeApp(TypeCon("IO"), UNIT_TYPE))
        env = env.extend("putStr", TypeScheme(set(), putstr_type))

        # show :: a -> String
        show_type = FunctionType(TypeVar("a"), TypeCon("String"))
        env = env.extend("show", TypeScheme({"a"}, show_type))

        # error :: String -> a
        error_type = FunctionType(STRING_TYPE, TypeVar("a"))
        env = env.extend("error", TypeScheme({"a"}, error_type))

        # Comparison operators
        # (==) :: a -> a -> Bool
        eq_type = FunctionType(
            TypeVar("a"),
            FunctionType(TypeVar("a"), TypeCon("Bool")),
        )
        env = env.extend("==", TypeScheme({"a"}, eq_type))

        # First pass: collect data declarations
        for stmt in ast.statements:
            match stmt:
                case DataDeclaration() as data_decl:
                    data_env = self.infer_data_decl(data_decl)
                    env = env.extend_many(data_env.bindings)
                case _:
                    continue

        # Second pass: create forward declarations for all functions
        # This allows functions to refer to each other regardless of order
        function_names = []
        for stmt in ast.statements:
            match stmt:
                case FunctionDefinition(function_name=function_name):
                    function_names.append(function_name)
                    # Create a fresh type variable for each function
                    func_type_var = self.fresh_type_var()
                    env = env.extend(
                        function_name,
                        TypeScheme(set(), func_type_var),
                    )
                case _:
                    continue

        # Third pass: group function definitions and infer them together
        from collections import defaultdict

        function_groups: Dict[str, List[FunctionDefinition]] = defaultdict(list)

        # Group function definitions by name
        for stmt in ast.statements:
            match stmt:
                case FunctionDefinition(function_name=function_name) as func_def:
                    function_groups[function_name].append(func_def)
                case _:
                    continue

        # Process each group of function definitions
        for function_name, func_defs in function_groups.items():
            try:
                if len(func_defs) == 1:
                    # Single function definition
                    scheme, _ = self.infer_function(func_defs[0], env)
                    env = env.extend(function_name, scheme)
                else:
                    # Multiple function clauses - group them together
                    scheme, _ = self.infer_function_group(func_defs, env)
                    env = env.extend(function_name, scheme)
            except TypeInferenceError as e:
                # Continue with other functions even if one fails
                raise TypeInferenceError(
                    f"Failed to infer type for function {function_name}: {e}",
                ) from e

        return env


def type_check_ast(ast: Program) -> TypeEnvironment:
    """Type check an AST and return the type environment."""
    inferrer = TypeInferrer()
    return inferrer.infer_program(ast)
