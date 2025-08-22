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
        if isinstance(node, TypeConstructor):
            type_name = node.name
            if type_name == "Int":
                return INT_TYPE
            elif type_name == "String":
                return STRING_TYPE
            elif type_name == "Float":
                return FLOAT_TYPE
            elif type_name == "Bool":
                return BOOL_TYPE
            else:
                return DataType(type_name, [])

        elif isinstance(node, TypeVariable):
            return TypeVar(node.name)

        elif isinstance(node, ArrowType):
            from_type = self.parse_type_expr(node.from_type)
            to_type = self.parse_type_expr(node.to_type)
            return FunctionType(from_type, to_type)

        elif isinstance(node, TypeApplication):
            constructor_type = self.parse_type_expr(node.constructor)
            argument_type = self.parse_type_expr(node.argument)

            if isinstance(constructor_type, DataType):
                # Apply type argument to data type
                new_args = constructor_type.type_args + [argument_type]
                return DataType(constructor_type.name, new_args)
            else:
                return TypeApp(constructor_type, argument_type)

        elif isinstance(node, GroupedType):
            return self.parse_type_expr(node.type_expr)

        else:
            raise TypeInferenceError(
                f"Cannot parse type expression: {type(node).__name__}",
            )

    def infer_expr(self, expr: Expression, env: TypeEnvironment) -> InferenceResult:
        """Infer the type of an expression."""

        # Literals
        if isinstance(expr, IntLiteral) or isinstance(expr, NegativeInt):
            expr.ty = INT_TYPE
            return INT_TYPE, TypeSubstitution()

        elif isinstance(expr, FloatLiteral) or isinstance(expr, NegativeFloat):
            expr.ty = FLOAT_TYPE
            return FLOAT_TYPE, TypeSubstitution()

        elif isinstance(expr, StringLiteral):
            expr.ty = STRING_TYPE
            return STRING_TYPE, TypeSubstitution()

        elif isinstance(expr, BoolLiteral):
            expr.ty = BOOL_TYPE
            return BOOL_TYPE, TypeSubstitution()

        elif isinstance(expr, ListLiteral):
            if not expr.elements:
                # Empty list: infer polymorphic list type
                element_type = self.fresh_type_var()
                list_type = TypeApp(TypeCon("List"), element_type)
                expr.ty = list_type
                return list_type, TypeSubstitution()

            # Non-empty list: infer element type from first element
            # and unify with all other elements
            first_type, subst1 = self.infer_expr(expr.elements[0], env)
            current_subst = subst1

            for element in expr.elements[1:]:
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
                    raise TypeInferenceError(
                        f"List elements have incompatible types: {e}",
                    )

            list_type = TypeApp(
                TypeCon("List"),
                first_type.apply_substitution(current_subst),
            )
            expr.ty = list_type
            return list_type, current_subst

        # Variables and constructors
        elif isinstance(expr, Variable):
            scheme = env.lookup(expr.name)
            if scheme is None:
                raise TypeInferenceError(f"Unknown variable: {expr.name}")
            inferred_type = scheme.instantiate(self.fresh_var_gen)
            expr.ty = inferred_type
            return inferred_type, TypeSubstitution()

        elif isinstance(expr, Constructor):
            scheme = env.lookup(expr.name)
            if scheme is None:
                raise TypeInferenceError(f"Unknown constructor: {expr.name}")
            inferred_type = scheme.instantiate(self.fresh_var_gen)
            expr.ty = inferred_type
            return inferred_type, TypeSubstitution()

        # Arithmetic operations
        elif isinstance(expr, AddOperation):
            result = self._infer_binary_numeric_op(expr.left, expr.right, env)
            expr.ty = result[0]
            return result

        elif isinstance(expr, SubOperation):
            result = self._infer_binary_numeric_op(expr.left, expr.right, env)
            expr.ty = result[0]
            return result

        elif isinstance(expr, MulOperation):
            result = self._infer_binary_numeric_op(expr.left, expr.right, env)
            expr.ty = result[0]
            return result

        elif isinstance(expr, DivOperation):
            result = self._infer_binary_numeric_op(expr.left, expr.right, env)
            expr.ty = result[0]
            return result

        elif isinstance(expr, PowIntOperation):
            result = self._infer_binary_numeric_op(expr.left, expr.right, env)
            expr.ty = result[0]
            return result

        elif isinstance(expr, PowFloatOperation):
            result = self._infer_binary_numeric_op(expr.left, expr.right, env)
            expr.ty = result[0]
            return result

        # Comparison operations
        elif isinstance(expr, EqualOperation):
            result = self._infer_binary_comparison_op(expr.left, expr.right, env)
            expr.ty = result[0]
            return result

        elif isinstance(expr, NotEqualOperation):
            result = self._infer_binary_comparison_op(expr.left, expr.right, env)
            expr.ty = result[0]
            return result

        elif isinstance(expr, LessThanOperation):
            result = self._infer_binary_comparison_op(expr.left, expr.right, env)
            expr.ty = result[0]
            return result

        elif isinstance(expr, LessEqualOperation):
            result = self._infer_binary_comparison_op(expr.left, expr.right, env)
            expr.ty = result[0]
            return result

        elif isinstance(expr, GreaterThanOperation):
            result = self._infer_binary_comparison_op(expr.left, expr.right, env)
            expr.ty = result[0]
            return result

        elif isinstance(expr, GreaterEqualOperation):
            result = self._infer_binary_comparison_op(expr.left, expr.right, env)
            expr.ty = result[0]
            return result

        # Logical operations
        elif isinstance(expr, AndOperation):
            result = self._infer_binary_logical_op(expr.left, expr.right, env)
            expr.ty = result[0]
            return result

        elif isinstance(expr, OrOperation):
            result = self._infer_binary_logical_op(expr.left, expr.right, env)
            expr.ty = result[0]
            return result

        elif isinstance(expr, NotOperation):
            operand_type, subst = self.infer_expr(expr.operand, env)
            try:
                bool_unify = unify_one(operand_type, BOOL_TYPE)
                final_subst = subst.compose(bool_unify)
                expr.ty = BOOL_TYPE
                return BOOL_TYPE, final_subst
            except UnificationError:
                raise TypeInferenceError(
                    f"NOT operation requires Bool operand, got {operand_type}",
                )

        # String/List operations
        elif isinstance(expr, ConcatOperation):
            left_type, left_subst = self.infer_expr(expr.left, env)
            right_type, right_subst = self.infer_expr(
                expr.right,
                env.apply_substitution(left_subst),
            )

            combined_subst = left_subst.compose(right_subst)

            # Try to unify both operands
            try:
                unify_subst = unify_one(
                    left_type.apply_substitution(combined_subst),
                    right_type.apply_substitution(combined_subst),
                )
                final_subst = combined_subst.compose(unify_subst)
                final_type = left_type.apply_substitution(final_subst)
                expr.ty = final_type
                return final_type, final_subst
            except UnificationError:
                raise TypeInferenceError(f"Concatenation operands must have same type")

        elif isinstance(expr, IndexOperation):
            indexed_list_type: Type
            indexed_list_type, list_subst = self.infer_expr(expr.list_expr, env)
            index_type, index_subst = self.infer_expr(
                expr.index_expr,
                env.apply_substitution(list_subst),
            )

            combined_subst = list_subst.compose(index_subst)

            # Index must be Int
            try:
                int_unify = unify_one(index_type, INT_TYPE)
                subst_with_int = combined_subst.compose(int_unify)
            except UnificationError:
                raise TypeInferenceError(f"List index must be Int, got {index_type}")

            # List must be List[T] for some T
            element_type = self.fresh_type_var()
            expected_list_type = TypeApp(TypeCon("List"), element_type)

            try:
                list_unify = unify_one(
                    indexed_list_type.apply_substitution(subst_with_int),
                    expected_list_type,
                )
                final_subst = subst_with_int.compose(list_unify)
                result_type = element_type.apply_substitution(final_subst)
                expr.ty = result_type
                return result_type, final_subst
            except UnificationError:
                raise TypeInferenceError(
                    f"Index operation requires a List, got {indexed_list_type}",
                )

        # Control flow
        elif isinstance(expr, IfElse):
            cond_type, cond_subst = self.infer_expr(expr.condition, env)

            # Condition must be Bool
            try:
                bool_unify = unify_one(cond_type, BOOL_TYPE)
                subst_after_cond = cond_subst.compose(bool_unify)
            except UnificationError:
                raise TypeInferenceError(f"If condition must be Bool, got {cond_type}")

            # Infer then branch
            then_type, then_subst = self.infer_expr(
                expr.then_expr,
                env.apply_substitution(subst_after_cond),
            )
            subst_after_then = subst_after_cond.compose(then_subst)

            # Infer else branch
            else_type, else_subst = self.infer_expr(
                expr.else_expr,
                env.apply_substitution(subst_after_then),
            )
            subst_after_else = subst_after_then.compose(else_subst)

            # Then and else branches must have same type
            try:
                branch_unify = unify_one(
                    then_type.apply_substitution(subst_after_else),
                    else_type,
                )
                final_subst = subst_after_else.compose(branch_unify)
                final_type = then_type.apply_substitution(final_subst)
                expr.ty = final_type
                return final_type, final_subst
            except UnificationError:
                raise TypeInferenceError(
                    f"If branches have incompatible types: {then_type} vs {else_type}",
                )

        # Function application
        elif isinstance(expr, FunctionApplication):
            func_type, func_subst = self.infer_expr(expr.function, env)
            arg_type, arg_subst = self.infer_expr(
                expr.argument,
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
                result_type = return_type.apply_substitution(final_subst)
                expr.ty = result_type
                return result_type, final_subst
            except UnificationError:
                raise TypeInferenceError(f"Function application type mismatch")

        # Grouping
        elif isinstance(expr, GroupedExpression):
            result = self.infer_expr(expr.expression, env)
            expr.ty = result[0]
            return result

        # Do blocks
        elif isinstance(expr, DoBlock):
            result = self.infer_do_block(expr.statements, env)
            expr.ty = result[0]
            return result

        # Constructor expressions
        elif isinstance(expr, ConstructorExpression):
            result = self.infer_constructor_expr(expr, env)
            expr.ty = result[0]
            return result

        else:
            raise TypeInferenceError(
                f"Unhandled expression type: {type(expr).__name__}",
            )

    def _infer_binary_numeric_op(
        self,
        left: Expression,
        right: Expression,
        env: TypeEnvironment,
    ) -> InferenceResult:
        """Helper for binary numeric operations."""
        left_type, left_subst = self.infer_expr(left, env)
        right_type, right_subst = self.infer_expr(
            right,
            env.apply_substitution(left_subst),
        )

        combined_subst = left_subst.compose(right_subst)

        # Both operands must have the same numeric type
        try:
            unify_subst = unify_one(
                left_type.apply_substitution(combined_subst),
                right_type.apply_substitution(combined_subst),
            )
            final_subst = combined_subst.compose(unify_subst)

            # Check that the unified type is numeric (Int or Float)
            unified_type = left_type.apply_substitution(final_subst)
            if unified_type == INT_TYPE or unified_type == FLOAT_TYPE:
                return unified_type, final_subst
            else:
                # Try to unify with Int
                try:
                    int_unify = unify_one(unified_type, INT_TYPE)
                    return INT_TYPE, final_subst.compose(int_unify)
                except UnificationError:
                    # Try to unify with Float
                    try:
                        float_unify = unify_one(unified_type, FLOAT_TYPE)
                        return FLOAT_TYPE, final_subst.compose(float_unify)
                    except UnificationError:
                        raise TypeInferenceError(
                            f"Numeric operation requires Int or Float, got {unified_type}",
                        )
        except UnificationError:
            raise TypeInferenceError(
                f"Binary numeric operation requires operands of same type",
            )

    def _infer_binary_comparison_op(
        self,
        left: Expression,
        right: Expression,
        env: TypeEnvironment,
    ) -> InferenceResult:
        """Helper for binary comparison operations."""
        left_type, left_subst = self.infer_expr(left, env)
        right_type, right_subst = self.infer_expr(
            right,
            env.apply_substitution(left_subst),
        )

        combined_subst = left_subst.compose(right_subst)

        # Both operands must have the same type (for comparison)
        try:
            unify_subst = unify_one(
                left_type.apply_substitution(combined_subst),
                right_type.apply_substitution(combined_subst),
            )
            final_subst = combined_subst.compose(unify_subst)
            return BOOL_TYPE, final_subst
        except UnificationError:
            raise TypeInferenceError(f"Comparison requires operands of same type")

    def _infer_binary_logical_op(
        self,
        left: Expression,
        right: Expression,
        env: TypeEnvironment,
    ) -> InferenceResult:
        """Helper for binary logical operations."""
        left_type, left_subst = self.infer_expr(left, env)
        right_type, right_subst = self.infer_expr(
            right,
            env.apply_substitution(left_subst),
        )

        combined_subst = left_subst.compose(right_subst)

        # Both operands must be Bool
        try:
            left_bool_unify = unify_one(left_type, BOOL_TYPE)
            subst_with_left = combined_subst.compose(left_bool_unify)

            right_bool_unify = unify_one(right_type, BOOL_TYPE)
            final_subst = subst_with_left.compose(right_bool_unify)

            return BOOL_TYPE, final_subst
        except UnificationError:
            raise TypeInferenceError(f"Logical operation requires Bool operands")

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
            scheme = generalize(
                env.apply_substitution(body_subst).free_type_vars(),
                body_type.apply_substitution(body_subst),
            )
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
        return scheme, env.extend(func_def.function_name, scheme)

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
            if isinstance(stmt, LetStatement):
                # Handle let statements
                value_type, value_subst = self.infer_expr(stmt.value, current_env)
                current_subst = current_subst.compose(value_subst)

                # Generalize and add to environment
                var_scheme = generalize(
                    current_env.apply_substitution(
                        current_subst,
                    ).free_type_vars(),
                    value_type,
                )
                current_env = current_env.extend(stmt.variable, var_scheme)

            # Check if it's an expression (not a declaration)
            elif isinstance(
                stmt,
                (
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
                ),
            ):
                # Type check but ignore result for intermediate expressions
                _, stmt_subst = self.infer_expr(
                    stmt,
                    current_env.apply_substitution(current_subst),
                )
                current_subst = current_subst.compose(stmt_subst)

        # Process the last statement and return its type
        last_stmt = statements[-1]
        if isinstance(last_stmt, LetStatement):
            # Handle let statement
            value_type, value_subst = self.infer_expr(last_stmt.value, current_env)
            final_subst = current_subst.compose(value_subst)

            # Generalize and add to environment
            var_scheme = generalize(
                current_env.apply_substitution(
                    final_subst,
                ).free_type_vars(),
                value_type,
            )
            current_env = current_env.extend(last_stmt.variable, var_scheme)

            return UNIT_TYPE, final_subst  # Let statements don't return values
        elif isinstance(
            last_stmt,
            (
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
            ),
        ):
            # It's an expression - return its type
            return self.infer_expr(
                last_stmt,
                current_env.apply_substitution(current_subst),
            )
        else:
            # Other statement types (like declarations) don't return values
            return UNIT_TYPE, current_subst

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
        if isinstance(constructor_type, FunctionType):
            # Walk through function type to get final return type
            result_type: Type = constructor_type
            while isinstance(result_type, FunctionType):
                result_type = result_type.result
            return result_type, current_subst
        else:
            return constructor_type, current_subst

    def infer_pattern(
        self,
        pattern: Pattern,
        pattern_type: Type,
        env: TypeEnvironment,
    ) -> Tuple[TypeEnvironment, TypeSubstitution]:
        """Infer pattern and return extended environment and substitution."""
        if isinstance(pattern, VariablePattern):
            # Variable patterns bind the variable to the pattern type
            param_scheme = TypeScheme(set(), pattern_type)
            return env.extend(pattern.name, param_scheme), TypeSubstitution()

        elif isinstance(pattern, ConstructorPattern):
            # Constructor patterns need to unify with constructor type
            current_subst = TypeSubstitution()
            extended_env = env

            # Look up constructor type
            constructor_name = pattern.constructor
            if constructor_name not in env:
                raise TypeInferenceError(
                    f"Unknown constructor in pattern: {constructor_name}",
                )

            constructor_scheme = env[constructor_name]
            constructor_type = constructor_scheme.instantiate(self.fresh_var_gen)

            # Unify pattern type with constructor result type
            if isinstance(constructor_type, FunctionType):
                # Walk through function type to get result type
                result_type: Type = constructor_type
                param_types = []
                while isinstance(result_type, FunctionType):
                    param_types.append(result_type.param)
                    result_type = result_type.result

                # Unify pattern type with constructor result type
                unify_subst = unify_one(pattern_type, result_type)
                current_subst = current_subst.compose(unify_subst)

                # Infer sub-patterns with their corresponding parameter types
                if len(pattern.patterns) != len(param_types):
                    raise TypeInferenceError(
                        f"Constructor {constructor_name} expects {len(param_types)} arguments, got {len(pattern.patterns)}",
                    )

                for subpattern, param_type in zip(pattern.patterns, param_types):
                    sub_env, sub_subst = self.infer_pattern(
                        subpattern,
                        param_type.apply_substitution(current_subst),
                        extended_env.apply_substitution(current_subst),
                    )
                    extended_env = sub_env
                    current_subst = current_subst.compose(sub_subst)
            else:
                # Constructor with no parameters
                unify_subst = unify_one(pattern_type, constructor_type)
                current_subst = current_subst.compose(unify_subst)

            return extended_env, current_subst

        elif isinstance(pattern, ConsPattern):
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
                pattern.head,
                elem_type.apply_substitution(current_subst),
                env.apply_substitution(current_subst),
            )
            current_subst = current_subst.compose(head_subst)

            # Infer tail pattern with list type
            tail_env, tail_subst = self.infer_pattern(
                pattern.tail,
                list_type.apply_substitution(current_subst),
                head_env.apply_substitution(current_subst),
            )
            current_subst = current_subst.compose(tail_subst)

            return tail_env, current_subst

        elif isinstance(pattern, ListLiteral):
            # List literal pattern [] or [p1, p2, ...]
            elem_type = self.fresh_type_var()
            list_type = TypeApp(TypeCon("List"), elem_type)

            # Unify pattern type with list type
            unify_subst = unify_one(pattern_type, list_type)
            current_subst = unify_subst
            extended_env = env

            # For now, only handle empty lists properly
            # TODO: Handle non-empty list patterns when elements are patterns, not expressions
            if len(pattern.elements) > 0:
                # This is a limitation - we can't properly handle non-empty list patterns yet
                # because elements are typed as expressions, not patterns
                pass

            return extended_env, current_subst

        else:
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

        # Comparison operators
        # (==) :: a -> a -> Bool
        eq_type = FunctionType(
            TypeVar("a"),
            FunctionType(TypeVar("a"), TypeCon("Bool")),
        )
        env = env.extend("==", TypeScheme({"a"}, eq_type))

        # First pass: collect data declarations
        for stmt in ast.statements:
            if isinstance(stmt, DataDeclaration):
                data_env = self.infer_data_decl(stmt)
                env = env.extend_many(data_env.bindings)

        # Second pass: create forward declarations for all functions
        # This allows functions to refer to each other regardless of order
        function_names = []
        for stmt in ast.statements:
            if isinstance(stmt, FunctionDefinition):
                function_names.append(stmt.function_name)
                # Create a fresh type variable for each function
                func_type_var = self.fresh_type_var()
                env = env.extend(stmt.function_name, TypeScheme(set(), func_type_var))

        # Third pass: infer function definitions with forward declarations available
        for stmt in ast.statements:
            if isinstance(stmt, FunctionDefinition):
                try:
                    scheme, _ = self.infer_function(stmt, env)
                    # Replace the forward declaration with the properly inferred type
                    env = env.extend(stmt.function_name, scheme)
                except TypeInferenceError as e:
                    # Continue with other functions even if one fails
                    raise TypeInferenceError(
                        f"Failed to infer type for function {stmt.function_name}: {e}",
                    ) from e

        return env


def type_check_ast(ast: Program) -> TypeEnvironment:
    """Type check an AST and return the type environment."""
    inferrer = TypeInferrer()
    return inferrer.infer_program(ast)
