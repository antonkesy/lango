"""
Main type inference engine using the Hindley-Milner algorithm
"""

from typing import Dict, List, Optional, Set, Tuple, Union

from lark import Token, Tree

from lango.minio.typechecker.types import (
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
from lango.minio.typechecker.unify import UnificationError, unify

ASTNode = Union[Tree, Token]
TypeBindings = Dict[str, TypeScheme]
InferenceResult = Tuple[Type, TypeSubstitution]
EnvironmentBinding = Dict[str, TypeScheme]


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

    def free_vars(self) -> Set[str]:
        """Return free variables in all type schemes in this environment."""
        result: Set[str] = set()
        for scheme in self.bindings.values():
            result |= scheme.free_vars()
        return result

    def substitute(self, subst: TypeSubstitution) -> "TypeEnvironment":
        """Apply a substitution to all type schemes in the environment."""
        new_bindings: TypeBindings = {}
        for name, scheme in self.bindings.items():
            new_bindings[name] = scheme.substitute(subst)
        return TypeEnvironment(new_bindings)


class TypeInferenceError(Exception):
    """Raised when type inference fails."""

    def __init__(self, message: str, node: Optional[ASTNode] = None) -> None:
        """Initialize with error message and optional AST node."""
        super().__init__(message)
        self.message = message
        self.node = node


class TypeInferrer:
    """Hindley-Milner type inference engine."""

    def __init__(self) -> None:
        """Initialize the type inferrer with built-in environment and fresh variable generator."""
        self.fresh_var_gen = FreshVarGenerator()
        self.constraints: List[Tuple[Type, Type]] = []

        # Built-in type environment
        self.builtin_env = self._create_builtin_env()

        # Data type definitions (constructor name -> type info)
        self.data_constructors: Dict[str, Tuple[str, List[Type]]] = (
            {}
        )  # constructor -> (data_type_name, field_types)
        self.data_types: Dict[str, List[str]] = (
            {}
        )  # data_type_name -> [constructor_names]

    def _create_builtin_env(self) -> TypeEnvironment:
        """Create the initial type environment with built-in functions"""
        env = TypeEnvironment()

        # Arithmetic operations
        int_binop = TypeScheme(
            set(),
            FunctionType(INT_TYPE, FunctionType(INT_TYPE, INT_TYPE)),
        )
        float_binop = TypeScheme(
            set(),
            FunctionType(FLOAT_TYPE, FunctionType(FLOAT_TYPE, FLOAT_TYPE)),
        )

        # Comparison operations (polymorphic for now, but could be restricted)
        a = TypeVar("a")
        comparison = TypeScheme({"a"}, FunctionType(a, FunctionType(a, BOOL_TYPE)))

        # String operations
        string_concat = TypeScheme(
            set(),
            FunctionType(STRING_TYPE, FunctionType(STRING_TYPE, STRING_TYPE)),
        )

        # IO operations - using ad-hoc polymorphism approach
        # putStr can work with strings or functions that return strings
        a = TypeVar("a")

        # Method 1: Give putStr overloaded types
        # For now, let's assume that putStr gets specialized based on usage
        put_str_string = TypeScheme(set(), FunctionType(STRING_TYPE, UNIT_TYPE))

        get_line = TypeScheme(set(), STRING_TYPE)
        read_int = TypeScheme(set(), INT_TYPE)
        read_string = TypeScheme(set(), STRING_TYPE)
        read_bool = TypeScheme(set(), BOOL_TYPE)
        show = TypeScheme({"a"}, FunctionType(a, STRING_TYPE))

        builtins = {
            # Arithmetic
            "+": int_binop,
            "-": int_binop,
            "*": int_binop,
            "/": float_binop,
            "^": int_binop,
            "**": float_binop,
            "^^": float_binop,
            "mod": int_binop,
            "quot": int_binop,
            # Comparison
            "==": comparison,
            "/=": comparison,
            "<": comparison,
            "<=": comparison,
            ">": comparison,
            ">=": comparison,
            # Logical
            "&&": TypeScheme(
                set(),
                FunctionType(BOOL_TYPE, FunctionType(BOOL_TYPE, BOOL_TYPE)),
            ),
            "||": TypeScheme(
                set(),
                FunctionType(BOOL_TYPE, FunctionType(BOOL_TYPE, BOOL_TYPE)),
            ),
            "not": TypeScheme(set(), FunctionType(BOOL_TYPE, BOOL_TYPE)),
            # String
            "++": string_concat,
            "concat": string_concat,
            # IO
            "putStr": put_str_string,
            "getLine": get_line,
            "readInt": read_int,
            "readString": read_string,
            "readBool": read_bool,
            "show": show,
        }

        for name, scheme in builtins.items():
            env = env.extend(name, scheme)

        return env

    def fresh_type_var(self) -> TypeVar:
        """Generate a fresh type variable"""
        return TypeVar(self.fresh_var_gen.fresh())

    def add_constraint(self, t1: Type, t2: Type) -> None:
        """Add a type constraint that t1 must equal t2."""
        self.constraints.append((t1, t2))

    def infer_data_decl(self, node: Tree) -> TypeEnvironment:
        """Process a data type declaration and return updated environment"""
        # data TYPE type_param* "=" constructor ("|" constructor)*
        type_name_token = node.children[0]
        type_name = (
            type_name_token.value
            if isinstance(type_name_token, Token)
            else str(type_name_token)
        )

        # Extract type parameters
        type_params = []
        constructors_start_idx = 1

        # Find where constructors start (after type parameters)
        for i, child in enumerate(node.children[1:], 1):
            if isinstance(child, Tree) and child.data == "type_param":
                param_token = child.children[0]
                param_name = (
                    param_token.value
                    if isinstance(param_token, Token)
                    else str(param_token)
                )
                type_params.append(param_name)
                constructors_start_idx = i + 1
            else:
                break

        constructors = node.children[constructors_start_idx:]

        constructor_names = []
        constructor_types = {}

        for constructor in constructors:
            if constructor.data == "constructor":
                ctor_name_token = constructor.children[0]
                ctor_name = (
                    ctor_name_token.value
                    if isinstance(ctor_name_token, Token)
                    else str(ctor_name_token)
                )
                constructor_names.append(ctor_name)

                # Check if this is a record constructor or positional constructor
                if (
                    len(constructor.children) > 1
                    and isinstance(constructor.children[1], Tree)
                    and constructor.children[1].data == "record_constructor"
                ):

                    # Record constructor: UIDENT record_constructor
                    record_def = constructor.children[1]
                    field_types = []

                    if len(record_def.children) > 0:
                        for field in record_def.children:
                            if field.data == "field":
                                # field has structure: ID "::" type_expr
                                field_type_expr = field.children[
                                    1
                                ]  # Get the type expression
                                field_type = self.parse_type_expr(field_type_expr)
                                field_types.append(field_type)

                    # For record constructors, we create a function that takes all fields
                    type_param_vars: List[Type] = [
                        TypeVar(param) for param in type_params
                    ]
                    result_type = DataType(type_name, type_param_vars)
                    ctor_type = result_type
                    for field_type in reversed(field_types):
                        ctor_type = FunctionType(field_type, ctor_type)

                    # Generalize over the type parameters
                    bound_vars = set(type_params)
                    constructor_types[ctor_name] = TypeScheme(bound_vars, ctor_type)
                    self.data_constructors[ctor_name] = (type_name, field_types)

                else:
                    # Positional constructor: UIDENT type_atom*
                    type_param_vars: List[Type] = [
                        TypeVar(param) for param in type_params
                    ]
                    result_type = DataType(type_name, type_param_vars)

                    if len(constructor.children) == 1:
                        # Nullary constructor
                        bound_vars = set(type_params)
                        ctor_type = TypeScheme(bound_vars, result_type)
                    else:
                        # Constructor with arguments
                        field_types = []
                        for type_expr in constructor.children[1:]:
                            field_type = self.parse_type_expr(type_expr)
                            field_types.append(field_type)

                        # Create function type: field1 -> field2 -> ... -> DataType
                        ctor_type = result_type
                        for field_type in reversed(field_types):
                            ctor_type = FunctionType(field_type, ctor_type)

                        # Generalize over the type parameters
                        bound_vars = set(type_params)
                        ctor_type = TypeScheme(bound_vars, ctor_type)

                    constructor_types[ctor_name] = ctor_type
                    self.data_constructors[ctor_name] = (
                        type_name,
                        [],
                    )  # Simplified for now

        self.data_types[type_name] = constructor_names

        # Return environment extended with constructor types
        env = TypeEnvironment()
        for name, scheme in constructor_types.items():
            env = env.extend(name, scheme)

        return env

    def parse_type_expr(self, node: ASTNode) -> Type:
        """Parse a type expression from the AST."""
        if isinstance(node, Token):
            type_name = node.value
            if type_name == "Int":
                return INT_TYPE
            elif type_name == "String":
                return STRING_TYPE
            elif type_name == "Float":
                return FLOAT_TYPE
            elif type_name == "Bool":
                return BOOL_TYPE
            else:
                # Custom data type or type variable
                if type_name[0].isupper():
                    return DataType(type_name, [])
                else:
                    return TypeVar(type_name)

        elif isinstance(node, Tree):
            if node.data == "type_constructor":
                # New grammar: Tree('type_constructor', [Token('TYPE', 'Int')])
                child = node.children[0]
                if isinstance(child, Token):
                    type_name = child.value
                else:
                    # If it's a Tree, get the first token
                    first_token = child.children[0]
                    if isinstance(first_token, Token):
                        type_name = first_token.value
                    else:
                        raise TypeInferenceError(
                            f"Expected token for type name, got {type(first_token)}",
                            child,
                        )

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

            elif node.data == "type_var":
                # New grammar: Tree('type_var', [Token('ID', 'a')])
                child = node.children[0]
                if isinstance(child, Token):
                    var_name = child.value
                else:
                    first_token = child.children[0]
                    if isinstance(first_token, Token):
                        var_name = first_token.value
                    else:
                        raise TypeInferenceError(
                            f"Expected token for type var name, got {type(first_token)}",
                            child,
                        )
                return TypeVar(var_name)

            elif node.data == "arrow_type":
                # New grammar: param_type -> result_type
                param_type = self.parse_type_expr(node.children[0])
                result_type = self.parse_type_expr(node.children[1])
                return FunctionType(param_type, result_type)

            elif node.data == "type_application":
                # New grammar: type constructor applied to arguments
                func_type = self.parse_type_expr(node.children[0])
                arg_type = self.parse_type_expr(node.children[1])

                # Handle parameterized types like "List Int" or "Pair Int String"
                if isinstance(func_type, DataType):
                    # Add the argument to the type constructor's arguments
                    return DataType(func_type.name, func_type.type_args + [arg_type])
                else:
                    # For now, we don't handle higher-order type constructors
                    raise TypeInferenceError(
                        f"Cannot apply non-constructor type: {func_type}",
                    )

            elif node.data == "grouped_type":
                # Grouped type expression in parentheses
                return self.parse_type_expr(node.children[0])

            elif node.data == "type_expr":
                # Legacy compatibility: old grammar structure
                if len(node.children) == 1:
                    return self.parse_type_expr(node.children[0])
                else:
                    # Function type
                    param_type = self.parse_type_expr(node.children[0])
                    result_type = self.parse_type_expr(node.children[1])
                    return FunctionType(param_type, result_type)

        raise TypeInferenceError(f"Cannot parse type expression: {node}")

    def infer_expr(self, expr: ASTNode, env: TypeEnvironment) -> InferenceResult:
        """Infer the type of an expression."""
        if isinstance(expr, Tree):
            match expr.data:
                case "int" | "neg_int":
                    return INT_TYPE, TypeSubstitution()

                case "float" | "neg_float":
                    return FLOAT_TYPE, TypeSubstitution()

                case "string":
                    return STRING_TYPE, TypeSubstitution()

                case "true" | "false":
                    return BOOL_TYPE, TypeSubstitution()

                case "list":
                    # List literal [expr1, expr2, ..., exprN]
                    # Note: empty lists have [None] as children due to Lark's optional matching
                    if len(expr.children) == 0 or (
                        len(expr.children) == 1 and expr.children[0] is None
                    ):
                        # Empty list: infer polymorphic list type
                        element_type = self.fresh_type_var()
                        return (
                            TypeApp(TypeCon("List"), element_type),
                            TypeSubstitution(),
                        )

                    # Non-empty list: infer element type from first element
                    # and unify with all other elements
                    first_type, s1 = self.infer_expr(expr.children[0], env)
                    current_subst = s1

                    # Unify all elements to have the same type
                    for element_expr in expr.children[1:]:
                        element_type, s2 = self.infer_expr(
                            element_expr,
                            env.substitute(current_subst),
                        )

                        # Unify with first element type
                        s3 = unify([(current_subst.apply(first_type), element_type)])
                        current_subst = s3.compose(s2).compose(current_subst)

                    # Return list type with unified element type
                    element_type = current_subst.apply(first_type)
                    return TypeApp(TypeCon("List"), element_type), current_subst

                case "var":
                    var_name_token = expr.children[0]
                    var_name = (
                        var_name_token.value
                        if isinstance(var_name_token, Token)
                        else str(var_name_token)
                    )
                    scheme = env.lookup(var_name)
                    if scheme is None:
                        raise TypeInferenceError(f"Unknown variable: {var_name}")

                    # Instantiate the type scheme
                    typ = scheme.instantiate(self.fresh_var_gen)
                    return typ, TypeSubstitution()

                case "constructor":
                    ctor_name_token = expr.children[0]
                    ctor_name = (
                        ctor_name_token.value
                        if isinstance(ctor_name_token, Token)
                        else str(ctor_name_token)
                    )
                    scheme = env.lookup(ctor_name)
                    if scheme is None:
                        # Try built-in constructors
                        if ctor_name in self.data_constructors:
                            data_type_name, field_types = self.data_constructors[
                                ctor_name
                            ]
                            result_type = DataType(data_type_name, [])
                            ctor_type = result_type
                            for field_type in reversed(field_types):
                                ctor_type = FunctionType(field_type, ctor_type)
                            scheme = TypeScheme(set(), ctor_type)
                        else:
                            raise TypeInferenceError(
                                f"Unknown constructor: {ctor_name}",
                            )

                    typ = scheme.instantiate(self.fresh_var_gen)
                    return typ, TypeSubstitution()

                case "app":
                    # Function application: f x
                    func_expr = expr.children[0]
                    arg_expr = expr.children[1]

                    # Special case: putStr/error show -> give it the curried type
                    if (
                        isinstance(func_expr, Tree)
                        and func_expr.data == "var"
                        and isinstance(func_expr.children[0], Token)
                        and func_expr.children[0].value == "putStr"
                        and isinstance(arg_expr, Tree)
                        and arg_expr.data == "var"
                        and isinstance(arg_expr.children[0], Token)
                        and arg_expr.children[0].value == "show"
                    ):

                        # putStr show has type: a -> () where a is showable
                        a = self.fresh_type_var()
                        result_type = FunctionType(a, UNIT_TYPE)
                        return result_type, TypeSubstitution()

                    # Normal function application
                    func_type, s1 = self.infer_expr(func_expr, env)
                    arg_type, s2 = self.infer_expr(arg_expr, env.substitute(s1))

                    result_type = self.fresh_type_var()
                    expected_func_type = FunctionType(arg_type, result_type)

                    # Unify function type with expected type
                    s3 = unify([(s2.apply(func_type), expected_func_type)])

                    final_subst = s3.compose(s2).compose(s1)
                    final_type = final_subst.apply(result_type)

                    return final_type, final_subst

                case (
                    "add"
                    | "sub"
                    | "mul"
                    | "div"
                    | "pow_int"
                    | "pow_float"
                    | "mod"
                    | "quot"
                ):
                    # Binary arithmetic operations
                    left_type, s1 = self.infer_expr(expr.children[0], env)
                    right_type, s2 = self.infer_expr(
                        expr.children[2],
                        env.substitute(s1),
                    )

                    if expr.data == "div":
                        # Division always returns float
                        # Both operands can be Int or Float, but result is always Float
                        left_after_s2 = s2.apply(left_type)

                        # Try to unify each operand with either Int or Float
                        try:
                            # Try Int first for left operand
                            s3_left = unify([(left_after_s2, INT_TYPE)])
                        except UnificationError:
                            try:
                                # If Int fails, try Float
                                s3_left = unify([(left_after_s2, FLOAT_TYPE)])
                            except UnificationError:
                                raise UnificationError(
                                    f"Left operand of division must be numeric, got {left_after_s2}",
                                )

                        try:
                            # Try Int first for right operand
                            s3_right = unify([(right_type, INT_TYPE)])
                        except UnificationError:
                            try:
                                # If Int fails, try Float
                                s3_right = unify([(right_type, FLOAT_TYPE)])
                            except UnificationError:
                                raise UnificationError(
                                    f"Right operand of division must be numeric, got {right_type}",
                                )

                        # Compose all substitutions
                        final_subst = s3_right.compose(s3_left).compose(s2).compose(s1)
                        return FLOAT_TYPE, final_subst
                    else:
                        # Other operations preserve type (Int or Float)
                        result_type = self.fresh_type_var()
                        s3 = unify(
                            [
                                (s2.apply(left_type), result_type),
                                (right_type, result_type),
                            ],
                        )
                        final_subst = s3.compose(s2).compose(s1)
                        final_type = final_subst.apply(result_type)
                        return final_type, final_subst

                case "eq" | "neq" | "lt" | "lteq" | "gt" | "gteq":
                    # Comparison operations
                    left_type, s1 = self.infer_expr(expr.children[0], env)
                    right_type, s2 = self.infer_expr(
                        expr.children[2],
                        env.substitute(s1),
                    )

                    # Both arguments must have the same type
                    s3 = unify([(s2.apply(left_type), right_type)])
                    final_subst = s3.compose(s2).compose(s1)

                    return BOOL_TYPE, final_subst

                case "and" | "or":
                    # Logical operations
                    left_type, s1 = self.infer_expr(expr.children[0], env)
                    right_type, s2 = self.infer_expr(
                        expr.children[2],
                        env.substitute(s1),
                    )

                    s3 = unify(
                        [(s2.apply(left_type), BOOL_TYPE), (right_type, BOOL_TYPE)],
                    )
                    final_subst = s3.compose(s2).compose(s1)

                    return BOOL_TYPE, final_subst

                case "not":
                    # Unary not
                    operand_type, s1 = self.infer_expr(expr.children[1], env)
                    s2 = unify([(operand_type, BOOL_TYPE)])
                    final_subst = s2.compose(s1)

                    return BOOL_TYPE, final_subst

                case "concat":
                    # Concatenation: can be string ++ string or list ++ list
                    left_type, s1 = self.infer_expr(expr.children[0], env)
                    right_type, s2 = self.infer_expr(
                        expr.children[1],
                        env.substitute(s1),
                    )

                    # Try string concatenation first
                    try:
                        s3 = unify(
                            [
                                (s2.apply(left_type), STRING_TYPE),
                                (right_type, STRING_TYPE),
                            ],
                        )
                        final_subst = s3.compose(s2).compose(s1)
                        return STRING_TYPE, final_subst
                    except UnificationError:
                        # Try list concatenation
                        element_type = self.fresh_type_var()
                        list_type_left = TypeApp(TypeCon("List"), element_type)
                        list_type_right = TypeApp(TypeCon("List"), element_type)

                        s3 = unify(
                            [
                                (s2.apply(left_type), list_type_left),
                                (right_type, list_type_right),
                            ],
                        )
                        final_subst = s3.compose(s2).compose(s1)
                        result_list_type = TypeApp(
                            TypeCon("List"),
                            final_subst.apply(element_type),
                        )
                        return result_list_type, final_subst

                case "index":
                    # List indexing: list !! int -> element_type
                    list_expr_type, s1 = self.infer_expr(expr.children[0], env)
                    index_type, s2 = self.infer_expr(
                        expr.children[1],
                        env.substitute(s1),
                    )

                    # Index must be Int
                    s3 = unify([(index_type, INT_TYPE)])

                    # List expression must be List element_type
                    element_type = self.fresh_type_var()
                    expected_list_type = TypeApp(TypeCon("List"), element_type)
                    s4 = unify(
                        [(s3.compose(s2).apply(list_expr_type), expected_list_type)],
                    )

                    final_subst = s4.compose(s3).compose(s2).compose(s1)
                    result_type = final_subst.apply(element_type)
                    return result_type, final_subst

                case "constructor_expr":
                    # Record construction: Constructor { field = value, ... }
                    ctor_name_token = expr.children[0]
                    ctor_name = (
                        ctor_name_token.value
                        if isinstance(ctor_name_token, Token)
                        else str(ctor_name_token)
                    )

                    # Get constructor type scheme
                    scheme = env.lookup(ctor_name)
                    if scheme is None:
                        raise TypeInferenceError(f"Unknown constructor: {ctor_name}")

                    # Instantiate the constructor type
                    ctor_type = scheme.instantiate(self.fresh_var_gen)

                    # Process field assignments
                    current_type = ctor_type
                    current_subst = TypeSubstitution()

                    # Apply constructor to each field value
                    for field_assign in expr.children[1:]:
                        if field_assign.data == "field_assign":
                            # Get field value expression
                            field_value = field_assign.children[1]

                            # Infer type of field value
                            value_type, s1 = self.infer_expr(
                                field_value,
                                env.substitute(current_subst),
                            )

                            # If current_type is a function type, apply it
                            if isinstance(
                                current_subst.apply(current_type),
                                FunctionType,
                            ):
                                func_type = current_subst.apply(current_type)
                                result_type = self.fresh_type_var()
                                expected_func_type = FunctionType(
                                    value_type,
                                    result_type,
                                )

                                # Unify function type with expected type
                                s2 = unify([(func_type, expected_func_type)])

                                current_subst = s2.compose(s1).compose(current_subst)
                                current_type = current_subst.apply(result_type)
                            else:
                                raise TypeInferenceError(
                                    f"Constructor {ctor_name} applied to too many arguments",
                                )

                    # Return the final constructed type
                    final_type = current_subst.apply(current_type)
                    return final_type, current_subst

                case "grouped":
                    # Parenthesized expression
                    return self.infer_expr(expr.children[0], env)

                case "do_block":
                    # Do block - infer type of last statement
                    stmt_list = expr.children[0]
                    return self.infer_stmt_list(stmt_list, env)

                case "if_else":
                    # If-else expression: if condition then expr1 else expr2
                    # condition must be Bool, expr1 and expr2 must have the same type
                    cond_type, s1 = self.infer_expr(expr.children[0], env)
                    then_type, s2 = self.infer_expr(
                        expr.children[1],
                        env.substitute(s1),
                    )
                    else_type, s3 = self.infer_expr(
                        expr.children[2],
                        env.substitute(s2.compose(s1)),
                    )

                    # Unify condition with Bool type
                    s4 = unify([(s3.compose(s2).apply(cond_type), BOOL_TYPE)])

                    # Unify then and else branches to have the same type
                    s5 = unify([(s4.compose(s3).apply(then_type), s4.apply(else_type))])

                    final_subst = s5.compose(s4).compose(s3).compose(s2).compose(s1)
                    result_type = final_subst.apply(then_type)

                    return result_type, final_subst

                case _:
                    raise TypeInferenceError(f"Unhandled expression: {expr.data}")

        elif isinstance(expr, Token):
            # This might be a variable reference
            var_name = expr.value
            scheme = env.lookup(var_name)
            if scheme is None:
                raise TypeInferenceError(f"Unknown variable: {var_name}")

            typ = scheme.instantiate(self.fresh_var_gen)
            return typ, TypeSubstitution()

        else:
            raise TypeInferenceError(f"Unknown expression type: {type(expr)}")

    def infer_stmt_list(self, stmt_list: Tree, env: TypeEnvironment) -> InferenceResult:
        """Infer type of a statement list (do block)."""
        current_env = env
        current_subst = TypeSubstitution()
        result_type = UNIT_TYPE

        for stmt in stmt_list.children:
            if isinstance(stmt, Tree) and stmt.data == "do_stmt":
                if len(stmt.children) >= 2 and len(stmt.children) % 2 == 0:
                    # Let block with multiple assignments
                    # Structure: var1, value1, var2, value2, ..., varN, valueN

                    # Process all variable-value pairs
                    for i in range(0, len(stmt.children), 2):
                        var_name_token = stmt.children[i]
                        var_name = (
                            var_name_token.value
                            if isinstance(var_name_token, Token)
                            else str(var_name_token)
                        )

                        # Infer type of the value expression
                        expr_type, s1 = self.infer_expr(
                            stmt.children[i + 1],
                            current_env.substitute(current_subst),
                        )

                        # Generalize the type
                        env_free_vars = current_env.substitute(
                            current_subst.compose(s1),
                        ).free_vars()
                        scheme = generalize(env_free_vars, s1.apply(expr_type))

                        # Extend current environment (persists for rest of do block)
                        current_env = current_env.extend(var_name, scheme)
                        current_subst = s1.compose(current_subst)

                    # The result of a let statement is Unit
                    result_type = UNIT_TYPE
                else:
                    # Expression statement
                    result_type, s1 = self.infer_expr(
                        stmt.children[0],
                        current_env.substitute(current_subst),
                    )
                    current_subst = s1.compose(current_subst)
            else:
                # Regular expression
                result_type, s1 = self.infer_expr(
                    stmt,
                    current_env.substitute(current_subst),
                )
                current_subst = s1.compose(current_subst)

        return current_subst.apply(result_type), current_subst

    def infer_function(
        self,
        func_def: Tree,
        env: TypeEnvironment,
    ) -> Tuple[str, TypeScheme]:
        """Infer the type of a function definition."""
        func_name_token = func_def.children[0]
        func_name = (
            func_name_token.value
            if isinstance(func_name_token, Token)
            else str(func_name_token)
        )
        patterns = func_def.children[1:-1]
        body_expr = func_def.children[-1]

        # Create fresh type variables for each parameter
        param_types = []
        extended_env = env

        for pattern in patterns:
            if isinstance(pattern, Token):
                # Simple variable pattern
                param_type = self.fresh_type_var()
                param_types.append(param_type)

                var_name = pattern.value
                scheme = TypeScheme(set(), param_type)
                extended_env = extended_env.extend(var_name, scheme)
            elif isinstance(pattern, Tree) and pattern.data == "constructor_pattern":
                # Constructor pattern like (Person id name)
                ctor_name_token = pattern.children[0]
                ctor_name = (
                    ctor_name_token.value
                    if isinstance(ctor_name_token, Token)
                    else str(ctor_name_token)
                )

                # Look up constructor type
                scheme = extended_env.lookup(ctor_name)
                if scheme is None:
                    raise TypeInferenceError(f"Unknown constructor: {ctor_name}")

                ctor_type = scheme.instantiate(self.fresh_var_gen)

                # Extract field types and result type
                field_types = self._extract_constructor_field_types(ctor_type)
                result_type = self._extract_constructor_result_type(ctor_type)

                # The parameter type is the data type that this constructor creates
                param_types.append(result_type)

                # Bind pattern variables to their respective field types
                pattern_vars = pattern.children[1:]  # Skip constructor name
                if len(pattern_vars) != len(field_types):
                    raise TypeInferenceError(
                        f"Constructor {ctor_name} expects {len(field_types)} arguments, "
                        f"but pattern has {len(pattern_vars)}",
                    )

                for var_token, field_type in zip(pattern_vars, field_types):
                    var_name = (
                        var_token.value
                        if isinstance(var_token, Token)
                        else str(var_token)
                    )
                    var_scheme = TypeScheme(set(), field_type)
                    extended_env = extended_env.extend(var_name, var_scheme)
            else:
                # Other patterns - create fresh type variable
                param_type = self.fresh_type_var()
                param_types.append(param_type)

        # Infer body type
        body_type, subst = self.infer_expr(body_expr, extended_env)

        # Build function type
        result_type = subst.apply(body_type)
        for param_type in reversed(param_types):
            result_type = FunctionType(subst.apply(param_type), result_type)

        # Generalize
        env_free_vars = env.substitute(subst).free_vars()
        scheme = generalize(env_free_vars, result_type)

        return func_name, scheme

    def _extract_constructor_result_type(self, ctor_type: Type) -> Type:
        """Extract the result type from a constructor type (the data type it constructs)"""
        if isinstance(ctor_type, FunctionType):
            # For constructor types like Int -> String -> Person,
            # we need to go to the rightmost type (Person)
            return self._extract_constructor_result_type(ctor_type.result)
        else:
            # This should be the data type
            return ctor_type

    def _extract_constructor_field_types(self, ctor_type: Type) -> List[Type]:
        """Extract the field types from a constructor type"""
        field_types = []
        current_type = ctor_type

        while isinstance(current_type, FunctionType):
            field_types.append(current_type.param)
            current_type = current_type.result

        return field_types

    def infer_program(self, tree: Tree) -> TypeEnvironment:
        """Infer types for an entire program"""
        env = self.builtin_env

        # First pass: collect data type declarations
        for stmt in tree.children:
            if isinstance(stmt, Tree) and stmt.data == "data_decl":
                data_env = self.infer_data_decl(stmt)
                # Merge data constructor types into environment
                for name, scheme in data_env.bindings.items():
                    env = env.extend(name, scheme)

        # Second pass: collect function signatures
        func_signatures = {}
        for stmt in tree.children:
            if isinstance(stmt, Tree) and stmt.data == "func_sig":
                func_name_token = stmt.children[0]
                func_name = (
                    func_name_token.value
                    if isinstance(func_name_token, Token)
                    else str(func_name_token)
                )
                # Parse the function signature
                type_exprs = stmt.children[1:]

                # Build function type from signature
                if len(type_exprs) == 1:
                    func_type = self.parse_type_expr(type_exprs[0])
                else:
                    func_type = self.parse_type_expr(type_exprs[-1])
                    for type_expr in reversed(type_exprs[:-1]):
                        param_type = self.parse_type_expr(type_expr)
                        func_type = FunctionType(param_type, func_type)

                # Generalize the function type - all free type variables become bound
                free_vars = func_type.free_vars()
                func_signatures[func_name] = TypeScheme(free_vars, func_type)

        # Add signatures to environment
        for name, scheme in func_signatures.items():
            env = env.extend(name, scheme)

        # Third pass: infer function definitions
        for stmt in tree.children:
            if isinstance(stmt, Tree) and stmt.data == "func_def":
                func_name, inferred_scheme = self.infer_function(stmt, env)

                # Check against declared signature if present
                if func_name in func_signatures:
                    declared_scheme = func_signatures[func_name]
                    # TODO: Check that inferred type matches declared type
                    # For now, use the declared type
                    env = env.extend(func_name, declared_scheme)
                else:
                    env = env.extend(func_name, inferred_scheme)

        return env

    def check_program(self, tree: Tree) -> TypeBindings:
        """Type check a program and return the final type environment."""
        try:
            env = self.infer_program(tree)
            return env.bindings
        except (UnificationError, TypeInferenceError) as e:
            raise TypeInferenceError(f"Type checking failed: {e}")


def type_check(tree: Tree) -> TypeBindings:
    """Main entry point for type checking."""
    inferrer = TypeInferrer()
    return inferrer.check_program(tree)
