"""
Main type inference engine using the Hindley-Milner algorithm
"""

from typing import Dict, List, Optional, Set, Tuple

from lark import Token, Tree

from .types import (
    BOOL_TYPE,
    FLOAT_TYPE,
    INT_TYPE,
    STRING_TYPE,
    UNIT_TYPE,
    DataType,
    FreshVarGenerator,
    FunctionType,
    Type,
    TypeScheme,
    TypeSubstitution,
    TypeVar,
    generalize,
)
from .unify import UnificationError, unify


class TypeEnvironment:
    """Type environment mapping identifiers to type schemes"""

    def __init__(self, bindings: Optional[Dict[str, TypeScheme]] = None):
        self.bindings = bindings or {}

    def lookup(self, name: str) -> Optional[TypeScheme]:
        return self.bindings.get(name)

    def extend(self, name: str, scheme: TypeScheme) -> "TypeEnvironment":
        """Return a new environment with an additional binding"""
        new_bindings = self.bindings.copy()
        new_bindings[name] = scheme
        return TypeEnvironment(new_bindings)

    def free_vars(self) -> Set[str]:
        """Return free variables in all type schemes in this environment"""
        result = set()
        for scheme in self.bindings.values():
            result |= scheme.free_vars()
        return result

    def substitute(self, subst: TypeSubstitution) -> "TypeEnvironment":
        """Apply a substitution to all type schemes in the environment"""
        new_bindings = {}
        for name, scheme in self.bindings.items():
            new_bindings[name] = scheme.substitute(subst)
        return TypeEnvironment(new_bindings)


class TypeInferenceError(Exception):
    """Raised when type inference fails"""

    pass


class TypeInferrer:
    """Hindley-Milner type inference engine"""

    def __init__(self):
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
        # putStr and putStrLn can work with strings or functions that return strings
        a = TypeVar("a")

        # Method 1: Give putStr/putStrLn overloaded types
        # For now, let's assume that putStr gets specialized based on usage
        put_str_string = TypeScheme(set(), FunctionType(STRING_TYPE, UNIT_TYPE))
        put_str_func = TypeScheme(
            {"a"},
            FunctionType(FunctionType(a, STRING_TYPE), FunctionType(a, UNIT_TYPE)),
        )

        put_str_ln = TypeScheme(set(), FunctionType(STRING_TYPE, UNIT_TYPE))
        put_str = put_str_string  # Start with string version, may need to be more sophisticated

        get_line = TypeScheme(set(), STRING_TYPE)
        read_int = TypeScheme(set(), INT_TYPE)
        read_string = TypeScheme(set(), STRING_TYPE)
        read_bool = TypeScheme(set(), BOOL_TYPE)
        show = TypeScheme({"a"}, FunctionType(a, STRING_TYPE))

        # String functions
        to_upper_case = TypeScheme(set(), FunctionType(STRING_TYPE, STRING_TYPE))
        to_lower_case = TypeScheme(set(), FunctionType(STRING_TYPE, STRING_TYPE))

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
            "toUpperCase": to_upper_case,
            "toLowerCase": to_lower_case,
            # IO
            "putStrLn": put_str_ln,
            "putStr": put_str,
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

    def add_constraint(self, t1: Type, t2: Type):
        """Add a type constraint"""
        self.constraints.append((t1, t2))

    def infer_data_decl(self, node: Tree) -> TypeEnvironment:
        """Process a data type declaration and return updated environment"""
        # data TYPE = constructor (| constructor)*
        type_name_token = node.children[0]
        type_name = (
            type_name_token.value
            if isinstance(type_name_token, Token)
            else str(type_name_token)
        )
        constructors = node.children[1:]

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
                    result_type = DataType(type_name, [])
                    ctor_type = result_type
                    for field_type in reversed(field_types):
                        ctor_type = FunctionType(field_type, ctor_type)

                    constructor_types[ctor_name] = TypeScheme(set(), ctor_type)
                    self.data_constructors[ctor_name] = (type_name, field_types)

                else:
                    # Positional constructor: UIDENT type_expr*
                    if len(constructor.children) == 1:
                        # Nullary constructor
                        ctor_type = TypeScheme(set(), DataType(type_name, []))
                    else:
                        # Constructor with arguments
                        field_types = []
                        for type_expr in constructor.children[1:]:
                            field_type = self.parse_type_expr(type_expr)
                            field_types.append(field_type)

                        # Create function type: field1 -> field2 -> ... -> DataType
                        result_type = DataType(type_name, [])
                        ctor_type = result_type
                        for field_type in reversed(field_types):
                            ctor_type = FunctionType(field_type, ctor_type)

                        ctor_type = TypeScheme(set(), ctor_type)

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

    def parse_type_expr(self, node) -> Type:
        """Parse a type expression from the AST"""
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
            if node.data == "type_expr":
                # This might be a function type: TYPE -> type_expr
                if len(node.children) == 1:
                    return self.parse_type_expr(node.children[0])
                else:
                    # Function type
                    param_type = self.parse_type_expr(node.children[0])
                    result_type = self.parse_type_expr(node.children[1])
                    return FunctionType(param_type, result_type)

        raise TypeInferenceError(f"Cannot parse type expression: {node}")

    def infer_expr(self, expr, env: TypeEnvironment) -> Tuple[Type, TypeSubstitution]:
        """Infer the type of an expression"""
        if isinstance(expr, Tree):
            match expr.data:
                case "int":
                    return INT_TYPE, TypeSubstitution()

                case "float":
                    return FLOAT_TYPE, TypeSubstitution()

                case "string":
                    return STRING_TYPE, TypeSubstitution()

                case "true" | "false":
                    return BOOL_TYPE, TypeSubstitution()

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

                    # Special case: putStr show -> give it the curried type
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

                    # Special case: putStrLn show -> give it the curried type
                    if (
                        isinstance(func_expr, Tree)
                        and func_expr.data == "var"
                        and isinstance(func_expr.children[0], Token)
                        and func_expr.children[0].value == "putStrLn"
                        and isinstance(arg_expr, Tree)
                        and arg_expr.data == "var"
                        and isinstance(arg_expr.children[0], Token)
                        and arg_expr.children[0].value == "show"
                    ):

                        # putStrLn show has type: a -> () where a is showable
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

                case "add" | "sub" | "mul" | "div" | "pow" | "mod" | "quot":
                    # Binary arithmetic operations
                    left_type, s1 = self.infer_expr(expr.children[0], env)
                    right_type, s2 = self.infer_expr(
                        expr.children[2],
                        env.substitute(s1),
                    )

                    if expr.data == "div":
                        # Division always returns float
                        s3 = unify(
                            [
                                (s2.apply(left_type), FLOAT_TYPE),
                                (right_type, FLOAT_TYPE),
                            ],
                        )
                        final_subst = s3.compose(s2).compose(s1)
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
                    # String concatenation
                    left_type, s1 = self.infer_expr(expr.children[0], env)
                    right_type, s2 = self.infer_expr(
                        expr.children[1],
                        env.substitute(s1),
                    )

                    s3 = unify(
                        [(s2.apply(left_type), STRING_TYPE), (right_type, STRING_TYPE)],
                    )
                    final_subst = s3.compose(s2).compose(s1)

                    return STRING_TYPE, final_subst

                case "constructor_expr":
                    # Record construction: Constructor { field = value, ... }
                    ctor_name_token = expr.children[0]
                    ctor_name = (
                        ctor_name_token.value
                        if isinstance(ctor_name_token, Token)
                        else str(ctor_name_token)
                    )

                    # Get constructor type
                    scheme = env.lookup(ctor_name)
                    if scheme is None and ctor_name in self.data_constructors:
                        data_type_name, field_types = self.data_constructors[ctor_name]
                        result_type = DataType(data_type_name, [])
                        scheme = TypeScheme(set(), result_type)

                    if scheme is None:
                        raise TypeInferenceError(f"Unknown constructor: {ctor_name}")

                    # For now, just return the data type
                    # TODO: Check field assignments
                    typ = scheme.instantiate(self.fresh_var_gen)
                    return typ, TypeSubstitution()

                case "grouped":
                    # Parenthesized expression
                    return self.infer_expr(expr.children[0], env)

                case "do_block":
                    # Do block - infer type of last statement
                    stmt_list = expr.children[0]
                    return self.infer_stmt_list(stmt_list, env)

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

    def infer_stmt_list(
        self,
        stmt_list: Tree,
        env: TypeEnvironment,
    ) -> Tuple[Type, TypeSubstitution]:
        """Infer type of a statement list (do block)"""
        current_env = env
        current_subst = TypeSubstitution()
        result_type = UNIT_TYPE

        for stmt in stmt_list.children:
            if isinstance(stmt, Tree) and stmt.data == "do_stmt":
                if len(stmt.children) == 2:
                    # Let statement: let var = expr
                    var_name_token = stmt.children[0]
                    var_name = (
                        var_name_token.value
                        if isinstance(var_name_token, Token)
                        else str(var_name_token)
                    )

                    expr_type, s1 = self.infer_expr(
                        stmt.children[1],
                        current_env.substitute(current_subst),
                    )

                    # Generalize the type
                    env_free_vars = current_env.substitute(
                        current_subst.compose(s1),
                    ).free_vars()
                    scheme = generalize(env_free_vars, s1.apply(expr_type))

                    # Extend environment
                    current_env = current_env.extend(var_name, scheme)
                    current_subst = s1.compose(current_subst)
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
        """Infer the type of a function definition"""
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

                func_signatures[func_name] = TypeScheme(set(), func_type)

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

    def check_program(self, tree: Tree) -> Dict[str, TypeScheme]:
        """Type check a program and return the final type environment"""
        try:
            env = self.infer_program(tree)
            return env.bindings
        except (UnificationError, TypeInferenceError) as e:
            raise TypeInferenceError(f"Type checking failed: {e}")


def type_check(tree: Tree) -> Dict[str, TypeScheme]:
    """Main entry point for type checking"""
    inferrer = TypeInferrer()
    return inferrer.check_program(tree)
