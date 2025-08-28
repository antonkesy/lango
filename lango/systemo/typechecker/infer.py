from collections import defaultdict
from typing import Dict, ItemsView, List, Optional, Set, Tuple

from lango.systemo.ast.nodes import (
    ArrowType,
    Associativity,
    ASTNode,
    BoolLiteral,
    CharLiteral,
    ConsPattern,
    Constructor,
    ConstructorExpression,
    ConstructorPattern,
    DataDeclaration,
    DoBlock,
    Expression,
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
    ListType,
    LiteralPattern,
    NegativeFloat,
    NegativeInt,
    Pattern,
    PrecedenceDeclaration,
    Program,
    Statement,
    StringLiteral,
    SymbolicOperation,
    TupleLiteral,
    TuplePattern,
)
from lango.systemo.ast.nodes import TupleType as ASTTupleType
from lango.systemo.ast.nodes import (
    TypeApplication,
    TypeConstructor,
    TypeExpression,
    TypeVariable,
    Variable,
    VariablePattern,
)
from lango.systemo.typechecker.systemo_types import (
    BOOL_TYPE,
    FLOAT_TYPE,
    INT_TYPE,
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
    generalize,
)
from lango.systemo.typechecker.unify import UnificationError, unify_one

TypeBindings = Dict[str, TypeScheme]
InferenceResult = Tuple[Type, TypeSubstitution]


class TypeEnvironment:
    def __init__(self, bindings: Optional[TypeBindings] = None) -> None:
        self.bindings: TypeBindings = bindings or {}

    def lookup(self, name: str) -> Optional[TypeScheme]:
        return self.bindings.get(name)

    def extend(self, name: str, scheme: TypeScheme) -> "TypeEnvironment":
        new_bindings = self.bindings.copy()
        new_bindings[name] = scheme
        return TypeEnvironment(new_bindings)

    def extend_many(self, new_bindings: Dict[str, TypeScheme]) -> "TypeEnvironment":
        combined_bindings = self.bindings.copy()
        combined_bindings.update(new_bindings)
        return TypeEnvironment(combined_bindings)

    def apply_substitution(self, subst: TypeSubstitution) -> "TypeEnvironment":
        new_bindings = {}
        for name, scheme in self.bindings.items():
            # Apply substitution to the scheme's type, keeping quantified vars
            applied_type = subst.apply(scheme.type)
            new_bindings[name] = TypeScheme(scheme.quantified_vars, applied_type)
        return TypeEnvironment(new_bindings)

    def free_type_vars(self) -> Set[str]:
        free_vars = set()
        for scheme in self.bindings.values():
            free_vars.update(scheme.free_vars())
        return free_vars

    def items(self) -> ItemsView[str, TypeScheme]:
        return self.bindings.items()

    def __contains__(self, name: str) -> bool:
        return name in self.bindings

    def __getitem__(self, name: str) -> TypeScheme:
        return self.bindings[name]


class TypeInferenceError(Exception):
    def __init__(self, message: str, node: Optional[ASTNode] = None) -> None:
        self.message = message
        self.node = node
        super().__init__(message)


class TypeInferrer:
    def __init__(self) -> None:
        self.fresh_var_gen = FreshVarGenerator()
        self.data_types: Dict[str, List[str]] = {}  # type_name -> [constructor_names]
        self.data_constructors: Dict[str, Tuple[str, List[Type]]] = (
            {}
        )  # constructor -> (type_name, field_types)
        self.instances: Dict[str, List[Tuple[Type, FunctionDefinition]]] = (
            {}
        )  # instance_name -> [(type, func_def)]
        self.precedences: Dict[str, Tuple[int, Associativity]] = (
            {}
        )  # operator -> (precedence, associativity)

    def fresh_type_var(self) -> TypeVar:
        var_name = self.fresh_var_gen.fresh()
        return TypeVar(var_name)

    def handle_precedence_decl(self, decl: PrecedenceDeclaration) -> None:
        """Store precedence declaration for an operator."""
        self.precedences[decl.operator] = (decl.precedence, decl.associativity)

    def infer_data_decl(self, node: DataDeclaration) -> TypeEnvironment:
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
                    self.data_constructors[ctor_name] = (type_name, [])
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
                    self.data_constructors[ctor_name] = (type_name, field_types)

        self.data_types[type_name] = constructor_names

        # Return environment extended with constructor types
        env = TypeEnvironment()
        for name, scheme in constructor_types.items():
            env = env.extend(name, scheme)

        return env

    def parse_type_expr(self, node: TypeExpression) -> Type:
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
            case ListType(element_type=element_type):
                element_type_parsed = self.parse_type_expr(element_type)
                return TypeApp(TypeCon("List"), element_type_parsed)
            case ASTTupleType(element_types=element_types):
                element_types_parsed = [
                    self.parse_type_expr(elem_type) for elem_type in element_types
                ]
                return TupleType(element_types_parsed)
            case GroupedType(type_expr=type_expr):
                return self.parse_type_expr(type_expr)
            case _:
                raise TypeInferenceError(
                    f"Cannot parse type expression: {type(node).__name__}",
                )

    def handle_instance_decl(self, inst_decl: InstanceDeclaration) -> None:
        """Store instance declaration and validate it against implementation"""
        instance_name = inst_decl.instance_name
        declared_type = self.parse_type_expr(inst_decl.type_signature)
        function_definition = inst_decl.function_definition

        # Validate that the implementation matches the declared type
        inferred_type = None
        try:
            # Create a temporary environment for validation
            temp_env = TypeEnvironment()

            # Infer the actual type of the function implementation
            inferred_scheme, _ = self.infer_function(function_definition, temp_env)
            inferred_type = inferred_scheme.type

            # Try to unify the declared type with the inferred type
            from lango.systemo.typechecker.unify import unify_one

            unify_one(declared_type, inferred_type)

        except Exception as e:
            # If we failed to infer the type, use a placeholder
            inferred_type_str = str(inferred_type) if inferred_type else "unknown"

            raise TypeInferenceError(
                f"Instance declaration for {instance_name} has type mismatch: "
                f"declared {declared_type} but implementation has type {inferred_type_str}. "
                f"Unification failed: {e}",
            )

        if instance_name not in self.instances:
            self.instances[instance_name] = []

        self.instances[instance_name].append((declared_type, function_definition))

    def _validate_instance_basic_mismatch(
        self,
        instance_name: str,
        declared_type: Type,
        function_definition: FunctionDefinition,
        env: TypeEnvironment,
    ) -> None:
        """Perform basic validation for obvious type mismatches in instance declarations"""
        # Check for the specific pattern where we have a simple pattern match
        # that returns a constructor field with the wrong type

        # Only validate for simple cases to avoid breaking complex instances
        if len(function_definition.patterns) == 1 and isinstance(
            function_definition.body,
            Variable,
        ):

            pattern = function_definition.patterns[0]
            body = function_definition.body

            # Check if it's a constructor pattern returning a field
            from lango.systemo.ast.nodes import ConstructorPattern

            if isinstance(pattern, ConstructorPattern) and isinstance(body, Variable):
                constructor_name = pattern.constructor
                if constructor_name in self.data_constructors:
                    _, field_types = self.data_constructors[constructor_name]

                    # Find which field is being returned
                    for i, pattern_param in enumerate(pattern.patterns):
                        from lango.systemo.ast.nodes import VariablePattern

                        if (
                            isinstance(pattern_param, VariablePattern)
                            and pattern_param.name == body.name
                        ):
                            # Found the field being returned - check bounds
                            if i < len(field_types):
                                actual_field_type = field_types[i]

                                # Extract the expected return type from declaration
                                expected_return_type = self._extract_return_type(
                                    declared_type,
                                )

                                # Check for obvious mismatches (Int vs Float)
                                if (
                                    str(expected_return_type) == "Int"
                                    and str(actual_field_type) == "Float"
                                ):
                                    raise TypeInferenceError(
                                        f"Instance declaration for {instance_name} has type mismatch: "
                                        f"declared to return {expected_return_type} but implementation "
                                        f"returns field of type {actual_field_type}",
                                    )
                                elif (
                                    str(expected_return_type) == "Float"
                                    and str(actual_field_type) == "Int"
                                ):
                                    raise TypeInferenceError(
                                        f"Instance declaration for {instance_name} has type mismatch: "
                                        f"declared to return {expected_return_type} but implementation "
                                        f"returns field of type {actual_field_type}",
                                    )
                            break

    def _extract_return_type(self, func_type: Type) -> Type:
        """Extract the return type from a function type"""
        match func_type:
            case FunctionType(result=result_type):
                return self._extract_return_type(result_type)
            case _:
                return func_type

    def _extract_operator_name(self, instance_name: str) -> str:
        """Extract operator name from Tree structure like "Tree(Token('RULE', 'inst_operator_name'), ['?'])" or "Tree(Token('RULE', 'inst_operator_name'), [Token('ID', 'xcoord')])" """
        import re

        # Look for pattern Tree(Token('RULE', 'inst_operator_name'), ['<operator>'])
        match = re.search(
            r"Tree\(Token\('RULE', 'inst_operator_name'\), \['([^']*)'\]\)",
            instance_name,
        )
        if match:
            return match.group(1)

        # Look for pattern Tree(Token('RULE', 'inst_operator_name'), [Token('ID', '<operator>')])
        match = re.search(
            r"Tree\(Token\('RULE', 'inst_operator_name'\), \[Token\('ID', '([^']*)'\)\]\)",
            instance_name,
        )
        if match:
            return match.group(1)

        # Fallback: return the instance_name as is
        return instance_name

    def handle_instance_decl_with_env(
        self,
        inst_decl: InstanceDeclaration,
        env: TypeEnvironment,
    ) -> None:
        """Store instance declaration and validate it against implementation with full environment"""
        raw_instance_name = inst_decl.instance_name
        # Extract the actual operator name from the Tree structure
        instance_name = self._extract_operator_name(raw_instance_name)
        declared_type = self.parse_type_expr(inst_decl.type_signature)
        function_definition = inst_decl.function_definition

        # Basic structural validation: check for obvious type constructor mismatches
        # We'll do a simple check for the specific case where the function body
        # is a direct pattern variable that has a different type than declared
        self._validate_instance_basic_mismatch(
            instance_name,
            declared_type,
            function_definition,
            env,
        )

        if instance_name not in self.instances:
            self.instances[instance_name] = []

        self.instances[instance_name].append((declared_type, function_definition))

    def resolve_overloaded_function(
        self,
        name: str,
        arg_type: Type,
        env: TypeEnvironment,
        expected_arity: Optional[int] = None,
    ) -> Optional[TypeScheme]:
        """Try to resolve an overloaded function based on argument type and expected arity"""
        if name not in self.instances:
            return None

        # If the argument type is a type variable, we don't have enough information to resolve overloading
        # Let the normal type inference handle it and constrain the type variable later
        if isinstance(arg_type, TypeVar):
            return None

        def count_function_arity(func_type: Type) -> int:
            """Count how many parameters a function type takes"""
            arity = 0
            current_type = func_type
            while isinstance(current_type, FunctionType):
                arity += 1
                current_type = current_type.result
            return arity

        # If we have an expected arity, prioritize functions matching that arity
        if expected_arity is not None:
            for instance_type, func_def in self.instances[name]:
                match instance_type:
                    case FunctionType(param=param_type, result=result_type):
                        # Check if arity matches expected
                        if count_function_arity(instance_type) == expected_arity:
                            try:
                                # Try to unify the parameter type with the argument type
                                unify_one(param_type, arg_type)
                                # If unification succeeds, return this instance's type
                                return TypeScheme(set(), instance_type)
                            except UnificationError:
                                continue
                    case _:
                        continue

        # Fallback: First pass - prefer multi-parameter functions that can be partially applied
        for instance_type, func_def in self.instances[name]:
            match instance_type:
                case FunctionType(param=param_type, result=result_type):
                    # Only consider multi-parameter functions in first pass
                    if isinstance(result_type, FunctionType):
                        try:
                            # Try to unify the parameter type with the argument type
                            unify_one(param_type, arg_type)
                            # If unification succeeds, return this instance's type
                            return TypeScheme(set(), instance_type)
                        except UnificationError:
                            continue
                case _:
                    continue

        # Second pass: if no multi-parameter function found, try unary functions
        for instance_type, func_def in self.instances[name]:
            match instance_type:
                case FunctionType(param=param_type, result=result_type):
                    # Check if this is a unary function (result is not a function type)
                    if not isinstance(result_type, FunctionType):
                        try:
                            # Try to unify the parameter type with the argument type
                            unify_one(param_type, arg_type)
                            # If unification succeeds, return this instance's type
                            return TypeScheme(set(), instance_type)
                        except UnificationError:
                            continue
                case _:
                    continue

        return None

    def _infer_operator_application(
        self,
        func_app: FunctionApplication,
        env: TypeEnvironment,
        expected_arity: int,
    ) -> InferenceResult:
        """Helper method to infer operator applications with expected arity"""
        # First, infer the argument type
        arg_type, arg_subst = self.infer_expr(func_app.argument, env)

        # Initialize combined_subst and func_type
        combined_subst = arg_subst
        func_type = None

        # Check if this is potentially an overloaded function
        match func_app.function:
            case Variable(name=func_name) if func_name in self.instances:
                # Try to resolve overloaded function based on argument type and expected arity
                resolved_scheme = self.resolve_overloaded_function(
                    func_name,
                    arg_type.apply_substitution(arg_subst),
                    env,
                    expected_arity=expected_arity,
                )
                if resolved_scheme is not None:
                    func_type = resolved_scheme.instantiate(self.fresh_var_gen)
                else:
                    # Fallback to normal resolution
                    func_type, func_subst = self.infer_expr(func_app.function, env)
                    combined_subst = func_subst.compose(arg_subst)
            case _:
                # Normal function application
                func_type, func_subst = self.infer_expr(func_app.function, env)
                combined_subst = func_subst.compose(arg_subst)

        if func_type is None:
            raise TypeInferenceError(
                f"Could not resolve function type for {func_app.function}",
            )

        # Create fresh return type
        return_type = self.fresh_type_var()
        expected_func_type = FunctionType(
            arg_type.apply_substitution(combined_subst),
            return_type,
        )

        # Unify the function type with the expected function type
        try:
            func_unify = unify_one(
                func_type.apply_substitution(combined_subst),
                expected_func_type,
            )
            final_subst = combined_subst.compose(func_unify)
            final_return_type = return_type.apply_substitution(final_subst)

            # Set the expression type and return
            func_app.ty = final_return_type
            return final_return_type, final_subst

        except UnificationError as e:
            raise TypeInferenceError(
                f"Function application type mismatch: expected {expected_func_type}, got {func_type.apply_substitution(combined_subst)}",
            ) from e

    def infer_expr(self, expr: Expression, env: TypeEnvironment) -> InferenceResult:
        match expr:
            # Literals
            case IntLiteral() | NegativeInt():
                return INT_TYPE, TypeSubstitution()

            case FloatLiteral() | NegativeFloat():
                return FLOAT_TYPE, TypeSubstitution()

            case StringLiteral():
                return STRING_TYPE, TypeSubstitution()

            case CharLiteral():
                return STRING_TYPE, TypeSubstitution()  # For now, treat char as string

            case BoolLiteral():
                return BOOL_TYPE, TypeSubstitution()

            case ListLiteral(elements=elements):
                if not elements:
                    # Empty list: infer polymorphic list type
                    element_type = self.fresh_type_var()
                    list_type = TypeApp(TypeCon("List"), element_type)
                    expr.ty = list_type
                    return list_type, TypeSubstitution()

                # Non-empty list: infer element type from first element
                # and unify with all other elements
                first_type, subst1 = self.infer_expr(elements[0], env)
                current_subst = subst1

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
                        raise TypeInferenceError(
                            f"List elements have incompatible types: {e}",
                        )

                list_type = TypeApp(
                    TypeCon("List"),
                    first_type.apply_substitution(current_subst),
                )
                expr.ty = list_type
                return list_type, current_subst

            case TupleLiteral(elements=elements):
                if not elements:
                    # Empty tuple: unit type
                    tuple_type = TupleType([])
                    expr.ty = tuple_type
                    return tuple_type, TypeSubstitution()

                # Non-empty tuple: infer type of each element
                element_types = []
                current_subst = TypeSubstitution()

                for element in elements:
                    elem_type, elem_subst = self.infer_expr(
                        element,
                        env.apply_substitution(current_subst),
                    )
                    current_subst = current_subst.compose(elem_subst)
                    element_types.append(elem_type.apply_substitution(current_subst))

                tuple_type = TupleType(element_types)
                expr.ty = tuple_type
                return tuple_type, current_subst

            # Variables and constructors
            case Variable(name=var_name):
                scheme = env.lookup(var_name)
                if scheme is None:
                    # Check if this is an overloaded function as fallback
                    if var_name in self.instances:
                        # For overloaded functions, we can't resolve the type without context,
                        # so we'll let the function application handler deal with it
                        # Return a placeholder type that will be resolved during function application
                        placeholder_type = self.fresh_type_var()
                        expr.ty = placeholder_type
                        return placeholder_type, TypeSubstitution()
                    raise TypeInferenceError(f"Unknown variable: {var_name}")
                inferred_type = scheme.instantiate(self.fresh_var_gen)
                expr.ty = inferred_type
                return inferred_type, TypeSubstitution()

            case Constructor(name=constr_name):
                scheme = env.lookup(constr_name)
                if scheme is None:
                    raise TypeInferenceError(f"Unknown constructor: {constr_name}")
                inferred_type = scheme.instantiate(self.fresh_var_gen)
                expr.ty = inferred_type
                return inferred_type, TypeSubstitution()

            # Generic symbolic operations - convert to function application
            case SymbolicOperation(operator=operator, operands=operands):
                # Transform symbolic operation into function application for type checking
                operator_var = Variable(operator)
                if len(operands) == 1:
                    # Unary operation: f x
                    func_app = FunctionApplication(operator_var, operands[0])
                    # Pass expected arity through a custom inference
                    return self._infer_operator_application(
                        func_app,
                        env,
                        expected_arity=1,
                    )
                elif len(operands) == 2:
                    # Binary operation: ((f x) y)
                    partial_app = FunctionApplication(operator_var, operands[0])
                    full_app = FunctionApplication(partial_app, operands[1])
                    return self._infer_operator_application(
                        full_app,
                        env,
                        expected_arity=2,
                    )
                else:
                    raise TypeInferenceError(
                        f"Unsupported arity for operator {operator}: {len(operands)}",
                    )

            # Control flow
            case IfElse(
                condition=condition,
                then_expr=then_branch,
                else_expr=else_branch,
            ):
                cond_type, cond_subst = self.infer_expr(condition, env)

                # Condition must be Bool
                try:
                    bool_unify = unify_one(cond_type, BOOL_TYPE)
                    subst_after_cond = cond_subst.compose(bool_unify)
                except UnificationError:
                    raise TypeInferenceError(
                        f"If condition must be Bool, got {cond_type}",
                    )

                # Infer then branch
                then_type, then_subst = self.infer_expr(
                    then_branch,
                    env.apply_substitution(subst_after_cond),
                )
                subst_after_then = subst_after_cond.compose(then_subst)

                # Infer else branch
                else_type, else_subst = self.infer_expr(
                    else_branch,
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
                        f"If branches have incompatible types: {then_type.apply_substitution(subst_after_else)} vs {else_type}",
                    )

            # Function application
            case FunctionApplication(function=func_expr, argument=arg_expr):
                # First, infer the argument type
                arg_type, arg_subst = self.infer_expr(arg_expr, env)

                # Initialize combined_subst and func_type
                combined_subst = arg_subst
                func_type = None

                # Check if this is potentially an overloaded function
                match func_expr:
                    case Variable(name=func_name) if func_name in self.instances:
                        # Try to resolve overloaded function based on argument type
                        resolved_scheme = self.resolve_overloaded_function(
                            func_name,
                            arg_type.apply_substitution(arg_subst),
                            env,
                        )
                        if resolved_scheme is not None:
                            func_type = resolved_scheme.instantiate(self.fresh_var_gen)
                        else:
                            # Fallback to normal resolution
                            func_type, func_subst = self.infer_expr(func_expr, env)
                            combined_subst = func_subst.compose(arg_subst)
                    case _:
                        # Normal function application
                        func_type, func_subst = self.infer_expr(func_expr, env)
                        combined_subst = func_subst.compose(arg_subst)

                if func_type is None:
                    raise TypeInferenceError(
                        f"Could not resolve function type for {func_expr}",
                    )

                # Create fresh return type
                return_type = self.fresh_type_var()
                expected_func_type = FunctionType(
                    arg_type.apply_substitution(combined_subst),
                    return_type,
                )

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
            case GroupedExpression(expression=inner_expr):
                result = self.infer_expr(inner_expr, env)
                expr.ty = result[0]
                return result

            # Do blocks
            case DoBlock(statements=stmts):
                result = self.infer_do_block(stmts, env)
                expr.ty = result[0]
                return result

            # Constructor expressions
            case ConstructorExpression(
                constructor_name=constructor_name,
                fields=fields,
            ):
                result = self.infer_constructor_expr(expr, env)
                expr.ty = result[0]
                return result

            case _:
                raise TypeInferenceError(
                    f"Unhandled expression type: {type(expr).__name__}",
                )

    def _infer_binary_numeric_op(
        self,
        left: Expression,
        right: Expression,
        env: TypeEnvironment,
    ) -> InferenceResult:
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

        # Create a fresh type variable for the function to support recursion
        # This will be unified with the inferred type later
        func_type_var = self.fresh_type_var()
        if first_arity == 0:
            # For nullary functions, the type variable is directly the result type
            recursive_env = env.extend(function_name, TypeScheme(set(), func_type_var))
        else:
            # For functions with parameters, create a function type with fresh parameter types
            param_types = [self.fresh_type_var() for _ in range(first_arity)]
            recursive_func_type = func_type_var
            for param_type in reversed(param_types):
                recursive_func_type = FunctionType(param_type, recursive_func_type)
            recursive_env = env.extend(
                function_name,
                TypeScheme(set(), recursive_func_type),
            )

        if first_arity == 0:
            # Nullary functions - all clauses should return the same type
            clause_types = []
            final_subst = TypeSubstitution()

            for func_def in func_defs:
                body_type, body_subst = self.infer_expr(
                    func_def.body,
                    recursive_env.apply_substitution(final_subst),
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

            # Unify the assumed function type with the inferred type
            try:
                unify_subst = unify_one(
                    func_type_var.apply_substitution(final_subst),
                    unified_type.apply_substitution(final_subst),
                )
                final_subst = final_subst.compose(unify_subst)
            except UnificationError as e:
                raise TypeInferenceError(
                    f"Function {function_name} recursive type mismatch: {e}",
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
            clause_env = recursive_env
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

        # Unify the assumed recursive function type with the inferred type
        assumed_recursive_type = recursive_env[function_name].instantiate(
            self.fresh_var_gen,
        )
        try:
            unify_subst = unify_one(
                assumed_recursive_type.apply_substitution(final_subst),
                func_type.apply_substitution(final_subst),
            )
            final_subst = final_subst.compose(unify_subst)
        except UnificationError as e:
            raise TypeInferenceError(
                f"Function {function_name} recursive type mismatch: {e}",
            )

        func_type = func_type.apply_substitution(final_subst)

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
        if not statements:
            return UNIT_TYPE, TypeSubstitution()

        current_env = env
        current_subst = TypeSubstitution()

        # Process all statements except the last
        for stmt in statements[:-1]:
            match stmt:
                case LetStatement(variable=let_variable, value=let_value):
                    # Handle let statements
                    value_type, value_subst = self.infer_expr(
                        let_value,
                        current_env,
                    )
                    current_subst = current_subst.compose(value_subst)

                    # Generalize and add to environment
                    var_scheme = generalize(
                        current_env.apply_substitution(
                            current_subst,
                        ).free_type_vars(),
                        value_type,
                    )
                    current_env = current_env.extend(let_variable, var_scheme)

                # Check if it's an expression (not a declaration)
                case (
                    IntLiteral()
                    | FloatLiteral()
                    | StringLiteral()
                    | BoolLiteral()
                    | ListLiteral()
                    | Variable()
                    | Constructor()
                    | IfElse()
                    | DoBlock()
                    | FunctionApplication()
                    | ConstructorExpression()
                    | GroupedExpression()
                    | NegativeInt()
                    | NegativeFloat() as expr_stmt
                ):
                    # Type check but ignore result for intermediate expressions
                    _, stmt_subst = self.infer_expr(
                        expr_stmt,
                        current_env.apply_substitution(current_subst),
                    )
                    current_subst = current_subst.compose(stmt_subst)

        # Process the last statement and return its type
        last_stmt = statements[-1]
        match last_stmt:
            case LetStatement(variable=let_variable, value=let_value):
                # Handle let statement
                value_type, value_subst = self.infer_expr(let_value, current_env)
                final_subst = current_subst.compose(value_subst)

                # Generalize and add to environment
                var_scheme = generalize(
                    current_env.apply_substitution(
                        final_subst,
                    ).free_type_vars(),
                    value_type,
                )
                current_env = current_env.extend(let_variable, var_scheme)

                return UNIT_TYPE, final_subst  # Let statements don't return values

            case (
                IntLiteral()
                | FloatLiteral()
                | StringLiteral()
                | BoolLiteral()
                | ListLiteral()
                | Variable()
                | Constructor()
                | IfElse()
                | DoBlock()
                | FunctionApplication()
                | ConstructorExpression()
                | GroupedExpression()
                | NegativeInt()
                | NegativeFloat() as expr_stmt
            ):
                # It's an expression - return its type
                return self.infer_expr(
                    expr_stmt,
                    current_env.apply_substitution(current_subst),
                )

            case _:
                # Other statement types (like declarations) don't return values
                return UNIT_TYPE, current_subst

    def infer_constructor_expr(
        self,
        expr: ConstructorExpression,
        env: TypeEnvironment,
    ) -> InferenceResult:
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

            case LiteralPattern(value=value):
                # Literal patterns constrain the pattern type to the literal's type
                literal_type: Type
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

            case TuplePattern(patterns=patterns):
                # Tuple pattern must match a tuple type with same arity
                if not patterns:
                    # Empty tuple pattern
                    empty_tuple_type = TupleType([])
                    unify_subst = unify_one(pattern_type, empty_tuple_type)
                    return env, unify_subst

                # Non-empty tuple pattern
                # Create type variables for each element
                element_types: List[Type] = [self.fresh_type_var() for _ in patterns]
                tuple_type = TupleType(element_types)

                # Unify pattern type with tuple type
                unify_subst = unify_one(pattern_type, tuple_type)
                current_subst = unify_subst
                extended_env = env

                # Infer each sub-pattern with its corresponding element type
                for i, sub_pattern in enumerate(patterns):
                    elem_type = element_types[i].apply_substitution(current_subst)
                    sub_env, sub_subst = self.infer_pattern(
                        sub_pattern,
                        elem_type,
                        extended_env.apply_substitution(current_subst),
                    )
                    extended_env = sub_env
                    current_subst = current_subst.compose(sub_subst)

                return extended_env, current_subst

            case _:
                # Other pattern types (literals, etc.)
                return env, TypeSubstitution()

    def infer_program(self, ast: Program) -> TypeEnvironment:
        env = TypeEnvironment()

        # Add built-in functions
        # putStr :: String -> IO ()
        putstr_type = FunctionType(TypeCon("String"), TypeApp(TypeCon("IO"), UNIT_TYPE))
        env = env.extend("putStr", TypeScheme(set(), putstr_type))

        # error :: String -> a
        error_type = FunctionType(STRING_TYPE, TypeVar("a"))
        env = env.extend("error", TypeScheme({"a"}, error_type))

        # builinPrimities to fix chicken and egg problem
        env = env.extend(
            "primIntAdd",
            TypeScheme(
                set(),
                FunctionType(
                    INT_TYPE,
                    FunctionType(INT_TYPE, INT_TYPE),
                ),
            ),
        )

        env = env.extend(
            "primFloatAdd",
            TypeScheme(
                set(),
                FunctionType(
                    FLOAT_TYPE,
                    FunctionType(FLOAT_TYPE, FLOAT_TYPE),
                ),
            ),
        )

        env = env.extend(
            "primIntSub",
            TypeScheme(
                set(),
                FunctionType(
                    INT_TYPE,
                    FunctionType(INT_TYPE, INT_TYPE),
                ),
            ),
        )

        env = env.extend(
            "primFloatSub",
            TypeScheme(
                set(),
                FunctionType(
                    FLOAT_TYPE,
                    FunctionType(FLOAT_TYPE, FLOAT_TYPE),
                ),
            ),
        )

        env = env.extend(
            "primIntMul",
            TypeScheme(
                set(),
                FunctionType(
                    INT_TYPE,
                    FunctionType(INT_TYPE, INT_TYPE),
                ),
            ),
        )

        env = env.extend(
            "primFloatMul",
            TypeScheme(
                set(),
                FunctionType(
                    FLOAT_TYPE,
                    FunctionType(FLOAT_TYPE, FLOAT_TYPE),
                ),
            ),
        )

        env = env.extend(
            "primFloatDiv",
            TypeScheme(
                set(),
                FunctionType(
                    FLOAT_TYPE,
                    FunctionType(FLOAT_TYPE, FLOAT_TYPE),
                ),
            ),
        )

        # Integer division primitive
        env = env.extend(
            "primIntDiv",
            TypeScheme(
                set(),
                FunctionType(
                    INT_TYPE,
                    FunctionType(INT_TYPE, INT_TYPE),
                ),
            ),
        )

        # Modulo primitive
        env = env.extend(
            "primIntMod",
            TypeScheme(
                set(),
                FunctionType(
                    INT_TYPE,
                    FunctionType(INT_TYPE, INT_TYPE),
                ),
            ),
        )

        # Exponentiation primitives
        env = env.extend(
            "primIntPow",
            TypeScheme(
                set(),
                FunctionType(
                    INT_TYPE,
                    FunctionType(INT_TYPE, INT_TYPE),
                ),
            ),
        )

        env = env.extend(
            "primFloatPow",
            TypeScheme(
                set(),
                FunctionType(
                    FLOAT_TYPE,
                    FunctionType(FLOAT_TYPE, FLOAT_TYPE),
                ),
            ),
        )

        # List indexing primitive
        env = env.extend(
            "primListIndex",
            TypeScheme(
                {"a"},
                FunctionType(
                    TypeApp(TypeCon("List"), TypeVar("a")),
                    FunctionType(INT_TYPE, TypeVar("a")),
                ),
            ),
        )

        # Unary negation primitives
        env = env.extend(
            "primIntNeg",
            TypeScheme(
                set(),
                FunctionType(INT_TYPE, INT_TYPE),
            ),
        )

        env = env.extend(
            "primFloatNeg",
            TypeScheme(
                set(),
                FunctionType(FLOAT_TYPE, FLOAT_TYPE),
            ),
        )

        # Comparison primitives
        env = env.extend(
            "primIntLt",
            TypeScheme(
                set(),
                FunctionType(
                    INT_TYPE,
                    FunctionType(INT_TYPE, TypeCon("Bool")),
                ),
            ),
        )

        env = env.extend(
            "primFloatLt",
            TypeScheme(
                set(),
                FunctionType(
                    FLOAT_TYPE,
                    FunctionType(FLOAT_TYPE, TypeCon("Bool")),
                ),
            ),
        )

        env = env.extend(
            "primIntLe",
            TypeScheme(
                set(),
                FunctionType(
                    INT_TYPE,
                    FunctionType(INT_TYPE, TypeCon("Bool")),
                ),
            ),
        )

        env = env.extend(
            "primFloatLe",
            TypeScheme(
                set(),
                FunctionType(
                    FLOAT_TYPE,
                    FunctionType(FLOAT_TYPE, TypeCon("Bool")),
                ),
            ),
        )

        env = env.extend(
            "primIntGt",
            TypeScheme(
                set(),
                FunctionType(
                    INT_TYPE,
                    FunctionType(INT_TYPE, TypeCon("Bool")),
                ),
            ),
        )

        env = env.extend(
            "primFloatGt",
            TypeScheme(
                set(),
                FunctionType(
                    FLOAT_TYPE,
                    FunctionType(FLOAT_TYPE, TypeCon("Bool")),
                ),
            ),
        )

        env = env.extend(
            "primIntGe",
            TypeScheme(
                set(),
                FunctionType(
                    INT_TYPE,
                    FunctionType(INT_TYPE, TypeCon("Bool")),
                ),
            ),
        )

        env = env.extend(
            "primFloatGe",
            TypeScheme(
                set(),
                FunctionType(
                    FLOAT_TYPE,
                    FunctionType(FLOAT_TYPE, TypeCon("Bool")),
                ),
            ),
        )

        env = env.extend(
            "primIntEq",
            TypeScheme(
                set(),
                FunctionType(
                    INT_TYPE,
                    FunctionType(INT_TYPE, TypeCon("Bool")),
                ),
            ),
        )

        env = env.extend(
            "primFloatEq",
            TypeScheme(
                set(),
                FunctionType(
                    FLOAT_TYPE,
                    FunctionType(FLOAT_TYPE, TypeCon("Bool")),
                ),
            ),
        )

        env = env.extend(
            "primBoolEq",
            TypeScheme(
                set(),
                FunctionType(
                    TypeCon("Bool"),
                    FunctionType(TypeCon("Bool"), TypeCon("Bool")),
                ),
            ),
        )

        env = env.extend(
            "primListEq",
            TypeScheme(
                {"a"},
                FunctionType(
                    TypeApp(TypeCon("List"), TypeVar("a")),
                    FunctionType(
                        TypeApp(TypeCon("List"), TypeVar("a")),
                        TypeCon("Bool"),
                    ),
                ),
            ),
        )

        # Logical primitives
        env = env.extend(
            "primBoolAnd",
            TypeScheme(
                set(),
                FunctionType(
                    TypeCon("Bool"),
                    FunctionType(TypeCon("Bool"), TypeCon("Bool")),
                ),
            ),
        )

        env = env.extend(
            "primBoolOr",
            TypeScheme(
                set(),
                FunctionType(
                    TypeCon("Bool"),
                    FunctionType(TypeCon("Bool"), TypeCon("Bool")),
                ),
            ),
        )

        env = env.extend(
            "primStringConcat",
            TypeScheme(
                set(),
                FunctionType(
                    TypeCon("String"),
                    FunctionType(TypeCon("String"), TypeCon("String")),
                ),
            ),
        )

        # List concatenation primitive
        env = env.extend(
            "primListConcat",
            TypeScheme(
                {"a"},
                FunctionType(
                    TypeApp(TypeCon("List"), TypeVar("a")),
                    FunctionType(
                        TypeApp(TypeCon("List"), TypeVar("a")),
                        TypeApp(TypeCon("List"), TypeVar("a")),
                    ),
                ),
            ),
        )

        # Create dummy function definition for built-in operators
        from lango.systemo.ast.nodes import FunctionDefinition, IntLiteral

        dummy_func_def = FunctionDefinition(
            function_name="_dummy",
            patterns=[],
            body=IntLiteral(0),
        )

        # Show function instances for different types
        self.instances["show"] = [
            (FunctionType(INT_TYPE, TypeCon("String")), dummy_func_def),
            (FunctionType(FLOAT_TYPE, TypeCon("String")), dummy_func_def),
            (FunctionType(TypeCon("Bool"), TypeCon("String")), dummy_func_def),
            (FunctionType(TypeCon("String"), TypeCon("String")), dummy_func_def),
        ]

        # First pass: collect data declarations and precedence declarations
        for stmt in ast.statements:
            match stmt:
                case DataDeclaration() as data_decl:
                    data_env = self.infer_data_decl(data_decl)
                    env = env.extend_many(data_env.bindings)
                case PrecedenceDeclaration() as prec_decl:
                    self.handle_precedence_decl(prec_decl)
                case _:
                    continue

        # Second pass: process instance declarations (now that data types are known)
        for stmt in ast.statements:
            match stmt:
                case InstanceDeclaration() as inst_decl:
                    # Handle instance declaration with full environment
                    self.handle_instance_decl_with_env(inst_decl, env)
                case _:
                    continue

        # Third pass: create forward declarations for all functions
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
                # Debug: print available instances for < operator
                if function_name == "abs":
                    print(
                        f"DEBUG: Available instances for '<': {self.instances.get('<', 'NOT FOUND')}",
                    )
                    print(f"DEBUG: All instance keys: {list(self.instances.keys())}")
                raise TypeInferenceError(
                    f"Failed to infer type for function {function_name}: {e}",
                ) from e

        return env


def type_check_ast(ast: Program) -> TypeEnvironment:
    inferrer = TypeInferrer()
    return inferrer.infer_program(ast)
