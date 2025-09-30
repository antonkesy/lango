import hashlib
import re
from typing import Any, Dict, List, Optional, Set

from lango.shared.ast.nodes import (
    ArrowType,
    BoolLiteral,
    CharLiteral,
    ConsPattern,
    Constructor,
    ConstructorExpression,
    ConstructorPattern,
    DataConstructor,
    DataDeclaration,
    DoBlock,
    Expression,
    FloatLiteral,
    FunctionApplication,
    FunctionDefinition,
    GroupedExpression,
    IfElse,
    InstanceDeclaration,
    IntLiteral,
    LetStatement,
    ListLiteral,
    ListPattern,
    LiteralPattern,
    NegativeFloat,
    NegativeInt,
    Pattern,
    Program,
    StringLiteral,
    SymbolicOperation,
    TupleLiteral,
    TuplePattern,
    TupleType,
    TypeApplication,
    TypeConstructor,
    TypeExpression,
    TypeVariable,
    Variable,
    VariablePattern,
    is_systemo_expression,
)
from lango.shared.compiler.python import (
    build_cons_pattern_match,
    build_list_pattern_match,
    build_literal_pattern_match,
    build_multi_arg_pattern_match,
    build_positional_pattern_match,
    build_record_pattern_match,
    build_simple_pattern_match,
    build_tuple_pattern_match,
    compile_literal_value,
)
from lango.shared.typechecker.lango_types import (
    DataType,
    FunctionType,
    TupleType,
    Type,
    TypeApp,
    TypeCon,
    TypeVar,
)


class SystemoCompiler:
    def __init__(self) -> None:
        self.indent_level = 0
        self.defined_functions: Set[str] = set()
        self.nullary_functions: Set[str] = set()  # Functions with no parameters
        self.lambda_lifted_functions: Set[str] = (
            set()
        )  # Functions that had lambdas lifted to parameters
        self.function_types: Dict[str, Type] = {}
        self.data_types: Dict[str, DataDeclaration] = {}
        self.local_variables: Set[str] = set()
        self.monomorphic_functions: Dict[str, str] = (
            {}
        )  # Maps name+fulltype -> monomorphic function name (legacy compatibility)
        self.type_dictionaries: Dict[str, Dict[str, str]] = (
            {}
        )  # Maps type name -> {operation -> primitive function}
        self.generated_dictionaries: Set[str] = set()
        self.generic_functions: Set[str] = set()
        self.generated_dispatchers: Set[str] = set()
        self.current_function_name: Optional[str] = None
        self.concrete_instantiations: Dict[str, Set[str]] = {}
        self.instance_declarations_registry: Dict[str, List] = {}
        self.generated_functions: List[str] = []

        # Reserved function names that conflict with built-in operators
        self.reserved_function_names: Set[str] = {
            "add",
            "subtract",
            "multiply",
            "divide",
            "mod",
            "pow",
            "neg",
            "equals",
            "not_equals",
            "less_than",
            "greater_than",
            "less_equal",
            "greater_equal",
            "and",
            "or",
            "not",
            "concat",
            "bangbang",
            "show",
            "first",
            "second",
            "third",
        }

    def _indent(self) -> str:
        return "    " * self.indent_level

    def _initialize_type_dictionaries(self) -> None:

        self.type_dictionaries["Int"] = {
            "add": "primIntAdd",
            "sub": "primIntSub",
            "mul": "primIntMul",
            "div": "primIntDiv",
            "pow": "primIntPow",
            "neg": "primIntNeg",
            "lt": "primIntLt",
            "le": "primIntLe",
            "gt": "primIntGt",
            "ge": "primIntGe",
            "eq": "primIntEq",
            "equals": "primIntEq",
            "show": "primIntShow",
            "slasheq": "lambda x, y: not primIntEq(x, y)",  # not equal (/=)
            "question": "lambda x: -x",  # unary minus (?)
        }

        self.type_dictionaries["Float"] = {
            "add": "primFloatAdd",
            "sub": "primFloatSub",
            "mul": "primFloatMul",
            "div": "primFloatDiv",
            "pow": "primFloatPow",
            "neg": "primFloatNeg",
            "lt": "primFloatLt",
            "le": "primFloatLe",
            "gt": "primFloatGt",
            "ge": "primFloatGe",
            "eq": "primFloatEq",
            "equals": "primFloatEq",
            "show": "primFloatShow",
            "slasheq": "lambda x, y: not primFloatEq(x, y)",  # not equal (/=)
            "question": "lambda x: -x",  # unary minus (?)
        }

        self.type_dictionaries["Bool"] = {
            "eq": "primBoolEq",
            "equals": "primBoolEq",
            "show": "primBoolShow",
            "and": "primBoolAnd",
            "or": "primBoolOr",
            "ampamp": "primBoolAnd",
            "pipepipe": "primBoolOr",
            "slasheq": "lambda x, y: not primBoolEq(x, y)",  # not equal (/=)
        }

        self.type_dictionaries["String"] = {
            "show": "primStringShow",
            "concat": "primStringConcat",
            "plusplus": "primStringConcat",
            "eq": "primStringEq",
            "equals": "primStringEq",
            "slasheq": "lambda x, y: not primStringEq(x, y)",  # not equal (/=)
        }

        # Char type dictionary
        self.type_dictionaries["Char"] = {
            "show": "primCharShow",
        }

        # List type dictionary (generic operations)
        self.type_dictionaries["List"] = {
            "concat": "primListConcat",
            "plusplus": "primListConcat",  # list concatenation (++)
            "show": "lambda x: '[' + ','.join(systemo_show_runtime_dispatch(item) for item in x) + ']'",
            "bangbang": "lambda x, i: x[i]",  # list indexing (!!)
            "equals": "lambda x, y: x == y",  # list equality
            "eq": "lambda x, y: x == y",
        }

        # Tuple type dictionary (generic operations for tuples)
        self.type_dictionaries["tuple"] = {
            "first": "lambda x: x[0]",
            "second": "lambda x: x[1]",
            "third": "lambda x: x[2]",
        }

    def _add_user_defined_operations_to_dictionaries(
        self,
        instance_declarations: Dict[str, List["InstanceDeclaration"]],
    ) -> None:
        # Process all instance declarations to populate type dictionaries
        for instance_name, instances in instance_declarations.items():
            # Extract the actual function name
            actual_function_name = self._extract_function_name(instance_name)

            # Only skip operations that are primitives or already have primitive implementations
            # Allow user-defined operations like xcoord, ycoord to be added to type dictionaries
            # Allow show operations for complex types that don't have primitive implementations (like lists, tuples)
            if actual_function_name == "show":
                # Check if this is for a type that already has a primitive show implementation
                has_primitive_show = False
                for instance in instances:
                    type_info = self._extract_instance_type_info(instance)
                    if type_info:
                        type_name, _ = type_info
                        # Types with primitive show implementations
                        primitive_show_types = {
                            "Int",
                            "Float",
                            "Bool",
                            "String",
                            "Char",
                        }
                        if type_name in primitive_show_types:
                            has_primitive_show = True
                            break

                if has_primitive_show:
                    continue

            # Group instances by their type
            type_to_instances: Dict[str, List[InstanceDeclaration]] = {}
            for instance in instances:
                type_info = self._extract_instance_type_info(instance)
                if type_info:
                    type_name, _ = type_info
                    if type_name not in type_to_instances:
                        type_to_instances[type_name] = []
                    type_to_instances[type_name].append(instance)

            # Generate comprehensive lambda implementations for each type
            for type_name, type_instances in type_to_instances.items():
                # Initialize type dictionary if it doesn't exist
                if type_name not in self.type_dictionaries:
                    self.type_dictionaries[type_name] = {}

                # Generate lambda implementation that handles all patterns for this type
                lambda_impl = self._generate_comprehensive_lambda_implementation(
                    type_instances,
                    type_name,
                    actual_function_name,
                )
                if lambda_impl:
                    # Add the operation to the type dictionary using the sanitized operation name
                    # This ensures consistency with symbolic operation compilation
                    operation_key = self._sanitize_operator_name(actual_function_name)
                    self.type_dictionaries[type_name][operation_key] = lambda_impl

    def _extract_instance_type_info(
        self,
        instance: "InstanceDeclaration",
    ) -> Optional[tuple]:
        # Get the type signature and extract the input type
        type_sig = instance.type_signature

        # Check for function type (ArrowType or FunctionType)
        if isinstance(type_sig, ArrowType):
            # Function type: from_type -> to_type
            param_type = type_sig.from_type

            if isinstance(param_type, TypeConstructor):
                type_name = param_type.name

                # Get the operation name for dictionary lookup
                instance_name = self._extract_function_name(instance.instance_name)

                # Sanitize the instance name for use in Python function names
                safe_instance_name = self._sanitize_operator_name(instance_name)
                if not safe_instance_name or not safe_instance_name[0].isalpha():
                    safe_instance_name = f"op_{safe_instance_name}"

                # Create the monomorphic function name based on the specific type
                # For example: xcoord_Point -> Int becomes systemo_xcoord_Point_to_Int
                func_def = instance.function_definition
                if func_def and func_def.patterns:
                    arity = len(func_def.patterns)

                    if arity == 1:
                        # Unary function - create monomorphic function name for this specific type
                        type_sig_str = self._type_expression_to_string(
                            instance.type_signature,
                        )
                        safe_type_sig = type_sig_str.replace(" -> ", "_to_").replace(
                            " ",
                            "_",
                        )
                        monomorphic_func_name = (
                            f"systemo_{safe_instance_name}_{safe_type_sig}"
                        )
                        return (type_name, monomorphic_func_name)
                    elif arity == 2:
                        # Binary function - create monomorphic function name for this specific type
                        type_sig_str = self._type_expression_to_string(
                            instance.type_signature,
                        )
                        safe_type_sig = type_sig_str.replace(" -> ", "_to_").replace(
                            " ",
                            "_",
                        )
                        monomorphic_func_name = (
                            f"systemo_{safe_instance_name}_{safe_type_sig}"
                        )
                        return (type_name, monomorphic_func_name)
        return None

    def _compile_expression_with_substitutions(
        self,
        expr: Any,
        substitutions: dict,
    ) -> str:
        if isinstance(expr, Variable):
            if expr.name in substitutions and substitutions[expr.name]:
                return substitutions[expr.name]
            else:
                # Track substituted variables as local to prevent prefixing
                old_local_vars = self.local_variables.copy()
                self.local_variables.update(substitutions.keys())

                try:
                    return self._compile_expression(expr)
                finally:
                    # Restore original local variables
                    self.local_variables = old_local_vars
        elif isinstance(expr, FunctionApplication):
            # Handle function applications recursively
            old_local_vars = self.local_variables.copy()
            self.local_variables.update(substitutions.keys())

            try:
                func_code = self._compile_expression_with_substitutions(
                    expr.function,
                    substitutions,
                )
                arg_code = self._compile_expression_with_substitutions(
                    expr.argument,
                    substitutions,
                )
                return f"{func_code}({arg_code})"
            finally:
                # Restore original local variables
                self.local_variables = old_local_vars
        else:
            # For other expression types (literals, etc.), use normal compilation
            old_local_vars = self.local_variables.copy()
            self.local_variables.update(substitutions.keys())

            try:
                return self._compile_expression(expr)
            finally:
                # Restore original local variables
                self.local_variables = old_local_vars

    def _substitute_variables_in_ast(self, expr: Any, substitutions: dict) -> Any:
        if isinstance(expr, Variable):
            if expr.name in substitutions:
                # Create a new Variable node with the substituted name
                return Variable(name=substitutions[expr.name], ty=expr.ty)
            else:
                return expr
        elif isinstance(expr, FunctionApplication):
            # Recursively substitute in function and argument
            new_function = self._substitute_variables_in_ast(
                expr.function,
                substitutions,
            )
            new_argument = self._substitute_variables_in_ast(
                expr.argument,
                substitutions,
            )
            return FunctionApplication(function=new_function, argument=new_argument)
        elif isinstance(expr, SymbolicOperation):
            # Recursively substitute in operands
            new_operands = [
                self._substitute_variables_in_ast(operand, substitutions)
                for operand in expr.operands
            ]
            return SymbolicOperation(
                operator=expr.operator,
                operands=new_operands,
                ty=expr.ty,
            )
        else:
            # For other expression types (literals, etc.), return as-is
            return expr

    def _generate_comprehensive_lambda_implementation(
        self,
        instances: List["InstanceDeclaration"],
        type_name: str,
        function_name: str,
    ) -> Optional[str]:
        if not instances:
            return None

        # Collect all patterns and their corresponding body expressions
        pattern_cases: List[Any] = []
        arity = None
        for instance in instances:
            func_def = instance.function_definition
            if func_def and func_def.patterns:
                current_arity = len(func_def.patterns)
                if arity is None:
                    arity = current_arity
                elif arity != current_arity:
                    # Mixed arity - not supported yet
                    continue

                if arity == 1:
                    pattern = func_def.patterns[0]
                    body_expr = func_def.body
                    pattern_cases.append((pattern, body_expr))
                elif arity == 2:
                    pattern1 = func_def.patterns[0]
                    pattern2 = func_def.patterns[1]
                    body_expr = func_def.body
                    pattern_cases.append(((pattern1, pattern2), body_expr))

        if not pattern_cases:
            return None

        # Generate lambda with pattern matching
        if arity == 1:
            return self._generate_pattern_matching_lambda(pattern_cases)
        elif arity == 2:
            return self._generate_binary_pattern_matching_lambda(pattern_cases)
        else:
            return None

    def _generate_pattern_matching_lambda(self, pattern_cases: List) -> str:
        # Start building the lambda
        lambda_body_parts = []
        default_case = None

        for pattern, body_expr in pattern_cases:
            if isinstance(pattern, LiteralPattern):
                # Literal pattern like ol 0 = 42 or ol 0.0 = 73.0
                literal_value = pattern.value
                body_code = self._compile_expression(body_expr)
                lambda_body_parts.append(
                    f"({body_code} if arg == {literal_value} else ",
                )
            elif isinstance(pattern, ConstructorPattern):
                # Constructor pattern like xcoord (MkPoint x y) = x
                constructor = pattern.constructor
                sub_patterns = pattern.patterns

                # Generate field extraction and substitution
                substitutions = {}
                for i, sub_pattern in enumerate(sub_patterns):
                    if isinstance(sub_pattern, VariablePattern):
                        # Map pattern variable to field access
                        substitutions[sub_pattern.name] = f"arg.arg_{i}"

                # Substitute variables in the body expression
                substituted_body = self._substitute_variables_in_ast(
                    body_expr,
                    substitutions,
                )

                # Add 'arg' as a local variable to prevent systemo_ prefixing
                old_local_vars = self.local_variables.copy()
                self.local_variables.add("arg")

                try:
                    body_code = self._compile_expression(substituted_body)
                finally:
                    self.local_variables = old_local_vars

                # Constructor patterns become conditions on the constructor type
                lambda_body_parts.append(
                    f"({body_code} if isinstance(arg, {constructor}) else ",
                )
            elif isinstance(pattern, VariablePattern):
                # Variable pattern like ol x = x + 1 - this becomes the default case
                var_name = pattern.name
                substituted_body = self._substitute_variables_in_ast(
                    body_expr,
                    {var_name: "arg"},
                )

                # Add 'arg' as a local variable to prevent systemo_ prefixing
                old_local_vars = self.local_variables.copy()
                self.local_variables.add("arg")

                try:
                    body_code = self._compile_expression(substituted_body)
                finally:
                    self.local_variables = old_local_vars

                default_case = body_code

        # Construct the full lambda expression
        if lambda_body_parts and default_case:
            # We have specific literal cases and a default case
            lambda_expr = (
                "".join(lambda_body_parts)
                + f"{default_case}"
                + ")" * len(lambda_body_parts)
            )
            return f"lambda arg: {lambda_expr}"
        elif default_case:
            # Only variable pattern
            return f"lambda arg: {default_case}"
        else:
            # Only literal patterns (shouldn't happen in practice)
            if lambda_body_parts:
                lambda_expr = (
                    "".join(lambda_body_parts) + "None" + ")" * len(lambda_body_parts)
                )
                return f"lambda arg: {lambda_expr}"

        return "lambda arg: None"  # Fallback

    def _generate_binary_pattern_matching_lambda(self, pattern_cases: List) -> str:
        # For now, handle simple variable patterns in binary operations
        # More complex pattern matching can be added later
        if len(pattern_cases) == 1:
            (pattern1, pattern2), body_expr = pattern_cases[0]

            if isinstance(pattern1, VariablePattern) and isinstance(
                pattern2,
                VariablePattern,
            ):
                # Simple binary variable pattern like (?) a b = a + b
                var1_name = pattern1.name
                var2_name = pattern2.name

                # Keep the original variable names to avoid transformation issues
                # Add them as local variables to prevent systemo_ prefixing
                old_local_vars = self.local_variables.copy()
                self.local_variables.add(var1_name)
                self.local_variables.add(var2_name)

                try:
                    body_code = self._compile_expression(body_expr)
                finally:
                    self.local_variables = old_local_vars

                return f"lambda {var1_name}, {var2_name}: {body_code}"

        # Fallback for unsupported patterns
        return "lambda arg_0, arg_1: None"

    def _generate_dictionary_definitions(self) -> List[str]:
        lines = ["# Type dictionaries for polymorphic operations"]

        for type_name, operations in self.type_dictionaries.items():
            dict_name = f"{type_name.lower()}_dict"
            lines.append(f"{dict_name} = {{")
            for op_name, prim_func in operations.items():
                # If prim_func is a string that looks like a lambda, emit it directly
                # Otherwise treat it as a function name that needs to be properly formatted
                if isinstance(prim_func, str) and prim_func.strip().startswith(
                    "lambda",
                ):
                    lines.append(f"    '{op_name}': {prim_func},")
                else:
                    lines.append(f"    '{op_name}': {prim_func},")
            lines.append("}")
            lines.append("")

        # Generate the generic dispatcher functions that use these dictionaries
        lines.extend(self._generate_generic_dispatch_functions())

        return lines

    def _generate_generic_dispatch_functions(self) -> List[str]:
        lines = ["# Generic dispatcher functions"]

        # Generate binary operation dispatchers with specific function names
        binary_ops = {
            ("plus", "systemo_add"): ("add", "+"),
            ("minus", "systemo_subtract"): ("sub", "-"),
            ("star", "systemo_multiply"): ("mul", "*"),
            ("slash", "systemo_divide"): ("div", "/"),
            ("pow", "systemo_power"): ("pow", "**"),
            ("mod", "systemo_modulo"): ("mod", "%"),
            ("eqeq", "systemo_equals"): ("eq", "=="),
            ("bangeq", "systemo_not_equals"): ("ne", "!="),
            ("lt", "systemo_less_than"): ("lt", "<"),
            ("lteq", "systemo_less_equal"): ("le", "<="),
            ("gt", "systemo_greater_than"): ("gt", ">"),
            ("gteq", "systemo_greater_equal"): ("ge", ">="),
            ("plusplus", "systemo_plusplus"): ("plusplus", "++"),
            ("ampamp", "systemo_ampamp"): ("ampamp", "&&"),
            ("pipepipe", "systemo_pipepipe"): ("pipepipe", "||"),
        }

        for (safe_op_name, func_name), (dict_key, python_op) in binary_ops.items():
            lines.append(f"def {func_name}(type_dict, x, y):")
            lines.append(f"    if '{dict_key}' in type_dict:")
            lines.append(f"        return type_dict['{dict_key}'](x, y)")
            lines.append(f"    else:")
            lines.append(
                f"        raise ValueError(f'Operation {dict_key} not supported for types {{type(x).__name__}} and {{type(y).__name__}}')",
            )
            lines.append("")

        # Generate unary operation dispatchers with specific function names
        unary_ops = {
            ("minus", "systemo_negate"): ("neg", "-"),
            ("bang", "systemo_not"): ("not", "not"),
        }

        for (safe_op_name, func_name), (dict_key, python_op) in unary_ops.items():
            lines.append(f"def {func_name}(type_dict, x):")
            lines.append(f"    if '{dict_key}' in type_dict:")
            lines.append(f"        return type_dict['{dict_key}'](x)")
            lines.append(f"    else:")
            lines.append(
                f"        raise ValueError(f'Operation {dict_key} not supported for type {{type(x).__name__}}')",
            )
            lines.append("")

        # Generate tuple operation dispatchers
        tuple_ops = {
            "systemo_first": "first",
            "systemo_second": "second",
            "systemo_third": "third",
        }

        for func_name, dict_key in tuple_ops.items():
            lines.append(f"def {func_name}(type_dict, x):")
            lines.append(f"    if '{dict_key}' in type_dict:")
            lines.append(f"        return type_dict['{dict_key}'](x)")
            lines.append(f"    else:")
            lines.append(
                f"        raise ValueError(f'Operation {dict_key} not supported for type {{type(x).__name__}}')",
            )
            lines.append("")

        # Generate dispatchers for user-defined polymorphic functions
        self._generate_user_defined_dispatchers(lines)

        # Generate runtime dispatch functions for fallback cases
        self._generate_runtime_dispatch_functions(lines)

        return lines

    def _generate_runtime_dispatch_functions(self, lines: List[str]) -> None:
        lines.append("# Runtime dispatch functions for fallback cases")

        # Binary operations runtime dispatch
        binary_ops = [
            ("systemo_add", "+"),
            ("systemo_subtract", "-"),
            ("systemo_multiply", "*"),
            ("systemo_divide", "/"),
            ("systemo_power", "**"),
            ("systemo_modulo", "%"),
            ("systemo_equals", "=="),
            ("systemo_not_equals", "!="),
            ("systemo_less_than", "<"),
            ("systemo_less_equal", "<="),
            ("systemo_greater_than", ">"),
            ("systemo_greater_equal", ">="),
            ("systemo_plusplus", "++"),
            ("systemo_ampamp", "&&"),
        ]

        for func_name, op_symbol in binary_ops:
            lines.append(f"def {func_name}_runtime_dispatch(x, y):")
            lines.append(f"    # Try to infer type dictionary from operand types")
            lines.append(f"    if isinstance(x, int) and isinstance(y, int):")
            lines.append(f"        return {func_name}(int_dict, x, y)")
            lines.append(f"    elif isinstance(x, float) and isinstance(y, float):")
            lines.append(f"        return {func_name}(float_dict, x, y)")
            lines.append(f"    elif isinstance(x, str) and isinstance(y, str):")
            lines.append(f"        return {func_name}(string_dict, x, y)")
            lines.append(f"    elif isinstance(x, bool) and isinstance(y, bool):")
            lines.append(f"        return {func_name}(bool_dict, x, y)")
            lines.append(f"    elif isinstance(x, list) and isinstance(y, list):")
            lines.append(f"        return {func_name}(list_dict, x, y)")
            lines.append(f"    else:")
            lines.append(
                f"        raise ValueError(f'Operation {op_symbol} not supported for types {{type(x).__name__}} and {{type(y).__name__}}')",
            )
            lines.append("")

        # Unary operations runtime dispatch
        unary_ops = [
            ("systemo_negate", "-"),
            ("systemo_not", "not"),
        ]

        for func_name, op_symbol in unary_ops:
            lines.append(f"def {func_name}_runtime_dispatch(x):")
            lines.append(f"    # Try to infer type dictionary from operand type")
            lines.append(f"    if isinstance(x, int):")
            lines.append(f"        return {func_name}(int_dict, x)")
            lines.append(f"    elif isinstance(x, float):")
            lines.append(f"        return {func_name}(float_dict, x)")
            lines.append(f"    elif isinstance(x, bool):")
            lines.append(f"        return {func_name}(bool_dict, x)")
            lines.append(f"    else:")
            lines.append(
                f"        raise ValueError(f'Operation {op_symbol} not supported for type {{type(x).__name__}}')",
            )
            lines.append("")

        # Add runtime dispatch for show function
        lines.append("def systemo_show_runtime_dispatch(x):")
        lines.append("    # Try to infer type dictionary from operand type")
        lines.append("    if isinstance(x, int):")
        lines.append("        return systemo_show(int_dict, x)")
        lines.append("    elif isinstance(x, float):")
        lines.append("        return systemo_show(float_dict, x)")
        lines.append("    elif isinstance(x, bool):")
        lines.append("        return systemo_show(bool_dict, x)")
        lines.append("    elif isinstance(x, str):")
        lines.append("        return systemo_show(string_dict, x)")
        lines.append("    elif isinstance(x, list):")
        lines.append("        return systemo_show(list_dict, x)")
        lines.append("    elif isinstance(x, tuple):")
        lines.append("        return systemo_show(tuple_dict, x)")
        lines.append("    else:")
        lines.append(
            "        raise ValueError(f'Operation show not supported for type {type(x).__name__}')",
        )
        lines.append("")

        # Add runtime dispatch for map function
        lines.append("def systemo_map_runtime_dispatch(f, xs):")
        lines.append("    # Map function works on lists regardless of element type")
        lines.append(
            "    # The type dictionary is for list operations, which we know is list_dict",
        )
        lines.append("    return systemo_map(list_dict, f, xs)")
        lines.append("")

    def _generate_user_defined_dispatchers(self, lines: List[str]) -> None:
        # Collect all user-defined polymorphic function names
        user_defined_functions = set()

        # From instance declarations
        for instance_name in self.instance_declarations_registry.keys():
            clean_name = self._sanitize_operator_name(instance_name)
            if not clean_name or not clean_name[0].isalpha():
                clean_name = f"op_{clean_name}"
            user_defined_functions.add((instance_name, clean_name))

        # From type dictionaries
        for type_dict in self.type_dictionaries.values():
            for op_name in type_dict.keys():
                clean_name = self._sanitize_operator_name(op_name)
                if not clean_name or not clean_name[0].isalpha():
                    clean_name = f"op_{clean_name}"
                user_defined_functions.add((op_name, clean_name))

        lines.append("# User-defined polymorphic function dispatchers")

        for original_name, clean_name in user_defined_functions:
            # Skip if this is a built-in operation already handled above
            built_in_ops = {
                "add",
                "sub",
                "mul",
                "div",
                "pow",
                "mod",
                "eq",
                "ne",
                "lt",
                "le",
                "gt",
                "ge",
                "neg",
                "not",
                "plusplus",
                "question",
                "ampamp",
                "pipepipe",
            }
            if clean_name in built_in_ops or original_name in built_in_ops:
                continue

            func_name = f"systemo_{clean_name}"

            # Skip if we already generated this dispatcher
            if func_name in self.generated_dispatchers:
                continue

            # Mark as generated to prevent duplicates
            self.generated_dispatchers.add(func_name)

            # Generate generic dispatcher - assume unary for now, can be extended for multi-arity
            lines.append(f"def {func_name}(type_dict, *args):")
            lines.append(f"    if '{clean_name}' in type_dict:")
            lines.append(f"        return type_dict['{clean_name}'](*args)")
            lines.append(f"    else:")
            lines.append(
                f"        raise ValueError(f'Operation {clean_name} not supported for type {{type(args[0]).__name__ if args else \"unknown\"}}')",
            )
            lines.append("")

    def _prefix_name(self, name: str) -> str:
        # Don't prefix field access expressions
        if "." in name and name.startswith("arg."):
            return name
        # Don't prefix local pattern variables
        if name in self.local_variables:
            return name
        # Don't prefix constructor names (they should be detected by their usage)
        # Add systemo_ prefix to all other names, with sanitization for operators
        sanitized_name = self._sanitize_operator_name(name)
        # Avoid conflicts with reserved function names by adding suffix
        if name in self.reserved_function_names:
            return f"systemo_{sanitized_name}_user"
        else:
            return f"systemo_{sanitized_name}"

    def _find_constructor_def(
        self,
        constructor_name: str,
    ) -> Optional["DataConstructor"]:
        for data_decl in self.data_types.values():
            for constructor in data_decl.constructors:
                if constructor.name == constructor_name:
                    return constructor
        return None

    def _get_function_arity(self, func_name: str) -> int:
        if func_name not in self.function_types:
            # If function type is not known, assume arity 1 as default
            return 1

        func_type = self.function_types[func_name]
        arity = 0
        while isinstance(func_type, FunctionType):
            arity += 1
            func_type = func_type.result
        return arity

    def _convert_type_expression_to_type(self, type_expr: Any) -> Optional[Type]:
        if hasattr(type_expr, "ty") and type_expr.ty is not None:
            return type_expr.ty
        # If no type information is available, return None
        return None

    def _systemo_type_to_python_hint(self, systemo_type: Optional[Type]) -> str:
        if systemo_type is None:
            return "Any"

        match systemo_type:
            case TypeCon(name="Int"):
                return "int"
            case TypeCon(name="String"):
                return "str"
            case TypeCon(name="Float"):
                return "float"
            case TypeCon(name="Bool"):
                return "bool"
            case TypeCon(name="Char"):
                return "str"  # Map Char to str at runtime
            case TypeCon(name="()"):
                return "None"
            case TypeApp(constructor=TypeCon(name="List"), argument=arg_type):
                inner_type = self._systemo_type_to_python_hint(arg_type)
                return f"List[{inner_type}]"
            case TypeApp(constructor=TypeCon(name="IO"), argument=arg_type):
                # IO types typically don't have meaningful return types in our compiled Python
                return "None"
            case FunctionType(param=param_type, result=result_type):
                # For function types, we'll use Callable
                param_hint = self._systemo_type_to_python_hint(param_type)
                result_hint = self._systemo_type_to_python_hint(result_type)
                return f"Callable[[{param_hint}], {result_hint}]"
            case DataType(name=name, type_args=type_args):
                # Custom data types - use Union of all constructors for the type
                if name in self.data_types:
                    constructors = [
                        ctor.name for ctor in self.data_types[name].constructors
                    ]
                    if len(constructors) == 1:
                        # Single constructor, use it directly
                        return constructors[0]
                    else:
                        # Multiple constructors, use Union
                        return f"Union[{', '.join(constructors)}]"
                else:
                    # Unknown data type, use Any
                    return "Any"
            case TypeVar(name=name):
                # Type variables become Any for now
                return "Any"
            case _:
                return "Any"

    def _extract_pattern_variables(self, pattern: Pattern) -> Set[str]:
        variables = set()
        match pattern:
            case VariablePattern(name=name):
                variables.add(name)
            case ConstructorPattern(patterns=patterns):
                for sub_pattern in patterns:
                    variables.update(self._extract_pattern_variables(sub_pattern))
            case ConsPattern(head=head, tail=tail):
                variables.update(self._extract_pattern_variables(head))
                variables.update(self._extract_pattern_variables(tail))
            case TuplePattern(patterns=patterns):
                for sub_pattern in patterns:
                    variables.update(self._extract_pattern_variables(sub_pattern))
            case _:
                # Other pattern types don't contain variables
                pass
        return variables

    def compile(self, program: Program) -> str:
        lines = []

        # Add type alias for Char
        lines.append("# Type aliases")
        lines.append("Char = str")
        lines.append("")

        with open("lango/systemo/compiler/python/prelude.py", "r") as f:
            lines.extend(f.read().splitlines())

        # Initialize type dictionaries (but don't generate code yet)
        self._initialize_type_dictionaries()

        # Collect data types
        for stmt in program.statements:
            match stmt:
                case DataDeclaration(type_name=type_name):
                    self.data_types[type_name] = stmt
                case _:
                    pass

        # Group function definitions by name
        function_definitions: Dict[str, List[FunctionDefinition]] = {}
        # Group instance declarations by name
        instance_declarations: Dict[str, List["InstanceDeclaration"]] = {}

        for stmt in program.statements:
            match stmt:
                case DataDeclaration():
                    lines.append(self._compile_data_declaration(stmt))
                case FunctionDefinition(function_name=function_name):
                    if function_name not in function_definitions:
                        function_definitions[function_name] = []
                    function_definitions[function_name].append(stmt)
                case InstanceDeclaration(instance_name=instance_name):
                    if instance_name not in instance_declarations:
                        instance_declarations[instance_name] = []
                    instance_declarations[instance_name].append(stmt)
                case LetStatement(variable=variable, value=value):
                    prefixed_var = self._prefix_name(variable)
                    lines.append(
                        f"{prefixed_var} = {self._compile_expression(value)}",
                    )

        # Store instance declarations and function definitions for later use
        self.instance_declarations_registry = instance_declarations
        self.function_definitions = function_definitions

        # Add user-defined operations to type dictionaries
        self._add_user_defined_operations_to_dictionaries(instance_declarations)

        # Now generate dictionary definitions including user-defined dictionaries
        lines.extend(self._generate_dictionary_definitions())

        # Generate generic polymorphic functions that use type dictionaries
        for instance_name, instances in instance_declarations.items():
            lines.extend(self._compile_generic_instances(instance_name, instances))

        # Add any dynamically generated functions
        if self.generated_functions:
            lines.extend(["", "# Dynamically generated specific instances"])
            lines.extend(self.generated_functions)

        # Collect concrete polymorphic instantiations needed in the program
        for stmt in program.statements:
            match stmt:
                case FunctionDefinition(body=body):
                    self._collect_concrete_polymorphic_instantiations(body)
                case InstanceDeclaration(function_definition=func_def):
                    self._collect_concrete_polymorphic_instantiations(func_def.body)
                case LetStatement(value=value):
                    self._collect_concrete_polymorphic_instantiations(value)
                case _:
                    pass

        # Generate specialized functions for concrete instantiations
        if self.concrete_instantiations:
            lines.extend(["", "# Specialized polymorphic function instances"])
            lines.extend(self._generate_concrete_function_instances())

        # Generate function definitions AFTER monomorphized functions are registered
        for func_name, definitions in function_definitions.items():
            lines.append(self._compile_function_group(func_name, definitions))

        # Add any dynamically generated functions before main execution
        if self.generated_functions:
            lines.extend(["", "# Dynamically generated specific instances"])
            lines.extend(self.generated_functions)

        # Generate monomorphic arithmetic functions if any were created during function compilation
        if (
            hasattr(self, "_generated_monomorphic_arithmetic")
            and self._generated_monomorphic_arithmetic
        ):
            lines.extend(["", "# Monomorphic arithmetic operation functions"])
            lines.extend(self._generated_monomorphic_arithmetic)

        # Add main execution
        if "main" in function_definitions:
            lines.extend(["", "if __name__ == '__main__':", "    systemo_main()"])

        return "\n".join(lines)

    def _compile_data_declaration(self, data_decl: DataDeclaration) -> str:
        lines = [f"# Data type: {data_decl.type_name}"]

        for constructor in data_decl.constructors:
            class_name = constructor.name

            # Handle both record constructors and type atom constructors
            if constructor.record_constructor:
                # Named fields like Person { id_ :: Int, name :: String }
                # Use a dictionary to store named fields
                field_names = [
                    field.name for field in constructor.record_constructor.fields
                ]
                field_types = [
                    field.field_type for field in constructor.record_constructor.fields
                ]

                # Create typed arguments
                typed_args = []
                for i, (field_name, field_type_expr) in enumerate(
                    zip(field_names, field_types),
                ):
                    # Convert field type expression to Python type hint
                    if field_type_expr:
                        converted_type = self._convert_type_expression_to_type(
                            field_type_expr,
                        )
                        type_hint = self._systemo_type_to_python_hint(converted_type)
                    else:
                        type_hint = "Any"
                    typed_args.append(f"arg_{i}: {type_hint}")

                lines.extend(
                    [
                        f"class {class_name}:",
                        f"    def __init__(self, {', '.join(typed_args)}) -> None:",
                    ],
                )
                # Store fields in a dictionary by name
                lines.append("        self.fields = {")
                for i, field_name in enumerate(field_names):
                    lines.append(f"            '{field_name}': arg_{i},")
                lines.append("        }")
            elif constructor.type_atoms:
                # Positional fields like MkPoint Float Float
                arg_count = len(constructor.type_atoms)

                # Create typed arguments
                typed_args = []
                for i, type_atom in enumerate(constructor.type_atoms):
                    # Convert type atom to Python type hint
                    if type_atom:
                        converted_type = self._convert_type_expression_to_type(
                            type_atom,
                        )
                        type_hint = self._systemo_type_to_python_hint(converted_type)
                    else:
                        type_hint = "Any"
                    typed_args.append(f"arg_{i}: {type_hint}")

                lines.extend(
                    [
                        f"class {class_name}:",
                        f"    def __init__(self, {', '.join(typed_args)}) -> None:",
                    ],
                )
                for i, arg in enumerate(range(arg_count)):
                    lines.append(f"        self.arg_{i} = arg_{i}")
            else:
                # No arguments
                lines.extend(
                    [
                        f"class {class_name}:",
                        "    def __init__(self) -> None:",
                        "        pass",
                    ],
                )

        lines.append("")
        return "\n".join(lines)

    def _compile_generic_instances(
        self,
        instance_name: str,
        instances: List["InstanceDeclaration"],
    ) -> List[str]:
        lines: List[str] = []

        # Extract the actual function name from parsing artifacts
        actual_function_name = self._extract_function_name(instance_name)

        # Use the same sanitization method as for symbolic operations
        safe_instance_name = self._sanitize_operator_name(actual_function_name)

        # Ensure it starts with a letter for valid Python identifier
        if not safe_instance_name or not safe_instance_name[0].isalpha():
            safe_instance_name = f"op_{safe_instance_name}"

        # Skip generating generic functions for functions already defined in prelude
        prelude_functions = {"error"}
        if actual_function_name in prelude_functions:
            return lines

        # Generate a single generic function that uses type dictionaries
        prefixed_name = f"systemo_{safe_instance_name}"

        if (
            prefixed_name in self.generic_functions
            or prefixed_name in self.generated_dispatchers
        ):
            return lines  # Already generated

        self.generic_functions.add(prefixed_name)

        # Determine arity from the first instance
        if not instances:
            return lines

        first_instance = instances[0]
        arity = len(first_instance.function_definition.patterns)

        # Generate generic function based on arity
        if arity == 1:
            lines.extend(
                self._generate_unary_generic_function(
                    prefixed_name,
                    safe_instance_name,
                    instances,
                ),
            )
        elif arity == 2:
            lines.extend(
                self._generate_binary_generic_function(
                    prefixed_name,
                    safe_instance_name,
                    instances,
                ),
            )
        else:
            # For higher arity, fallback to original behavior for now
            pass
        return lines

    def _generate_unary_generic_function(
        self,
        prefixed_name: str,
        operation_name: str,
        instances: List["InstanceDeclaration"],
    ) -> List[str]:
        lines = [
            f"def {prefixed_name}(type_dict: Dict[str, Any], x: Any) -> Any:",
            f"    if '{operation_name}' in type_dict:",
            f"        return type_dict['{operation_name}'](x)",
            f"    else:",
            f"        raise ValueError(f'Operation {operation_name} not supported for type {{type(x).__name__}}')",
            "",
        ]
        return lines

    def _generate_binary_generic_function(
        self,
        prefixed_name: str,
        operation_name: str,
        instances: List["InstanceDeclaration"],
    ) -> List[str]:
        lines = [
            f"def {prefixed_name}(type_dict: Dict[str, Any], x: Any, y: Any) -> Any:",
            f"    if '{operation_name}' in type_dict:",
            f"        return type_dict['{operation_name}'](x, y)",
            f"    else:",
            f"        raise ValueError(f'Operation {operation_name} not supported for type {{type(x).__name__}}')",
            "",
        ]
        return lines

    def _extract_function_name(self, instance_name: Any) -> str:
        # Handle Tree objects from lark parser
        if hasattr(instance_name, "children") and hasattr(instance_name, "data"):
            # This is a Tree object from lark
            if instance_name.children:
                child = instance_name.children[0]
                if hasattr(child, "value"):
                    # This is a Token with a value
                    return child.value
                elif isinstance(child, str):
                    return child
                else:
                    # Handle other types by converting to string
                    return str(child)

        # Handle string representations (fallback for existing code)
        if isinstance(instance_name, str):
            import re

            # Look for operator patterns like Tree(Token('RULE', 'inst_operator_name'), ['/'])
            operator_match = re.search(
                r"Tree\(Token\('RULE', 'inst_operator_name'\), \['([^']+)'\]\)",
                instance_name,
            )
            if operator_match:
                return operator_match.group(1)

            # Look for patterns like Token('ID', 'xcoord') in the string
            token_match = re.search(r"Token\('ID', '(\w+)'\)", instance_name)
            if token_match:
                return token_match.group(1)

        # Final fallback
        return str(instance_name)

        # Look for simple word patterns
        word_match = re.search(r"\b([a-zA-Z]\w*)\b", instance_name)
        if word_match:
            return word_match.group(1)

        # Fallback
        return "instance"

    def _legacy_ensure_specific_monomorphic_function(
        self,
        operator: str,
        safe_op_name: str,
        param1_type: str,
        param2_type: str,
        result_type: str,
    ) -> None:
        # Legacy compatibility - keep monomorphic function tracking
        type_signature = f"{param1_type} -> {param2_type} -> {result_type}"
        monomorphic_key = f"{safe_op_name}+{type_signature}"

        # Generate the monomorphic function name
        func_name = f"systemo_{safe_op_name}_{param1_type}_{param2_type}_{result_type}"
        self.monomorphic_functions[monomorphic_key] = func_name

        # Generate the function body that delegates to primitive operations
        if not hasattr(self, "_generated_monomorphic_arithmetic"):
            self._generated_monomorphic_arithmetic = []

        # Handle equality operators for mixed types
        if safe_op_name == "eqeq" and param1_type != param2_type:
            # Different types are never equal
            func_body = [
                f"def {func_name}(arg_0: Any, arg_1: Any) -> Any:",
                f"    if True:",
                f"        x = arg_0",
                f"        y = arg_1",
                f"        return False",
                f"    raise ValueError(f'No matching pattern for {func_name} with args: {{arg_0, arg_1}}')",
                "",
            ]
            self._generated_monomorphic_arithmetic.extend(func_body)
            return

        if param1_type == "Int" and param2_type == "Int" and result_type == "Int":
            # Map operator names to primitive function names
            prim_name_map = {
                "plus": "Add",
                "minus": "Sub",
                "star": "Mul",
                "slash": "Div",
            }
            prim_suffix = prim_name_map.get(safe_op_name, safe_op_name.capitalize())
            prim_func = f"primInt{prim_suffix}"
        elif (
            param1_type == "Float" and param2_type == "Float" and result_type == "Float"
        ):
            # Map operator names to primitive function names
            prim_name_map = {
                "plus": "Add",
                "minus": "Sub",
                "star": "Mul",
                "slash": "Div",
            }
            prim_suffix = prim_name_map.get(safe_op_name, safe_op_name.capitalize())
            prim_func = f"primFloat{prim_suffix}"
        elif param1_type == "Int" and param2_type == "Float" and result_type == "Float":
            prim_name_map = {
                "plus": "Add",
                "minus": "Sub",
                "star": "Mul",
                "slash": "Div",
            }
            prim_suffix = prim_name_map.get(safe_op_name, safe_op_name.capitalize())
            prim_func = f"primFloat{prim_suffix}"
            func_body = [
                f"def {func_name}(x, y):",
                f"    return {prim_func}(float(x), y)",
                "",
            ]
            self._generated_monomorphic_arithmetic.extend(func_body)
            return
        elif param1_type == "Float" and param2_type == "Int" and result_type == "Float":
            prim_name_map = {
                "plus": "Add",
                "minus": "Sub",
                "star": "Mul",
                "slash": "Div",
            }
            prim_suffix = prim_name_map.get(safe_op_name, safe_op_name.capitalize())
            prim_func = f"primFloat{prim_suffix}"
            func_body = [
                f"def {func_name}(x, y):",
                f"    return {prim_func}(x, float(y))",
                "",
            ]
            self._generated_monomorphic_arithmetic.extend(func_body)
            return
        else:
            return  # Skip unsupported combinations

        # Standard case without coercion
        func_body = [
            f"def {func_name}(x, y):",
            f"    return {prim_func}(x, y)",
            "",
        ]
        self._generated_monomorphic_arithmetic.extend(func_body)

    def _ensure_arithmetic_monomorphic_functions(
        self,
        operator: str,
        safe_op_name: str,
    ) -> None:
        # Generate monomorphic functions for common type combinations
        type_combinations = [
            ("Int", "Int", "Int"),  # Int + Int -> Int
            ("Float", "Float", "Float"),  # Float + Float -> Float
            ("Int", "Float", "Float"),  # Int + Float -> Float (with coercion)
            ("Float", "Int", "Float"),  # Float + Int -> Float (with coercion)
        ]

        for param1_type, param2_type, result_type in type_combinations:
            # Create the monomorphic function signature
            type_signature = f"{param1_type} -> {param2_type} -> {result_type}"
            monomorphic_key = f"{safe_op_name}+{type_signature}"

            if monomorphic_key not in self.monomorphic_functions:
                # Generate the monomorphic function name
                func_name = (
                    f"systemo_{safe_op_name}_{param1_type}_{param2_type}_{result_type}"
                )
                self.monomorphic_functions[monomorphic_key] = func_name

                # Generate the function body that delegates to primitive operations
                if not hasattr(self, "_generated_monomorphic_arithmetic"):
                    self._generated_monomorphic_arithmetic = []

                if (
                    param1_type == "Int"
                    and param2_type == "Int"
                    and result_type == "Int"
                ):
                    # Map operator names to primitive function names
                    prim_name_map = {
                        "plus": "Add",
                        "minus": "Sub",
                        "star": "Mul",
                        "slash": "Div",
                    }
                    prim_suffix = prim_name_map.get(
                        safe_op_name,
                        safe_op_name.capitalize(),
                    )
                    prim_func = f"primInt{prim_suffix}"
                    func_body = [
                        f"def {func_name}(x, y):",
                        f"    return {prim_func}(x, y)",
                        "",
                    ]
                elif (
                    param1_type == "Float"
                    and param2_type == "Float"
                    and result_type == "Float"
                ):
                    prim_name_map = {
                        "plus": "Add",
                        "minus": "Sub",
                        "star": "Mul",
                        "slash": "Div",
                    }
                    prim_suffix = prim_name_map.get(
                        safe_op_name,
                        safe_op_name.capitalize(),
                    )
                    prim_func = f"primFloat{prim_suffix}"
                    func_body = [
                        f"def {func_name}(x, y):",
                        f"    return {prim_func}(x, y)",
                        "",
                    ]
                elif (
                    param1_type == "Int"
                    and param2_type == "Float"
                    and result_type == "Float"
                ):
                    prim_name_map = {
                        "plus": "Add",
                        "minus": "Sub",
                        "star": "Mul",
                        "slash": "Div",
                    }
                    prim_suffix = prim_name_map.get(
                        safe_op_name,
                        safe_op_name.capitalize(),
                    )
                    prim_func = f"primFloat{prim_suffix}"
                    func_body = [
                        f"def {func_name}(x, y):",
                        f"    return {prim_func}(float(x), y)",
                        "",
                    ]
                elif (
                    param1_type == "Float"
                    and param2_type == "Int"
                    and result_type == "Float"
                ):
                    prim_name_map = {
                        "plus": "Add",
                        "minus": "Sub",
                        "star": "Mul",
                        "slash": "Div",
                    }
                    prim_suffix = prim_name_map.get(
                        safe_op_name,
                        safe_op_name.capitalize(),
                    )
                    prim_func = f"primFloat{prim_suffix}"
                    func_body = [
                        f"def {func_name}(x, y):",
                        f"    return {prim_func}(x, float(y))",
                        "",
                    ]
                else:
                    continue  # Skip unsupported combinations

                self._generated_monomorphic_arithmetic.extend(func_body)

    def _generate_type_dispatcher_simple(
        self,
        instance_name: str,
        instances: List["InstanceDeclaration"],
    ) -> str:
        prefixed_name = f"systemo_{instance_name}"
        lines = [f"def {prefixed_name}(arg: Any) -> Any:"]

        if len(instances) == 0:
            lines.append(f"    raise ValueError(f'No instances of {instance_name}')")
            lines.append("")
            return "\n".join(lines)

        # Try each instance in sequence using individual try/except blocks
        for i, instance in enumerate(instances):
            monomorphized_name = f"systemo_{instance_name}_{i}"
            lines.append(f"    try:")

            # Check if this is a binary function (curried) by examining the function definition patterns
            func_def = instance.function_definition
            if len(func_def.patterns) == 2:
                # This is a binary function - return the partial application with the first argument
                lines.append(f"        return {monomorphized_name}(arg)")
            else:
                # This is a unary function - call it directly
                lines.append(f"        return {monomorphized_name}(arg)")
            lines.append(f"    except (ValueError, AttributeError):")
            lines.append(f"        pass")

        # Final fallback
        lines.append(
            f"    raise ValueError(f'No instance of {instance_name} for type {{type(arg).__name__}}')",
        )

        lines.append("")
        return "\n".join(lines)

    def _generate_type_dispatcher(
        self,
        instance_name: str,
        instances: List["InstanceDeclaration"],
    ) -> str:
        prefixed_name = f"systemo_{instance_name}"
        lines = [f"def {prefixed_name}(arg: Any) -> Any:"]

        # Generate type checks for each instance
        type_handled = False
        for i, instance in enumerate(instances):
            param_type = self._extract_first_param_type_name(instance.type_signature)

            if param_type and param_type in self.data_types:
                # Check for constructor types
                constructor_names = [
                    ctor.name for ctor in self.data_types[param_type].constructors
                ]

                if constructor_names:
                    # Generate type check based on constructor
                    type_checks = [
                        f"type(arg).__name__ == '{ctor_name}'"
                        for ctor_name in constructor_names
                    ]
                    condition = " or ".join(type_checks)
                    monomorphized_name = f"systemo_{instance_name}_{param_type}"
                    lines.append(f"    if {condition}:")
                    lines.append(f"        return {monomorphized_name}(arg)")
                    type_handled = True

        if not type_handled:
            # Fallback: if we can't determine types, just use the first implementation
            if instances:
                monomorphized_name = f"systemo_{instance_name}_0"
                lines.append(f"    return {monomorphized_name}(arg)")
            else:
                lines.append(
                    f"    raise ValueError(f'No instances of {instance_name}')",
                )
        else:
            # Fallback error for unknown types
            lines.append(
                f"    raise ValueError(f'No instance of {instance_name} for type {{type(arg).__name__}}')",
            )

        lines.append("")
        return "\n".join(lines)

    def _compile_function_group(
        self,
        func_name: str,
        definitions: List[FunctionDefinition],
    ) -> str:
        # Avoid conflicts with reserved function names by adding suffix
        if func_name in self.reserved_function_names:
            prefixed_func_name = f"systemo_{func_name}_user"
        else:
            prefixed_func_name = f"systemo_{func_name}"

        # Skip generating function definition if this is a polymorphic function
        # that already has a dispatcher generated
        if prefixed_func_name in self.generated_dispatchers:
            return ""  # Don't generate duplicate function

        self.defined_functions.add(func_name)

        # Store function type information
        if definitions and definitions[0].ty:
            self.function_types[func_name] = definitions[0].ty

        # Check if any definition is nullary
        if any(len(defn.patterns) == 0 for defn in definitions):
            self.nullary_functions.add(func_name)

        # Check if this is a polymorphic function
        is_polymorphic = self._is_polymorphic_function(func_name)

        if len(definitions) == 1 and len(definitions[0].patterns) <= 1:
            return self._compile_simple_function(definitions[0], prefixed_func_name)

        # Get return type hint from the first function definition
        return_type_hint = "Any"
        if definitions and definitions[0].ty:
            # For function types, extract the final return type
            current_type = definitions[0].ty
            while isinstance(current_type, FunctionType):
                current_type = current_type.result
            return_type_hint = self._systemo_type_to_python_hint(current_type)

        # Find maximum number of parameters needed
        max_params = (
            max(len(defn.patterns) for defn in definitions) if definitions else 0
        )

        # Use standard function signature for multi-parameter functions
        if max_params > 1:
            # For multi-parameter functions, get parameter types from the function type
            param_types: List[str] = []
            if definitions and definitions[0].ty:
                current_type = definitions[0].ty
                while (
                    isinstance(current_type, FunctionType)
                    and len(param_types) < max_params
                ):
                    param_type_hint = self._systemo_type_to_python_hint(
                        current_type.param,
                    )
                    param_types.append(param_type_hint)
                    current_type = current_type.result

            # Fill remaining with Any if we don't have enough type information
            while len(param_types) < max_params:
                param_types.append("Any")

            param_list = [f"arg_{i}: {param_types[i]}" for i in range(max_params)]

            if is_polymorphic:
                # Add dictionary parameter for polymorphic functions
                param_list.insert(0, "type_dict: Dict[str, Any]")

            lines = [
                f"def {prefixed_func_name}({', '.join(param_list)}) -> {return_type_hint}:",
            ]
        else:
            # For single parameter functions, try to get the parameter type
            param_type = "Any"
            if (
                definitions
                and definitions[0].ty
                and isinstance(definitions[0].ty, FunctionType)
            ):
                param_type = self._systemo_type_to_python_hint(definitions[0].ty.param)

            if is_polymorphic:
                # Add dictionary parameter for polymorphic functions
                lines = [
                    f"def {prefixed_func_name}(type_dict: Dict[str, Any], arg_0: {param_type}) -> {return_type_hint}:",
                ]
            else:
                lines = [
                    f"def {prefixed_func_name}(arg_0: {param_type}) -> {return_type_hint}:",
                ]

        self.indent_level += 1

        # Collect all pattern variables from all definitions
        old_local_vars = self.local_variables.copy()
        for func_def in definitions:
            for pattern in func_def.patterns:
                self.local_variables.update(self._extract_pattern_variables(pattern))

        # Track if we have exhaustive patterns (ending with catch-all)
        has_exhaustive_patterns = False

        for i, func_def in enumerate(definitions):
            if len(func_def.patterns) == 0:
                # Nullary function - no arguments expected
                lines.append(
                    self._indent()
                    + f"return {self._compile_expression(func_def.body)}",
                )
                has_exhaustive_patterns = True  # Nullary is always exhaustive
            else:
                # Pattern matching - check each pattern
                pattern_matches = []
                assignments = []

                for j, pattern in enumerate(func_def.patterns):
                    arg_name = f"arg_{j}"
                    match pattern:
                        case VariablePattern(name=name):
                            assignments.append(f"{name} = {arg_name}")
                        case LiteralPattern(value=value):
                            pattern_matches.append(
                                f"{arg_name} == {self._compile_literal_value(value)}",
                            )
                        case ConsPattern(head=head, tail=tail):
                            # Cons pattern (x:xs) - check if list is non-empty and destructure
                            pattern_matches.append(f"len({arg_name}) > 0")
                            match head:
                                case VariablePattern(name=name):
                                    assignments.append(f"{name} = {arg_name}[0]")
                                case _:
                                    pass
                            match tail:
                                case VariablePattern(name=name):
                                    assignments.append(f"{name} = {arg_name}[1:]")
                                case _:
                                    pass
                        case TuplePattern(patterns=patterns):
                            # Tuple pattern - check tuple length and destructure
                            pattern_matches.append(
                                f"len({arg_name}) == {len(patterns)}",
                            )
                            for i, sub_pattern in enumerate(patterns):
                                match sub_pattern:
                                    case VariablePattern(name=name):
                                        assignments.append(f"{name} = {arg_name}[{i}]")
                                    case _:
                                        # For non-variable patterns, add recursive matching
                                        pass
                        case ListPattern(patterns=patterns):
                            # List pattern - check list length and destructure
                            pattern_matches.append(
                                f"len({arg_name}) == {len(patterns)}",
                            )
                            for i, sub_pattern in enumerate(patterns):
                                match sub_pattern:
                                    case VariablePattern(name=name):
                                        assignments.append(f"{name} = {arg_name}[{i}]")
                                    case _:
                                        # For non-variable patterns, add recursive matching
                                        pass
                        case ConstructorPattern(
                            constructor=constructor,
                            patterns=sub_patterns,
                        ):
                            # Constructor pattern - check type and destructure
                            pattern_matches.append(
                                f"type({arg_name}).__name__ == '{constructor}'",
                            )

                            constructor_def = self._find_constructor_def(constructor)

                            if constructor_def and constructor_def.record_constructor:
                                # Record constructor - use dictionary access by field name
                                for k, sub_pattern in enumerate(sub_patterns):
                                    match sub_pattern:
                                        case VariablePattern(name=name):
                                            field_name = constructor_def.record_constructor.fields[
                                                k
                                            ].name
                                            assignments.append(
                                                f"{name} = {arg_name}.fields['{field_name}']",
                                            )
                                        case _:
                                            pass
                            else:
                                # Positional constructor - use arg_ access
                                for k, sub_pattern in enumerate(sub_patterns):
                                    match sub_pattern:
                                        case VariablePattern(name=name):
                                            assignments.append(
                                                f"{name} = {arg_name}.arg_{k}",
                                            )
                                        case _:
                                            pass
                            # Could add more sub-pattern types here if needed

                # Generate pattern matching condition
                if pattern_matches:
                    condition = " and ".join(pattern_matches)
                    lines.append(self._indent() + f"if {condition}:")
                    self.indent_level += 1
                    for assignment in assignments:
                        lines.append(self._indent() + assignment)
                    lines.append(
                        self._indent()
                        + f"return {self._compile_expression(func_def.body)}",
                    )
                    self.indent_level -= 1
                else:
                    # Only variable patterns, always matches - this is a catch-all
                    for assignment in assignments:
                        lines.append(self._indent() + assignment)
                    lines.append(
                        self._indent()
                        + f"return {self._compile_expression(func_def.body)}",
                    )
                    # If this is the last definition and only has variable patterns, it's exhaustive
                    if i == len(definitions) - 1:
                        has_exhaustive_patterns = True

        # Only add fallback error if patterns are not exhaustive
        if not has_exhaustive_patterns:
            if max_params > 1:
                # For curried functions, show the individual arguments
                arg_names = ", ".join([f"arg_{i}" for i in range(max_params)])
                error_msg = f"raise ValueError(f'No matching pattern for {prefixed_func_name} with args: {{{arg_names}}}')"
            else:
                # For single parameter functions, show arg_0
                error_msg = f"raise ValueError(f'No matching pattern for {prefixed_func_name} with args: {{arg_0}}')"

            lines.append(self._indent() + error_msg)
        self.indent_level -= 1

        # Restore local variables
        self.local_variables = old_local_vars

        lines.append("")
        return "\n".join(lines)

    def _compile_binary_instance_function(
        self,
        func_def: FunctionDefinition,
        prefixed_name: str,
    ) -> str:
        lines = [
            f"def {prefixed_name}(arg_0: Any, arg_1: Any) -> Any:",
        ]

        self.indent_level += 1

        # Track pattern variables
        old_local_vars = self.local_variables.copy()

        # Extract variables from patterns
        for pattern in func_def.patterns:
            self.local_variables.update(self._extract_pattern_variables(pattern))

        # Generate pattern matching for both arguments
        pattern_0 = func_def.patterns[0]
        pattern_1 = func_def.patterns[1]

        # Build the pattern matching logic
        condition_parts = []
        assignments = []

        # Handle first pattern
        match pattern_0:
            case LiteralPattern(value=value):
                condition_parts.append(f"arg_0 == {self._compile_literal_value(value)}")
            case VariablePattern(name=name):
                # Don't assign if the name is an operator symbol or contains parentheses/operators
                # Skip assignment for patterns that look like operator names: (?), (+), etc.
                if name.isidentifier() and not (
                    name.startswith("(") and name.endswith(")")
                ):
                    assignments.append(f"{name} = arg_0")
            case ListPattern(patterns=patterns):
                # List pattern - check list length and destructure
                condition_parts.append(f"len(arg_0) == {len(patterns)}")
                for i, sub_pattern in enumerate(patterns):
                    match sub_pattern:
                        case VariablePattern(name=name):
                            assignments.append(f"{name} = arg_0[{i}]")
            case ConsPattern(head=head, tail=tail):
                # Cons pattern (x:xs) - check if list is non-empty and destructure
                condition_parts.append("len(arg_0) > 0")
                match head:
                    case VariablePattern(name=name):
                        assignments.append(f"{name} = arg_0[0]")
                match tail:
                    case VariablePattern(name=name):
                        assignments.append(f"{name} = arg_0[1:]")
            case _:
                # More complex patterns would need additional handling
                condition_parts.append("True")  # Placeholder

        # Handle second pattern
        match pattern_1:
            case LiteralPattern(value=value):
                condition_parts.append(f"arg_1 == {self._compile_literal_value(value)}")
            case VariablePattern(name=name):
                # Don't assign if the name is an operator symbol or contains parentheses/operators
                # Skip assignment for patterns that look like operator names: (?), (+), etc.
                if name.isidentifier() and not (
                    name.startswith("(") and name.endswith(")")
                ):
                    assignments.append(f"{name} = arg_1")
            case ListPattern(patterns=patterns):
                # List pattern - check list length and destructure
                condition_parts.append(f"len(arg_1) == {len(patterns)}")
                for i, sub_pattern in enumerate(patterns):
                    match sub_pattern:
                        case VariablePattern(name=name):
                            assignments.append(f"{name} = arg_1[{i}]")
            case ConsPattern(head=head, tail=tail):
                # Cons pattern (x:xs) - check if list is non-empty and destructure
                condition_parts.append("len(arg_1) > 0")
                match head:
                    case VariablePattern(name=name):
                        assignments.append(f"{name} = arg_1[0]")
                match tail:
                    case VariablePattern(name=name):
                        assignments.append(f"{name} = arg_1[1:]")
            case _:
                # More complex patterns would need additional handling
                condition_parts.append("True")  # Placeholder

        # Build the complete condition
        if condition_parts:
            condition = " and ".join(condition_parts)
            lines.append(f"{self._indent()}if {condition}:")
        else:
            # All patterns are variables, no condition needed
            lines.append(f"{self._indent()}if True:")

        self.indent_level += 1

        # Add variable assignments
        for assignment in assignments:
            lines.append(f"{self._indent()}{assignment}")

        # Workaround for parsing issue: if the body references variables that aren't assigned,
        # try to map them from positional arguments. This handles cases like a (?) b = a + b
        # where the pattern is parsed as [(?, b)] but the body uses [a, b]
        body_text = str(func_def.body)
        assigned_vars = {assignment.split(" = ")[0] for assignment in assignments}

        # Common variable names that might be missing due to operator parsing issues
        if "a" in body_text and "a" not in assigned_vars:
            lines.append(f"{self._indent()}a = arg_0")
            self.local_variables.add("a")
        if "x" in body_text and "x" not in assigned_vars:
            lines.append(f"{self._indent()}x = arg_0")
            self.local_variables.add("x")

        # Compile the function body
        lines.append(
            f"{self._indent()}return {self._compile_expression(func_def.body)}",
        )

        self.indent_level -= 1

        # Add fallback
        lines.append(
            f"{self._indent()}raise ValueError(f'No matching pattern for {prefixed_name} with args: {{arg_0, arg_1}}')",
        )

        self.indent_level -= 1
        self.local_variables = old_local_vars

        lines.append("")
        return "\n".join(lines)

    def _compile_simple_function(
        self,
        func_def: FunctionDefinition,
        prefixed_name: Optional[str] = None,
    ) -> str:
        if prefixed_name:
            func_name = prefixed_name
        elif func_def.function_name in self.reserved_function_names:
            func_name = f"systemo_{func_def.function_name}_user"
        else:
            func_name = f"systemo_{func_def.function_name}"

        # Track pattern variables
        old_local_vars = self.local_variables.copy()
        for pattern in func_def.patterns:
            self.local_variables.update(self._extract_pattern_variables(pattern))

        # Get type hints from the function's type annotation
        return_type_hint = "Any"
        param_type_hint = "Any"

        # Check if this is a polymorphic function that needs dictionary parameter
        is_polymorphic = self._is_polymorphic_function(func_def.function_name)

        if func_def.ty:
            match func_def.ty:
                case FunctionType(param=param_type, result=result_type):
                    param_type_hint = self._systemo_type_to_python_hint(param_type)
                    return_type_hint = self._systemo_type_to_python_hint(result_type)
                case _:
                    # Not a function type, use it as return type
                    return_type_hint = self._systemo_type_to_python_hint(func_def.ty)

        if len(func_def.patterns) == 0:
            # Nullary function
            if is_polymorphic:
                # Polymorphic nullary function needs dictionary parameter
                # Check if the body is a do block with multiple statements
                match func_def.body:
                    case DoBlock(statements=statements) if len(statements) > 1:
                        lines = [
                            f"def {func_name}(type_dict: Dict[str, Any]) -> {return_type_hint}:",
                        ]
                        lines.extend(
                            self._compile_do_block_as_statements(func_def.body),
                        )
                    case _:
                        compiled_body = self._compile_expression(func_def.body)
                        lines = [
                            f"def {func_name}(type_dict: Dict[str, Any]) -> {return_type_hint}:",
                            f"    return {compiled_body}",
                        ]
            else:
                # Non-polymorphic nullary function
                # Check if the body is a do block with multiple statements
                match func_def.body:
                    case DoBlock(statements=statements) if len(statements) > 1:
                        lines = [f"def {func_name}() -> {return_type_hint}:"]
                        lines.extend(
                            self._compile_do_block_as_statements(func_def.body),
                        )
                    case _:
                        # Compile the expression first to see if it's a partial application
                        compiled_body = self._compile_expression(func_def.body)

                        # Check if the compiled expression is a lambda (partial application)
                        if compiled_body.startswith("lambda "):
                            # Extract lambda parameters and body
                            # Format: "lambda param1, param2: body"
                            lambda_part = compiled_body[7:]  # Remove "lambda "
                            colon_index = lambda_part.find(":")
                            if colon_index != -1:
                                params_str = lambda_part[:colon_index].strip()
                                lambda_body = lambda_part[colon_index + 1 :].strip()

                                # Add type hints to parameters
                                if params_str:
                                    # Mark this function as lambda-lifted (not truly nullary)
                                    self.lambda_lifted_functions.add(
                                        func_def.function_name,
                                    )
                                    params_with_types = ", ".join(
                                        f"{param}: Any"
                                        for param in params_str.split(", ")
                                    )
                                    lines = [
                                        f"def {func_name}({params_with_types}) -> {return_type_hint}:",
                                        f"    return {lambda_body}",
                                    ]
                                else:
                                    # No parameters in lambda
                                    lines = [
                                        f"def {func_name}() -> {return_type_hint}:",
                                        f"    return {lambda_body}",
                                    ]
                            else:
                                # Fallback to original behavior
                                lines = [
                                    f"def {func_name}() -> {return_type_hint}:",
                                    f"    return {compiled_body}",
                                ]
                        else:
                            lines = [
                                f"def {func_name}() -> {return_type_hint}:",
                                f"    return {compiled_body}",
                            ]
        else:
            pattern = func_def.patterns[0]
            match pattern:
                case VariablePattern(name=name):
                    if is_polymorphic:
                        # Polymorphic function with one parameter
                        # Check if the body is a do block with multiple statements
                        match func_def.body:
                            case DoBlock(statements=statements) if len(statements) > 1:
                                lines = [
                                    f"def {func_name}(type_dict: Dict[str, Any], {name}: {param_type_hint}) -> {return_type_hint}:",
                                ]
                                lines.extend(
                                    self._compile_do_block_as_statements(func_def.body),
                                )
                            case _:
                                lines = [
                                    f"def {func_name}(type_dict: Dict[str, Any], {name}: {param_type_hint}) -> {return_type_hint}:",
                                    f"    return {self._compile_expression(func_def.body)}",
                                ]
                    else:
                        # Non-polymorphic function with one parameter
                        # Check if the body is a do block with multiple statements
                        match func_def.body:
                            case DoBlock(statements=statements) if len(statements) > 1:
                                lines = [
                                    f"def {func_name}({name}: {param_type_hint}) -> {return_type_hint}:",
                                ]
                                lines.extend(
                                    self._compile_do_block_as_statements(func_def.body),
                                )
                            case _:
                                lines = [
                                    f"def {func_name}({name}: {param_type_hint}) -> {return_type_hint}:",
                                    f"    return {self._compile_expression(func_def.body)}",
                                ]
                case _:
                    if is_polymorphic:
                        # Polymorphic function with pattern matching
                        lines = [
                            f"def {func_name}(type_dict: Dict[str, Any], arg: {param_type_hint}) -> {return_type_hint}:",
                            f"    {self._compile_pattern_match(pattern, 'arg', func_def.body)}",
                        ]
                    else:
                        # Non-polymorphic function with pattern matching
                        lines = [
                            f"def {func_name}(arg: {param_type_hint}) -> {return_type_hint}:",
                            f"    {self._compile_pattern_match(pattern, 'arg', func_def.body)}",
                        ]

        lines.append("")

        # Restore local variables
        self.local_variables = old_local_vars

        return "\n".join(lines)

    def _compile_do_block_as_statements(self, do_block: DoBlock) -> List[str]:
        lines = []

        # Process all statements except the last one
        for stmt in do_block.statements[:-1]:
            match stmt:
                case LetStatement(variable=variable, value=value):
                    prefixed_var = self._prefix_name(variable)
                    lines.append(
                        f"    {prefixed_var} = {self._compile_expression(value)}",
                    )
                case _ if self._is_expression(stmt):
                    # Handle expression statements (like putStr calls)
                    lines.append(f"    {self._compile_expression_safe(stmt)}")
                case _:
                    pass

        # Handle the last statement (which becomes the return value)
        last_stmt = do_block.statements[-1]
        match last_stmt:
            case LetStatement(variable=variable, value=value):
                prefixed_var = self._prefix_name(variable)
                lines.append(
                    f"    {prefixed_var} = {self._compile_expression(value)}",
                )
                lines.append(f"    return {prefixed_var}")
            case _ if self._is_expression(last_stmt):
                lines.append(f"    return {self._compile_expression_safe(last_stmt)}")
            case _:
                lines.append("    return None")

        return lines

    def _compile_pattern_match(
        self,
        pattern: Pattern,
        value_expr: str,
        body: Expression,
    ) -> str:
        match pattern:
            case VariablePattern(name=name):
                self.local_variables.add(name)
                return f"{name} = {value_expr}\n    return {self._compile_expression(body)}"
            case LiteralPattern(value=value):
                return build_literal_pattern_match(
                    value_expr,
                    value,
                    body,
                    self._compile_expression,
                    self._compile_literal_value,
                )
            case ConsPattern(head=head, tail=tail):
                # Cons pattern (x:xs) - destructure list
                head_var = None
                tail_var = None
                match head:
                    case VariablePattern(name=name):
                        head_var = name
                        self.local_variables.add(head_var)
                    case _:
                        pass
                match tail:
                    case VariablePattern(name=name):
                        tail_var = name
                        self.local_variables.add(tail_var)
                    case _:
                        pass

                return build_cons_pattern_match(
                    value_expr,
                    head_var,
                    tail_var,
                    body,
                    self._compile_expression,
                )
            case TuplePattern(patterns=patterns):
                # Tuple pattern - destructure tuple
                tuple_vars = []
                for i, sub_pattern in enumerate(patterns):
                    match sub_pattern:
                        case VariablePattern(name=name):
                            tuple_vars.append(name)
                            self.local_variables.add(name)
                        case _:
                            tuple_vars.append(f"_tuple_elem_{i}")

                return build_tuple_pattern_match(
                    value_expr,
                    tuple_vars,
                    body,
                    self._compile_expression,
                )
            case ConstructorPattern(constructor=constructor, patterns=patterns):
                # Extract variables from constructor pattern
                for sub_pattern in patterns:
                    self.local_variables.update(
                        self._extract_pattern_variables(sub_pattern),
                    )

                # Generate destructuring assignment for constructor pattern
                if len(patterns) == 1:
                    match patterns[0]:
                        case VariablePattern(name=var_name):
                            constructor_def = self._find_constructor_def(constructor)

                            if constructor_def and constructor_def.record_constructor:
                                # Record constructor - use dictionary access
                                field_name = constructor_def.record_constructor.fields[
                                    0
                                ].name
                                return build_record_pattern_match(
                                    value_expr,
                                    constructor,
                                    field_name,
                                    var_name,
                                    body,
                                    self._compile_expression,
                                )
                            else:
                                # Positional constructor - use arg_ access
                                return build_positional_pattern_match(
                                    value_expr,
                                    constructor,
                                    var_name,
                                    body,
                                    self._compile_expression,
                                    arg_index=0,
                                )
                        case _:
                            # Handle non-variable patterns with single argument
                            return build_simple_pattern_match(
                                value_expr,
                                constructor,
                                body,
                                self._compile_expression,
                            )
                elif len(patterns) > 1:
                    # Multiple variables in constructor pattern
                    constructor_def = self._find_constructor_def(constructor)

                    assignments = []
                    if constructor_def and constructor_def.record_constructor:
                        # Record constructor - use dictionary access
                        for k, sub_pattern in enumerate(patterns):
                            match sub_pattern:
                                case VariablePattern(name=name):
                                    field_name = (
                                        constructor_def.record_constructor.fields[
                                            k
                                        ].name
                                    )
                                    assignments.append(
                                        f"{name} = {value_expr}.fields['{field_name}']",
                                    )
                                case _:
                                    pass
                    else:
                        # Positional constructor - use arg_ access
                        for k, sub_pattern in enumerate(patterns):
                            match sub_pattern:
                                case VariablePattern(name=name):
                                    assignments.append(
                                        f"{name} = {value_expr}.arg_{k}",
                                    )
                                case _:
                                    pass

                    return build_multi_arg_pattern_match(
                        value_expr,
                        constructor,
                        assignments,
                        body,
                        self._compile_expression,
                    )
                else:
                    # Constructor with no arguments
                    return build_simple_pattern_match(
                        value_expr,
                        constructor,
                        body,
                        self._compile_expression,
                    )
            case ListPattern(patterns=patterns):
                # List pattern - destructure list
                list_vars = []
                for i, sub_pattern in enumerate(patterns):
                    match sub_pattern:
                        case VariablePattern(name=name):
                            list_vars.append(name)
                            self.local_variables.add(name)
                        case _:
                            list_vars.append(f"_list_elem_{i}")

                return build_list_pattern_match(
                    value_expr,
                    list_vars,
                    body,
                    self._compile_expression,
                )
            case _:
                return f"return {self._compile_expression(body)}"

    def _compile_literal_value(self, value: Any) -> str:
        return compile_literal_value(value)

    def _compile_symbolic_operation(
        self,
        operator: str,
        operands: List["Expression"],
        ty: Optional[Type] = None,
    ) -> str:

        # Handle binary operators
        if len(operands) == 2:
            left = self._compile_expression(operands[0])
            right = self._compile_expression(operands[1])

            # Get the safe operation name and determine binary function name
            safe_op_name = self._sanitize_operator_name(operator)

            # Use specific names for binary operations to distinguish from unary
            binary_func_name = self._get_binary_function_name(safe_op_name)

            # Check for recursive calls to the current function being compiled
            if (
                self.current_function_name
                and binary_func_name in self.current_function_name
            ):
                # This is a recursive call - use the current function being compiled
                return f"{self.current_function_name}({left}, {right})"

            # Determine type dictionary and operation name from type information
            if ty is not None and all(op.ty is not None for op in operands):
                # Get the type of the first operand to determine which dictionary to use
                arg1_type = self._type_expression_to_string(operands[0].ty)
                dict_name = self._get_type_dict_name(arg1_type)

                # If type inference fails, use runtime dispatch
                if dict_name == "unknown_dict":
                    return f"{binary_func_name}_runtime_dispatch({left}, {right})"

                # Generate dictionary-passing call
                return f"{binary_func_name}({dict_name}, {left}, {right})"
            else:
                # Fallback: try to infer type from operand structure
                inferred_type = self._infer_type_from_operand(operands[0])
                if inferred_type:
                    dict_name = self._get_type_dict_name(inferred_type)
                    # If type inference fails, use runtime dispatch
                    if dict_name == "unknown_dict":
                        return f"{binary_func_name}_runtime_dispatch({left}, {right})"
                    return f"{binary_func_name}({dict_name}, {left}, {right})"
                else:
                    # Improve type inference by checking operand types more carefully
                    left_type = self._infer_type_from_operand(operands[0])
                    if left_type:
                        dict_name = self._get_type_dict_name(left_type)
                        # If type inference fails, use runtime dispatch
                        if dict_name == "unknown_dict":
                            return (
                                f"{binary_func_name}_runtime_dispatch({left}, {right})"
                            )
                        return f"{binary_func_name}({dict_name}, {left}, {right})"
                    else:
                        # Last resort: generate runtime dispatch function call
                        return f"{binary_func_name}_runtime_dispatch({left}, {right})"

        # Handle unary operators
        elif len(operands) == 1:
            operand = self._compile_expression(operands[0])
            safe_op_name = self._sanitize_operator_name(operator)

            # Use specific names for unary operations to distinguish from binary
            unary_func_name = self._get_unary_function_name(safe_op_name)

            # Determine type dictionary from type information
            if ty is not None and operands[0].ty is not None:
                arg_type = self._type_expression_to_string(operands[0].ty)
                dict_name = self._get_type_dict_name(arg_type)

                # If type inference fails, use runtime dispatch
                if dict_name == "unknown_dict":
                    return f"{unary_func_name}_runtime_dispatch({operand})"

                # Generate dictionary-passing call
                return f"{unary_func_name}({dict_name}, {operand})"
            else:
                # Fallback: try to infer type from operand structure
                inferred_type = self._infer_type_from_operand(operands[0])
                if inferred_type:
                    dict_name = self._get_type_dict_name(inferred_type)
                    # If type inference fails, use runtime dispatch
                    if dict_name == "unknown_dict":
                        return f"{unary_func_name}_runtime_dispatch({operand})"
                    return f"{unary_func_name}({dict_name}, {operand})"
                else:
                    # Default fallback for unary operations
                    return f"{unary_func_name}_runtime_dispatch({operand})"

        # Fallback for other cases (should not happen with proper type checking)
        else:
            raise RuntimeError(
                f"Unsupported operator arity: {operator} with {len(operands)} operands",
            )

    def _get_type_dict_name(self, type_name: str) -> str:
        # Normalize the type name and return the corresponding dictionary
        if type_name in ["Int", "int"]:
            return "int_dict"
        elif type_name in ["Float", "float"]:
            return "float_dict"
        elif type_name in ["Bool", "bool"]:
            return "bool_dict"
        elif type_name in ["String", "str"]:
            return "string_dict"
        elif type_name in ["Char", "char"]:
            return "char_dict"
        elif type_name.startswith("List") or type_name.startswith("["):
            return "list_dict"
        elif type_name in ["tuple", "Tuple"] or type_name.startswith("TupleType"):
            return "tuple_dict"
        else:
            # Check if it's a user-defined type with a dictionary
            dict_name = f"{type_name.lower()}_dict"
            if type_name in self.type_dictionaries:
                return dict_name
            else:
                # For truly unknown types, return unknown so we can use runtime dispatch
                return "unknown_dict"

    def _get_binary_function_name(self, safe_op_name: str) -> str:
        # Map specific binary operations to distinct names
        binary_name_map = {
            "minus": "systemo_subtract",  # Binary subtraction
            "plus": "systemo_add",
            "star": "systemo_multiply",
            "slash": "systemo_divide",
            "eqeq": "systemo_equals",
            "bangeq": "systemo_not_equals",
            "lt": "systemo_less_than",
            "gt": "systemo_greater_than",
            "lteq": "systemo_less_equal",
            "gteq": "systemo_greater_equal",
        }
        return binary_name_map.get(safe_op_name, f"systemo_{safe_op_name}")

    def _get_unary_function_name(self, safe_op_name: str) -> str:
        # Map specific unary operations to distinct names
        unary_name_map = {
            "minus": "systemo_negate",  # Unary negation
            "bang": "systemo_not",
        }
        if safe_op_name in unary_name_map:
            return unary_name_map[safe_op_name]

        # For user-defined functions (alphanumeric names), use simple systemo_ prefix
        if safe_op_name.isalnum():
            return f"systemo_{safe_op_name}"
        else:
            # For symbolic operators, use unary prefix to distinguish from binary
            return f"systemo_unary_{safe_op_name}"

    def _infer_type_from_operand(self, operand: "Expression") -> Optional[str]:
        # Check if the operand has a type annotation
        if hasattr(operand, "ty") and operand.ty:
            return self._type_expression_to_string(operand.ty)

        if isinstance(operand, (IntLiteral, NegativeInt)):
            return "Int"
        elif isinstance(operand, (FloatLiteral, NegativeFloat)):
            return "Float"
        elif isinstance(operand, BoolLiteral):
            return "Bool"
        elif isinstance(operand, StringLiteral):
            return "String"
        elif isinstance(operand, CharLiteral):
            return "Char"

        # For other cases, return None (cannot infer)
        return None

    def _sanitize_operator_name(self, operator: str) -> str:
        replacements = {
            "!": "bang",
            "@": "at",
            "#": "hash",
            "$": "dollar",
            "%": "percent",
            "^": "caret",
            "&": "amp",
            "*": "star",
            "+": "plus",
            "-": "minus",
            "=": "eq",
            "|": "pipe",
            "\\": "backslash",
            "/": "slash",
            "?": "question",
            "<": "lt",
            ">": "gt",
            "~": "tilde",
            "`": "backtick",
            ":": "colon",
            ";": "semicolon",
            "'": "quote",
            '"': "doublequote",
            ",": "comma",
            ".": "dot",
            "(": "lparen",
            ")": "rparen",
            "[": "lbracket",
            "]": "rbracket",
            "{": "lbrace",
            "}": "rbrace",
        }

        result = operator
        for char, replacement in replacements.items():
            result = result.replace(char, replacement)

        return result

    def _compile_literal_expression(self, expr: Expression) -> Optional[str]:
        match expr:
            case IntLiteral(value=value) | NegativeInt(value=value):
                return str(value)
            case FloatLiteral(value=value) | NegativeFloat(value=value):
                return str(value)
            case StringLiteral(value=value):
                return f'"{value}"'
            case CharLiteral(value=value):
                return f"'{value}'"  # Char is just a string in Python
            case BoolLiteral(value=value):
                return str(value)
            case ListLiteral(elements=elements):
                compiled_elements = [
                    self._compile_expression(elem) for elem in elements
                ]
                return f"[{','.join(compiled_elements)}]"
            case TupleLiteral(elements=elements):
                compiled_elements = [
                    self._compile_expression(elem) for elem in elements
                ]
                # Ensure we have proper tuple syntax - add comma for single element
                if len(compiled_elements) == 1:
                    return f"({compiled_elements[0]},)"
                return f"({', '.join(compiled_elements)})"
            case _:
                return None  # Not a literal

    def _compile_variable_expression(self, name: str) -> str:
        # Primitive functions should not be prefixed
        if name.startswith("prim"):
            if (
                name in self.nullary_functions
                and name not in self.lambda_lifted_functions
            ):
                return f"{name}()"
            else:
                return name
        else:
            prefixed_name = self._prefix_name(name)

            # Pattern variables (local variables) should never be called as functions
            if name in self.local_variables:
                return prefixed_name
            elif (
                name in self.nullary_functions
                and name not in self.lambda_lifted_functions
            ):
                return f"{prefixed_name}()"
            else:
                return prefixed_name

    def _compile_constructor_expression(self, name: str) -> str:
        # Nullary constructors should be instantiated
        constructor_def = self._find_constructor_def(name)
        if (
            constructor_def
            and (
                constructor_def.type_atoms is None
                or len(constructor_def.type_atoms) == 0
            )
            and constructor_def.record_constructor is None
        ):
            return f"{name}()"
        else:
            return name

    def _compile_expression(self, expr: Expression) -> str:
        # Try literal expressions first
        literal_result = self._compile_literal_expression(expr)
        if literal_result is not None:
            return literal_result

        match expr:
            # Literals
            case IntLiteral(value=value):
                return str(value)
            case FloatLiteral(value=value):
                return str(value)
            case NegativeInt(value=value):
                return str(value)
            case NegativeFloat(value=value):
                return str(value)
            case StringLiteral(value=value):
                return f'"{value}"'
            case CharLiteral(value=value):
                return f"'{value}'"  # Char is just a string in Python
            case BoolLiteral(value=value):
                return str(value)
            case ListLiteral(elements=elements):
                compiled_elements = [
                    self._compile_expression(elem) for elem in elements
                ]
                return f"[{','.join(compiled_elements)}]"

            case TupleLiteral(elements=elements):
                compiled_elements = [
                    self._compile_expression(elem) for elem in elements
                ]
                # Ensure we have proper tuple syntax - add comma for single element
                if len(compiled_elements) == 1:
                    return f"({compiled_elements[0]},)"
                return f"({', '.join(compiled_elements)})"
            case Variable(name=name):
                # TODO: just create a call to the exact name with type suffix
                return self._compile_variable_expression(name)
            case Constructor(name=name):
                return self._compile_constructor_expression(name)

            case IfElse(condition=condition, then_expr=then_expr, else_expr=else_expr):
                return f"({self._compile_expression(then_expr)} if {self._compile_expression(condition)} else {self._compile_expression(else_expr)})"

            # Function application
            case FunctionApplication(function, ty=ty):
                match function:
                    case Variable(name=name):
                        pass
                    case FunctionApplication(function=fInner, ty=ty):
                        match fInner:
                            case Variable(name=innerName):
                                pass
                            case _:
                                pass
                    case _:
                        pass
                # Collect all arguments for function calls
                args: List[Expression] = []
                current: Expression = expr

                # Collect all arguments from nested function applications
                while True:
                    match current:
                        case FunctionApplication(argument=argument, function=function):
                            args.insert(0, argument)
                            current = function
                        case _:
                            break

                # If the base function is a constructor, generate single call with all args
                match current:
                    case Constructor(name=name):
                        arg_exprs = [self._compile_expression(arg) for arg in args]
                        return f"{name}({', '.join(arg_exprs)})"
                    case Variable(name=name):
                        # Handle primitive functions specially - they are not curried
                        if name.startswith("prim"):
                            arg_exprs = [self._compile_expression(arg) for arg in args]
                            return f"{name}({', '.join(arg_exprs)})"

                        # For polymorphic functions, use pure dictionary-passing
                        if self._is_polymorphic_function(name):
                            arg_exprs = [self._compile_expression(arg) for arg in args]
                            clean_name = self._sanitize_operator_name(name)
                            if not clean_name or not clean_name[0].isalpha():
                                clean_name = f"op_{clean_name}"

                            # Check if this is partial application
                            expected_arity = self._get_function_arity(name)
                            provided_args = len(args)
                            dispatcher_name = f"systemo_{clean_name}"

                            if provided_args == expected_arity:
                                # Exact match - call function directly
                                if expected_arity == 1:
                                    # Unary function - determine type dictionary from argument
                                    arg_type = getattr(args[0], "ty", None)
                                    if arg_type:
                                        arg_type_str = self._type_expression_to_string(
                                            arg_type,
                                        )
                                        dict_name = self._get_type_dict_name(
                                            arg_type_str,
                                        )
                                        # If type inference fails or is uncertain, use runtime dispatch
                                        if (
                                            dict_name == "unknown_dict"
                                            or arg_type_str in ["unknown", "u"]
                                        ):
                                            return f"{dispatcher_name}_runtime_dispatch({arg_exprs[0]})"
                                        return f"{dispatcher_name}({dict_name}, {arg_exprs[0]})"
                                    else:
                                        # Add runtime type dispatch for better type inference
                                        return f"{dispatcher_name}_runtime_dispatch({arg_exprs[0]})"
                                elif expected_arity == 2:
                                    # Binary function - determine type dictionary from first argument
                                    arg_type = getattr(args[0], "ty", None)
                                    if arg_type:
                                        arg_type_str = self._type_expression_to_string(
                                            arg_type,
                                        )
                                        dict_name = self._get_type_dict_name(
                                            arg_type_str,
                                        )
                                        # If type inference fails or is uncertain, use runtime dispatch
                                        if (
                                            dict_name == "unknown_dict"
                                            or arg_type_str in ["unknown", "u"]
                                        ):
                                            return f"{dispatcher_name}_runtime_dispatch({arg_exprs[0]}, {arg_exprs[1]})"
                                        return f"{dispatcher_name}({dict_name}, {arg_exprs[0]}, {arg_exprs[1]})"
                                    else:
                                        # Add runtime type dispatch for better type inference
                                        return f"{dispatcher_name}_runtime_dispatch({arg_exprs[0]}, {arg_exprs[1]})"
                                else:
                                    # For higher arity functions
                                    return f"{dispatcher_name}({', '.join(arg_exprs)})"
                            elif provided_args < expected_arity:
                                # Partial application for polymorphic functions - generate lambda
                                remaining_args = expected_arity - provided_args
                                lambda_params = [
                                    f"_arg{i}" for i in range(remaining_args)
                                ]

                                # For polymorphic functions, we need to determine the type dictionary
                                # from the first argument and pass it along with all arguments
                                if arg_exprs:
                                    arg_type = getattr(args[0], "ty", None)
                                    if arg_type:
                                        arg_type_str = self._type_expression_to_string(
                                            arg_type,
                                        )
                                        dict_name = self._get_type_dict_name(
                                            arg_type_str,
                                        )
                                        if (
                                            dict_name == "unknown_dict"
                                            or arg_type_str in ["unknown", "u"]
                                        ):
                                            # Use runtime dispatch for partial application
                                            all_args = arg_exprs + lambda_params
                                            return f"lambda {', '.join(lambda_params)}: {dispatcher_name}_runtime_dispatch({', '.join(all_args)})"
                                        else:
                                            # Use dictionary-passing for partial application
                                            all_args = (
                                                [dict_name] + arg_exprs + lambda_params
                                            )
                                            return f"lambda {', '.join(lambda_params)}: {dispatcher_name}({', '.join(all_args)})"
                                    else:
                                        # Fallback to runtime dispatch
                                        all_args = arg_exprs + lambda_params
                                        return f"lambda {', '.join(lambda_params)}: {dispatcher_name}_runtime_dispatch({', '.join(all_args)})"
                                else:
                                    # No provided args, just create lambda for all parameters
                                    lambda_params = [
                                        f"_arg{i}" for i in range(expected_arity)
                                    ]
                                    return f"lambda {', '.join(lambda_params)}: {dispatcher_name}_runtime_dispatch({', '.join(lambda_params)})"
                            else:
                                # More arguments than expected - shouldn't happen in well-typed code
                                return f"{dispatcher_name}({', '.join(arg_exprs)})"

                        # Try to find a monomorphic function for this call using actual type information
                        return_type = getattr(expr, "ty", None)
                        monomorphic_name = self._find_monomorphic_function_by_types(
                            name,
                            args,
                            return_type,
                        )
                        if monomorphic_name:
                            # Use the monomorphic function directly
                            arg_exprs = [self._compile_expression(arg) for arg in args]
                            return f"{monomorphic_name}({', '.join(arg_exprs)})"

                        # Try alternative monomorphic function resolution
                        monomorphic_call = self._compile_monomorphic_function_call(
                            name,
                            args,
                            expr,
                        )
                        if monomorphic_call:
                            return monomorphic_call

                        # For non-polymorphic functions, handle as regular function
                        # Check the arity of the function to handle partial application
                        expected_arity = self._get_function_arity(name)
                        provided_args = len(args)
                        prefixed_name = self._prefix_name(name)

                        if provided_args == expected_arity:
                            # Exact match - call function directly
                            arg_exprs = [self._compile_expression(arg) for arg in args]
                            return f"{prefixed_name}({', '.join(arg_exprs)})"
                        elif provided_args < expected_arity:
                            # Partial application - generate lambda for remaining arguments
                            arg_exprs = [self._compile_expression(arg) for arg in args]
                            remaining_args = expected_arity - provided_args

                            # Generate lambda parameters for remaining arguments
                            lambda_params = [f"_arg{i}" for i in range(remaining_args)]
                            all_args = arg_exprs + lambda_params

                            return f"lambda {', '.join(lambda_params)}: {prefixed_name}({', '.join(all_args)})"
                        else:
                            # More arguments than expected - this shouldn't happen in well-typed code
                            # Fall back to the current behavior
                            arg_exprs = [self._compile_expression(arg) for arg in args]
                            return f"{prefixed_name}({', '.join(arg_exprs)})"
                    case _:
                        # Regular curried function application - fall back to nested calls
                        func_expr = self._compile_expression(expr.function)
                        arg_expr = self._compile_expression(expr.argument)
                        return f"{func_expr}({arg_expr})"

            # Constructor expressions
            case ConstructorExpression(
                constructor_name=constructor_name,
                fields=fields,
            ):
                if fields:
                    constructor_def = self._find_constructor_def(constructor_name)

                    if constructor_def and constructor_def.record_constructor:
                        # Reorder fields according to declaration order
                        declared_fields = {
                            field.name: field
                            for field in constructor_def.record_constructor.fields
                        }
                        provided_fields = {field.field_name: field for field in fields}

                        # Create ordered field arguments
                        field_args = []
                        for declared_field in constructor_def.record_constructor.fields:
                            if declared_field.name in provided_fields:
                                field_args.append(
                                    self._compile_expression(
                                        provided_fields[declared_field.name].value,
                                    ),
                                )
                            else:
                                # Field not provided - this should be an error, but for now use None
                                field_args.append("None")

                        return f"{constructor_name}({', '.join(field_args)})"
                    else:
                        # Fallback: use fields in provided order (for non-record constructors)
                        field_args = [
                            self._compile_expression(field.value) for field in fields
                        ]
                        return f"{constructor_name}({', '.join(field_args)})"
                else:
                    return f"{constructor_name}()"

            # Other expressions
            case DoBlock():
                return self._compile_do_block(expr)
            case GroupedExpression(expression=expression):
                return f"({self._compile_expression(expression)})"

            # Symbolic operations (operators)
            case SymbolicOperation(operator=operator, operands=operands, ty=ty):
                return self._compile_symbolic_operation(operator, operands, ty)

            case _:
                return f"None  # Unsupported: {type(expr)}"

    def _compile_do_block(self, do_block: DoBlock) -> str:
        # For single statements, we can still inline them
        if len(do_block.statements) == 1:
            stmt = do_block.statements[0]
            match stmt:
                case LetStatement(value=value):
                    return f"(lambda: {self._compile_expression(value)})()"
                case _ if self._is_expression(stmt):
                    return self._compile_expression_safe(stmt)
                case _:
                    return "None"

        # For multiple statements in expression context, fall back to sequential execution
        # This is a bit hacky but works for simple cases
        parts = []
        for stmt in do_block.statements[:-1]:
            match stmt:
                case LetStatement(variable=variable, value=value):
                    prefixed_var = self._prefix_name(variable)
                    parts.append(
                        f"globals().update({{'{prefixed_var}': {self._compile_expression(value)}}})",
                    )
                case _ if self._is_expression(stmt):
                    parts.append(self._compile_expression_safe(stmt))
                case _:
                    pass

        # Handle the last statement (which becomes the return value)
        last_stmt = do_block.statements[-1]
        match last_stmt:
            case LetStatement(variable=variable, value=value):
                prefixed_var = self._prefix_name(variable)
                final_expr = f"globals().update({{'{prefixed_var}': {self._compile_expression(value)}}})"
            case _ if self._is_expression(last_stmt):
                final_expr = self._compile_expression_safe(last_stmt)
            case _:
                final_expr = "None"

        if parts:
            return f"({' or '.join(parts)} or {final_expr})"
        else:
            return final_expr

    def _is_expression(self, stmt: Any) -> bool:
        return is_systemo_expression(stmt)

    def _compile_expression_safe(self, stmt: Any) -> str:
        return self._compile_expression(stmt)

    def _type_expression_to_string(self, type_expr: Any) -> str:
        if type_expr is None:
            return "unknown"

        # Use match pattern for proper type handling
        match type_expr:
            case TypeApp(constructor=TypeCon(name="List"), argument=arg_type):
                return "List"
            case TypeCon(name=name):
                if name == "String":
                    return "str"
                return name
            case TupleType():
                return "tuple"
            case _:
                pass  # Fall through to legacy handling

        if hasattr(type_expr, "__class__"):
            class_name = type_expr.__class__.__name__

            # Handle tuple types specifically
            if class_name == "TupleType":
                return "tuple"

            if hasattr(type_expr, "name"):
                # Basic types like Int, Float, Bool
                type_name = str(type_expr.name)
                # Map to match monomorphic function naming conventions
                if type_name == "List":
                    return "listtype"
                elif type_name == "String":
                    return "str"  # Match the registration
                return type_name
            elif hasattr(type_expr, "from_type") and hasattr(type_expr, "to_type"):
                # Arrow types (function types)
                from_str = self._type_expression_to_string(type_expr.from_type)
                to_str = self._type_expression_to_string(type_expr.to_type)
                return f"{from_str} -> {to_str}"
            elif hasattr(type_expr, "param") and hasattr(type_expr, "result"):
                # Function types
                param_str = self._type_expression_to_string(type_expr.param)
                result_str = self._type_expression_to_string(type_expr.result)
                return f"{param_str} -> {result_str}"
            elif hasattr(type_expr, "constructor"):
                # Constructor types
                constructor_name = str(type_expr.constructor)
                # Map constructor names to match monomorphic function naming
                if constructor_name == "List":
                    return "listtype"
                elif constructor_name == "String":
                    return "str"
                return constructor_name
            else:
                return class_name.lower()

        # Handle DataType objects (user-defined types)
        if (
            hasattr(type_expr, "name")
            and hasattr(type_expr, "type_args")
            and not hasattr(type_expr, "from_type")
        ):
            # This looks like a DataType - has name and type_args but not from_type (which would be ArrowType)
            return type_expr.name
        else:
            return str(type_expr)

    def _type_expression_to_show_type_string(self, type_expr: Any) -> str:
        if type_expr is None:
            return "unknown"

        # Use match for type objects from lango_types to extract specific list element types
        match type_expr:
            case TypeApp(constructor=TypeCon(name="List"), argument=arg_type):
                # Extract the element type for lists
                element_type_str = self._type_expression_to_show_type_string(arg_type)
                return f"list_{element_type_str}"
            case TypeApp(constructor=constructor, argument=arg_type):
                # Other type applications
                constructor_str = self._type_expression_to_show_type_string(constructor)
                arg_str = self._type_expression_to_show_type_string(arg_type)
                return f"{constructor_str}_{arg_str}"
            case TypeCon(name="Int"):
                return "int"
            case TypeCon(name="Float"):
                return "float"
            case TypeCon(name="Bool"):
                return "bool"
            case TypeCon(name="String"):
                return "str"
            case TypeCon(name="Char"):
                return "char"
            case TypeCon(name="()"):
                return "unit"
            case TypeCon(name=name):
                return name.lower()
            case FunctionType(param=param_type, result=result_type):
                # Function types - convert -> to _to_ for valid Python function names
                param_str = self._type_expression_to_show_type_string(param_type)
                result_str = self._type_expression_to_show_type_string(result_type)
                return f"{param_str}_to_{result_str}"
            case _:
                # Fallback to the original function but sanitize for function names
                type_str = self._type_expression_to_string(type_expr)
                # Replace -> with _to_ and remove special characters
                import re

                sanitized = type_str.replace(" -> ", "_to_").replace("->", "_to_")
                sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", sanitized)
                return sanitized

    def _type_signature_to_monomorphic_suffix(self, type_expr: Any) -> str:
        if type_expr is None:
            return "unknown"

        type_str = self._type_expression_to_string(type_expr)
        if not type_str:
            return "unknown"

        # Convert type names to lowercase and replace arrows with underscores
        # Remove spaces and convert -> to _
        normalized = type_str.lower().replace(" -> ", "_").replace("->", "_")

        # Remove any special characters that would be invalid in function names
        import re

        normalized = re.sub(r"[^a-zA-Z0-9_]", "", normalized)

        # Handle special cases for built-in types
        type_mappings = {
            "int": "int",
            "float": "float",
            "bool": "bool",
            "string": "str",
            "list": "list",
            "()": "unit",
            "": "unknown",
        }

        # Split by underscore and map each type
        parts = normalized.split("_")
        mapped_parts = []
        for part in parts:
            if part.strip():  # Skip empty parts
                mapped_parts.append(type_mappings.get(part.strip(), part.strip()))

        if not mapped_parts:
            return "unknown"

        return "_".join(mapped_parts)

    def _generate_monomorphic_function_name(
        self,
        base_name: str,
        type_signature: Any,
    ) -> str:
        try:
            type_suffix = self._type_signature_to_monomorphic_suffix(type_signature)
            return f"systemo_{base_name}_{type_suffix}"
        except Exception as e:
            # Fallback to index-based naming if type processing fails
            import hashlib

            type_hash = hashlib.md5(str(type_signature).encode()).hexdigest()[:8]
            return f"systemo_{base_name}_{type_hash}"

    def _compile_monomorphic_function_call(
        self,
        function_name: str,
        args: List[Expression],
        type_info: Any = None,
    ) -> Optional[str]:
        # First try using the type information from the function application itself
        if type_info and hasattr(type_info, "ty") and type_info.ty:
            return_type = type_info.ty
            monomorphic_name = self._find_monomorphic_function_by_types(
                function_name,
                args,
                return_type,
            )
            if monomorphic_name:
                arg_exprs = [self._compile_expression(arg) for arg in args]
                return f"{monomorphic_name}({', '.join(arg_exprs)})"

        # Try without return type information
        monomorphic_name = self._find_monomorphic_function_by_types(function_name, args)
        if monomorphic_name:
            arg_exprs = [self._compile_expression(arg) for arg in args]
            return f"{monomorphic_name}({', '.join(arg_exprs)})"

        # Legacy specific handling for show function on tuple types (can be removed when type info is complete)
        if function_name == "show" and len(args) == 1:
            arg = args[0]
            if hasattr(arg, "ty") and hasattr(arg.ty, "__class__"):

                if isinstance(arg.ty, TupleType):
                    # Generate specialized tuple show function name
                    tuple_type_str = self._tuple_type_to_string(arg.ty.element_types)
                    specialized_func_name = f"systemo_show_{tuple_type_str}_str"

                    # Check if we have this specialized function
                    registry_key = f"show+{tuple_type_str} -> str"
                    if registry_key in self.monomorphic_functions:
                        arg_exprs = [self._compile_expression(arg) for arg in args]
                        return f"{specialized_func_name}({', '.join(arg_exprs)})"

        return None

    def _generate_typed_dispatcher(
        self,
        instance_name: str,
        instances: List["InstanceDeclaration"],
        type_to_function: Dict[str, str],
    ) -> str:
        prefixed_name = f"systemo_{instance_name}"
        lines = [f"def {prefixed_name}(arg: Any) -> Any:"]

        self.indent_level += 1

        # Generate type-based dispatch similar to interpreter
        type_handled = False

        # Sort instances to put Bool before Int (since bool is subclass of int in Python)
        sorted_instances = []
        bool_instances = []
        other_instances = []

        for instance in instances:
            arg_type = self._extract_first_param_type_name(instance.type_signature)
            if arg_type == "Bool":
                bool_instances.append(instance)
            else:
                other_instances.append(instance)

        # Put Bool instances first, then others
        sorted_instances = bool_instances + other_instances

        for instance in sorted_instances:
            type_sig = self._type_expression_to_string(instance.type_signature)
            monomorphized_name = type_to_function[type_sig]

            # Extract the argument type from the type signature object
            arg_type = self._extract_first_param_type_name(instance.type_signature)

            if arg_type:
                # Generate appropriate type check
                if arg_type in ["Int", "Float", "Bool", "String"]:
                    # Built-in types
                    python_type = {
                        "Int": "int",
                        "Float": "float",
                        "Bool": "bool",
                        "String": "str",
                    }[arg_type]
                    condition = f"isinstance(arg, {python_type})"
                elif arg_type == "List":
                    # List type - use isinstance for Python lists
                    condition = "isinstance(arg, list)"
                elif arg_type == "Tuple":
                    # Tuple type - check both isinstance and length
                    # Need to get tuple length from type signature
                    tuple_length = self._extract_tuple_length(instance.type_signature)
                    if tuple_length is not None:
                        condition = (
                            f"isinstance(arg, tuple) and len(arg) == {tuple_length}"
                        )
                    else:
                        condition = "isinstance(arg, tuple)"
                elif arg_type in self.data_types:
                    # Custom data types - check constructor names
                    constructor_names = [
                        ctor.name for ctor in self.data_types[arg_type].constructors
                    ]
                    type_checks = [
                        f"type(arg).__name__ == '{ctor_name}'"
                        for ctor_name in constructor_names
                    ]
                    condition = " or ".join(type_checks)
                elif arg_type == "Char":
                    # Special handling for Char - check for char tuples
                    condition = (
                        "isinstance(arg, tuple) and len(arg) == 2 and arg[0] == 'char'"
                    )
                else:
                    # Unknown type, try by name (but avoid complex expressions)
                    if arg_type in ["Unknown"]:
                        condition = "True"  # Fallback that always matches
                    else:
                        condition = f"type(arg).__name__ == '{arg_type}'"

                lines.append(self._indent() + f"if {condition}:")
                self.indent_level += 1

                # Check if this is a binary function (curried)
                func_def = instance.function_definition
                if len(func_def.patterns) == 2:
                    lines.append(self._indent() + f"return {monomorphized_name}(arg)")
                else:
                    lines.append(self._indent() + f"return {monomorphized_name}(arg)")

                self.indent_level -= 1
                type_handled = True

        # Add fallback error
        if type_handled:
            lines.append(
                self._indent()
                + f"raise ValueError(f'No instance of {instance_name} for type {{type(arg).__name__}}')",
            )
        else:
            lines.append(
                self._indent()
                + f"raise ValueError(f'No instances of {instance_name}')",
            )

        self.indent_level -= 1
        lines.append("")
        return "\n".join(lines)

    def _generate_typed_binary_dispatcher(
        self,
        instance_name: str,
        instances: List["InstanceDeclaration"],
        type_to_function: Dict[str, str],
    ) -> str:
        prefixed_name = f"systemo_{instance_name}_binary"
        lines = [f"def {prefixed_name}(arg_0: Any, arg_1: Any) -> Any:"]

        self.indent_level += 1

        # Filter to only binary instances (those with 2 patterns)
        binary_instances = [
            instance
            for instance in instances
            if len(instance.function_definition.patterns) == 2
        ]

        if not binary_instances:
            lines.append(
                self._indent()
                + f"raise ValueError(f'No binary instances of {instance_name}')",
            )
            self.indent_level -= 1
            lines.append("")
            return "\n".join(lines)

        # Generate type-based dispatch for binary operations
        for instance in binary_instances:
            type_sig = self._type_expression_to_string(instance.type_signature)
            monomorphized_name = type_to_function[type_sig]

            # For binary operations, extract both argument types from the type expression object
            arg_types = self._extract_binary_param_types_from_type_expression(
                instance.type_signature,
            )

            if len(arg_types) >= 2:
                conditions = []
                for i, arg_type in enumerate(arg_types[:2]):
                    if arg_type in ["Int", "Float", "Bool", "String"]:
                        python_type = {
                            "Int": "int",
                            "Float": "float",
                            "Bool": "bool",
                            "String": "str",
                        }[arg_type]
                        conditions.append(f"isinstance(arg_{i}, {python_type})")
                    elif arg_type == "List":
                        # List type - use isinstance for Python lists
                        conditions.append(f"isinstance(arg_{i}, list)")
                    elif arg_type == "Tuple":
                        # Tuple type - use isinstance for Python tuples
                        conditions.append(f"isinstance(arg_{i}, tuple)")
                    elif arg_type in self.data_types:
                        constructor_names = [
                            ctor.name for ctor in self.data_types[arg_type].constructors
                        ]
                        type_checks = [
                            f"type(arg_{i}).__name__ == '{ctor_name}'"
                            for ctor_name in constructor_names
                        ]
                        conditions.append("(" + " or ".join(type_checks) + ")")
                    else:
                        # Avoid complex type expressions that can't be valid Python
                        if arg_type in ["Unknown"]:
                            conditions.append("True")  # Fallback that always matches
                        else:
                            conditions.append(f"type(arg_{i}).__name__ == '{arg_type}'")

                condition = " and ".join(conditions)
                lines.append(self._indent() + f"if {condition}:")
                self.indent_level += 1
                lines.append(
                    self._indent() + f"return {monomorphized_name}(arg_0, arg_1)",
                )
                self.indent_level -= 1

        # Add fallback
        lines.append(
            self._indent()
            + f"raise ValueError(f'No binary instance of {instance_name} for types {{type(arg_0).__name__}}, {{type(arg_1).__name__}}')",
        )

        self.indent_level -= 1
        lines.append("")
        return "\n".join(lines)

    def _generate_type_check_condition(self, arg_name: str, arg_type: str) -> str:
        if arg_type == "Point":
            return f"hasattr({arg_name}, '__class__') and {arg_name}.__class__.__name__ == 'MkPoint'"
        elif arg_type in ["Int", "int"]:
            return f"isinstance({arg_name}, int)"
        elif arg_type in ["Float", "float"]:
            return f"isinstance({arg_name}, float)"
        elif arg_type in ["Bool", "bool"]:
            return f"isinstance({arg_name}, bool)"
        elif arg_type in ["Char"]:
            # Char should match str that are single characters
            return f"isinstance({arg_name}, str) and len({arg_name}) == 1"
        elif arg_type in ["String", "str"]:
            return f"isinstance({arg_name}, str)"
        elif arg_type.startswith("list_"):
            # Handle specific list types like list_int, list_str
            element_type = arg_type[5:]  # Remove "list_" prefix
            if element_type == "int":
                return f"isinstance({arg_name}, list) and all(isinstance(x, int) for x in {arg_name})"
            elif element_type == "str":
                return f"isinstance({arg_name}, list) and all(isinstance(x, str) for x in {arg_name})"
            elif element_type == "bool":
                return f"isinstance({arg_name}, list) and all(isinstance(x, bool) for x in {arg_name})"
            elif element_type == "float":
                return f"isinstance({arg_name}, list) and all(isinstance(x, float) for x in {arg_name})"
            else:
                # For other element types, just check if it's a list
                return f"isinstance({arg_name}, list)"
        elif arg_type.startswith("[") or arg_type == "listtype":
            return f"isinstance({arg_name}, list)"
        elif arg_type in self.data_types:
            # General data type check
            constructor_names = [
                ctor.name for ctor in self.data_types[arg_type].constructors
            ]
            type_checks = [
                f"type({arg_name}).__name__ == '{ctor_name}'"
                for ctor_name in constructor_names
            ]
            return "(" + " or ".join(type_checks) + ")"
        else:
            return "True"  # Fallback

    def _generate_instance_dispatcher(
        self,
        safe_instance_name: str,
        actual_function_name: str,
        type_to_instances: dict,
    ) -> str:
        prefixed_name = f"systemo_{safe_instance_name}"

        # Determine the arity based on the first instance
        first_instance = next(iter(type_to_instances.values()))[0]
        num_patterns = len(first_instance.function_definition.patterns)

        if num_patterns == 1:
            # Unary function dispatcher
            lines = [f"def {prefixed_name}(arg: Any) -> Any:"]
            self.indent_level += 1

            # Generate dispatch logic for each type signature
            for type_sig, instances in type_to_instances.items():
                registry_key = f"{safe_instance_name}+{type_sig}"
                monomorphic_name = self.monomorphic_functions.get(registry_key)

                if monomorphic_name:
                    # Extract the argument type from the type signature
                    arg_type = (
                        type_sig.split("_to_")[0] if "_to_" in type_sig else type_sig
                    )

                    # Generate type check condition
                    condition = self._generate_type_check_condition("arg", arg_type)

                    lines.append(f"    if {condition}:")
                    lines.append(f"        return {monomorphic_name}(arg)")

            # Add fallback error
            lines.append(
                f"    raise ValueError(f'No instance of {actual_function_name} for type {{type(arg).__name__}}')",
            )

            self.indent_level -= 1
            lines.append("")
            return "\n".join(lines)

        elif num_patterns == 2:
            # Binary function dispatcher - generate runtime type-based dispatch
            lines = [f"def {prefixed_name}(arg_0: Any, arg_1: Any) -> Any:"]

            # Generate type checks for each monomorphic function
            for type_sig, instances in type_to_instances.items():
                registry_key = f"{safe_instance_name}+{type_sig}"
                monomorphic_name = self.monomorphic_functions.get(registry_key)

                if monomorphic_name:
                    # Parse the type signature to get argument types
                    # Type signature format: "Type1 -> Type2 -> ReturnType"
                    parts = type_sig.split(" -> ")
                    if len(parts) >= 2:
                        arg1_type = parts[0].strip()
                        arg2_type = parts[1].strip()

                        # Generate runtime type check
                        def get_type_condition(arg_name: str, arg_type: str) -> str:
                            if arg_type == "Point":
                                return f"hasattr({arg_name}, '__class__') and {arg_name}.__class__.__name__ == 'MkPoint'"
                            elif arg_type == "Int":
                                return f"isinstance({arg_name}, int)"
                            elif arg_type == "Float":
                                return f"isinstance({arg_name}, float)"
                            elif arg_type == "Bool":
                                return f"isinstance({arg_name}, bool)"
                            elif arg_type == "String":
                                return f"isinstance({arg_name}, str)"
                            elif arg_type.startswith("[") or arg_type == "listtype":
                                return f"isinstance({arg_name}, list)"
                            elif arg_type in self.data_types:
                                # General data type check
                                constructor_names = [
                                    ctor.name
                                    for ctor in self.data_types[arg_type].constructors
                                ]
                                type_checks = [
                                    f"type({arg_name}).__name__ == '{ctor_name}'"
                                    for ctor_name in constructor_names
                                ]
                                return "(" + " or ".join(type_checks) + ")"
                            else:
                                return "True"  # Fallback

                        type1_check = get_type_condition("arg_0", arg1_type)
                        type2_check = get_type_condition("arg_1", arg2_type)

                        lines.append(f"    if {type1_check} and {type2_check}:")
                        lines.append(f"        return {monomorphic_name}(arg_0, arg_1)")

            # Add fallback error
            lines.append(
                f"    raise ValueError(f'No instance of {actual_function_name} for types {{type(arg_0).__name__}} and {{type(arg_1).__name__}}')",
            )

            return "\n".join(lines)
        else:
            # Unsupported arity
            return f"# Unsupported arity {num_patterns} for dispatcher {prefixed_name}"

    def _extract_binary_param_types_from_type_expression(
        self,
        type_expr: Any,
    ) -> List[str]:
        result: List[str] = []
        current = type_expr

        # For ArrowType like "Int -> Int -> String", extract the first two types
        while hasattr(current, "from_type") and len(result) < 2:
            result.append(self._extract_type_name_from_expression(current.from_type))
            if hasattr(current, "to_type"):
                current = current.to_type
            else:
                break

        return result

    def _compile_grouped_instance_function(
        self,
        grouped_instances: List["InstanceDeclaration"],
        function_name: str,
    ) -> str:
        if not grouped_instances:
            return ""

        # Set the current function name for recursive call handling
        old_function_name = self.current_function_name
        self.current_function_name = function_name

        try:
            if len(grouped_instances) == 1:
                # Single instance - use existing method based on number of patterns
                instance = grouped_instances[0]
                func_def = instance.function_definition
                if len(func_def.patterns) == 2:
                    return self._compile_binary_instance_function(
                        func_def,
                        function_name,
                    )
                else:
                    return self._compile_simple_function(func_def, function_name)

            # Multiple instances - determine if they are binary or unary
            first_instance = grouped_instances[0]
            is_binary = len(first_instance.function_definition.patterns) == 2

            if is_binary:
                # Binary function with multiple patterns
                lines = [
                    f"def {function_name}(arg_0: Any, arg_1: Any) -> Any:",
                ]
            else:
                # Unary function with multiple patterns
                lines = [f"def {function_name}(arg: Any) -> Any:"]

            self.indent_level += 1

            # Track pattern variables
            old_local_vars = self.local_variables.copy()

            # Generate pattern matching for each instance
            for instance in grouped_instances:
                func_def = instance.function_definition

                # Extract variables from patterns
                for pattern in func_def.patterns:
                    self.local_variables.update(
                        self._extract_pattern_variables(pattern),
                    )

                # Generate pattern matching for each function definition
                if is_binary and len(func_def.patterns) >= 2:
                    # Binary function - generate proper pattern conditions
                    pattern_0 = func_def.patterns[0]
                    pattern_1 = func_def.patterns[1]

                    condition_0 = self._compile_pattern_condition(pattern_0, "arg_0")
                    condition_1 = self._compile_pattern_condition(pattern_1, "arg_1")
                    combined_condition = f"({condition_0}) and ({condition_1})"

                    lines.append(self._indent() + f"if {combined_condition}:")
                    self.indent_level += 1

                    # Add variable bindings
                    bindings_0 = self._compile_pattern_bindings(pattern_0, "arg_0")
                    bindings_1 = self._compile_pattern_bindings(pattern_1, "arg_1")

                    for binding in bindings_0 + bindings_1:
                        lines.append(self._indent() + binding)

                    lines.append(
                        self._indent()
                        + f"return {self._compile_expression(func_def.body)}",
                    )
                    self.indent_level -= 1

                elif not is_binary and len(func_def.patterns) >= 1:
                    # Unary function - generate proper pattern condition
                    pattern = func_def.patterns[0]
                    condition = self._compile_pattern_condition(pattern, "arg")

                    lines.append(self._indent() + f"if {condition}:")
                    self.indent_level += 1

                    # Add variable bindings
                    bindings = self._compile_pattern_bindings(pattern, "arg")
                    for binding in bindings:
                        lines.append(self._indent() + binding)

                    lines.append(
                        self._indent()
                        + f"return {self._compile_expression(func_def.body)}",
                    )
                    self.indent_level -= 1

            # Add fallback error
            lines.append(self._indent() + f"raise ValueError('Pattern match failed')")

            self.indent_level -= 1
            # Restore local variables
            self.local_variables = old_local_vars

            lines.append("")
            return "\n".join(lines)
        finally:
            # Always restore the previous function name
            self.current_function_name = old_function_name

    def _extract_first_param_type_name(self, type_expression: Any) -> str:
        # Handle ArrowType objects
        if hasattr(type_expression, "from_type"):
            return self._extract_type_name_from_expression(type_expression.from_type)
        else:
            return self._extract_type_name_from_expression(type_expression)

    def _extract_type_name_from_expression(self, type_expr: Any) -> str:
        if hasattr(type_expr, "name"):
            # TypeConstructor like Int, Float, Bool, String
            return type_expr.name
        elif hasattr(type_expr, "constructor"):
            # TypeApplication like Either Int Bool
            return self._extract_type_name_from_expression(type_expr.constructor)
        elif hasattr(type_expr, "element_type"):
            # ListType - return "List" as the type name
            return "List"
        elif hasattr(type_expr, "element_types"):
            # TupleType - return "Tuple" as the type name
            return "Tuple"
        else:
            # For any complex type, extract a safe name or fall back to a generic name
            type_str = str(type_expr)
            if "TupleType" in type_str:
                return "Tuple"
            elif "ListType" in type_str:
                return "List"
            else:
                return "Unknown"

    def _extract_tuple_length(self, type_signature: Any) -> Optional[int]:
        if isinstance(type_signature, ArrowType):
            # For ArrowType, check the from_type
            from_type = type_signature.from_type
            if isinstance(from_type, TupleType):
                return len(from_type.element_types)
        elif isinstance(type_signature, TupleType):
            # Direct tuple type
            return len(type_signature.element_types)

        return None

    def _compile_pattern_condition(self, pattern: Any, arg_name: str) -> str:
        if isinstance(pattern, LiteralPattern):
            # Check if argument equals the literal value
            return f"{arg_name} == {self._compile_literal(pattern.value)}"
        elif isinstance(pattern, VariablePattern):
            # Variable patterns always match
            return "True"
        elif isinstance(pattern, ListPattern):
            if not pattern.patterns:
                # Empty list pattern []
                return f"isinstance({arg_name}, list) and len({arg_name}) == 0"
            else:
                # Non-empty list pattern [a, b, c] (rare)
                conditions = [
                    f"isinstance({arg_name}, list)",
                    f"len({arg_name}) == {len(pattern.patterns)}",
                ]
                return " and ".join(conditions)
        elif isinstance(pattern, ConsPattern):
            # List cons pattern x:xs - check if it's a non-empty list
            return f"isinstance({arg_name}, list) and len({arg_name}) > 0"
        elif isinstance(pattern, TuplePattern):
            # Tuple pattern (a, b, c)
            conditions = [
                f"isinstance({arg_name}, tuple)",
                f"len({arg_name}) == {len(pattern.patterns)}",
            ]
            return " and ".join(conditions)
        elif isinstance(pattern, ConstructorPattern):
            # Constructor pattern like Just x, Left y
            return f"type({arg_name}).__name__ == '{getattr(pattern, 'constructor', 'Unknown')}'"
        else:
            # Fallback for unknown patterns
            return "True"

    def _compile_pattern_bindings(self, pattern: Any, arg_name: str) -> List[str]:
        bindings = []

        if isinstance(pattern, VariablePattern):
            bindings.append(f"{pattern.name} = {arg_name}")
        elif isinstance(pattern, ConsPattern):
            # x:xs pattern - bind head and tail
            if isinstance(pattern.head, VariablePattern):
                bindings.append(f"{pattern.head.name} = {arg_name}[0]")
            if isinstance(pattern.tail, VariablePattern):
                bindings.append(f"{pattern.tail.name} = {arg_name}[1:]")
        elif isinstance(pattern, TuplePattern):
            # (a, b, c) pattern - bind each element
            for i, subpattern in enumerate(pattern.patterns):
                if isinstance(subpattern, VariablePattern):
                    bindings.append(f"{subpattern.name} = {arg_name}[{i}]")
        elif isinstance(pattern, ListPattern):
            # [a, b, c] pattern - bind each element (rare)
            for i, subpattern in enumerate(pattern.patterns):
                if isinstance(subpattern, VariablePattern):
                    bindings.append(f"{subpattern.name} = {arg_name}[{i}]")
        # LiteralPattern and ConstructorPattern don't bind variables

        return bindings

    def _compile_literal(self, value: Any) -> str:
        if isinstance(value, bool):
            return str(value)
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            return str(value)
        elif isinstance(value, str):
            return repr(value)
        else:
            return str(value)

    def _collect_concrete_polymorphic_instantiations(self, expr: Expression) -> None:
        match expr:
            case FunctionApplication(function=func, argument=arg):
                # Check if this is a call to a polymorphic function with concrete types
                if isinstance(func, Variable) and hasattr(arg, "ty") and arg.ty:
                    func_name = func.name

                    # Check if this function has overloaded instances (is polymorphic)
                    # We'll check this by looking at our instance declarations
                    if self._is_polymorphic_function(func_name):
                        arg_type_str = self._type_expression_to_string(arg.ty)

                        # Get the return type from the function application
                        if hasattr(expr, "ty") and expr.ty:
                            return_type_str = self._type_expression_to_string(expr.ty)
                            full_type_sig = f"{arg_type_str} -> {return_type_str}"

                            # Track this concrete instantiation
                            if func_name not in self.concrete_instantiations:
                                self.concrete_instantiations[func_name] = set()
                            self.concrete_instantiations[func_name].add(full_type_sig)

                # Recursively process function and argument
                self._collect_concrete_polymorphic_instantiations(func)
                self._collect_concrete_polymorphic_instantiations(arg)

            case TupleLiteral(elements=elements):
                # Recursively collect from tuple elements
                for elem in elements:
                    self._collect_concrete_polymorphic_instantiations(elem)

            case Variable():
                # Variables don't need recursion
                pass

            case IfElse(condition=cond, then_expr=then_expr, else_expr=else_expr):
                self._collect_concrete_polymorphic_instantiations(cond)
                self._collect_concrete_polymorphic_instantiations(then_expr)
                self._collect_concrete_polymorphic_instantiations(else_expr)

            case DoBlock(statements=stmts):
                for stmt in stmts:
                    if isinstance(stmt, LetStatement):
                        self._collect_concrete_polymorphic_instantiations(stmt.value)
                    # Skip non-expression statements
                    elif isinstance(  # TODO: replace with is_expression
                        stmt,
                        (
                            IntLiteral,
                            FloatLiteral,
                            StringLiteral,
                            CharLiteral,
                            BoolLiteral,
                            ListLiteral,
                            TupleLiteral,
                            Variable,
                            Constructor,
                            SymbolicOperation,
                            IfElse,
                            DoBlock,
                            FunctionApplication,
                            ConstructorExpression,
                            GroupedExpression,
                            NegativeInt,
                            NegativeFloat,
                        ),
                    ):
                        self._collect_concrete_polymorphic_instantiations(stmt)

            case GroupedExpression(expression=inner_expr):
                self._collect_concrete_polymorphic_instantiations(inner_expr)

            case ConstructorExpression(fields=fields):
                for field in fields:
                    if hasattr(field, "value"):  # FieldAssignment has a value
                        self._collect_concrete_polymorphic_instantiations(field.value)

            case ListLiteral(elements=elements):
                for elem in elements:
                    self._collect_concrete_polymorphic_instantiations(elem)

            case _:
                # Other expression types
                pass

    def _is_polymorphic_function(self, func_name: str) -> bool:
        # In pure dictionary-passing, we identify polymorphic functions based on:
        # 1. Known built-in polymorphic functions
        # 2. Functions with instance declarations
        # 3. Functions that appear in type dictionaries

        # IMPORTANT: Check if this is a user-defined function first
        # User-defined functions take precedence over built-in polymorphic functions
        if (
            hasattr(self, "function_definitions")
            and func_name in self.function_definitions
        ):
            return False  # User-defined functions are not polymorphic

        # Clean the function name first
        clean_name = self._sanitize_operator_name(func_name)
        if not clean_name or not clean_name[0].isalpha():
            clean_name = f"op_{clean_name}"

        # Built-in polymorphic functions
        built_in_polymorphic = {
            "show",
            "eq",
            "add",
            "sub",
            "mul",
            "div",
            "pow",
            "neg",
            "lt",
            "le",
            "gt",
            "ge",
            "slasheq",
            "ampamp",
            "pipepipe",
            "plusplus",
            "question",
            "abs",
            "elem",
            "listEq",
            "map",
            "length",
            "bangbang",
            "equals",
        }

        if clean_name in built_in_polymorphic or func_name in built_in_polymorphic:
            return True

        # Check if this function appears in any type dictionary
        for type_dict in self.type_dictionaries.values():
            if func_name in type_dict or clean_name in type_dict:
                return True

        # Check if there are instance declarations for this function
        if hasattr(self, "instance_declarations_registry"):
            for instance_name in self.instance_declarations_registry.keys():
                if instance_name == func_name or instance_name == clean_name:
                    return True

        # Check if there are multiple monomorphic functions for this name (fallback)
        matching_keys = [
            key
            for key in self.monomorphic_functions.keys()
            if key.startswith(f"{clean_name}+")
        ]

        # If there are multiple instances, it's polymorphic
        return len(matching_keys) > 1

    def _tuple_type_to_string(self, element_types: Any) -> str:
        type_strings = [self._type_expression_to_string(t) for t in element_types]
        return "_".join(type_strings)

    def _generate_concrete_function_instances(self) -> List[str]:
        functions = []

        for func_name, type_sigs in self.concrete_instantiations.items():
            for type_sig in type_sigs:
                # Generate a concrete instance for this function and type signature
                concrete_func = self._generate_concrete_instance(func_name, type_sig)
                if concrete_func:
                    functions.append(concrete_func)

        return functions

    def _generate_concrete_instance(
        self,
        func_name: str,
        type_sig: str,
    ) -> Optional[str]:

        # Parse the type signature to get argument and return types
        parts = type_sig.split(" -> ")
        if len(parts) != 2:
            return None

        arg_type_str, return_type_str = parts

        # Clean the function name
        clean_name = self._sanitize_operator_name(func_name)
        if not clean_name or not clean_name[0].isalpha():
            clean_name = f"op_{clean_name}"

        # Check if we already have a monomorphic function for this exact type signature
        registry_key = f"{clean_name}+{type_sig}"
        if registry_key in self.monomorphic_functions:
            # Already exists, no need to generate
            return None

        # Find the generic instance definition that matches this function
        generic_instance = self._find_generic_instance(func_name, arg_type_str)
        if not generic_instance:
            return None

        # Generate the concrete function name
        concrete_func_name = self._generate_monomorphic_function_name(
            clean_name,
            type_sig,
        )

        # Register this new concrete function
        self.monomorphic_functions[registry_key] = concrete_func_name

        # For now, we don't generate concrete function implementations
        # The monomorphic functions are already registered from the systemo code
        return None

    def _find_generic_instance(
        self,
        func_name: str,
        arg_type_str: str,
    ) -> Optional["InstanceDeclaration"]:
        # This is a simplified approach - in practice, we'd need to do proper type matching
        # For now, we'll look for instances that have compatible type patterns

        # Look through all registered monomorphic functions to find one that matches
        clean_name = self._sanitize_operator_name(func_name)
        if not clean_name or not clean_name[0].isalpha():
            clean_name = f"op_{clean_name}"

        # Find any existing instance for this function (we'll use it as a template)
        for key, _ in self.monomorphic_functions.items():
            if key.startswith(f"{clean_name}+"):
                # This gives us the pattern that we can use
                # For now, return None to indicate we need the generic tuple handling
                return None

        return None

    def _should_generate_specific_instance(
        self,
        function_name: str,
        arg_type: Type,
    ) -> bool:
        # For show functions, we should generate specific instances for all types
        if function_name == "show":
            match arg_type:
                case TypeApp(constructor=TypeCon(name="List"), argument=_):
                    return True
                case TypeCon(name="Char"):
                    return True
                case TypeCon(name="Int"):
                    return True
                case TypeCon(name="Float"):
                    return True
                case TypeCon(name="Bool"):
                    return True
                case TypeCon(name="String"):
                    return True
                case _:
                    return False
        return False

    def _generate_specific_show_instance(
        self,
        arg_type: Type,
        function_name: str,
    ) -> None:
        match arg_type:
            case TypeApp(constructor=TypeCon(name="List"), argument=element_type):
                # Generate a specific list show function that calls the generic one
                # This eliminates runtime type checking by using compile-time determined functions
                function_code = f"""
def {function_name}(arg: Any) -> Any:
    return systemo_show_listtype_to_str_str(arg)
"""

                # Add it to the generated functions
                self.generated_functions.append(function_code)

                # Register it in the monomorphic functions registry
                type_sig = self._type_expression_to_show_type_string(arg_type)
                registry_key = f"show+{type_sig} -> String"
                self.monomorphic_functions[registry_key] = function_name

            case TypeCon(name="Char"):
                # Generate specific char show function
                function_code = f"""
def {function_name}(arg: Any) -> Any:
    return primCharShow(arg)
"""
                self.generated_functions.append(function_code)
                type_sig = self._type_expression_to_show_type_string(arg_type)
                registry_key = f"show+{type_sig} -> String"
                self.monomorphic_functions[registry_key] = function_name

            case TypeCon(name="Int"):
                # Generate specific int show function
                function_code = f"""
def {function_name}(arg: Any) -> Any:
    return primIntShow(arg)
"""
                self.generated_functions.append(function_code)
                type_sig = self._type_expression_to_show_type_string(arg_type)
                registry_key = f"show+{type_sig} -> String"
                self.monomorphic_functions[registry_key] = function_name

            case TypeCon(name="Float"):
                # Generate specific float show function
                function_code = f"""
def {function_name}(arg: Any) -> Any:
    return primFloatShow(arg)
"""
                self.generated_functions.append(function_code)
                type_sig = self._type_expression_to_show_type_string(arg_type)
                registry_key = f"show+{type_sig} -> String"
                self.monomorphic_functions[registry_key] = function_name

            case TypeCon(name="Bool"):
                # Generate specific bool show function
                function_code = f"""
def {function_name}(arg: Any) -> Any:
    return primBoolShow(arg)
"""
                self.generated_functions.append(function_code)
                type_sig = self._type_expression_to_show_type_string(arg_type)
                registry_key = f"show+{type_sig} -> String"
                self.monomorphic_functions[registry_key] = function_name

            case TypeCon(name="String"):
                # Generate specific string show function
                function_code = f"""
def {function_name}(arg: Any) -> Any:
    return primStringShow(arg)
"""
                self.generated_functions.append(function_code)
                type_sig = self._type_expression_to_show_type_string(arg_type)
                registry_key = f"show+{type_sig} -> String"
                self.monomorphic_functions[registry_key] = function_name

    def _find_specific_monomorphic_function(
        self,
        function_name: str,
        args: List[Expression],
        expr: Expression,
    ) -> Optional[str]:
        if not args:
            return None

        # Get the argument type information
        arg_types = []
        for arg in args:
            if hasattr(arg, "ty") and arg.ty:
                arg_types.append(arg.ty)
            else:
                # If we don't have complete type information, return None
                return None

        # Get the return type information from the function application expression
        return_type = getattr(expr, "ty", None)
        if not return_type:
            return None

        # Build the type signature for the function
        clean_name = self._sanitize_operator_name(function_name)
        if not clean_name or not clean_name[0].isalpha():
            clean_name = f"op_{clean_name}"

        # For show functions, construct the monomorphic function name
        if clean_name == "show" and len(arg_types) == 1:
            arg_type = arg_types[0]
            # Use the show-specific type string conversion
            arg_type_str = self._type_expression_to_show_type_string(arg_type)

            # Construct the monomorphic function name matching the current pattern
            monomorphic_name = f"systemo_show_{arg_type_str}_to_str_str"

            # Check if this function exists in our registry
            existing_functions = list(self.monomorphic_functions.values())
            if monomorphic_name in existing_functions:
                return monomorphic_name

            # If the function doesn't exist but we need it, try to generate it on-demand
            if self._should_generate_specific_instance(clean_name, arg_type):
                # Generate the function immediately
                self._generate_specific_show_instance(arg_type, monomorphic_name)
                return monomorphic_name

            # Fallback to generic dispatcher if specific function couldn't be generated
            return None

        # For other functions, use the general approach
        return self._find_monomorphic_function_by_types(
            function_name,
            args,
            return_type,
        )

    def _find_monomorphic_function_by_types(
        self,
        function_name: str,
        args: List[Expression],
        return_type: Any = None,
    ) -> Optional[str]:
        if not args:
            return None

        # Build the type signature from the actual argument types
        arg_type_strs = []
        for arg in args:
            if hasattr(arg, "ty") and arg.ty:
                arg_type_str = self._type_expression_to_string(arg.ty)
                arg_type_strs.append(arg_type_str)
            else:
                # If we don't have type information, we can't do type-based selection
                return None

        # Add return type if available
        if return_type:
            return_type_str = self._type_expression_to_string(return_type)
        else:
            # For now, try without return type or use a fallback
            return_type_str = None

        # Build the complete type signature
        if len(arg_type_strs) == 1 and return_type_str:
            type_signature = f"{arg_type_strs[0]} -> {return_type_str}"
        elif len(arg_type_strs) == 2 and return_type_str:
            type_signature = (
                f"{arg_type_strs[0]} -> {arg_type_strs[1]} -> {return_type_str}"
            )
        else:
            # Fallback to just the argument types
            type_signature = " -> ".join(arg_type_strs)

        # Look up the monomorphic function using the actual type signature
        clean_name = self._sanitize_operator_name(function_name)
        registry_key = f"{clean_name}+{type_signature}"
        monomorphic_name = self.monomorphic_functions.get(registry_key)

        if monomorphic_name:
            return monomorphic_name

        # If exact match fails, try to find compatible function by argument types only
        for key, monomorphic_name in self.monomorphic_functions.items():
            if "+" in key:
                func_part, registered_type = key.split("+", 1)
                if func_part == clean_name:
                    # Check if the argument types match
                    if len(arg_type_strs) == 1 and registered_type.startswith(
                        arg_type_strs[0],
                    ):
                        return monomorphic_name
                    elif len(arg_type_strs) == 2:
                        expected_pattern = f"{arg_type_strs[0]} -> {arg_type_strs[1]}"
                        if registered_type.startswith(expected_pattern):
                            return monomorphic_name

        return None


def compile_program(program: Program) -> str:
    compiler = SystemoCompiler()
    return compiler.compile(program)
