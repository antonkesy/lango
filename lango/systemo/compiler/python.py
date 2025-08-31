from typing import Any, Dict, List, Optional, Set

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
    Type,
    TypeApp,
    TypeCon,
    TypeVar,
)
from lango.systemo.ast.nodes import (
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
    Variable,
    VariablePattern,
    is_expression,
)


class systemoCompiler:
    def __init__(self) -> None:
        self.indent_level = 0
        self.defined_functions: Set[str] = set()
        self.nullary_functions: Set[str] = set()  # Functions with no parameters
        self.lambda_lifted_functions: Set[str] = (
            set()
        )  # Functions that had lambdas lifted to parameters
        self.function_types: Dict[str, Type] = {}  # Track function types
        self.data_types: Dict[str, DataDeclaration] = {}
        self.local_variables: Set[str] = set()  # Track local pattern variables
        self.do_block_counter = 0  # Counter for unique do block function names

    def _indent(self) -> str:
        return "    " * self.indent_level

    def _prefix_name(self, name: str) -> str:
        """Add systemo_ prefix to user-defined names, but not built-ins, constructors, or pattern variables."""
        # Don't prefix local pattern variables
        if name in self.local_variables:
            return name
        # Don't prefix constructor names (they should be detected by their usage)
        # Add systemo_ prefix to all other names, with sanitization for operators
        sanitized_name = self._sanitize_operator_name(name)
        return f"systemo_{sanitized_name}"

    def _find_constructor_def(
        self,
        constructor_name: str,
    ) -> Optional["DataConstructor"]:
        """Find the constructor definition for a given constructor name."""
        for data_decl in self.data_types.values():
            for constructor in data_decl.constructors:
                if constructor.name == constructor_name:
                    return constructor
        return None

    def _get_function_final_return_type(self, func_name: str) -> str:
        """Get the final return type of a function, unwrapping curried functions."""
        if func_name not in self.function_types:
            return "Any"

        func_type = self.function_types[func_name]
        # Unwrap function types to get the final return type
        while isinstance(func_type, FunctionType):
            func_type = func_type.result

        return self._systemo_type_to_python_hint(func_type)

    def _get_function_arity(self, func_name: str) -> int:
        """Get the number of parameters (arity) of a function."""
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
        """Convert a TypeExpression to a Type, using the ty field if available."""
        if hasattr(type_expr, "ty") and type_expr.ty is not None:
            return type_expr.ty
        # If no type information is available, return None
        return None

    def _systemo_type_to_python_hint(self, systemo_type: Optional[Type]) -> str:
        """Convert a systemo type to a Python type hint string."""
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
        """Extract all variable names from a pattern recursively."""
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

    def _build_record_pattern_match(
        self,
        value_expr: str,
        constructor: str,
        field_name: str,
        var_name: str,
        body: Expression,
    ) -> str:
        """Build a readable pattern match for record constructors."""
        return build_record_pattern_match(
            value_expr,
            constructor,
            field_name,
            var_name,
            body,
            self._compile_expression,
        )

    def _build_positional_pattern_match(
        self,
        value_expr: str,
        constructor: str,
        var_name: str,
        body: Expression,
        arg_index: int = 0,
    ) -> str:
        return build_positional_pattern_match(
            value_expr,
            constructor,
            var_name,
            body,
            self._compile_expression,
            arg_index,
        )

    def _build_multi_arg_pattern_match(
        self,
        value_expr: str,
        constructor: str,
        assignments: List[str],
        body: Expression,
    ) -> str:
        return build_multi_arg_pattern_match(
            value_expr,
            constructor,
            assignments,
            body,
            self._compile_expression,
        )

    def _build_literal_pattern_match(
        self,
        value_expr: str,
        value: Any,
        body: Expression,
    ) -> str:
        return build_literal_pattern_match(
            value_expr,
            value,
            body,
            self._compile_expression,
            self._compile_literal_value,
        )

    def _build_cons_pattern_match(
        self,
        value_expr: str,
        head_var: Optional[str],
        tail_var: Optional[str],
        body: Expression,
    ) -> str:
        """Build a readable pattern match for cons patterns (x:xs)."""
        return build_cons_pattern_match(
            value_expr,
            head_var,
            tail_var,
            body,
            self._compile_expression,
        )

    def _build_tuple_pattern_match(
        self,
        value_expr: str,
        tuple_vars: List[str],
        body: Expression,
    ) -> str:
        """Build a readable pattern match for tuple patterns."""
        return build_tuple_pattern_match(
            value_expr,
            tuple_vars,
            body,
            self._compile_expression,
        )

    def _build_list_pattern_match(
        self,
        value_expr: str,
        list_vars: List[str],
        body: Expression,
    ) -> str:
        """Build a readable pattern match for list patterns."""
        return build_list_pattern_match(
            value_expr,
            list_vars,
            body,
            self._compile_expression,
        )

    def _build_simple_pattern_match(
        self,
        value_expr: str,
        constructor: str,
        body: Expression,
    ) -> str:
        """Build a readable pattern match for constructors with no arguments."""
        return build_simple_pattern_match(
            value_expr,
            constructor,
            body,
            self._compile_expression,
        )

    def compile(self, program: Program) -> str:
        lines = []
        with open("lango/systemo/compiler/prelude.py", "r") as f:
            lines.extend(f.read().splitlines())

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

        # Generate function definitions
        for func_name, definitions in function_definitions.items():
            # Skip built-in functions to avoid conflicts
            if func_name not in ["error"]:
                lines.append(self._compile_function_group(func_name, definitions))

        # Generate instance declarations (monomorphized functions)
        for instance_name, instances in instance_declarations.items():
            lines.extend(self._compile_instances(instance_name, instances))

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

    def _compile_instances(
        self,
        instance_name: str,
        instances: List["InstanceDeclaration"],
    ) -> List[str]:
        """Compile instance declarations into monomorphized functions with proper type dispatch."""
        lines = []

        # Extract the actual function name from parsing artifacts
        actual_function_name = self._extract_function_name(instance_name)

        # Skip generating dispatcher for functions already defined in prelude
        # These functions already have proper implementations that handle all cases
        prelude_functions = {"error"}
        if actual_function_name in prelude_functions:
            return []  # Don't generate any instance functions

        # Use the same sanitization method as for symbolic operations
        safe_instance_name = self._sanitize_operator_name(actual_function_name)

        # Ensure it starts with a letter for valid Python identifier
        if not safe_instance_name or not safe_instance_name[0].isalpha():
            safe_instance_name = f"op_{safe_instance_name}"

        # Group instances by type signature to handle multiple patterns for same type
        type_to_instances = {}
        for instance in instances:
            type_sig = self._type_expression_to_string(instance.type_signature)
            if type_sig not in type_to_instances:
                type_to_instances[type_sig] = []
            type_to_instances[type_sig].append(instance)

        # Generate monomorphized functions for each unique type signature
        type_to_function = {}

        for i, (type_sig, grouped_instances) in enumerate(type_to_instances.items()):
            monomorphized_name = f"systemo_{safe_instance_name}_{i}"
            type_to_function[type_sig] = monomorphized_name

            if len(grouped_instances) == 1:
                # Single instance - use existing compilation methods
                instance = grouped_instances[0]
                func_def = instance.function_definition

                if len(func_def.patterns) == 2:
                    # Binary operator - generate a curried function
                    compiled_func = self._compile_binary_instance_function(
                        func_def,
                        monomorphized_name,
                    )
                else:
                    # Single argument or nullary - use existing method
                    compiled_func = self._compile_simple_function(
                        func_def,
                        monomorphized_name,
                    )
            else:
                # Multiple instances with same type signature - combine into one function
                compiled_func = self._compile_grouped_instance_function(
                    grouped_instances,
                    monomorphized_name,
                )

            lines.append(compiled_func)

        # Generate a type-aware dispatcher function
        lines.append(
            self._generate_typed_dispatcher(
                safe_instance_name,
                instances,
                type_to_function,
            ),
        )

        # Generate a binary-specific dispatcher for binary operations
        lines.append(
            self._generate_typed_binary_dispatcher(
                safe_instance_name,
                instances,
                type_to_function,
            ),
        )

        return lines

    def _extract_function_name(self, instance_name) -> str:
        """Extract the actual function name from parsing artifacts."""
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

    def _generate_type_dispatcher_simple(
        self,
        instance_name: str,
        instances: List["InstanceDeclaration"],
    ) -> str:
        """Generate a simple dispatcher function that tries each monomorphized version."""
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

    def _generate_binary_dispatcher(
        self,
        instance_name: str,
        instances: List["InstanceDeclaration"],
    ) -> str:
        """Generate a binary-specific dispatcher that only considers binary instances."""
        prefixed_name = f"systemo_{instance_name}_binary"
        lines = [f"def {prefixed_name}(arg_0: Any, arg_1: Any) -> Any:"]

        # Filter instances to only include binary ones (those with 2 patterns)
        binary_instances = [
            (i, instance)
            for i, instance in enumerate(instances)
            if len(instance.function_definition.patterns) == 2
        ]

        if len(binary_instances) == 0:
            lines.append(
                f"    raise ValueError(f'No binary instances of {instance_name}')",
            )
            lines.append("")
            return "\n".join(lines)

        # Try each binary instance in sequence
        for original_index, instance in binary_instances:
            monomorphized_name = f"systemo_{instance_name}_{original_index}"
            lines.append(f"    try:")
            lines.append(f"        return {monomorphized_name}(arg_0, arg_1)")
            lines.append(f"    except (ValueError, AttributeError):")
            lines.append(f"        pass")

        # Final fallback
        lines.append(
            f"    raise ValueError(f'No binary instance of {instance_name} for types {{type(arg_0).__name__}}, {{type(arg_1).__name__}}')",
        )

        lines.append("")
        return "\n".join(lines)

    def _generate_type_dispatcher(
        self,
        instance_name: str,
        instances: List["InstanceDeclaration"],
    ) -> str:
        """Generate a dispatcher function that calls the right monomorphized version."""
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
        """Compile function definitions with pattern matching."""
        prefixed_func_name = f"systemo_{func_name}"
        self.defined_functions.add(func_name)

        # Store function type information
        if definitions and definitions[0].ty:
            self.function_types[func_name] = definitions[0].ty

        # Check if any definition is nullary
        if any(len(defn.patterns) == 0 for defn in definitions):
            self.nullary_functions.add(func_name)

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
        """Compile a binary function (2 parameters) for instance declarations."""
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
        """Compile a simple function with at most one parameter."""
        func_name = prefixed_name or f"systemo_{func_def.function_name}"

        # Track pattern variables
        old_local_vars = self.local_variables.copy()
        for pattern in func_def.patterns:
            self.local_variables.update(self._extract_pattern_variables(pattern))

        # Get type hints from the function's type annotation
        return_type_hint = "Any"
        param_type_hint = "Any"

        if func_def.ty:
            match func_def.ty:
                case FunctionType(param=param_type, result=result_type):
                    param_type_hint = self._systemo_type_to_python_hint(param_type)
                    return_type_hint = self._systemo_type_to_python_hint(result_type)
                case _:
                    # Not a function type, use it as return type
                    return_type_hint = self._systemo_type_to_python_hint(func_def.ty)

        if len(func_def.patterns) == 0:
            # Check if the body is a do block with multiple statements
            match func_def.body:
                case DoBlock(statements=statements) if len(statements) > 1:
                    lines = [f"def {func_name}() -> {return_type_hint}:"]
                    lines.extend(self._compile_do_block_as_statements(func_def.body))
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
                                self.lambda_lifted_functions.add(func_def.function_name)
                                params_with_types = ", ".join(
                                    f"{param}: Any" for param in params_str.split(", ")
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
                    lines = [
                        f"def {func_name}(arg: {param_type_hint}) -> {return_type_hint}:",
                        f"    {self._compile_pattern_match(pattern, 'arg', func_def.body)}",
                    ]

        lines.append("")

        # Restore local variables
        self.local_variables = old_local_vars

        return "\n".join(lines)

    def _compile_do_block_as_statements(self, do_block: DoBlock) -> List[str]:
        """Compile a do block as a series of Python statements within a function."""
        lines = []

        # Process all statements except the last one
        for stmt in do_block.statements[:-1]:
            match stmt:
                case LetStatement(variable=variable, value=value):
                    prefixed_var = self._prefix_name(variable)
                    lines.append(
                        f"    {prefixed_var} = {self._compile_expression(value)}",
                    )
                case _ if is_expression(stmt):
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
            case _ if is_expression(last_stmt):
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
                return self._build_literal_pattern_match(value_expr, value, body)
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

                return self._build_cons_pattern_match(
                    value_expr,
                    head_var,
                    tail_var,
                    body,
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

                return self._build_tuple_pattern_match(
                    value_expr,
                    tuple_vars,
                    body,
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
                                return self._build_record_pattern_match(
                                    value_expr,
                                    constructor,
                                    field_name,
                                    var_name,
                                    body,
                                )
                            else:
                                # Positional constructor - use arg_ access
                                return self._build_positional_pattern_match(
                                    value_expr,
                                    constructor,
                                    var_name,
                                    body,
                                    arg_index=0,
                                )
                        case _:
                            # Handle non-variable patterns with single argument
                            return self._build_simple_pattern_match(
                                value_expr,
                                constructor,
                                body,
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

                    return self._build_multi_arg_pattern_match(
                        value_expr,
                        constructor,
                        assignments,
                        body,
                    )
                else:
                    # Constructor with no arguments
                    return self._build_simple_pattern_match(
                        value_expr,
                        constructor,
                        body,
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

                return self._build_list_pattern_match(
                    value_expr,
                    list_vars,
                    body,
                )
            case _:
                return f"return {self._compile_expression(body)}"

    def _compile_literal_value(self, value: Any) -> str:
        return compile_literal_value(value)

    def _compile_symbolic_operation(
        self,
        operator: str,
        operands: List["Expression"],
    ) -> str:
        """Compile symbolic operations (operators) to Python code."""
        # Handle binary operators
        if len(operands) == 2:
            left = self._compile_expression(operands[0])
            right = self._compile_expression(operands[1])

            # For binary operators, call the dispatcher with both arguments directly
            # to avoid the issue with unary/binary instance ambiguity
            safe_op_name = self._sanitize_operator_name(operator)

            # Use a special binary calling convention to distinguish from unary operations
            return f"systemo_{safe_op_name}_binary({left}, {right})"

        # Handle unary operators
        elif len(operands) == 1:
            operand = self._compile_expression(operands[0])

            # For unary operators, use the regular dispatcher
            safe_op_name = self._sanitize_operator_name(operator)
            return f"systemo_{safe_op_name}({operand})"

        # Fallback for other cases
        else:
            compiled_operands = [self._compile_expression(op) for op in operands]
            safe_op_name = self._sanitize_operator_name(operator)
            return f"systemo_{safe_op_name}({', '.join(compiled_operands)})"

    def _sanitize_operator_name(self, operator: str) -> str:
        """Convert operator symbols to safe Python function names."""
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

    def _compile_expression(self, expr: Expression) -> str:
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
                return f"('char', '{value}')"  # Use tuple to distinguish from strings
            case BoolLiteral(value=value):
                return str(value)
            case ListLiteral(elements=elements):
                compiled_elements = [
                    self._compile_expression(elem) for elem in elements
                ]
                return f"[{', '.join(compiled_elements)}]"

            case TupleLiteral(elements=elements):
                compiled_elements = [
                    self._compile_expression(elem) for elem in elements
                ]
                # Ensure we have proper tuple syntax - add comma for single element
                if len(compiled_elements) == 1:
                    return f"({compiled_elements[0]},)"
                return f"({', '.join(compiled_elements)})"

                # Variables and constructors
                return "systemo_put_str"
            case Variable(name="error"):
                return "systemo_error"
            case Variable(name=name):
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
            case Constructor(name=name):
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

            case IfElse(condition=condition, then_expr=then_expr, else_expr=else_expr):
                return f"({self._compile_expression(then_expr)} if {self._compile_expression(condition)} else {self._compile_expression(else_expr)})"

            # Function application
            case FunctionApplication():
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
            case SymbolicOperation(operator=operator, operands=operands):
                return self._compile_symbolic_operation(operator, operands)

            # Default case
            case _:
                return f"None  # Unsupported: {type(expr)}"

    def _compile_do_block(self, do_block: DoBlock) -> str:
        # For single statements, we can still inline them
        if len(do_block.statements) == 1:
            stmt = do_block.statements[0]
            match stmt:
                case LetStatement(value=value):
                    return f"(lambda: {self._compile_expression(value)})()"
                case _ if is_expression(stmt):
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
                case _ if is_expression(stmt):
                    parts.append(self._compile_expression_safe(stmt))
                case _:
                    pass

        # Handle the last statement (which becomes the return value)
        last_stmt = do_block.statements[-1]
        match last_stmt:
            case LetStatement(variable=variable, value=value):
                prefixed_var = self._prefix_name(variable)
                final_expr = f"globals().update({{'{prefixed_var}': {self._compile_expression(value)}}})"
            case _ if is_expression(last_stmt):
                final_expr = self._compile_expression_safe(last_stmt)
            case _:
                final_expr = "None"

        if parts:
            return f"({' or '.join(parts)} or {final_expr})"
        else:
            return final_expr

    def _compile_expression_safe(self, stmt: Any) -> str:
        """Safely compile a statement as an expression."""
        return self._compile_expression(stmt) if is_expression(stmt) else "None"

    def _type_expression_to_string(self, type_expr) -> str:
        """Convert a type expression to a string representation."""
        if hasattr(type_expr, "__class__"):
            class_name = type_expr.__class__.__name__

            if hasattr(type_expr, "name"):
                # Basic types like Int, Float, Bool
                return str(type_expr.name)
            elif hasattr(type_expr, "param") and hasattr(type_expr, "result"):
                # Function types
                param_str = self._type_expression_to_string(type_expr.param)
                result_str = self._type_expression_to_string(type_expr.result)
                return f"{param_str} -> {result_str}"
            elif hasattr(type_expr, "constructor"):
                # Constructor types
                return str(type_expr.constructor)
            else:
                return str(type_expr)
        else:
            return str(type_expr)

    def _generate_typed_dispatcher(
        self,
        instance_name: str,
        instances: List["InstanceDeclaration"],
        type_to_function: Dict[str, str],
    ) -> str:
        """Generate a typed dispatcher function using proper type-based selection."""
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
        """Generate a typed binary dispatcher for operations with two arguments."""
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

    def _extract_binary_param_types_from_type_expression(self, type_expr) -> List[str]:
        """Extract parameter types from a binary function type expression."""
        result = []
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
        """Compile multiple instance declarations with same type signature into one function."""
        if not grouped_instances:
            return ""

        if len(grouped_instances) == 1:
            # Single instance - use existing method based on number of patterns
            instance = grouped_instances[0]
            func_def = instance.function_definition
            if len(func_def.patterns) == 2:
                return self._compile_binary_instance_function(func_def, function_name)
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
                self.local_variables.update(self._extract_pattern_variables(pattern))

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

    def _extract_first_param_type_from_signature(self, type_sig: str) -> str:
        """Extract the first parameter type from a type signature string."""
        if " -> " in type_sig:
            # Function type like "Int -> Int -> Int"
            parts = type_sig.split(" -> ")
            return parts[0].strip()
        else:
            # Simple type
            return type_sig.strip()

    def _extract_binary_param_types_from_signature(self, type_sig: str) -> List[str]:
        """Extract parameter types from a binary function type signature."""
        if " -> " in type_sig:
            parts = type_sig.split(" -> ")
            # For a binary function like "Int -> Int -> Int", we want the first two parts
            return [part.strip() for part in parts[:2]]
        else:
            return [type_sig.strip()]

    def _extract_first_param_type_name(self, type_expression) -> str:
        """Extract the first parameter type name from a type expression object."""
        # Handle ArrowType objects
        if hasattr(type_expression, "from_type"):
            return self._extract_type_name_from_expression(type_expression.from_type)
        else:
            return self._extract_type_name_from_expression(type_expression)

    def _extract_type_name_from_expression(self, type_expr) -> str:
        """Extract the type name from a type expression object."""
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

    def _extract_tuple_length(self, type_signature) -> Optional[int]:
        """Extract the length of a tuple type from a type signature."""
        from lango.systemo.ast.nodes import ArrowType, TupleType

        if isinstance(type_signature, ArrowType):
            # For ArrowType, check the from_type
            from_type = type_signature.from_type
            if isinstance(from_type, TupleType):
                return len(from_type.element_types)
        elif isinstance(type_signature, TupleType):
            # Direct tuple type
            return len(type_signature.element_types)

        return None

    def _compile_pattern_condition(self, pattern, arg_name: str) -> str:
        """Generate a Python condition to check if an argument matches a pattern."""
        from lango.systemo.ast.nodes import (
            ConsPattern,
            ConstructorPattern,
            ListPattern,
            LiteralPattern,
            TuplePattern,
            VariablePattern,
        )

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

    def _compile_pattern_bindings(self, pattern, arg_name: str) -> List[str]:
        """Generate Python statements to bind variables from a pattern match."""
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

    def _compile_literal(self, value) -> str:
        """Compile a literal value to Python code."""
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


def compile_program(program: Program) -> str:
    """Compile a systemo program to Python code."""
    compiler = systemoCompiler()
    return compiler.compile(program)
