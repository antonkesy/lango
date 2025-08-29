from typing import Any, Dict, List, Optional, Set

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
)


class systemoCompiler:
    def __init__(self) -> None:
        self.indent_level = 0
        self.defined_functions: Set[str] = set()
        self.nullary_functions: Set[str] = set()  # Functions with no parameters
        self.function_types: Dict[str, Type] = {}  # Track function types
        self.data_types: Dict[str, DataDeclaration] = {}
        self.local_variables: Set[str] = set()  # Track local pattern variables
        self.do_block_counter = 0  # Counter for unique do block function names

    def _indent(self) -> str:
        return "    " * self.indent_level

    def _prefix_name(self, name: str) -> str:
        """Add systemo_ prefix to user-defined names, but not built-ins, constructors, or pattern variables."""
        # Don't prefix built-in functions
        if name in ["show", "putStr"]:
            return name
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
        lines = [
            f"match {value_expr}:",
            f"        case {constructor}() if hasattr({value_expr}, 'fields') and '{field_name}' in {value_expr}.fields:",
            f"            {var_name} = {value_expr}.fields['{field_name}']",
            f"            return {self._compile_expression(body)}",
            "        case _:",
            "            raise ValueError('Pattern match failed')",
        ]
        return "\n".join(lines)

    def _build_positional_pattern_match(
        self,
        value_expr: str,
        constructor: str,
        var_name: str,
        body: Expression,
        arg_index: int = 0,
    ) -> str:
        lines = [
            f"match {value_expr}:",
            f"        case {constructor}():",
            f"            {var_name} = {value_expr}.arg_{arg_index}",
            f"            return {self._compile_expression(body)}",
            "        case _:",
            "            raise ValueError('Pattern match failed')",
        ]
        return "\n".join(lines)

    def _build_multi_arg_pattern_match(
        self,
        value_expr: str,
        constructor: str,
        assignments: List[str],
        body: Expression,
    ) -> str:
        lines = [f"match {value_expr}:", f"        case {constructor}():"]
        for assignment in assignments:
            lines.append(f"            {assignment}")
        lines.extend(
            [
                f"            return {self._compile_expression(body)}",
                "        case _:",
                "            raise ValueError('Pattern match failed')",
            ],
        )
        return "\n".join(lines)

    def _build_literal_pattern_match(
        self,
        value_expr: str,
        value: Any,
        body: Expression,
    ) -> str:
        lines = [
            f"if {value_expr} == {self._compile_literal_value(value)}:",
            f"        return {self._compile_expression(body)}",
            "    else:",
            "        raise ValueError('Pattern match failed')",
        ]
        return "\n".join(lines)

    def _build_cons_pattern_match(
        self,
        value_expr: str,
        head_var: Optional[str],
        tail_var: Optional[str],
        body: Expression,
    ) -> str:
        """Build a readable pattern match for cons patterns (x:xs)."""
        assignments = []
        if head_var:
            assignments.append(f"        {head_var} = {value_expr}[0]")
        if tail_var:
            assignments.append(f"        {tail_var} = {value_expr}[1:]")

        lines = [f"if len({value_expr}) > 0:"]
        lines.extend(assignments)
        lines.extend(
            [
                f"        return {self._compile_expression(body)}",
                "    else:",
                "        raise ValueError('Pattern match failed')",
            ],
        )
        return "\n".join(lines)

    def _build_tuple_pattern_match(
        self,
        value_expr: str,
        tuple_vars: List[str],
        body: Expression,
    ) -> str:
        """Build a readable pattern match for tuple patterns."""
        assignments = []
        for i, var in enumerate(tuple_vars):
            if not var.startswith("_tuple_elem_"):
                assignments.append(f"        {var} = {value_expr}[{i}]")

        lines = [f"if len({value_expr}) == {len(tuple_vars)}:"]
        lines.extend(assignments)
        lines.extend(
            [
                f"        return {self._compile_expression(body)}",
                "    else:",
                "        raise ValueError('Pattern match failed')",
            ],
        )
        return "\n".join(lines)

    def _build_list_pattern_match(
        self,
        value_expr: str,
        list_vars: List[str],
        body: Expression,
    ) -> str:
        """Build a readable pattern match for list patterns."""
        assignments = []
        for i, var in enumerate(list_vars):
            if not var.startswith("_list_elem_"):
                assignments.append(f"        {var} = {value_expr}[{i}]")

        lines = [f"if len({value_expr}) == {len(list_vars)}:"]
        lines.extend(assignments)
        lines.extend(
            [
                f"        return {self._compile_expression(body)}",
                "    else:",
                "        raise ValueError('Pattern match failed')",
            ],
        )
        return "\n".join(lines)

    def _build_simple_pattern_match(
        self,
        value_expr: str,
        constructor: str,
        body: Expression,
    ) -> str:
        """Build a readable pattern match for constructors with no arguments."""
        lines = [
            f"match {value_expr}:",
            f"        case {constructor}():",
            f"            return {self._compile_expression(body)}",
            "        case _:",
            "            raise ValueError('Pattern match failed')",
        ]
        return "\n".join(lines)

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
            if func_name not in ["show", "putStr", "error"]:
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
        """Compile instance declarations into monomorphized functions."""
        lines = []

        # Extract the actual function name from parsing artifacts
        actual_function_name = self._extract_function_name(instance_name)

        # Skip generating dispatcher for functions already defined in prelude
        # These functions already have proper implementations that handle all cases
        prelude_functions = {"show", "putStr", "error"}
        if actual_function_name in prelude_functions:
            return []  # Don't generate any instance functions

        # Use the same sanitization method as for symbolic operations
        safe_instance_name = self._sanitize_operator_name(actual_function_name)

        # Ensure it starts with a letter for valid Python identifier
        if not safe_instance_name or not safe_instance_name[0].isalpha():
            safe_instance_name = f"op_{safe_instance_name}"

        for i, instance in enumerate(instances):
            # Generate a simple monomorphized function name using index
            monomorphized_name = f"systemo_{safe_instance_name}_{i}"

            # Check if this is a binary operator (2 patterns) or unary/nullary
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

            lines.append(compiled_func)

        # Generate a dispatcher function that calls the appropriate monomorphized version
        # based on the runtime type of the argument
        lines.append(
            self._generate_type_dispatcher_simple(safe_instance_name, instances),
        )

        # Generate a binary-specific dispatcher for binary operations to avoid unary/binary conflicts
        lines.append(
            self._generate_binary_dispatcher(safe_instance_name, instances),
        )

        return lines

    def _extract_function_name(self, instance_name: str) -> str:
        """Extract the actual function name from parsing artifacts."""
        import re

        # Look for operator patterns like Tree(Token('RULE', 'inst_operator_name'), ['/'])
        # Extract the operator from the list at the end
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

        # Look for simple word patterns
        word_match = re.search(r"\b([a-zA-Z]\w*)\b", instance_name)
        if word_match:
            return word_match.group(1)

        # Fallback
        return "instance"

    def _extract_first_param_type_name(self, type_signature: Any) -> Optional[str]:
        """Extract the first parameter type name from a type signature."""
        # Simplified approach: if we have type information, try to extract it
        if hasattr(type_signature, "ty") and type_signature.ty:
            type_obj = type_signature.ty
            if hasattr(type_obj, "param"):
                param_type = type_obj.param
                if hasattr(param_type, "name"):
                    return param_type.name
                # Handle TypeCon case
                if (
                    hasattr(param_type, "__class__")
                    and param_type.__class__.__name__ == "TypeCon"
                ):
                    return getattr(param_type, "name", None)

        # Fallback: try to extract from string representation
        type_str = str(type_signature)
        # Look for pattern like "TypeName -> ..."
        import re

        match = re.search(r"(\w+)\s*->", type_str)
        if match:
            return match.group(1)

        return None

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

        # Use @curry decorator for multi-parameter functions
        if max_params > 1:
            # For curried functions, get parameter types from the function type
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

            curried_return_type = (
                f"Union[Callable[[Any], {return_type_hint}], {return_type_hint}]"
            )

            lines = [
                "@curry",
                f"def {prefixed_func_name}({', '.join(param_list)}) -> {curried_return_type}:",
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
            f"@curry",
            f"def {prefixed_name}(arg_0: Any, arg_1: Any) -> Union[Callable[[Any], Any], Any]:",
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
                    lines = [
                        f"def {func_name}() -> {return_type_hint}:",
                        f"    return {self._compile_expression(func_def.body)}",
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
        match value:
            case str():
                return f'"{value}"'
            case bool():
                return str(value)
            case list():
                return "[]"  # Handle empty list explicitly
            case _:
                return str(value)

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
            case Variable(name="show"):
                return "systemo_show"
            case Variable(name="putStr"):
                return "systemo_put_str"
            case Variable(name="error"):
                return "systemo_error"
            case Variable(name=name):
                # Primitive functions should not be prefixed
                if name.startswith("prim"):
                    if name in self.nullary_functions:
                        return f"{name}()"
                    else:
                        return name
                else:
                    prefixed_name = self._prefix_name(name)
                    if name in self.nullary_functions:
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

                        # Check if this is a user-defined function that might be curried
                        prefixed_name = self._prefix_name(name)
                        if len(args) > 1 and name in self.defined_functions:
                            # Multi-argument call for curried functions
                            arg_exprs = [self._compile_expression(arg) for arg in args]
                            # Use cast with proper return type to resolve Union type issues
                            return_type = self._get_function_final_return_type(name)
                            return f"cast({return_type}, {prefixed_name}({', '.join(arg_exprs)}))"
                        else:
                            # Single argument or built-in function
                            func_expr = self._compile_expression(current)
                            if len(args) == 1:
                                arg_expr = self._compile_expression(args[0])
                                return f"{func_expr}({arg_expr})"
                            else:
                                # Multiple args but not a defined function, use nested calls
                                result = func_expr
                                for arg in args:
                                    arg_expr = self._compile_expression(arg)
                                    result = f"{result}({arg_expr})"
                                return result
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
        """Check if a statement is an expression."""
        match stmt:
            case (
                IntLiteral()
                | FloatLiteral()
                | StringLiteral()
                | CharLiteral()
                | BoolLiteral()
                | ListLiteral()
                | Variable()
                | Constructor()
                | SymbolicOperation()
                | FunctionApplication()
                | ConstructorExpression()
                | DoBlock()
                | GroupedExpression()
                | IfElse()
            ):
                return True
            case _:
                return False

    def _compile_expression_safe(self, stmt: Any) -> str:
        """Safely compile a statement as an expression."""
        match stmt:
            case _ if self._is_expression(stmt):
                return self._compile_expression(stmt)  # type: ignore
            case _:
                return "None"


def compile_program(program: Program) -> str:
    """Compile a systemo program to Python code."""
    compiler = systemoCompiler()
    return compiler.compile(program)
