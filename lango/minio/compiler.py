from typing import Any, Dict, List, Optional, Set

from lango.minio.ast_nodes import *


class MinioCompiler:
    def __init__(self):
        self.indent_level = 0
        self.defined_functions: Set[str] = set()
        self.nullary_functions: Set[str] = set()  # Functions with no parameters
        self.data_types: Dict[str, DataDeclaration] = {}
        self.local_variables: Set[str] = set()  # Track local pattern variables
        self.do_block_counter = 0  # Counter for unique do block function names

    def _indent(self) -> str:
        return "    " * self.indent_level

    def _prefix_name(self, name: str) -> str:
        """Add minio_ prefix to user-defined names, but not built-ins, constructors, or pattern variables."""
        # Don't prefix built-in functions
        if name in ["show", "putStr"]:
            return name
        # Don't prefix local pattern variables
        if name in self.local_variables:
            return name
        # Don't prefix constructor names (they should be detected by their usage)
        # Add minio_ prefix to all other names
        return f"minio_{name}"

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

    def _extract_pattern_variables(self, pattern: Pattern) -> Set[str]:
        """Extract all variable names from a pattern recursively."""
        variables = set()
        if isinstance(pattern, VariablePattern):
            variables.add(pattern.name)
        elif isinstance(pattern, ConstructorPattern):
            for sub_pattern in pattern.patterns:
                variables.update(self._extract_pattern_variables(sub_pattern))
        elif isinstance(pattern, ConsPattern):
            variables.update(self._extract_pattern_variables(pattern.head))
            variables.update(self._extract_pattern_variables(pattern.tail))
        # Other pattern types don't contain variables
        return variables

    def compile(self, program: Program) -> str:
        lines = [
            "# Generated Python code from Minio",
            "from typing import Any, Dict, List, Union, Optional",
            "",
            "# Runtime support functions",
            "def minio_show(value):",
            "    if isinstance(value, bool):",
            "        return 'True' if value else 'False'",
            "    elif isinstance(value, str):",
            "        return value",
            "    elif isinstance(value, list):",
            "        elements = [minio_show(x) for x in value]",
            "        return '[' + ', '.join(elements) + ']'",
            "    else:",
            "        return str(value)",
            "",
            "def minio_put_str(s):",
            "    print(s, end='')",
            "",
            "",
        ]

        # Collect data types
        for stmt in program.statements:
            if isinstance(stmt, DataDeclaration):
                self.data_types[stmt.type_name] = stmt

        # Group function definitions by name
        function_definitions: Dict[str, List[FunctionDefinition]] = {}

        for stmt in program.statements:
            if isinstance(stmt, DataDeclaration):
                lines.append(self._compile_data_declaration(stmt))
            elif isinstance(stmt, FunctionDefinition):
                if stmt.function_name not in function_definitions:
                    function_definitions[stmt.function_name] = []
                function_definitions[stmt.function_name].append(stmt)
            elif isinstance(stmt, LetStatement):
                prefixed_var = self._prefix_name(stmt.variable)
                lines.append(
                    f"{prefixed_var} = {self._compile_expression(stmt.value)}",
                )

        # Generate function definitions
        for func_name, definitions in function_definitions.items():
            # Skip built-in functions to avoid conflicts
            if func_name not in ["show", "putStr"]:
                lines.append(self._compile_function_group(func_name, definitions))

        # Add main execution
        if "main" in function_definitions:
            lines.extend(["", "if __name__ == '__main__':", "    minio_main()"])

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
                args = [f"arg_{i}" for i in range(len(field_names))]

                lines.extend(
                    [
                        f"class {class_name}:",
                        f"    def __init__(self, {', '.join(args)}):",
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
                args = [f"arg_{i}" for i in range(arg_count)]

                lines.extend(
                    [
                        f"class {class_name}:",
                        f"    def __init__(self, {', '.join(args)}):",
                    ],
                )
                for i, arg in enumerate(args):
                    lines.append(f"        self.arg_{i} = {arg}")
            else:
                # No arguments
                lines.extend(
                    [
                        f"class {class_name}:",
                        "    def __init__(self):",
                        "        pass",
                    ],
                )

        lines.append("")
        return "\n".join(lines)

    def _compile_function_group(
        self,
        func_name: str,
        definitions: List[FunctionDefinition],
    ) -> str:
        """Compile function definitions with pattern matching."""
        prefixed_func_name = f"minio_{func_name}"
        self.defined_functions.add(func_name)

        # Check if any definition is nullary
        if any(len(defn.patterns) == 0 for defn in definitions):
            self.nullary_functions.add(func_name)

        if len(definitions) == 1 and len(definitions[0].patterns) <= 1:
            return self._compile_simple_function(definitions[0], prefixed_func_name)

        # Multiple definitions or pattern matching - use *args approach
        lines = [f"def {prefixed_func_name}(*args):"]
        self.indent_level += 1

        # Collect all pattern variables from all definitions
        old_local_vars = self.local_variables.copy()
        for func_def in definitions:
            for pattern in func_def.patterns:
                self.local_variables.update(self._extract_pattern_variables(pattern))

        for i, func_def in enumerate(definitions):
            if len(func_def.patterns) == 0:
                # Nullary function - no arguments expected
                lines.append(
                    self._indent()
                    + f"return {self._compile_expression(func_def.body)}",
                )
            else:
                # Pattern matching - check each pattern
                pattern_matches = []
                assignments = []

                for j, pattern in enumerate(func_def.patterns):
                    if isinstance(pattern, VariablePattern):
                        assignments.append(f"{pattern.name} = args[{j}]")
                    elif isinstance(pattern, LiteralPattern):
                        pattern_matches.append(
                            f"args[{j}] == {self._compile_literal_value(pattern.value)}",
                        )
                    elif isinstance(pattern, ConsPattern):
                        # Cons pattern (x:xs) - check if list is non-empty and destructure
                        pattern_matches.append(f"len(args[{j}]) > 0")
                        if isinstance(pattern.head, VariablePattern):
                            assignments.append(f"{pattern.head.name} = args[{j}][0]")
                        if isinstance(pattern.tail, VariablePattern):
                            assignments.append(f"{pattern.tail.name} = args[{j}][1:]")
                    elif isinstance(pattern, ConstructorPattern):
                        # Constructor pattern - check type and destructure
                        pattern_matches.append(
                            f"isinstance(args[{j}], {pattern.constructor})",
                        )

                        constructor_def = self._find_constructor_def(
                            pattern.constructor,
                        )

                        if constructor_def and constructor_def.record_constructor:
                            # Record constructor - use dictionary access by field name
                            for k, sub_pattern in enumerate(pattern.patterns):
                                if isinstance(sub_pattern, VariablePattern):
                                    field_name = (
                                        constructor_def.record_constructor.fields[
                                            k
                                        ].name
                                    )
                                    assignments.append(
                                        f"{sub_pattern.name} = args[{j}].fields['{field_name}']",
                                    )
                        else:
                            # Positional constructor - use arg_ access
                            for k, sub_pattern in enumerate(pattern.patterns):
                                if isinstance(sub_pattern, VariablePattern):
                                    assignments.append(
                                        f"{sub_pattern.name} = args[{j}].arg_{k}",
                                    )
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
                    # Only variable patterns, always matches
                    for assignment in assignments:
                        lines.append(self._indent() + assignment)
                    lines.append(
                        self._indent()
                        + f"return {self._compile_expression(func_def.body)}",
                    )

        lines.append(
            self._indent()
            + f"raise ValueError(f'No matching pattern for {prefixed_func_name} with args: {{args}}')",
        )
        self.indent_level -= 1

        # Restore local variables
        self.local_variables = old_local_vars

        lines.append("")
        return "\n".join(lines)

    def _compile_simple_function(
        self,
        func_def: FunctionDefinition,
        prefixed_name: Optional[str] = None,
    ) -> str:
        """Compile a simple function with at most one parameter."""
        func_name = prefixed_name or f"minio_{func_def.function_name}"

        # Track pattern variables
        old_local_vars = self.local_variables.copy()
        for pattern in func_def.patterns:
            self.local_variables.update(self._extract_pattern_variables(pattern))

        if len(func_def.patterns) == 0:
            # Check if the body is a do block with multiple statements
            if isinstance(func_def.body, DoBlock) and len(func_def.body.statements) > 1:
                lines = [f"def {func_name}():"]
                lines.extend(self._compile_do_block_as_statements(func_def.body))
            else:
                lines = [
                    f"def {func_name}():",
                    f"    return {self._compile_expression(func_def.body)}",
                ]
        else:
            pattern = func_def.patterns[0]
            if isinstance(pattern, VariablePattern):
                # Check if the body is a do block with multiple statements
                if (
                    isinstance(func_def.body, DoBlock)
                    and len(func_def.body.statements) > 1
                ):
                    lines = [f"def {func_name}({pattern.name}):"]
                    lines.extend(self._compile_do_block_as_statements(func_def.body))
                else:
                    lines = [
                        f"def {func_name}({pattern.name}):",
                        f"    return {self._compile_expression(func_def.body)}",
                    ]
            else:
                lines = [
                    f"def {func_name}(arg):",
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
            if isinstance(stmt, LetStatement):
                prefixed_var = self._prefix_name(stmt.variable)
                lines.append(
                    f"    {prefixed_var} = {self._compile_expression(stmt.value)}",
                )
            elif self._is_expression(stmt):
                # Handle expression statements (like putStr calls)
                lines.append(f"    {self._compile_expression_safe(stmt)}")

        # Handle the last statement (which becomes the return value)
        last_stmt = do_block.statements[-1]
        if isinstance(last_stmt, LetStatement):
            prefixed_var = self._prefix_name(last_stmt.variable)
            lines.append(
                f"    {prefixed_var} = {self._compile_expression(last_stmt.value)}",
            )
            lines.append(f"    return {prefixed_var}")
        elif self._is_expression(last_stmt):
            lines.append(f"    return {self._compile_expression_safe(last_stmt)}")
        else:
            lines.append("    return None")

        return lines

    def _compile_pattern_match(
        self,
        pattern: Pattern,
        value_expr: str,
        body: Expression,
    ) -> str:
        if isinstance(pattern, VariablePattern):
            self.local_variables.add(pattern.name)
            return f"{pattern.name} = {value_expr}\n    return {self._compile_expression(body)}"
        elif isinstance(pattern, LiteralPattern):
            return f"if {value_expr} == {self._compile_literal_value(pattern.value)}:\n        return {self._compile_expression(body)}\n    else:\n        raise ValueError('Pattern match failed')"
        elif isinstance(pattern, ConsPattern):
            # Cons pattern (x:xs) - destructure list
            head_var = None
            tail_var = None
            if isinstance(pattern.head, VariablePattern):
                head_var = pattern.head.name
                self.local_variables.add(head_var)
            if isinstance(pattern.tail, VariablePattern):
                tail_var = pattern.tail.name
                self.local_variables.add(tail_var)

            assignments = []
            if head_var:
                assignments.append(f"        {head_var} = {value_expr}[0]")
            if tail_var:
                assignments.append(f"        {tail_var} = {value_expr}[1:]")

            assignment_lines = "\n".join(assignments)
            return f"if len({value_expr}) > 0:\n{assignment_lines}\n        return {self._compile_expression(body)}\n    else:\n        raise ValueError('Pattern match failed')"
        elif isinstance(pattern, ConstructorPattern):
            # Extract variables from constructor pattern
            for sub_pattern in pattern.patterns:
                self.local_variables.update(
                    self._extract_pattern_variables(sub_pattern),
                )

            # Generate destructuring assignment for constructor pattern
            if len(pattern.patterns) == 1 and isinstance(
                pattern.patterns[0],
                VariablePattern,
            ):
                var_name = pattern.patterns[0].name
                constructor_def = self._find_constructor_def(pattern.constructor)

                if constructor_def and constructor_def.record_constructor:
                    # Record constructor - use dictionary access
                    field_name = constructor_def.record_constructor.fields[0].name
                    return f"if isinstance({value_expr}, {pattern.constructor}):\n        {var_name} = {value_expr}.fields['{field_name}']\n        return {self._compile_expression(body)}\n    else:\n        raise ValueError('Pattern match failed')"
                else:
                    # Positional constructor - use arg_ access
                    return f"if isinstance({value_expr}, {pattern.constructor}):\n        {var_name} = {value_expr}.arg_0\n        return {self._compile_expression(body)}\n    else:\n        raise ValueError('Pattern match failed')"
            elif len(pattern.patterns) > 1:
                # Multiple variables in constructor pattern
                constructor_def = self._find_constructor_def(pattern.constructor)

                assignments = []
                if constructor_def and constructor_def.record_constructor:
                    # Record constructor - use dictionary access
                    for k, sub_pattern in enumerate(pattern.patterns):
                        if isinstance(sub_pattern, VariablePattern):
                            field_name = constructor_def.record_constructor.fields[
                                k
                            ].name
                            assignments.append(
                                f"        {sub_pattern.name} = {value_expr}.fields['{field_name}']",
                            )
                else:
                    # Positional constructor - use arg_ access
                    for k, sub_pattern in enumerate(pattern.patterns):
                        if isinstance(sub_pattern, VariablePattern):
                            assignments.append(
                                f"        {sub_pattern.name} = {value_expr}.arg_{k}",
                            )

                assignment_lines = "\n".join(assignments)
                return f"if isinstance({value_expr}, {pattern.constructor}):\n{assignment_lines}\n        return {self._compile_expression(body)}\n    else:\n        raise ValueError('Pattern match failed')"
            else:
                # Constructor with no arguments
                return f"if isinstance({value_expr}, {pattern.constructor}):\n        return {self._compile_expression(body)}\n    else:\n        raise ValueError('Pattern match failed')"
        else:
            return f"return {self._compile_expression(body)}"

    def _compile_literal_value(self, value: Any) -> str:
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, list):
            return "[]"  # Handle empty list explicitly
        else:
            return str(value)

    def _compile_expression(self, expr: Expression) -> str:
        if isinstance(expr, IntLiteral):
            return str(expr.value)
        elif isinstance(expr, FloatLiteral):
            return str(expr.value)
        elif isinstance(expr, StringLiteral):
            return f'"{expr.value}"'
        elif isinstance(expr, BoolLiteral):
            return str(expr.value)
        elif isinstance(expr, ListLiteral):
            elements = [self._compile_expression(elem) for elem in expr.elements]
            return f"[{', '.join(elements)}]"
        elif isinstance(expr, Variable):
            # Handle built-in functions - these take precedence
            if expr.name == "show":
                return "minio_show"
            elif expr.name == "putStr":
                return "minio_put_str"
            else:
                # User-defined variables and functions get minio_ prefix
                prefixed_name = self._prefix_name(expr.name)
                # Check if this is a nullary function (needs to be called)
                if expr.name in self.nullary_functions:
                    return f"{prefixed_name}()"
                else:
                    return prefixed_name
        elif isinstance(expr, Constructor):
            return expr.name
        elif isinstance(expr, AddOperation):
            return f"({self._compile_expression(expr.left)} + {self._compile_expression(expr.right)})"
        elif isinstance(expr, SubOperation):
            return f"({self._compile_expression(expr.left)} - {self._compile_expression(expr.right)})"
        elif isinstance(expr, MulOperation):
            return f"({self._compile_expression(expr.left)} * {self._compile_expression(expr.right)})"
        elif isinstance(expr, DivOperation):
            return f"({self._compile_expression(expr.left)} / {self._compile_expression(expr.right)})"
        elif isinstance(expr, EqualOperation):
            return f"({self._compile_expression(expr.left)} == {self._compile_expression(expr.right)})"
        elif isinstance(expr, ConcatOperation):
            return f"({self._compile_expression(expr.left)} + {self._compile_expression(expr.right)})"
        elif isinstance(expr, IfElse):
            return f"({self._compile_expression(expr.then_expr)} if {self._compile_expression(expr.condition)} else {self._compile_expression(expr.else_expr)})"
        elif isinstance(expr, FunctionApplication):
            # Check if this is a constructor application
            args = []
            current = expr

            # Collect all arguments for potential constructor calls
            while isinstance(current, FunctionApplication):
                args.insert(0, current.argument)
                current = current.function

            # If the base function is a constructor, generate single call with all args
            if isinstance(current, Constructor):
                arg_exprs = [self._compile_expression(arg) for arg in args]
                return f"{current.name}({', '.join(arg_exprs)})"
            else:
                # Regular curried function application
                func_expr = self._compile_expression(expr.function)
                arg_expr = self._compile_expression(expr.argument)
                return f"{func_expr}({arg_expr})"
        elif isinstance(expr, ConstructorExpression):
            constructor_name = expr.constructor_name
            if expr.fields:
                constructor_def = self._find_constructor_def(constructor_name)

                if constructor_def and constructor_def.record_constructor:
                    # Reorder fields according to declaration order
                    declared_fields = {
                        field.name: field
                        for field in constructor_def.record_constructor.fields
                    }
                    provided_fields = {field.field_name: field for field in expr.fields}

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
                        self._compile_expression(field.value) for field in expr.fields
                    ]
                    return f"{constructor_name}({', '.join(field_args)})"
            else:
                return f"{constructor_name}()"
        elif isinstance(expr, DoBlock):
            return self._compile_do_block(expr)
        elif isinstance(expr, GroupedExpression):
            return f"({self._compile_expression(expr.expression)})"
        else:
            return f"None  # Unsupported: {type(expr)}"

    def _compile_do_block(self, do_block: DoBlock) -> str:
        # For single statements, we can still inline them
        if len(do_block.statements) == 1:
            stmt = do_block.statements[0]
            if isinstance(stmt, LetStatement):
                return f"(lambda: {self._compile_expression(stmt.value)})()"
            elif self._is_expression(stmt):
                return self._compile_expression_safe(stmt)
            else:
                return "None"

        # For multiple statements in expression context, fall back to sequential execution
        # This is a bit hacky but works for simple cases
        parts = []
        for stmt in do_block.statements[:-1]:
            if isinstance(stmt, LetStatement):
                prefixed_var = self._prefix_name(stmt.variable)
                parts.append(
                    f"globals().update({{'{prefixed_var}': {self._compile_expression(stmt.value)}}})",
                )
            elif self._is_expression(stmt):
                parts.append(self._compile_expression_safe(stmt))

        # Handle the last statement (which becomes the return value)
        last_stmt = do_block.statements[-1]
        if isinstance(last_stmt, LetStatement):
            prefixed_var = self._prefix_name(last_stmt.variable)
            final_expr = f"globals().update({{'{prefixed_var}': {self._compile_expression(last_stmt.value)}}})"
        elif self._is_expression(last_stmt):
            final_expr = self._compile_expression_safe(last_stmt)
        else:
            final_expr = "None"

        if parts:
            return f"({' or '.join(parts)} or {final_expr})"
        else:
            return final_expr

    def _is_expression(self, stmt: Any) -> bool:
        """Check if a statement is an expression."""
        return isinstance(
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
                EqualOperation,
                ConcatOperation,
                IfElse,
                FunctionApplication,
                ConstructorExpression,
                DoBlock,
                GroupedExpression,
            ),
        )

    def _compile_expression_safe(self, stmt: Any) -> str:
        """Safely compile a statement as an expression."""
        if self._is_expression(stmt):
            return self._compile_expression(stmt)  # type: ignore
        else:
            return "None"


def compile_program(program: Program) -> str:
    """Compile a Minio program to Python code."""
    compiler = MinioCompiler()
    return compiler.compile(program)
