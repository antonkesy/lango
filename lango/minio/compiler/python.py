from typing import Any, Dict, List, Optional, Set

from lango.minio.ast.nodes import (
    AddOperation,
    AndOperation,
    BoolLiteral,
    ConcatOperation,
    ConsPattern,
    Constructor,
    ConstructorExpression,
    ConstructorPattern,
    DataConstructor,
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
    Program,
    StringLiteral,
    SubOperation,
    Variable,
    VariablePattern,
)


class MinioCompiler:
    def __init__(self) -> None:
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
        match pattern:
            case VariablePattern(name=name):
                variables.add(name)
            case ConstructorPattern(patterns=patterns):
                for sub_pattern in patterns:
                    variables.update(self._extract_pattern_variables(sub_pattern))
            case ConsPattern(head=head, tail=tail):
                variables.update(self._extract_pattern_variables(head))
                variables.update(self._extract_pattern_variables(tail))
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
        lines = [
            "# Generated Python code from Minio",
            "from typing import Any, Dict, List, Union, Optional",
            "",
            "# Runtime support functions",
            "def minio_show(value):",
            "    match value:",
            "        case bool():",
            "            return 'True' if value else 'False'",
            "        case str():",
            "            return f'\"{value}\"'",
            "        case list():",
            "            elements = [minio_show(x) for x in value]",
            "            return '[' + ','.join(elements) + ']'",
            "        case _:",
            "            return str(value)",
            "",
            "def minio_put_str(s):",
            "    match s:",
            "        case str():",
            '            s = s.encode().decode("unicode_escape")',
            "    print(s, end='')",
            "",
            "",
        ]

        # Collect data types
        for stmt in program.statements:
            match stmt:
                case DataDeclaration(type_name=type_name):
                    self.data_types[type_name] = stmt
                case _:
                    pass

        # Group function definitions by name
        function_definitions: Dict[str, List[FunctionDefinition]] = {}

        for stmt in program.statements:
            match stmt:
                case DataDeclaration():
                    lines.append(self._compile_data_declaration(stmt))
                case FunctionDefinition(function_name=function_name):
                    if function_name not in function_definitions:
                        function_definitions[function_name] = []
                    function_definitions[function_name].append(stmt)
                case LetStatement(variable=variable, value=value):
                    prefixed_var = self._prefix_name(variable)
                    lines.append(
                        f"{prefixed_var} = {self._compile_expression(value)}",
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

        # Find maximum number of parameters needed
        max_params = (
            max(len(defn.patterns) for defn in definitions) if definitions else 0
        )

        # Add currying support for multi-parameter functions
        if max_params > 1:
            lines.append(self._indent() + f"if len(args) < {max_params}:")
            self.indent_level += 1
            lines.append(
                self._indent()
                + "return lambda *more_args: "
                + prefixed_func_name
                + "(*(args + more_args))",
            )
            self.indent_level -= 1

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
                    match pattern:
                        case VariablePattern(name=name):
                            assignments.append(f"{name} = args[{j}]")
                        case LiteralPattern(value=value):
                            pattern_matches.append(
                                f"args[{j}] == {self._compile_literal_value(value)}",
                            )
                        case ConsPattern(head=head, tail=tail):
                            # Cons pattern (x:xs) - check if list is non-empty and destructure
                            pattern_matches.append(f"len(args[{j}]) > 0")
                            match head:
                                case VariablePattern(name=name):
                                    assignments.append(f"{name} = args[{j}][0]")
                                case _:
                                    pass
                            match tail:
                                case VariablePattern(name=name):
                                    assignments.append(f"{name} = args[{j}][1:]")
                                case _:
                                    pass
                        case ConstructorPattern(
                            constructor=constructor,
                            patterns=sub_patterns,
                        ):
                            # Constructor pattern - check type and destructure
                            pattern_matches.append(
                                f"type(args[{j}]).__name__ == '{constructor}'",
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
                                                f"{name} = args[{j}].fields['{field_name}']",
                                            )
                                        case _:
                                            pass
                            else:
                                # Positional constructor - use arg_ access
                                for k, sub_pattern in enumerate(sub_patterns):
                                    match sub_pattern:
                                        case VariablePattern(name=name):
                                            assignments.append(
                                                f"{name} = args[{j}].arg_{k}",
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
            match func_def.body:
                case DoBlock(statements=statements) if len(statements) > 1:
                    lines = [f"def {func_name}():"]
                    lines.extend(self._compile_do_block_as_statements(func_def.body))
                case _:
                    lines = [
                        f"def {func_name}():",
                        f"    return {self._compile_expression(func_def.body)}",
                    ]
        else:
            pattern = func_def.patterns[0]
            match pattern:
                case VariablePattern(name=name):
                    # Check if the body is a do block with multiple statements
                    match func_def.body:
                        case DoBlock(statements=statements) if len(statements) > 1:
                            lines = [f"def {func_name}({name}):"]
                            lines.extend(
                                self._compile_do_block_as_statements(func_def.body),
                            )
                        case _:
                            lines = [
                                f"def {func_name}({name}):",
                                f"    return {self._compile_expression(func_def.body)}",
                            ]
                case _:
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
            case BoolLiteral(value=value):
                return str(value)
            case ListLiteral(elements=elements):
                compiled_elements = [
                    self._compile_expression(elem) for elem in elements
                ]
                return f"[{', '.join(compiled_elements)}]"

            # Variables and constructors
            case Variable(name="show"):
                return "minio_show"
            case Variable(name="putStr"):
                return "minio_put_str"
            case Variable(name=name):
                prefixed_name = self._prefix_name(name)
                if name in self.nullary_functions:
                    return f"{prefixed_name}()"
                else:
                    return prefixed_name
            case Constructor(name=name):
                return name

            # Binary operations
            case AddOperation(left=left, right=right):
                return f"({self._compile_expression(left)} + {self._compile_expression(right)})"
            case SubOperation(left=left, right=right):
                return f"({self._compile_expression(left)} - {self._compile_expression(right)})"
            case MulOperation(left=left, right=right):
                return f"({self._compile_expression(left)} * {self._compile_expression(right)})"
            case DivOperation(left=left, right=right):
                return f"({self._compile_expression(left)} / {self._compile_expression(right)})"
            case EqualOperation(left=left, right=right):
                return f"({self._compile_expression(left)} == {self._compile_expression(right)})"
            case NotEqualOperation(left=left, right=right):
                return f"({self._compile_expression(left)} != {self._compile_expression(right)})"
            case LessThanOperation(left=left, right=right):
                return f"({self._compile_expression(left)} < {self._compile_expression(right)})"
            case LessEqualOperation(left=left, right=right):
                return f"({self._compile_expression(left)} <= {self._compile_expression(right)})"
            case GreaterThanOperation(left=left, right=right):
                return f"({self._compile_expression(left)} > {self._compile_expression(right)})"
            case GreaterEqualOperation(left=left, right=right):
                return f"({self._compile_expression(left)} >= {self._compile_expression(right)})"
            case ConcatOperation(left=left, right=right):
                return f"({self._compile_expression(left)} + {self._compile_expression(right)})"
            case AndOperation(left=left, right=right):
                return f"({self._compile_expression(left)} and {self._compile_expression(right)})"
            case OrOperation(left=left, right=right):
                return f"({self._compile_expression(left)} or {self._compile_expression(right)})"

            # Unary operations
            case NotOperation(operand=operand):
                return f"(not {self._compile_expression(operand)})"

            # Other operations
            case IndexOperation(list_expr=list_expr, index_expr=index_expr):
                return f"({self._compile_expression(list_expr)}[{self._compile_expression(index_expr)}])"
            case IfElse(condition=condition, then_expr=then_expr, else_expr=else_expr):
                return f"({self._compile_expression(then_expr)} if {self._compile_expression(condition)} else {self._compile_expression(else_expr)})"

            # Function application
            case FunctionApplication():
                # Check if this is a constructor application
                args: List[Expression] = []
                current: Expression = expr

                # Collect all arguments for potential constructor calls
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
                    case _:
                        # Regular curried function application
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
                | BoolLiteral()
                | ListLiteral()
                | Variable()
                | Constructor()
                | AddOperation()
                | SubOperation()
                | MulOperation()
                | DivOperation()
                | EqualOperation()
                | ConcatOperation()
                | IfElse()
                | FunctionApplication()
                | ConstructorExpression()
                | DoBlock()
                | GroupedExpression()
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
    """Compile a Minio program to Python code."""
    compiler = MinioCompiler()
    return compiler.compile(program)
