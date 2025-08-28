from collections import Counter
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
    NegOperation,
    NotEqualOperation,
    NotOperation,
    OrOperation,
    Pattern,
    PowFloatOperation,
    PowIntOperation,
    Program,
    StringLiteral,
    SubOperation,
    Variable,
    VariablePattern,
)
from lango.minio.typechecker.minio_types import (
    DataType,
    FunctionType,
    Type,
    TypeApp,
    TypeCon,
    TypeVar,
)


class MinioGoCompiler:
    def __init__(self) -> None:
        self.indent_level = 0
        self.defined_functions: Set[str] = set()
        self.nullary_functions: Set[str] = set()  # Functions with no parameters
        self.nullary_constructors: Set[str] = set()  # Constructors with no parameters
        self.function_types: Dict[str, Type] = {}  # Track function types
        self.data_types: Dict[str, DataDeclaration] = {}
        self.local_variables: Set[str] = set()  # Track local pattern variables
        self.do_block_counter = 0  # Counter for unique do block function names

    def _indent(self) -> str:
        return "\t" * self.indent_level

    def _get_type_assertion_for_pattern(
        self,
        pattern: Pattern,
        func_type: Optional[Type],
    ) -> Optional[str]:
        """Generate appropriate type assertion for a pattern in function registry."""
        match pattern:
            case VariablePattern():
                # For variable patterns, we need to determine the expected type
                if func_type and isinstance(func_type, FunctionType):
                    param_type = func_type.param
                    return self._get_type_assertion_for_type(param_type, "arg")
            case ConstructorPattern(constructor=constructor):
                # For constructor patterns, assert to the appropriate interface type
                constructor_def = self._find_constructor_def(constructor)
                if constructor_def:
                    # Find which data type this constructor belongs to
                    for data_name, data_decl in self.data_types.items():
                        if constructor in [c.name for c in data_decl.constructors]:
                            return f"arg.({data_name}Interface)"
            case LiteralPattern():
                # For literal patterns, we can infer the type from the literal
                # This is less common in function parameters but handle basic cases
                return "arg"
            case _:
                return None
        return None

    def _get_type_assertion_for_type(self, minio_type: Type, var_name: str) -> str:
        """Generate Go type assertion for a Minio type."""
        match minio_type:
            case TypeCon(name="Int"):
                return f"{var_name}.(int)"
            case TypeCon(name="Float"):
                return f"{var_name}.(float64)"
            case TypeCon(name="String"):
                return f"{var_name}.(string)"
            case TypeCon(name="Bool"):
                return f"{var_name}.(bool)"
            case TypeApp(constructor=TypeCon(name="List")):
                return f"{var_name}.([]any)"
            case DataType(name=name):
                return f"{var_name}.({name}Interface)"
            case TypeVar():
                # Type variables remain as any
                return var_name
            case _:
                return var_name

    def _prefix_name(self, name: str) -> str:
        """Add Minio prefix to user-defined names, but not built-ins, constructors, or pattern variables."""
        # Don't prefix built-in functions
        if name in ["show", "putStr", "error"]:
            return name
        # Don't prefix local pattern variables
        if name in self.local_variables:
            return name
        # Add Minio prefix to all other names
        return f"Minio{name.capitalize()}"

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

    def _find_referenced_variables(self, expr: Any) -> Set[str]:
        """Find all variable names referenced in an expression."""
        referenced = set()

        match expr:
            case Variable(name=name):
                referenced.add(name)
            case FunctionApplication(function=function, argument=argument):
                referenced.update(self._find_referenced_variables(function))
                referenced.update(self._find_referenced_variables(argument))
            case IfElse(condition=condition, then_expr=then_expr, else_expr=else_expr):
                referenced.update(self._find_referenced_variables(condition))
                referenced.update(self._find_referenced_variables(then_expr))
                if else_expr:
                    referenced.update(self._find_referenced_variables(else_expr))
            case IndexOperation(list_expr=list_expr, index_expr=index_expr):
                referenced.update(self._find_referenced_variables(list_expr))
                referenced.update(self._find_referenced_variables(index_expr))
            case (
                AddOperation(left=left, right=right)
                | SubOperation(left=left, right=right)
                | MulOperation(left=left, right=right)
                | DivOperation(left=left, right=right)
                | AndOperation(left=left, right=right)
                | OrOperation(left=left, right=right)
                | EqualOperation(left=left, right=right)
                | NotEqualOperation(left=left, right=right)
                | GreaterThanOperation(left=left, right=right)
                | GreaterEqualOperation(left=left, right=right)
                | LessThanOperation(left=left, right=right)
                | LessEqualOperation(left=left, right=right)
                | ConcatOperation(left=left, right=right)
            ):
                referenced.update(self._find_referenced_variables(left))
                referenced.update(self._find_referenced_variables(right))
            case NotOperation(operand=operand):
                referenced.update(self._find_referenced_variables(operand))
            case NegOperation(operand=operand):
                referenced.update(self._find_referenced_variables(operand))
            case (
                ConstructorExpression()
                | IntLiteral()
                | FloatLiteral()
                | StringLiteral()
                | BoolLiteral()
                | NegativeInt()
                | NegativeFloat()
            ):
                # These don't reference variables
                pass
            case ListLiteral(elements=elements):
                # List literals can contain expressions that reference variables
                for element in elements:
                    referenced.update(self._find_referenced_variables(element))
            case DoBlock(statements=statements):
                for stmt in statements:
                    referenced.update(self._find_referenced_variables(stmt))
            case GroupedExpression(expression=inner):
                referenced.update(self._find_referenced_variables(inner))
            case _:
                # For other expression types, conservatively assume no references
                pass

        return referenced

    def _minio_type_to_go_type(self, minio_type: Optional[Type]) -> str:
        """Convert a Minio type to a Go type string."""
        if minio_type is None:
            return "any"

        match minio_type:
            case TypeCon(name="Int"):
                return "int"
            case TypeCon(name="String"):
                return "string"
            case TypeCon(name="Float"):
                return "float64"
            case TypeCon(name="Bool"):
                return "bool"
            case TypeCon(name="()"):
                return "any"
            case TypeApp(constructor=TypeCon(name="List"), argument=arg_type):
                # For now, use []any for all lists to simplify
                return "[]any"
            case TypeApp(constructor=TypeCon(name="IO"), argument=arg_type):
                # IO types typically don't have meaningful return types in our compiled Go
                return "any"
            case FunctionType(param=param_type, result=result_type):
                # For function types, we'll use func signatures
                param_type_str = self._minio_type_to_go_type(param_type)
                result_type_str = self._minio_type_to_go_type(result_type)
                return f"func({param_type_str}) {result_type_str}"
            case DataType(name=name, type_args=type_args):
                # Custom data types - use interface type
                return f"{name}Interface"
            case TypeVar(name=name):
                # Type variables become any for now
                return "any"
            case _:
                return "any"

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

    def compile(self, program: Program) -> str:
        lines = []
        with open("lango/minio/compiler/prelude.go", "r") as f:
            lines.extend(f.read().splitlines())

        # Collect data types
        for stmt in program.statements:
            match stmt:
                case DataDeclaration(type_name=type_name):
                    self.data_types[type_name] = stmt
                case _:
                    pass

        # Generate data type declarations
        for stmt in program.statements:
            match stmt:
                case DataDeclaration():
                    lines.append(self._compile_data_declaration(stmt))
                case _:
                    pass

        # Group function definitions by name
        function_definitions: Dict[str, List[FunctionDefinition]] = {}

        for stmt in program.statements:
            match stmt:
                case FunctionDefinition(function_name=function_name):
                    if function_name not in function_definitions:
                        function_definitions[function_name] = []
                    function_definitions[function_name].append(stmt)
                case LetStatement(variable=variable, value=value):
                    prefixed_var = self._prefix_name(variable)

                    # Special handling for function applications that might be partial applications
                    match value:
                        case FunctionApplication(
                            function=Variable(name=func_name),
                            argument=arg,
                        ):
                            # This might be a partial application - generate a function instead of a variable
                            if func_name in function_definitions and func_name not in [
                                "show",
                                "putStr",
                                "error",
                            ]:
                                # Get the arity of the target function
                                target_func_defs = function_definitions[func_name]
                                if target_func_defs:
                                    target_arity = (
                                        len(target_func_defs[0].patterns)
                                        if target_func_defs[0].patterns
                                        else 0
                                    )
                                    if (
                                        target_arity > 1
                                    ):  # Multi-argument function, create partial application
                                        prefixed_func = self._prefix_name(func_name)
                                        arg_expr = self._compile_expression(arg)
                                        lines.append(
                                            f"func {prefixed_var}() interface{{}} {{",
                                        )
                                        lines.append(
                                            f"\treturn {prefixed_func}({arg_expr})",
                                        )
                                        lines.append("}")
                                        continue

                            # Not a partial application, treat as normal variable assignment
                            go_type = "any"
                            lines.append(
                                f"var {prefixed_var} {go_type} = {self._compile_expression(value)}",
                            )
                        case _:
                            # Normal let statement
                            go_type = "any"  # Default type
                            lines.append(
                                f"var {prefixed_var} {go_type} = {self._compile_expression(value)}",
                            )
                case _:
                    pass

        # Generate function definitions
        for func_name, definitions in function_definitions.items():
            # Skip built-in functions to avoid conflicts
            if func_name not in ["show", "putStr", "error"]:
                lines.append(self._compile_function_group(func_name, definitions))

        # Generate function registry initialization
        lines.append("")
        lines.append("func init() {")
        lines.append("\t// Populate function registry for first-class function support")
        for func_name in function_definitions:
            if func_name not in ["show", "putStr", "error", "main"]:
                prefixed_name = self._prefix_name(func_name)
                # Check if this is a nullary function
                if func_name in self.nullary_functions:
                    # Nullary functions need to be wrapped
                    lines.append(
                        f'\tminioFunctionRegistry["{func_name}"] = func(arg any) any {{',
                    )
                    lines.append(f"\t\t// Nullary function - ignore the argument")
                    lines.append(f"\t\treturn {prefixed_name}()")
                    lines.append("\t}")
                else:
                    # Regular functions - need proper type assertion
                    lines.append(
                        f'\tminioFunctionRegistry["{func_name}"] = func(arg any) any {{',
                    )

                    # Get function type information for proper type casting
                    func_defs = function_definitions[func_name]
                    if func_defs and len(func_defs[0].patterns) > 0:
                        first_pattern = func_defs[0].patterns[0]
                        type_assertion = self._get_type_assertion_for_pattern(
                            first_pattern,
                            func_defs[0].ty,
                        )
                        if type_assertion:
                            lines.append(
                                f"\t\treturn {prefixed_name}({type_assertion})",
                            )
                        else:
                            lines.append(f"\t\treturn {prefixed_name}(arg)")
                    else:
                        lines.append(f"\t\treturn {prefixed_name}(arg)")
                    lines.append("\t}")
        lines.append("}")

        # Add main function
        if "main" in function_definitions:
            lines.extend(["", "func main() {", "\tMinioMain()", "}"])

        return "\n".join(lines)

    def _compile_data_declaration(self, data_decl: DataDeclaration) -> str:
        lines = [f"// Data type: {data_decl.type_name}"]

        # Create interface for the data type
        interface_name = f"{data_decl.type_name}Interface"
        lines.append(f"type {interface_name} interface {{")
        lines.append(f"\tIs{data_decl.type_name}() bool")
        lines.append("}")
        lines.append("")

        for constructor in data_decl.constructors:
            class_name = constructor.name

            # Track nullary constructors
            is_nullary = not constructor.record_constructor and (
                not constructor.type_atoms or len(constructor.type_atoms) == 0
            )
            if is_nullary:
                self.nullary_constructors.add(class_name)

            # Generate struct for constructor
            if constructor.record_constructor:
                # Named fields like Person { id_ :: Int, name :: String }
                lines.append(f"type {class_name} struct {{")
                for field in constructor.record_constructor.fields:
                    field_type = "any"  # Default type
                    if field.field_type and hasattr(field.field_type, "ty"):
                        field_type = self._minio_type_to_go_type(field.field_type.ty)
                    lines.append(f"\t{field.name.capitalize()} {field_type}")
                lines.append("}")
            else:
                # Positional arguments like MkPoint Float Float
                if constructor.type_atoms:
                    lines.append(f"type {class_name} struct {{")
                    for i, type_atom in enumerate(constructor.type_atoms):
                        field_type = "any"
                        if hasattr(type_atom, "ty"):
                            field_type = self._minio_type_to_go_type(type_atom.ty)
                        lines.append(f"\tArg{i} {field_type}")
                    lines.append("}")
                else:
                    # No arguments - simple constructor
                    lines.append(f"type {class_name} struct {{}}")

            # Implement the interface - fix syntax
            receiver_name = class_name[0].lower()
            lines.append(
                f"func ({receiver_name} {class_name}) Is{data_decl.type_name}() bool {{ return true }}",
            )
            lines.append("")

        lines.append("")
        return "\n".join(lines)

    def _compile_function_group(
        self,
        func_name: str,
        definitions: List[FunctionDefinition],
    ) -> str:
        """Compile function definitions with pattern matching."""
        prefixed_func_name = self._prefix_name(func_name)
        self.defined_functions.add(func_name)

        # Store function type information
        if definitions and definitions[0].ty:
            self.function_types[func_name] = definitions[0].ty

        # Check if any definition is nullary
        if any(len(defn.patterns) == 0 for defn in definitions):
            self.nullary_functions.add(func_name)

        # Simple case: single function with at most one parameter
        if len(definitions) == 1 and len(definitions[0].patterns) <= 1:
            return self._compile_simple_function(definitions[0], prefixed_func_name)

        # Complex case: multiple patterns or multiple definitions
        lines = [f"func {prefixed_func_name}(args ...any) any {{"]
        self.indent_level += 1

        # Collect all pattern variables from all definitions
        old_local_vars = self.local_variables.copy()
        for func_def in definitions:
            for pattern in func_def.patterns:
                self.local_variables.update(self._extract_pattern_variables(pattern))

        # For functions with multiple definitions, we need to ensure variables are accessible
        # across all branches. We'll pre-declare variables but handle name conflicts by
        # using the most common variable name for each argument position
        arg_var_names: Dict[int, List[str]] = {}  # arg_index -> list of var_names
        for func_def in definitions:
            for j, pattern in enumerate(func_def.patterns):
                if isinstance(pattern, VariablePattern):
                    if j not in arg_var_names:
                        arg_var_names[j] = []
                    arg_var_names[j].append(pattern.name)

        # Choose the most common name for each argument, or the first one if tied
        final_arg_vars = {}  # arg_index -> var_name
        for arg_index, var_names in arg_var_names.items():
            # Count occurrences
            counter = Counter(var_names)
            most_common = counter.most_common(1)[0][0]
            final_arg_vars[arg_index] = most_common

        # Check which variables are actually used in function bodies to avoid unused variable warnings
        used_vars = set()
        for func_def in definitions:
            # Simple heuristic: check if variable names appear in the stringified body
            body_str = str(func_def.body)
            for var_name in final_arg_vars.values():
                if var_name in body_str:
                    used_vars.add(var_name)

        # Only declare variables that are actually used
        used_final_arg_vars = {
            arg_index: var_name
            for arg_index, var_name in final_arg_vars.items()
            if var_name in used_vars
        }

        # Declare argument variables at the top of the function
        if used_final_arg_vars:
            lines.append(f"{self._indent()}// Declare argument variables")
            for arg_index in sorted(used_final_arg_vars.keys()):
                var_name = used_final_arg_vars[arg_index]
                lines.append(f"{self._indent()}var {var_name} any")
                lines.append(
                    f"{self._indent()}if len(args) > {arg_index} {{ {var_name} = args[{arg_index}] }}",
                )
            lines.append("")

        # Generate pattern matching logic - group by arity first
        arity_groups: Dict[int, List[FunctionDefinition]] = {}
        for func_def in definitions:
            arity = len(func_def.patterns)
            if arity not in arity_groups:
                arity_groups[arity] = []
            arity_groups[arity].append(func_def)

        # Calculate maximum arity for partial application support
        max_arity = max(arity_groups.keys()) if arity_groups else 0

        # Generate branches for each arity group
        first_arity = True
        for arity, func_defs in arity_groups.items():
            if first_arity:
                lines.append(f"{self._indent()}if len(args) == {arity} {{")
                first_arity = False
            else:
                lines.append(f"{self._indent()}}} else if len(args) == {arity} {{")

            # Within this arity group, generate pattern matching for each function definition
            first_pattern = True
            for func_def in func_defs:
                condition_parts = []
                assignments = []

                for j, pattern in enumerate(func_def.patterns):
                    condition, assignment = (
                        self._compile_pattern_condition_and_assignment(
                            pattern,
                            f"args[{j}]",
                            f"arg{j}",
                            func_def.body,
                        )
                    )
                    if condition:
                        condition_parts.append(condition)
                    if assignment:
                        if isinstance(pattern, VariablePattern):
                            # Handle variable aliasing if needed
                            if (
                                j in used_final_arg_vars
                                and pattern.name != used_final_arg_vars[j]
                            ):
                                assignments.append(
                                    f"{pattern.name} := {used_final_arg_vars[j]}",
                                )
                        else:
                            assignments.append(assignment)

                # Generate condition check for this specific pattern
                if condition_parts:
                    # Check if any condition contains variable declarations (type assertions)
                    has_type_assertions = any(":=" in cond for cond in condition_parts)

                    if has_type_assertions:
                        # Generate nested if statements for type assertions
                        if first_pattern:
                            lines.append(
                                f"{self._indent()}\tif {condition_parts[0]} {{",
                            )
                        else:
                            lines.append(
                                f"{self._indent()}\t}} else if {condition_parts[0]} {{",
                            )

                        # Add additional nested conditions if needed
                        for cond in condition_parts[1:]:
                            if ":=" in cond:
                                lines.append(f"{self._indent()}\t\tif {cond} {{")
                            else:
                                lines.append(f"{self._indent()}\t\tif {cond} {{")

                        # Calculate correct indentation based on nested conditions
                        indent_level = 2 + len(condition_parts) - 1
                    else:
                        # No type assertions - can use && to combine conditions
                        combined_condition = " && ".join(condition_parts)
                        if first_pattern:
                            lines.append(
                                f"{self._indent()}\tif {combined_condition} {{",
                            )
                        else:
                            lines.append(
                                f"{self._indent()}\t}} else if {combined_condition} {{",
                            )
                        indent_level = 2
                else:
                    # No pattern conditions - this is a variable-only pattern (always matches)
                    if first_pattern:
                        lines.append(f"{self._indent()}\t{{")  # No condition needed
                    else:
                        lines.append(f"{self._indent()}\t}} else {{")  # Fallback case
                    indent_level = 2

                # Add assignments
                for assignment in assignments:
                    lines.append(f"{self._indent()}{'\t' * indent_level}{assignment}")

                # Compile function body
                lines.append(
                    f"{self._indent()}{'\t' * indent_level}return {self._compile_expression(func_def.body)}",
                )

                # Close nested conditions if needed
                if condition_parts and any(":=" in cond for cond in condition_parts):
                    for _ in condition_parts[1:]:  # Close additional nested conditions
                        lines.append(f"{self._indent()}\t\t}}")

                first_pattern = False

            # Close the pattern matching for this function definition group
            if not first_pattern:  # Only if we had at least one pattern
                lines.append(f"{self._indent()}\t}}")

            # Add a fallback return for any unmatched patterns within this arity
            # This should not normally be reached if patterns are exhaustive
            lines.append(
                f'{self._indent()}\tpanic("Pattern match failure in {prefixed_func_name}")',
            )

        # Add fallback for partial application - if fewer arguments provided, return a closure
        if max_arity > 1:  # Only for multi-argument functions
            for partial_arity in range(1, max_arity):
                lines.append(
                    f"{self._indent()}}} else if len(args) == {partial_arity} {{",
                )
                lines.append(
                    f"{self._indent()}\t// Partial application with {partial_arity} argument(s)",
                )
                lines.append(
                    f"{self._indent()}\treturn func(remainingArgs ...any) any {{",
                )
                lines.append(
                    f"{self._indent()}\t\tallArgs := make([]any, {partial_arity}+len(remainingArgs))",
                )
                for i in range(partial_arity):
                    lines.append(f"{self._indent()}\t\tallArgs[{i}] = args[{i}]")
                lines.append(
                    f"{self._indent()}\t\tcopy(allArgs[{partial_arity}:], remainingArgs)",
                )
                lines.append(
                    f"{self._indent()}\t\treturn {prefixed_func_name}(allArgs...)",
                )
                lines.append(f"{self._indent()}\t}}")

        lines.append(f"{self._indent()}}} else {{")
        lines.append(
            f'{self._indent()}\tpanic("No matching pattern for {prefixed_func_name}")',
        )
        lines.append(f"{self._indent()}}}")

        self.indent_level -= 1
        lines.append("}")

        # Restore local variables
        self.local_variables = old_local_vars

        lines.append("")
        return "\n".join(lines)

    def _compile_simple_function(
        self,
        func_def: FunctionDefinition,
        prefixed_name: str,
    ) -> str:
        """Compile a simple function with at most one parameter."""

        # Track pattern variables
        old_local_vars = self.local_variables.copy()
        for pattern in func_def.patterns:
            self.local_variables.update(self._extract_pattern_variables(pattern))

        # Determine return type
        return_type = "any"
        # Try to infer return type from the function body for simple cases
        if len(func_def.patterns) == 0:  # nullary function
            match func_def.body:
                case ConstructorExpression(constructor_name=constructor_name):
                    # If returning a constructor, return the appropriate interface type
                    for data_name, data_decl in self.data_types.items():
                        for constructor in data_decl.constructors:
                            if constructor.name == constructor_name:
                                return_type = f"{data_name}Interface"
                                break
                        if return_type != "any":
                            break
                case Variable(name=name) if name in self.data_types:
                    return_type = f"{name}Interface"

        if len(func_def.patterns) == 0:
            # No parameters
            lines = [f"func {prefixed_name}() {return_type} {{"]
            lines.append(f"\treturn {self._compile_expression(func_def.body)}")
            lines.append("}")
        else:
            # One parameter
            pattern = func_def.patterns[0]
            param_type = "any"

            # Try to get parameter type from function type
            if func_def.ty and isinstance(func_def.ty, FunctionType):
                param_type = self._minio_type_to_go_type(func_def.ty.param)

            match pattern:
                case VariablePattern(name=name):
                    # Simple variable pattern
                    lines = [
                        f"func {prefixed_name}({name} {param_type}) {return_type} {{",
                    ]
                    lines.append(f"\treturn {self._compile_expression(func_def.body)}")
                    lines.append("}")
                case _:
                    # Complex pattern - need pattern matching
                    lines = [f"func {prefixed_name}(arg {param_type}) {return_type} {{"]
                    condition, assignment = (
                        self._compile_pattern_condition_and_assignment(
                            pattern,
                            "arg",
                            "matched",
                            func_def.body,  # Pass function body to determine used variables
                        )
                    )

                    if condition:
                        lines.append(f"\tif {condition} {{")
                        if assignment:
                            lines.append(f"\t\t{assignment}")
                        lines.append(
                            f"\t\treturn {self._compile_expression(func_def.body)}",
                        )
                        lines.append("\t} else {")
                        lines.append('\t\tpanic("Pattern match failed")')
                        lines.append("\t}")
                    else:
                        if assignment:
                            lines.append(f"\t{assignment}")
                        lines.append(
                            f"\treturn {self._compile_expression(func_def.body)}",
                        )

                    lines.append("}")

        lines.append("")

        # Restore local variables
        self.local_variables = old_local_vars

        return "\n".join(lines)

    def _compile_pattern_condition_and_assignment(
        self,
        pattern: Pattern,
        value_expr: str,
        var_prefix: str,
        function_body: Any = None,  # Optional function body to determine used variables
    ) -> tuple[Optional[str], Optional[str]]:
        """Compile a pattern into a condition check and variable assignment."""
        match pattern:
            case VariablePattern(name=name):
                # Variable patterns always match
                return None, f"{name} := {value_expr}"
            case LiteralPattern(value=value):
                # Literal patterns need equality check
                match value:
                    case []:
                        # Empty list comparison
                        return f"len({value_expr}.([]any)) == 0", None
                    case _:
                        go_value = self._compile_literal_value(value)
                        return f"{value_expr} == {go_value}", None
            case ConstructorPattern(constructor=constructor, patterns=patterns):
                # Constructor pattern - type check and field access
                condition = f"_, ok := {value_expr}.({constructor}); ok"
                assignments = []

                if patterns:
                    typed_var = f"{var_prefix}_{constructor}"
                    assignments.append(f"{typed_var} := {value_expr}.({constructor})")

                    # Find constructor definition to determine field access method
                    constructor_def = self._find_constructor_def(constructor)

                    # Determine which variables are actually used in the function body
                    used_variables = set()
                    if function_body:
                        used_variables = self._find_referenced_variables(function_body)

                    for i, sub_pattern in enumerate(patterns):
                        # Skip processing if this is a variable pattern that's not used
                        if isinstance(sub_pattern, VariablePattern) and function_body:
                            if sub_pattern.name not in used_variables:
                                continue  # Skip unused variables

                        # Determine field access based on constructor type
                        if constructor_def and constructor_def.record_constructor:
                            # Record constructor - use field names
                            field_name = constructor_def.record_constructor.fields[
                                i
                            ].name
                            field_expr = f"{typed_var}.{field_name.capitalize()}"
                        else:
                            # Positional constructor - use Arg fields
                            field_expr = f"{typed_var}.Arg{i}"

                        sub_condition, sub_assignment = (
                            self._compile_pattern_condition_and_assignment(
                                sub_pattern,
                                field_expr,
                                f"{var_prefix}_{i}",
                                function_body,
                            )
                        )
                        if sub_condition:
                            condition = f"{condition} && {sub_condition}"
                        if sub_assignment:
                            assignments.append(sub_assignment)

                assignment_str = "; ".join(assignments) if assignments else None
                return condition, assignment_str
            case ConsPattern(head=head, tail=tail):
                # Cons pattern - list with at least one element
                condition = f"len({value_expr}.([]any)) > 0"
                assignments = []

                # Determine which variables are actually used in the function body
                used_variables = set()
                if function_body:
                    used_variables = self._find_referenced_variables(function_body)

                if isinstance(head, VariablePattern):
                    # Check if head variable is actually used
                    if function_body and head.name not in used_variables:
                        # Variable is not used - use blank identifier
                        assignments.append(
                            f"_ = {value_expr}.([]any)[0]",
                        )
                    else:
                        # Variable is used - generate normal assignment
                        assignments.append(
                            f"{head.name} := {value_expr}.([]any)[0]",
                        )
                else:
                    # For non-variable patterns, still need to evaluate the head for pattern matching
                    head_condition, head_assignment = (
                        self._compile_pattern_condition_and_assignment(
                            head,
                            f"{value_expr}.([]any)[0]",
                            f"{var_prefix}_head",
                            function_body,
                        )
                    )
                    if head_condition:
                        condition = f"{condition} && {head_condition}"
                    if head_assignment:
                        assignments.append(head_assignment)

                if isinstance(tail, VariablePattern):
                    # Check if tail variable is actually used
                    if function_body and tail.name not in used_variables:
                        # Variable is not used - use blank identifier
                        assignments.append(
                            f"_ = {value_expr}.([]any)[1:]",
                        )
                    else:
                        # Variable is used - generate normal assignment
                        assignments.append(
                            f"{tail.name} := {value_expr}.([]any)[1:]",
                        )
                else:
                    # For non-variable patterns, still need to evaluate the tail for pattern matching
                    tail_condition, tail_assignment = (
                        self._compile_pattern_condition_and_assignment(
                            tail,
                            f"{value_expr}.([]any)[1:]",
                            f"{var_prefix}_tail",
                            function_body,
                        )
                    )
                    if tail_condition:
                        condition = f"{condition} && {tail_condition}"
                    if tail_assignment:
                        assignments.append(tail_assignment)

                assignment_str = "; ".join(assignments) if assignments else None
                return condition, assignment_str
            case _:
                return None, None

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
                return "true" if value else "false"
            case ListLiteral(elements=elements):
                compiled_elements = [
                    self._compile_expression(elem) for elem in elements
                ]
                return "[]any{" + ", ".join(compiled_elements) + "}"

            # Variables and constructors
            case Variable(name="show"):
                return "minioShow"
            case Variable(name="putStr"):
                return "minioPutStr"
            case Variable(name="error"):
                return "minioError"
            case Variable(name=name):
                prefixed_name = self._prefix_name(name)
                # Only treat as function call if it's in nullary_functions AND not a local variable
                if name in self.nullary_functions and name not in self.local_variables:
                    return f"{prefixed_name}()"
                else:
                    return prefixed_name
            case Constructor(name=name):
                # For nullary constructors, return an instance, not the type
                if name in self.nullary_constructors:
                    return f"{name}{{}}"
                else:
                    return name

            # Binary operations
            case AddOperation(left=left, right=right):
                return f"minioAdd({self._compile_expression(left)}, {self._compile_expression(right)})"
            case SubOperation(left=left, right=right):
                return f"minioSub({self._compile_expression(left)}, {self._compile_expression(right)})"
            case MulOperation(left=left, right=right):
                return f"minioMul({self._compile_expression(left)}, {self._compile_expression(right)})"
            case DivOperation(left=left, right=right):
                return f"minioDiv({self._compile_expression(left)}, {self._compile_expression(right)})"
            case PowIntOperation(left=left, right=right):
                return f"minioPowInt({self._compile_expression(left)}, {self._compile_expression(right)})"
            case PowFloatOperation(left=left, right=right):
                return f"minioPowFloat({self._compile_expression(left)}, {self._compile_expression(right)})"
            case EqualOperation(left=left, right=right):
                return f"minioEqual({self._compile_expression(left)}, {self._compile_expression(right)})"
            case NotEqualOperation(left=left, right=right):
                return f"minioNotEqual({self._compile_expression(left)}, {self._compile_expression(right)})"
            case LessThanOperation(left=left, right=right):
                return f"minioLessThan({self._compile_expression(left)}, {self._compile_expression(right)})"
            case LessEqualOperation(left=left, right=right):
                return f"minioLessEqual({self._compile_expression(left)}, {self._compile_expression(right)})"
            case GreaterThanOperation(left=left, right=right):
                return f"minioGreaterThan({self._compile_expression(left)}, {self._compile_expression(right)})"
            case GreaterEqualOperation(left=left, right=right):
                return f"minioGreaterEqual({self._compile_expression(left)}, {self._compile_expression(right)})"
            case ConcatOperation(left=left, right=right):
                return f"minioConcat({self._compile_expression(left)}, {self._compile_expression(right)})"
            case AndOperation(left=left, right=right):
                return f"({self._compile_expression(left)} && {self._compile_expression(right)})"
            case OrOperation(left=left, right=right):
                return f"({self._compile_expression(left)} || {self._compile_expression(right)})"

            # Unary operations
            case NotOperation(operand=operand):
                return f"(!{self._compile_expression(operand)})"
            case NegOperation(operand=operand):
                return f"minioNeg({self._compile_expression(operand)})"

            # Other operations
            case IndexOperation(list_expr=list_expr, index_expr=index_expr):
                list_compiled = self._compile_expression(list_expr)
                index_compiled = self._compile_expression(index_expr)
                # Only add type assertion if we're not sure it's already a slice
                # If it's a simple variable (starts with Minio and contains only letters/numbers)
                # or already contains type assertion, don't double-assert
                if (
                    ".([]any)" in list_compiled
                    or list_compiled.startswith("[]any")
                    or (
                        list_compiled.startswith("Minio")
                        and list_compiled.replace("Minio", "")
                        .replace("_", "")
                        .isalnum()
                    )
                ):
                    return f"({list_compiled}[{index_compiled}])"
                else:
                    return f"({list_compiled}.([]any)[{index_compiled}])"
            case IfElse(condition=condition, then_expr=then_expr, else_expr=else_expr):
                # Use a more compact ternary-like expression in Go
                cond_expr = self._compile_expression(condition)
                then_val = self._compile_expression(then_expr)
                else_val = self._compile_expression(else_expr)
                return f"(func() any {{ if minioBool({cond_expr}) {{ return {then_val} }} else {{ return {else_val} }} }}())"

            # Function application
            case FunctionApplication(function=function, argument=argument):
                func_expr = self._compile_expression(function)
                arg_expr = self._compile_expression(argument)

                # Helper function to unwrap GroupedExpressions
                def unwrap_grouped(expr: Any) -> Any:
                    while isinstance(expr, GroupedExpression):
                        expr = expr.expression
                    return expr

                # Handle constructor calls specially
                unwrapped_function = unwrap_grouped(function)
                match unwrapped_function:
                    case Constructor(name=name):
                        # For constructors, we need to determine if it's record or positional
                        constructor_def = self._find_constructor_def(name)
                        if constructor_def and constructor_def.record_constructor:
                            # This is a record constructor, but we're applying it positionally
                            # This shouldn't happen in well-typed code, but handle it gracefully
                            return f"{name}{{{arg_expr}}}"
                        else:
                            # Positional constructor
                            return f"{name}{{{arg_expr}}}"
                    case FunctionApplication():
                        # Nested function application - collect all arguments
                        args: List[Expression] = []
                        current: Expression = expr

                        # Collect all arguments from nested function applications
                        while True:
                            match current:
                                case FunctionApplication(
                                    argument=argument,
                                    function=function,
                                ):
                                    args.insert(0, argument)
                                    current = unwrap_grouped(function)
                                case _:
                                    break

                        # Handle the base function
                        match current:
                            case Constructor(name=name):
                                # Multiple arguments to constructor
                                arg_exprs = [
                                    self._compile_expression(arg) for arg in args
                                ]
                                constructor_def = self._find_constructor_def(name)
                                if (
                                    constructor_def
                                    and constructor_def.record_constructor
                                    and len(args)
                                    == len(constructor_def.record_constructor.fields)
                                ):
                                    # Match positional arguments to field names
                                    field_assignments = []
                                    for field, arg_expr in zip(
                                        constructor_def.record_constructor.fields,
                                        arg_exprs,
                                    ):
                                        field_assignments.append(
                                            f"{field.name.capitalize()}: {arg_expr}",
                                        )
                                    return f"{name}{{{', '.join(field_assignments)}}}"
                                else:
                                    # Positional constructor with multiple args
                                    if len(args) == 1:
                                        return f"{name}{{{arg_exprs[0]}}}"
                                    else:
                                        # Multiple args as struct fields
                                        field_assignments = []
                                        for i, arg_expr in enumerate(arg_exprs):
                                            field_assignments.append(
                                                f"Arg{i}: {arg_expr}",
                                            )
                                        return (
                                            f"{name}{{{', '.join(field_assignments)}}}"
                                        )
                            case _:
                                # Regular function call with multiple arguments
                                return f"{self._compile_expression(current)}({', '.join(self._compile_expression(arg) for arg in args)})"
                    case _:
                        # Regular function call
                        func_expr = self._compile_expression(function)
                        arg_expr = self._compile_expression(argument)

                        # Check various cases where we should use minioCall instead of direct call
                        should_use_minio_call = False

                        # Case 1: Function is a local variable (could be any)
                        if (
                            isinstance(function, Variable)
                            and function.name in self.local_variables
                        ):
                            should_use_minio_call = True

                        # Case 2: Function expression contains function calls that might return closures
                        # This handles cases like addOneAndDouble where the function is the result of another function call
                        elif isinstance(function, FunctionApplication):
                            should_use_minio_call = True

                        # Case 3: Function is a nullary function (like addOneAndDouble) that returns a closure
                        elif (
                            isinstance(function, Variable)
                            and function.name in self.nullary_functions
                        ):
                            should_use_minio_call = True

                        if should_use_minio_call:
                            return f"minioCall({func_expr}, {arg_expr})"
                        else:
                            return f"{func_expr}({arg_expr})"

            # Constructor expressions
            case ConstructorExpression(
                constructor_name=constructor_name,
                fields=fields,
            ):
                if fields:
                    field_assignments = []
                    for field_assignment in fields:
                        field_assignments.append(
                            f"{field_assignment.field_name.capitalize()}: {self._compile_expression(field_assignment.value)}",
                        )
                    return f"{constructor_name}{{{', '.join(field_assignments)}}}"
                else:
                    return f"{constructor_name}{{}}"

            # Other expressions
            case DoBlock():
                return self._compile_do_block(expr)
            case GroupedExpression(expression=expression):
                return f"({self._compile_expression(expression)})"

            case _:
                return f"nil  // Unsupported: {type(expr)}"

    def _compile_do_block(self, do_block: DoBlock) -> str:
        """Compile a do block as an immediately invoked function expression."""
        lines = ["func() any {"]

        # Process all statements except the last one
        for stmt in do_block.statements[:-1]:
            match stmt:
                case LetStatement(variable=variable, value=value):
                    prefixed_var = self._prefix_name(variable)
                    lines.append(
                        f"\t{prefixed_var} := {self._compile_expression(value)}",
                    )
                    # Add a blank identifier assignment to avoid "declared and not used" errors
                    lines.append(
                        f"\t_ = {prefixed_var}",
                    )  # This suppresses unused variable warnings
                case _ if self._is_expression(stmt):
                    lines.append(f"\t{self._compile_expression(stmt)}")  # type: ignore

        # Handle the last statement (which becomes the return value)
        last_stmt = do_block.statements[-1]
        match last_stmt:
            case LetStatement(variable=variable, value=value):
                prefixed_var = self._prefix_name(variable)
                lines.append(f"\t{prefixed_var} := {self._compile_expression(value)}")
                lines.append(f"\treturn {prefixed_var}")
            case _ if self._is_expression(last_stmt):
                # Check if it's a statement that doesn't return a value (like putStr)
                expr_str = self._compile_expression(last_stmt)  # type: ignore
                if "minioPutStr(" in expr_str:
                    # This is a print statement - execute it and return nil
                    lines.append(f"\t{expr_str}")
                    lines.append("\treturn nil")
                else:
                    lines.append(f"\treturn {expr_str}")
            case _:
                lines.append("\treturn nil")

        lines.append("}()")
        return "\n".join(lines)

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

    def _compile_literal_value(self, value: Any) -> str:
        match value:
            case str():
                return f'"{value}"'
            case bool():
                return "true" if value else "false"
            case list():
                return "[]any{}"  # Handle empty list explicitly
            case _:
                return str(value)


def compile_program(program: Program) -> str:
    """Compile a Minio program to Go code."""
    compiler = MinioGoCompiler()
    return compiler.compile(program)
