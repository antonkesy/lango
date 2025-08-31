from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from lango.systemo.ast.nodes import (
    SymbolicOperation,
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
    TupleLiteral,
    TuplePattern,
    Variable,
    VariablePattern,
    is_expression,
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
    Type,
    TypeApp,
    TypeCon,
    TypeVar,
)


@dataclass
class FunctionInfo:
    arity: int
    type_info: Optional[Type] = None


class systemoCompiler:
    def __init__(self) -> None:
        self.indent_level = 0
        self.functions: Dict[str, FunctionInfo] = (
            {}
        )  # Consolidated function information
        self.data_types: Dict[str, DataDeclaration] = {}
        self.local_variables: Set[str] = set()  # Track local pattern variables

        # Initialize built-in function information
        builtin_functions = {
            "primIntAdd": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Int"),
                    result=FunctionType(param=TypeCon("Int"), result=TypeCon("Int")),
                ),
            ),
            "primIntSub": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Int"),
                    result=FunctionType(param=TypeCon("Int"), result=TypeCon("Int")),
                ),
            ),
            "primIntMul": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Int"),
                    result=FunctionType(param=TypeCon("Int"), result=TypeCon("Int")),
                ),
            ),
            "primIntDiv": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Int"),
                    result=FunctionType(param=TypeCon("Int"), result=TypeCon("Int")),
                ),
            ),
            "primIntPow": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Int"),
                    result=FunctionType(param=TypeCon("Int"), result=TypeCon("Int")),
                ),
            ),
            "primIntNeg": FunctionInfo(
                arity=1,
                type_info=FunctionType(param=TypeCon("Int"), result=TypeCon("Int")),
            ),
            "primIntLt": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Int"),
                    result=FunctionType(param=TypeCon("Int"), result=TypeCon("Bool")),
                ),
            ),
            "primIntLe": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Int"),
                    result=FunctionType(param=TypeCon("Int"), result=TypeCon("Bool")),
                ),
            ),
            "primIntGt": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Int"),
                    result=FunctionType(param=TypeCon("Int"), result=TypeCon("Bool")),
                ),
            ),
            "primIntGe": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Int"),
                    result=FunctionType(param=TypeCon("Int"), result=TypeCon("Bool")),
                ),
            ),
            "primIntEq": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Int"),
                    result=FunctionType(param=TypeCon("Int"), result=TypeCon("Bool")),
                ),
            ),
            "primIntShow": FunctionInfo(
                arity=1,
                type_info=FunctionType(param=TypeCon("Int"), result=TypeCon("String")),
            ),
            "primFloatAdd": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Float"),
                    result=FunctionType(
                        param=TypeCon("Float"), result=TypeCon("Float")
                    ),
                ),
            ),
            "primFloatSub": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Float"),
                    result=FunctionType(
                        param=TypeCon("Float"), result=TypeCon("Float")
                    ),
                ),
            ),
            "primFloatMul": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Float"),
                    result=FunctionType(
                        param=TypeCon("Float"), result=TypeCon("Float")
                    ),
                ),
            ),
            "primFloatDiv": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Float"),
                    result=FunctionType(
                        param=TypeCon("Float"), result=TypeCon("Float")
                    ),
                ),
            ),
            "primFloatPow": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Float"),
                    result=FunctionType(
                        param=TypeCon("Float"), result=TypeCon("Float")
                    ),
                ),
            ),
            "primFloatNeg": FunctionInfo(
                arity=1,
                type_info=FunctionType(param=TypeCon("Float"), result=TypeCon("Float")),
            ),
            "primFloatLt": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Float"),
                    result=FunctionType(param=TypeCon("Float"), result=TypeCon("Bool")),
                ),
            ),
            "primFloatLe": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Float"),
                    result=FunctionType(param=TypeCon("Float"), result=TypeCon("Bool")),
                ),
            ),
            "primFloatGt": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Float"),
                    result=FunctionType(param=TypeCon("Float"), result=TypeCon("Bool")),
                ),
            ),
            "primFloatGe": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Float"),
                    result=FunctionType(param=TypeCon("Float"), result=TypeCon("Bool")),
                ),
            ),
            "primFloatEq": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Float"),
                    result=FunctionType(param=TypeCon("Float"), result=TypeCon("Bool")),
                ),
            ),
            "primFloatShow": FunctionInfo(
                arity=1,
                type_info=FunctionType(
                    param=TypeCon("Float"), result=TypeCon("String")
                ),
            ),
            "primStringConcat": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("String"),
                    result=FunctionType(
                        param=TypeCon("String"), result=TypeCon("String")
                    ),
                ),
            ),
            "primStringEq": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("String"),
                    result=FunctionType(
                        param=TypeCon("String"), result=TypeCon("Bool")
                    ),
                ),
            ),
            "primStringShow": FunctionInfo(
                arity=1,
                type_info=FunctionType(
                    param=TypeCon("String"), result=TypeCon("String")
                ),
            ),
            "primCharEq": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Char"),
                    result=FunctionType(param=TypeCon("Char"), result=TypeCon("Bool")),
                ),
            ),
            "primCharShow": FunctionInfo(
                arity=1,
                type_info=FunctionType(param=TypeCon("Char"), result=TypeCon("String")),
            ),
            "primBoolAnd": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Bool"),
                    result=FunctionType(param=TypeCon("Bool"), result=TypeCon("Bool")),
                ),
            ),
            "primBoolOr": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Bool"),
                    result=FunctionType(param=TypeCon("Bool"), result=TypeCon("Bool")),
                ),
            ),
            "primBoolNot": FunctionInfo(
                arity=1,
                type_info=FunctionType(param=TypeCon("Bool"), result=TypeCon("Bool")),
            ),
            "primBoolEq": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeCon("Bool"),
                    result=FunctionType(param=TypeCon("Bool"), result=TypeCon("Bool")),
                ),
            ),
            "primBoolShow": FunctionInfo(
                arity=1,
                type_info=FunctionType(param=TypeCon("Bool"), result=TypeCon("String")),
            ),
            "putStr": FunctionInfo(
                arity=1,
                type_info=FunctionType(
                    param=TypeCon("String"),
                    result=TypeApp(constructor=TypeCon("IO"), argument=TypeCon("()")),
                ),
            ),
            "primListConcat": FunctionInfo(
                arity=2,
                type_info=FunctionType(
                    param=TypeApp(constructor=TypeCon("List"), argument=TypeVar("a")),
                    result=FunctionType(
                        param=TypeApp(
                            constructor=TypeCon("List"), argument=TypeVar("a")
                        ),
                        result=TypeApp(
                            constructor=TypeCon("List"), argument=TypeVar("a")
                        ),
                    ),
                ),
            ),
            "primListShow": FunctionInfo(
                arity=1,
                type_info=FunctionType(
                    param=TypeApp(constructor=TypeCon("List"), argument=TypeVar("a")),
                    result=TypeCon("String"),
                ),
            ),
            "error": FunctionInfo(
                arity=1,
                type_info=FunctionType(param=TypeCon("String"), result=TypeVar("a")),
            ),
        }
        self.functions.update(builtin_functions)

    def _indent(self) -> str:
        return "    " * self.indent_level

    def _prefix_name(self, name: str) -> str:
        """Add systemo_ prefix to user-defined names, but not built-ins, constructors, or pattern variables."""
        # Don't prefix local pattern variables
        if name in self.local_variables:
            return name
        # Don't prefix constructor names (they should be detected by their usage)
        # Add systemo_ prefix to all other names
        name = self._sanitize_operator_name(name)
        return f"systemo_{name}"

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
            lines.append(self._compile_function_group(func_name, definitions))

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

    def _compile_function_group(
        self,
        func_name: str,
        definitions: List[FunctionDefinition],
    ) -> str:
        """Compile function definitions with pattern matching."""
        prefixed_func_name = self._prefix_name(func_name)

        # Track function arity (number of parameters)
        max_params = (
            max(len(defn.patterns) for defn in definitions) if definitions else 0
        )

        # Store function information
        function_type = definitions[0].ty if definitions and definitions[0].ty else None
        self.functions[func_name] = FunctionInfo(
            arity=max_params,
            type_info=function_type,
        )

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

        # Create parameter list with type hints for multi-parameter functions
        param_list = []
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
        else:
            # For single parameter functions, try to get the parameter type
            param_type = "Any"
            if (
                definitions
                and definitions[0].ty
                and isinstance(definitions[0].ty, FunctionType)
            ):
                param_type = self._systemo_type_to_python_hint(definitions[0].ty.param)
            param_list = [f"arg_0: {param_type}"]

        lines = [
            f"def {prefixed_func_name}({', '.join(param_list)}) -> {return_type_hint}:",
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
            # Show all arguments in error message
            arg_names = ", ".join([f"arg_{i}" for i in range(max_params)])
            if max_params == 0:
                error_msg = (
                    f"raise ValueError(f'No matching pattern for {prefixed_func_name}')"
                )
            elif max_params == 1:
                error_msg = f"raise ValueError(f'No matching pattern for {prefixed_func_name} with args: {{arg_0}}')"
            else:
                error_msg = f"raise ValueError(f'No matching pattern for {prefixed_func_name} with args: {{{arg_names}}}')"

            lines.append(self._indent() + error_msg)
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
                                    0,
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
            case Variable(name=name):
                prefixed_name = self._prefix_name(name)
                # Check if this is a nullary function (0 parameters) and automatically call it
                if name in self.functions and self.functions[name].arity == 0:
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

            case SymbolicOperation(operator=operator, operands=operands):
                return self._compile_symbolic_operation(operator, operands)

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
                        # Check if this is a user-defined function
                        prefixed_name = self._prefix_name(name)

                        # Get the expected arity for this function
                        expected_arity = self.functions.get(
                            name,
                            FunctionInfo(arity=1),
                        ).arity

                        if len(args) == expected_arity:
                            # Full application - call function directly
                            arg_exprs = [self._compile_expression(arg) for arg in args]
                            return f"{prefixed_name}({', '.join(arg_exprs)})"
                        elif len(args) < expected_arity and name in self.functions:
                            # Partial application - create lambda for remaining args
                            arg_exprs = [self._compile_expression(arg) for arg in args]
                            remaining_params = expected_arity - len(args)
                            lambda_params = [
                                f"__arg_{i}" for i in range(remaining_params)
                            ]
                            all_args = arg_exprs + lambda_params
                            return f"lambda {', '.join(lambda_params)}: {prefixed_name}({', '.join(all_args)})"
                        elif len(args) > expected_arity:
                            # Over-application - call function with exact args, then apply rest
                            exact_args = args[:expected_arity]
                            remaining_args = args[expected_arity:]

                            arg_exprs = [
                                self._compile_expression(arg) for arg in exact_args
                            ]
                            result = f"{prefixed_name}({', '.join(arg_exprs)})"

                            # Apply remaining arguments one by one
                            for arg in remaining_args:
                                arg_expr = self._compile_expression(arg)
                                result = f"{result}({arg_expr})"
                            return result
                        else:
                            # Single argument or built-in function - use nested calls
                            func_expr = self._compile_expression(current)
                            if len(args) == 1:
                                arg_expr = self._compile_expression(args[0])
                                return f"{func_expr}({arg_expr})"
                            else:
                                # Multiple args - use nested calls
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

            # Default case
            case _:
                return f"None  # Unsupported: {type(expr)}"

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


def compile_program(program: Program) -> str:
    """Compile a systemo program to Python code."""
    compiler = systemoCompiler()
    return compiler.compile(program)
