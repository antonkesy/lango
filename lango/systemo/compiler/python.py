from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from lango.systemo.ast.nodes import (
    ArrowType,
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
    TuplePattern,
    TupleType,
    TupleLiteral,
    TypeApplication,
    TypeConstructor,
    TypeVariable,
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
class FunctionOverload:
    """Represents a single overloaded version of a function."""

    arity: int
    type_info: Optional[Type] = None
    monomorphized_name: Optional[str] = None


@dataclass
class FunctionInfo:
    """Represents all overloaded versions of a function."""

    overloads: List[FunctionOverload] = field(default_factory=list)

    def add_overload(
        self, arity: int, type_info: Optional[Type], monomorphized_name: str
    ) -> None:
        """Add a new overloaded version of this function."""
        overload = FunctionOverload(
            arity=arity, type_info=type_info, monomorphized_name=monomorphized_name
        )
        self.overloads.append(overload)

    def find_best_overload(
        self, arg_types: List[Optional[Type]]
    ) -> Optional[FunctionOverload]:
        """Find the best matching overload for the given argument types."""
        # For now, do simple matching based on arity and type compatibility
        for overload in self.overloads:
            if overload.arity == len(arg_types) and self._types_match(
                overload.type_info, arg_types
            ):
                return overload
        # Fallback to first overload with matching arity
        for overload in self.overloads:
            if overload.arity == len(arg_types):
                return overload
        # Fallback to first overload
        return self.overloads[0] if self.overloads else None

    def _types_match(
        self, func_type: Optional[Type], arg_types: List[Optional[Type]]
    ) -> bool:
        """Check if function type matches the argument types."""
        if func_type is None or not arg_types:
            return True

        # Extract parameter types from function type
        param_types = []
        current_type = func_type
        while isinstance(current_type, FunctionType):
            param_types.append(current_type.param)
            current_type = current_type.result

        if len(param_types) != len(arg_types):
            return False

        # Simple type matching (can be enhanced later)
        for param_type, arg_type in zip(param_types, arg_types):
            if arg_type is not None and not self._type_compatible(param_type, arg_type):
                return False

        return True

    def _type_compatible(self, expected: Type, actual: Type) -> bool:
        """Check if actual type is compatible with expected type."""
        # Handle tuple type compatibility
        if isinstance(actual, TypeCon) and actual.name.startswith("Tuple"):
            # Check if expected is a nested tuple TypeApp structure
            if isinstance(expected, TypeApp):
                # Extract the base tuple constructor from nested structure
                base_constructor = self._extract_tuple_base(expected)
                if (isinstance(base_constructor, TypeCon) and 
                    base_constructor.name.startswith("Tuple")):
                    return actual.name == base_constructor.name
            elif isinstance(expected, TypeCon) and expected.name.startswith("Tuple"):
                return actual.name == expected.name
        
        # Simple equality check for non-tuple types
        if isinstance(expected, TypeCon) and isinstance(actual, TypeCon):
            return expected.name == actual.name
        # Add more sophisticated type checking as needed
        return True

    def _extract_tuple_base(self, type_app: Type) -> Optional[Type]:
        """Extract the base tuple constructor from a nested TypeApp structure."""
        current = type_app
        while isinstance(current, TypeApp):
            current = current.constructor
        return current

    @property
    def arity(self) -> int:
        """Get the arity of the first overload (for backward compatibility)."""
        return self.overloads[0].arity if self.overloads else 0


class systemoCompiler:
    def __init__(self) -> None:
        self.indent_level = 0
        self.functions: Dict[str, FunctionInfo] = (
            {}
        )  # Consolidated function information
        self.data_types: Dict[str, DataDeclaration] = {}
        self.local_variables: Set[str] = set()  # Track local pattern variables
        self.variable_types: Dict[str, Type] = {}  # Track variable assignments and their types

        # Initialize built-in function information
        self._initialize_builtin_functions()

    def _create_monomorphized_name(
        self, base_name: str, func_type: Optional[Type]
    ) -> str:
        """Create a monomorphized function name based on the type signature."""
        # Sanitize the base name first
        sanitized_base_name = self._sanitize_operator_name(base_name)

        if func_type is None:
            return f"systemo_{sanitized_base_name}"

        type_suffix = self._type_to_suffix(func_type)
        return f"systemo_{sanitized_base_name}_{type_suffix}"

    def _type_to_suffix(self, func_type: Type) -> str:
        """Convert a type to a suffix for monomorphized names."""

        def type_name(t: Type) -> str:
            match t:
                case TypeCon(name=name):
                    # Handle tuple types by arity
                    if name == "Tuple2":
                        return "tuple2"
                    elif name == "Tuple3":
                        return "tuple3"
                    elif name.startswith("Tuple"):
                        # Extract arity from TupleN
                        arity = name[5:]  # Remove "Tuple" prefix
                        return f"tuple{arity}"
                    else:
                        return name.lower()
                case FunctionType(param=param, result=result):
                    return f"{type_name(param)}_to_{type_name(result)}"
                case TypeApp(constructor=TypeCon(name=name), argument=arg):
                    return f"{name.lower()}_{type_name(arg)}"
                case TypeApp(constructor=TypeApp(constructor=TypeCon(name="Tuple2"), argument=arg1), argument=arg2):
                    # Handle Tuple2 structure: ((Tuple2 a) b) -> tuple2_a_b
                    return f"tuple2_{type_name(arg1)}_{type_name(arg2)}"
                case TypeApp(constructor=TypeApp(constructor=TypeApp(constructor=TypeCon(name="Tuple3"), argument=arg1), argument=arg2), argument=arg3):
                    # Handle Tuple3 structure: (((Tuple3 a) b) c) -> tuple3_a_b_c
                    return f"tuple3_{type_name(arg1)}_{type_name(arg2)}_{type_name(arg3)}"
                case TypeApp(constructor=constructor, argument=arg):
                    # Generic TypeApp case
                    return f"{type_name(constructor)}_{type_name(arg)}"
                case TypeVar(name=name):
                    return name.lower()
                case _:
                    return "any"

        # For function types, extract parameter types
        param_types = []
        current_type = func_type
        while isinstance(current_type, FunctionType):
            param_types.append(current_type.param)
            current_type = current_type.result

        if param_types:
            return "_".join(type_name(pt) for pt in param_types)
        else:
            return type_name(func_type)

    def _add_builtin_function(self, name: str, arity: int, type_info: Type) -> None:
        """Add a builtin function with its type information."""
        if name not in self.functions:
            self.functions[name] = FunctionInfo()

        monomorphized_name = self._create_monomorphized_name(name, type_info)
        self.functions[name].add_overload(arity, type_info, monomorphized_name)

    def _initialize_builtin_functions(self) -> None:
        """Initialize all builtin functions with their type signatures."""
        # Integer operations
        int_int_to_int = FunctionType(
            param=TypeCon("Int"),
            result=FunctionType(param=TypeCon("Int"), result=TypeCon("Int")),
        )
        int_to_int = FunctionType(param=TypeCon("Int"), result=TypeCon("Int"))
        int_int_to_bool = FunctionType(
            param=TypeCon("Int"),
            result=FunctionType(param=TypeCon("Int"), result=TypeCon("Bool")),
        )
        int_to_string = FunctionType(param=TypeCon("Int"), result=TypeCon("String"))

        self._add_builtin_function("primIntAdd", 2, int_int_to_int)
        self._add_builtin_function("primIntSub", 2, int_int_to_int)
        self._add_builtin_function("primIntMul", 2, int_int_to_int)
        self._add_builtin_function("primIntDiv", 2, int_int_to_int)
        self._add_builtin_function("primIntPow", 2, int_int_to_int)
        self._add_builtin_function("primIntNeg", 1, int_to_int)
        self._add_builtin_function("primIntLt", 2, int_int_to_bool)
        self._add_builtin_function("primIntLe", 2, int_int_to_bool)
        self._add_builtin_function("primIntGt", 2, int_int_to_bool)
        self._add_builtin_function("primIntGe", 2, int_int_to_bool)
        self._add_builtin_function("primIntEq", 2, int_int_to_bool)
        self._add_builtin_function("primIntShow", 1, int_to_string)

        # Float operations
        float_float_to_float = FunctionType(
            param=TypeCon("Float"),
            result=FunctionType(param=TypeCon("Float"), result=TypeCon("Float")),
        )
        float_to_float = FunctionType(param=TypeCon("Float"), result=TypeCon("Float"))
        float_float_to_bool = FunctionType(
            param=TypeCon("Float"),
            result=FunctionType(param=TypeCon("Float"), result=TypeCon("Bool")),
        )
        float_to_string = FunctionType(param=TypeCon("Float"), result=TypeCon("String"))

        self._add_builtin_function("primFloatAdd", 2, float_float_to_float)
        self._add_builtin_function("primFloatSub", 2, float_float_to_float)
        self._add_builtin_function("primFloatMul", 2, float_float_to_float)
        self._add_builtin_function("primFloatDiv", 2, float_float_to_float)
        self._add_builtin_function("primFloatPow", 2, float_float_to_float)
        self._add_builtin_function("primFloatNeg", 1, float_to_float)
        self._add_builtin_function("primFloatLt", 2, float_float_to_bool)
        self._add_builtin_function("primFloatLe", 2, float_float_to_bool)
        self._add_builtin_function("primFloatGt", 2, float_float_to_bool)
        self._add_builtin_function("primFloatGe", 2, float_float_to_bool)
        self._add_builtin_function("primFloatEq", 2, float_float_to_bool)
        self._add_builtin_function("primFloatShow", 1, float_to_string)

        # String operations
        string_string_to_string = FunctionType(
            param=TypeCon("String"),
            result=FunctionType(param=TypeCon("String"), result=TypeCon("String")),
        )
        string_string_to_bool = FunctionType(
            param=TypeCon("String"),
            result=FunctionType(param=TypeCon("String"), result=TypeCon("Bool")),
        )
        string_to_string = FunctionType(
            param=TypeCon("String"), result=TypeCon("String")
        )

        self._add_builtin_function("primStringConcat", 2, string_string_to_string)
        self._add_builtin_function("primStringEq", 2, string_string_to_bool)
        self._add_builtin_function("primStringShow", 1, string_to_string)

        # Character operations
        char_char_to_bool = FunctionType(
            param=TypeCon("Char"),
            result=FunctionType(param=TypeCon("Char"), result=TypeCon("Bool")),
        )
        char_to_string = FunctionType(param=TypeCon("Char"), result=TypeCon("String"))

        self._add_builtin_function("primCharEq", 2, char_char_to_bool)
        self._add_builtin_function("primCharShow", 1, char_to_string)

        # Boolean operations
        bool_bool_to_bool = FunctionType(
            param=TypeCon("Bool"),
            result=FunctionType(param=TypeCon("Bool"), result=TypeCon("Bool")),
        )
        bool_to_bool = FunctionType(param=TypeCon("Bool"), result=TypeCon("Bool"))
        bool_to_string = FunctionType(param=TypeCon("Bool"), result=TypeCon("String"))

        self._add_builtin_function("primBoolAnd", 2, bool_bool_to_bool)
        self._add_builtin_function("primBoolOr", 2, bool_bool_to_bool)
        self._add_builtin_function("primBoolNot", 1, bool_to_bool)
        self._add_builtin_function("primBoolEq", 2, bool_bool_to_bool)
        self._add_builtin_function("primBoolShow", 1, bool_to_string)

        # IO operations
        string_to_io_unit = FunctionType(
            param=TypeCon("String"),
            result=TypeApp(constructor=TypeCon("IO"), argument=TypeCon("()")),
        )

        # List operations
        list_a = TypeApp(constructor=TypeCon("List"), argument=TypeVar("a"))
        list_list_to_list = FunctionType(
            param=list_a,
            result=FunctionType(param=list_a, result=list_a),
        )
        list_to_string = FunctionType(param=list_a, result=TypeCon("String"))

        self._add_builtin_function("primListConcat", 2, list_list_to_list)
        self._add_builtin_function("primListShow", 1, list_to_string)

        # Error function
        string_to_a = FunctionType(param=TypeCon("String"), result=TypeVar("a"))
        self._add_builtin_function("error", 1, string_to_a)

        # Add operator overloads
        # + operator for different types
        self._add_builtin_function("+", 2, int_int_to_int)
        self._add_builtin_function("+", 2, float_float_to_float)

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
        """Convert a TypeExpression AST node to a Type object."""
        if hasattr(type_expr, "ty") and type_expr.ty is not None:
            return type_expr.ty

        # Convert from AST type expression nodes to Type objects
        match type_expr:
            case ArrowType(from_type=from_type, to_type=to_type):
                # Convert function type: A -> B becomes FunctionType(param=A, result=B)
                param_type = self._convert_type_expression_to_type(from_type)
                result_type = self._convert_type_expression_to_type(to_type)
                if param_type and result_type:
                    return FunctionType(param=param_type, result=result_type)
                return None
            case TypeConstructor(name=name):
                # Convert type constructor to TypeCon
                return TypeCon(name)
            case TypeVariable(name=name):
                # Convert type variable to TypeVar
                return TypeVar(name)
            case TypeApplication(constructor=constructor, argument=argument):
                # Convert type application: F A becomes TypeApp(constructor=F, argument=A)
                constructor_type = self._convert_type_expression_to_type(constructor)
                argument_type = self._convert_type_expression_to_type(argument)
                if constructor_type and argument_type:
                    return TypeApp(constructor=constructor_type, argument=argument_type)
                return None
            case TupleType(element_types=element_types):
                # Convert tuple types to a specialized tuple type representation
                # For now, we create a synthetic TypeCon for tuples
                if len(element_types) == 2:
                    # Binary tuple (a, b)
                    elem1_type = self._convert_type_expression_to_type(element_types[0])
                    elem2_type = self._convert_type_expression_to_type(element_types[1])
                    if elem1_type and elem2_type:
                        # Create a type application: Tuple2 a b
                        tuple_con = TypeCon("Tuple2")
                        tuple_with_first = TypeApp(
                            constructor=tuple_con, argument=elem1_type
                        )
                        return TypeApp(
                            constructor=tuple_with_first, argument=elem2_type
                        )
                elif len(element_types) == 3:
                    # Triple tuple (a, b, c)
                    elem1_type = self._convert_type_expression_to_type(element_types[0])
                    elem2_type = self._convert_type_expression_to_type(element_types[1])
                    elem3_type = self._convert_type_expression_to_type(element_types[2])
                    if elem1_type and elem2_type and elem3_type:
                        # Create a type application: Tuple3 a b c
                        tuple_con = TypeCon("Tuple3")
                        tuple_with_first = TypeApp(
                            constructor=tuple_con, argument=elem1_type
                        )
                        tuple_with_second = TypeApp(
                            constructor=tuple_with_first, argument=elem2_type
                        )
                        return TypeApp(
                            constructor=tuple_with_second, argument=elem3_type
                        )
                else:
                    # Generic tuple type for other arities
                    return TypeCon("Tuple")
                return None
            case list() if all(
                hasattr(elem, "name") or hasattr(elem, "__class__")
                for elem in type_expr
            ):
                # Handle list of type expressions (for complex types)
                converted_types = []
                for elem in type_expr:
                    converted = self._convert_type_expression_to_type(elem)
                    if converted:
                        converted_types.append(converted)

                if len(converted_types) == 1:
                    return converted_types[0]
                elif len(converted_types) > 1:
                    # Multiple types - could be a function chain or tuple
                    # Try to build a function type chain
                    result_type = converted_types[-1]
                    for param_type in reversed(converted_types[:-1]):
                        result_type = FunctionType(param=param_type, result=result_type)
                    return result_type
                return None
            case str():
                # Handle string type names directly
                return TypeCon(type_expr)
            case _:
                # Unknown type expression - try to extract name if available
                if hasattr(type_expr, "name"):
                    return TypeCon(str(type_expr.name))
                elif hasattr(type_expr, "__class__"):
                    # Last resort: use the class name
                    class_name = type_expr.__class__.__name__
                    if class_name.endswith("Type"):
                        return TypeCon(class_name[:-4])  # Remove 'Type' suffix
                    return TypeCon(class_name)
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
            case TypeCon(name="Tuple"):
                return "tuple"
            case TypeCon(name="Tuple2"):
                return "tuple"
            case TypeCon(name="Tuple3"):
                return "tuple"
            case TypeApp(constructor=TypeCon(name="List"), argument=arg_type):
                inner_type = self._systemo_type_to_python_hint(arg_type)
                return f"List[{inner_type}]"
            case TypeApp(constructor=TypeCon(name="IO"), argument=arg_type):
                # IO types typically don't have meaningful return types in our compiled Python
                return "None"
            case TypeApp(
                constructor=TypeApp(
                    constructor=TypeCon(name="Tuple2"), argument=arg1_type
                ),
                argument=arg2_type,
            ):
                # Binary tuple type: Tuple2 A B -> Tuple[A, B]
                type1_hint = self._systemo_type_to_python_hint(arg1_type)
                type2_hint = self._systemo_type_to_python_hint(arg2_type)
                return f"Tuple[{type1_hint}, {type2_hint}]"
            case TypeApp(
                constructor=TypeApp(
                    constructor=TypeApp(
                        constructor=TypeCon(name="Tuple3"), argument=arg1_type
                    ),
                    argument=arg2_type,
                ),
                argument=arg3_type,
            ):
                # Triple tuple type: Tuple3 A B C -> Tuple[A, B, C]
                type1_hint = self._systemo_type_to_python_hint(arg1_type)
                type2_hint = self._systemo_type_to_python_hint(arg2_type)
                type3_hint = self._systemo_type_to_python_hint(arg3_type)
                return f"Tuple[{type1_hint}, {type2_hint}, {type3_hint}]"
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
            case TypeCon(name=name):
                # Generic type constructor - use the name as-is
                return name
            case _:
                return "Any"

    def _infer_expression_type(self, expr: Expression) -> Optional[Type]:
        """Infer the type of an expression (enhanced type inference)."""
        match expr:
            case IntLiteral() | NegativeInt():
                return TypeCon("Int")
            case FloatLiteral() | NegativeFloat():
                return TypeCon("Float")
            case StringLiteral():
                return TypeCon("String")
            case CharLiteral():
                return TypeCon("Char")
            case BoolLiteral():
                return TypeCon("Bool")
            case ListLiteral(elements=elements):
                if elements:
                    elem_type = self._infer_expression_type(elements[0])
                    return (
                        TypeApp(constructor=TypeCon("List"), argument=elem_type)
                        if elem_type
                        else None
                    )
                return TypeApp(constructor=TypeCon("List"), argument=TypeVar("a"))
            case TupleLiteral(elements=elements):
                # Create simple tuple type based on element count
                arity = len(elements)
                if arity == 2:
                    return TypeCon("Tuple2")
                elif arity == 3:
                    return TypeCon("Tuple3")
                else:
                    return TypeCon(f"Tuple{arity}")
            case Variable(name=name):
                # First, try to get type from variable type tracker
                if name in self.variable_types:
                    print(f"DEBUG: Found tracked variable '{name}' with type: {self.variable_types[name]}")
                    return self.variable_types[name]
                
                # Try to get type from function info
                if name in self.functions and self.functions[name].overloads:
                    overload = self.functions[name].overloads[0]
                    if overload.type_info:
                        # Extract return type from function type
                        current_type = overload.type_info
                        while isinstance(current_type, FunctionType):
                            current_type = current_type.result
                        return current_type
                return None
            case Constructor(name=name):
                # For constructors, try to infer from data type definitions
                constructor_def = self._find_constructor_def(name)
                if constructor_def:
                    # Find which data type this constructor belongs to
                    for data_type_name, data_decl in self.data_types.items():
                        if any(ctor.name == name for ctor in data_decl.constructors):
                            return TypeCon(data_type_name)
                return None
            case FunctionApplication(function=function, argument=argument):
                # Infer function type and apply argument type
                func_type = self._infer_expression_type(function)
                arg_type = self._infer_expression_type(argument)

                # Special handling for monomorphized function calls
                if isinstance(function, Variable):
                    func_name = function.name
                    # If this is a monomorphized function name, look it up directly
                    if func_name.startswith("systemo_"):
                        # Find the function info that matches this monomorphized name
                        for base_name, func_info in self.functions.items():
                            for overload in func_info.overloads:
                                if (
                                    overload.monomorphized_name == func_name
                                    and overload.type_info
                                ):
                                    # Extract the return type from the function type
                                    current_type = overload.type_info
                                    while isinstance(current_type, FunctionType):
                                        current_type = current_type.result
                                    return current_type
                    # Try regular function lookup
                    elif (
                        func_name in self.functions
                        and self.functions[func_name].overloads
                    ):
                        # Use the first overload for now - in a full implementation we'd match arg types
                        overload = self.functions[func_name].overloads[0]
                        if overload.type_info:
                            current_type = overload.type_info
                            while isinstance(current_type, FunctionType):
                                current_type = current_type.result
                            return current_type

                if isinstance(func_type, FunctionType):
                    # Return the result type of the function
                    return func_type.result
                return None
            case ConstructorExpression(constructor_name=constructor_name):
                # Constructor application results in the data type
                constructor_def = self._find_constructor_def(constructor_name)
                if constructor_def:
                    for data_type_name, data_decl in self.data_types.items():
                        if any(
                            ctor.name == constructor_name
                            for ctor in data_decl.constructors
                        ):
                            return TypeCon(data_type_name)
                return None
            case SymbolicOperation(operator=operator, operands=operands):
                # Try to infer from operator overloads
                if operator in self.functions and self.functions[operator].overloads:
                    # For now, use the first overload's return type
                    overload = self.functions[operator].overloads[0]
                    if overload.type_info:
                        current_type = overload.type_info
                        while isinstance(current_type, FunctionType):
                            current_type = current_type.result
                        return current_type
                return None
            case GroupedExpression(expression=expression):
                # Grouped expressions have the same type as their content
                return self._infer_expression_type(expression)
            case IfElse(then_expr=then_expr, else_expr=else_expr):
                # If-else expressions have the type of the then/else branches
                then_type = self._infer_expression_type(then_expr)
                else_type = self._infer_expression_type(else_expr)
                # For now, return the first non-None type
                return then_type or else_type
            case DoBlock(statements=statements):
                # Do blocks have the type of their last statement
                if statements:
                    last_stmt = statements[-1]
                    if is_expression(last_stmt):
                        return self._infer_expression_type(last_stmt)  # type: ignore
                return None
            case _:
                return None

    def _resolve_function_call(self, func_name: str, args: List[Expression]) -> str:
        """Resolve function call to the appropriate monomorphized version."""
        # Check if this is a primitive function - don't monomorphize these
        if func_name.startswith("prim"):
            return func_name

        if func_name not in self.functions:
            # Unknown function, use default naming
            prefixed_name = self._prefix_name(func_name)
            print(f"DEBUG: Unknown function '{func_name}', using '{prefixed_name}'")
            return prefixed_name

        func_info = self.functions[func_name]

        # Infer argument types
        arg_types = [self._infer_expression_type(arg) for arg in args]
        print(
            f"DEBUG: Resolving '{func_name}' with inferred arg types: {[str(t) if t else 'None' for t in arg_types]}"
        )

        # Find best matching overload
        best_overload = func_info.find_best_overload(arg_types)

        if best_overload and best_overload.monomorphized_name:
            print(f"DEBUG: Selected overload: {best_overload.monomorphized_name}")
            return best_overload.monomorphized_name
        else:
            # Fallback to default naming
            fallback = self._prefix_name(func_name)
            print(f"DEBUG: No matching overload, using fallback: {fallback}")
            return fallback

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
        # Group instance declarations by name and type for proper pattern matching
        instance_groups: Dict[str, Dict[str, List[tuple]]] = (
            {}
        )  # name -> type_key -> [(func_def, type_info, monomorphized_name)]

        for stmt in program.statements:
            match stmt:
                case DataDeclaration():
                    lines.append(self._compile_data_declaration(stmt))
                case FunctionDefinition(function_name=function_name):
                    if function_name not in function_definitions:
                        function_definitions[function_name] = []
                    function_definitions[function_name].append(stmt)
                case InstanceDeclaration(
                    instance_name=instance_name,
                    function_definition=func_def,
                    ty=instance_ty,
                    type_signature=type_sig,
                ):
                    # Group instance declarations by name and type
                    func_def.function_name = instance_name  # Ensure the name matches

                    # Use the type from the instance declaration
                    type_info = (
                        instance_ty
                        if instance_ty is not None
                        else self._convert_type_expression_to_type(type_sig)
                    )
                    type_key = str(type_info) if type_info else "unknown"

                    # Create monomorphized name
                    monomorphized_name = self._create_monomorphized_name(
                        instance_name, type_info
                    )

                    # Group by name and type
                    if instance_name not in instance_groups:
                        instance_groups[instance_name] = {}
                    if type_key not in instance_groups[instance_name]:
                        instance_groups[instance_name][type_key] = []
                    instance_groups[instance_name][type_key].append(
                        (func_def, type_info, monomorphized_name)
                    )
                case LetStatement(variable=variable, value=value):
                    # Infer the type of the assigned value and track it
                    value_type = self._infer_expression_type(value)
                    if value_type:
                        self.variable_types[variable] = value_type
                        print(f"DEBUG: Tracked variable '{variable}' with type: {value_type}")
                    
                    prefixed_var = self._prefix_name(variable)
                    lines.append(
                        f"{prefixed_var} = {self._compile_expression(value)}",
                    )

        # First pass: Register all instance functions in the function registry
        for instance_name, type_groups in instance_groups.items():
            if instance_name not in self.functions:
                self.functions[instance_name] = FunctionInfo()

            for type_key, func_data_list in type_groups.items():
                for func_def, type_info, monomorphized_name in func_data_list:
                    max_params = len(func_def.patterns) if func_def.patterns else 0
                    self.functions[instance_name].add_overload(
                        max_params, type_info, monomorphized_name
                    )

        # First pass: Register all regular function definitions in the function registry
        for func_name, definitions in function_definitions.items():
            if func_name not in self.functions:
                self.functions[func_name] = FunctionInfo()

            # Group definitions by their type signature to create separate overloads
            for func_def in definitions:
                # Convert AST type to internal type if available
                type_info = func_def.ty if func_def.ty else None

                # Calculate arity
                max_params = len(func_def.patterns) if func_def.patterns else 0

                # Create monomorphized name
                monomorphized_name = self._create_monomorphized_name(
                    func_name, type_info
                )

                # Add overload
                self.functions[func_name].add_overload(
                    max_params, type_info, monomorphized_name
                )

        # Generate function definitions
        for func_name, definitions in function_definitions.items():
            lines.append(self._compile_function_group(func_name, definitions))

        # Generate instance declarations
        for instance_name, type_groups in instance_groups.items():
            print(
                f"DEBUG: Generating instances for '{instance_name}' with {len(type_groups)} type groups"
            )
            for type_key, func_data_list in type_groups.items():
                print(f"  Type group '{type_key}' has {len(func_data_list)} functions")
                # Extract the function definitions and type info
                func_defs = []
                type_info = None
                monomorphized_name = None

                for func_def, t_info, m_name in func_data_list:
                    func_defs.append(func_def)
                    if type_info is None:
                        type_info = t_info
                        monomorphized_name = m_name

                # Function registry already updated in first pass

                # If multiple function definitions with the same type, use pattern matching
                if len(func_defs) > 1:
                    lines.append(
                        self._compile_pattern_matching_function_with_custom_name(
                            monomorphized_name or f"systemo_{instance_name}",
                            instance_name,
                            func_defs,
                        )
                    )
                else:
                    # Single function definition - use simple compilation
                    lines.append(
                        self._compile_simple_function(
                            func_defs[0],
                            monomorphized_name or f"systemo_{instance_name}",
                        )
                    )

        # Add main execution
        if "main" in function_definitions:
            # Use the monomorphized main function name
            main_func_info = self.functions.get("main")
            if main_func_info and main_func_info.overloads:
                main_name = main_func_info.overloads[0].monomorphized_name
                lines.extend(["", "if __name__ == '__main__':", f"    {main_name}()"])
            else:
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
                    # Use Any for all constructor parameters to avoid forward reference issues
                    typed_args.append(f"arg_{i}: Any")

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
                    # Use Any for all constructor parameters to avoid forward reference issues
                    typed_args.append(f"arg_{i}: Any")

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
        """Compile function definitions, grouping by type signature for overloading."""
        # Group definitions by their type signature to create separate overloads
        type_groups: Dict[str, List[FunctionDefinition]] = {}

        for func_def in definitions:
            # Convert AST type to internal type if available
            type_info = None
            if func_def.ty:
                type_info = func_def.ty

            # Use type signature as the key for grouping
            type_key = str(type_info) if type_info else "unknown"
            if type_key not in type_groups:
                type_groups[type_key] = []
            type_groups[type_key].append(func_def)

        # Generate code for each type group (each becomes a separate overload)
        compiled_functions = []
        for type_key, group_definitions in type_groups.items():
            compiled_functions.append(
                self._compile_pattern_matching_function(func_name, group_definitions)
            )

        return "\n\n".join(compiled_functions)

    def _compile_pattern_matching_function(
        self,
        func_name: str,
        definitions: List[FunctionDefinition],
    ) -> str:
        """Compile function definitions with pattern matching for a single type signature."""
        # Track function arity (number of parameters)
        max_params = (
            max(len(defn.patterns) for defn in definitions) if definitions else 0
        )

        # Get function type from the first definition
        function_type = definitions[0].ty if definitions and definitions[0].ty else None
        monomorphized_name = self._create_monomorphized_name(func_name, function_type)

        if len(definitions) == 1:
            # Single definition - use simple compilation if it doesn't have complex patterns
            func_def = definitions[0]
            if len(func_def.patterns) <= 1 or all(
                isinstance(p, VariablePattern) for p in func_def.patterns
            ):
                return self._compile_simple_function(func_def, monomorphized_name)

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
            f"def {monomorphized_name}({', '.join(param_list)}) -> {return_type_hint}:",
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
                    f"raise ValueError(f'No matching pattern for {monomorphized_name}')"
                )
            elif max_params == 1:
                error_msg = f"raise ValueError(f'No matching pattern for {monomorphized_name} with args: {{arg_0}}')"
            else:
                error_msg = f"raise ValueError(f'No matching pattern for {monomorphized_name} with args: {{{arg_names}}}')"

            lines.append(self._indent() + error_msg)
        self.indent_level -= 1

        # Restore local variables
        self.local_variables = old_local_vars

        lines.append("")
        return "\n".join(lines)

    def _compile_pattern_matching_function_with_custom_name(
        self,
        custom_name: str,
        func_name: str,
        definitions: List[FunctionDefinition],
    ) -> str:
        """Compile function definitions with pattern matching using a custom function name."""
        # Track function arity (number of parameters)
        max_params = (
            max(len(defn.patterns) for defn in definitions) if definitions else 0
        )

        # Get return type hint from the first function definition
        return_type_hint = "Any"
        if definitions and definitions[0].ty:
            # For function types, extract the final return type
            current_type = definitions[0].ty
            while isinstance(current_type, FunctionType):
                current_type = current_type.result
            return_type_hint = self._systemo_type_to_python_hint(current_type)

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
            f"def {custom_name}({', '.join(param_list)}) -> {return_type_hint}:",
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
                        case _:
                            # Add other pattern types as needed
                            pass

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
                error_msg = f"raise ValueError('No matching pattern for {custom_name}')"
            elif max_params == 1:
                error_msg = f"raise ValueError(f'No matching pattern for {custom_name} with args: {{arg_0}}')"
            else:
                error_msg = f"raise ValueError(f'No matching pattern for {custom_name} with args: {{{arg_names}}}')"

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
        """Compile a simple function with multiple parameters."""
        func_name = prefixed_name or f"systemo_{func_def.function_name}"

        # Track pattern variables
        old_local_vars = self.local_variables.copy()
        for pattern in func_def.patterns:
            self.local_variables.update(self._extract_pattern_variables(pattern))

        # Get type hints from the function's type annotation
        return_type_hint = "Any"
        param_type_hints = []

        if func_def.ty:
            # Extract parameter types from function type chain
            current_type = func_def.ty
            while isinstance(current_type, FunctionType):
                param_type_hints.append(
                    self._systemo_type_to_python_hint(current_type.param)
                )
                current_type = current_type.result
            return_type_hint = self._systemo_type_to_python_hint(current_type)

        # Ensure we have at least as many type hints as patterns
        while len(param_type_hints) < len(func_def.patterns):
            param_type_hints.append("Any")

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
        elif len(func_def.patterns) == 1:
            # Single parameter function
            pattern = func_def.patterns[0]
            param_type_hint = param_type_hints[0] if param_type_hints else "Any"

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
        else:
            # Multiple parameter function - handle all patterns as parameters
            param_list = []
            for i, pattern in enumerate(func_def.patterns):
                param_type_hint = (
                    param_type_hints[i] if i < len(param_type_hints) else "Any"
                )
                match pattern:
                    case VariablePattern(name=name):
                        param_list.append(f"{name}: {param_type_hint}")
                    case _:
                        param_list.append(f"arg_{i}: {param_type_hint}")

            # Check if the body is a do block with multiple statements
            match func_def.body:
                case DoBlock(statements=statements) if len(statements) > 1:
                    lines = [
                        f"def {func_name}({', '.join(param_list)}) -> {return_type_hint}:",
                    ]
                    lines.extend(self._compile_do_block_as_statements(func_def.body))
                case _:
                    lines = [
                        f"def {func_name}({', '.join(param_list)}) -> {return_type_hint}:",
                        f"    return {self._compile_expression(func_def.body)}",
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
                    # Track the variable type for tuple resolution
                    inferred_type = self._infer_expression_type(value)
                    if inferred_type:
                        # Store both prefixed and original variable names for lookup
                        self.variable_types[variable] = inferred_type
                        self.variable_types[prefixed_var] = inferred_type
                        print(f"DEBUG: Tracked variable '{variable}' (prefixed: '{prefixed_var}') with type: {inferred_type}")
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
                # Track the variable type for tuple resolution
                inferred_type = self._infer_expression_type(value)
                if inferred_type:
                    # Store both prefixed and original variable names for lookup
                    self.variable_types[variable] = inferred_type
                    self.variable_types[prefixed_var] = inferred_type
                    print(f"DEBUG: Tracked variable '{variable}' (prefixed: '{prefixed_var}') with type: {inferred_type}")
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
                # Check if this is a nullary function (0 parameters) and automatically call it
                if name in self.functions and self.functions[name].arity == 0:
                    resolved_name = self._resolve_function_call(name, [])
                    return f"{resolved_name}()"
                else:
                    resolved_name = self._resolve_function_call(name, [])
                    return resolved_name
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
                        # Resolve to the correct monomorphized function name
                        resolved_name = self._resolve_function_call(name, args)

                        # Get the expected arity for this function
                        func_info = self.functions.get(name)
                        if func_info is None:
                            # Create default function info with arity 1
                            func_info = FunctionInfo()
                            func_info.add_overload(1, None, self._prefix_name(name))
                        expected_arity = func_info.arity

                        if len(args) == expected_arity:
                            # Full application - call function directly
                            arg_exprs = [self._compile_expression(arg) for arg in args]
                            return f"{resolved_name}({', '.join(arg_exprs)})"
                        elif len(args) < expected_arity and name in self.functions:
                            # Partial application - create lambda for remaining args
                            arg_exprs = [self._compile_expression(arg) for arg in args]
                            remaining_params = expected_arity - len(args)
                            lambda_params = [
                                f"__arg_{i}" for i in range(remaining_params)
                            ]
                            all_args = arg_exprs + lambda_params
                            return f"lambda {', '.join(lambda_params)}: {resolved_name}({', '.join(all_args)})"
                        elif len(args) > expected_arity:
                            # Over-application - call function with exact args, then apply rest
                            exact_args = args[:expected_arity]
                            remaining_args = args[expected_arity:]

                            arg_exprs = [
                                self._compile_expression(arg) for arg in exact_args
                            ]
                            result = f"{resolved_name}({', '.join(arg_exprs)})"

                            # Apply remaining arguments one by one
                            for arg in remaining_args:
                                arg_expr = self._compile_expression(arg)
                                result = f"{result}({arg_expr})"
                            return result
                        else:
                            # Single argument or built-in function - use resolved name
                            if len(args) == 1:
                                arg_expr = self._compile_expression(args[0])
                                return f"{resolved_name}({arg_expr})"
                            else:
                                # Multiple args - use nested calls
                                result = resolved_name
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

            # Try to resolve to monomorphized function name
            resolved_name = self._resolve_function_call(operator, operands)
            return f"{resolved_name}({left}, {right})"

        # Handle unary operators
        elif len(operands) == 1:
            operand = self._compile_expression(operands[0])

            # Try to resolve to monomorphized function name
            resolved_name = self._resolve_function_call(operator, operands)
            return f"{resolved_name}({operand})"

        # Fallback for other cases
        else:
            compiled_operands = [self._compile_expression(op) for op in operands]
            resolved_name = self._resolve_function_call(operator, operands)
            return f"{resolved_name}({', '.join(compiled_operands)})"

    def _compile_do_block(self, do_block: DoBlock) -> str:
        print(f"DEBUG: _compile_do_block called with {len(do_block.statements)} statements")
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
                    # Track the variable type for tuple resolution
                    inferred_type = self._infer_expression_type(value)
                    if inferred_type:
                        # Store both prefixed and original variable names for lookup
                        self.variable_types[variable] = inferred_type
                        self.variable_types[prefixed_var] = inferred_type
                        print(f"DEBUG: Tracked variable '{variable}' (prefixed: '{prefixed_var}') with type: {inferred_type}")
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
                # Track the variable type for tuple resolution
                inferred_type = self._infer_expression_type(value)
                if inferred_type:
                    # Store both prefixed and original variable names for lookup
                    self.variable_types[variable] = inferred_type
                    self.variable_types[prefixed_var] = inferred_type
                    print(f"DEBUG: Tracked variable '{variable}' (prefixed: '{prefixed_var}') with type: {inferred_type}")
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
