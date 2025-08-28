from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from lango.systemo.ast.nodes import (
    BoolLiteral,
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
    IfElse,
    IntLiteral,
    LetStatement,
    ListLiteral,
    LiteralPattern,
    NegativeFloat,
    NegativeInt,
    Pattern,
    Program,
    Statement,
    StringLiteral,
    SymbolicOperation,
    Variable,
    VariablePattern,
)
from lango.systemo.typechecker.typecheck import type_check

# Type aliases for the interpreter
Value = Any  # Any runtime value
# Type definitions for runtime values
Record = Dict[str, Any]  # Dictionary representing a record/object
FunctionClause = Tuple[List[Pattern], Expression]
FunctionValue = Tuple[str, List[FunctionClause]]  # ("pattern_match", clauses)
Environment = Dict[str, FunctionValue]
ConstructorInfo = Tuple[int, str]  # (arity, data_type_name)
ConstructorEnvironment = Dict[str, ConstructorInfo]


@dataclass
class RunReturn:
    output: str
    exit_code: int


def interpret(
    ast: Program,
    collectStdOut: bool = False,
) -> RunReturn:
    if not type_check(ast):
        print("Type checking failed, cannot interpret.")
        return RunReturn("", 1)

    env, constructors = build_environment(ast)
    interp = Interpreter(env, constructors)

    if "main" not in env:
        raise RuntimeError("No main function defined")

    if collectStdOut:
        f = StringIO()
        with redirect_stdout(f):
            result = interp.eval_func("main")
        output = f.getvalue()
    else:
        result = interp.eval_func("main")
        output = ""

    if not collectStdOut:
        if result is not None and not callable(result):
            print(f"{result}\n")
        elif callable(result):
            print("[main] is a function")
    return RunReturn(output, 0)


def build_environment(ast: Program) -> Tuple[Environment, ConstructorEnvironment]:
    env: Environment = {}
    constructors: ConstructorEnvironment = {}

    for stmt in ast.statements:
        match stmt:
            case FunctionDefinition(
                function_name=func_name,
                patterns=patterns,
                body=body,
            ):
                # Support multiple function clauses for pattern matching
                if func_name not in env:
                    env[func_name] = ("pattern_match", [])

                # Add this clause to the function's clauses list
                env[func_name][1].append((patterns, body))

            case DataDeclaration(type_name=type_name, constructors=data_constructors):
                # Process data type constructors
                for constructor in data_constructors:
                    ctor_name = constructor.name

                    if constructor.record_constructor:
                        # Record constructor - arity is number of fields
                        arity = len(constructor.record_constructor.fields)
                    elif constructor.type_atoms:
                        # Positional constructor - arity is number of type atoms
                        arity = len(constructor.type_atoms)
                    else:
                        # Nullary constructor - arity 0
                        arity = 0

                    constructors[ctor_name] = (arity, type_name)
            case _:
                pass

    return env, constructors


def flexible_putStr(
    arg: Union[Value, Callable[[Value], Value]],
) -> Optional[Callable[[Value], None]]:
    """putStr that can handle both single argument and curried usage."""
    if callable(arg):
        # If given a function (like 'show'), return a curried function
        def curried_putStr(value: Value) -> None:
            result = arg(value)  # Apply the function to the value
            match result:
                case str():
                    result = result.encode().decode("unicode_escape")
            print(result, end="")

        return curried_putStr
    else:
        # If given a direct value, print it
        match arg:
            case str():
                arg = arg.encode().decode("unicode_escape")
        print(arg, end="")
        return None


def _error(message: str) -> Any:
    raise RuntimeError(f"Runtime error: {message}")


def _show(value: Value) -> str:
    match value:
        case str():
            return f'"{value}"'
        case list():
            return "[" + ",".join(_show(i) for i in value) + "]"
        case float():
            if value == float("inf"):
                return "Infinity"
            if value == float("-inf"):
                return "-Infinity"
            return str(value)
        case dict() if "_constructor" in value:
            # Handle constructor values
            constructor = value["_constructor"]
            # Get field values in order
            field_values = []
            i = 0
            while f"field_{i}" in value:
                field_values.append(value[f"field_{i}"])
                i += 1

            if field_values:
                field_strs = [_show(field_val) for field_val in field_values]
                return f"{constructor}({', '.join(field_strs)})"
            else:
                return constructor
        case _:
            return str(value)


# Built-in functions available in the language
builtins: Dict[str, Callable[..., Any]] = {
    "putStr": flexible_putStr,
    "show": lambda x: _show(x),
    "error": lambda x: _error(x),
    "mod": lambda x: lambda y: x % y,
}


class Interpreter:
    def __init__(self, env: Environment, constructors: ConstructorEnvironment) -> None:
        self.env = env
        self.constructors = constructors
        self.variables: Dict[str, Value] = {}

    def eval(self, node: Expression) -> Value:
        match node:
            # Literals
            case IntLiteral(value=value):
                return value
            case FloatLiteral(value=value):
                return value
            case StringLiteral(value=value):
                return value
            case BoolLiteral(value=value):
                return value
            case NegativeInt(value=value):
                return value
            case NegativeFloat(value=value):
                return value
            case ListLiteral(elements=elements):
                return [self.eval(elem) for elem in elements]

            # Variables and constructors
            case Variable(name=name):
                if name in self.variables:
                    return self.variables[name]
                elif name in self.env:
                    return self.eval_func(name)
                elif name in builtins:
                    return builtins[name]
                else:
                    raise RuntimeError(f"Unknown variable: {name}")

            case Constructor(name=constructor_name):
                # Check if this is a known constructor
                if constructor_name in self.constructors:
                    arity, data_type = self.constructors[constructor_name]

                    if arity == 0:
                        # Nullary constructor - return the value directly
                        return {"_constructor": constructor_name}
                    else:
                        # Return a curried constructor function
                        def make_constructor(
                            collected_args: Optional[List[Value]] = None,
                        ) -> Callable[[Value], Value]:
                            if collected_args is None:
                                collected_args = []

                            def constructor_fn(
                                arg: Value,
                            ) -> Union[Record, Callable[[Value], Value]]:
                                new_args = collected_args + [arg]
                                # Create when we have enough arguments
                                if len(new_args) >= arity:
                                    # Create the record with the collected arguments
                                    fields: Record = {}
                                    for i, value in enumerate(new_args):
                                        fields[f"field_{i}"] = value
                                    return {"_constructor": constructor_name, **fields}
                                else:
                                    # Still collecting arguments
                                    return make_constructor(new_args)

                            return constructor_fn

                        return make_constructor()
                else:
                    # Unknown constructor - treat as variable
                    if constructor_name in self.variables:
                        return self.variables[constructor_name]
                    elif constructor_name in self.env:
                        return self.eval_func(constructor_name)
                    elif constructor_name in builtins:
                        return builtins[constructor_name]
                    else:
                        raise RuntimeError(
                            f"Unknown constructor or variable: {constructor_name}",
                        )

            # Generic symbolic operations - convert to function application
            case SymbolicOperation(operator=operator, operands=operands):
                # Transform symbolic operation into function application
                operator_var = Variable(operator)
                if len(operands) == 1:
                    # Unary operation: f x
                    return self.eval(FunctionApplication(operator_var, operands[0]))
                elif len(operands) == 2:
                    # Binary operation: ((f x) y)
                    partial_app = FunctionApplication(operator_var, operands[0])
                    return self.eval(FunctionApplication(partial_app, operands[1]))
                else:
                    raise RuntimeError(
                        f"Unsupported arity for operator {operator}: {len(operands)}",
                    )

            # Control flow
            case IfElse(condition=condition, then_expr=then_expr, else_expr=else_expr):
                cond_val = self.eval(condition)
                if cond_val:
                    return self.eval(then_expr)
                else:
                    return self.eval(else_expr)

            case DoBlock(statements=statements):
                return self.eval_do_block(statements)

            # Function application
            case FunctionApplication(function=function, argument=argument):
                func = self.eval(function)
                arg = self.eval(argument)
                return func(arg)

            # Constructor expressions
            case ConstructorExpression(
                constructor_name=constructor_name,
                fields=fields,
            ):
                field_dict: Record = {}
                for field_assign in fields:
                    field_value = self.eval(field_assign.value)
                    field_dict[field_assign.field_name] = field_value
                return {"_constructor": constructor_name, **field_dict}

            # Grouping
            case GroupedExpression(expression=expression):
                return self.eval(expression)

            # Default case
            case _:
                raise NotImplementedError(
                    f"Unhandled expression type: {type(node).__name__}",
                )

    def eval_do_block(self, statements: List[Statement]) -> Value:
        result: Value = None
        for stmt in statements:
            match stmt:
                case LetStatement(variable=variable, value=value):
                    eval_value = self.eval(value)
                    self.variables[variable] = eval_value
                case (
                    IntLiteral()
                    | FloatLiteral()
                    | StringLiteral()
                    | BoolLiteral()
                    | ListLiteral()
                    | Variable()
                    | Constructor()
                    | SymbolicOperation()
                    | IfElse()
                    | DoBlock()
                    | FunctionApplication()
                    | ConstructorExpression()
                    | GroupedExpression()
                    | NegativeInt()
                    | NegativeFloat()
                ) as expr:
                    # It's an expression
                    result = self.eval(expr)
                case _:
                    pass
        return result

    def eval_func(self, func_name: str) -> Value:
        if func_name not in self.env:
            raise RuntimeError(f"Unknown function: {func_name}")

        func_type, clauses = self.env[func_name]

        if func_type != "pattern_match":
            raise RuntimeError(f"Unsupported function type: {func_type}")

        # Handle only nullary functions (no arguments)
        for patterns, body in clauses:
            if len(patterns) == 0:
                # Nullary function - just evaluate the body
                return self.eval(body)

        # If no nullary clause found, return a curried function
        def curried_function(*args: Any) -> Value:
            return self._apply_function_with_patterns(clauses, list(args))

        return curried_function

    def _apply_function_with_patterns(
        self,
        clauses: List[FunctionClause],
        args: List[Value],
    ) -> Value:
        for patterns, body in clauses:
            if len(patterns) == len(args):
                # Try to match this clause
                old_vars = self.variables.copy()  # Save current variable state
                try:
                    if self._match_patterns(patterns, args):
                        result = self.eval(body)
                        self.variables = old_vars  # Restore variables after evaluation
                        return result
                finally:
                    # Ensure variables are restored even if evaluation fails
                    self.variables = old_vars

        # If no patterns matched and we don't have enough args, return partial application
        if len(args) < max(len(patterns) for patterns, _ in clauses):

            def partial_function(next_arg: Value) -> Value:
                return self._apply_function_with_patterns(clauses, args + [next_arg])

            return partial_function

        raise RuntimeError(
            f"No matching pattern found for function call with {len(args)} arguments",
        )

    def _match_patterns(self, patterns: List[Pattern], args: List[Value]) -> bool:
        if len(patterns) != len(args):
            return False

        # Save current variable state
        old_vars = self.variables.copy()

        try:
            for pattern, arg in zip(patterns, args):
                if not self._match_pattern(pattern, arg):
                    # Restore variables if match failed
                    self.variables = old_vars
                    return False
            return True
        except Exception:
            # Restore variables if match failed
            self.variables = old_vars
            return False

    def _match_pattern(self, pattern: Pattern, value: Value) -> bool:
        """Match a single pattern against a value."""
        match pattern:
            case VariablePattern(name=name):
                # Variable pattern always matches and binds the value
                self.variables[name] = value
                return True

            case LiteralPattern(value=pattern_value):
                # Literal pattern must match exactly
                return pattern_value == value

            case ConstructorPattern(constructor=constructor, patterns=patterns):
                # Match constructor pattern
                match value:
                    case dict() if "_constructor" in value:
                        pass  # Valid constructor value
                    case _:
                        return False

                if value["_constructor"] != constructor:
                    return False

                # Match sub-patterns against constructor fields
                if len(patterns) == 0:
                    # Constructor with no patterns (like Zero or Nil used as pattern)
                    return True

                # Check if this is a record constructor or positional constructor
                if constructor in self.constructors:
                    arity, data_type = self.constructors[constructor]

                    # If the value has named fields (not field_0, field_1, etc), it's a record
                    has_named_fields = any(
                        key not in ["_constructor"] and not key.startswith("field_")
                        for key in value.keys()
                    )

                    if has_named_fields:
                        # Record constructor pattern matching
                        # For record patterns like (Person id_ name), we need to match against record fields in the order they were declared
                        # This requires looking up the field names from the data type definition

                        # Assume the pattern variables match the field names in declaration order
                        field_names = [
                            key for key in value.keys() if key != "_constructor"
                        ]
                        field_names.sort()  # Sort to ensure consistent order

                        if len(patterns) != len(field_names):
                            return False

                        # Match each pattern against its corresponding field value
                        for sub_pattern, field_name in zip(patterns, field_names):
                            if not self._match_pattern(sub_pattern, value[field_name]):
                                return False

                        return True
                    else:
                        # Positional constructor pattern matching
                        field_values = []
                        for i in range(len(patterns)):
                            field_key = f"field_{i}"
                            if field_key in value:
                                field_values.append(value[field_key])
                            else:
                                return False  # Not enough fields

                        if len(patterns) != len(field_values):
                            return False

                        # Match each pattern against its corresponding field value
                        for sub_pattern, field_value in zip(patterns, field_values):
                            if not self._match_pattern(sub_pattern, field_value):
                                return False

                        return True
                else:
                    # Unknown constructor, treat as positional
                    field_values = []
                    for i in range(len(patterns)):
                        field_key = f"field_{i}"
                        if field_key in value:
                            field_values.append(value[field_key])
                        else:
                            return False  # Not enough fields

                    if len(patterns) != len(field_values):
                        return False

                    # Match each pattern against its corresponding field value
                    for sub_pattern, field_value in zip(patterns, field_values):
                        if not self._match_pattern(sub_pattern, field_value):
                            return False

                    return True

            case ConsPattern(head=head, tail=tail):
                # Match cons pattern (head : tail)
                match value:
                    case list() if len(value) > 0:
                        pass  # Valid non-empty list
                    case _:
                        return False

                head_val = value[0]
                tail_val = value[1:]

                return self._match_pattern(head, head_val) and self._match_pattern(
                    tail,
                    tail_val,
                )

            case _:
                raise NotImplementedError(
                    f"Unhandled pattern type: {type(pattern).__name__}",
                )
