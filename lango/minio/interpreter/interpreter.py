"""
Rewritten interpreter that uses custom AST nodes instead of raw Lark objects.
"""

from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from lango.minio.ast.nodes import (
    AddOperation,
    AndOperation,
    BoolLiteral,
    ConcatOperation,
    ConsPattern,
    Constructor,
    ConstructorExpression,
    ConstructorPattern,
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
    PowFloatOperation,
    PowIntOperation,
    Program,
    Statement,
    StringLiteral,
    SubOperation,
    Variable,
    VariablePattern,
)
from lango.minio.typechecker.typecheck import type_check

# Type aliases for the interpreter
Value = Any  # Any runtime value
# Type definitions for runtime values
Record = Dict[str, Any]  # Dictionary representing a record/object
FunctionClause = Tuple[List[Pattern], Expression]
FunctionValue = Tuple[str, List[FunctionClause]]  # ("pattern_match", clauses)
Environment = Dict[str, FunctionValue]


def interpret(
    ast: Program,
    collectStdOut: bool = False,
) -> str:
    # TODO: return code
    if not type_check(ast):
        print("Type checking failed, cannot interpret.")
        return ""

    env = build_environment(ast)
    interp = Interpreter(env)

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
    return output


def build_environment(ast: Program) -> Environment:
    """Build environment from AST collecting function definitions."""
    env: Environment = {}

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
            case _:
                pass

    return env


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
    """Runtime error function that throws an exception."""
    raise RuntimeError(f"Runtime error: {message}")


def _show(value: Value) -> str:
    """Convert a value to its string representation."""
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
        case _:
            return str(value)


# Built-in functions available in the language
builtins: Dict[str, Callable[..., Any]] = {
    "putStr": flexible_putStr,
    "show": lambda x: _show(x),
    "error": lambda x: _error(x),
}


class Interpreter:
    """Interpreter class for evaluating AST expressions."""

    def __init__(self, env: Environment) -> None:
        """Initialize interpreter with environment of functions."""
        self.env = env
        self.variables: Dict[str, Value] = {}

    def eval(self, node: Expression) -> Value:
        """Evaluate an AST expression node and return its value."""

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
                        # For now, assume we need 2 or more args and create when we get them
                        if len(new_args) >= 2:
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

            # Arithmetic operations
            case AddOperation(left=left, right=right):
                return self.eval(left) + self.eval(right)
            case SubOperation(left=left, right=right):
                return self.eval(left) - self.eval(right)
            case MulOperation(left=left, right=right):
                return self.eval(left) * self.eval(right)
            case DivOperation(left=left, right=right):
                if self.eval(right) == 0:
                    return float("inf")  # special case because haskell does this
                return self.eval(left) / self.eval(right)
            case PowIntOperation(left=left, right=right):
                return int(self.eval(left) ** self.eval(right))
            case PowFloatOperation(left=left, right=right):
                return float(self.eval(left) ** self.eval(right))

            # Comparison operations
            case EqualOperation(left=left, right=right):
                return self.eval(left) == self.eval(right)
            case NotEqualOperation(left=left, right=right):
                return self.eval(left) != self.eval(right)
            case LessThanOperation(left=left, right=right):
                return self.eval(left) < self.eval(right)
            case LessEqualOperation(left=left, right=right):
                return self.eval(left) <= self.eval(right)
            case GreaterThanOperation(left=left, right=right):
                return self.eval(left) > self.eval(right)
            case GreaterEqualOperation(left=left, right=right):
                return self.eval(left) >= self.eval(right)

            # Logical operations
            case AndOperation(left=left, right=right):
                return self.eval(left) and self.eval(right)
            case OrOperation(left=left, right=right):
                return self.eval(left) or self.eval(right)
            case NotOperation(operand=operand):
                return not self.eval(operand)

            # String/List operations
            case ConcatOperation(left=left, right=right):
                left_val = self.eval(left)
                right_val = self.eval(right)
                match (left_val, right_val):
                    case (list(), list()):
                        # List concatenation
                        return left_val + right_val
                    case _:
                        # String concatenation
                        return str(left_val) + str(right_val)

            case IndexOperation(list_expr=list_expr, index_expr=index_expr):
                list_val = self.eval(list_expr)
                index_val = self.eval(index_expr)
                match list_val:
                    case list():
                        pass  # Valid list
                    case _:
                        raise RuntimeError(
                            f"Cannot index non-list value: {type(list_val)}",
                        )
                match index_val:
                    case int():
                        pass  # Valid index
                    case _:
                        raise RuntimeError(
                            f"List index must be an integer, got: {type(index_val)}",
                        )
                if index_val < 0 or index_val >= len(list_val):
                    raise RuntimeError(
                        f"List index {index_val} out of bounds for list of length {len(list_val)}",
                    )
                return list_val[index_val]

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
        """Evaluate a do block with statements."""
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
                    | AddOperation()
                    | SubOperation()
                    | MulOperation()
                    | DivOperation()
                    | PowIntOperation()
                    | PowFloatOperation()
                    | EqualOperation()
                    | NotEqualOperation()
                    | LessThanOperation()
                    | LessEqualOperation()
                    | GreaterThanOperation()
                    | GreaterEqualOperation()
                    | AndOperation()
                    | OrOperation()
                    | NotOperation()
                    | ConcatOperation()
                    | IndexOperation()
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
        """Evaluate a function by name."""
        if func_name not in self.env:
            raise RuntimeError(f"Unknown function: {func_name}")

        func_type, clauses = self.env[func_name]

        if func_type != "pattern_match":
            raise RuntimeError(f"Unsupported function type: {func_type}")

        # For now, handle only nullary functions (no arguments)
        # This is a simplified implementation
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
        """Apply function with pattern matching."""
        for patterns, body in clauses:
            if len(patterns) == len(args):
                # Try to match this clause
                if self._match_patterns(patterns, args):
                    return self.eval(body)

        # If no patterns matched and we don't have enough args, return partial application
        if len(args) < max(len(patterns) for patterns, _ in clauses):

            def partial_function(next_arg: Value) -> Value:
                return self._apply_function_with_patterns(clauses, args + [next_arg])

            return partial_function

        raise RuntimeError(
            f"No matching pattern found for function call with {len(args)} arguments",
        )

    def _match_patterns(self, patterns: List[Pattern], args: List[Value]) -> bool:
        """Check if patterns match arguments and bind variables."""
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

            case ListLiteral(elements=elements):
                # Match list literal pattern against list value
                match value:
                    case list():
                        pass  # Valid list
                    case _:
                        return False
                if len(elements) != len(value):
                    return False
                # For list literals as patterns, evaluate elements and compare values
                for pattern_elem, value_elem in zip(elements, value):
                    pattern_val = self.eval(pattern_elem)
                    if pattern_val != value_elem:
                        return False
                return True

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
                # For record constructors like Person {id_: Int, name: String},
                # the patterns will match against the field values
                if len(patterns) == 0:
                    # Constructor with no patterns (like MkPoint used as value)
                    return True

                # Get the constructor fields in order
                # For now, assume field order matches pattern order
                # This is a simplification - a full implementation would need
                # to look up the data type definition
                field_values = []
                for key in sorted(value.keys()):
                    if key != "_constructor":
                        field_values.append(value[key])

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
