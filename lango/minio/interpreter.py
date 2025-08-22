"""
Rewritten interpreter that uses custom AST nodes instead of raw Lark objects.
"""

from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from lango.minio.ast_nodes import (
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
from lango.minio.typecheck import type_check

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
        if isinstance(stmt, FunctionDefinition):
            func_name = stmt.function_name

            # Support multiple function clauses for pattern matching
            if func_name not in env:
                env[func_name] = ("pattern_match", [])

            # Add this clause to the function's clauses list
            env[func_name][1].append((stmt.patterns, stmt.body))

    return env


def flexible_putStr(
    arg: Union[Value, Callable[[Value], Value]],
) -> Optional[Callable[[Value], None]]:
    """putStr that can handle both single argument and curried usage."""
    if callable(arg):
        # If given a function (like 'show'), return a curried function
        def curried_putStr(value: Value) -> None:
            result = arg(value)  # Apply the function to the value
            if isinstance(result, str):
                result = result.encode().decode("unicode_escape")
            print(result, end="")

        return curried_putStr
    else:
        # If given a direct value, print it
        if isinstance(arg, str):
            arg = arg.encode().decode("unicode_escape")
        print(arg, end="")
        return None


def _show(value: Value) -> str:
    """Convert a value to its string representation."""
    if isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, list):
        return "[" + ",".join(_show(i) for i in value) + "]"
    return str(value)


# Built-in functions available in the language
builtins: Dict[str, Callable[..., Any]] = {
    "putStr": flexible_putStr,
    "show": lambda x: _show(x),
}


class Interpreter:
    """Interpreter class for evaluating AST expressions."""

    def __init__(self, env: Environment) -> None:
        """Initialize interpreter with environment of functions."""
        self.env = env
        self.variables: Dict[str, Value] = {}

    def eval(self, node: Expression) -> Value:
        """Evaluate an AST expression node and return its value."""

        # Literals
        if isinstance(node, IntLiteral):
            return node.value

        elif isinstance(node, FloatLiteral):
            return node.value

        elif isinstance(node, StringLiteral):
            return node.value

        elif isinstance(node, BoolLiteral):
            return node.value

        elif isinstance(node, NegativeInt):
            return node.value

        elif isinstance(node, NegativeFloat):
            return node.value

        elif isinstance(node, ListLiteral):
            return [self.eval(elem) for elem in node.elements]

        # Variables and constructors
        elif isinstance(node, Variable):
            name = node.name
            if name in self.variables:
                return self.variables[name]
            elif name in self.env:
                return self.eval_func(name)
            elif name in builtins:
                return builtins[name]
            else:
                raise RuntimeError(f"Unknown variable: {name}")

        elif isinstance(node, Constructor):
            constructor_name = node.name

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
        elif isinstance(node, AddOperation):
            return self.eval(node.left) + self.eval(node.right)

        elif isinstance(node, SubOperation):
            return self.eval(node.left) - self.eval(node.right)

        elif isinstance(node, MulOperation):
            return self.eval(node.left) * self.eval(node.right)

        elif isinstance(node, DivOperation):
            return self.eval(node.left) / self.eval(node.right)

        elif isinstance(node, PowIntOperation):
            return int(self.eval(node.left) ** self.eval(node.right))

        elif isinstance(node, PowFloatOperation):
            return float(self.eval(node.left) ** self.eval(node.right))

        # Comparison operations
        elif isinstance(node, EqualOperation):
            return self.eval(node.left) == self.eval(node.right)

        elif isinstance(node, NotEqualOperation):
            return self.eval(node.left) != self.eval(node.right)

        elif isinstance(node, LessThanOperation):
            return self.eval(node.left) < self.eval(node.right)

        elif isinstance(node, LessEqualOperation):
            return self.eval(node.left) <= self.eval(node.right)

        elif isinstance(node, GreaterThanOperation):
            return self.eval(node.left) > self.eval(node.right)

        elif isinstance(node, GreaterEqualOperation):
            return self.eval(node.left) >= self.eval(node.right)

        # Logical operations
        elif isinstance(node, AndOperation):
            return self.eval(node.left) and self.eval(node.right)

        elif isinstance(node, OrOperation):
            return self.eval(node.left) or self.eval(node.right)

        elif isinstance(node, NotOperation):
            return not self.eval(node.operand)

        # String/List operations
        elif isinstance(node, ConcatOperation):
            left = self.eval(node.left)
            right = self.eval(node.right)
            if isinstance(left, list) and isinstance(right, list):
                # List concatenation
                return left + right
            else:
                # String concatenation
                return str(left) + str(right)

        elif isinstance(node, IndexOperation):
            list_val = self.eval(node.list_expr)
            index_val = self.eval(node.index_expr)
            if not isinstance(list_val, list):
                raise RuntimeError(f"Cannot index non-list value: {type(list_val)}")
            if not isinstance(index_val, int):
                raise RuntimeError(
                    f"List index must be an integer, got: {type(index_val)}",
                )
            if index_val < 0 or index_val >= len(list_val):
                raise RuntimeError(
                    f"List index {index_val} out of bounds for list of length {len(list_val)}",
                )
            return list_val[index_val]

        # Control flow
        elif isinstance(node, IfElse):
            condition = self.eval(node.condition)
            if condition:
                return self.eval(node.then_expr)
            else:
                return self.eval(node.else_expr)

        elif isinstance(node, DoBlock):
            return self.eval_do_block(node.statements)

        # Function application
        elif isinstance(node, FunctionApplication):
            func = self.eval(node.function)
            arg = self.eval(node.argument)
            return func(arg)

        # Constructor expressions
        elif isinstance(node, ConstructorExpression):
            fields: Record = {}
            for field_assign in node.fields:
                field_value = self.eval(field_assign.value)
                fields[field_assign.field_name] = field_value
            return {"_constructor": node.constructor_name, **fields}

        # Grouping
        elif isinstance(node, GroupedExpression):
            return self.eval(node.expression)

        else:
            raise NotImplementedError(
                f"Unhandled expression type: {type(node).__name__}",
            )

    def eval_do_block(self, statements: List[Statement]) -> Value:
        """Evaluate a do block with statements."""
        result: Value = None
        for stmt in statements:
            if isinstance(stmt, LetStatement):
                value = self.eval(stmt.value)
                self.variables[stmt.variable] = value
            elif hasattr(stmt, "__dict__") and any(
                isinstance(stmt, cls)
                for cls in [
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
                    PowIntOperation,
                    PowFloatOperation,
                    EqualOperation,
                    NotEqualOperation,
                    LessThanOperation,
                    LessEqualOperation,
                    GreaterThanOperation,
                    GreaterEqualOperation,
                    AndOperation,
                    OrOperation,
                    NotOperation,
                    ConcatOperation,
                    IndexOperation,
                    IfElse,
                    DoBlock,
                    FunctionApplication,
                    ConstructorExpression,
                    GroupedExpression,
                    NegativeInt,
                    NegativeFloat,
                ]
            ):
                # It's an expression
                result = self.eval(stmt)  # type: ignore
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
        def curried_function(*args):
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

            def partial_function(next_arg):
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
        if isinstance(pattern, VariablePattern):
            # Variable pattern always matches and binds the value
            self.variables[pattern.name] = value
            return True

        elif isinstance(pattern, LiteralPattern):
            # Literal pattern must match exactly
            return pattern.value == value

        elif isinstance(pattern, ListLiteral):
            # Match list literal pattern against list value
            if not isinstance(value, list):
                return False
            if len(pattern.elements) != len(value):
                return False
            # For list literals as patterns, evaluate elements and compare values
            for pattern_elem, value_elem in zip(pattern.elements, value):
                pattern_value = self.eval(pattern_elem)
                if pattern_value != value_elem:
                    return False
            return True

        elif isinstance(pattern, ConstructorPattern):
            # Match constructor pattern
            if not isinstance(value, dict) or "_constructor" not in value:
                return False

            if value["_constructor"] != pattern.constructor:
                return False

            # Match sub-patterns against constructor fields
            # For record constructors like Person {id_: Int, name: String},
            # the patterns will match against the field values
            if len(pattern.patterns) == 0:
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

            if len(pattern.patterns) != len(field_values):
                return False

            # Match each pattern against its corresponding field value
            for sub_pattern, field_value in zip(pattern.patterns, field_values):
                if not self._match_pattern(sub_pattern, field_value):
                    return False

            return True

        elif isinstance(pattern, ConsPattern):
            # Match cons pattern (head : tail)
            if not isinstance(value, list) or len(value) == 0:
                return False

            head = value[0]
            tail = value[1:]

            return self._match_pattern(pattern.head, head) and self._match_pattern(
                pattern.tail,
                tail,
            )

        else:
            raise NotImplementedError(
                f"Unhandled pattern type: {type(pattern).__name__}",
            )
