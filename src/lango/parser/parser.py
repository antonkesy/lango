from contextlib import redirect_stdout
from io import StringIO

from lark import Lark, Token, Tree


def build_environment(tree):
    env = {}

    def visit(node):
        if isinstance(node, Tree):
            if node.data == "func_def":
                func_name = node.children[0].value
                patterns = node.children[
                    1:-1
                ]  # all children between func_name and expr
                expr = node.children[-1]

                # Support multiple function clauses for pattern matching
                if func_name not in env:
                    env[func_name] = ("pattern_match", [])

                # Add this clause to the function's clauses list
                env[func_name][1].append((patterns, expr))
            for child in node.children:
                visit(child)

    visit(tree)
    return env


def collect_functions(tree):
    env = {}

    def visit(node):
        if isinstance(node, Tree):
            if node.data == "func_def":
                func_name = node.children[0].value
                patterns = node.children[
                    1:-1
                ]  # all children between func_name and expr
                expr = node.children[-1]

                # Support multiple function clauses for pattern matching
                if func_name not in env:
                    env[func_name] = ("pattern_match", [])

                # Add this clause to the function's clauses list
                env[func_name][1].append((patterns, expr))
            for child in node.children:
                visit(child)

    visit(tree)
    return env


def flexible_putStrLn(arg):
    """putStrLn that can handle both single argument and curried usage"""
    if callable(arg):
        # If given a function (like 'printPerson'), return a curried function
        def curried_putStrLn(value):
            result = arg(value)  # Apply the function to the value
            print(result)
            return None  # Ensure we return None like the original

        return curried_putStrLn
    else:
        # If given a direct value, print it
        print(arg)
        return None


def flexible_putStr(arg):
    """putStr that can handle both single argument and curried usage"""
    if callable(arg):
        # If given a function (like 'show'), return a curried function
        def curried_putStr(value):
            result = arg(value)  # Apply the function to the value
            print(result, end="")
            return None  # Ensure we return None like the original

        return curried_putStr
    else:
        # If given a direct value, print it
        print(arg, end="")
        return None


builtins = {
    "putStrLn": flexible_putStrLn,
    "putStr": flexible_putStr,
    "getLine": lambda: input(),
    "readInt": lambda: int(input()),
    "readString": lambda: input(),
    "readBool": lambda: input().lower() == "true",
    "concat": lambda x, y: x + y,
    "toUpperCase": lambda x: x.upper(),
    "toLowerCase": lambda x: x.lower(),
    "show": lambda x: str(x),
}


class Interpreter:
    def __init__(self, env):
        self.env = env
        self.variables = {}

    def eval(self, node):
        if isinstance(node, Tree):
            match node.data:
                case "int":
                    return int(node.children[0])
                case "float":
                    return float(node.children[0])
                case "string":
                    return node.children[0][1:-1]
                case "true":
                    return True
                case "false":
                    return False
                case "var":
                    name = node.children[0].value
                    if name in self.variables:
                        return self.variables[name]
                    elif name in self.env:
                        return self.eval_func(name)
                    elif name in builtins:
                        return builtins[name]
                    else:
                        raise RuntimeError(f"Unknown variable: {name}")
                # Arithmetic
                case "add":
                    return self.eval(node.children[0]) + self.eval(node.children[2])
                case "sub":
                    return self.eval(node.children[0]) - self.eval(node.children[2])
                case "mul":
                    return self.eval(node.children[0]) * self.eval(node.children[2])
                case "div":
                    return self.eval(node.children[0]) / self.eval(node.children[2])
                case "pow":
                    return self.eval(node.children[0]) ** self.eval(node.children[2])
                case "mod":
                    return self.eval(node.children[0]) % self.eval(node.children[2])
                case "neg":
                    return -self.eval(node.children[0])
                case "quot":
                    return self.eval(node.children[0]) // self.eval(node.children[2])
                # TODO: rem, div?
                # Comparison
                case "eq":
                    return self.eval(node.children[0]) == self.eval(node.children[2])
                case "neq":
                    return self.eval(node.children[0]) != self.eval(node.children[2])
                case "lt":
                    return self.eval(node.children[0]) < self.eval(node.children[2])
                case "lteq":
                    return self.eval(node.children[0]) <= self.eval(node.children[2])
                case "gt":
                    return self.eval(node.children[0]) > self.eval(node.children[2])
                case "gteq":
                    return self.eval(node.children[0]) >= self.eval(node.children[2])
                # Logical
                case "and":
                    return self.eval(node.children[0]) and self.eval(node.children[2])
                case "or":
                    return self.eval(node.children[0]) or self.eval(node.children[2])
                case "not":
                    return not self.eval(node.children[1])
                # String
                case "concat":
                    return self.eval(node.children[0]) + self.eval(node.children[1])
                case "app":
                    # handle function application: f x
                    func_node = node.children[0]
                    arg_node = node.children[1]
                    func = self.eval(func_node)
                    arg = self.eval(arg_node)
                    return func(arg)
                case "do_block":
                    return self.eval_do_block(node.children)
                case "let":
                    var_name = node.children[0].value
                    value = self.eval(node.children[1])
                    self.variables[var_name] = value
                    return value
                case "stmt_list":
                    result = None
                    for stmt in node.children:
                        result = self.eval(stmt)
                    return result
                case "do_stmt":
                    # handle do statements
                    return self.eval_do_block(node.children)
                case "grouped":
                    # handle grouped expressions
                    return self.eval(node.children[0])
                case "constructor_expr":
                    # handle record constructor expressions like Person { id = 1, name = "Alice" }
                    constructor_name = node.children[0].value
                    fields = {}

                    # Process field assignments
                    for field_assign in node.children[1:]:
                        if field_assign.data == "field_assign":
                            field_name = field_assign.children[0].value
                            field_value = self.eval(field_assign.children[1])
                            fields[field_name] = field_value

                    # Return a dictionary representing the record
                    return {"_constructor": constructor_name, **fields}
                case _:
                    raise NotImplementedError(f"Unhandled expression: {node.data}")
        elif isinstance(node, Token):
            return node.value
        else:
            raise TypeError(f"Unknown node type: {type(node)}")

    def eval_func(self, name):
        env_data = self.env[name]

        # Handle both old format (kind, patterns, expr) and new format (kind, data)
        if len(env_data) == 3:
            # Old format: ("lambda", patterns, expr)
            kind, patterns, expr = env_data
            if kind != "lambda":
                raise RuntimeError("Only lambdas supported in old format")
            if len(patterns) == 0:
                return self.eval(expr)
            else:
                return self._create_curried_function([(patterns, expr)])
        elif len(env_data) == 2:
            # New format: ("pattern_match", clauses)
            kind, data = env_data
            if kind == "pattern_match":
                clauses = data
                return self._create_pattern_match_function(clauses)
            else:
                raise RuntimeError(f"Unknown function kind: {kind}")
        else:
            raise RuntimeError(f"Invalid environment data format: {env_data}")

    def _create_pattern_match_function(self, clauses):
        """Create a function that handles pattern matching across multiple clauses"""

        def pattern_match_fn(*args):
            # Try each clause in order
            for patterns, expr in clauses:
                if self._try_match_patterns(patterns, args):
                    # This clause matches, evaluate it with the matched bindings
                    old_variables = self.variables.copy()
                    try:
                        self._bind_patterns(patterns, args)
                        result = self.eval(expr)
                        return result
                    finally:
                        self.variables = old_variables

            # No clause matched
            raise RuntimeError(f"No pattern matched for arguments: {args}")

        # Create curried version for partial application
        def curried_fn(collected_args=[]):
            if not clauses:
                raise RuntimeError("No clauses defined")

            # Check if we have enough arguments for any clause
            expected_arity = len(clauses[0][0])  # Assume all clauses have same arity

            if len(collected_args) == expected_arity:
                return pattern_match_fn(*collected_args)
            else:
                # Return a function that collects the next argument
                def next_fn(arg):
                    return curried_fn(collected_args + [arg])

                return next_fn

        return curried_fn()

    def _create_curried_function(self, clauses):
        """Create a curried function that handles multiple parameters (backward compatibility)"""
        # Convert old format to new format
        return self._create_pattern_match_function(clauses)

    def _try_match_patterns(self, patterns, args):
        """Check if patterns match the given arguments"""
        if len(patterns) != len(args):
            return False

        for pattern, arg in zip(patterns, args):
            if not self._match_pattern(pattern, arg):
                return False
        return True

    def _match_pattern(self, pattern, arg):
        """Check if a single pattern matches an argument"""
        if isinstance(pattern, Token):
            # Variable pattern - always matches
            return True
        elif isinstance(pattern, Tree):
            if pattern.data == "constructor_pattern":
                # Constructor pattern like (Person id name)
                constructor_name = pattern.children[0].value

                # Check if the argument is a record with matching constructor
                if (
                    isinstance(arg, dict)
                    and "_constructor" in arg
                    and arg["_constructor"] == constructor_name
                ):

                    # Check if the pattern has the right number of fields
                    pattern_vars = pattern.children[1:]  # Skip constructor name
                    expected_fields = len(pattern_vars)
                    actual_fields = len(arg) - 1  # Subtract 1 for _constructor key

                    return expected_fields == actual_fields
                return False
            else:
                # Literal pattern - must match exactly
                pattern_value = self._get_pattern_value(pattern)
                return pattern_value == arg
        else:
            # Unknown pattern type
            return False

    def _bind_patterns(self, patterns, args):
        """Bind matched patterns to variables in the current scope"""
        for pattern, arg in zip(patterns, args):
            if isinstance(pattern, Token):
                # Variable pattern - bind to the argument value
                self.variables[pattern.value] = arg
            elif isinstance(pattern, Tree) and pattern.data == "constructor_pattern":
                # Constructor pattern like (Person id name)
                pattern_vars = pattern.children[1:]  # Skip constructor name

                # Extract field values from the record in the order they were defined
                if isinstance(arg, dict) and "_constructor" in arg:
                    # For now, assume fields are bound in the order they appear in the pattern
                    # This is a simplified approach - in a full implementation, you'd want
                    # to match field names properly
                    field_values = []
                    for key, value in arg.items():
                        if key != "_constructor":
                            field_values.append(value)

                    # Bind pattern variables to field values
                    for var_token, field_value in zip(pattern_vars, field_values):
                        if isinstance(var_token, Token):
                            self.variables[var_token.value] = field_value
            # Literal patterns don't bind anything

    def _get_pattern_value(self, pattern):
        """Extract the value from a pattern Tree node"""
        if pattern.data == "int":
            return int(pattern.children[0])
        elif pattern.data == "float":
            return float(pattern.children[0])
        elif pattern.data == "string":
            return pattern.children[0][1:-1]
        elif pattern.data == "true":
            return True
        elif pattern.data == "false":
            return False
        else:
            raise RuntimeError(f"Unknown literal pattern: {pattern.data}")

    def eval_do_block(self, stmts):
        result = None
        for stmt in stmts:
            if stmt.data == "let":
                var_name = stmt.children[0].value
                value = self.eval(stmt.children[1])
                self.variables[var_name] = value
            else:
                result = self.eval(stmt)
        return result


def example(
    path: str = "test/files/minio/datatypes/custom/named.minio",
    isTest: bool = False,
) -> str:
    parser = Lark.open(
        "./src/lango/parser/minio.lark",
        parser="lalr",
    )

    with open(path) as f:
        tree = parser.parse(f.read())

    # print(tree.pretty())
    # print("\n")

    env = collect_functions(tree)
    interp = Interpreter(env)

    if "main" not in env:
        raise RuntimeError("No main function defined")

    if isTest:
        f = StringIO()
        with redirect_stdout(f):
            result = interp.eval_func("main")
        output = f.getvalue()
    else:
        result = interp.eval_func("main")
        output = ""

    if not isTest:
        print(
            f"\n[main] => {result}" if not callable(result) else "[main] is a function",
        )
    return output
