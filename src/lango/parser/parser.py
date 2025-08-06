import io
from contextlib import redirect_stdout

from lark import Lark, Token, Transformer, Tree


def collect_functions(tree):
    env = {}

    def visit(node):
        if isinstance(node, Tree):
            if node.data == "func_def":
                func_name = node.children[0].value
                patterns = node.children[1:-2]  # if using pattern* as children
                expr = node.children[-1]
                env[func_name] = ("lambda", patterns, expr)
            for child in node.children:
                visit(child)

    visit(tree)
    return env


builtins = {
    "putStrLn": lambda x: print(x),
    "putStr": lambda x: print(x, end=""),
    "getLine": lambda: input(),
    "readInt": lambda: int(input()),
    "readString": lambda: input(),
    "readBool": lambda: input().lower() == "true",
    "add": lambda x, y: x + y,
    "concat": lambda x, y: x + y,
    "length": lambda x: len(x),
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
                case _:
                    raise NotImplementedError(f"Unhandled expression: {node.data}")
        elif isinstance(node, Token):
            return node.value
        else:
            raise TypeError(f"Unknown node type: {type(node)}")

    def eval_func(self, name):
        kind, patterns, expr = self.env[name]
        if kind != "lambda":
            raise RuntimeError("Only lambdas supported")
        if len(patterns) == 0:
            return self.eval(expr)
        else:

            def fn(arg):
                self.variables[patterns[0].children[0].value] = arg
                return self.eval(expr)

            return fn

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
    path: str = "./test/files/minio/math/logical/not.minio",
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
        f = io.StringIO()
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
