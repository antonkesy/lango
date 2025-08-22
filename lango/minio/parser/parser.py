import os

from lark import Lark, ParseTree

from lango.minio.ast.nodes import FunctionDefinition, Program
from lango.minio.ast.transformer import transform_parse_tree


def _parse_lark(path: str) -> ParseTree:
    parser = Lark.open(
        "./lango/minio/parser/minio.lark",
        parser="lalr",
    )

    prelude_dir = "./lango/minio/prelude"
    prelude_content = ""

    if os.path.exists(prelude_dir):
        for filename in sorted(os.listdir(prelude_dir)):
            if filename.endswith(".minio"):
                prelude_file_path = os.path.join(prelude_dir, filename)
                try:
                    with open(prelude_file_path, "r") as prelude_file:
                        prelude_content += prelude_file.read() + "\n"
                except FileNotFoundError:
                    pass

    with open(path) as f:
        main_content = f.read()

    return parser.parse(main_content + prelude_content)


def parse(path: str) -> Program:
    program = transform_parse_tree(_parse_lark(path))

    has_main = any(
        isinstance(stmt, FunctionDefinition) and stmt.function_name == "main"
        for stmt in program.statements
    )

    if not has_main:
        raise RuntimeError(f"No main function defined in {path}")

    return program
