import os

from lark import Lark, ParseTree

from .ast_nodes import Program
from .ast_transformer import transform_parse_tree


def parse_lark(path: str) -> ParseTree:
    """Parse file and return raw Lark ParseTree (for backward compatibility)."""
    parser = Lark.open(
        "./lango/minio/minio.lark",
        parser="lalr",
    )

    # Load prelude files
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
                    pass  # Skip missing files

    # Read main file
    with open(path) as f:
        main_content = f.read()

    # Combine prelude and main content (prelude first)
    combined_content = prelude_content + main_content

    return parser.parse(combined_content)


def parse_string(code: str) -> Program:
    """Parse code string and return custom AST."""
    parser = Lark.open(
        "./lango/minio/minio.lark",
        parser="lalr",
    )

    # Load prelude files
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
                    pass  # Skip missing files

    # Combine prelude and main content (prelude first)
    combined_content = prelude_content + code

    parse_tree = parser.parse(combined_content)
    # Transform to custom AST
    return transform_parse_tree(parse_tree)


def parse(path: str) -> Program:
    """Parse file and return custom AST."""
    # First get the Lark parse tree
    parse_tree = parse_lark(path)
    # Then transform to custom AST
    return transform_parse_tree(parse_tree)
