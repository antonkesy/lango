"""
Type checker that works with custom AST nodes instead of raw Lark objects.
"""

from .ast_nodes import Program
from .parser import parse
from .typechecker.infer_ast import type_check_ast as type_check_ast_impl


def get_type_str(file_path: str) -> str:
    """Get type information for a Minio program file."""
    res = ""

    try:
        ast = parse(file_path)
        type_env = type_check_ast_impl(ast)
        res += "Inferred types:\n"
        for name, scheme in type_env.items():
            res += f"  {name} :: {scheme}\n"
    except Exception as e:
        res += f"Type checking failed: {e}"
    return res


def type_check(file_path: str) -> bool:
    """Type check a Minio program file."""
    try:
        ast = parse(file_path)
        type_check_ast_impl(ast)
        return True
    except Exception as e:
        print(f"Type checking failed: {e}")
        return False


def type_check_ast(ast: Program) -> bool:
    """Type check a Minio program AST."""
    try:
        type_check_ast_impl(ast)
        return True
    except Exception as e:
        print(f"Type checking failed: {e}")
        return False
