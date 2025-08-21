from lango.minio.ast_nodes import Program
from lango.minio.typechecker.infer_ast import type_check_ast as type_check_ast_impl


def get_type_str(ast: Program) -> str:
    res = ""

    try:
        type_env = type_check_ast_impl(ast)
        for name, scheme in type_env.items():
            res += f"  {name} :: {scheme}\n"
    except Exception as e:
        res += f"Type checking failed: {e}"
    return res


def type_check(ast: Program) -> bool:
    try:
        type_check_ast_impl(ast)
        return True
    except Exception as e:
        print(f"Type checking failed: {e}")
        return False
