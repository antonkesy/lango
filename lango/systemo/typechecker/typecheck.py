from lango.systemo.ast.nodes import Program
from lango.systemo.typechecker.infer import type_check_ast as type_check_ast_impl
from lango.systemo.typechecker.systemo_types import normalize_type_scheme


def get_type_str(ast: Program) -> str:
    res = ""

    try:
        type_env = type_check_ast_impl(ast)
        for name, scheme in type_env.items():
            normalized_scheme = normalize_type_scheme(scheme)
            res += f"  {name} :: {normalized_scheme}\n"
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
