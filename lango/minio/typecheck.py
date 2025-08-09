from lark import ParseTree

from lango.minio.typechecker.infer import type_check as type_check_infer


def get_type_str(tree: ParseTree) -> str:
    res = ""

    try:
        type_env = type_check_infer(tree)
        res += "Inferred types:\n"
        for name, scheme in type_env.items():
            res += f"  {name} :: {scheme}\n"
    except Exception as e:
        res += f"Type checking failed: {e}"
    return res


def type_check(tree: ParseTree) -> bool:
    try:
        type_check_infer(tree)
        return True
    except Exception as e:
        print(f"Type checking failed: {e}")
        return False
