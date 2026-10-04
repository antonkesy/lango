from lango.minio.typechecker.infer import TypeEnvironment, type_check_ast
from lango.shared.ast.nodes import Program
from lango.shared.typechecker.lango_types import normalize_type_scheme


def type_check(ast: Program) -> TypeEnvironment:
    """Type check a program; raises ``TypeInferenceError`` if it is ill-typed."""
    return type_check_ast(ast)


def get_type_str(ast: Program) -> str:
    try:
        env = type_check_ast(ast)
    except Exception as e:
        return f"Type checking failed: {e}"
    return "".join(
        f"  {name} :: {normalize_type_scheme(scheme)}\n" for name, scheme in env.items()
    )
