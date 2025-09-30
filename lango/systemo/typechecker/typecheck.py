from lango.shared.ast.nodes import Program
from lango.systemo.typechecker.infer import FunctionDecl, TypedProgram, infer_program
from lango.systemo.typechecker.types import display_name, scheme_to_str


def type_check(ast: Program) -> TypedProgram:
    """Type check a program; raises ``TypeInferenceError`` if it is ill-typed."""
    return infer_program(ast)


def get_type_str(ast: Program) -> str:
    typed = infer_program(ast)
    lines = []
    for decl in typed.decls:
        match decl:
            case FunctionDecl(name=name, scheme=scheme):
                lines.append(f"  {name} :: {scheme_to_str(scheme)}")
            case _:
                lines.append(
                    f"  inst {display_name(decl.name)} :: {scheme_to_str(decl.scheme)}",
                )
    return "\n".join(lines) + "\n"
