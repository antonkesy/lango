from lango.shared.typechecker.lango_types import normalize_type_scheme
from lango.systemo.ast.nodes import Program
from lango.systemo.typechecker.infer import TypeInferrer
from lango.systemo.typechecker.infer import type_check_ast as type_check_ast_impl


def get_type_str(ast: Program) -> str:
    res = ""

    inferrer = TypeInferrer()
    type_env = inferrer.infer_program(ast)

    # Display regular functions and types
    for name, scheme in type_env.items():
        normalized_scheme = normalize_type_scheme(scheme)
        res += f"  {name} :: {normalized_scheme}\n"

    # Display overloaded functions
    for instance_name, instances in inferrer.instances.items():
        for instance_type, _ in instances:
            # Normalize the instance type
            from lango.systemo.typechecker.systemo_types import TypeScheme

            free_vars = instance_type.free_vars()
            scheme = TypeScheme(free_vars, instance_type)
            normalized_scheme = normalize_type_scheme(scheme)
            res += f"  {instance_name} :: {normalized_scheme}\n"
    return res


def type_check(ast: Program) -> bool:
    try:
        type_check_ast_impl(ast)
        return True
    except Exception as e:
        raise e
