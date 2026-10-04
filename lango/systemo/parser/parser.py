from pathlib import Path

from lango.shared.ast.nodes import FunctionDefinition, Program
from lango.shared.parser import parse_lark
from lango.systemo.ast.desugar import desugar_program
from lango.systemo.ast.transformer import transform_parse_tree

GRAMMAR = Path(__file__).with_name("systemo.lark")
PRELUDE_DIR = Path(__file__).parents[1] / "prelude"


def parse(path: Path) -> Program:
    program = transform_parse_tree(
        parse_lark(
            path,
            GRAMMAR,
            PRELUDE_DIR,
            file_extension="syso",
            prelude_first=True,
            # rebinding a unique variable is a type error (checked by the type checker)
            check_prelude_conflicts=False,
        ),
    )
    program = desugar_program(program)
    _validate_program(program)
    return program


def _validate_program(program: Program) -> None:
    mains = [
        stmt
        for stmt in program.statements
        if isinstance(stmt, FunctionDefinition) and stmt.function_name == "main"
    ]
    if not mains:
        raise RuntimeError("No main function defined")
    if len(mains) > 1:
        raise RuntimeError("Multiple main functions defined")
    _validate_function_clauses(program)


def _validate_function_clauses(program: Program) -> None:
    """The clauses of a function must be contiguous and of equal arity.

    A unique variable is bound at most once in a System O program; the
    clauses of one function together form its single (recursive) binding.
    """
    seen: list[str] = []
    previous: FunctionDefinition | None = None
    for stmt in program.statements:
        if not isinstance(stmt, FunctionDefinition):
            previous = None
            continue
        if previous is not None and previous.function_name == stmt.function_name:
            if len(previous.patterns) != len(stmt.patterns):
                raise RuntimeError(
                    f"Function '{stmt.function_name}' has clauses with "
                    f"{len(previous.patterns)} and {len(stmt.patterns)} parameters",
                )
        elif stmt.function_name in seen:
            raise RuntimeError(
                f"Function '{stmt.function_name}' is defined more than once",
            )
        else:
            seen.append(stmt.function_name)
        previous = stmt
