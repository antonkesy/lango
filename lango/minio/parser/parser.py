from collections import defaultdict
from pathlib import Path

from lango.minio.ast.nodes import FunctionDefinition, Program
from lango.minio.ast.transformer import transform_parse_tree
from lango.shared.parser import parse_lark


def parse(path: Path) -> Program:
    program = transform_parse_tree(
        parse_lark(
            path,
            grammar=Path("./lango/minio/parser/minio.lark"),
            prelude_dir=Path("./lango/minio/prelude"),
            file_extension="minio",
        ),
    )

    _validate_program(program)

    return program


def _validate_program(program: Program) -> None:
    # at least one main
    if not any(
        isinstance(stmt, FunctionDefinition) and stmt.function_name == "main"
        for stmt in program.statements
    ):
        raise RuntimeError("No main function defined")

    # only one main
    main_functions = [
        stmt
        for stmt in program.statements
        if isinstance(stmt, FunctionDefinition) and stmt.function_name == "main"
    ]
    if len(main_functions) > 1:
        raise RuntimeError("Multiple main functions defined")

    _validate_function_pattern_consistency(program)


def _validate_function_pattern_consistency(program: Program) -> None:
    """
    patterns of the same function must have same amount of variables for different matches
    t 0 = 1
    t 1 1 = 2 <- not allowed
    """

    function_patterns = defaultdict(list)

    for stmt in program.statements:
        if isinstance(stmt, FunctionDefinition):
            pattern_count = len(stmt.patterns)
            function_patterns[stmt.function_name].append(pattern_count)

    for function_name, pattern_counts in function_patterns.items():
        if len(set(pattern_counts)) > 1:
            unique_counts = sorted(set(pattern_counts))
            raise RuntimeError(
                f"Function '{function_name}' has inconsistent pattern counts: "
                f"found definitions with {unique_counts} parameters respectively. "
                f"All pattern matches for the same function must have the same number of parameters.",
            )
