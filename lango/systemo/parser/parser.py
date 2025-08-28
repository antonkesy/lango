import os
from collections import defaultdict
from pathlib import Path

from lark import Lark, ParseTree

from lango.systemo.ast.nodes import FunctionDefinition, Program
from lango.systemo.ast.precedence_rewriter import rewrite_precedence
from lango.systemo.ast.transformer import transform_parse_tree


def _parse_lark(path: Path) -> ParseTree:
    parser = Lark.open(
        "./lango/systemo/parser/systemo.lark",
        parser="earley",
    )

    prelude_dir = "./lango/systemo/prelude"
    prelude_content = ""

    if os.path.exists(prelude_dir):
        for filename in sorted(os.listdir(prelude_dir)):
            if filename.endswith(".syso"):
                prelude_file_path = os.path.join(prelude_dir, filename)
                try:
                    with open(prelude_file_path, "r") as prelude_file:
                        prelude_content += prelude_file.read() + "\n"
                except FileNotFoundError:
                    pass

    with open(path) as f:
        main_content = f.read()

    return parser.parse(main_content + prelude_content)


def parse(path: Path) -> Program:
    program = transform_parse_tree(_parse_lark(path))
    program = rewrite_precedence(program)

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
