from typing import Any

import typer
from rich.console import Console

from lango.minio.ast.printer import (
    print_annotated_ast,
    print_ast_colored,
    print_ast_compact,
    print_ast_types_only,
)
from lango.minio.compiler.go import compile_program as go_compile_program
from lango.minio.compiler.python import compile_program as python_compile_program
from lango.minio.interpreter.interpreter import interpret
from lango.minio.parser.parser import parse
from lango.minio.typechecker.typecheck import get_type_str, type_check

app = typer.Typer()
console = Console()


@app.command()
def run(
    input_file: str = typer.Option(
        "examples/minio/example.minio",
        "--input_file",
        "-i",
        help="Path to .minio file to run",
    ),
    enable_type_check: bool = typer.Option(
        True,
        "--type_check/--no_type_check",
        help="Enable or disable type checking",
    ),
) -> int:
    ast = parse(input_file)
    if enable_type_check:
        if not type_check(ast):
            print("Type checking failed, cannot interpret.")
            return 1

    return interpret(ast).exit_code


@app.command()
def types(
    input_file: str = typer.Argument(
        help="Path to .minio file to type check",
    ),
) -> None:
    print(get_type_str(parse(input_file)))


@app.command()
def ast(
    input_file: str = typer.Argument(
        help="Path to .minio file to show annotated AST",
    ),
    mode: str = typer.Option(
        "full",
        "--mode",
        "-m",
        help="AST display mode: full, compact, summary, types-only, colored",
    ),
    max_depth: int = typer.Option(
        None,
        "--max-depth",
        "-d",
        help="Maximum depth to display (for summary mode)",
    ),
    show_types: bool = typer.Option(
        True,
        "--types/--no-types",
        help="Show type annotations",
    ),
) -> None:
    ast_parsed = parse(input_file)
    if not type_check(ast_parsed):
        console.print("Type checking failed, cannot print AST.", style="bold red")
        return

    match mode:
        case "full":
            print_annotated_ast(ast_parsed, show_types=show_types)
        case "compact":
            print_ast_compact(ast_parsed, show_types=show_types)
        case "summary":
            depth = max_depth if max_depth is not None else 3
            print_annotated_ast(ast_parsed, show_types=show_types, max_depth=depth)
        case "types-only":
            print_ast_types_only(ast_parsed)
        case "colored":
            print_ast_colored(ast_parsed, show_types=show_types, compact=False)
        case _:
            console.print(
                f"Unknown mode: {mode}. Use: full, compact, summary, types-only, or colored",
                style="bold red",
            )


@app.command()
def typecheck(
    input_file: str = typer.Argument(
        help="Path to .minio file to type check",
    ),
) -> int:
    if type_check(parse(input_file)):
        console.print("Type checking succeeded", style="bold green")
        return 0
    else:
        console.print("Type checking failed", style="bold red")
        return 1


@app.command()
def compile(
    input_file: str = typer.Argument(
        help="Path to .minio file to compile to Python",
    ),
    output_file: str = typer.Option(
        "out.py",
        "--output",
        "-o",
        help="Output Python file path",
    ),
    target: str = typer.Option(
        "python",
        "--target",
        "-t",
        help="Target language (python|go)",
    ),
) -> int:
    try:
        ast = parse(input_file)
        type_check(ast)
        match target:
            case "python":
                compile_program = python_compile_program
            case "go":
                compile_program = go_compile_program
            case _:
                console.print(f"Unknown target: {target}", style="bold red")
                return 1
        compiled_code = compile_program(ast)

        with open(output_file, "w") as f:
            f.write(compiled_code)

        console.print(f"Compiled {input_file} to {output_file}", style="bold green")
        return 0

    except Exception as e:
        console.print(f"Compilation failed: {e}", style="bold red")
        return 1


def main() -> Any:
    return app()
