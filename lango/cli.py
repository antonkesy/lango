import pprint
from pathlib import Path
from typing import Any, Union

import typer
from rich.console import Console

from lango.minio.ast.nodes import Program as MinioProgram
from lango.minio.compiler.go import compile_program as minio_go_compile_program
from lango.minio.compiler.python import compile_program as minio_python_compile_program
from lango.minio.interpreter.interpreter import interpret as minio_interpret
from lango.minio.parser.parser import parse as minio_parse
from lango.minio.typechecker.typecheck import get_type_str as minio_get_type_str
from lango.minio.typechecker.typecheck import type_check as minio_type_check
from lango.systemo.ast.nodes import Program as SystemoProgram
from lango.systemo.compiler.python import (
    compile_program as systemo_python_compile_program,
)
from lango.systemo.interpreter.interpreter import interpret as systemo_interpret
from lango.systemo.overloaded import collect_all_functions
from lango.systemo.parser.parser import parse as systemo_parse
from lango.systemo.typechecker.typecheck import get_type_str as systemo_get_type_str
from lango.systemo.typechecker.typecheck import type_check as systemo_type_check

app = typer.Typer(pretty_exceptions_enable=False)
console = Console()


@app.command()
def parse(
    lang: str = typer.Argument(..., help="systemo|minio"),
    input_file: Path = typer.Argument(..., exists=True, help="Path to input file"),
) -> int:
    ast: Union[SystemoProgram, MinioProgram]
    match lang:
        case "systemo":
            ast = systemo_parse(input_file)
            systemo_type_check(ast)
        case "minio":
            ast = minio_parse(input_file)
            minio_type_check(ast)
        case _:
            console.print(f"Unknown language: {lang}", style="bold red")
            return 1
    console.print(ast)
    return 0


@app.command()
def functions(
    input_file: Path = typer.Argument(..., exists=True, help="Path to input file"),
) -> int:
    ast = systemo_parse(input_file)

    functions = collect_all_functions(ast)
    pprint.pprint(functions, width=120, depth=10)
    return 0


@app.command()
def run(
    lang: str = typer.Argument(..., help="systemo|minio"),
    input_file: Path = typer.Argument(..., exists=True, help="Path to input file"),
) -> int:
    match lang:
        case "systemo":
            systemo_ast = systemo_parse(input_file)
            return systemo_interpret(systemo_ast).exit_code
        case "minio":
            minio_ast = minio_parse(input_file)
            return minio_interpret(minio_ast).exit_code
        case _:
            console.print(f"Unknown language: {lang}", style="bold red")
            return 1


@app.command()
def types(
    lang: str = typer.Argument(..., help="systemo|minio"),
    input_file: Path = typer.Argument(..., exists=True, help="Path to input file"),
) -> int:
    match lang:
        case "systemo":
            print(systemo_get_type_str(systemo_parse(input_file)))
        case "minio":
            print(minio_get_type_str(minio_parse(input_file)))
        case _:
            console.print(f"Unknown language: {lang}", style="bold red")
            return -1
    return 0


@app.command()
def typecheck(
    lang: str = typer.Argument(..., help="systemo|minio"),
    input_file: Path = typer.Argument(..., exists=True, help="Path to input file"),
) -> int:
    match lang:
        case "systemo":
            try:
                systemo_type_check(systemo_parse(input_file))
                console.print("Type checking succeeded", style="bold green")
                return 0
            except Exception:
                console.print("Type checking failed", style="bold red")
                return 1
        case "minio":
            try:
                minio_type_check(minio_parse(input_file))
                console.print("Type checking succeeded", style="bold green")
                return 0
            except Exception:
                console.print("Type checking failed", style="bold red")
                return 1
        case _:
            console.print(f"Unknown language: {lang}", style="bold red")
            return 1


@app.command()
def compile(
    lang: str = typer.Argument(..., help="systemo|minio"),
    input_file: Path = typer.Argument(..., exists=True, help="Path to input file"),
    output_file: str = typer.Option(
        "",
        "--output",
        "-o",
        help="Output compiled target file path",
    ),
    target: str = typer.Option(
        "python",
        "--target",
        "-t",
        help="Target language (python|go)",
    ),
) -> int:

    compiled_code: str
    match lang:
        case "systemo":
            systemo_ast = systemo_parse(input_file)
            systemo_type_check(systemo_ast)
            match target:
                case "python":
                    compile_program = systemo_python_compile_program
                case _:
                    console.print(
                        f"[red]Error: Target '{target}' not supported for systemo.[/red]",
                    )
                    return 1
            compiled_code = compile_program(systemo_ast)
        case "minio":
            minio_ast = minio_parse(input_file)
            minio_type_check(minio_ast)
            match target:
                case "python":
                    compile_program = minio_python_compile_program
                case "go":
                    compile_program = minio_go_compile_program
                case _:
                    console.print(f"Unknown target: {target}", style="bold red")
                    return 1

            compiled_code = compile_program(minio_ast)
        case _:
            console.print(f"Unknown language: {lang}", style="bold red")
            return 1

    if not output_file:
        output_file = f"out.{'py' if target == 'python' else 'go'}"

    with open(output_file, "w") as f:
        f.write(compiled_code)

    console.print(f"Compiled {input_file} to {output_file}", style="bold green")
    return 0


def main() -> Any:
    return app()
