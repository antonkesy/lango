import typer
from rich.console import Console

from lango.minio.ast_printer import print_annotated_ast
from lango.minio.interpreter import interpret
from lango.minio.parser import parse
from lango.minio.typecheck import get_type_str, type_check

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
    """Run a Minio program"""
    ast = parse(input_file)
    if enable_type_check:
        # Use new AST-based type checker
        if not type_check(ast):
            print("Type checking failed, cannot interpret.")
            return 1

    # Use new AST-based interpreter
    interpret(ast)
    # TODO: add return value handling
    return 0


@app.command()
def types(
    input_file: str = typer.Argument(
        help="Path to .minio file to type check",
    ),
):
    print(get_type_str(parse(input_file)))


@app.command()
def ast(
    input_file: str = typer.Argument(
        help="Path to .minio file to show annotated AST",
    ),
):
    """Show AST with type annotations after type checking"""

    ast = parse(input_file)
    type_check(ast)  # This will annotate the AST
    print_annotated_ast(ast)


@app.command()
def typecheck(
    input_file: str = typer.Argument(
        help="Path to .minio file to type check",
    ),
) -> int:
    """Type check a Minio program"""
    if type_check(parse(input_file)):
        console.print("Type checking succeeded", style="bold green")
        return 0
    else:
        console.print("Type checking failed", style="bold red")
        return 1


def main():
    return app()
