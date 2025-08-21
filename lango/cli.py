import typer
from rich.console import Console

from lango.minio.interpreter import interpret
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
    if enable_type_check:
        # Use new AST-based type checker
        if not type_check(input_file):
            print("Type checking failed, cannot interpret.")
            return 1

    # Use new AST-based interpreter
    interpret(input_file)
    # TODO: add return value handling
    return 0


@app.command()
def types(
    input_file: str = typer.Argument(
        help="Path to .minio file to type check",
    ),
):
    print(get_type_str(input_file))


@app.command()
def typecheck(
    input_file: str = typer.Argument(
        help="Path to .minio file to type check",
    ),
) -> int:
    """Type check a Minio program"""
    if type_check(input_file):
        console.print("Type checking succeeded", style="bold green")
        return 0
    else:
        console.print("Type checking failed", style="bold red")
        return 1


def main():
    return app()
