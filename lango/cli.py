import typer
from rich.console import Console

from .parser.parser import example, type_check_file

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
    test: bool = typer.Option(
        False,
        "--test",
        "-t",
        help="Run in test mode (capture output)",
    ),
    check_types: bool = typer.Option(
        False,
        "--type-check",
        "-tc",
        help="Run type checker before execution",
    ),
):
    """Run a Minio program"""
    example(input_file, test, check_types)


@app.command()
def typecheck(
    input_file: str = typer.Argument(
        help="Path to .minio file to type check",
    ),
):
    """Type check a Minio program"""
    type_check_file(input_file)


def main():
    app()
    return 0
