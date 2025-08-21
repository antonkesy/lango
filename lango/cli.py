import typer
from rich.console import Console

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
):
    """Run a Minio program"""
    tree = parse(input_file)
    if enable_type_check:
        if not type_check(tree):
            print("Type checking failed, cannot interpret.")
            return
    interpret(tree)


@app.command()
def types(
    input_file: str = typer.Argument(
        help="Path to .minio file to type check",
    ),
):
    tree = parse(input_file)
    print(get_type_str(tree))


@app.command()
def typecheck(
    input_file: str = typer.Argument(
        help="Path to .minio file to type check",
    ),
):
    """Type check a Minio program"""
    tree = parse(input_file)
    if type_check(tree):
        console.print("Type checking succeeded", style="bold green")
    else:
        console.print("Type checking failed", style="bold red")


def main():
    app()
    return 0
