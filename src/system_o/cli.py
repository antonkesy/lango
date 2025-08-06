import typer
from rich.console import Console

from system_o.parser.parser import example

app = typer.Typer()
console = Console()


@app.command()
def text(
    input_file: str = typer.Option(
        "",
        "--input_file",
        "-i",
        help="TODO",
    ),
):
    # TODO
    console.print("TODO")


def main():
    example()
    return 0
