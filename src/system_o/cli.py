import typer
from rich.console import Console

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
    console.print(f"Input file: {input_file}")


if __name__ == "__main__":
    app()
