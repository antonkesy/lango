"""The ``lango`` command line interface."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, NoReturn

import typer
from lark.exceptions import LarkError
from rich.console import Console
from rich.markup import escape

from lango.minio.compiler.go import compile_program as minio_go_compile
from lango.minio.compiler.python import compile_program as minio_python_compile
from lango.minio.interpreter.interpreter import interpret as minio_interpret
from lango.minio.parser.parser import parse as minio_parse
from lango.minio.typechecker.typecheck import get_type_str as minio_get_type_str
from lango.minio.typechecker.typecheck import type_check as minio_type_check
from lango.shared.ast.nodes import Program
from lango.shared.typechecker.errors import TypeInferenceError
from lango.systemo.compiler.python.codegen import Strategy
from lango.systemo.compiler.python.dictionary_passing import (
    compile_program as systemo_dp_compile,
)
from lango.systemo.compiler.python.monomorphization import (
    compile_program as systemo_mono_compile,
)
from lango.systemo.interpreter.interpreter import interpret as systemo_interpret
from lango.systemo.parser.parser import parse as systemo_parse
from lango.systemo.typechecker.infer import InstanceInfo
from lango.systemo.typechecker.typecheck import get_type_str as systemo_get_type_str
from lango.systemo.typechecker.typecheck import type_check as systemo_type_check
from lango.systemo.typechecker.types import display_name, scheme_to_str


class Lang(StrEnum):
    SYSTEMO = "systemo"
    MINIO = "minio"


class Target(StrEnum):
    PYTHON = "python"
    GO = "go"


EXTENSIONS = {Target.PYTHON: "py", Target.GO: "go"}

type Compiler = Callable[[Program], str]


@dataclass(frozen=True)
class Language:
    parse: Callable[[Path], Program]
    type_check: Callable[[Program], Any]
    interpret: Callable[[Program], Any]
    type_str: Callable[[Program], str]


LANGUAGES = {
    Lang.MINIO: Language(
        minio_parse, minio_type_check, minio_interpret, minio_get_type_str
    ),
    Lang.SYSTEMO: Language(
        systemo_parse,
        systemo_type_check,
        systemo_interpret,
        systemo_get_type_str,
    ),
}
MINIO_COMPILERS: dict[Target, Compiler] = {
    Target.PYTHON: minio_python_compile,
    Target.GO: minio_go_compile,
}
SYSTEMO_COMPILERS: dict[Strategy, Compiler] = {
    Strategy.MONOMORPHIZATION: systemo_mono_compile,
    Strategy.DICTIONARY_PASSING: systemo_dp_compile,
}

app = typer.Typer(pretty_exceptions_enable=False)
console = Console()

LANG = typer.Argument(..., help="systemo|minio")
INPUT_FILE = typer.Argument(..., exists=True, help="Path to input file")


def _fail(message: str) -> NoReturn:
    console.print(escape(message), style="bold red")
    raise typer.Exit(1)


def _load(lang: Lang, input_file: Path) -> Program:
    """Parse and type check; a type error ends the program with exit code 1."""
    language = LANGUAGES[lang]
    try:
        program = language.parse(input_file)
    except (LarkError, RuntimeError, ValueError) as e:
        _fail(f"Parse error: {e}")
    try:
        language.type_check(program)
    except TypeInferenceError as e:
        _fail(f"Type checking failed: {e}")
    return program


@app.command()
def parse(lang: Lang = LANG, input_file: Path = INPUT_FILE) -> None:
    """Print the (typed) abstract syntax tree of a program."""
    console.print(_load(lang, input_file))


@app.command()
def functions(input_file: Path = INPUT_FILE) -> None:
    """List the overloaded identifiers of a SystemO program and their instances."""
    typed = systemo_type_check(systemo_parse(input_file))
    instances: dict[str, list[str]] = {}
    for decl in typed.decls:
        if isinstance(decl, InstanceInfo):
            instances.setdefault(decl.name, []).append(scheme_to_str(decl.scheme))
    for name, schemes in instances.items():
        console.print(escape(display_name(name)))
        for scheme in schemes:
            console.print(escape(f"  inst {display_name(name)} :: {scheme}"))


@app.command()
def run(lang: Lang = LANG, input_file: Path = INPUT_FILE) -> None:
    """Interpret a program."""
    result = LANGUAGES[lang].interpret(_load(lang, input_file))
    if result.exit_code:
        raise typer.Exit(result.exit_code)


@app.command()
def types(lang: Lang = LANG, input_file: Path = INPUT_FILE) -> None:
    """Print the inferred types of the top-level declarations."""
    language = LANGUAGES[lang]
    print(language.type_str(language.parse(input_file)))


@app.command()
def typecheck(lang: Lang = LANG, input_file: Path = INPUT_FILE) -> None:
    """Type check a program."""
    _load(lang, input_file)
    console.print("Type checking succeeded", style="bold green")


@app.command()
def compile(
    lang: Lang = LANG,
    input_file: Path = INPUT_FILE,
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output compiled target file path",
    ),
    target: Target = typer.Option(
        Target.PYTHON,
        "--target",
        "-t",
        help="Target language",
    ),
    strategy: Strategy | None = typer.Option(
        None,
        "--strategy",
        "-s",
        help="SystemO compilation strategy",
    ),
) -> None:
    """Compile a program to Python (or, for MiniO, Go)."""
    compiler: Compiler
    match lang:
        case Lang.SYSTEMO:
            if target != Target.PYTHON:
                _fail(f"Error: Target '{target}' not supported for systemo.")
            if strategy is None:
                _fail("Error: systemo needs a compilation strategy (--strategy).")
            compiler = SYSTEMO_COMPILERS[strategy]
        case Lang.MINIO:
            if strategy is not None:
                _fail(f"Error: Strategy '{strategy}' not supported for minio.")
            compiler = MINIO_COMPILERS[target]

    compiled_code = compiler(_load(lang, input_file))
    output = output_file or Path(f"out.{EXTENSIONS[target]}")
    output.write_text(compiled_code)
    console.print(f"Compiled {input_file} to {output}", style="bold green")


def main() -> None:
    app()
