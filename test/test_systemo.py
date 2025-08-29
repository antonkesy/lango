from pathlib import Path
from typing import Any

import pytest

from lango.systemo.compiler.python import compile_program as compile_to_python
from lango.systemo.interpreter.interpreter import interpret
from lango.systemo.parser.parser import parse
from lango.systemo.typechecker.typecheck import type_check

from .utility.file_tester import file_test_output, file_test_type, get_all_test_files
from .utility.runners.external import run_python_code

MINIO_BASE_TEST_FILES_PATH = Path("./test/files/minio/")
SYSO_BASE_TEST_FILES_PATH = Path("./test/files/systemo/")
EXAMPLE = Path("./examples/systemo/example.syso")


@pytest.mark.parametrize(
    "file_name",
    list(get_all_test_files(SYSO_BASE_TEST_FILES_PATH, "syso")),
    ids=lambda p: str(p),
)
def test_python_compiler(file_name: Path) -> None:
    def run_compiler_and_output(f: Path) -> str:
        ast = parse(f)
        type_check(ast)
        return run_python_code(compile_to_python(ast))

    file_test_output(file_name, run_compiler_and_output)


@pytest.mark.parametrize(
    "file_name",
    list(get_all_test_files(MINIO_BASE_TEST_FILES_PATH, "minio")),
    ids=lambda p: str(p),
)
def test_interpreter_is_superset_of_minio(file_name: Path) -> None:
    def run_interpreter(f: Path) -> str:
        return interpret(parse(f), collectStdOut=True).output

    file_test_output(file_name, run_interpreter)


@pytest.mark.parametrize(
    "file_name",
    list(get_all_test_files(SYSO_BASE_TEST_FILES_PATH, "syso")),
    ids=lambda p: str(p),
)
def test_interpreter(file_name: Path) -> None:
    def run_interpreter(f: Path) -> str:
        return interpret(parse(f), collectStdOut=True).output

    file_test_output(file_name, run_interpreter)


@pytest.mark.parametrize(
    "file_name",
    list(get_all_test_files(MINIO_BASE_TEST_FILES_PATH, "minio")),
    ids=lambda p: str(p),
)
def test_types_is_superset_of_minio(file_name: Path) -> None:
    def run_type_check(f: Path) -> Any:
        return type_check(parse(f))

    file_test_type(file_name, run_type_check)


@pytest.mark.parametrize(
    "file_name",
    list(get_all_test_files(SYSO_BASE_TEST_FILES_PATH, "syso")) + [EXAMPLE],
    ids=lambda p: str(p),
)
def test_is_type_valid(file_name: Path) -> None:
    def run_type_check(f: Path) -> Any:
        return type_check(parse(f))

    file_test_type(file_name, run_type_check)
