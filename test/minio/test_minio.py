from typing import Any

import pytest

from lango.minio.compiler.go import compile_program as compile_to_go
from lango.minio.compiler.python import compile_program as compile_to_python

# from lango.minio.compiler.systemf import compile_program as compile_to_systemf
from lango.minio.interpreter.interpreter import interpret
from lango.minio.parser.parser import parse
from lango.minio.typechecker.typecheck import type_check

from ..utility.file_tester import file_test_output, file_test_type, get_all_test_files
from ..utility.runners.external import (  # run_systemf_code,
    run_go_code,
    run_haskell_file,
    run_python_code,
)

BASE_TEST_FILES_PATH = "./test/minio/files/"
EXAMPLE = "./examples/minio/example.minio"


@pytest.mark.parametrize("file_name", list(get_all_test_files(BASE_TEST_FILES_PATH)))
def test_interpreter(file_name: str) -> None:
    def run_interpreter(f: str) -> str:
        return interpret(parse(f), collectStdOut=True)

    file_test_output(file_name, run_interpreter)


@pytest.mark.parametrize("file_name", list(get_all_test_files(BASE_TEST_FILES_PATH)))
def test_python_compiler(file_name: str) -> None:
    def run_compiler_and_output(f: str) -> str:
        return run_python_code(compile_to_python(parse(f)))

    file_test_output(file_name, run_compiler_and_output)


# @pytest.mark.parametrize("file_name", list(get_all_test_files(BASE_TEST_FILES_PATH)))
# def test_systemf_compiler(file_name: str) -> None:
#     def run_compiler_and_output(f: str) -> str:
#         return run_systemf_code(compile_to_systemf(parse(f)))

#     file_test_output(file_name, run_compiler_and_output)


@pytest.mark.parametrize("file_name", list(get_all_test_files(BASE_TEST_FILES_PATH)))
def test_go_compiler(file_name: str) -> None:
    def run_compiler_and_output(f: str) -> str:
        return run_go_code(compile_to_go(parse(f)))

    file_test_output(file_name, run_compiler_and_output)


@pytest.mark.parametrize("file_name", list(get_all_test_files(BASE_TEST_FILES_PATH)))
def test_is_haskell_compliant(file_name: str) -> None:
    file_test_output(file_name, run_haskell_file)


@pytest.mark.parametrize(
    "file_name",
    list(get_all_test_files(BASE_TEST_FILES_PATH)) + [EXAMPLE],
)
def test_is_type_valid(file_name: str) -> None:
    def run_type_check(f: str) -> Any:
        return type_check(parse(f))

    file_test_type(file_name, run_type_check)
