import pytest

from lango.minio.compiler.python import compile_program
from lango.minio.interpreter.interpreter import interpret
from lango.minio.parser.parser import parse
from lango.minio.typechecker.typecheck import type_check

from ..utility.file_tester import file_test_output, file_test_type, get_all_test_files
from ..utility.runners.external import run_compiled_python, run_haskell_file

BASE_TEST_FILES_PATH = "./test/minio/files/"
EXAMPLE = "./examples/minio/example.minio"


@pytest.mark.parametrize("file_name", list(get_all_test_files(BASE_TEST_FILES_PATH)))
def test_interpreter(file_name):
    def run_interpreter(f):
        return interpret(parse(f), collectStdOut=True)

    file_test_output(file_name, run_interpreter)


@pytest.mark.parametrize("file_name", list(get_all_test_files(BASE_TEST_FILES_PATH)))
def test_compiler(file_name):
    def run_compiler_and_output(f):
        return run_compiled_python(compile_program(parse(f)))

    file_test_output(file_name, run_compiler_and_output)


@pytest.mark.parametrize("file_name", list(get_all_test_files(BASE_TEST_FILES_PATH)))
def test_is_haskell_compliant(file_name):
    file_test_output(file_name, run_haskell_file)


@pytest.mark.parametrize(
    "file_name",
    list(get_all_test_files(BASE_TEST_FILES_PATH)) + [EXAMPLE],
)
def test_is_type_valid(file_name):
    def run_type_check(f):
        return type_check(parse(f))

    file_test_type(file_name, run_type_check)
