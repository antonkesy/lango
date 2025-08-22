import os
import subprocess
import tempfile

import pytest

from lango.minio.compiler.python import compile_program
from lango.minio.interpreter.interpreter import interpret
from lango.minio.parser.parser import parse
from lango.minio.typechecker.typecheck import type_check

BASE_TEST_FILES_PATH = "./test/files/minio/"
EXAMPLE = "./examples/minio/example.minio"


def get_all_test_files():
    for root, _, files in os.walk(BASE_TEST_FILES_PATH):
        for file in files:
            if file.endswith(".minio"):
                yield os.path.join(root, file)


def get_test_output(file_name: str) -> str:
    # TODO make more robust
    with open(file_name, "r") as file:
        first_line = file.readline().strip()
    return first_line[4:-1]


def run_haskell_file(path: str) -> str:
    try:
        result = subprocess.run(
            ["runghc", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return result.stdout.decode("utf-8")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr.decode("utf-8").strip())


def run_compiled_python(python_code: str) -> str:
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            temp_file = f.name
            f.write(python_code)
            f.flush()

            result = subprocess.run(
                ["python3", temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return result.stdout.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Python execution failed: {e.stderr.decode('utf-8').strip()}",
        )
    finally:
        if temp_file:
            os.unlink(temp_file)


@pytest.mark.parametrize("file_name", list(get_all_test_files()))
def test_interpreter(file_name):
    expected = get_test_output(file_name)
    result = interpret(parse(file_name), collectStdOut=True)
    assert result == expected, f"{file_name}: Expected '{expected}', got '{result}'"


@pytest.mark.parametrize("file_name", list(get_all_test_files()))
def test_compiler(file_name):
    expected = get_test_output(file_name)
    ast = parse(file_name)
    python_code = compile_program(ast)
    result = run_compiled_python(python_code)
    assert result == expected, f"{file_name}: Expected '{expected}', got '{result}'"


@pytest.mark.parametrize("file_name", list(get_all_test_files()))
def test_is_haskell_compliant(file_name):
    expected = get_test_output(file_name)
    haskell_result = run_haskell_file(file_name)
    assert expected == haskell_result, (
        f"Haskell mismatch for {file_name}: "
        f"Haskell: '{haskell_result}', expected: '{expected}'"
    )


@pytest.mark.parametrize("file_name", list(get_all_test_files()) + [EXAMPLE])
def test_is_type_valid(file_name):
    result = type_check(parse(file_name))
    assert result, f"Type checking failed for {file_name}"
