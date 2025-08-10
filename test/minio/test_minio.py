import os
import subprocess

import pytest

from lango.minio.interpreter import interpret
from lango.minio.parser import parse
from lango.minio.typecheck import type_check

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


@pytest.mark.parametrize("file_name", list(get_all_test_files()))
def test_output_matches_expected(file_name):
    expected = get_test_output(file_name)
    tree = parse(file_name)
    result = interpret(tree, collectStdOut=True)
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
    try:
        tree = parse(file_name)
        type_check(tree)
    except Exception:
        assert False
