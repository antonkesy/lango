import os
import re
import subprocess

from lango.parser.parser import are_types_correct, example, get_type_str


def get_test_output(file_name: str) -> str:
    with open(file_name, "r") as file:
        lines = file.readlines()
    match = re.search(r'"(.*?)"', lines[0])
    if match:
        return match.group(1)
    return ""


def get_all_test_files() -> list:
    test_file_path = "./test/files/minio/"
    test_files = []
    for root, _, files in os.walk(test_file_path):
        for file in files:
            if file.endswith(".minio"):
                test_files.append(os.path.join(root, file))
    return test_files


def test_all():
    for file_name in get_all_test_files():
        expected = get_test_output(file_name)
        try:
            result = example(file_name, True)
            assert (
                result == expected
            ), f"Failed for {file_name}, expected '{expected}', got '{result}'"

        except Exception as e:
            assert False, f"Exception for {file_name}: {type(e).__name__}: {e}"


def test_against_haskell():
    for file_name in get_all_test_files():
        try:
            result = example(file_name, True)
            haskell_result = run_haskell_file(file_name)
            assert (
                result == haskell_result
            ), f"Haskell result mismatch for {file_name}, Haskell: '{haskell_result}', lango: '{result}'"
        except Exception as e:
            assert False, f"Exception for {file_name}: {type(e).__name__}: {e}"


def test_type_check():
    for file_name in get_all_test_files():
        assert are_types_correct(
            file_name,
        ), f"Type check failed for {file_name}, {get_type_str(file_name)}"


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
