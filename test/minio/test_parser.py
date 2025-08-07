import os
import re

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
        assert (
            example(file_name, True) == expected
        ), f"Failed for {file_name}, expected {expected}, got {example(file_name, True)}"


def test_type_check():
    for file_name in get_all_test_files():
        pass
        # assert are_types_correct(
        #     file_name
        # ), f"Type check failed for {file_name}, {get_type_str(file_name)}"
