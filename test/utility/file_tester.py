import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generator


@dataclass
class TestOutput:
    runtimeFails: bool
    typecheckFails: bool
    expected_output: str


def _get_test_output(file_name: Path) -> TestOutput:
    """Parses the first lines of the test file to determine the expected output.
    Format:
        -- RUN: OK|FAIL
        -- TYPECHECK: OK|FAIL
        -- "expected output"
    """
    with open(file_name, "r") as file:
        first_line = file.readline().strip()
        second_line = file.readline().strip()
        third_line = file.readline().strip()
        return TestOutput(
            runtimeFails=first_line == "-- RUN: FAIL",
            typecheckFails=second_line == "-- TYPECHECK: FAIL",
            expected_output=third_line[4:-1],  # remove -- " and "
        )


def get_all_test_files(base_path: Path, extension: str) -> Generator[Path, None, None]:
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith("." + extension):
                yield Path(os.path.join(root, file))


def file_test_output(file_name: Path, runFn: Callable[[Path], Any]) -> None:
    expected = _get_test_output(file_name)
    try:
        result = runFn(file_name)
    except Exception as e:
        if not expected.runtimeFails:
            assert False, f"{file_name}: Expected '{expected}', but got exception '{e}'"
        return  # if expected.fails, we can return here

    if expected.runtimeFails:
        assert False, f"{file_name}: Expected failure, but got '{result}'"
    assert (
        result == expected.expected_output
    ), f"{file_name}: Expected '{expected.expected_output}', got '{result}'"


def file_test_type(file_name: Path, runFn: Callable[[Path], Any]) -> None:
    expected = _get_test_output(file_name)
    try:
        runFn(file_name)
    except Exception as e:
        if not expected.typecheckFails:
            assert (
                False
            ), f"{file_name}: Expected type check to pass, but got exception '{e}'"
        return

    if expected.typecheckFails:
        assert False, f"{file_name}: Expected failure, but type check passed"
