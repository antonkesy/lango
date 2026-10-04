"""Test files describe their expected behaviour in their first three lines:

-- RUN: OK|FAIL
-- TYPECHECK: OK|FAIL
-- "expected output"
"""

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HEADER = re.compile(r'-- RUN: (OK|FAIL)\n-- TYPECHECK: (OK|FAIL)\n-- "(.*)"\n')


@dataclass(frozen=True)
class Expectation:
    run_fails: bool
    typecheck_fails: bool
    output: str


def expectation_of(file_name: Path) -> Expectation:
    """A file without a header (the examples) is expected to be well-typed."""
    header = _HEADER.match(file_name.read_text())
    if header is None:
        return Expectation(run_fails=False, typecheck_fails=False, output="")
    run, typecheck, output = header.groups()
    return Expectation(run == "FAIL", typecheck == "FAIL", output)


def get_all_test_files(base_path: Path, extension: str) -> Iterator[Path]:
    return iter(sorted(base_path.rglob(f"*.{extension}")))


def file_test_output(file_name: Path, run: Callable[[Path], Any]) -> None:
    """``run`` must produce exactly the expected output, or fail if expected to."""
    expected = expectation_of(file_name)
    try:
        result = run(file_name)
    except Exception as e:
        assert (
            expected.run_fails
        ), f"{file_name}: expected {expected.output!r}, got {e!r}"
        return
    assert not expected.run_fails, f"{file_name}: expected a failure, got {result!r}"
    assert (
        result == expected.output
    ), f"{file_name}: expected {expected.output!r}, got {result!r}"


def file_test_type(file_name: Path, type_check: Callable[[Path], Any]) -> None:
    """``type_check`` must raise exactly when the file says it is ill-typed."""
    expected = expectation_of(file_name)
    try:
        type_check(file_name)
    except Exception as e:
        assert (
            expected.typecheck_fails
        ), f"{file_name}: expected to type check, got {e!r}"
        return
    assert not expected.typecheck_fails, f"{file_name}: expected a type error"
