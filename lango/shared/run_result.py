"""The result of interpreting a program."""

from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO


@dataclass(frozen=True, slots=True)
class RunResult:
    output: str = ""  # what the program printed, when it was collected
    exit_code: int = 0


def run_program(run: Callable[[], object], collect_stdout: bool) -> RunResult:
    """Run ``run``, collecting what it prints if ``collect_stdout``."""
    if not collect_stdout:
        run()
        return RunResult()
    buffer = StringIO()
    with redirect_stdout(buffer):
        run()
    return RunResult(buffer.getvalue())
