import os
import subprocess
import tempfile


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


def _run_x_code(code: str, executable: list[str], extension: str) -> str:
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=extension, delete=False) as f:
            temp_file = f.name
            f.write(code)
            f.flush()

            result = subprocess.run(
                executable + [temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return result.stdout.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Execution failed: {e.stderr.decode('utf-8').strip()}",
        )
    finally:
        if temp_file:
            os.unlink(temp_file)


def run_python_code(python_code: str) -> str:
    return _run_x_code(python_code, ["python3"], ".py")


def run_systemf_code(code: str) -> str:
    return _run_x_code(code, ["fullpoly"], ".sf")


def run_go_code(go_code: str) -> str:
    return _run_x_code(go_code, ["go", "run"], ".go")
