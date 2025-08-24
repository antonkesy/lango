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
