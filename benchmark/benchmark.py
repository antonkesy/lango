import subprocess
import time

# pre compile the benchmark
prepare_cmd = [
    "lango compile minio ./benchmark/fib.minio -o ./benchmark/minio.py --target python",
    "lango compile minio ./benchmark/fib.minio -o ./benchmark/minio.go --target go",
    "lango compile systemo ./benchmark/fib.minio -o ./benchmark/syso_mono.py --strategy monomorphization",
    "lango compile systemo ./benchmark/fib.minio -o ./benchmark/syso_dict.py --strategy dictionary_passing",
    "go build -o ./benchmark/fibGo ./benchmark/minio.go",
    "ghc ./benchmark/fib.hs",
]
for prep in prepare_cmd:
    subprocess.run(prep, shell=True, check=True)

utilities = [
    "lango run minio ./benchmark/fib.minio",
    "lango run systemo ./benchmark/fib.minio",
    "python3 ./benchmark/minio.py",
    "./benchmark/fibGo",
    "python3 ./benchmark/syso_mono.py",
    "python3 ./benchmark/syso_dict.py",
    "./benchmark/fib",
]

n = 25
expected_result = 75025  # fib(25)

results = {}

for cmd in utilities:
    times = []
    for i in range(n):
        print(f"Running command: {cmd} (Run {i+1}/{n})")
        start_time = time.time()
        try:
            output = subprocess.check_output(cmd, shell=True, text=True).strip()

            num = float(output)
            if num != expected_result:
                print(
                    f"Command '{cmd}' returned incorrect result: {num} (expected {expected_result})",
                )
                times.append(None)
                continue

            elapsed = time.time() - start_time
            times.append(elapsed)

        except subprocess.CalledProcessError as e:
            print(f"Command '{cmd}' failed with exit code {e.returncode}")
            times.append(None)
        except ValueError:
            print(f"Command '{cmd}' output is not a number: '{output}'")
            times.append(None)

    results[cmd] = times

for cmd, times in results.items():
    valid_times = [t for t in times if t is not None]
    if valid_times:
        avg_time = sum(valid_times) / len(valid_times)
        print(f"Command: {cmd}")
        print(f"  Runs: {len(valid_times)}/{n} successful")
        print(f"  Times: {valid_times}")
        print(f"  Average time: {avg_time:.6f} sec")
    else:
        print(f"Command: {cmd} failed all runs")
