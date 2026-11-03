#!/usr/bin/env python3
"""Runs the verification and writes RESULT_VERIFICATION.md from what it measured.

Build first, then point --bin-dir at the folder holding the executables:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j
    python3 verification/run_verification.py --bin-dir build

Needs NumPy. Exits non-zero if any check fails.

The CUDA checks run when the build has benchmark_cuda (-DMINITENSOR_USE_CUDA=ON).
Use --cuda-bin-dir if the CUDA build is in a different folder from the CPU build.
Otherwise the report marks them as not done.
"""

import argparse
import csv
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "verification" / "results"
REPEATS = 15
WARMUPS = 3
SEED = 20261103

GROUP_TITLES = {
    "matmul": "Matrix multiplication",
    "matmul_incompatible": "Matrix multiplication, incompatible shapes",
    "broadcast": "Broadcasting, compatible shapes",
    "broadcast_incompatible": "Broadcasting, incompatible shapes",
    "transpose": "Transpose",
    "elementwise": "Same-shape element-wise operations",
    "reduction": "Reductions (sum, mean, min, max)",
    "reshape": "Reshape",
    "reshape_incompatible": "Reshape, incompatible sizes",
    "scope_limit": "Rejected by design (NumPy accepts these)",
    "control": "Negative controls (deliberately wrong results)",
}


def run(command, threads=None):
    env = dict(os.environ)
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
    return subprocess.run(command, env=env, text=True, capture_output=True)


def output_of(command):
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def executable(bin_dir, name):
    suffix = ".exe" if os.name == "nt" else ""
    path = Path(bin_dir) / (name + suffix)
    if not path.exists():
        raise SystemExit(f"Not found: {path} (build the project first)")
    return str(path)


def cpu_name():
    if platform.system() == "Windows":
        return output_of(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Processor).Name"]) or platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def memory_gb():
    if platform.system() == "Windows":
        text = output_of(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
        return f"{int(text) / 2**30:.1f} GB" if text else "unknown"
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return f"{int(line.split()[1]) / 2**20:.1f} GB"
    except OSError:
        pass
    return "unknown"


def blas_name():
    try:
        config = np.show_config(mode="dicts")
        blas = config["Build Dependencies"]["blas"]
        return f"{blas.get('name')} {blas.get('version')}"
    except Exception:
        return "unknown"


def time_numpy(workload, size, operation):
    for _ in range(WARMUPS):
        operation()
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        operation()
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    return {
        "implementation": "NumPy", "threads": "BLAS default", "workload": workload, "size": size,
        "min_ms": times[0], "median_ms": times[len(times) // 2],
        "mean_ms": float(np.mean(times)), "max_ms": times[-1],
        "stddev_ms": float(np.std(times, ddof=1)),
    }


def benchmark_numpy():
    rng = np.random.default_rng(SEED)
    rows = []
    for n in (64, 128, 256, 512, 1024):
        a, b = rng.uniform(-1, 1, (n, n)), rng.uniform(-1, 1, (n, n))
        rows.append(time_numpy("matmul", f"{n}x{n}", lambda: a @ b))
    for n in (1024, 4096):
        a = rng.uniform(-1, 1, (n, n))
        # .copy() because NumPy's transpose is otherwise a view with no work in it.
        rows.append(time_numpy("transpose", f"{n}x{n}", lambda: a.T.copy()))
    n = 2048
    a, b, row = rng.uniform(-1, 1, (n, n)), rng.uniform(-1, 1, (n, n)), rng.uniform(-1, 1, n)
    rows.append(time_numpy("add", f"{n}x{n}", lambda: a + b))
    rows.append(time_numpy("add_broadcast", f"{n}x{n} + {n}", lambda: a + row))
    rows.append(time_numpy("sum", f"{n}x{n}", lambda: np.sum(a)))
    return rows


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def ms(value):
    return f"{float(value):.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True)
    parser.add_argument("--cuda-bin-dir", help="CUDA build folder, if not the same as --bin-dir")
    parser.add_argument("--build-flags", default="CMake Release: -O3 -DNDEBUG, OpenMP on")
    parser.add_argument("--cuda-build-flags", default="CMake Release with -DMINITENSOR_USE_CUDA=ON")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--output", default=str(ROOT / "RESULT_VERIFICATION.md"))
    args = parser.parse_args()

    verify = executable(args.bin_dir, "verify_numerical")
    bench = executable(args.bin_dir, "benchmark_cpu")
    generator = str(ROOT / "verification" / "generate_reference.py")
    threads = os.cpu_count() or 1

    cuda_dir = Path(args.cuda_bin_dir or args.bin_dir)
    has_cuda = (cuda_dir / ("benchmark_cuda" + (".exe" if os.name == "nt" else ""))).exists()
    cuda_rerun_same = False
    cuda_info = ""

    RESULTS.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="minitensor_verify_"))

    try:
        print("Generating the NumPy reference twice...", flush=True)
        reference, reference2 = temp / "reference.bin", temp / "reference2.bin"
        for path in (reference, reference2):
            subprocess.run([sys.executable, generator, str(path)], check=True)
        reference_hash = sha256(reference)
        reference_same = reference_hash == sha256(reference2)

        print("Comparing MiniTensor with NumPy...", flush=True)
        numerical = RESULTS / "numerical.csv"
        all_threads = run([verify, str(reference), str(numerical)], threads)
        print(all_threads.stdout, all_threads.stderr, flush=True)
        if all_threads.returncode not in (0, 1):
            raise SystemExit(f"verify_numerical could not run (exit code {all_threads.returncode})")
        openmp = "openmp=1" in all_threads.stdout

        again = temp / "again.csv"
        one_thread = temp / "one_thread.csv"
        run([verify, str(reference), str(again)], threads)
        run([verify, str(reference), str(one_thread)], 1)
        numerical_hash = sha256(numerical)
        rerun_same = numerical_hash == sha256(again)
        threads_same = numerical_hash == sha256(one_thread)

        print(f"CPU benchmark with {threads} threads...", flush=True)
        bench_all = RESULTS / f"benchmark_cpu_{threads}_threads.csv"
        result = run([bench, str(bench_all), str(REPEATS)], threads)
        if result.returncode != 0:
            raise SystemExit("benchmark_cpu failed: " + result.stderr)

        print("CPU benchmark with 1 thread...", flush=True)
        bench_one = RESULTS / "benchmark_cpu_1_thread.csv"
        result = run([bench, str(bench_one), str(REPEATS)], 1)
        if result.returncode != 0:
            raise SystemExit("benchmark_cpu failed: " + result.stderr)

        print("NumPy benchmark...", flush=True)
        numpy_rows = benchmark_numpy()

        if has_cuda:
            print("Comparing the CUDA backend with NumPy...", flush=True)
            verify_cuda = executable(cuda_dir, "verify_numerical")
            numerical_cuda = RESULTS / "numerical_cuda.csv"
            result = run([verify_cuda, str(reference), str(numerical_cuda), "cuda"])
            print(result.stdout, result.stderr, flush=True)
            if result.returncode not in (0, 1):
                raise SystemExit(f"verify_numerical could not run on the CUDA backend (exit code {result.returncode})")

            cuda_again = temp / "cuda_again.csv"
            run([verify_cuda, str(reference), str(cuda_again), "cuda"])
            cuda_rerun_same = sha256(numerical_cuda) == sha256(cuda_again)

            print("CUDA benchmark...", flush=True)
            bench_cuda = RESULTS / "benchmark_cuda.csv"
            result = run([executable(cuda_dir, "benchmark_cuda"), str(bench_cuda), str(REPEATS)])
            if result.returncode != 0:
                raise SystemExit("benchmark_cuda failed: " + result.stderr)
            cuda_info = result.stdout.strip().splitlines()[-1]
    finally:
        shutil.rmtree(temp, ignore_errors=True)

    fields = ["implementation", "threads", "workload", "size", "min_ms", "median_ms",
              "mean_ms", "max_ms", "stddev_ms"]
    with open(RESULTS / "benchmark_numpy.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(numpy_rows)

    cases = read_csv(numerical)
    one_rows = read_csv(bench_one)
    all_rows = read_csv(bench_all)
    numpy_by_key = {(r["workload"], r["size"]): r for r in numpy_rows}
    all_by_key = {(r["workload"], r["size"]): r for r in all_rows}

    def group_cases(*groups):
        return [c for c in cases if c["group"] in groups]

    def passed(*groups):
        selected = group_cases(*groups)
        return bool(selected) and all(c["status"] == "PASS" for c in selected)

    def verdict(ok):
        return "PASS" if ok else "FAIL"

    bits_repeatable = all(r["identical_bits"] == "yes" for r in one_rows + all_rows)
    bits_across_threads = all(
        r["result_hash"] == all_by_key[(r["workload"], r["size"])]["result_hash"] for r in one_rows)
    reproducible = (reference_same and rerun_same and threads_same
                    and bits_repeatable and bits_across_threads)
    numerical_ok = all(c["status"] == "PASS" for c in cases)

    cuda_cases = read_csv(numerical_cuda) if has_cuda else []
    cuda_rows = read_csv(bench_cuda) if has_cuda else []
    cuda_numerical_ok = bool(cuda_cases) and all(c["status"] == "PASS" for c in cuda_cases)
    cuda_bits_repeatable = bool(cuda_rows) and all(r["identical_bits"] == "yes" for r in cuda_rows)
    cuda_ok = cuda_numerical_ok and cuda_rerun_same and cuda_bits_repeatable

    compared = [c for c in cases if c["expected_outcome"] == "matches NumPy"]
    rejected = [c for c in cases if c["expected_outcome"] == "rejected"]
    worst = max(float(c["relative_error"]) for c in compared)

    commit = output_of(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]) or "unknown"
    dirty = output_of(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "include"])
    cxx = (output_of([args.cxx, "--version"]) or "unknown").splitlines()[0]
    gpu = output_of(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                     "--format=csv,noheader"])

    report = []
    add = report.append
    add("# MiniTensor result verification\n")
    add(f"Generated by `verification/run_verification.py` on {time.strftime('%Y-%m-%d')}. "
        "Every number below was measured by that run; none is typed by hand. "
        "The raw data is in `verification/results/`.\n")

    add("## Summary\n")
    add(table(["Verification", "What to demonstrate", "Result", "Evidence"], [
        ["Numerical correctness", "Compare MiniTensor outputs against NumPy/PyTorch",
         f"**{verdict(numerical_ok)}**",
         f"{len(compared)} results compared with NumPy, worst relative error {worst:.2e}. PyTorch not used"],
        ["Matrix multiplication", "Test multiple matrix dimensions",
         f"**{verdict(passed('matmul', 'matmul_incompatible'))}**",
         f"{len(group_cases('matmul'))} results across 1x1x1 to 1024x1024x1024, "
         f"{len(group_cases('matmul_incompatible'))} incompatible shapes rejected"],
        ["Broadcasting", "Test compatible and incompatible shapes",
         f"**{verdict(passed('broadcast', 'broadcast_incompatible'))}**",
         f"{len(group_cases('broadcast'))} compatible results, "
         f"{len(group_cases('broadcast_incompatible'))} incompatible rejected"],
        ["Transpose", "Verify output dimensions and values",
         f"**{verdict(passed('transpose'))}**",
         f"{len(group_cases('transpose'))} shapes, dimensions and values both checked"],
        ["CPU benchmarking", "Repeated measurements", "**DONE**",
         f"{WARMUPS} warm-ups + {REPEATS} timed runs per workload, at 1 and {threads} threads"],
        ["CUDA benchmarking", "Warm-up, CUDA synchronization and repeated measurements",
         f"**{'DONE' if cuda_ok else 'FAIL'}**" if has_cuda else "**NOT DONE**",
         f"{WARMUPS} warm-ups + {REPEATS} timed runs per workload, stopping the clock after "
         f"`cudaDeviceSynchronize()`. {len(cuda_cases)} GPU checks against NumPy"
         if has_cuda else "This build has no CUDA backend, see below"],
        ["Reproducibility", "Fixed seeds and documented hardware/software",
         f"**{verdict(reproducible)}**", "Seeds and environment below, identical output across reruns"],
    ]))
    add("")

    add("## Method\n")
    add(f"`generate_reference.py` creates inputs with a fixed seed ({SEED}), computes the answers "
        "with NumPy and writes both to a file. `verify_numerical.cpp` runs the same operations "
        "through MiniTensor and compares.\n")
    add("A result passes when its shape matches exactly and the largest element-wise difference "
        "is at most `1e-10 * max(1, largest |expected|)`. This limit was set before the first run. "
        "Exact equality isn't required, because NumPy's BLAS adds terms in a different order. "
        "An incompatible case passes when NumPy raises an error and MiniTensor throws "
        "`std::invalid_argument`.\n")

    add("## Numerical results\n")
    rows = []
    for group, title in GROUP_TITLES.items():
        selected = group_cases(group)
        if not selected:
            continue
        errors = [float(c["relative_error"]) for c in selected if c["relative_error"] != "-"]
        rows.append([title, len(selected), sum(c["status"] == "PASS" for c in selected),
                     f"{max(errors):.2e}" if errors else "-"])
    add(table(["Group", "Checks", "Passed", "Worst relative error"], rows))
    add("")
    add(f"Total: {sum(c['status'] == 'PASS' for c in cases)} of {len(cases)} checks passed.\n")
    add("The negative controls feed the comparison a wrong value, a wrong shape and a NaN. "
        "Each must be caught, which shows the comparison can fail.\n")

    add("### Matrix multiplication\n")
    add(table(["Inputs", "Output", "Function", "Max abs error", "Relative error", "Status"],
              [[c["input_shapes"], c["output_shape"], f"`{c['op']}`", c["max_abs_error"],
                c["relative_error"], c["status"]] for c in group_cases("matmul")]))
    add("")
    add("### Broadcasting\n")
    add("Compatible shapes. Each row is checked for add, subtract and multiply, and shows the worst of the three.\n")
    pairs = {}
    for c in group_cases("broadcast"):
        pairs.setdefault((c["input_shapes"], c["output_shape"]), []).append(c)
    add(table(["Inputs", "Output", "Worst relative error", "Status"],
              [[key[0], key[1], f"{max(float(c['relative_error']) for c in group):.2e}",
                verdict(all(c["status"] == "PASS" for c in group))] for key, group in pairs.items()]))
    add("")
    add("Incompatible shapes. NumPy raised an error for every one of these.\n")
    pairs = {}
    for c in group_cases("broadcast_incompatible"):
        pairs.setdefault(c["input_shapes"], []).append(c)
    add(table(["Inputs", "Operations rejected by MiniTensor"],
              [[key, f"{sum(c['status'] == 'PASS' for c in group)} of {len(group)}"]
               for key, group in pairs.items()]))
    add("")
    add("### Transpose\n")
    add(table(["Input", "Output", "Max abs error", "Status"],
              [[c["input_shapes"], c["output_shape"], c["max_abs_error"], c["status"]]
               for c in group_cases("transpose")]))
    add("")
    add("### Known differences from NumPy\n")
    add("MiniTensor's `matmul` and `transpose` are defined for 2-D tensors only. NumPy also accepts "
        "other ranks. MiniTensor rejects them with an exception instead of returning a different "
        "answer, and the \"Rejected by design\" row confirms this.\n")

    add("## CPU benchmarks\n")
    add(f"{WARMUPS} warm-up runs are discarded, then {REPEATS} runs are timed. Times are in milliseconds. "
        "\"Spread\" is the standard deviation as a percentage of the mean. Lower times are better.\n")
    rows = []
    for row in one_rows:
        key = (row["workload"], row["size"])
        multi, reference_row = all_by_key[key], numpy_by_key[key]
        spread = 100.0 * float(multi["stddev_ms"]) / float(multi["mean_ms"]) if float(multi["mean_ms"]) else 0.0
        rows.append([row["workload"], row["size"], ms(row["median_ms"]), ms(multi["median_ms"]),
                     f"{ms(multi['min_ms'])} to {ms(multi['max_ms'])}", f"{spread:.1f}%",
                     ms(reference_row["median_ms"])])
    add(table(["Workload", "Size", "MiniTensor median, 1 thread", f"MiniTensor median, {threads} threads",
               f"Range, {threads} threads", f"Spread, {threads} threads", "NumPy median"], rows))
    add("")
    add("Only `matmul` uses OpenMP, and only for large inputs (at least 128 rows and "
        "rows x inner x columns of at least 256^3). The other workloads run on one thread in both columns, "
        "so differences between those columns are measurement noise.\n")
    add("NumPy is shown for scale. It uses an optimised BLAS, so a gap in `matmul` is expected. "
        "The machine is a laptop, and neither its power mode nor other running programs were controlled, "
        "so compare these timings to each other, not to timings from another machine.\n")

    add("## CUDA benchmarks\n")
    if has_cuda:
        add("These numbers measure the optional CUDA backend (`-DMINITENSOR_USE_CUDA=ON`, double precision). "
            "`matmul` calls cuBLAS `dgemm`. The other operations are MiniTensor's own kernels. "
            "The CPU tables above are not affected by this backend.\n")
        add("### How it was measured\n")
        add(f"- **Warm-up:** {WARMUPS} runs per workload are discarded. They also cover the one-time "
            "CUDA and cuBLAS setup.")
        add("- **Synchronization:** `cudaDeviceSynchronize()` runs before the clock starts and again before "
            "it stops, because a GPU call returns before the work has finished.")
        add(f"- **Repeats:** {REPEATS} timed runs per workload. The same seeded input data as the CPU benchmark is used.")
        add("- **sum:** it has to return its answer to the host, so its GPU time includes small copies "
            "back and forth, and it has no separate launch time.\n")
        add("### Timings\n")
        add("Times are in milliseconds, and lower is better.\n")
        add("- **GPU median:** inputs already on the GPU, clock stopped after synchronizing. This is the real GPU time.")
        add("- **Launch only:** the same call with the clock stopped before synchronizing. "
            "This is what a benchmark without synchronization would wrongly report.")
        add("- **End to end:** host tensors in and out, including both copies. This is what a caller with host data pays.\n")
        rows = []
        for row in cuda_rows:
            key = (row["workload"], row["size"])
            spread = 100.0 * float(row["stddev_ms"]) / float(row["mean_ms"]) if float(row["mean_ms"]) else 0.0
            rows.append([row["workload"], row["size"], ms(row["median_ms"]),
                         f"{ms(row['min_ms'])} to {ms(row['max_ms'])}", f"{spread:.1f}%",
                         ms(row["launch_median_ms"]), ms(row["end_to_end_median_ms"]),
                         ms(all_by_key[key]["median_ms"]), ms(numpy_by_key[key]["median_ms"])])
        add(table(["Workload", "Size", "GPU median", "GPU range", "Spread", "Launch only", "End to end",
                   f"CPU median, {threads} threads", "NumPy median"], rows))
        add("")
        add("The GPU column leaves out copies, so compare End to end with the CPU and NumPy columns when "
            "the data starts on the host. Other programs using the GPU and laptop power limits were not "
            "controlled, so these timings apply to this run on this machine.\n")
        add("### GPU numerical results\n")
        add("The same reference file and pass limit as the CPU. The backend has no `matmul_parallel`, "
            "`mean`, `min`, `max` or `reshape`, so those cases are skipped.\n")
        rows = []
        for group, title in GROUP_TITLES.items():
            selected = [c for c in cuda_cases if c["group"] == group]
            if not selected:
                continue
            errors = [float(c["relative_error"]) for c in selected if c["relative_error"] != "-"]
            rows.append([title, len(selected), sum(c["status"] == "PASS" for c in selected),
                         f"{max(errors):.2e}" if errors else "-"])
        add(table(["Group", "Checks", "Passed", "Worst relative error"], rows))
        add("")
    else:
        add("**Not performed.** This build has no CUDA backend, so there is nothing of MiniTensor's to "
            "time on a GPU. Configure with `-DMINITENSOR_USE_CUDA=ON` to include it. "
            "This section doesn't estimate any numbers.\n")

    add("## Reproducibility\n")
    add(table(["Check", "Result"], [
        ["Reference file is byte-identical when generated twice", verdict(reference_same)],
        ["Numerical results file is byte-identical on a rerun", verdict(rerun_same)],
        [f"Numerical results file is byte-identical at 1 and {threads} threads", verdict(threads_same)],
        ["Every timed repeat produced identical result bits", verdict(bits_repeatable)],
        [f"Benchmark results are bit-identical at 1 and {threads} threads", verdict(bits_across_threads)],
    ] + ([
        ["GPU numerical results file is byte-identical on a rerun", verdict(cuda_rerun_same)],
        ["Every timed GPU repeat and its host-copy run produced identical result bits", verdict(cuda_bits_repeatable)],
    ] if has_cuda else [])))
    add("")
    add(f"- Reference data seed: `{SEED}` (NumPy PCG64). Reference file SHA-256: `{reference_hash}`")
    add("- Benchmark data seeds: 1 to 6 (`std::mt19937_64`, raw output)")
    add("- Timings differ between runs. Only the computed results are expected to be identical.\n")

    add("### Hardware and software\n")
    environment = [
        ["MiniTensor commit", commit + (" (src/include modified)" if dirty else "")],
        ["CPU", f"{cpu_name()}, {threads} logical processors"],
        ["Memory", memory_gb()],
        ["GPU" if has_cuda else "GPU (not used)", gpu or "none detected"],
        ["OS", platform.platform()],
        ["Compiler", cxx],
        ["Build", args.build_flags],
        ["OpenMP in the tested build", "yes" if openmp else "no"],
        ["Python", platform.python_version()],
        ["NumPy", np.__version__],
        ["NumPy BLAS", blas_name()],
    ]
    if has_cuda:
        environment += [
            ["CUDA build", args.cuda_build_flags],
            ["CUDA device and runtime", cuda_info],
        ]
    add(table(["Item", "Value"], environment))
    add("")
    if platform.system() != "Linux":
        add(f"> These results were measured on {platform.system()}, while the challenge targets Ubuntu 22.04+. "
            "Run the commands below on Ubuntu for results from the target platform.\n")

    add("### Run it yourself\n")
    add("```bash\nsudo apt-get install -y cmake g++ python3-numpy\n"
        "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n"
        "cmake --build build -j\n"
        "python3 verification/run_verification.py --bin-dir build\n```\n")
    add("To include the GPU results, install the CUDA Toolkit and add `-DMINITENSOR_USE_CUDA=ON` "
        "to the first `cmake` command.\n")

    Path(args.output).write_text("\n".join(report), encoding="utf-8", newline="\n")
    print("Wrote", args.output)

    return 0 if numerical_ok and reproducible and (cuda_ok or not has_cuda) else 1


if __name__ == "__main__":
    sys.exit(main())
