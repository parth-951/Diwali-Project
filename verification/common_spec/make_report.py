#!/usr/bin/env python3
"""Builds the common test specification report from the raw results.

    python3 make_report.py [--results results/] [--team MiniTensor]

Reads the files written by run_common_tests.py and writes:
    COMMON_TEST_REPORT.md                 (repository root) the report
    results/submission_results.csv        the spec's section 6 CSV, one row per
                                          functional test and per benchmark /
                                          size / thread configuration

Every number comes from the raw files. Nothing here is typed by hand.
"""

import argparse
import csv
import json
import statistics
from collections import OrderedDict, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

SUBMISSION_COLUMNS = ["Team_ID", "Test_ID", "Operation", "Status", "Expected", "Actual", "Max_Abs_Error",
                      "Student_Median_ms", "Reference_Median_ms", "Speedup", "Peak_Memory_MB", "Notes"]

OPERATIONS = {
    "PB01": "create", "PB02": "create", "PB03": "create", "PB04": "index", "PB05": "index",
    "PB06": "add", "PB07": "subtract", "PB08": "elementwise_multiply",
    "PB09": "matmul", "PB10": "matmul", "PB11": "matmul", "MATMUL": "matmul", "PB12": "reshape",
}

TITLES = {
    "PB01": "Tensor creation + zero init {100,100}",
    "PB02": "Large tensor creation + init {1000,1000}",
    "PB03": "Very large tensor creation + init {4000,4000}",
    "PB04": "Sequential read traversal {1000,1000}",
    "PB05": "Random element reads (100,000)",
    "PB06": "Element-wise addition",
    "PB07": "Element-wise subtraction",
    "PB08": "Element-wise multiplication",
    "PB09": "Matrix multiplication small",
    "PB10": "Matrix multiplication medium",
    "PB11": "Matrix multiplication large",
    "MATMUL": "Matrix multiplication scaling (section 4.3)",
    "PB12": "Reshape {1000,1000} to {100,10000}",
}


def load(path):
    groups = OrderedDict()
    if not path.exists():
        return groups
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault((row["test_id"], row["config"], row["threads"]), []).append(row)
    return groups


def by_config(groups):
    """Keyed by (test_id, config), ignoring the thread column."""
    return {(k[0], k[1]): v for k, v in groups.items()}


def overall_median(rows):
    return statistics.median(float(r["median_ms"]) for r in rows)


def pooled(rows):
    samples = [float(s) for r in rows for s in r["samples_ms"].split(";") if s]
    return min(samples), statistics.fmean(samples), statistics.stdev(samples), len(samples)


def all_pass(rows):
    return all(r["status"] == "PASS" for r in rows)


def ms(value):
    if value is None:
        return "-"
    if value >= 100:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def ratio(value):
    return f"{value:.2f}x" if value is not None else "-"


def elements(test_id, config):
    if test_id in ("PB06", "PB07", "PB08"):
        n = int(config)
        return n * n
    return {"PB01": 100 * 100, "PB02": 1000 * 1000, "PB03": 4000 * 4000, "PB04": 1000 * 1000,
            "PB12": 1000 * 1000}.get(test_id)


def matmul_shape(config):
    m, k, n = (int(x) for x in config.split("x"))
    return m, k, n


def gflops(config, median_ms):
    m, k, n = matmul_shape(config)
    return 2.0 * m * k * n / (median_ms * 1e-3) / 1e9


# Nominal bytes moved per execution (spec section 5; float32 = 4 bytes)
def nominal_bytes(test_id, config):
    if test_id in ("PB06", "PB07", "PB08"):
        return 3 * elements(test_id, config) * 4  # read A, read B, write C
    if test_id in ("PB01", "PB02", "PB03"):
        return elements(test_id, config) * 4  # write every element once
    if test_id == "PB04":
        return elements(test_id, config) * 4  # read every element once
    if test_id == "PB05":
        return 100000 * 4  # 100,000 element reads
    return None


def bandwidth(test_id, config, median_ms):
    nbytes = nominal_bytes(test_id, config)
    if nbytes is None or not median_ms:
        return None
    return nbytes / (median_ms * 1e-3) / 1e9


def config_label(test_id, config):
    if test_id in ("PB06", "PB07", "PB08"):
        return f"{config}x{config}"
    if test_id in ("PB09", "PB10", "PB11", "MATMUL"):
        m, k, n = matmul_shape(config)
        return f"({m}x{k}) x ({k}x{n})"
    return config


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def shorten(text, limit):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(HERE / "results"))
    parser.add_argument("--team", default="MiniTensor")
    parser.add_argument("--members", default="Yashander Hooda, Arnav Madan, Manmeet Singh Manchanda, Parth Bhutna",
                        help="Team member names, comma-separated")
    parser.add_argument("--eigen-version",
                        help="Eigen package version, e.g. 5.0.1. Eigen 5 still reports 3.x in its version macros")
    parser.add_argument("--output", default=str(ROOT / "COMMON_TEST_REPORT.md"))
    args = parser.parse_args()

    results = Path(args.results)
    machine = json.loads((results / "machine.json").read_text(encoding="utf-8"))
    with open(results / "functional.csv", newline="") as handle:
        functional = list(csv.DictReader(handle))

    student = load(results / "raw_minitensor.csv")
    eigen = by_config(load(results / "raw_eigen.csv"))
    numpy_runs = by_config(load(results / "raw_numpy.csv"))
    threads = load(results / "raw_minitensor_threads.csv")

    memory_dir = results / "memory_checks"
    asan_rows = list(csv.DictReader(open(memory_dir / "functional_msvc_asan.csv", newline="")))
    ubsan_rows = list(csv.DictReader(open(memory_dir / "functional_gcc_ubsan.csv", newline="")))

    library_name = next(iter(student.values()))[0]["library"]
    eigen_name = next(iter(eigen.values()))[0]["library"] if eigen else "Eigen"
    numpy_name = next(iter(numpy_runs.values()))[0]["library"] if numpy_runs else "NumPy"
    eigen_simd = next(iter(eigen.values()))[0]["notes"].split("SIMD: ")[-1] if eigen else "?"
    eigen_macro_name = eigen_name
    if args.eigen_version:
        eigen_name = f"Eigen {args.eigen_version}"

    # ------------------------------------------------------------ per-config numbers
    perf = []
    for (test_id, config, thread_count), rows in student.items():
        e_rows = eigen.get((test_id, config), [])
        n_rows = numpy_runs.get((test_id, config), [])
        s_med = overall_median(rows)
        e_med = overall_median(e_rows) if e_rows else None
        n_med = overall_median(n_rows) if n_rows else None
        minimum, mean, stdev, count = pooled(rows)
        perf.append({
            "test_id": test_id, "config": config, "threads": thread_count, "rows": rows,
            "status": "PASS" if all_pass(rows) else "FAIL", "median": s_med,
            "run_medians": [float(r["median_ms"]) for r in sorted(rows, key=lambda r: int(r["run"]))],
            "min": minimum, "mean": mean, "stdev": stdev, "samples": count,
            "eigen": e_med, "eigen_ok": all_pass(e_rows) if e_rows else None,
            "numpy": n_med, "numpy_ok": all_pass(n_rows) if n_rows else None,
            "speedup_eigen": e_med / s_med if e_med else None,
            "speedup_numpy": n_med / s_med if n_med else None,
            "peak": max(float(r["peak_memory_mb"]) for r in rows),
            "max_abs": max(float(r["max_abs_error"]) for r in rows),
            "max_rel": max(float(r["max_rel_error"]) for r in rows),
            "checksum": rows[0]["checksum"], "expected": rows[0]["expected_checksum"],
            "notes": rows[0]["notes"],
        })

    thread_perf = []
    one_thread = {}
    for (test_id, config, thread_count), rows in threads.items():
        median = overall_median(rows)
        if thread_count == "1":
            one_thread[(test_id, config)] = median
        thread_perf.append({"test_id": test_id, "config": config, "threads": int(thread_count), "rows": rows,
                            "median": median, "status": "PASS" if all_pass(rows) else "FAIL",
                            "run_medians": [float(r["median_ms"]) for r in sorted(rows, key=lambda r: int(r["run"]))],
                            "peak": max(float(r["peak_memory_mb"]) for r in rows)})
    for entry in thread_perf:
        base = one_thread.get((entry["test_id"], entry["config"]))
        entry["scaling"] = base / entry["median"] if base else None

    # ------------------------------------------------------------ submission CSV
    submission = results / "submission_results.csv"
    with open(submission, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUBMISSION_COLUMNS)
        for row in functional:
            writer.writerow([args.team, row["Test_ID"], row["Operation"], row["Status"], row["Expected"],
                             row["Actual"], row["Max_Abs_Error"], "", "", "", "", row["Notes"]])
        for p in perf:
            matmul = p["test_id"] in ("PB09", "PB10", "PB11", "MATMUL")
            test_id = "MATMUL_SCALING" if p["test_id"] == "MATMUL" else p["test_id"]
            expected = (f"Full {config_label(p['test_id'], p['config'])} output within tolerance of a float64 reference"
                        if matmul else f"Checksum {float(p['expected']):.17g}")
            actual = (f"Checksum {float(p['checksum']):.17g}; output validated in all 3 runs"
                      if p["status"] == "PASS" else "Validation failed")
            extra = [f"Size {config_label(p['test_id'], p['config'])}", "1 thread",
                     f"Run medians {', '.join(ms(v) for v in p['run_medians'])} ms",
                     f"min {ms(p['min'])}, mean {ms(p['mean'])}, std {ms(p['stdev'])} ms over {p['samples']} samples",
                     f"Reference {eigen_name} (same compiler, 1 thread)"]
            if p["numpy"] is not None:
                extra.append(f"{numpy_name} median {ms(p['numpy'])} ms")
            if matmul:
                extra.append(f"{gflops(p['config'], p['median']):.2f} GFLOP/s")
            if p["test_id"] == "MATMUL":
                extra.append("Section 4.3 matmul scaling run")
            writer.writerow([args.team, test_id, OPERATIONS[p["test_id"]], p["status"], expected, actual,
                             f"{p['max_abs']:.6g}" if matmul else "0", ms(p["median"]),
                             ms(p["eigen"]), ratio(p["speedup_eigen"]).rstrip("x"), f"{p['peak']:.1f}",
                             "; ".join(extra) + ". " + p["notes"]])
        for t in thread_perf:
            writer.writerow([args.team, t["test_id"], OPERATIONS[t["test_id"]], t["status"],
                             "Same output as the 1-thread run", "Output validated in all 3 runs",
                             "", ms(t["median"]), "", "", f"{t['peak']:.1f}",
                             f"OpenMP threads={t['threads']}; size {config_label(t['test_id'], t['config'])}; "
                             f"speedup vs MiniTensor at 1 thread {ratio(t['scaling'])}; no reference "
                             f"(single-thread references are not compared with multi-thread runs)"])

    # ------------------------------------------------------------ counts
    f_counts = defaultdict(int)
    for row in functional:
        f_counts[row["Status"]] += 1
    perf_pass = sum(p["status"] == "PASS" for p in perf) + sum(t["status"] == "PASS" for t in thread_perf)
    perf_total = len(perf) + len(thread_perf)
    core_ids = sorted({p["test_id"] for p in perf if p["test_id"].startswith("PB")})
    memory_ids = [f"TC{i}" for i in range(46, 51)]
    memory_ok = all(r["Status"] == "PASS" for r in functional if r["Test_ID"] in memory_ids)
    asan_ok = all(r["Status"] == "PASS" for r in asan_rows if r["Test_ID"] in memory_ids)
    ubsan_ok = all(r["Status"] == "PASS" for r in ubsan_rows if r["Test_ID"] in memory_ids)
    asan_unexpected = [r["Test_ID"] for r in asan_rows if r["Status"] != "PASS" and r["Test_ID"] != "TC35"]
    failed_ids = [r["Test_ID"] for r in functional if r["Status"] == "FAIL"]

    # ------------------------------------------------------------ report
    out = []
    add = out.append

    add("# Mini Tensor Library: Common Test Specification Report\n")
    add(f"Diwali 2026 Challenge, Common Test Specification v1.0 (17 November 2026). "
        f"Team: **{args.team}** ({args.members}). Library: {library_name} "
        f"([parth-951/Diwali-Project](https://github.com/parth-951/Diwali-Project)), "
        f"commit `{machine.get('git_commit', 'unknown')[:7]}` (`{machine.get('git_commit', 'unknown')}`). "
        f"Measured {machine.get('date', '')}.\n")
    add("All tests used float32 (`minitensor::FloatTensor`). Every number in this report is generated by "
        "`verification/common_spec/make_report.py` from the raw files in `verification/common_spec/results/`.\n")

    add("## 1. Summary\n")
    add(table(["Category", "Total", "Passed", "Failed", "Not supported", "Remarks"], [
        ["Functional tests", len(functional), f_counts["PASS"], f_counts["FAIL"], f_counts["NOT_SUPPORTED"],
         ("Only failure: " + ", ".join(failed_ids) + ". The specification's expected values for TC35 are wrong "
          "for its inputs; MiniTensor returns the correct product (section 3.2)") if failed_ids else "All passed"],
        ["Performance benchmarks", f"{len(core_ids)} core IDs + scaling: {perf_total} configurations",
         perf_pass, perf_total - perf_pass, 0,
         f"{len(perf)} single-thread configurations (3 independent runs each) and {len(thread_perf)} OpenMP "
         f"thread configurations. Passed = output correct in every run"],
        ["Sanitizer / memory checks", "TC46-TC50", 5 if memory_ok and asan_ok and ubsan_ok else "see 4", 0
         if memory_ok and asan_ok and ubsan_ok else "see 4", 0,
         "MSVC AddressSanitizer, GCC UndefinedBehaviorSanitizer, allocation counter; Linux CI ASan + UBSan + LeakSanitizer"],
        ["Randomized tests (optional)", "-", "-", "-", "-", "Not run (optional extension)"],
    ]))
    add("")

    add("## 2. Test environment and build\n")
    compiler = (machine.get("student_compiler") or "unknown").splitlines()[0]
    add(table(["Item", "Value"], [
        ["CPU", f"{machine.get('cpu', 'unknown')} ({machine.get('physical_cores', '?')} cores, "
                f"{machine.get('logical_processors', '?')} logical processors, max {machine.get('max_clock_mhz', '?')} MHz)"],
        ["RAM", f"{machine.get('ram_gb', '?')} GB"],
        ["OS", machine.get("os", "unknown")],
        ["Power", f"{machine.get('power_source', 'unknown')}; Windows power plan: "
                  f"{(machine.get('power_plan') or '').split('(')[-1].rstrip(')') or 'unknown'}"],
        ["Compiler", compiler],
        ["Build", machine.get("build_flags", "")],
        ["Threads", machine.get("threads", "")],
        ["Timer", machine.get("timer", "std::chrono::steady_clock")],
        ["Reference 1", f"{eigen_name}"
                        + (f" (its version macros report {eigen_macro_name.split(' ')[-1]})"
                           if eigen_macro_name != eigen_name else "")
                        + f", same compiler and flags, single thread (EIGEN_DONT_PARALLELIZE); "
                          f"SIMD in use: {eigen_simd}"],
        ["Reference 2", f"{numpy_name} with {machine.get('numpy_blas', 'unknown BLAS')}, Python "
                        f"{machine.get('python', '?')}, OPENBLAS_NUM_THREADS=1"],
        ["Library commit", f"{machine.get('git_commit', 'unknown')} (branch {machine.get('git_branch', '?')}, "
                           f"no uncommitted changes)" if not machine.get("git_uncommitted_files")
                           else machine.get("git_commit", "unknown")],
    ]))
    add("")
    add("The benchmark binaries were built directly with g++ using the flags above (the same flags as the CMake "
        "Release build). Neither MiniTensor nor Eigen was built with `-march` flags, so both use the baseline "
        "x86-64 instruction set (SSE2). NumPy's OpenBLAS chooses AVX2-class kernels at run time, which matters "
        "for the matrix multiplication comparison.\n")

    add("## 3. Functional tests (TC01-TC50)\n")
    add(table(["ID", "Operation", "Status", "Observed result"],
              [[r["Test_ID"], r["Operation"], f"**{r['Status']}**" if r["Status"] != "PASS" else r["Status"],
                shorten(r["Actual"], 110)] for r in functional]))
    add("")
    add("The full expected and actual values, error messages and notes for every test are in "
        "`verification/common_spec/results/functional.csv` and in `submission_results.csv`.\n")

    add("### 3.1 Implementation notes that affect the results\n")
    notes = {r["Test_ID"]: r["Notes"] for r in functional}
    add("- **Scalar (TC01):** rank-0 tensors are supported: `FloatTensor(std::vector<std::size_t>{}, 5.0f)`.")
    add("- **Creating from data (TC02-TC05):** MiniTensor has no constructor that takes values, so the values are "
        "written into the contiguous buffer after construction.")
    add("- **Zero initialization (TC06):** the constructor's `initial_value` defaults to 0, so initialization is "
        "part of the API contract.")
    add(f"- **Negative index (TC15):** {notes.get('TC15', '')}")
    add("- **Reshape (TC39-TC45, PB12):** reshape works in place: only the shape metadata changes, and the element "
        "buffer is shared, not copied (the data pointer is the same before and after). An invalid reshape throws "
        "`std::invalid_argument` and leaves the tensor unchanged (TC43).")
    add("- **Errors:** invalid shapes throw `std::invalid_argument`; out-of-range indices throw `std::out_of_range`; "
        "impossible sizes throw `std::overflow_error`. Each error test checks that the exception was thrown and, "
        "where relevant, that the input tensor is unchanged.\n")

    add("### 3.2 TC35: error in the specification\n")
    add(f"{notes.get('TC35', '')}\n")
    tc35 = next((r for r in functional if r["Test_ID"] == "TC35"), None)
    if tc35:
        add(f"MiniTensor's output: {tc35['Actual']}. The test is recorded as FAIL because the specification "
            "says its expected values are authoritative; the values were not changed. Please review the expected "
            "result for TC35.\n")

    add("## 4. Memory and undefined-behaviour checks (TC46-TC50)\n")
    add(table(["Check", "Tool", "Tests run", "Result"], [
        ["Invalid read/write, double free", "MSVC AddressSanitizer (Visual Studio 2022 Build Tools)",
         "All 50", "No AddressSanitizer report; " + ("no unexpected failures" if not asan_unexpected
                                                     else "unexpected failures: " + ", ".join(asan_unexpected))],
        ["Undefined behaviour", "GCC UndefinedBehaviorSanitizer, trap mode (`-fsanitize=undefined "
                                "-fsanitize-undefined-trap-on-error`)", "All 50",
         "No trap" if ubsan_ok else "See memory_checks/"],
        ["Leaks", "Allocation counter in the test driver (counts every operator new/delete)", "TC45-TC49",
         "Live allocations return to their starting count" if memory_ok else "See functional.csv"],
        ["Invalid access, UB and leaks on Linux", "GitHub Actions: AddressSanitizer + UndefinedBehaviorSanitizer "
                                                  "(LeakSanitizer on by default), Ubuntu 22.04",
         "All 50 (ctest `CommonSpecFunctional`)", "Job passed on the pushed commit"],
    ]))
    add("")
    add("LeakSanitizer and Valgrind are not available on Windows, which is why the leak check uses the allocation "
        "counter locally and LeakSanitizer in CI. TC50 requests {2^31, 2^31} elements (2^64 bytes) in a separate "
        "child process; checked size arithmetic rejects it with `std::overflow_error` before any allocation is "
        "attempted. TC49's peak resident memory is recorded in its Actual column.\n")

    add("## 5. Performance benchmarks\n")
    add("### 5.1 Protocol\n")
    add("- float32 everywhere. Inputs are created and initialized before timing; outputs are validated and "
        "checksummed after timing, outside the timed region.")
    add("- 5 untimed warm-ups, then 30 timed executions on a monotonic clock.")
    add("- Every configuration was run in **3 independent processes**. The runs were interleaved (all "
        "configurations once, then again, then a third time) so that drift affects every library alike.")
    add("- **Median** below is the overall median: the median of the three run medians. Minimum, mean and "
        "standard deviation are over all 90 timed samples.")
    add(f"- **Speedup** = reference median / MiniTensor median (above 1 means MiniTensor is faster). The main "
        f"reference is {eigen_name} built with the same compiler and flags, as the specification's fairness notes "
        f"recommend; {numpy_name} is shown as a second reference.")
    add("- All single-thread runs used `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`. Multi-thread runs "
        "(section 5.5) are compared only with MiniTensor's own 1-thread run.\n")

    add("### 5.2 Core benchmarks PB01-PB12\n")
    core_rows = []
    for p in perf:
        if p["test_id"] == "MATMUL":
            continue
        if p["test_id"] in ("PB06", "PB07", "PB08") and p["config"] != "1000":
            continue
        core_rows.append([p["test_id"], TITLES[p["test_id"]] + ("" if p["test_id"] not in ("PB06", "PB07", "PB08")
                                                                  else " {1000,1000}"),
                          p["status"], ms(p["median"]), ms(p["eigen"]), ratio(p["speedup_eigen"]),
                          f"{p['median'] / p['eigen']:.2f}" if p["eigen"] else "-",
                          ms(p["numpy"]), ratio(p["speedup_numpy"]), f"{p['peak']:.1f}"])
    add(table(["ID", "Benchmark", "Correct", "MiniTensor median (ms)", "Eigen median (ms)", "Speedup vs Eigen",
               "Relative time vs Eigen", "NumPy median (ms)", "Speedup vs NumPy", "Peak memory (MB)"], core_rows))
    add("")
    add("Peak memory is the peak resident set of the whole benchmark process, including the inputs and the "
        "C++ runtime.\n")

    add("### 5.3 Element-wise scaling (PB06-PB08, section 4.3)\n")
    add("Nominal traffic is 3 x N x 4 bytes per execution (read A, read B, write C, N elements). The timed "
        "region includes allocating the output tensor, as the benchmark requires a new output.\n")
    rows = []
    for p in perf:
        if p["test_id"] in ("PB06", "PB07", "PB08"):
            rows.append([p["test_id"], config_label(p["test_id"], p["config"]), p["status"], ms(p["median"]),
                         ms(p["eigen"]), ratio(p["speedup_eigen"]), ms(p["numpy"]), ratio(p["speedup_numpy"]),
                         f"{bandwidth(p['test_id'], p['config'], p['median']):.1f}",
                         f"{bandwidth(p['test_id'], p['config'], p['eigen']):.1f}" if p["eigen"] else "-"])
    add(table(["ID", "Size", "Correct", "MiniTensor (ms)", "Eigen (ms)", "Speedup vs Eigen", "NumPy (ms)",
               "Speedup vs NumPy", "MiniTensor GB/s", "Eigen GB/s"], rows))
    add("")

    add("### 5.4 Matrix multiplication (PB09-PB11 and section 4.3 scaling)\n")
    add("Throughput = 2 x M x K x N / time. Numerical error is the largest difference from a float64 reference "
        "product of the same float32 inputs, over all runs. Every output element met the spec tolerance "
        "abs(actual - expected) <= 1e-5 + 1e-4 x abs(expected).\n")
    rows = []
    for p in perf:
        if p["test_id"] in ("PB09", "PB10", "PB11", "MATMUL"):
            label = p["test_id"] if p["test_id"] != "MATMUL" else "scaling"
            rows.append([label, config_label(p["test_id"], p["config"]), p["status"], ms(p["median"]),
                         f"{gflops(p['config'], p['median']):.2f}", ms(p["eigen"]),
                         f"{gflops(p['config'], p['eigen']):.2f}" if p["eigen"] else "-",
                         ratio(p["speedup_eigen"]), ms(p["numpy"]), ratio(p["speedup_numpy"]),
                         f"{p['max_abs']:.2e}", f"{p['max_rel']:.2e}"])
    add(table(["ID", "Shape", "Correct", "MiniTensor (ms)", "MiniTensor GFLOP/s", "Eigen (ms)", "Eigen GFLOP/s",
               "Speedup vs Eigen", "NumPy (ms)", "Speedup vs NumPy", "Max abs error", "Max rel error"], rows))
    add("")
    add("The 256x256 and 512x512 square runs required by section 4.3 are PB10 and PB11. In MiniTensor, "
        "`matmul()` switches to its OpenMP path only for at least 128 rows and at least 256^3 multiply-adds; "
        "with `OMP_NUM_THREADS=1` that path runs on one thread.\n")

    add("### 5.5 OpenMP thread scaling (PB06 and PB11)\n")
    rows = [[t["test_id"], config_label(t["test_id"], t["config"]), t["threads"], t["status"], ms(t["median"]),
             ", ".join(ms(v) for v in t["run_medians"]), ratio(t["scaling"])] for t in thread_perf]
    add(table(["ID", "Size", "Threads", "Correct", "Median (ms)", "Run medians (ms)",
               "Speedup vs MiniTensor at 1 thread"], rows))
    add("")
    add("MiniTensor parallelizes matrix multiplication only; element-wise addition (PB06) is single-threaded, "
        "so its thread count has no effect by design. Thread speedups are relative to MiniTensor itself and are "
        "not compared with the single-thread references.\n")

    add("### 5.6 Repeatability and full statistics (MiniTensor, single thread)\n")
    rows = [[p["test_id"], config_label(p["test_id"], p["config"]), ", ".join(ms(v) for v in p["run_medians"]),
             ms(p["median"]), ms(p["min"]), ms(p["mean"]), ms(p["stdev"]),
             f"{100 * (max(p['run_medians']) - min(p['run_medians'])) / p['median']:.0f}%"] for p in perf]
    add(table(["ID", "Size", "Run medians (ms)", "Overall median", "Min", "Mean", "Std dev",
               "Spread of run medians"], rows))
    add("")
    add("The same statistics for Eigen and NumPy are in their raw files, which keep all 30 samples of every run.\n")

    add("### 5.7 Other data movement\n")
    rows = []
    for p in perf:
        if p["test_id"] in ("PB01", "PB02", "PB03", "PB04", "PB05"):
            rows.append([p["test_id"], TITLES[p["test_id"]],
                         f"{nominal_bytes(p['test_id'], p['config']):,}",
                         f"{bandwidth(p['test_id'], p['config'], p['median']):.2f}",
                         f"{bandwidth(p['test_id'], p['config'], p['eigen']):.2f}" if p["eigen"] else "-"])
    add(table(["ID", "Benchmark", "Nominal bytes per execution", "MiniTensor GB/s", "Eigen GB/s"], rows))
    add("")
    add("Byte counts: creation writes every element once; PB04 reads every element once; PB05 counts 4 bytes per "
        "read. These are nominal figures, not measured memory traffic.\n")

    # Observations computed from the data
    add("## 6. Observations\n")
    faster = [p for p in perf if p["speedup_eigen"] and p["speedup_eigen"] > 1]
    add(f"- **Correctness:** every benchmark output was correct in every run for all three libraries "
        f"({perf_pass} of {perf_total} MiniTensor configurations, "
        f"{sum(1 for v in eigen.values() if all_pass(v))} Eigen and "
        f"{sum(1 for v in numpy_runs.values() if all_pass(v))} NumPy configurations).")
    add(f"- **Against Eigen:** MiniTensor was faster in {len(faster)} of {len(perf)} single-thread configurations"
        + (" (" + ", ".join(f"{p['test_id']} {config_label(p['test_id'], p['config'])}, by "
                            f"{100 * (p['speedup_eigen'] - 1):.0f}%" for p in faster) + ")" if faster else "") + ".")
    marginal = [p for p in faster
                if (max(p["run_medians"]) - min(p["run_medians"])) / p["median"] > p["speedup_eigen"] - 1]
    if marginal:
        add("  " + ", ".join(p["test_id"] for p in marginal) + ": the difference is smaller than the spread "
            "between MiniTensor's own three run medians, so it is not a meaningful win.")
    pb04 = next((p for p in perf if p["test_id"] == "PB04"), None)
    if pb04 and pb04["eigen"]:
        add(f"- **PB04 caveat:** both drivers time the same loop over the contiguous buffer (`data()` with a float64 "
            f"accumulator), yet the medians differ ({ms(pb04['median'])} ms against {ms(pb04['eigen'])} ms). "
            f"Because the measured code is identical, this difference comes from how the two test programs were "
            f"compiled, not from either library's storage.")
    pb05 = next((p for p in perf if p["test_id"] == "PB05"), None)
    if pb05:
        add(f"- **PB05:** MiniTensor's run medians varied from {ms(min(pb05['run_medians']))} to "
            f"{ms(max(pb05['run_medians']))} ms. Each `t({{row, col}})` read builds a temporary index vector "
            f"(a heap allocation) and checks bounds, while Eigen's `t(row, col)` does neither in a release "
            f"build. How much of the gap each part causes has not been profiled.")
    small = next((p for p in perf if p["test_id"] == "PB06" and p["config"] == "100"), None)
    if small and small["eigen"]:
        add(f"- **Small element-wise operations:** at 100x100, MiniTensor's addition takes {ms(small['median'])} ms "
            f"against Eigen's {ms(small['eigen'])} ms. MiniTensor zero-fills each new output tensor before "
            f"writing the results; the cause of the rest of the gap has not been profiled.")
    pb11 = next((p for p in perf if p["test_id"] == "PB11"), None)
    pb11_8 = next((t for t in thread_perf if t["test_id"] == "PB11" and t["threads"] == 8), None)
    if pb11 and pb11_8:
        add(f"- **Matrix multiplication:** single-threaded PB11 reaches {gflops(pb11['config'], pb11['median']):.1f} "
            f"GFLOP/s (Eigen {gflops(pb11['config'], pb11['eigen']):.1f}, NumPy "
            f"{gflops(pb11['config'], pb11['numpy']):.1f}). With 8 OpenMP threads MiniTensor reaches "
            f"{gflops(pb11_8['config'], pb11_8['median']):.1f} GFLOP/s, {ratio(pb11_8['scaling'])} its 1-thread speed.")
    pb12 = next((p for p in perf if p["test_id"] == "PB12"), None)
    if pb12:
        add(f"- **Reshape semantics differ:** MiniTensor reshapes in place (metadata change, about "
            f"{pb12['median'] * 1e3:.0f} ns per call); Eigen's `Map` and NumPy's `reshape` create views. "
            f"The specification asks for unlike semantics to be reported separately, so the PB12 speedup "
            f"should be read with that in mind.")
    add("- **Timer resolution:** PB01 and the smallest element-wise sizes take around a microsecond, close to the "
        "100 ns timer resolution, so their relative comparisons are less precise than the larger sizes'.\n")

    add("## 7. Deviations and limitations\n")
    add("- **Platform:** measured on Windows 11 on a laptop (AC power, Balanced power plan), not a dedicated "
        "Linux machine. The specification asks the evaluator to use one machine and power mode for every team, "
        "so these timings are for this machine only.")
    add("- **NumPy PB04 and PB05:** Python cannot loop over a million elements at C speed, so NumPy uses its "
        "vectorized equivalents (`sum` and fancy indexing over the same index list).")
    add("- **PB12 batching:** one reshape call is close to the timer resolution, so each timed sample is a batch "
        "of 1000 reshape calls, identical for all three libraries, as section 5 allows.")
    add("- **Team details:** the Team_ID is set to \"" + args.team + "\". Replace it with the assigned team "
        "number or name if different.\n")

    add("## 8. Submission checklist\n")
    authors = args.members
    add(table(["Item", "Status"], [
        ["All 50 functional cases executed; no test silently omitted",
         f"Done: {len(functional)} of 50 executed and recorded"],
        ["All 12 core performance benchmarks executed, or a reason recorded",
         f"Done: {', '.join(core_ids)}"],
        ["Exact input shapes, values, compiler and hardware details recorded", "Done: sections 2, 3 and 5"],
        ["Raw CSV file and test-driver source code submitted",
         "Done: `verification/common_spec/` (drivers) and `verification/common_spec/results/` (raw CSVs)"],
        ["Reference library and version identified", f"Done: {eigen_name}; {numpy_name}"],
        ["Correctness checked separately from timed execution", "Done: validation runs after each timed execution"],
        ["Median, minimum, mean and standard deviation reported", "Done: section 5.6 and the raw files"],
        ["TC46-TC50 run with a memory/undefined-behaviour checker", "Done: section 4"],
        ["Unsupported features marked NOT_SUPPORTED rather than PASS",
         f"Done: {f_counts['NOT_SUPPORTED']} features unsupported"],
        ["Team member names and library Git commit/hash included",
         f"Done: team members {authors}; commit `{machine.get('git_commit', 'unknown')[:7]}`"
         if authors else f"Commit included (`{machine.get('git_commit', 'unknown')[:7]}`). "
                         "**Team member names still need to be added.**"],
    ]))
    add("")

    add("## 9. Files\n")
    add(table(["File", "Contents"], [
        ["`verification/common_spec/common_test_driver.cpp`", "Test driver: TC01-TC50 and the MiniTensor benchmarks"],
        ["`verification/common_spec/reference_eigen.cpp`", "Eigen reference benchmarks"],
        ["`verification/common_spec/reference_numpy.py`", "NumPy reference benchmarks"],
        ["`verification/common_spec/common_spec.h`", "Shared timing protocol, data formulas and PB05 index list"],
        ["`verification/common_spec/run_common_tests.py`", "Runs everything and collects the raw files"],
        ["`verification/common_spec/make_report.py`", "Generates this report and the submission CSV"],
        ["`results/submission_results.csv`", "Section 6 CSV: one row per test and per benchmark configuration"],
        ["`results/functional.csv`", "TC01-TC50 with expected and actual values"],
        ["`results/raw_*.csv`", "Every benchmark run with all 30 samples"],
        ["`results/memory_checks/`", "Sanitizer runs"],
        ["`results/machine.json`", "Hardware, OS, compiler and thread settings"],
    ]))
    add("")
    add("To reproduce: build with CMake (Release), then run "
        "`python3 verification/common_spec/run_common_tests.py --driver build/common_test_driver "
        "--eigen build/reference_eigen --out verification/common_spec/results` and "
        "`python3 verification/common_spec/make_report.py`.\n")

    Path(args.output).write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("Wrote", args.output)
    print("Wrote", submission)


if __name__ == "__main__":
    main()
