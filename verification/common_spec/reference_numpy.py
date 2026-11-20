#!/usr/bin/env python3
"""NumPy reference for the Diwali 2026 common test specification benchmarks.

    python3 reference_numpy.py bench <test_id> <config> <run> <raw.csv> <indices.bin>

Same protocol as common_test_driver.cpp: float32 data, 5 untimed warm-ups,
30 timed executions on time.perf_counter_ns (monotonic), inputs created
before timing and checked after it. Set OPENBLAS_NUM_THREADS=1 (or the
equivalent for your BLAS) before starting it for single-thread runs.

Python cannot loop over a million elements at C speed, so PB04 and PB05 use
NumPy's vectorized equivalents (sum and fancy indexing). The notes column says so.
"""

import csv
import os
import platform
import statistics
import sys
import time

import numpy as np

WARMUPS = 5
REPETITIONS = 30
F32 = np.float32


def within_tolerance(actual, expected):
    return np.abs(actual - expected) <= 1e-5 + 1e-4 * np.abs(expected)


def measure(run, consume):
    for _ in range(WARMUPS):
        run()
        consume()
    samples = []
    checksum = 0.0
    for _ in range(REPETITIONS):
        start = time.perf_counter_ns()
        run()
        stop = time.perf_counter_ns()
        samples.append((stop - start) / 1e6)
        checksum = consume()
    return samples, checksum


def formula(rows, cols, is_a):
    i = np.arange(rows).reshape(-1, 1)
    j = np.arange(cols).reshape(1, -1)
    if is_a:
        return (((i + j) % 7).astype(F32) * F32(0.1)).astype(F32)
    return (((2 * i + j) % 5).astype(F32) * F32(0.2)).astype(F32)


def peak_memory_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / (1024.0 * 1024.0)
    except (OSError, AttributeError):
        pass
    return -1.0


def bench(test_id, config, indices_path):
    """Returns (status, checksum, expected, max_abs, max_rel, samples, notes)."""
    holder = {}

    if test_id in ("PB01", "PB02", "PB03"):
        n = {"PB01": 100, "PB02": 1000, "PB03": 4000}[test_id]
        value = 0.0 if test_id == "PB01" else 1.0
        ok = [True]

        def run():
            holder["t"] = np.zeros((n, n), F32) if value == 0.0 else np.full((n, n), value, F32)

        def consume():
            t = holder.pop("t")
            ok[0] = ok[0] and t[0, 0] == value and t[-1, -1] == value
            return float(t.sum(dtype=np.float64))

        samples, checksum = measure(run, consume)
        expected = n * n * value
        status = "PASS" if ok[0] and checksum == expected else "FAIL"
        return status, checksum, expected, 0.0, 0.0, samples, \
            "np.zeros" if value == 0.0 else "np.full(..., 1.0)"

    if test_id == "PB04":
        t = np.arange(1, 1000001, dtype=F32).reshape(1000, 1000)

        def run():
            holder["s"] = t.sum(dtype=np.float64)

        samples, checksum = measure(run, lambda: float(holder["s"]))
        expected = 500000500000.0
        return ("PASS" if checksum == expected else "FAIL"), checksum, expected, 0.0, 0.0, samples, \
            "Vectorized t.sum(dtype=float64), not a Python loop"

    if test_id == "PB05":
        t = np.arange(0, 1000000, dtype=F32).reshape(1000, 1000)
        pairs = np.fromfile(indices_path, dtype="<u4").reshape(-1, 2)
        rows = pairs[:, 0].astype(np.intp)
        cols = pairs[:, 1].astype(np.intp)
        expected = float((rows * 1000 + cols).astype(np.float64).sum())

        def run():
            holder["s"] = t[rows, cols].sum(dtype=np.float64)

        samples, checksum = measure(run, lambda: float(holder["s"]))
        return ("PASS" if checksum == expected else "FAIL"), checksum, expected, 0.0, 0.0, samples, \
            "Vectorized fancy indexing t[rows, cols] in the same order, from the C++ driver's index list"

    if test_id in ("PB06", "PB07", "PB08"):
        n = int(config)
        a_value, b_value, value = {"PB06": (1.25, 2.5, 3.75), "PB07": (5.0, 1.5, 3.5),
                                   "PB08": (1.25, 2.0, 2.5)}[test_id]
        operation = {"PB06": np.add, "PB07": np.subtract, "PB08": np.multiply}[test_id]
        a = np.full((n, n), a_value, F32)
        b = np.full((n, n), b_value, F32)
        ok = [True]

        def run():
            holder["c"] = operation(a, b)

        def consume():
            c = holder.pop("c")
            ok[0] = ok[0] and bool(np.all(c == F32(value)))
            return float(c.sum(dtype=np.float64))

        samples, checksum = measure(run, consume)
        expected = n * n * value
        return ("PASS" if ok[0] and checksum == expected else "FAIL"), checksum, expected, 0.0, 0.0, samples, \
            "Result written to a new array"

    if test_id in ("PB09", "PB10", "PB11", "MATMUL"):
        m, k, n = (int(x) for x in config.split("x"))
        a = formula(m, k, True)
        b = formula(k, n, False)
        reference = a.astype(np.float64) @ b.astype(np.float64)
        state = {"ok": True, "abs": 0.0, "rel": 0.0}

        def run():
            holder["c"] = a @ b

        def consume():
            c = holder.pop("c").astype(np.float64)
            error = np.abs(c - reference)
            state["ok"] = state["ok"] and bool(np.all(within_tolerance(c, reference)))
            state["abs"] = max(state["abs"], float(error.max()))
            nonzero = reference != 0
            if nonzero.any():
                state["rel"] = max(state["rel"], float((error[nonzero] / np.abs(reference[nonzero])).max()))
            return float(c.sum())

        samples, checksum = measure(run, consume)
        return ("PASS" if state["ok"] else "FAIL"), checksum, float(reference.sum()), state["abs"], state["rel"], \
            samples, "Full output compared with a float64 reference of the same inputs"

    if test_id == "PB12":
        t = np.arange(0, 1000000, dtype=F32).reshape(1000, 1000)
        batch = 1000

        def run():
            for _ in range(batch // 2):
                wide = t.reshape(100, 10000)
                square = wide.reshape(1000, 1000)
            holder["v"] = square

        samples, _ = measure(run, lambda: 0.0)
        view = t.reshape(100, 10000)
        ok = np.shares_memory(view, t) and view[37, 4567] == 374567.0 and view[99, 9999] == 999999.0
        checksum = float(view.sum(dtype=np.float64))
        expected = 499999500000.0
        return ("PASS" if ok and checksum == expected else "FAIL"), checksum, expected, 0.0, 0.0, samples, \
            "View (ndarray.reshape returns a view sharing memory). Each sample is one batch of 1000 reshape " \
            "calls alternating {100,10000} and {1000,1000}"

    raise SystemExit(f"Unknown benchmark {test_id}")


def main():
    if len(sys.argv) != 7 or sys.argv[1] != "bench":
        raise SystemExit("Usage: reference_numpy.py bench <test_id> <config> <run> <raw.csv> <indices.bin>")
    _, _, test_id, config, run, path, indices_path = sys.argv

    try:
        status, checksum, expected, max_abs, max_rel, samples, notes = bench(test_id, config, indices_path)
    except MemoryError:
        status, checksum, expected, max_abs, max_rel, samples, notes = \
            "OUT_OF_MEMORY", 0.0, 0.0, 0.0, 0.0, [], "MemoryError"

    sorted_samples = sorted(samples)
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_MINIMAL)
        if fresh:
            writer.writerow(["library", "test_id", "config", "threads", "run", "status", "checksum",
                             "expected_checksum", "max_abs_error", "max_rel_error", "min_ms", "median_ms",
                             "mean_ms", "stddev_ms", "peak_memory_mb", "samples_ms", "notes"])
        writer.writerow([
            f"NumPy {np.__version__}", test_id, config, os.environ.get("OPENBLAS_NUM_THREADS", "default"),
            run, status, repr(checksum), repr(expected), f"{max_abs:.6g}", f"{max_rel:.6g}",
            f"{sorted_samples[0]:.8g}" if samples else "", f"{statistics.median(samples):.8g}" if samples else "",
            f"{statistics.fmean(samples):.8g}" if samples else "",
            f"{statistics.stdev(samples):.8g}" if len(samples) > 1 else "",
            f"{peak_memory_mb():.6g}", ";".join(f"{s:.8g}" for s in samples),
            f"{notes}. Python {platform.python_version()}",
        ])
    print(f"{test_id} {config} run {run}: {status}, median {statistics.median(samples) if samples else 0:.4f} ms")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
