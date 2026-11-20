#!/usr/bin/env python3
"""Runs the Diwali 2026 common test specification and collects raw results.

    python3 run_common_tests.py --driver build/common_test_driver \
        --eigen build/reference_eigen --out results/

Writes to --out:
    machine.json                   hardware, OS, compilers, power mode, thread settings
    functional.csv                 TC01-TC50 in the specification's CSV columns
    raw_minitensor.csv             every benchmark run, single thread (all 30 samples)
    raw_eigen.csv, raw_numpy.csv   the same benchmarks for the references
    raw_minitensor_threads.csv     PB06 and PB11 at 1, 2, 4 and 8 OpenMP threads

Every configuration runs three independent times (separate processes). The
runs are interleaved: all configurations once, then all again, so slow drift
in clock speed or temperature affects every library alike.

This script only collects data. It does not summarise or compare it.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = 3

# (test_id, config). Element-wise sizes are N for NxN; matmul is MxKxN.
CONFIGS = [
    ("PB01", "100x100"),
    ("PB02", "1000x1000"),
    ("PB03", "4000x4000"),
    ("PB04", "1000x1000"),
    ("PB05", "1000x1000"),
    *[(op, size) for op in ("PB06", "PB07", "PB08") for size in ("100", "500", "1000", "4000")],
    ("PB09", "64x64x64"),
    ("PB10", "256x256x256"),
    ("PB11", "512x512x512"),
    # Section 4.3 matmul scaling. 256x256 and 512x512 are PB10 and PB11.
    ("MATMUL", "32x32x32"),
    ("MATMUL", "128x128x128"),
    ("MATMUL", "128x256x64"),
    ("MATMUL", "512x128x256"),
    ("MATMUL", "1024x64x512"),
    ("PB12", "1000x1000 to 100x10000"),
]

THREAD_CONFIGS = [("PB06", "1000"), ("PB11", "512x512x512")]
THREAD_COUNTS = [1, 2, 4, 8]


def output_of(command):
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (result.stdout or result.stderr).strip() or None


def machine_info(args):
    info = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "os": platform.platform(),
        "logical_processors": os.cpu_count(),
        "python": platform.python_version(),
        "git_commit": output_of(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_branch": output_of(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_uncommitted_files": output_of(["git", "-C", str(ROOT), "status", "--porcelain"]),
        "student_compiler": output_of([args.cxx, "--version"]),
        "build_flags": args.build_flags,
        "threads": "OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 for every single-thread run; "
                   "Eigen built with EIGEN_DONT_PARALLELIZE",
    }
    try:
        import numpy
        info["numpy"] = numpy.__version__
        blas = numpy.show_config(mode="dicts")["Build Dependencies"]["blas"]
        info["numpy_blas"] = f"{blas.get('name')} {blas.get('version')}"
    except Exception as error:  # noqa: BLE001 - informational only
        info["numpy_error"] = str(error)

    if platform.system() == "Windows":
        query = ("$p=Get-CimInstance Win32_Processor; $c=Get-CimInstance Win32_ComputerSystem; "
                 "$b=Get-CimInstance Win32_Battery; "
                 "'{0}|{1}|{2}|{3}|{4}' -f $p.Name,$p.NumberOfCores,$p.MaxClockSpeed,"
                 "[math]::Round($c.TotalPhysicalMemory/1GB,1),$b.BatteryStatus")
        text = output_of(["powershell", "-NoProfile", "-Command", query])
        if text:
            name, cores, mhz, ram, battery = (text.split("|") + [""] * 5)[:5]
            info.update(cpu=name.strip(), physical_cores=cores, max_clock_mhz=mhz, ram_gb=ram,
                        power_source="AC" if battery.strip() == "2" else f"battery status {battery.strip()}")
        info["power_plan"] = output_of(["powercfg", "/getactivescheme"])
        frequency = output_of(["powershell", "-NoProfile", "-Command", "[Diagnostics.Stopwatch]::Frequency"])
        if frequency:
            info["timer"] = (f"std::chrono::steady_clock uses QueryPerformanceCounter at {frequency} Hz "
                             f"(resolution {1e9 / int(frequency):.0f} ns)")
    else:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    info["ram_gb"] = round(int(line.split()[1]) / 2**20, 1)
            info["cpu_governor"] = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
        except OSError:
            pass
    return info


def run(command, threads=1, log=None):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    line = (result.stdout.strip().splitlines() or [""])[-1]
    print(f"  {line}" if result.returncode in (0, 1) else f"  ERROR {result.returncode}: {result.stderr.strip()}",
          flush=True)
    if log is not None:
        log.write(f"{' '.join(command)} [threads={threads}] -> {result.returncode}\n{result.stdout}{result.stderr}\n")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--driver", required=True, help="common_test_driver executable")
    parser.add_argument("--eigen", help="reference_eigen executable (skipped if omitted)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--team", default="MiniTensor")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--build-flags", default="-std=c++17 -O3 -DNDEBUG -fopenmp")
    parser.add_argument("--skip-numpy", action="store_true")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("raw_minitensor.csv", "raw_eigen.csv", "raw_numpy.csv", "raw_minitensor_threads.csv"):
        (out / name).unlink(missing_ok=True)

    driver = str(Path(args.driver).resolve())
    eigen = str(Path(args.eigen).resolve()) if args.eigen else None
    numpy_script = str(HERE / "reference_numpy.py")
    indices = str(out / "pb05_indices.bin")

    (out / "machine.json").write_text(json.dumps(machine_info(args), indent=2), encoding="utf-8")

    with open(out / "run_log.txt", "w", encoding="utf-8") as log:
        print("Functional tests TC01-TC50", flush=True)
        run([driver, "functional", str(out / "functional.csv"), args.team, "--expect-fail=TC35"], log=log)
        run([driver, "indices", indices], log=log)

        for repeat in range(1, RUNS + 1):
            for test_id, config in CONFIGS:
                print(f"Run {repeat}: {test_id} {config}", flush=True)
                run([driver, "bench", test_id, config, str(repeat), str(out / "raw_minitensor.csv")], log=log)
                if eigen:
                    run([eigen, "bench", test_id, config, str(repeat), str(out / "raw_eigen.csv")], log=log)
                if not args.skip_numpy:
                    run([sys.executable, numpy_script, "bench", test_id, config, str(repeat),
                         str(out / "raw_numpy.csv"), indices], log=log)

        for repeat in range(1, RUNS + 1):
            for test_id, config in THREAD_CONFIGS:
                for threads in THREAD_COUNTS:
                    print(f"Run {repeat}: {test_id} {config} at {threads} threads", flush=True)
                    run([driver, "bench", test_id, config, str(repeat), str(out / "raw_minitensor_threads.csv")],
                        threads=threads, log=log)

    print("Raw results in", out)


if __name__ == "__main__":
    main()
