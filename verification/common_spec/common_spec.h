#ifndef MINITENSOR_COMMON_SPEC_H
#define MINITENSOR_COMMON_SPEC_H

// Shared by the Diwali 2026 common test specification drivers
// (common_test_driver.cpp for MiniTensor, reference_eigen.cpp for Eigen), so
// that both follow the same protocol on the same data.
//
// Protocol (spec section 4.1): 5 untimed warm-ups, then 30 timed executions
// on a monotonic clock (std::chrono::steady_clock). Inputs are created before
// timing; outputs are checked and checksummed after timing.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace common_spec {

constexpr int warmups = 5;
constexpr int repetitions = 30;

// PB05 index seed from the specification
constexpr std::uint32_t index_seed = 20261117u;

// Spec section 2: abs(actual - expected) <= 1e-5 + 1e-4 * abs(expected)
inline bool within_tolerance(double actual, double expected)
{
    return std::fabs(actual - expected) <= 1e-5 + 1e-4 * std::fabs(expected);
}

// Keeps the compiler from deleting work whose result is otherwise unused
template <typename T>
inline void do_not_optimize(const T& value)
{
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r,m"(value) : "memory");
#else
    static volatile const void* sink;
    sink = &value;
#endif
}

inline int thread_count()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Peak resident memory of this process so far, in MB
inline double peak_memory_mb()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return static_cast<double>(counters.PeakWorkingSetSize) / (1024.0 * 1024.0);
    }
    return -1.0;
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<double>(usage.ru_maxrss) / 1024.0;  // Linux reports KB
    }
    return -1.0;
#endif
}

// PB09-PB11 and the matmul scaling runs (spec 4.2)
inline float matrix_a(std::size_t i, std::size_t j)
{
    return static_cast<float>((i + j) % 7) * 0.1f;
}

inline float matrix_b(std::size_t i, std::size_t j)
{
    return static_cast<float>((2 * i + j) % 5) * 0.2f;
}

// PB05: 100,000 (row, column) pairs for a 1000x1000 tensor. Raw mt19937
// output is fully specified by the C++ standard, so the list is identical on
// every platform. Element value = row * 1000 + column.
inline std::vector<std::pair<std::size_t, std::size_t>> random_indices()
{
    std::mt19937 engine(index_seed);
    std::vector<std::pair<std::size_t, std::size_t>> indices(100000);
    for (auto& index : indices) {
        index.first = engine() % 1000u;
        index.second = engine() % 1000u;
    }
    return indices;
}

struct Stats {
    double min = 0.0;
    double median = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
};

inline Stats summarize(std::vector<double> samples)
{
    std::sort(samples.begin(), samples.end());

    Stats stats;
    stats.min = samples.front();
    stats.mean = std::accumulate(samples.begin(), samples.end(), 0.0)
        / static_cast<double>(samples.size());

    const std::size_t middle = samples.size() / 2;
    stats.median = samples.size() % 2 == 1
        ? samples[middle]
        : (samples[middle - 1] + samples[middle]) / 2.0;

    double squares = 0.0;
    for (const double sample : samples) {
        squares += (sample - stats.mean) * (sample - stats.mean);
    }
    stats.stddev = samples.size() > 1
        ? std::sqrt(squares / static_cast<double>(samples.size() - 1))
        : 0.0;

    return stats;
}

// One timed execution. `run` is timed; `consume` runs after the clock stops,
// returns the checksum, and is where outputs are checked and destroyed.
struct Measurement {
    std::vector<double> samples_ms;
    double checksum = 0.0;
};

inline Measurement measure(const std::function<void()>& run,
                           const std::function<double()>& consume)
{
    using clock = std::chrono::steady_clock;

    for (int i = 0; i < warmups; ++i) {
        run();
        (void)consume();
    }

    Measurement result;
    for (int i = 0; i < repetitions; ++i) {
        const auto start = clock::now();
        run();
        const auto stop = clock::now();

        result.samples_ms.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());

        // Every repetition must produce the same checksum
        result.checksum = consume();
    }
    return result;
}

inline std::string csv_field(const std::string& value)
{
    std::string quoted = "\"";
    for (const char c : value) {
        quoted += c;
        if (c == '"') {
            quoted += '"';
        }
    }
    return quoted + "\"";
}

inline std::string format_number(double value, int digits = 6)
{
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*g", digits, value);
    return buffer;
}

// Raw benchmark row, the same columns for every library
struct BenchRow {
    std::string library;
    std::string test_id;
    std::string config;
    int run = 0;
    std::string status;  // PASS, FAIL, OUT_OF_MEMORY
    double checksum = 0.0;
    double expected_checksum = 0.0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    Stats stats;
    double peak_mb = 0.0;
    std::vector<double> samples_ms;
    std::string notes;
};

inline const char* bench_header()
{
    return "library,test_id,config,threads,run,status,checksum,expected_checksum,"
           "max_abs_error,max_rel_error,min_ms,median_ms,mean_ms,stddev_ms,peak_memory_mb,"
           "samples_ms,notes\n";
}

inline void append_bench_row(const std::string& path, const BenchRow& row)
{
    std::ifstream probe(path);
    const bool fresh = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    probe.close();

    std::ofstream out(path, std::ios::app | std::ios::binary);
    if (fresh) {
        out << bench_header();
    }

    std::string samples;
    for (std::size_t i = 0; i < row.samples_ms.size(); ++i) {
        samples += (i == 0 ? "" : ";") + format_number(row.samples_ms[i], 8);
    }

    out << csv_field(row.library) << ',' << csv_field(row.test_id) << ','
        << csv_field(row.config) << ',' << thread_count() << ',' << row.run << ','
        << row.status << ',' << format_number(row.checksum, 17) << ','
        << format_number(row.expected_checksum, 17) << ','
        << format_number(row.max_abs_error, 6) << ',' << format_number(row.max_rel_error, 6) << ','
        << format_number(row.stats.min, 8) << ',' << format_number(row.stats.median, 8) << ','
        << format_number(row.stats.mean, 8) << ',' << format_number(row.stats.stddev, 8) << ','
        << format_number(row.peak_mb, 6) << ',' << csv_field(samples) << ','
        << csv_field(row.notes) << '\n';
}

// Compares a float32 result with a double reference using the spec tolerance
struct Accuracy {
    bool ok = true;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
};

inline void accumulate_accuracy(Accuracy& accuracy, double actual, double expected)
{
    const double error = std::fabs(actual - expected);
    accuracy.max_abs_error = std::max(accuracy.max_abs_error, error);
    if (expected != 0.0) {
        accuracy.max_rel_error = std::max(accuracy.max_rel_error, error / std::fabs(expected));
    }
    if (!within_tolerance(actual, expected)) {
        accuracy.ok = false;
    }
}

// Double-precision reference product of the PB09 formulas, computed outside timing
inline std::vector<double> reference_matmul(std::size_t m, std::size_t k, std::size_t n)
{
    std::vector<double> c(m * n, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            const double a = matrix_a(i, p);
            for (std::size_t j = 0; j < n; ++j) {
                c[i * n + j] += a * static_cast<double>(matrix_b(p, j));
            }
        }
    }
    return c;
}

// Benchmark configurations, so that both drivers parse identical arguments.
// Square sizes "N" mean NxN; matmul shapes are "MxKxN".
struct MatmulShape {
    std::size_t m = 0;
    std::size_t k = 0;
    std::size_t n = 0;
};

inline MatmulShape parse_matmul(const std::string& text)
{
    MatmulShape shape;
    char x1 = 0;
    char x2 = 0;
    std::istringstream in(text);
    in >> shape.m >> x1 >> shape.k >> x2 >> shape.n;
    if (!in || x1 != 'x' || x2 != 'x') {
        throw std::invalid_argument("matmul shape must look like 256x256x256");
    }
    return shape;
}

} // namespace common_spec

#endif
