// Repeated CPU timings of the MiniTensor operations, written as CSV.
//
// Usage: benchmark_cpu <results.csv> [repeats]
//
// Each workload runs 3 warm-ups that are discarded, then `repeats` timed runs.
// Only the operation itself is inside the timer. Inputs come from a fixed seed
// and every run's result is hashed, which keeps the compiler from removing the
// work and shows whether all repeats produced identical bits.
// The thread count comes from OMP_NUM_THREADS.

#include "benchmark_common.h"
#include "minitensor/minitensor.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace minitensor;
using namespace benchmark_common;

namespace {

int thread_count() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void measure(std::ofstream& csv, const std::string& workload, const std::string& size,
             int repeats, const std::function<Tensor()>& operation) {
    using clock = std::chrono::steady_clock;

    for (int i = 0; i < warmups; ++i) {
        (void)hash_of(operation());
    }

    std::vector<double> times;
    std::uint64_t first_hash = 0;
    bool repeatable = true;

    for (int i = 0; i < repeats; ++i) {
        const auto start = clock::now();
        const Tensor result = operation();
        const auto stop = clock::now();

        times.push_back(std::chrono::duration<double, std::milli>(stop - start).count());

        const std::uint64_t hash = hash_of(result);
        if (i == 0) {
            first_hash = hash;
        }
        repeatable = repeatable && hash == first_hash;
    }

    const Stats stats = summarize(times);

    csv << "MiniTensor," << thread_count() << ',' << workload << ',' << size << ','
        << warmups << ',' << repeats << ',' << stats.min << ',' << stats.median << ','
        << stats.mean << ',' << stats.max << ',' << stats.stddev << ','
        << (repeatable ? "yes" : "no") << ',' << first_hash << '\n';
    csv.flush();

    std::cout << workload << ' ' << size << ": median " << stats.median << " ms\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: benchmark_cpu <results.csv> [repeats]\n";
        return 2;
    }

    try {
        const int repeats = argc > 2 ? std::atoi(argv[2]) : 15;
        if (repeats < 1) {
            throw std::invalid_argument("repeats must be positive");
        }

        std::ofstream csv(argv[1], std::ios::binary);
        if (!csv) {
            throw std::runtime_error(std::string("Cannot write ") + argv[1]);
        }
        csv << "implementation,threads,workload,size,warmups,repeats,min_ms,median_ms,"
               "mean_ms,max_ms,stddev_ms,identical_bits,result_hash\n";

        for (const std::size_t n : {64, 128, 256, 512, 1024}) {
            const Tensor a = seeded({n, n}, 1);
            const Tensor b = seeded({n, n}, 2);
            measure(csv, "matmul", std::to_string(n) + "x" + std::to_string(n), repeats,
                    [&] { return matmul(a, b); });
        }

        for (const std::size_t n : {1024, 4096}) {
            const Tensor a = seeded({n, n}, 3);
            measure(csv, "transpose", std::to_string(n) + "x" + std::to_string(n), repeats,
                    [&] { return transpose(a); });
        }

        {
            const std::size_t n = 2048;
            const Tensor a = seeded({n, n}, 4);
            const Tensor b = seeded({n, n}, 5);
            const Tensor row = seeded({n}, 6);
            const std::string size = std::to_string(n) + "x" + std::to_string(n);

            measure(csv, "add", size, repeats, [&] { return add(a, b); });
            measure(csv, "add_broadcast", size + " + " + std::to_string(n), repeats,
                    [&] { return add_broadcast(a, row); });
            measure(csv, "sum", size, repeats, [&] { return Tensor({1}, sum(a)); });
        }

        std::cout << "openmp=" << (openmp_enabled() ? 1 : 0)
                  << " threads=" << thread_count() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
