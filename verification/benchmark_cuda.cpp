// Repeated GPU timings of the MiniTensor CUDA backend, written as CSV.
//
// Usage: benchmark_cuda <results.csv> [repeats]
//
// Uses the same workloads and input data as benchmark_cpu.cpp. Each workload
// runs 3 warm-ups that are discarded (the first GPU call also pays for setting
// up the CUDA context and cuBLAS), then `repeats` timed runs.
//
// Three times are recorded for each run:
//   gpu        inputs already on the GPU; the clock stops after
//              cuda::synchronize(), so this is the real GPU work
//   launch     the same call with the clock stopped before synchronizing. It
//              shows what a benchmark without synchronization would report.
//   end_to_end Tensor in, Tensor out, including both copies. This is what a
//              caller with host data pays.

#include "benchmark_common.h"
#include "minitensor/Cuda.h"
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

using namespace minitensor;
using namespace benchmark_common;

namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start, Clock::time_point stop)
{
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

// `gpu_operation` queues work on device tensors. `sum` has to block for its
// answer, so its launch time is the same as its GPU time.
void measure(std::ofstream& csv, const std::string& workload, const std::string& size,
             int repeats, const std::function<cuda::DeviceTensor()>& gpu_operation,
             const std::function<Tensor()>& end_to_end)
{
    for (int i = 0; i < warmups; ++i) {
        const cuda::DeviceTensor discarded = gpu_operation();
        cuda::synchronize();
    }

    std::vector<double> gpu_times;
    std::vector<double> launch_times;
    std::vector<double> end_to_end_times;
    std::uint64_t first_hash = 0;
    bool repeatable = true;

    for (int i = 0; i < repeats; ++i) {
        // Nothing from the previous run may still be in the queue.
        cuda::synchronize();

        const auto start = Clock::now();
        const cuda::DeviceTensor result = gpu_operation();
        const auto launched = Clock::now();
        cuda::synchronize();
        const auto finished = Clock::now();

        launch_times.push_back(milliseconds(start, launched));
        gpu_times.push_back(milliseconds(start, finished));

        const std::uint64_t hash = hash_of(result.download());
        if (i == 0) {
            first_hash = hash;
        }
        repeatable = repeatable && hash == first_hash;

        const auto copy_start = Clock::now();
        const Tensor host_result = end_to_end();
        const auto copy_stop = Clock::now();
        end_to_end_times.push_back(milliseconds(copy_start, copy_stop));

        // The copy path has to give the same bits as the queued path.
        repeatable = repeatable && hash_of(host_result) == first_hash;
    }

    const Stats gpu = summarize(gpu_times);
    const Stats launch = summarize(launch_times);
    const Stats copies = summarize(end_to_end_times);

    csv << "MiniTensor CUDA," << workload << ',' << size << ',' << warmups << ',' << repeats
        << ',' << gpu.min << ',' << gpu.median << ',' << gpu.mean << ',' << gpu.max << ','
        << gpu.stddev << ',' << launch.median << ',' << copies.median << ','
        << (repeatable ? "yes" : "no") << ',' << first_hash << '\n';
    csv.flush();

    std::cout << workload << ' ' << size << ": gpu median " << gpu.median
              << " ms, launch " << launch.median << " ms, end to end " << copies.median
              << " ms\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: benchmark_cuda <results.csv> [repeats]\n";
        return 2;
    }

    try {
        const int repeats = argc > 2 ? std::atoi(argv[2]) : 15;
        if (repeats < 1) {
            throw std::invalid_argument("repeats must be positive");
        }

        if (!cuda::available()) {
            throw std::runtime_error("No CUDA device is available");
        }

        std::ofstream csv(argv[1], std::ios::binary);
        if (!csv) {
            throw std::runtime_error(std::string("Cannot write ") + argv[1]);
        }
        csv << "implementation,workload,size,warmups,repeats,min_ms,median_ms,mean_ms,"
               "max_ms,stddev_ms,launch_median_ms,end_to_end_median_ms,identical_bits,"
               "result_hash\n";

        for (const std::size_t n : {64, 128, 256, 512, 1024}) {
            const Tensor a = seeded({n, n}, 1);
            const Tensor b = seeded({n, n}, 2);
            const cuda::DeviceTensor gpu_a(a);
            const cuda::DeviceTensor gpu_b(b);
            measure(csv, "matmul", std::to_string(n) + "x" + std::to_string(n), repeats,
                    [&] { return cuda::matmul(gpu_a, gpu_b); },
                    [&] { return cuda::matmul(a, b); });
        }

        for (const std::size_t n : {1024, 4096}) {
            const Tensor a = seeded({n, n}, 3);
            const cuda::DeviceTensor gpu_a(a);
            measure(csv, "transpose", std::to_string(n) + "x" + std::to_string(n), repeats,
                    [&] { return cuda::transpose(gpu_a); },
                    [&] { return cuda::transpose(a); });
        }

        {
            const std::size_t n = 2048;
            const Tensor a = seeded({n, n}, 4);
            const Tensor b = seeded({n, n}, 5);
            const Tensor row = seeded({n}, 6);
            const cuda::DeviceTensor gpu_a(a);
            const cuda::DeviceTensor gpu_b(b);
            const cuda::DeviceTensor gpu_row(row);
            const std::string size = std::to_string(n) + "x" + std::to_string(n);

            measure(csv, "add", size, repeats,
                    [&] { return cuda::add(gpu_a, gpu_b); },
                    [&] { return cuda::add(a, b); });
            measure(csv, "add_broadcast", size + " + " + std::to_string(n), repeats,
                    [&] { return cuda::add_broadcast(gpu_a, gpu_row); },
                    [&] { return cuda::add_broadcast(a, row); });

            // The total is written into a device tensor so that sum fits the
            // same measuring code. That one-value copy is inside the timer.
            measure(csv, "sum", size, repeats,
                    [&] { return cuda::DeviceTensor(Tensor({1}, cuda::sum(gpu_a))); },
                    [&] { return Tensor({1}, cuda::sum(a)); });
        }

        std::cout << "device=" << cuda::device_name()
                  << " runtime=" << cuda::runtime_version() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
