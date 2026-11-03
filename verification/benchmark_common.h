#ifndef MINITENSOR_BENCHMARK_COMMON_H
#define MINITENSOR_BENCHMARK_COMMON_H

// Shared by benchmark_cpu.cpp and benchmark_cuda.cpp, so both time the same data.

#include "minitensor/Tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace benchmark_common {

constexpr int warmups = 3;

// Raw mt19937_64 output is specified by the standard; distributions are not.
inline minitensor::Tensor seeded(
    const std::vector<std::size_t>& dimensions, std::uint64_t seed)
{
    std::mt19937_64 engine(seed);
    minitensor::Tensor result(dimensions);
    for (std::size_t i = 0; i < result.size(); ++i) {
        result.data()[i] = static_cast<double>(engine() >> 11) * 0x1p-52 - 1.0;
    }
    return result;
}

// FNV-1a over the raw result bytes
inline std::uint64_t hash_of(const minitensor::Tensor& tensor)
{
    std::uint64_t hash = 14695981039346656037ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(tensor.data());
    for (std::size_t i = 0; i < tensor.size() * sizeof(double); ++i) {
        hash = (hash ^ bytes[i]) * 1099511628211ULL;
    }
    return hash;
}

struct Stats {
    double min = 0.0;
    double median = 0.0;
    double mean = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

inline Stats summarize(std::vector<double> times)
{
    std::sort(times.begin(), times.end());

    Stats stats;
    stats.min = times.front();
    stats.max = times.back();
    stats.mean = std::accumulate(times.begin(), times.end(), 0.0)
        / static_cast<double>(times.size());

    const std::size_t middle = times.size() / 2;
    stats.median = times.size() % 2 == 1
        ? times[middle]
        : (times[middle - 1] + times[middle]) / 2.0;

    double squares = 0.0;
    for (const double time : times) {
        squares += (time - stats.mean) * (time - stats.mean);
    }
    stats.stddev = times.size() > 1
        ? std::sqrt(squares / static_cast<double>(times.size() - 1))
        : 0.0;

    return stats;
}

} // namespace benchmark_common

#endif
