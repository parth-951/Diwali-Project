// Compares MiniTensor against the NumPy answers from generate_reference.py.
//
// Usage: verify_numerical <reference.bin> <results.csv> [cpu|cuda]
//
// With "cuda", operations run on the GPU backend, and the ones it doesn't have
// (matmul_parallel, mean, min, max, reshape) are skipped.
//
// A case passes when the shape matches exactly and the largest element-wise
// difference is at most 1e-10 * max(1, largest |expected|). Results are not
// expected to be bit-identical because NumPy's BLAS adds terms in a different
// order. Exits non-zero if any case fails.

#include "minitensor/minitensor.h"

#ifdef MINITENSOR_WITH_CUDA
#include "minitensor/Cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace minitensor;

namespace {

constexpr double tolerance = 1e-10;

class Reader {
public:
    explicit Reader(const std::string& path) : in_(path, std::ios::binary) {
        if (!in_) {
            throw std::runtime_error("Cannot open " + path);
        }
        const std::string magic = "MTREF1\n";
        std::string found(magic.size(), '\0');
        in_.read(&found[0], static_cast<std::streamsize>(found.size()));
        if (found != magic) {
            throw std::runtime_error("Not a reference file: " + path);
        }
    }

    bool at_end() {
        return in_.peek() == std::char_traits<char>::eof();
    }

    std::uint32_t u32() {
        std::uint32_t value = 0;
        read_bytes(&value, sizeof(value));
        return value;
    }

    std::string text() {
        std::string value(u32(), '\0');
        read_bytes(&value[0], value.size());
        return value;
    }

    Tensor tensor() {
        std::vector<std::size_t> dimensions(u32());
        for (std::size_t& dimension : dimensions) {
            std::uint64_t value = 0;
            read_bytes(&value, sizeof(value));
            dimension = static_cast<std::size_t>(value);
        }
        Tensor result(dimensions);
        read_bytes(result.data(), result.size() * sizeof(double));
        return result;
    }

private:
    void read_bytes(void* target, std::size_t count) {
        in_.read(static_cast<char*>(target), static_cast<std::streamsize>(count));
        if (!in_) {
            throw std::runtime_error("Reference file is truncated");
        }
    }

    std::ifstream in_;
};

Tensor scalar(double value) {
    return Tensor({1}, value);
}

Tensor run_operation(const std::string& op, const std::vector<Tensor>& in) {
    if (op == "matmul") return matmul(in[0], in[1]);
    if (op == "matmul_parallel") return matmul_parallel(in[0], in[1]);
    if (op == "add") return add(in[0], in[1]);
    if (op == "subtract") return subtract(in[0], in[1]);
    if (op == "multiply") return multiply(in[0], in[1]);
    if (op == "add_broadcast") return add_broadcast(in[0], in[1]);
    if (op == "subtract_broadcast") return subtract_broadcast(in[0], in[1]);
    if (op == "multiply_broadcast") return multiply_broadcast(in[0], in[1]);
    if (op == "transpose") return transpose(in[0]);
    if (op == "sum") return scalar(sum(in[0]));
    if (op == "mean") return scalar(mean(in[0]));
    if (op == "min") return scalar(min(in[0]));
    if (op == "max") return scalar(max(in[0]));
    if (op == "reshape") {
        std::vector<std::size_t> target;
        for (std::size_t i = 0; i < in[1].size(); ++i) {
            target.push_back(static_cast<std::size_t>(in[1].data()[i]));
        }
        Tensor result = in[0];
        result.reshape(target);
        return result;
    }
    throw std::logic_error("Unknown operation in reference file: " + op);
}

#ifdef MINITENSOR_WITH_CUDA
bool on_cuda_backend(const std::string& op) {
    return op != "matmul_parallel" && op != "mean" && op != "min" && op != "max"
        && op != "reshape";
}

Tensor run_on_cuda(const std::string& op, const std::vector<Tensor>& in) {
    if (op == "matmul") return cuda::matmul(in[0], in[1]);
    if (op == "add") return cuda::add(in[0], in[1]);
    if (op == "subtract") return cuda::subtract(in[0], in[1]);
    if (op == "multiply") return cuda::multiply(in[0], in[1]);
    if (op == "add_broadcast") return cuda::add_broadcast(in[0], in[1]);
    if (op == "subtract_broadcast") return cuda::subtract_broadcast(in[0], in[1]);
    if (op == "multiply_broadcast") return cuda::multiply_broadcast(in[0], in[1]);
    if (op == "transpose") return cuda::transpose(in[0]);
    if (op == "sum") return scalar(cuda::sum(in[0]));
    throw std::logic_error("Not on the CUDA backend: " + op);
}
#endif

struct Comparison {
    bool shape_ok = false;
    double max_abs_error = 0.0;
    double relative_error = 0.0;
    bool passed = false;
};

Comparison compare(const Tensor& actual, const Tensor& expected) {
    Comparison result;
    result.shape_ok = actual.shape().dimensions() == expected.shape().dimensions();
    if (!result.shape_ok) {
        return result;
    }

    double scale = 1.0;
    bool values_ok = true;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double difference = std::fabs(actual.data()[i] - expected.data()[i]);
        // Written so that a NaN in the result counts as a failure.
        if (!(difference <= result.max_abs_error)) {
            result.max_abs_error = difference;
            values_ok = values_ok && std::isfinite(difference);
        }
        scale = std::max(scale, std::fabs(expected.data()[i]));
    }

    result.relative_error = result.max_abs_error / scale;
    result.passed = values_ok && result.relative_error <= tolerance;
    return result;
}

// FNV-1a over the raw result bytes, to detect any change between runs.
std::uint64_t hash_of(const Tensor& tensor) {
    std::uint64_t hash = 14695981039346656037ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(tensor.data());
    for (std::size_t i = 0; i < tensor.size() * sizeof(double); ++i) {
        hash = (hash ^ bytes[i]) * 1099511628211ULL;
    }
    return hash;
}

std::string shape_text(const std::vector<std::size_t>& dimensions) {
    std::string text;
    for (std::size_t i = 0; i < dimensions.size(); ++i) {
        text += (i == 0 ? "" : "x") + std::to_string(dimensions[i]);
    }
    return text;
}

std::string format_error(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.3e", value);
    return buffer;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: verify_numerical <reference.bin> <results.csv> [cpu|cuda]\n";
        return 2;
    }

    const bool use_cuda = argc == 4 && std::string(argv[3]) == "cuda";

    if (argc == 4 && !use_cuda && std::string(argv[3]) != "cpu") {
        std::cerr << "Unknown backend: " << argv[3] << '\n';
        return 2;
    }

#ifndef MINITENSOR_WITH_CUDA
    if (use_cuda) {
        std::cerr << "error: built without MINITENSOR_USE_CUDA\n";
        return 2;
    }
#endif

    const auto execute = [&](const std::string& op, const std::vector<Tensor>& in) {
#ifdef MINITENSOR_WITH_CUDA
        if (use_cuda) {
            return run_on_cuda(op, in);
        }
#endif
        return run_operation(op, in);
    };

    const auto skipped = [&](const std::string& op) {
#ifdef MINITENSOR_WITH_CUDA
        return use_cuda && !on_cuda_backend(op);
#else
        (void)op;
        return false;
#endif
    };

    try {
        Reader reader(argv[1]);
        std::ofstream csv(argv[2], std::ios::binary);
        if (!csv) {
            throw std::runtime_error(std::string("Cannot write ") + argv[2]);
        }
        csv << "group,case,op,input_shapes,output_shape,expected_outcome,"
               "max_abs_error,relative_error,result_hash,status\n";

        std::size_t total = 0;
        std::size_t failed = 0;

        const auto record = [&](const std::string& group, const std::string& name,
                                const std::string& op, const std::string& inputs,
                                const std::string& output, const std::string& outcome,
                                const std::string& abs_error, const std::string& rel_error,
                                const std::string& hash, bool passed) {
            csv << group << ',' << name << ',' << op << ',' << inputs << ',' << output << ','
                << outcome << ',' << abs_error << ',' << rel_error << ',' << hash << ','
                << (passed ? "PASS" : "FAIL") << '\n';
            ++total;
            if (!passed) {
                ++failed;
                std::cout << "FAIL " << group << ' ' << name << ' ' << op << '\n';
            }
        };

        // Kept for the negative controls below.
        bool have_control = false;
        Tensor control_actual({1});
        Tensor control_expected({1});

        while (!reader.at_end()) {
            const std::string group = reader.text();
            const std::string name = reader.text();
            const std::string op = reader.text();
            const bool expect_throw = reader.u32() != 0;

            std::vector<Tensor> inputs;
            const std::uint32_t input_count = reader.u32();
            std::string input_shapes;
            for (std::uint32_t i = 0; i < input_count; ++i) {
                inputs.push_back(reader.tensor());
                input_shapes += (i == 0 ? "" : " & ") + shape_text(inputs.back().shape().dimensions());
            }

            if (skipped(op)) {
                if (!expect_throw) {
                    (void)reader.tensor();
                }
                continue;
            }

            if (expect_throw) {
                bool threw = false;
                try {
                    (void)execute(op, inputs);
                } catch (const std::invalid_argument&) {
                    threw = true;
                }
                record(group, name, op, input_shapes, "-", "rejected", "-", "-", "-", threw);
                continue;
            }

            const Tensor expected = reader.tensor();
            const Tensor actual = execute(op, inputs);
            const Comparison result = compare(actual, expected);
            record(group, name, op, input_shapes, shape_text(actual.shape().dimensions()),
                   "matches NumPy", format_error(result.max_abs_error),
                   format_error(result.relative_error),
                   std::to_string(hash_of(actual)), result.passed);

            if (!have_control && group == "matmul" && expected.size() > 1) {
                control_actual = actual;
                control_expected = expected;
                have_control = true;
            }
        }

        if (!have_control) {
            throw std::runtime_error("Reference file has no matmul case for the controls");
        }

        // Negative controls: the comparison above must reject wrong results,
        // otherwise its PASS lines mean nothing.
        Tensor wrong_value = control_actual;
        wrong_value.data()[wrong_value.size() / 2] += 1e-6 * std::max(1.0, std::fabs(wrong_value.data()[0]));
        record("control", "one_value_off_by_1e-6", "compare", "-", "-", "must be caught", "-", "-", "-",
               !compare(wrong_value, control_expected).passed);

        Tensor wrong_shape = control_actual;
        wrong_shape.reshape({wrong_shape.size()});
        record("control", "wrong_shape", "compare", "-", "-", "must be caught", "-", "-", "-",
               !compare(wrong_shape, control_expected).passed);

        Tensor not_a_number = control_actual;
        not_a_number.data()[0] = std::nan("");
        record("control", "nan_in_result", "compare", "-", "-", "must be caught", "-", "-", "-",
               !compare(not_a_number, control_expected).passed);

        std::cout << "openmp=" << (openmp_enabled() ? 1 : 0) << " cases=" << total
                  << " failed=" << failed << '\n';
        return failed == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
