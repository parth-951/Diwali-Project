// Eigen reference for the Diwali 2026 common test specification benchmarks.
//
//   reference_eigen bench <test_id> <config> <run> <raw.csv>
//
// Same protocol, data and checks as common_test_driver.cpp (shared through
// common_spec.h), using row-major float32 Eigen matrices. Built with the same
// compiler and flags as MiniTensor, with EIGEN_DONT_PARALLELIZE so it runs on
// one thread.

#define EIGEN_DONT_PARALLELIZE

#include "common_spec.h"

#include <Eigen/Dense>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <optional>
#include <string>
#include <vector>

using namespace common_spec;

namespace {

using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

double matrix_sum(const Matrix& m)
{
    double total = 0.0;
    for (Eigen::Index i = 0; i < m.size(); ++i) {
        total += m.data()[i];
    }
    return total;
}

Matrix filled_with_index(Eigen::Index rows, Eigen::Index cols, float first)
{
    Matrix m(rows, cols);
    for (Eigen::Index i = 0; i < m.size(); ++i) {
        m.data()[i] = first + static_cast<float>(i);
    }
    return m;
}

Matrix formula_matrix(std::size_t rows, std::size_t cols, bool is_a)
{
    Matrix m(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            m(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = is_a ? matrix_a(i, j) : matrix_b(i, j);
        }
    }
    return m;
}

void creation_bench(BenchRow& row, Eigen::Index n, float value, bool zero)
{
    std::optional<Matrix> holder;
    bool values_ok = true;

    const Measurement m = measure(
        [&] {
            if (zero) {
                holder.emplace(Matrix::Zero(n, n));
            } else {
                holder.emplace(Matrix::Constant(n, n, value));
            }
        },
        [&] {
            const double total = matrix_sum(*holder);
            values_ok = values_ok && holder->data()[0] == value && holder->data()[holder->size() - 1] == value;
            holder.reset();
            return total;
        });

    row.samples_ms = m.samples_ms;
    row.checksum = m.checksum;
    row.expected_checksum = static_cast<double>(n) * static_cast<double>(n) * value;
    row.status = (values_ok && row.checksum == row.expected_checksum) ? "PASS" : "FAIL";
    row.notes = zero ? "Matrix::Zero(N,N)" : "Matrix::Constant(N,N,1.0f)";
}

void elementwise_bench(BenchRow& row, Eigen::Index n, char op)
{
    const float a_value = op == '+' ? 1.25f : op == '-' ? 5.0f : 1.25f;
    const float b_value = op == '+' ? 2.5f : op == '-' ? 1.5f : 2.0f;
    const float expected = op == '+' ? 3.75f : op == '-' ? 3.5f : 2.5f;

    const Matrix a = Matrix::Constant(n, n, a_value);
    const Matrix b = Matrix::Constant(n, n, b_value);
    std::optional<Matrix> c;
    bool values_ok = true;

    const Measurement m = measure(
        [&] {
            if (op == '+') {
                c.emplace(a + b);
            } else if (op == '-') {
                c.emplace(a - b);
            } else {
                c.emplace(a.cwiseProduct(b));
            }
        },
        [&] {
            for (Eigen::Index i = 0; i < c->size(); ++i) {
                values_ok = values_ok && c->data()[i] == expected;
            }
            const double total = matrix_sum(*c);
            c.reset();
            return total;
        });

    row.samples_ms = m.samples_ms;
    row.checksum = m.checksum;
    row.expected_checksum = static_cast<double>(n) * static_cast<double>(n) * expected;
    row.status = (values_ok && row.checksum == row.expected_checksum) ? "PASS" : "FAIL";
    row.notes = "Result evaluated into a new matrix";
}

void matmul_bench(BenchRow& row, const MatmulShape& s)
{
    const Matrix a = formula_matrix(s.m, s.k, true);
    const Matrix b = formula_matrix(s.k, s.n, false);
    const std::vector<double> reference = reference_matmul(s.m, s.k, s.n);

    std::optional<Matrix> c;
    Accuracy accuracy;
    double expected_sum = 0.0;
    for (const double v : reference) {
        expected_sum += v;
    }

    const Measurement m = measure(
        [&] { c.emplace(a * b); },
        [&] {
            Accuracy this_run;
            for (std::size_t i = 0; i < reference.size(); ++i) {
                accumulate_accuracy(this_run, c->data()[i], reference[i]);
            }
            accuracy.ok = accuracy.ok && this_run.ok;
            accuracy.max_abs_error = std::max(accuracy.max_abs_error, this_run.max_abs_error);
            accuracy.max_rel_error = std::max(accuracy.max_rel_error, this_run.max_rel_error);
            const double total = matrix_sum(*c);
            c.reset();
            return total;
        });

    row.samples_ms = m.samples_ms;
    row.checksum = m.checksum;
    row.expected_checksum = expected_sum;
    row.max_abs_error = accuracy.max_abs_error;
    row.max_rel_error = accuracy.max_rel_error;
    row.status = accuracy.ok ? "PASS" : "FAIL";
    row.notes = "Full output compared with a double-precision reference";
}

int run_bench(const std::string& test_id, const std::string& config, int run, const std::string& path)
{
    BenchRow row;
    row.library = "Eigen " + std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION)
        + "." + std::to_string(EIGEN_MINOR_VERSION);
    row.test_id = test_id;
    row.config = config;
    row.run = run;

    try {
        if (test_id == "PB01") {
            creation_bench(row, 100, 0.0f, true);
        } else if (test_id == "PB02") {
            creation_bench(row, 1000, 1.0f, false);
        } else if (test_id == "PB03") {
            creation_bench(row, 4000, 1.0f, false);
        } else if (test_id == "PB04") {
            const Matrix t = filled_with_index(1000, 1000, 1.0f);
            double result = 0.0;
            const Measurement m = measure(
                [&] {
                    double total = 0.0;
                    const float* data = t.data();
                    for (Eigen::Index i = 0; i < t.size(); ++i) {
                        total += data[i];
                    }
                    result = total;
                },
                [&] { return result; });
            row.samples_ms = m.samples_ms;
            row.checksum = m.checksum;
            row.expected_checksum = 500000500000.0;
            row.status = row.checksum == row.expected_checksum ? "PASS" : "FAIL";
            row.notes = "Contiguous row-major traversal through data(); float64 accumulator";
        } else if (test_id == "PB05") {
            const Matrix t = filled_with_index(1000, 1000, 0.0f);
            const auto indices = random_indices();
            double expected = 0.0;
            for (const auto& index : indices) {
                expected += static_cast<double>(index.first * 1000 + index.second);
            }
            double result = 0.0;
            const Measurement m = measure(
                [&] {
                    double total = 0.0;
                    for (const auto& index : indices) {
                        total += t(static_cast<Eigen::Index>(index.first), static_cast<Eigen::Index>(index.second));
                    }
                    result = total;
                },
                [&] { return result; });
            row.samples_ms = m.samples_ms;
            row.checksum = m.checksum;
            row.expected_checksum = expected;
            row.status = row.checksum == row.expected_checksum ? "PASS" : "FAIL";
            row.notes = "t(row, col) with no bounds check (NDEBUG). Same index list as MiniTensor";
        } else if (test_id == "PB06" || test_id == "PB07" || test_id == "PB08") {
            const char op = test_id == "PB06" ? '+' : test_id == "PB07" ? '-' : '*';
            elementwise_bench(row, static_cast<Eigen::Index>(std::stol(config)), op);
        } else if (test_id == "PB09" || test_id == "PB10" || test_id == "PB11" || test_id == "MATMUL") {
            matmul_bench(row, parse_matmul(config));
        } else if (test_id == "PB12") {
            Matrix t = filled_with_index(1000, 1000, 0.0f);
            constexpr int batch = 1000;
            const Measurement m = measure(
                [&] {
                    for (int i = 0; i < batch / 2; ++i) {
                        Eigen::Map<Matrix> wide(t.data(), 100, 10000);
                        do_not_optimize(wide);
                        Eigen::Map<Matrix> square(t.data(), 1000, 1000);
                        do_not_optimize(square);
                    }
                },
                [&] { return 0.0; });
            Eigen::Map<Matrix> view(t.data(), 100, 10000);
            const bool ok = view(37, 4567) == 374567.0f && view(99, 9999) == 999999.0f;
            row.samples_ms = m.samples_ms;
            row.checksum = matrix_sum(t);
            row.expected_checksum = 499999500000.0;
            row.status = (ok && row.checksum == row.expected_checksum) ? "PASS" : "FAIL";
            row.notes = "View (Eigen::Map over the same buffer, no copy). Each sample is one batch of 1000 "
                        "view creations alternating {100,10000} and {1000,1000}";
        } else {
            std::cerr << "Unknown benchmark " << test_id << '\n';
            return 2;
        }
    } catch (const std::bad_alloc&) {
        row.status = "OUT_OF_MEMORY";
        row.notes = "std::bad_alloc";
    }

    if (!row.samples_ms.empty()) {
        row.stats = summarize(row.samples_ms);
    }
    row.peak_mb = peak_memory_mb();
    row.notes += std::string(". SIMD: ") + Eigen::SimdInstructionSetsInUse();
    append_bench_row(path, row);

    std::cout << test_id << ' ' << config << " run " << run << ": " << row.status << ", median "
              << row.stats.median << " ms\n";
    return row.status == "PASS" ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 6 && std::string(argv[1]) == "bench") {
            return run_bench(argv[2], argv[3], std::atoi(argv[4]), argv[5]);
        }
        std::cerr << "Usage: reference_eigen bench <test_id> <config> <run> <raw.csv>\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 2;
    }
}
