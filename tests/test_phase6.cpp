// Phase 6: Integration, property and edge-case tests across the whole library.

#include "minitensor/minitensor.h"
#include "test_utils.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace minitensor;
using minitensor_test::nearly_equal;
using minitensor_test::run_test;
using minitensor_test::tensors_close;

namespace {

// Deterministic small integers in [-5, 5]; sums of products stay exact in double
Tensor make_pattern(const std::vector<std::size_t>& dimensions, std::uint64_t seed)
{
    Tensor tensor(dimensions);

    for (std::size_t i = 0; i < tensor.size(); ++i)
    {
        const std::uint64_t mixed = (i + 1) * 2654435761ULL + seed * 40503ULL;
        tensor.data()[i] = static_cast<double>(static_cast<int>(mixed % 11) - 5);
    }

    return tensor;
}

// Straightforward i-j-k matrix product used as the reference result
Tensor reference_matmul(const Tensor& a, const Tensor& b)
{
    const std::size_t rows = a.shape().dimensions()[0];
    const std::size_t inner = a.shape().dimensions()[1];
    const std::size_t cols = b.shape().dimensions()[1];

    Tensor result({rows, cols});

    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            double total = 0.0;

            for (std::size_t k = 0; k < inner; ++k)
            {
                total += a.data()[i * inner + k] * b.data()[k * cols + j];
            }

            result.data()[i * cols + j] = total;
        }
    }

    return result;
}

Tensor identity(std::size_t n)
{
    Tensor result({n, n});

    for (std::size_t i = 0; i < n; ++i)
    {
        result.at({i, i}) = 1.0;
    }

    return result;
}

Tensor scalar(double value)
{
    return Tensor(std::vector<std::size_t>{}, value);
}

// ---------------------------------------------------------------- matmul

void test_matmul_small_matches_reference()
{
    const Tensor a = make_pattern({3, 4}, 1);
    const Tensor b = make_pattern({4, 5}, 2);
    const Tensor expected = reference_matmul(a, b);

    MT_CHECK(tensors_close(matmul(a, b), expected));
    MT_CHECK(tensors_close(matmul_parallel(a, b), expected));
}

void test_matmul_parallel_path_matches_reference()
{
    // 300 x 260 x 280 is above the hybrid threshold, so matmul() goes parallel
    const Tensor a = make_pattern({300, 260}, 3);
    const Tensor b = make_pattern({260, 280}, 4);
    const Tensor expected = reference_matmul(a, b);

    MT_CHECK(tensors_close(matmul(a, b), expected));
    MT_CHECK(tensors_close(matmul_parallel(a, b), expected));
}

void test_matmul_wide_sequential_path_matches_reference()
{
    // Large work but fewer than 128 rows keeps matmul() on the sequential path
    const Tensor a = make_pattern({64, 512}, 5);
    const Tensor b = make_pattern({512, 520}, 6);

    MT_CHECK(tensors_close(matmul(a, b), reference_matmul(a, b)));
}

void test_matmul_edge_shapes()
{
    const Tensor no_rows = matmul(Tensor({0, 3}), Tensor({3, 2}));
    MT_CHECK(no_rows.shape().dimensions() == std::vector<std::size_t>({0, 2}));
    MT_CHECK(no_rows.size() == 0);

    // An empty inner dimension gives an all-zero result
    const Tensor no_inner = matmul(Tensor({2, 0}), Tensor({0, 3}));
    MT_CHECK(no_inner.shape().dimensions() == std::vector<std::size_t>({2, 3}));
    MT_CHECK(sum(no_inner) == 0.0);

    const Tensor single = matmul(Tensor({1, 1}, 3.0), Tensor({1, 1}, 4.0));
    MT_CHECK(single.at({0, 0}) == 12.0);

    // Row vector times column vector is a dot product
    const Tensor row = make_pattern({1, 6}, 7);
    const Tensor column = make_pattern({6, 1}, 8);
    MT_CHECK(tensors_close(matmul(row, column), reference_matmul(row, column)));
}

void test_matmul_errors()
{
    MT_CHECK_THROWS(matmul(Tensor({2, 3}), Tensor({2, 3})), std::invalid_argument);
    MT_CHECK_THROWS(matmul(Tensor({2, 3, 4}), Tensor({4, 2})), std::invalid_argument);
    MT_CHECK_THROWS(matmul(Tensor({3}), Tensor({3, 1})), std::invalid_argument);
    MT_CHECK_THROWS(matmul_parallel(Tensor({2, 3}), Tensor({4, 2})), std::invalid_argument);
    MT_CHECK_THROWS(matmul_parallel(Tensor({2}), Tensor({2})), std::invalid_argument);
}

void test_matmul_algebraic_properties()
{
    const Tensor a = make_pattern({7, 5}, 9);
    const Tensor b = make_pattern({5, 6}, 10);
    const Tensor c = make_pattern({6, 4}, 11);

    MT_CHECK(tensors_close(matmul(identity(7), a), a));
    MT_CHECK(tensors_close(matmul(a, identity(5)), a));

    // (AB)C == A(BC)
    MT_CHECK(tensors_close(matmul(matmul(a, b), c), matmul(a, matmul(b, c))));

    // A(B + B') == AB + AB'
    const Tensor b2 = make_pattern({5, 6}, 12);
    MT_CHECK(tensors_close(matmul(a, b + b2), matmul(a, b) + matmul(a, b2)));
}

// ------------------------------------------------------------- transpose

void test_transpose_properties()
{
    const Tensor a = make_pattern({4, 7}, 13);
    const Tensor b = make_pattern({7, 3}, 14);

    MT_CHECK(tensors_close(transpose(transpose(a)), a));

    // (AB)^T == B^T A^T
    MT_CHECK(tensors_close(transpose(matmul(a, b)), matmul(transpose(b), transpose(a))));

    const Tensor row = make_pattern({1, 5}, 15);
    const Tensor column = transpose(row);
    MT_CHECK(column.shape().dimensions() == std::vector<std::size_t>({5, 1}));

    for (std::size_t i = 0; i < 5; ++i)
    {
        MT_CHECK(column.at({i, 0}) == row.at({0, i}));
    }

    MT_CHECK_THROWS(transpose(Tensor({4})), std::invalid_argument);
}

// ------------------------------------------------------ reshape/indexing

void test_reshape_round_trip()
{
    Tensor tensor({2, 3, 4});

    for (std::size_t i = 0; i < tensor.size(); ++i)
    {
        tensor.flat_at(i) = static_cast<double>(i);
    }

    const double* data_before = tensor.data();

    // Row-major: (i, j, k) lives at i*12 + j*4 + k
    MT_CHECK(tensor.at({1, 2, 3}) == 23.0);
    MT_CHECK(tensor.at({1, 0, 2}) == 14.0);

    tensor.reshape({4, 6});
    MT_CHECK(tensor.at({3, 5}) == 23.0);
    MT_CHECK(tensor.at({2, 2}) == 14.0);

    tensor.reshape({24});
    MT_CHECK(tensor.at({23}) == 23.0);

    tensor.reshape({2, 3, 4});
    MT_CHECK(tensor.at({1, 2, 3}) == 23.0);

    // Reshape never reallocates or moves the data
    MT_CHECK(tensor.data() == data_before);

    Tensor single({1}, 7.0);
    single.reshape(std::vector<std::size_t>{});
    MT_CHECK(single.rank() == 0);
    MT_CHECK(single.at(std::vector<std::size_t>{}) == 7.0);
}

void test_failed_reshape_leaves_tensor_unchanged()
{
    Tensor tensor({2, 3}, 1.0);

    MT_CHECK_THROWS(tensor.reshape({4, 2}), std::invalid_argument);
    MT_CHECK_THROWS(tensor.reshape({7}), std::invalid_argument);

    MT_CHECK(tensor.shape().dimensions() == std::vector<std::size_t>({2, 3}));
    MT_CHECK(tensor.size() == 6);
}

void test_indexing_errors()
{
    Tensor tensor({2, 3});
    const Tensor& const_tensor = tensor;

    MT_CHECK_THROWS(tensor.at({1}), std::invalid_argument);
    MT_CHECK_THROWS(tensor.at({0, 0, 0}), std::invalid_argument);
    MT_CHECK_THROWS(tensor.at({2, 0}), std::out_of_range);
    MT_CHECK_THROWS(tensor.at({0, 3}), std::out_of_range);
    MT_CHECK_THROWS(const_tensor.at({5, 5}), std::out_of_range);
    MT_CHECK_THROWS(tensor({0, 3}), std::out_of_range);
    MT_CHECK_THROWS(tensor.flat_at(6), std::out_of_range);
    MT_CHECK_THROWS(const_tensor.flat_at(100), std::out_of_range);

    // Writes through operator() land in the right flat slot
    tensor({1, 2}) = 42.0;
    MT_CHECK(tensor.flat_at(5) == 42.0);
}

// ---------------------------------------------------- element-wise ops

void test_elementwise_operators_match_functions()
{
    const Tensor a = make_pattern({3, 4}, 16);
    const Tensor b = make_pattern({3, 4}, 17);
    const Tensor c = make_pattern({3, 4}, 18);

    MT_CHECK(tensors_close(a + b, add(a, b)));
    MT_CHECK(tensors_close(a - b, subtract(a, b)));
    MT_CHECK(tensors_close(a * b, multiply(a, b)));

    // A chained expression checked element by element
    const Tensor result = (a + b) * c - a;

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const double expected = (a.data()[i] + b.data()[i]) * c.data()[i] - a.data()[i];
        MT_CHECK(result.data()[i] == expected);
    }

    // Element-wise ops do not modify their inputs
    MT_CHECK(tensors_close(a, make_pattern({3, 4}, 16)));
}

void test_elementwise_shape_errors()
{
    const Tensor a({2, 3});
    const Tensor b({3, 2});
    const Tensor c({6});

    MT_CHECK_THROWS(a + b, std::invalid_argument);
    MT_CHECK_THROWS(a - c, std::invalid_argument);
    MT_CHECK_THROWS(a * b, std::invalid_argument);
}

// ---------------------------------------------------------- broadcasting

void test_broadcasting_three_dimensions()
{
    const Tensor a = make_pattern({2, 1, 3}, 19);
    const Tensor b = make_pattern({4, 1}, 20);

    const Tensor result = add_broadcast(a, b);
    MT_CHECK(result.shape().dimensions() == std::vector<std::size_t>({2, 4, 3}));

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 4; ++j)
        {
            for (std::size_t k = 0; k < 3; ++k)
            {
                MT_CHECK(result.at({i, j, k}) == a.at({i, 0, k}) + b.at({j, 0}));
            }
        }
    }
}

void test_broadcasting_outer_product_shape()
{
    const Tensor column = make_pattern({3, 1}, 21);
    const Tensor row = make_pattern({1, 4}, 22);

    // A (3x1) * (1x4) broadcast equals the matrix product of the two vectors
    const Tensor outer = multiply_broadcast(column, row);
    MT_CHECK(tensors_close(outer, matmul(column, row)));
}

void test_broadcasting_consistency()
{
    const Tensor a = make_pattern({3, 4}, 23);
    const Tensor b = make_pattern({3, 4}, 24);
    const Tensor row = make_pattern({4}, 25);

    // Same shapes behave exactly like the plain element-wise functions
    MT_CHECK(tensors_close(add_broadcast(a, b), add(a, b)));
    MT_CHECK(tensors_close(subtract_broadcast(a, b), subtract(a, b)));
    MT_CHECK(tensors_close(multiply_broadcast(a, b), multiply(a, b)));

    // Addition and multiplication are commutative under broadcasting
    MT_CHECK(tensors_close(add_broadcast(a, row), add_broadcast(row, a)));
    MT_CHECK(tensors_close(multiply_broadcast(a, row), multiply_broadcast(row, a)));

    // Scalar on the left of a subtraction
    const Tensor flipped = subtract_broadcast(scalar(10.0), a);

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        MT_CHECK(flipped.data()[i] == 10.0 - a.data()[i]);
    }

    const Tensor empty = add_broadcast(Tensor({0, 3}), Tensor({3}));
    MT_CHECK(empty.shape().dimensions() == std::vector<std::size_t>({0, 3}));

    MT_CHECK_THROWS(add_broadcast(Tensor({2, 3}), Tensor({4})), std::invalid_argument);
    MT_CHECK_THROWS(multiply_broadcast(Tensor({2, 3, 4}), Tensor({3, 1, 2})), std::invalid_argument);
}

// ------------------------------------------------------------ reductions

void test_reductions_large_tensor()
{
    constexpr std::size_t n = 100000;
    Tensor tensor({n});

    for (std::size_t i = 0; i < n; ++i)
    {
        tensor.flat_at(i) = static_cast<double>(i + 1);
    }

    MT_CHECK(sum(tensor) == static_cast<double>(n) * (n + 1) / 2.0);
    MT_CHECK(nearly_equal(mean(tensor), (n + 1) / 2.0));
    MT_CHECK(min(tensor) == 1.0);
    MT_CHECK(max(tensor) == static_cast<double>(n));

    // Reductions ignore shape; reshaping does not change the results
    tensor.reshape({100, 1000});
    MT_CHECK(sum(tensor) == static_cast<double>(n) * (n + 1) / 2.0);
}

// -------------------------------------------------------- copy semantics

void test_copy_and_move_semantics()
{
    Tensor original = make_pattern({4, 4}, 26);
    Tensor copy({1});
    copy = original;

    MT_CHECK(tensors_close(copy, original));
    MT_CHECK(copy.data() != original.data());

    copy.at({0, 0}) = 1000.0;
    MT_CHECK(original.at({0, 0}) != 1000.0);

    // Self-assignment through an alias keeps the data intact
    const Tensor expected = original;
    Tensor& alias = original;
    original = alias;
    MT_CHECK(tensors_close(original, expected));

    // Move assignment steals the buffer instead of copying it
    const double* buffer = copy.data();
    Tensor moved({1});
    moved = std::move(copy);
    MT_CHECK(moved.data() == buffer);
    MT_CHECK(moved.at({0, 0}) == 1000.0);
}

// ------------------------------------------------------- printing/version

void test_printing()
{
    Tensor matrix({2, 2});
    matrix.flat_at(0) = 1;
    matrix.flat_at(1) = 2;
    matrix.flat_at(2) = 3;
    matrix.flat_at(3) = 4;
    MT_CHECK(to_string(matrix) == "[[1, 2], [3, 4]]");

    Tensor vector({3});
    vector.flat_at(0) = 1.5;
    vector.flat_at(1) = -2;
    vector.flat_at(2) = 0;
    MT_CHECK(to_string(vector) == "[1.5, -2, 0]");

    MT_CHECK(to_string(scalar(42)) == "42");
    MT_CHECK(to_string(Tensor({0, 3})) == "[]");
    MT_CHECK(to_string(Tensor({2, 0})) == "[[], []]");
    MT_CHECK(to_string(Tensor({2, 1, 2}, 0.5)) == "[[[0.5, 0.5]], [[0.5, 0.5]]]");
}

void test_version()
{
    MT_CHECK(std::string(version()) == MINITENSOR_VERSION_STRING);
    MT_CHECK(MINITENSOR_VERSION_MAJOR >= 1);
}

}

int main()
{
    run_test("matmul small matches reference", test_matmul_small_matches_reference);
    run_test("matmul parallel path matches reference", test_matmul_parallel_path_matches_reference);
    run_test("matmul wide sequential path matches reference", test_matmul_wide_sequential_path_matches_reference);
    run_test("matmul edge shapes", test_matmul_edge_shapes);
    run_test("matmul errors", test_matmul_errors);
    run_test("matmul algebraic properties", test_matmul_algebraic_properties);
    run_test("transpose properties", test_transpose_properties);
    run_test("reshape round trip", test_reshape_round_trip);
    run_test("failed reshape leaves tensor unchanged", test_failed_reshape_leaves_tensor_unchanged);
    run_test("indexing errors", test_indexing_errors);
    run_test("element-wise operators match functions", test_elementwise_operators_match_functions);
    run_test("element-wise shape errors", test_elementwise_shape_errors);
    run_test("broadcasting in three dimensions", test_broadcasting_three_dimensions);
    run_test("broadcasting outer product", test_broadcasting_outer_product_shape);
    run_test("broadcasting consistency", test_broadcasting_consistency);
    run_test("reductions on a large tensor", test_reductions_large_tensor);
    run_test("copy and move semantics", test_copy_and_move_semantics);
    run_test("printing", test_printing);
    run_test("version", test_version);

    std::cout << "OpenMP " << (openmp_enabled() ? "enabled" : "disabled") << '\n';

    return minitensor_test::summary("Phase 6 tests");
}
