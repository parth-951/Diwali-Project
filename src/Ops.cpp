
#include "minitensor/Ops.h"

#include <stdexcept>
#include <vector>
#include <cstddef>

namespace minitensor {

namespace {

// Validate identical shapes for normal element-wise operations
void validate_same_shape(const Shape& a, const Shape& b)
{
    if (a.dimensions() != b.dimensions())
    {
        throw std::invalid_argument(
            "Element-wise operations require identical tensor shapes"
        );
    }
}

// Calculate the broadcast-compatible output shape
std::vector<std::size_t> broadcast_shape(
    const Shape& a,
    const Shape& b)
{
    const auto& a_dims = a.dimensions();
    const auto& b_dims = b.dimensions();

    const std::size_t rank =
        a_dims.size() > b_dims.size()
        ? a_dims.size()
        : b_dims.size();

    std::vector<std::size_t> result(rank);

    for (std::size_t i = 0; i < rank; ++i)
    {
        const std::size_t a_index =
            i < rank - a_dims.size()
            ? 1
            : a_dims[i - (rank - a_dims.size())];

        const std::size_t b_index =
            i < rank - b_dims.size()
            ? 1
            : b_dims[i - (rank - b_dims.size())];

        if (a_index == b_index)
        {
            result[i] = a_index;
        }
        else if (a_index == 1)
        {
            result[i] = b_index;
        }
        else if (b_index == 1)
        {
            result[i] = a_index;
        }
        else
        {
            throw std::invalid_argument(
                "Tensor shapes are not broadcast-compatible"
            );
        }
    }

    return result;
}

// Calculate the input offset for a broadcasted output element
std::size_t broadcast_offset(
    std::size_t flat_index,
    const std::vector<std::size_t>& output_shape,
    const std::vector<std::size_t>& input_shape)
{
    const std::size_t output_rank = output_shape.size();
    const std::size_t input_rank = input_shape.size();

    std::size_t offset = 0;
    std::size_t input_stride = 1;

    for (std::size_t i = 0; i < input_rank; ++i)
    {
        const std::size_t input_dim =
            input_shape[input_rank - 1 - i];

        const std::size_t output_dim =
            output_shape[output_rank - 1 - i];

        const std::size_t coordinate =
            flat_index % output_dim;

        flat_index /= output_dim;

        if (input_dim != 1)
        {
            offset += coordinate * input_stride;
        }

        input_stride *= input_dim;
    }

    return offset;
}

// Generic broadcasting operation
template <typename T, typename Operation>
BasicTensor<T> broadcast_operation(
    const BasicTensor<T>& a,
    const BasicTensor<T>& b,
    Operation operation)
{
    const std::vector<std::size_t> output_shape =
        broadcast_shape(a.shape(), b.shape());

    BasicTensor<T> result(output_shape);

    const T* a_data = a.data();
    const T* b_data = b.data();
    T* result_data = result.data();

    const auto& a_dims = a.shape().dimensions();
    const auto& b_dims = b.shape().dimensions();

    for (std::size_t i = 0; i < result.size(); ++i)
    {
        const std::size_t a_offset =
            broadcast_offset(i, output_shape, a_dims);

        const std::size_t b_offset =
            broadcast_offset(i, output_shape, b_dims);

        result_data[i] = operation(
            a_data[a_offset],
            b_data[b_offset]
        );
    }

    return result;
}

}

// Element-wise addition
template <typename T>
BasicTensor<T> add(const BasicTensor<T>& a, const BasicTensor<T>& b)
{
    validate_same_shape(a.shape(), b.shape());

    BasicTensor<T> result(a.shape().dimensions());

    const T* a_data = a.data();
    const T* b_data = b.data();
    T* result_data = result.data();

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        result_data[i] = a_data[i] + b_data[i];
    }

    return result;
}

// Element-wise subtraction
template <typename T>
BasicTensor<T> subtract(const BasicTensor<T>& a, const BasicTensor<T>& b)
{
    validate_same_shape(a.shape(), b.shape());

    BasicTensor<T> result(a.shape().dimensions());

    const T* a_data = a.data();
    const T* b_data = b.data();
    T* result_data = result.data();

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        result_data[i] = a_data[i] - b_data[i];
    }

    return result;
}

// Element-wise multiplication
template <typename T>
BasicTensor<T> multiply(const BasicTensor<T>& a, const BasicTensor<T>& b)
{
    validate_same_shape(a.shape(), b.shape());

    BasicTensor<T> result(a.shape().dimensions());

    const T* a_data = a.data();
    const T* b_data = b.data();
    T* result_data = result.data();

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        result_data[i] = a_data[i] * b_data[i];
    }

    return result;
}

// Broadcasting addition
template <typename T>
BasicTensor<T> add_broadcast(const BasicTensor<T>& a, const BasicTensor<T>& b)
{
    return broadcast_operation(
        a, b,
        [](T x, T y) { return x + y; }
    );
}

// Broadcasting subtraction
template <typename T>
BasicTensor<T> subtract_broadcast(const BasicTensor<T>& a, const BasicTensor<T>& b)
{
    return broadcast_operation(
        a, b,
        [](T x, T y) { return x - y; }
    );
}

// Broadcasting multiplication
template <typename T>
BasicTensor<T> multiply_broadcast(const BasicTensor<T>& a, const BasicTensor<T>& b)
{
    return broadcast_operation(
        a, b,
        [](T x, T y) { return x * y; }
    );
}

// Hybrid matrix multiplication
template <typename T>
BasicTensor<T> matmul(const BasicTensor<T>& a, const BasicTensor<T>& b)
{
    if (a.rank() != 2 || b.rank() != 2)
    {
        throw std::invalid_argument(
            "Matrix multiplication requires 2D tensors"
        );
    }

    const auto& a_shape = a.shape().dimensions();
    const auto& b_shape = b.shape().dimensions();

    const std::size_t rows_a = a_shape[0];
    const std::size_t cols_a = a_shape[1];
    const std::size_t rows_b = b_shape[0];
    const std::size_t cols_b = b_shape[1];

    if (cols_a != rows_b)
    {
        throw std::invalid_argument(
            "Matrix dimensions are incompatible for multiplication"
        );
    }

    constexpr long double parallel_threshold =
        256.0L * 256.0L * 256.0L;

    const long double estimated_work =
        static_cast<long double>(rows_a) *
        static_cast<long double>(cols_a) *
        static_cast<long double>(cols_b);

    if (rows_a >= 128 && estimated_work >= parallel_threshold)
    {
        return matmul_parallel(a, b);
    }

    BasicTensor<T> result({rows_a, cols_b}, T{});

    const T* a_data = a.data();
    const T* b_data = b.data();
    T* result_data = result.data();

    for (std::size_t i = 0; i < rows_a; ++i)
    {
        for (std::size_t k = 0; k < cols_a; ++k)
        {
            const T value = a_data[i * cols_a + k];

            for (std::size_t j = 0; j < cols_b; ++j)
            {
                result_data[i * cols_b + j] +=
                    value * b_data[k * cols_b + j];
            }
        }
    }

    return result;
}

// OpenMP parallel matrix multiplication
template <typename T>
BasicTensor<T> matmul_parallel(const BasicTensor<T>& a, const BasicTensor<T>& b)
{
    if (a.rank() != 2 || b.rank() != 2)
    {
        throw std::invalid_argument(
            "Matrix multiplication requires 2D tensors"
        );
    }

    const auto& a_shape = a.shape().dimensions();
    const auto& b_shape = b.shape().dimensions();

    const std::size_t rows_a = a_shape[0];
    const std::size_t cols_a = a_shape[1];
    const std::size_t rows_b = b_shape[0];
    const std::size_t cols_b = b_shape[1];

    if (cols_a != rows_b)
    {
        throw std::invalid_argument(
            "Invalid matrix dimensions"
        );
    }

    BasicTensor<T> result({rows_a, cols_b}, T{});

    const T* A = a.data();
    const T* B = b.data();
    T* C = result.data();

    // Without OpenMP (MINITENSOR_USE_OPENMP=OFF) this runs sequentially.
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long long i = 0;
         i < static_cast<long long>(rows_a); ++i)
    {
        for (std::size_t k = 0; k < cols_a; ++k)
        {
            const T value = A[i * cols_a + k];

            for (std::size_t j = 0; j < cols_b; ++j)
            {
                C[i * cols_b + j] +=
                    value * B[k * cols_b + j];
            }
        }
    }

    return result;
}

// Tensor transpose
template <typename T>
BasicTensor<T> transpose(const BasicTensor<T>& a)
{
    if (a.rank() != 2)
    {
        throw std::invalid_argument(
            "Transpose requires a 2D tensor"
        );
    }

    const auto& dimensions = a.shape().dimensions();

    const std::size_t rows = dimensions[0];
    const std::size_t cols = dimensions[1];

    BasicTensor<T> result({cols, rows});

    const T* input = a.data();
    T* output = result.data();

    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            output[j * rows + i] = input[i * cols + j];
        }
    }

    return result;
}

// Addition operator
template <typename T>
BasicTensor<T> operator+(const BasicTensor<T>& a, const BasicTensor<T>& b)
{
    return add(a, b);
}

// Subtraction operator
template <typename T>
BasicTensor<T> operator-(const BasicTensor<T>& a, const BasicTensor<T>& b)
{
    return subtract(a, b);
}

// Multiplication operator
template <typename T>
BasicTensor<T> operator*(const BasicTensor<T>& a, const BasicTensor<T>& b)
{
    return multiply(a, b);
}

// Phase 4: Reduction Operations

template <typename T>
double sum(const BasicTensor<T>& a)
{
    double result = 0.0;

    const T* input = a.data();

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        result += input[i];
    }

    return result;
}

template <typename T>
double mean(const BasicTensor<T>& a)
{
    if (a.size() == 0)
    {
        throw std::invalid_argument(
            "Mean of an empty tensor is undefined"
        );
    }

    return sum(a) / static_cast<double>(a.size());
}

template <typename T>
T min(const BasicTensor<T>& a)
{
    if (a.size() == 0)
    {
        throw std::invalid_argument(
            "Minimum of an empty tensor is undefined"
        );
    }

    const T* input = a.data();
    T result = input[0];

    for (std::size_t i = 1; i < a.size(); ++i)
    {
        if (input[i] < result)
        {
            result = input[i];
        }
    }

    return result;
}

template <typename T>
T max(const BasicTensor<T>& a)
{
    if (a.size() == 0)
    {
        throw std::invalid_argument(
            "Maximum of an empty tensor is undefined"
        );
    }

    const T* input = a.data();
    T result = input[0];

    for (std::size_t i = 1; i < a.size(); ++i)
    {
        if (input[i] > result)
        {
            result = input[i];
        }
    }

    return result;
}

// The two supported element types
#define MINITENSOR_INSTANTIATE_OPS(T)                                              \
    template BasicTensor<T> add(const BasicTensor<T>&, const BasicTensor<T>&);           \
    template BasicTensor<T> subtract(const BasicTensor<T>&, const BasicTensor<T>&);      \
    template BasicTensor<T> multiply(const BasicTensor<T>&, const BasicTensor<T>&);      \
    template BasicTensor<T> matmul(const BasicTensor<T>&, const BasicTensor<T>&);        \
    template BasicTensor<T> matmul_parallel(const BasicTensor<T>&, const BasicTensor<T>&); \
    template BasicTensor<T> transpose(const BasicTensor<T>&);                            \
    template BasicTensor<T> add_broadcast(const BasicTensor<T>&, const BasicTensor<T>&); \
    template BasicTensor<T> subtract_broadcast(const BasicTensor<T>&, const BasicTensor<T>&); \
    template BasicTensor<T> multiply_broadcast(const BasicTensor<T>&, const BasicTensor<T>&); \
    template double sum(const BasicTensor<T>&);                                         \
    template double mean(const BasicTensor<T>&);                                        \
    template T min(const BasicTensor<T>&);                                              \
    template T max(const BasicTensor<T>&);                                              \
    template BasicTensor<T> operator+(const BasicTensor<T>&, const BasicTensor<T>&);     \
    template BasicTensor<T> operator-(const BasicTensor<T>&, const BasicTensor<T>&);     \
    template BasicTensor<T> operator*(const BasicTensor<T>&, const BasicTensor<T>&);

MINITENSOR_INSTANTIATE_OPS(float)
MINITENSOR_INSTANTIATE_OPS(double)

#undef MINITENSOR_INSTANTIATE_OPS

}
