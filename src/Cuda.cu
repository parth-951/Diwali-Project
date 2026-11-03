#include "minitensor/Cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace minitensor {
namespace cuda {

namespace {

constexpr unsigned block_threads = 256;

// Enough for every shape used in practice, and it keeps kernel arguments plain
constexpr std::size_t max_broadcast_rank = 8;

// Elements summed by one GPU thread in sum()
constexpr std::size_t sum_chunk = 1024;

void check(cudaError_t status, const char* action)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(action) + " failed: " + cudaGetErrorString(status)
        );
    }
}

// Runs body(i) for every i in [0, count), with one GPU thread per index
template <typename Body>
__global__ void for_each_kernel(std::size_t count, Body body)
{
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * block_threads + threadIdx.x;

    if (index < count)
    {
        body(index);
    }
}

template <typename Body>
void launch(std::size_t count, const Body& body)
{
    const std::size_t blocks = (count + block_threads - 1) / block_threads;

    if (blocks > std::numeric_limits<unsigned>::max())
    {
        throw std::invalid_argument("Tensor is too large for the CUDA backend");
    }

    for_each_kernel<<<static_cast<unsigned>(blocks), block_threads>>>(count, body);
    check(cudaGetLastError(), "Launching a CUDA kernel");
}

struct AddOp {
    __device__ double operator()(double x, double y) const { return x + y; }
};

struct SubtractOp {
    __device__ double operator()(double x, double y) const { return x - y; }
};

struct MultiplyOp {
    __device__ double operator()(double x, double y) const { return x * y; }
};

template <typename Op>
struct ElementwiseBody {
    const double* a;
    const double* b;
    double* result;
    Op op;

    __device__ void operator()(std::size_t i) const
    {
        result[i] = op(a[i], b[i]);
    }
};

// Dimensions of one input, aligned to the last output dimension
struct BroadcastInput {
    const double* data;
    std::size_t rank;
    std::size_t dims[max_broadcast_rank];

    __device__ double at(
        std::size_t flat_index,
        const std::size_t* output_dims,
        std::size_t output_rank) const
    {
        std::size_t offset = 0;
        std::size_t stride = 1;

        for (std::size_t i = 0; i < rank; ++i)
        {
            const std::size_t input_dim = dims[rank - 1 - i];
            const std::size_t output_dim = output_dims[output_rank - 1 - i];
            const std::size_t coordinate = flat_index % output_dim;

            flat_index /= output_dim;

            if (input_dim != 1)
            {
                offset += coordinate * stride;
            }

            stride *= input_dim;
        }

        return data[offset];
    }
};

template <typename Op>
struct BroadcastBody {
    BroadcastInput a;
    BroadcastInput b;
    std::size_t output_rank;
    std::size_t output_dims[max_broadcast_rank];
    double* result;
    Op op;

    __device__ void operator()(std::size_t i) const
    {
        result[i] = op(
            a.at(i, output_dims, output_rank),
            b.at(i, output_dims, output_rank)
        );
    }
};

struct TransposeBody {
    const double* input;
    double* output;
    std::size_t rows;
    std::size_t cols;

    __device__ void operator()(std::size_t index) const
    {
        const std::size_t j = index / rows;
        const std::size_t i = index % rows;

        output[index] = input[i * cols + j];
    }
};

struct SumBody {
    const double* input;
    double* partial;
    std::size_t size;

    __device__ void operator()(std::size_t chunk) const
    {
        const std::size_t begin = chunk * sum_chunk;
        const std::size_t end =
            begin + sum_chunk < size ? begin + sum_chunk : size;

        double total = 0.0;

        for (std::size_t i = begin; i < end; ++i)
        {
            total += input[i];
        }

        partial[chunk] = total;
    }
};

// Same rules as the CPU broadcast_shape() in Ops.cpp
std::vector<std::size_t> broadcast_shape(
    const std::vector<std::size_t>& a_dims,
    const std::vector<std::size_t>& b_dims)
{
    const std::size_t rank =
        a_dims.size() > b_dims.size() ? a_dims.size() : b_dims.size();

    if (rank > max_broadcast_rank)
    {
        throw std::invalid_argument(
            "The CUDA backend broadcasts at most 8 dimensions"
        );
    }

    std::vector<std::size_t> result(rank);

    for (std::size_t i = 0; i < rank; ++i)
    {
        const std::size_t a_dim =
            i < rank - a_dims.size() ? 1 : a_dims[i - (rank - a_dims.size())];

        const std::size_t b_dim =
            i < rank - b_dims.size() ? 1 : b_dims[i - (rank - b_dims.size())];

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1)
        {
            throw std::invalid_argument(
                "Tensor shapes are not broadcast-compatible"
            );
        }

        result[i] = a_dim == 1 ? b_dim : a_dim;
    }

    return result;
}

BroadcastInput broadcast_input(const DeviceTensor& tensor)
{
    BroadcastInput input{};
    input.data = tensor.data();
    input.rank = tensor.dimensions().size();

    for (std::size_t i = 0; i < input.rank; ++i)
    {
        input.dims[i] = tensor.dimensions()[i];
    }

    return input;
}

template <typename Op>
DeviceTensor elementwise(const DeviceTensor& a, const DeviceTensor& b)
{
    if (a.dimensions() != b.dimensions())
    {
        throw std::invalid_argument(
            "Element-wise operations require identical tensor shapes"
        );
    }

    DeviceTensor result(a.dimensions());

    launch(a.size(), ElementwiseBody<Op>{a.data(), b.data(), result.data(), Op{}});

    return result;
}

template <typename Op>
DeviceTensor broadcast(const DeviceTensor& a, const DeviceTensor& b)
{
    const std::vector<std::size_t> output_dims =
        broadcast_shape(a.dimensions(), b.dimensions());

    DeviceTensor result(output_dims);

    BroadcastBody<Op> body{};
    body.a = broadcast_input(a);
    body.b = broadcast_input(b);
    body.output_rank = output_dims.size();
    body.result = result.data();

    for (std::size_t i = 0; i < output_dims.size(); ++i)
    {
        body.output_dims[i] = output_dims[i];
    }

    launch(result.size(), body);

    return result;
}

cublasHandle_t blas_handle()
{
    // Never destroyed: the CUDA runtime may already be gone at static
    // destruction time, and the process is ending anyway.
    static const cublasHandle_t handle = [] {
        cublasHandle_t created = nullptr;

        if (cublasCreate(&created) != CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error("Could not create a cuBLAS handle");
        }

        return created;
    }();

    return handle;
}

int blas_dimension(std::size_t value)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::invalid_argument("Matrix is too large for cuBLAS");
    }

    return static_cast<int>(value);
}

std::size_t element_count(const std::vector<std::size_t>& dimensions)
{
    std::size_t count = 1;

    for (const std::size_t dimension : dimensions)
    {
        count *= dimension;
    }

    return count;
}

} // namespace

DeviceTensor::DeviceTensor(const std::vector<std::size_t>& dimensions)
    : dimensions_(dimensions), size_(element_count(dimensions))
{
    check(
        cudaMalloc(&data_, size_ * sizeof(double)),
        "Allocating GPU memory"
    );
}

DeviceTensor::DeviceTensor(const Tensor& host)
    : DeviceTensor(host.shape().dimensions())
{
    check(
        cudaMemcpy(data_, host.data(), size_ * sizeof(double), cudaMemcpyHostToDevice),
        "Copying to the GPU"
    );
}

DeviceTensor::~DeviceTensor()
{
    // cudaFree waits for queued work that uses this memory.
    if (data_ != nullptr)
    {
        cudaFree(data_);
    }
}

DeviceTensor::DeviceTensor(DeviceTensor&& other) noexcept
    : dimensions_(std::move(other.dimensions_)),
      size_(other.size_),
      data_(other.data_)
{
    other.size_ = 0;
    other.data_ = nullptr;
}

DeviceTensor& DeviceTensor::operator=(DeviceTensor&& other) noexcept
{
    if (this != &other)
    {
        if (data_ != nullptr)
        {
            cudaFree(data_);
        }

        dimensions_ = std::move(other.dimensions_);
        size_ = other.size_;
        data_ = other.data_;

        other.size_ = 0;
        other.data_ = nullptr;
    }

    return *this;
}

const std::vector<std::size_t>& DeviceTensor::dimensions() const noexcept
{
    return dimensions_;
}

std::size_t DeviceTensor::size() const noexcept
{
    return size_;
}

double* DeviceTensor::data() noexcept
{
    return data_;
}

const double* DeviceTensor::data() const noexcept
{
    return data_;
}

Tensor DeviceTensor::download() const
{
    synchronize();

    Tensor host(dimensions_);

    check(
        cudaMemcpy(host.data(), data_, size_ * sizeof(double), cudaMemcpyDeviceToHost),
        "Copying from the GPU"
    );

    return host;
}

bool available() noexcept
{
    int count = 0;

    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string device_name()
{
    int device = 0;
    check(cudaGetDevice(&device), "Reading the current CUDA device");

    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "Reading CUDA device properties");

    return properties.name;
}

std::string runtime_version()
{
    int version = 0;
    check(cudaRuntimeGetVersion(&version), "Reading the CUDA runtime version");

    return std::to_string(version / 1000) + "." + std::to_string(version % 1000 / 10);
}

void synchronize()
{
    check(cudaDeviceSynchronize(), "Waiting for the GPU");
}

DeviceTensor add(const DeviceTensor& a, const DeviceTensor& b)
{
    return elementwise<AddOp>(a, b);
}

DeviceTensor subtract(const DeviceTensor& a, const DeviceTensor& b)
{
    return elementwise<SubtractOp>(a, b);
}

DeviceTensor multiply(const DeviceTensor& a, const DeviceTensor& b)
{
    return elementwise<MultiplyOp>(a, b);
}

DeviceTensor add_broadcast(const DeviceTensor& a, const DeviceTensor& b)
{
    return broadcast<AddOp>(a, b);
}

DeviceTensor subtract_broadcast(const DeviceTensor& a, const DeviceTensor& b)
{
    return broadcast<SubtractOp>(a, b);
}

DeviceTensor multiply_broadcast(const DeviceTensor& a, const DeviceTensor& b)
{
    return broadcast<MultiplyOp>(a, b);
}

DeviceTensor matmul(const DeviceTensor& a, const DeviceTensor& b)
{
    if (a.dimensions().size() != 2 || b.dimensions().size() != 2)
    {
        throw std::invalid_argument(
            "Matrix multiplication requires 2D tensors"
        );
    }

    if (a.dimensions()[1] != b.dimensions()[0])
    {
        throw std::invalid_argument(
            "Matrix dimensions are incompatible for multiplication"
        );
    }

    const std::size_t rows = a.dimensions()[0];
    const std::size_t inner = a.dimensions()[1];
    const std::size_t cols = b.dimensions()[1];

    DeviceTensor result(std::vector<std::size_t>{rows, cols});

    // cuBLAS is column-major and MiniTensor is row-major. Read as column-major,
    // the stored data is the transpose of each matrix, and (AB)^T = B^T A^T,
    // so swapping the inputs gives C in MiniTensor's layout.
    const double one = 1.0;
    const double zero = 0.0;

    const cublasStatus_t status = cublasDgemm(
        blas_handle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        blas_dimension(cols), blas_dimension(rows), blas_dimension(inner),
        &one,
        b.data(), blas_dimension(cols),
        a.data(), blas_dimension(inner),
        &zero,
        result.data(), blas_dimension(cols)
    );

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("cublasDgemm failed");
    }

    return result;
}

DeviceTensor transpose(const DeviceTensor& a)
{
    if (a.dimensions().size() != 2)
    {
        throw std::invalid_argument("Transpose requires a 2D tensor");
    }

    const std::size_t rows = a.dimensions()[0];
    const std::size_t cols = a.dimensions()[1];

    DeviceTensor result(std::vector<std::size_t>{cols, rows});

    launch(a.size(), TransposeBody{a.data(), result.data(), rows, cols});

    return result;
}

double sum(const DeviceTensor& a)
{
    // Chunk totals are added on the host in a fixed order, so the answer does
    // not depend on which GPU thread finishes first.
    const std::size_t chunks = (a.size() + sum_chunk - 1) / sum_chunk;

    DeviceTensor partial(std::vector<std::size_t>{chunks});

    launch(chunks, SumBody{a.data(), partial.data(), a.size()});

    const Tensor totals = partial.download();

    double total = 0.0;

    for (std::size_t i = 0; i < chunks; ++i)
    {
        total += totals.data()[i];
    }

    return total;
}

Tensor add(const Tensor& a, const Tensor& b)
{
    return add(DeviceTensor(a), DeviceTensor(b)).download();
}

Tensor subtract(const Tensor& a, const Tensor& b)
{
    return subtract(DeviceTensor(a), DeviceTensor(b)).download();
}

Tensor multiply(const Tensor& a, const Tensor& b)
{
    return multiply(DeviceTensor(a), DeviceTensor(b)).download();
}

Tensor add_broadcast(const Tensor& a, const Tensor& b)
{
    return add_broadcast(DeviceTensor(a), DeviceTensor(b)).download();
}

Tensor subtract_broadcast(const Tensor& a, const Tensor& b)
{
    return subtract_broadcast(DeviceTensor(a), DeviceTensor(b)).download();
}

Tensor multiply_broadcast(const Tensor& a, const Tensor& b)
{
    return multiply_broadcast(DeviceTensor(a), DeviceTensor(b)).download();
}

Tensor matmul(const Tensor& a, const Tensor& b)
{
    return matmul(DeviceTensor(a), DeviceTensor(b)).download();
}

Tensor transpose(const Tensor& a)
{
    return transpose(DeviceTensor(a)).download();
}

double sum(const Tensor& a)
{
    return sum(DeviceTensor(a));
}

} // namespace cuda
} // namespace minitensor
