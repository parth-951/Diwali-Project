#ifndef MINITENSOR_CUDA_H
#define MINITENSOR_CUDA_H

// Optional GPU backend, built with -DMINITENSOR_USE_CUDA=ON.
//
// A Tensor always lives in host memory. A DeviceTensor is a copy of it in GPU
// memory. Operations on DeviceTensors return as soon as the work is queued, so
// call synchronize() before reading a clock. Operations on Tensors are the
// simple form: they upload, compute, wait and download.

#include "minitensor/Tensor.h"

#include <cstddef>
#include <string>
#include <vector>

namespace minitensor {
namespace cuda {

class DeviceTensor {
public:
    // Uninitialized GPU memory with the given dimensions
    explicit DeviceTensor(const std::vector<std::size_t>& dimensions);

    // Copies a host tensor to the GPU
    explicit DeviceTensor(const Tensor& host);

    ~DeviceTensor();

    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    DeviceTensor(DeviceTensor&& other) noexcept;
    DeviceTensor& operator=(DeviceTensor&& other) noexcept;

    const std::vector<std::size_t>& dimensions() const noexcept;
    std::size_t size() const noexcept;

    double* data() noexcept;
    const double* data() const noexcept;

    // Waits for queued work, then copies the values back to the host
    Tensor download() const;

private:
    std::vector<std::size_t> dimensions_;
    std::size_t size_ = 0;
    double* data_ = nullptr;
};

// True when a CUDA device can be used
bool available() noexcept;

std::string device_name();

// CUDA runtime version, such as "13.4"
std::string runtime_version();

// Blocks until all queued GPU work has finished
void synchronize();

// Queued on the GPU: these return before the work has finished.
// They throw std::invalid_argument for the same shapes as the CPU versions.
DeviceTensor add(const DeviceTensor& a, const DeviceTensor& b);
DeviceTensor subtract(const DeviceTensor& a, const DeviceTensor& b);
DeviceTensor multiply(const DeviceTensor& a, const DeviceTensor& b);

DeviceTensor add_broadcast(const DeviceTensor& a, const DeviceTensor& b);
DeviceTensor subtract_broadcast(const DeviceTensor& a, const DeviceTensor& b);
DeviceTensor multiply_broadcast(const DeviceTensor& a, const DeviceTensor& b);

DeviceTensor matmul(const DeviceTensor& a, const DeviceTensor& b);
DeviceTensor transpose(const DeviceTensor& a);

// Blocks, because the answer has to come back to the host
double sum(const DeviceTensor& a);

// Host tensors in and out. These block until the result is ready.
Tensor add(const Tensor& a, const Tensor& b);
Tensor subtract(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);

Tensor add_broadcast(const Tensor& a, const Tensor& b);
Tensor subtract_broadcast(const Tensor& a, const Tensor& b);
Tensor multiply_broadcast(const Tensor& a, const Tensor& b);

Tensor matmul(const Tensor& a, const Tensor& b);
Tensor transpose(const Tensor& a);

double sum(const Tensor& a);

} // namespace cuda
} // namespace minitensor

#endif
