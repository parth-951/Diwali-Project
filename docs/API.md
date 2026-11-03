# MiniTensor API Reference

Everything lives in the `minitensor` namespace. Include `<minitensor/minitensor.h>` for the whole API, or include individual headers.

| Header | Contents |
|---|---|
| `minitensor/Tensor.h` | `Tensor` |
| `minitensor/Ops.h` | Arithmetic, broadcasting, matmul, transpose, reductions |
| `minitensor/IO.h` | Printing (`operator<<`, `to_string`) |
| `minitensor/Version.h` | Version macros, `version()`, `openmp_enabled()` |
| `minitensor/Shape.h` | `Shape` (dimensions and strides) |
| `minitensor/Storage.h` | `Storage` (aligned buffer) |

Elements are `double`. Tensors are stored contiguously in **row-major** order, so the last index changes fastest.

---

## Tensor

```cpp
class Tensor;
```

An n-dimensional array that owns its data. Copying a tensor copies the data (deep copy). Moving a tensor transfers the buffer without copying.

### Construction

```cpp
explicit Tensor(const std::vector<std::size_t>& dimensions, double initial_value = 0.0);
Tensor(std::initializer_list<std::size_t> dimensions, double initial_value = 0.0);
```

Creates a tensor of the given shape with every element set to `initial_value`.

```cpp
Tensor matrix({2, 3});                            // 2x3 of zeros
Tensor filled({4}, 1.5);                          // [1.5, 1.5, 1.5, 1.5]
Tensor scalar(std::vector<std::size_t>{}, 7.0);   // rank 0, one element
Tensor empty({0, 3});                             // zero elements
```

- **Scalar:** an empty dimension list gives a rank-0 tensor with exactly one element. Pass `std::vector<std::size_t>{}` instead of `{}` to avoid an ambiguous overload.
- **Empty:** any dimension of size 0 gives a tensor with zero elements.
- **Throws** `std::overflow_error` if the element count or byte size overflows `std::size_t`.

### Shape queries

| Member | Returns |
|---|---|
| `std::size_t size() const noexcept` | Total number of elements |
| `std::size_t rank() const noexcept` | Number of dimensions |
| `const Shape& shape() const noexcept` | The shape, with `shape().dimensions()` and `shape().strides()` |

### Element access

```cpp
double& at(std::initializer_list<std::size_t> indices);
double& at(const std::vector<std::size_t>& indices);
double& operator()(std::initializer_list<std::size_t> indices);
double& operator()(const std::vector<std::size_t>& indices);
double& flat_at(std::size_t index);
// const overloads of all of the above
```

`at()` and `operator()` take one index per dimension, and `flat_at()` takes a position in the underlying contiguous buffer. All of them are bounds-checked.

```cpp
Tensor t({2, 3});
t({1, 2}) = 5.0;       // same element as t.flat_at(5)
double v = t.at({1, 2});
```

| Error | Exception |
|---|---|
| Wrong number of indices | `std::invalid_argument` |
| An index ≥ its dimension | `std::out_of_range` |
| `flat_at` index ≥ `size()` | `std::out_of_range` |

### Raw data

```cpp
double* data() noexcept;
const double* data() const noexcept;
void fill(double value);
```

`data()` points to `size()` contiguous doubles, aligned to 64 bytes. It may be `nullptr` for an empty tensor. Use it for fast loops without per-element bounds checks.

### Reshape

```cpp
void reshape(std::initializer_list<std::size_t> dimensions);
void reshape(const std::vector<std::size_t>& dimensions);
```

Changes the shape **without moving or copying data**. Because the layout is row-major, the flat order of elements stays the same.

- **Throws** `std::invalid_argument` if the new shape has a different number of elements. The tensor is left unchanged.

---

## Element-wise operations (`Ops.h`)

```cpp
Tensor add(const Tensor& a, const Tensor& b);
Tensor subtract(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);   // Hadamard (element-wise) product

Tensor operator+(const Tensor& a, const Tensor& b);  // = add
Tensor operator-(const Tensor& a, const Tensor& b);  // = subtract
Tensor operator*(const Tensor& a, const Tensor& b);  // = multiply, NOT matmul
```

Returns a new tensor and leaves the inputs unchanged. Runs in O(n).

- **Throws** `std::invalid_argument` unless both shapes are identical. Use the broadcasting versions for differing shapes.

## Broadcasting

```cpp
Tensor add_broadcast(const Tensor& a, const Tensor& b);
Tensor subtract_broadcast(const Tensor& a, const Tensor& b);
Tensor multiply_broadcast(const Tensor& a, const Tensor& b);
```

These follow NumPy broadcasting rules. Shapes are compared from the last dimension backwards, and two dimensions are compatible when they are equal or one of them is 1. Missing leading dimensions count as 1.

| `a` | `b` | Result |
|---|---|---|
| `(2, 3)` | `(3)` | `(2, 3)`: `b` added to every row |
| `(2, 3)` | `(2, 1)` | `(2, 3)`: `b` added to every column |
| `(2, 1, 3)` | `(4, 1)` | `(2, 4, 3)` |
| `(2, 2)` | scalar | `(2, 2)` |

- **Throws** `std::invalid_argument` if the shapes are not broadcast-compatible.

Multiplying by a scalar tensor is the idiom for scaling:

```cpp
const Tensor half(std::vector<std::size_t>{}, 0.5);
Tensor scaled = multiply_broadcast(t, half);
```

## Matrix operations

```cpp
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor matmul_parallel(const Tensor& a, const Tensor& b);
Tensor transpose(const Tensor& a);
```

| Function | Behavior |
|---|---|
| `matmul` | `(m × k) @ (k × n) → (m × n)`. Hybrid: picks sequential or OpenMP-parallel execution from the problem size (see [DESIGN.md](DESIGN.md)). Use this one. |
| `matmul_parallel` | Always uses the OpenMP path. It runs sequentially if the library was built without OpenMP. |
| `transpose` | `(m × n) → (n × m)` |

- `matmul` / `matmul_parallel` **throw** `std::invalid_argument` if either input is not 2-D or the inner dimensions differ.
- `transpose` **throws** `std::invalid_argument` if the input is not 2-D.
- Zero-sized dimensions are allowed. For example, `(2 × 0) @ (0 × 3)` is a `2 × 3` tensor of zeros.

## Reductions

```cpp
double sum(const Tensor& a);
double mean(const Tensor& a);
double min(const Tensor& a);
double max(const Tensor& a);
```

These reduce over every element regardless of shape.

- `sum` of an empty tensor is `0.0`.
- `mean`, `min` and `max` **throw** `std::invalid_argument` on an empty tensor.

---

## Printing (`IO.h`)

```cpp
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
std::string to_string(const Tensor& tensor);
```

Prints tensors as nested brackets in row-major order. The output respects the stream's formatting flags, such as `std::setprecision`.

| Tensor | Output |
|---|---|
| 2×2 | `[[1, 2], [3, 4]]` |
| vector | `[1.5, -2, 0]` |
| scalar | `42` |
| shape `(0, 3)` | `[]` |

## Version (`Version.h`)

```cpp
#define MINITENSOR_VERSION_MAJOR 1
#define MINITENSOR_VERSION_MINOR 0
#define MINITENSOR_VERSION_PATCH 0
#define MINITENSOR_VERSION_STRING "1.0.0"

const char* version() noexcept;       // version of the compiled library
bool openmp_enabled() noexcept;       // whether it was built with OpenMP
```

`Version.h` is generated by CMake at configure time.

---

## CUDA backend (`Cuda.h`)

Optional, in namespace `minitensor::cuda`. Available only when the library is built with `-DMINITENSOR_USE_CUDA=ON`. Set `CMAKE_CUDA_ARCHITECTURES` to your GPU, such as `89` for an RTX 40 series, to build native code.

```cpp
#include "minitensor/Cuda.h"

// Simple form: upload, compute, wait, download
Tensor c = cuda::matmul(a, b);

// Keeping data on the GPU between operations
cuda::DeviceTensor gpu_a(a);
cuda::DeviceTensor gpu_b(b);
cuda::DeviceTensor gpu_c = cuda::matmul(gpu_a, gpu_b);  // returns before the GPU has finished
cuda::synchronize();                                    // call this before reading a clock
Tensor result = gpu_c.download();
```

| Function | Notes |
|---|---|
| `available()` | True when a CUDA device can be used |
| `device_name()`, `runtime_version()` | Describe the device and CUDA runtime |
| `synchronize()` | Blocks until all queued GPU work has finished |
| `add`, `subtract`, `multiply` | Same-shape element-wise operations |
| `add_broadcast`, `subtract_broadcast`, `multiply_broadcast` | NumPy broadcasting rules, up to 8 dimensions |
| `matmul` | 2-D only, uses cuBLAS `dgemm` |
| `transpose` | 2-D only |
| `sum` | Always blocks, because the answer has to return to the host |

Each operation accepts `Tensor` or `DeviceTensor`. Shape errors throw `std::invalid_argument`, like the CPU functions. CUDA failures, such as running out of GPU memory, throw `std::runtime_error`. `mean`, `min`, `max`, `reshape` and `matmul_parallel` have no GPU version.

## Lower-level building blocks

These are public, but most users only need `Tensor`.

### Shape

Holds the dimensions, the row-major strides and the element count.

| Member | Returns |
|---|---|
| `rank()` | Number of dimensions |
| `num_elements()` | Product of dimensions (1 for a scalar, 0 if any dimension is 0) |
| `dimensions()` | `const std::vector<std::size_t>&` |
| `strides()` | Elements to skip per step along each axis. The last axis has stride 1. |

### Storage

An owning, 64-byte aligned buffer of `double`s with deep-copy and move semantics. Its members are `size()`, `capacity()`, `data()`, `fill(value)` and `swap(other)`.
