# MiniTensor

[![CI](https://github.com/parth-951/Diwali-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/parth-951/Diwali-Project/actions/workflows/ci.yml)

A small, fast, dependency-free **C++17 multidimensional tensor library**, built from scratch for the Diwali 2026 Mini Tensor Library challenge. It provides contiguous 64-byte aligned storage, multidimensional indexing, element-wise arithmetic, broadcasting, reductions, and OpenMP-parallel matrix multiplication.

```cpp
#include <minitensor/minitensor.h>
#include <iostream>

int main()
{
    using namespace minitensor;

    Tensor a({2, 3});
    for (std::size_t i = 0; i < a.size(); ++i)
        a.flat_at(i) = static_cast<double>(i + 1);   // [[1, 2, 3], [4, 5, 6]]

    Tensor b({2, 3}, 10.0);

    std::cout << a + b << '\n';                       // [[11, 12, 13], [14, 15, 16]]
    std::cout << matmul(a, transpose(a)) << '\n';     // [[14, 32], [32, 77]]
    std::cout << sum(a) << ' ' << mean(a) << '\n';    // 21 3.5
}
```

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Build, test and install](#build-test-and-install)
- [Using MiniTensor in your project](#using-minitensor-in-your-project)
- [Example applications](#example-applications)
- [Performance](#performance)
- [Project layout](#project-layout)
- [Documentation](#documentation)
- [Releasing](#releasing)
- [License](#license)

## Features

The library was developed in seven phases, following the challenge brief.

| Phase | Topic | What it provides |
|---|---|---|
| 1 | Tensor representation and memory management | Arbitrary-rank shapes, contiguous 64-byte aligned storage, overflow-checked sizes, deep copy and move, scalar and empty tensors |
| 2 | Multidimensional indexing and memory layout | Row-major layout with strides, bounds-checked `at()` / `operator()`, `flat_at()`, zero-copy `reshape()` |
| 3 | Mathematical operations | Element-wise `+ - *`, matrix multiplication, OpenMP parallel and hybrid matmul |
| 4 | Advanced tensor operations | `transpose()`, NumPy-style broadcasting, `sum` / `mean` / `min` / `max` reductions |
| 5 | Performance and parallel computing | Benchmarks for baseline, blocked, OpenMP and hybrid matmul |
| 6 | Testing and application development | A Phase 6 test suite with 135 checks (property, integration and edge-case), tensor printing, and four example applications that check their own results |
| 7 | Packaging, documentation and public release | CMake install with `find_package` support, pkg-config, `.deb`/`.tar.gz` packages, CI on GCC and Clang, sanitizer runs, tagged GitHub releases, API and design docs |

## Requirements

- Ubuntu Linux 22.04 or later (the primary target). Other platforms with a C++17 compiler also work.
- GCC 9+ or Clang 10+ (any C++17 compiler)
- CMake 3.16 or later
- OpenMP *(optional)*. Without it, everything still works and `matmul_parallel()` runs sequentially.

On Ubuntu:

```bash
sudo apt install build-essential cmake
```

For OpenMP with Clang, also install `libomp-dev`. GCC ships with OpenMP support.

## Build, test and install

```bash
git clone https://github.com/parth-951/Diwali-Project.git
cd Diwali-Project
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
sudo cmake --install build          # installs to /usr/local by default
```

`ctest` runs the six phase test suites and the four example applications. A passing run ends with `100% tests passed`.

### Build options

| Option | Default | Purpose |
|---|---|---|
| `MINITENSOR_USE_OPENMP` | `ON` | Parallelize `matmul_parallel()` / large `matmul()` with OpenMP |
| `MINITENSOR_BUILD_TESTS` | `ON`* | Build the test suite |
| `MINITENSOR_BUILD_EXAMPLES` | `ON`* | Build the example applications |
| `MINITENSOR_BUILD_BENCHMARKS` | `ON`* | Build the benchmarks |
| `MINITENSOR_INSTALL` | `ON`* | Generate install and packaging rules |
| `BUILD_SHARED_LIBS` | `OFF` | Build a shared library instead of a static one |

\* `ON` when MiniTensor is the top-level project and `OFF` when it is used as a subproject, so it doesn't add targets to your build.

Install to a custom location with `cmake --install build --prefix ~/.local`.

### Windows (MinGW)

```powershell
cmake -S . -B build-release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
ctest --test-dir build-release --output-on-failure
```

## Using MiniTensor in your project

### 1. `find_package` after installing

```cmake
find_package(MiniTensor 1.0 REQUIRED)
target_link_libraries(my_app PRIVATE minitensor::minitensor)
```

If you installed to a non-standard prefix, pass `-DCMAKE_PREFIX_PATH=/that/prefix`. [examples/consumer](examples/consumer) is a complete standalone project that does this.

### 2. FetchContent (no install needed)

```cmake
include(FetchContent)
FetchContent_Declare(
    minitensor
    GIT_REPOSITORY https://github.com/parth-951/Diwali-Project.git
    GIT_TAG v1.0.0
)
FetchContent_MakeAvailable(minitensor)

target_link_libraries(my_app PRIVATE minitensor::minitensor)
```

### 3. `add_subdirectory`

Copy or submodule the repository into your project, then:

```cmake
add_subdirectory(third_party/minitensor)
target_link_libraries(my_app PRIVATE minitensor::minitensor)
```

### 4. pkg-config (plain Makefiles or a compiler command)

```bash
g++ -std=c++17 main.cpp $(pkg-config --cflags --libs minitensor) -o main
```

### 5. Debian package

Download `minitensor_<version>_amd64.deb` from the [Releases](https://github.com/parth-951/Diwali-Project/releases) page:

```bash
sudo apt install ./minitensor_1.0.0_amd64.deb
```

## Example applications

All examples build into the build directory. Each one checks its own result and exits non-zero on failure, so they also run as tests under `ctest`.

| Program | What it does | Library features used |
|---|---|---|
| [`quickstart`](examples/quickstart.cpp) | A tour of the whole API | Everything |
| [`linear_regression`](examples/linear_regression.cpp) | Learns `y = 3x₁ − 2x₂ + 5` from 1000 noisy samples with gradient descent | `matmul`, `transpose`, broadcasting, `mean` |
| [`xor_neural_network`](examples/xor_neural_network.cpp) | Trains a 2-4-1 neural network with backpropagation to learn XOR | `matmul`, `transpose`, broadcasting, element-wise ops, `reshape` |
| [`image_processing`](examples/image_processing.cpp) | RGB to grayscale, contrast stretch and vignette on an image tensor, rendered as ASCII art | `reshape`, `matmul`, `min`/`max`, broadcasting |
| [`basic_example`](examples/main.cpp) | Minimal tensor creation | `Tensor` |

```bash
./build/xor_neural_network
```

```
Training a 2-4-1 network on XOR
  epoch     1  loss 0.693579
  ...
  epoch 10000  loss 0.000411

  x1  x2  target  prediction
   0   0       0      0.0001
   0   1       1      0.9996
   1   0       1      0.9996
   1   1       0      0.0007

XOR learned.
```

## Performance

One Release-build benchmark sample on Windows with MinGW:

| Matrix Size | Baseline | OpenMP | Library Hybrid | Hybrid Speedup |
|---|---:|---:|---:|---:|
| 256 × 256 | 2.660 ms | 0.643 ms | 0.561 ms | 4.74× |
| 512 × 512 | 20.453 ms | 3.962 ms | 3.949 ms | 5.18× |
| 1024 × 1024 | 173.768 ms | 35.499 ms | 31.633 ms | 5.49× |

The blocked implementation did not improve performance in this sample. OpenMP overhead can make parallel execution slower for smaller matrices, which is why `matmul()` only switches to the parallel path above a work threshold (see [docs/DESIGN.md](docs/DESIGN.md)). Timings depend on hardware, compiler, thread count and system load, so these results are not a universal performance guarantee.

Run the benchmarks yourself:

```bash
./build/benchmark_matmul
```

## Project layout

```
include/minitensor/   Public headers (minitensor.h includes everything)
src/                  Library implementation
tests/                Phase 1–6 test suites and the test harness
examples/             Example applications and a find_package consumer project
benchmarks/           Performance benchmarks
docs/                 API reference and design notes
cmake/                Package config, pkg-config and version header templates
.github/workflows/    CI and release automation
```

## Documentation

- [docs/API.md](docs/API.md): reference for every public class and function, including the exceptions each one throws
- [docs/DESIGN.md](docs/DESIGN.md): memory layout, broadcasting, the hybrid matmul strategy, the testing approach and known limitations
- [CHANGELOG.md](CHANGELOG.md): release history

## Releasing

1. Update `VERSION` in `project()` in [CMakeLists.txt](CMakeLists.txt) and add an entry to [CHANGELOG.md](CHANGELOG.md).
2. Commit, then tag and push:

   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

3. The [release workflow](.github/workflows/release.yml) builds and tests the library, then publishes a GitHub Release with the `.deb`, binary `.tar.gz` and source tarball attached.

## License

MiniTensor is released under the [MIT License](LICENSE).
