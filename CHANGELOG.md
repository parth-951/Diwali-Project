# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Optional CUDA backend (`minitensor/Cuda.h`, `-DMINITENSOR_USE_CUDA=ON`, off by default): GPU `matmul` through cuBLAS, plus `transpose`, element-wise operations, broadcasting and `sum`.
- `verification/`: a comparison with NumPy, CPU and CUDA benchmarks, and reproducibility checks. `run_verification.py` generates `RESULT_VERIFICATION.md` from a real run.

## [1.0.0] - 2026-11-01

First public release.

### Added: Phase 6 (testing and application development)

- `tests/test_phase6.cpp`: 19 integration, property and edge-case tests (135 checks), covering both matmul dispatch paths against a reference implementation, algebraic identities, 3-D broadcasting, reshape round trips, indexing errors, copy/move semantics and printing.
- `tests/test_utils.h`: a dependency-free test harness that keeps running after failures and works in Release builds.
- Example applications that check their own results and run under `ctest`:
  - `quickstart`: an API tour
  - `linear_regression`: gradient descent on 1000 samples
  - `xor_neural_network`: a 2-4-1 network trained with backpropagation
  - `image_processing`: grayscale conversion, contrast stretch and vignette with ASCII rendering
- `operator<<` and `to_string()` for printing tensors (`minitensor/IO.h`).
- `version()`, `openmp_enabled()` and `MINITENSOR_VERSION_*` macros (`minitensor/Version.h`).
- `minitensor/minitensor.h` umbrella header.

### Added: Phase 7 (packaging, documentation and public release)

- CMake install rules with an exported `minitensor::minitensor` target, so installed copies work with `find_package(MiniTensor)`.
- Support for use via `FetchContent` and `add_subdirectory`. Tests, examples and benchmarks are skipped automatically when MiniTensor is a subproject.
- A relocatable pkg-config file (`minitensor.pc`).
- CPack packaging: `.deb` and `.tar.gz` on Linux, `.zip` on Windows, plus a source tarball.
- CI on GitHub Actions: GCC and Clang on Ubuntu 22.04 and 24.04, with and without OpenMP, an AddressSanitizer + UndefinedBehaviorSanitizer job, and an install-and-consume check.
- A release workflow that publishes packages to GitHub Releases when a `v*` tag is pushed.
- `docs/API.md`, `docs/DESIGN.md`, a rewritten README and an MIT license.
- Build options `MINITENSOR_USE_OPENMP`, `MINITENSOR_BUILD_TESTS`, `MINITENSOR_BUILD_EXAMPLES`, `MINITENSOR_BUILD_BENCHMARKS` and `MINITENSOR_INSTALL`.

### Changed

- OpenMP is now optional, as the challenge allows. Without it the library builds normally and `matmul_parallel()` runs sequentially.
- The default build type is `Release` when none is given.

## [0.1.0]

Phases 1–5: tensor representation and aligned storage, multidimensional indexing and reshape, element-wise arithmetic and matrix multiplication, transpose, broadcasting and reductions, and OpenMP parallel and hybrid matrix multiplication with benchmarks.
