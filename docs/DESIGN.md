# MiniTensor Design Notes

This document explains *why* the library is built the way it is. It is meant to help contributors and reviewers understand and defend the implementation.

## 1. Architecture

```
Tensor  ── owns ──▶  Shape    (dimensions, strides, element count)
        ── owns ──▶  Storage  (64-byte aligned double buffer)

Ops.cpp:  free functions that take Tensors and return new Tensors
IO.cpp:   printing
```

- **Separation of concerns.** `Storage` only manages memory, and `Shape` only does index arithmetic. `Tensor` combines them and adds bounds-checked access. Each piece can be tested on its own.
- **Value semantics.** A `Tensor` owns its data. Copies are deep, and moves are O(1) pointer transfers. Operations return new tensors instead of modifying inputs, which makes them safe to compose (`(a + b) * c - a`). Thanks to move semantics, returning results costs no extra copy.
- **Free functions for operations.** `add`, `matmul` and the others are non-member functions, so the class stays small and new operations can be added without touching `Tensor`.

## 2. Memory layout

### Contiguous, aligned storage

All elements live in one heap block allocated with aligned `operator new` (`std::align_val_t{64}`):

- **One allocation per tensor**, with no per-row pointers. That means fewer allocations and no pointer chasing.
- **64-byte alignment** matches the cache-line size on x86-64 and most ARM cores, so a tensor never starts mid-line. It is also the alignment that AVX-512 loads prefer, which helps the compiler auto-vectorize the element-wise loops.
- **Overflow checks.** `Shape` checks that the product of the dimensions fits in `std::size_t`, and `Storage` checks the byte count before allocating. A huge shape throws `std::overflow_error` instead of silently allocating too little.

### Row-major strides

For dimensions `(d0, d1, …, dn-1)` the strides are:

```
stride[n-1] = 1
stride[i]   = stride[i+1] * d[i+1]
offset(i0, …, in-1) = Σ ik * stride[k]
```

For example, a `2×3×4` tensor has strides `(12, 4, 1)`, so element `(1, 2, 3)` is at offset `12 + 8 + 3 = 23`.

Because the data is always contiguous and row-major, **`reshape` is O(1)**: it swaps in a new `Shape` and never touches the buffer. Tests check that the data pointer doesn't change.

## 3. Element-wise operations

Same-shape operations are single flat loops over raw pointers. They involve no index arithmetic or bounds checks, and they are auto-vectorized by the compiler at `-O2`/`-O3`.

## 4. Broadcasting

`broadcast_shape()` aligns the two shapes from the right and computes the output shape, rejecting incompatible pairs. For each output element, `broadcast_offset()` converts the flat output index into coordinates. Any coordinate where the input dimension is 1 is skipped, which repeats that input element along the axis.

A scalar tensor (rank 0) broadcasts against anything, because every dimension of it counts as 1.

**Trade-off:** computing coordinates costs a division and a modulo per dimension per element. That keeps the code simple and correct for any rank, but it is slower than the same-shape fast path. See *Future work*.

## 5. Matrix multiplication

### Loop order

The textbook `i-j-k` order reads `B` down a column, which jumps `n` elements in memory on every step and wastes most of every cache line. MiniTensor uses **`i-k-j`** instead:

```cpp
for i:
    for k:
        a_ik = A[i][k]                // reused across the whole inner loop
        for j:
            C[i][j] += a_ik * B[k][j] // B and C walked sequentially
```

The inner loop now streams through `B` and `C` contiguously. That is cache-friendly and easy for the compiler to vectorize.

### OpenMP parallelism

`matmul_parallel` splits the outer `i` loop across threads with `#pragma omp parallel for schedule(static)`. Each thread writes to its own rows of `C`, so no locks or atomics are needed, and static scheduling suits the equal work per row.

### Hybrid dispatch

Creating threads costs roughly tens of microseconds, so parallelism hurts small problems. `matmul` estimates the work as `m·k·n` multiply-adds and goes parallel only when **both** of these hold:

- `m·k·n ≥ 256³`, so there is enough work to pay for the threads
- `m ≥ 128`, so there are enough rows to split among threads

Otherwise it runs the sequential loop. The Phase 5 benchmarks show the hybrid matching or beating pure OpenMP at every size. The Phase 6 tests exercise both branches against a reference implementation.

### OpenMP is optional

OpenMP is linked `PRIVATE` and the pragma is guarded by `#ifdef _OPENMP`, so the library builds and passes every test without it (`-DMINITENSOR_USE_OPENMP=OFF`). CI checks this configuration. `openmp_enabled()` reports how the library was built.

## 6. Error handling

Errors throw standard exceptions with clear messages. The library never returns error codes or fails silently.

| Situation | Exception |
|---|---|
| Size overflow | `std::overflow_error` |
| Wrong index count, mismatched shapes, wrong rank, invalid reshape, reduction of an empty tensor | `std::invalid_argument` |
| Index out of bounds | `std::out_of_range` |

Operations validate their inputs before allocating or writing anything, so a failed call leaves its inputs untouched. For example, a failed `reshape` keeps the old shape, and a test checks this.

## 7. Testing strategy (Phase 6)

| Layer | Where | What it catches |
|---|---|---|
| Unit tests per phase | `tests/test_phase1..5.cpp` | Each feature against hand-computed values |
| Reference comparison | `test_phase6.cpp` | `matmul` and `matmul_parallel`, on both the sequential and parallel paths, compared with a naive implementation on large non-uniform inputs |
| Algebraic properties | `test_phase6.cpp` | `(AB)C = A(BC)`, `A(B+C) = AB+AC`, `(AB)ᵀ = BᵀAᵀ`, `(Aᵀ)ᵀ = A`, `IA = A` |
| Edge cases | `test_phase6.cpp` | Scalars, zero-sized dimensions, out-of-range indices, bad shapes, self-assignment, failed reshape |
| End-to-end applications | `examples/*.cpp` | Linear regression, a neural network and an image pipeline must produce correct results |
| Memory and UB checks | CI `sanitizers` job | AddressSanitizer and UndefinedBehaviorSanitizer over the whole suite |
| Portability | CI matrix | GCC and Clang, Ubuntu 22.04 and 24.04, with and without OpenMP |
| Packaging | CI install job | A separate project builds against the *installed* library via `find_package` and via pkg-config |

Test inputs use small integers, so floating-point sums are exact and comparisons don't depend on rounding. The applications generate data from raw `std::mt19937` output, which the C++ standard fixes exactly, so results are the same on every platform. The standard does not fix `<random>` distributions, so they are avoided.

`tests/test_utils.h` is a tiny harness. Unlike `assert`, it keeps running after a failure, reports `file:line` for each failed check, and works in Release builds.

## 8. Known limitations and future work

- **Only `double` elements.** A `Tensor<T>` template would add `float` (twice the SIMD throughput) and integer types.
- **No views or slices.** Every operation allocates a new result. Strided views would make transpose and slicing O(1).
- **Reductions cover the whole tensor only.** Per-axis reductions (`sum(t, axis)`) are not supported yet. The examples work around this with `ones @ matrix`.
- **Faster broadcasting.** Precomputed per-input strides with a running odometer index would remove the per-element division and modulo.
- **Matmul.** Cache blocking tuned per architecture, register tiling or explicit SIMD would increase throughput further. The Phase 5 blocked version did not yet beat `i-k-j` on the test machine.
- **No in-place operators** (`+=`, `*=`), which would avoid allocations in training loops.
