#ifndef MINITENSOR_OPS_H
#define MINITENSOR_OPS_H

#include "minitensor/Tensor.h"

namespace minitensor {

// Every operation works on Tensor (double) and FloatTensor (float).
// Both inputs must have the same element type.

template <typename T>
BasicTensor<T> add(const BasicTensor<T>& a, const BasicTensor<T>& b);
template <typename T>
BasicTensor<T> subtract(const BasicTensor<T>& a, const BasicTensor<T>& b);
template <typename T>
BasicTensor<T> multiply(const BasicTensor<T>& a, const BasicTensor<T>& b);

template <typename T>
BasicTensor<T> matmul(const BasicTensor<T>& a, const BasicTensor<T>& b);
template <typename T>
BasicTensor<T> matmul_parallel(const BasicTensor<T>& a, const BasicTensor<T>& b);

template <typename T>
BasicTensor<T> transpose(const BasicTensor<T>& a);

// Phase 4: Broadcasting
template <typename T>
BasicTensor<T> add_broadcast(const BasicTensor<T>& a, const BasicTensor<T>& b);
template <typename T>
BasicTensor<T> subtract_broadcast(const BasicTensor<T>& a, const BasicTensor<T>& b);
template <typename T>
BasicTensor<T> multiply_broadcast(const BasicTensor<T>& a, const BasicTensor<T>& b);

// Phase 4: Reduction Operations
// sum and mean accumulate in double for both element types.
template <typename T>
double sum(const BasicTensor<T>& a);
template <typename T>
double mean(const BasicTensor<T>& a);
template <typename T>
T min(const BasicTensor<T>& a);
template <typename T>
T max(const BasicTensor<T>& a);

template <typename T>
BasicTensor<T> operator+(const BasicTensor<T>& a, const BasicTensor<T>& b);
template <typename T>
BasicTensor<T> operator-(const BasicTensor<T>& a, const BasicTensor<T>& b);
template <typename T>
BasicTensor<T> operator*(const BasicTensor<T>& a, const BasicTensor<T>& b);

}

#endif
