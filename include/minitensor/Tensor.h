#ifndef MINITENSOR_TENSOR_H
#define MINITENSOR_TENSOR_H

#include "minitensor/Shape.h"
#include "minitensor/Storage.h"

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace minitensor {

// T is the element type: float or double. Use the aliases Tensor (double)
// and FloatTensor (float) below.
template <typename T>
class BasicTensor {
private:
    Shape shape_;
    BasicStorage<T> storage_;

    std::size_t offset(
        const std::vector<std::size_t>& indices
    ) const;

public:
    using value_type = T;

    explicit BasicTensor(
        const std::vector<std::size_t>& dimensions,
        T initial_value = T{}
    );

    BasicTensor(
        std::initializer_list<std::size_t> dimensions,
        T initial_value = T{}
    );

    BasicTensor(const BasicTensor&) = default;
    BasicTensor& operator=(const BasicTensor&) = default;

    BasicTensor(BasicTensor&&) noexcept = default;
    BasicTensor& operator=(BasicTensor&&) noexcept = default;

    std::size_t size() const noexcept;
    std::size_t rank() const noexcept;

    const Shape& shape() const noexcept;

    T* data() noexcept;
    const T* data() const noexcept;

    void fill(T value);

    T& at(
        const std::vector<std::size_t>& indices
    );

    const T& at(
        const std::vector<std::size_t>& indices
    ) const;

    T& at(
        std::initializer_list<std::size_t> indices
    );

    const T& at(
        std::initializer_list<std::size_t> indices
    ) const;

    T& operator()(
        std::initializer_list<std::size_t> indices
    );

    const T& operator()(
        std::initializer_list<std::size_t> indices
    ) const;

    T& operator()(
        const std::vector<std::size_t>& indices
    );

    const T& operator()(
        const std::vector<std::size_t>& indices
    ) const;

    T& flat_at(std::size_t index);
    const T& flat_at(std::size_t index) const;

    void reshape(
        const std::vector<std::size_t>& dimensions
    );

    void reshape(
        std::initializer_list<std::size_t> dimensions
    );
};

// Defined in Tensor.cpp for these two element types only
extern template class BasicTensor<float>;
extern template class BasicTensor<double>;

using Tensor = BasicTensor<double>;
using FloatTensor = BasicTensor<float>;

} // namespace minitensor

#endif
