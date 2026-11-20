#include "minitensor/Tensor.h"

#include <stdexcept>
#include <utility>

namespace minitensor {

template <typename T>
BasicTensor<T>::BasicTensor(
    const std::vector<std::size_t>& dimensions,
    T initial_value
)
    : shape_(dimensions),
      storage_(shape_.num_elements(), initial_value) {
}

template <typename T>
BasicTensor<T>::BasicTensor(
    std::initializer_list<std::size_t> dimensions,
    T initial_value
)
    : BasicTensor(
        std::vector<std::size_t>(dimensions),
        initial_value
    ) {
}

template <typename T>
std::size_t BasicTensor<T>::size() const noexcept {
    return storage_.size();
}

template <typename T>
std::size_t BasicTensor<T>::rank() const noexcept {
    return shape_.rank();
}

template <typename T>
const Shape& BasicTensor<T>::shape() const noexcept {
    return shape_;
}

template <typename T>
T* BasicTensor<T>::data() noexcept {
    return storage_.data();
}

template <typename T>
const T* BasicTensor<T>::data() const noexcept {
    return storage_.data();
}

template <typename T>
void BasicTensor<T>::fill(T value) {
    storage_.fill(value);
}

template <typename T>
std::size_t BasicTensor<T>::offset(
    const std::vector<std::size_t>& indices
) const {
    if (indices.size() != rank()) {
        throw std::invalid_argument(
            "Tensor index rank does not match tensor rank"
        );
    }

    const auto& dimensions = shape_.dimensions();
    const auto& strides = shape_.strides();

    std::size_t result = 0;

    for (std::size_t axis = 0; axis < indices.size(); ++axis) {
        if (indices[axis] >= dimensions[axis]) {
            throw std::out_of_range(
                "Tensor index is out of bounds"
            );
        }

        result += indices[axis] * strides[axis];
    }

    return result;
}

template <typename T>
T& BasicTensor<T>::at(
    const std::vector<std::size_t>& indices
) {
    return storage_.data()[offset(indices)];
}

template <typename T>
const T& BasicTensor<T>::at(
    const std::vector<std::size_t>& indices
) const {
    return storage_.data()[offset(indices)];
}

template <typename T>
T& BasicTensor<T>::at(
    std::initializer_list<std::size_t> indices
) {
    return at(std::vector<std::size_t>(indices));
}

template <typename T>
const T& BasicTensor<T>::at(
    std::initializer_list<std::size_t> indices
) const {
    return at(std::vector<std::size_t>(indices));
}

template <typename T>
T& BasicTensor<T>::operator()(
    std::initializer_list<std::size_t> indices
) {
    return at(indices);
}

template <typename T>
const T& BasicTensor<T>::operator()(
    std::initializer_list<std::size_t> indices
) const {
    return at(indices);
}

template <typename T>
T& BasicTensor<T>::operator()(
    const std::vector<std::size_t>& indices
) {
    return at(indices);
}

template <typename T>
const T& BasicTensor<T>::operator()(
    const std::vector<std::size_t>& indices
) const {
    return at(indices);
}

template <typename T>
T& BasicTensor<T>::flat_at(std::size_t index) {
    if (index >= size()) {
        throw std::out_of_range(
            "Tensor flat index is out of bounds"
        );
    }

    return storage_.data()[index];
}

template <typename T>
const T& BasicTensor<T>::flat_at(std::size_t index) const {
    if (index >= size()) {
        throw std::out_of_range(
            "Tensor flat index is out of bounds"
        );
    }

    return storage_.data()[index];
}

template <typename T>
void BasicTensor<T>::reshape(
    const std::vector<std::size_t>& dimensions
) {
    Shape new_shape(dimensions);

    if (new_shape.num_elements() != size()) {
        throw std::invalid_argument(
            "Reshape must preserve the number of elements"
        );
    }

    shape_ = std::move(new_shape);
}

template <typename T>
void BasicTensor<T>::reshape(
    std::initializer_list<std::size_t> dimensions
) {
    reshape(std::vector<std::size_t>(dimensions));
}

template class BasicTensor<float>;
template class BasicTensor<double>;

} // namespace minitensor
