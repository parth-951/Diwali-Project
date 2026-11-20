#include "minitensor/Storage.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace minitensor {

template <typename T>
void BasicStorage<T>::allocate(std::size_t size) {
    if (size == 0) {
        buffer_ = nullptr;
        size_ = 0;
        return;
    }

    if (size > std::numeric_limits<std::size_t>::max() /
                   sizeof(T)) {
        throw std::overflow_error("Storage size overflow");
    }

    const std::size_t bytes = size * sizeof(T);

    buffer_ = static_cast<T*>(
        ::operator new(
            bytes,
            std::align_val_t{ALIGNMENT}
        )
    );

    size_ = size;
}

template <typename T>
void BasicStorage<T>::release() noexcept {
    if (buffer_ != nullptr) {
        ::operator delete(
            buffer_,
            std::align_val_t{ALIGNMENT}
        );
    }

    buffer_ = nullptr;
    size_ = 0;
}

template <typename T>
BasicStorage<T>::BasicStorage(
    std::size_t size,
    T initialValue
)
    : buffer_(nullptr),
      size_(0) {
    allocate(size);

    if (size_ > 0) {
        std::fill_n(buffer_, size_, initialValue);
    }
}

template <typename T>
BasicStorage<T>::~BasicStorage() {
    release();
}

template <typename T>
BasicStorage<T>::BasicStorage(const BasicStorage& other)
    : buffer_(nullptr),
      size_(0) {
    allocate(other.size_);

    if (size_ > 0) {
        std::memcpy(
            buffer_,
            other.buffer_,
            size_ * sizeof(T)
        );
    }
}

template <typename T>
BasicStorage<T>& BasicStorage<T>::operator=(const BasicStorage& other) {
    if (this != &other) {
        BasicStorage temp(other);
        swap(temp);
    }

    return *this;
}

template <typename T>
BasicStorage<T>::BasicStorage(BasicStorage&& other) noexcept
    : buffer_(other.buffer_),
      size_(other.size_) {
    other.buffer_ = nullptr;
    other.size_ = 0;
}

template <typename T>
BasicStorage<T>& BasicStorage<T>::operator=(BasicStorage&& other) noexcept {
    if (this != &other) {
        release();

        buffer_ = other.buffer_;
        size_ = other.size_;

        other.buffer_ = nullptr;
        other.size_ = 0;
    }

    return *this;
}

template <typename T>
std::size_t BasicStorage<T>::size() const noexcept {
    return size_;
}

template <typename T>
std::size_t BasicStorage<T>::capacity() const noexcept {
    return size_;
}

template <typename T>
T* BasicStorage<T>::data() noexcept {
    return buffer_;
}

template <typename T>
const T* BasicStorage<T>::data() const noexcept {
    return buffer_;
}

template <typename T>
void BasicStorage<T>::fill(T value) {
    if (size_ > 0) {
        std::fill_n(buffer_, size_, value);
    }
}

template <typename T>
void BasicStorage<T>::swap(BasicStorage& other) noexcept {
    std::swap(buffer_, other.buffer_);
    std::swap(size_, other.size_);
}

template class BasicStorage<float>;
template class BasicStorage<double>;

} // namespace minitensor
