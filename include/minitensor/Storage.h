
#ifndef MINITENSOR_STORAGE_H
#define MINITENSOR_STORAGE_H

#include <cstddef>
#include <type_traits>

namespace minitensor {

// Aligned, contiguous element buffer. T is float or double.
template <typename T>
class BasicStorage {
    static_assert(
        std::is_same<T, float>::value || std::is_same<T, double>::value,
        "MiniTensor elements must be float or double"
    );

private:
    static constexpr std::size_t ALIGNMENT = 64;

    T* buffer_;
    std::size_t size_;

    void allocate(std::size_t size);
    void release() noexcept;

public:
    explicit BasicStorage(
        std::size_t size,
        T initialValue = T{}
    );

    ~BasicStorage();

    BasicStorage(const BasicStorage& other);
    BasicStorage& operator=(const BasicStorage& other);

    BasicStorage(BasicStorage&& other) noexcept;
    BasicStorage& operator=(BasicStorage&& other) noexcept;

    std::size_t size() const noexcept;
    std::size_t capacity() const noexcept;

    T* data() noexcept;
    const T* data() const noexcept;

    void fill(T value);

    void swap(BasicStorage& other) noexcept;
};

// Defined in Storage.cpp for these two element types only
extern template class BasicStorage<float>;
extern template class BasicStorage<double>;

using Storage = BasicStorage<double>;

}

#endif
