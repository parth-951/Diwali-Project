#include "minitensor/IO.h"

#include <cstddef>
#include <ostream>
#include <sstream>

namespace minitensor {

namespace {

// Print one axis, recursing into the next axis until the innermost one
template <typename T>
void print_axis(
    std::ostream& os,
    const BasicTensor<T>& tensor,
    std::size_t axis,
    std::size_t offset)
{
    const auto& dimensions = tensor.shape().dimensions();
    const auto& strides = tensor.shape().strides();

    os << '[';

    for (std::size_t i = 0; i < dimensions[axis]; ++i)
    {
        if (i > 0)
        {
            os << ", ";
        }

        const std::size_t position = offset + i * strides[axis];

        if (axis + 1 == dimensions.size())
        {
            os << tensor.data()[position];
        }
        else
        {
            print_axis(os, tensor, axis + 1, position);
        }
    }

    os << ']';
}

}

template <typename T>
std::ostream& operator<<(std::ostream& os, const BasicTensor<T>& tensor)
{
    if (tensor.rank() == 0)
    {
        return os << tensor.data()[0];
    }

    print_axis(os, tensor, 0, 0);

    return os;
}

template <typename T>
std::string to_string(const BasicTensor<T>& tensor)
{
    std::ostringstream stream;
    stream << tensor;
    return stream.str();
}

template std::ostream& operator<<(std::ostream&, const BasicTensor<float>&);
template std::ostream& operator<<(std::ostream&, const BasicTensor<double>&);
template std::string to_string(const BasicTensor<float>&);
template std::string to_string(const BasicTensor<double>&);

}
