#include "minitensor/IO.h"

#include <cstddef>
#include <ostream>
#include <sstream>

namespace minitensor {

namespace {

// Print one axis, recursing into the next axis until the innermost one
void print_axis(
    std::ostream& os,
    const Tensor& tensor,
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

std::ostream& operator<<(std::ostream& os, const Tensor& tensor)
{
    if (tensor.rank() == 0)
    {
        return os << tensor.data()[0];
    }

    print_axis(os, tensor, 0, 0);

    return os;
}

std::string to_string(const Tensor& tensor)
{
    std::ostringstream stream;
    stream << tensor;
    return stream.str();
}

}
