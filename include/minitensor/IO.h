#ifndef MINITENSOR_IO_H
#define MINITENSOR_IO_H

#include "minitensor/Tensor.h"

#include <iosfwd>
#include <string>

namespace minitensor {

// Phase 6: Print tensors as nested row-major brackets, e.g. [[1, 2], [3, 4]].
// Scalars print as a bare value and empty tensors as [].
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

std::string to_string(const Tensor& tensor);

}

#endif
