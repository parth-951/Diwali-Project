// A tour of the MiniTensor API. This is the program shown in the README.

#include "minitensor/minitensor.h"

#include <iostream>

int main()
{
    using namespace minitensor;

    std::cout << "MiniTensor " << version()
              << " (OpenMP " << (openmp_enabled() ? "on" : "off") << ")\n\n";

    // Create a 2x3 tensor and fill it with 1..6
    Tensor a({2, 3});

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        a.flat_at(i) = static_cast<double>(i + 1);
    }

    Tensor b({2, 3}, 10.0);

    std::cout << "a              = " << a << '\n';
    std::cout << "b              = " << b << '\n';

    // Element access with multidimensional indices
    a({1, 2}) = 60.0;
    std::cout << "a(1, 2) = 60   -> " << a << '\n';
    a({1, 2}) = 6.0;

    // Element-wise arithmetic
    std::cout << "a + b          = " << (a + b) << '\n';
    std::cout << "b - a          = " << (b - a) << '\n';
    std::cout << "a * a          = " << (a * a) << '\n';

    // Broadcasting: add a row vector to every row
    Tensor row({3});
    row({0}) = 100.0;
    row({1}) = 200.0;
    row({2}) = 300.0;
    std::cout << "a + row        = " << add_broadcast(a, row) << '\n';

    // Matrix operations
    std::cout << "transpose(a)   = " << transpose(a) << '\n';
    std::cout << "a @ a^T        = " << matmul(a, transpose(a)) << '\n';

    // Reshape keeps the data and changes only the view of it
    a.reshape({3, 2});
    std::cout << "a as 3x2       = " << a << '\n';

    // Reductions
    std::cout << "sum / mean     = " << sum(a) << " / " << mean(a) << '\n';
    std::cout << "min / max      = " << min(a) << " / " << max(a) << '\n';

    return 0;
}
