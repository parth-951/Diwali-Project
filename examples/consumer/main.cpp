#include <minitensor/minitensor.h>

#include <iostream>

int main()
{
    using namespace minitensor;

    Tensor a({2, 2}, 1.0);
    Tensor b({2, 2}, 2.0);

    const Tensor product = matmul(a, b);

    std::cout << "Using installed MiniTensor " << version() << '\n';
    std::cout << "[[1, 1], [1, 1]] @ [[2, 2], [2, 2]] = " << product << '\n';

    return to_string(product) == "[[4, 4], [4, 4]]" ? 0 : 1;
}
