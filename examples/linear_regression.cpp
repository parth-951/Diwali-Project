// Application: linear regression trained with batch gradient descent.
//
// Learns y = 3*x1 - 2*x2 + 5 from 1000 noisy samples. All of the math
// (predictions, errors, gradients and updates) uses MiniTensor operations.
// Exits with status 1 if the learned weights are not close to the true ones.

#include "minitensor/minitensor.h"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace minitensor;

namespace {

// The output of std::mt19937 is fixed by the C++ standard, unlike the <random>
// distributions, so this gives the same data on every compiler and platform.
double uniform(std::mt19937& generator, double low, double high)
{
    return low + (high - low) * (static_cast<double>(generator()) / 4294967296.0);
}

Tensor scalar(double value)
{
    return Tensor(std::vector<std::size_t>{}, value);
}

}

int main()
{
    constexpr std::size_t samples = 1000;
    constexpr std::size_t features = 3; // x1, x2 and a bias column of ones
    constexpr int epochs = 500;
    constexpr double learning_rate = 0.1;
    constexpr double tolerance = 0.05;

    const double true_weights[features] = {3.0, -2.0, 5.0};
    const char* names[features] = {"x1", "x2", "bias"};

    // Build the dataset: X is (samples x 3), y is (samples x 1)
    std::mt19937 generator(42);
    Tensor x({samples, features});
    Tensor y({samples, 1});

    for (std::size_t i = 0; i < samples; ++i)
    {
        const double x1 = uniform(generator, -1.0, 1.0);
        const double x2 = uniform(generator, -1.0, 1.0);
        const double noise = uniform(generator, -0.1, 0.1);

        x({i, 0}) = x1;
        x({i, 1}) = x2;
        x({i, 2}) = 1.0;
        y({i, 0}) = true_weights[0] * x1 + true_weights[1] * x2 + true_weights[2] + noise;
    }

    // Gradient of the mean squared error: (2 / n) * X^T (Xw - y)
    Tensor weights({features, 1}, 0.0);
    const Tensor x_transposed = transpose(x);
    const Tensor step = scalar(2.0 * learning_rate / static_cast<double>(samples));

    std::cout << "Training linear regression on " << samples << " samples\n";
    std::cout << std::fixed << std::setprecision(6);

    for (int epoch = 1; epoch <= epochs; ++epoch)
    {
        const Tensor error = matmul(x, weights) - y;
        weights = weights - multiply_broadcast(matmul(x_transposed, error), step);

        if (epoch == 1 || epoch % 100 == 0)
        {
            std::cout << "  epoch " << std::setw(3) << epoch
                      << "  mse " << mean(error * error) << '\n';
        }
    }

    std::cout << "\nweight   learned     true\n";
    bool converged = true;

    for (std::size_t i = 0; i < features; ++i)
    {
        const double learned = weights({i, 0});
        std::cout << "  " << std::left << std::setw(5) << names[i] << std::right
                  << std::setw(10) << learned << std::setw(10) << true_weights[i] << '\n';

        if (std::abs(learned - true_weights[i]) > tolerance)
        {
            converged = false;
        }
    }

    std::cout << (converged ? "\nConverged.\n" : "\nDid not converge!\n");

    return converged ? 0 : 1;
}
