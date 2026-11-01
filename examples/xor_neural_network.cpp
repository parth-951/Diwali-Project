// Application: a two-layer neural network that learns XOR.
//
// XOR is the classic problem a single linear layer cannot solve. The network
// is 2 inputs -> 4 sigmoid hidden units -> 1 sigmoid output, trained with full
// batch gradient descent on binary cross-entropy. Forward and backward passes
// are written with matmul, transpose, broadcasting and element-wise ops.
// Exits with status 1 if any of the four XOR cases is predicted wrong.

#include "minitensor/minitensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace minitensor;

namespace {

// See linear_regression.cpp: raw mt19937 output is identical everywhere.
double uniform(std::mt19937& generator, double low, double high)
{
    return low + (high - low) * (static_cast<double>(generator()) / 4294967296.0);
}

Tensor scalar(double value)
{
    return Tensor(std::vector<std::size_t>{}, value);
}

Tensor random_tensor(const std::vector<std::size_t>& dimensions, std::mt19937& generator)
{
    Tensor tensor(dimensions);

    for (std::size_t i = 0; i < tensor.size(); ++i)
    {
        tensor.flat_at(i) = uniform(generator, -1.0, 1.0);
    }

    return tensor;
}

Tensor sigmoid(Tensor tensor)
{
    double* data = tensor.data();

    for (std::size_t i = 0; i < tensor.size(); ++i)
    {
        data[i] = 1.0 / (1.0 + std::exp(-data[i]));
    }

    return tensor;
}

// Column sums of an (m x k) matrix as a length-k vector, computed as
// ones(1 x m) @ matrix. Used for the bias gradients.
Tensor column_sums(const Tensor& matrix)
{
    const std::size_t rows = matrix.shape().dimensions()[0];

    Tensor sums = matmul(Tensor({1, rows}, 1.0), matrix);
    sums.reshape({sums.size()});

    return sums;
}

double binary_cross_entropy(const Tensor& predicted, const Tensor& target)
{
    double total = 0.0;

    for (std::size_t i = 0; i < predicted.size(); ++i)
    {
        const double p = std::min(std::max(predicted.data()[i], 1e-12), 1.0 - 1e-12);
        const double t = target.data()[i];
        total -= t * std::log(p) + (1.0 - t) * std::log(1.0 - p);
    }

    return total / static_cast<double>(predicted.size());
}

}

int main()
{
    constexpr std::size_t samples = 4;
    constexpr std::size_t hidden_units = 4;
    constexpr int epochs = 10000;
    constexpr double learning_rate = 2.0;

    // The XOR truth table
    Tensor x({samples, 2});
    Tensor y({samples, 1});
    const double table[samples][3] = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};

    for (std::size_t i = 0; i < samples; ++i)
    {
        x({i, 0}) = table[i][0];
        x({i, 1}) = table[i][1];
        y({i, 0}) = table[i][2];
    }

    std::mt19937 generator(7);
    Tensor w1 = random_tensor({2, hidden_units}, generator);
    Tensor b1({hidden_units}, 0.0);
    Tensor w2 = random_tensor({hidden_units, 1}, generator);
    Tensor b2({1}, 0.0);

    const Tensor one = scalar(1.0);
    const Tensor step = scalar(learning_rate / static_cast<double>(samples));

    std::cout << "Training a 2-" << hidden_units << "-1 network on XOR\n";
    std::cout << std::fixed << std::setprecision(6);

    Tensor output({samples, 1});

    for (int epoch = 1; epoch <= epochs; ++epoch)
    {
        // Forward pass; the bias vectors broadcast across the batch
        const Tensor hidden = sigmoid(add_broadcast(matmul(x, w1), b1));
        output = sigmoid(add_broadcast(matmul(hidden, w2), b2));

        if (epoch == 1 || epoch % 2000 == 0)
        {
            std::cout << "  epoch " << std::setw(5) << epoch
                      << "  loss " << binary_cross_entropy(output, y) << '\n';
        }

        // Backward pass. With a sigmoid output and cross-entropy loss the
        // output gradient simplifies to (prediction - target).
        const Tensor d_output = output - y;
        const Tensor d_hidden = matmul(d_output, transpose(w2))
                              * hidden
                              * subtract_broadcast(one, hidden);

        w2 = w2 - multiply_broadcast(matmul(transpose(hidden), d_output), step);
        b2 = b2 - multiply_broadcast(column_sums(d_output), step);
        w1 = w1 - multiply_broadcast(matmul(transpose(x), d_hidden), step);
        b1 = b1 - multiply_broadcast(column_sums(d_hidden), step);
    }

    std::cout << "\n  x1  x2  target  prediction\n";
    bool all_correct = true;

    for (std::size_t i = 0; i < samples; ++i)
    {
        const double prediction = output({i, 0});
        const bool correct = (prediction > 0.5) == (y({i, 0}) > 0.5);

        std::cout << "  " << std::setprecision(0) << std::setw(2) << x({i, 0})
                  << "  " << std::setw(2) << x({i, 1})
                  << "  " << std::setw(6) << y({i, 0})
                  << "  " << std::setprecision(4) << std::setw(10) << prediction
                  << (correct ? "" : "  <-- wrong") << '\n';

        if (!correct)
        {
            all_correct = false;
        }
    }

    std::cout << (all_correct ? "\nXOR learned.\n" : "\nXOR not learned!\n");

    return all_correct ? 0 : 1;
}
