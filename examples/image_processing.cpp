// Application: a small image-processing pipeline on an RGB image tensor.
//
//   1. Grayscale: reshape (H x W x 3) to (H*W x 3) and matmul with the
//      ITU-R BT.601 luma weights (3 x 1), then reshape back to (H x W).
//   2. Contrast stretch to [0, 1] with min/max reductions and broadcasting.
//   3. Vignette: multiply by an (H x W) mask built as an outer product.
//   4. Render the result as ASCII art.
//
// Exits with status 1 if any pipeline stage produces an unexpected value.

#include "minitensor/minitensor.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

using namespace minitensor;

namespace {

Tensor scalar(double value)
{
    return Tensor(std::vector<std::size_t>{}, value);
}

// Synthetic test image: red rises left to right, green top to bottom and a
// bright blue disc sits in the middle. Channel values are in [0, 255].
Tensor make_image(std::size_t height, std::size_t width)
{
    Tensor image({height, width, 3});

    const double center_y = static_cast<double>(height - 1) / 2.0;
    const double center_x = static_cast<double>(width - 1) / 2.0;

    for (std::size_t row = 0; row < height; ++row)
    {
        for (std::size_t col = 0; col < width; ++col)
        {
            const double dy = (static_cast<double>(row) - center_y) / center_y;
            const double dx = (static_cast<double>(col) - center_x) / center_x;
            const bool in_disc = dx * dx + dy * dy < 0.35;

            image({row, col, 0}) = 255.0 * static_cast<double>(col) / static_cast<double>(width - 1);
            image({row, col, 1}) = 255.0 * static_cast<double>(row) / static_cast<double>(height - 1);
            image({row, col, 2}) = in_disc ? 255.0 : 0.0;
        }
    }

    return image;
}

bool near(double a, double b)
{
    return std::abs(a - b) < 1e-9;
}

}

int main()
{
    constexpr std::size_t height = 16;
    constexpr std::size_t width = 48;

    const Tensor image = make_image(height, width);
    bool ok = true;

    // 1. Grayscale via reshape + matmul
    Tensor pixels = image;
    pixels.reshape({height * width, 3});

    Tensor luma({3, 1});
    luma({0, 0}) = 0.299;
    luma({1, 0}) = 0.587;
    luma({2, 0}) = 0.114;

    Tensor gray = matmul(pixels, luma);
    gray.reshape({height, width});

    // The top-left pixel is pure black, the bottom-right is red + green
    ok = ok && near(gray({0, 0}), 0.0);
    ok = ok && near(gray({height - 1, width - 1}), 255.0 * (0.299 + 0.587));

    // 2. Contrast stretch: (gray - min) / (max - min)
    const double low = min(gray);
    const double high = max(gray);
    const Tensor normalized = multiply_broadcast(
        subtract_broadcast(gray, scalar(low)),
        scalar(1.0 / (high - low)));

    ok = ok && near(min(normalized), 0.0) && near(max(normalized), 1.0);

    // 3. Vignette mask as an outer product: (H x 1) @ (1 x W)
    Tensor row_profile({height, 1});
    Tensor col_profile({1, width});

    for (std::size_t row = 0; row < height; ++row)
    {
        const double t = 2.0 * static_cast<double>(row) / static_cast<double>(height - 1) - 1.0;
        row_profile({row, 0}) = 1.0 - 0.4 * t * t;
    }

    for (std::size_t col = 0; col < width; ++col)
    {
        const double t = 2.0 * static_cast<double>(col) / static_cast<double>(width - 1) - 1.0;
        col_profile({0, col}) = 1.0 - 0.4 * t * t;
    }

    const Tensor mask = matmul(row_profile, col_profile);
    const Tensor result = normalized * mask;

    ok = ok && min(result) >= 0.0 && max(result) <= 1.0;
    // Corners are darkened by 0.6 along each axis
    ok = ok && normalized({0, width - 1}) > 0.0;
    ok = ok && near(result({0, width - 1}), normalized({0, width - 1}) * 0.6 * 0.6);

    // 4. Report and render
    std::cout << "Image " << height << "x" << width << "x3"
              << "  gray range [" << low << ", " << high << "]"
              << "  mean brightness " << mean(result) << "\n\n";

    const std::string ramp = " .:-=+*#%@";

    for (std::size_t row = 0; row < height; ++row)
    {
        std::string line;

        for (std::size_t col = 0; col < width; ++col)
        {
            const double value = result({row, col});
            const std::size_t level = static_cast<std::size_t>(value * (ramp.size() - 1) + 0.5);
            line += ramp[level];
        }

        std::cout << "  " << line << '\n';
    }

    std::cout << (ok ? "\nAll pipeline checks passed.\n" : "\nPipeline check failed!\n");

    return ok ? 0 : 1;
}
