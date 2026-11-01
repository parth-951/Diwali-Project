#ifndef MINITENSOR_TEST_UTILS_H
#define MINITENSOR_TEST_UTILS_H

// Phase 6: A tiny dependency-free test harness. Unlike assert(), checks keep
// running after a failure, report file:line, and work in Release builds.

#include "minitensor/Tensor.h"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>

namespace minitensor_test {

inline int& check_count()
{
    static int count = 0;
    return count;
}

inline int& failure_count()
{
    static int count = 0;
    return count;
}

inline void report(bool passed, const char* expression, const char* file, int line)
{
    ++check_count();

    if (!passed)
    {
        ++failure_count();
        std::cerr << "  FAILED: " << expression
                  << " (" << file << ':' << line << ")\n";
    }
}

inline bool nearly_equal(double a, double b, double tolerance = 1e-9)
{
    return std::abs(a - b) <= tolerance * (1.0 + std::abs(a) + std::abs(b));
}

inline bool tensors_close(
    const minitensor::Tensor& a,
    const minitensor::Tensor& b,
    double tolerance = 1e-9)
{
    if (a.shape().dimensions() != b.shape().dimensions())
    {
        return false;
    }

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (!nearly_equal(a.data()[i], b.data()[i], tolerance))
        {
            return false;
        }
    }

    return true;
}

// Run one test function; an unexpected exception counts as a failure
template <typename TestFunction>
void run_test(const char* name, TestFunction test)
{
    const int failures_before = failure_count();

    try
    {
        test();
    }
    catch (const std::exception& error)
    {
        report(false, error.what(), name, 0);
    }
    catch (...)
    {
        report(false, "unknown exception", name, 0);
    }

    std::cout << (failure_count() == failures_before ? "[PASS] " : "[FAIL] ")
              << name << '\n';
}

// Print the summary and return the process exit code
inline int summary(const char* suite)
{
    std::cout << suite << ": " << check_count() << " checks, "
              << failure_count() << " failures\n";

    return failure_count() == 0 ? 0 : 1;
}

}

#define MT_CHECK(condition) \
    minitensor_test::report(static_cast<bool>(condition), #condition, __FILE__, __LINE__)

#define MT_CHECK_THROWS(expression, ExceptionType)                          \
    do                                                                      \
    {                                                                       \
        bool mt_threw_expected = false;                                     \
        try                                                                 \
        {                                                                   \
            static_cast<void>(expression);                                  \
        }                                                                   \
        catch (const ExceptionType&)                                        \
        {                                                                   \
            mt_threw_expected = true;                                       \
        }                                                                   \
        catch (...)                                                         \
        {                                                                   \
        }                                                                   \
        minitensor_test::report(mt_threw_expected,                          \
            #expression " throws " #ExceptionType, __FILE__, __LINE__);     \
    } while (false)

#endif
