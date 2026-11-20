// MiniTensor driver for the Diwali 2026 Mini Tensor Library Common Test
// Specification v1.0 (17 November 2026).
//
//   common_test_driver functional <results.csv> [team_id] [--expect-fail=TCxx ...]
//       Runs TC01-TC50 and writes one row per test in the spec's CSV columns.
//   common_test_driver bench <test_id> <config> <run> <raw.csv>
//       Runs one benchmark configuration (5 warm-ups, 30 timed runs) and
//       appends one row to raw.csv. Threads come from OMP_NUM_THREADS.
//   common_test_driver tc50-child
//       The impossible allocation of TC50, run by `functional` in a child process.
//   common_test_driver indices <file>
//       Writes the PB05 index list for the NumPy reference.
//
// All tests use float32 (minitensor::FloatTensor). The spec's function calls
// are mapped to MiniTensor's API; the mathematical operations are unchanged.

#include "common_spec.h"
#include "minitensor/minitensor.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

using namespace minitensor;
using namespace common_spec;

// ---------------------------------------------------------------- leak check
//
// Outside sanitizer builds, every operator new/delete in this process is
// counted, so TC45-TC49 can check that the number of live allocations returns
// to where it started. AddressSanitizer replaces these operators itself, so
// sanitizer builds rely on it instead.

#if defined(__SANITIZE_ADDRESS__)
#define MT_SANITIZER_BUILD 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define MT_SANITIZER_BUILD 1
#endif
#endif

namespace {
std::atomic<long long> live_allocations{0};
}

#ifndef MT_SANITIZER_BUILD
#define MT_COUNT_ALLOCATIONS 1

namespace {

void* counted_alloc(std::size_t size)
{
    void* p = std::malloc(size == 0 ? 1 : size);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    ++live_allocations;
    return p;
}

void counted_free(void* p) noexcept
{
    if (p != nullptr) {
        --live_allocations;
        std::free(p);
    }
}

void* counted_aligned_alloc(std::size_t size, std::align_val_t alignment)
{
    const std::size_t align = static_cast<std::size_t>(alignment);
#ifdef _WIN32
    void* p = _aligned_malloc(size == 0 ? 1 : size, align);
#else
    const std::size_t rounded = ((size == 0 ? 1 : size) + align - 1) / align * align;
    void* p = std::aligned_alloc(align, rounded);
#endif
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    ++live_allocations;
    return p;
}

void counted_aligned_free(void* p) noexcept
{
    if (p != nullptr) {
        --live_allocations;
#ifdef _WIN32
        _aligned_free(p);
#else
        std::free(p);
#endif
    }
}

} // namespace

void* operator new(std::size_t size) { return counted_alloc(size); }
void* operator new[](std::size_t size) { return counted_alloc(size); }
void operator delete(void* p) noexcept { counted_free(p); }
void operator delete[](void* p) noexcept { counted_free(p); }
void operator delete(void* p, std::size_t) noexcept { counted_free(p); }
void operator delete[](void* p, std::size_t) noexcept { counted_free(p); }

void* operator new(std::size_t size, std::align_val_t a) { return counted_aligned_alloc(size, a); }
void* operator new[](std::size_t size, std::align_val_t a) { return counted_aligned_alloc(size, a); }
void operator delete(void* p, std::align_val_t) noexcept { counted_aligned_free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { counted_aligned_free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { counted_aligned_free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { counted_aligned_free(p); }
#endif

namespace {

std::string checker_name()
{
#ifdef MT_SANITIZER_BUILD
    return "AddressSanitizer build";
#else
    return "allocation counter (no sanitizer in this build)";
#endif
}

// ------------------------------------------------------------ test helpers

struct Outcome {
    std::string status = "PASS";
    std::string actual;
    std::string max_abs_error;
    std::string notes;
};

struct Row {
    std::string id;
    std::string operation;
    std::string expected;
    Outcome outcome;
};

std::string shape_text(const std::vector<std::size_t>& dims)
{
    std::string text = "{";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        text += (i == 0 ? "" : ",") + std::to_string(dims[i]);
    }
    return text + "}";
}

std::string shape_text(const FloatTensor& t)
{
    return shape_text(t.shape().dimensions());
}

std::string values_text(const FloatTensor& t)
{
    std::string text = "[";
    for (std::size_t i = 0; i < t.size(); ++i) {
        text += (i == 0 ? "" : ",") + format_number(t.data()[i], 9);
    }
    return text + "]";
}

FloatTensor make(const std::vector<std::size_t>& dims, const std::vector<float>& values)
{
    FloatTensor t(dims);
    if (values.size() != t.size()) {
        throw std::logic_error("test setup: value count does not match shape");
    }
    std::memcpy(t.data(), values.data(), values.size() * sizeof(float));
    return t;
}

std::vector<float> iota_values(std::size_t count, float first)
{
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = first + static_cast<float>(i);
    }
    return values;
}

// Shape and values must match exactly
bool equals(const FloatTensor& t, const std::vector<std::size_t>& dims,
            const std::vector<float>& values)
{
    if (t.shape().dimensions() != dims || t.size() != values.size()) {
        return false;
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (t.data()[i] != values[i]) {
            return false;
        }
    }
    return true;
}

// Shape exact, values within the spec's float32 tolerance
bool close(const FloatTensor& t, const std::vector<std::size_t>& dims,
           const std::vector<double>& values, double& max_error)
{
    max_error = 0.0;
    if (t.shape().dimensions() != dims || t.size() != values.size()) {
        return false;
    }
    bool ok = true;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double error = std::fabs(static_cast<double>(t.data()[i]) - values[i]);
        max_error = std::max(max_error, error);
        ok = ok && within_tolerance(t.data()[i], values[i]);
    }
    return ok;
}

Outcome numeric_result(const FloatTensor& t, const std::vector<std::size_t>& dims,
                       const std::vector<double>& values)
{
    Outcome o;
    double error = 0.0;
    const bool ok = close(t, dims, values, error);
    o.status = ok ? "PASS" : "FAIL";
    o.actual = "shape " + shape_text(t) + "; values " + values_text(t);
    o.max_abs_error = format_number(error, 6);
    return o;
}

// Runs `operation` and expects a library exception
template <typename Operation>
Outcome expect_error(Operation operation, const std::string& extra_check_name = "",
                     const std::function<bool()>& extra_check = nullptr)
{
    Outcome o;
    try {
        operation();
        o.status = "FAIL";
        o.actual = "no error was reported";
        return o;
    } catch (const std::out_of_range& e) {
        o.actual = std::string("std::out_of_range: ") + e.what();
    } catch (const std::invalid_argument& e) {
        o.actual = std::string("std::invalid_argument: ") + e.what();
    } catch (const std::overflow_error& e) {
        o.actual = std::string("std::overflow_error: ") + e.what();
    } catch (const std::bad_alloc& e) {
        o.actual = std::string("std::bad_alloc: ") + e.what();
    } catch (const std::exception& e) {
        o.actual = std::string(typeid(e).name()) + ": " + e.what();
    }
    o.status = "PASS";
    if (extra_check && !extra_check()) {
        o.status = "FAIL";
        o.notes = extra_check_name + " check failed";
    } else if (extra_check) {
        o.notes = extra_check_name + " verified";
    }
    return o;
}

// ------------------------------------------------------------ TC50 child

int tc50_child()
{
    const std::size_t huge = std::size_t{1} << 31;
    try {
        FloatTensor t({huge, huge});
        std::cout << "CONSTRUCTED size=" << t.size() << '\n';
        return 1;
    } catch (const std::overflow_error& e) {
        std::cout << "std::overflow_error: " << e.what() << '\n';
    } catch (const std::bad_alloc& e) {
        std::cout << "std::bad_alloc: " << e.what() << '\n';
    } catch (const std::exception& e) {
        std::cout << typeid(e).name() << ": " << e.what() << '\n';
    }
    return 0;
}

Outcome run_tc50(const std::string& self, const std::string& scratch)
{
    Outcome o;
    const std::string output = scratch + ".tc50.txt";
    std::string command = "\"" + self + "\" tc50-child > \"" + output + "\"";
#ifdef _WIN32
    // cmd.exe strips one pair of outer quotes
    command = "\"" + command + "\"";
#endif
    const int code = std::system(command.c_str());

    std::ifstream in(output);
    std::string line;
    std::getline(in, line);
    in.close();
    std::remove(output.c_str());

    o.actual = "child exit code " + std::to_string(code) + "; child reported: " + line;
    const bool reported_error = line.find("error") != std::string::npos
        || line.find("bad_alloc") != std::string::npos;
    o.status = (code == 0 && reported_error) ? "PASS" : "FAIL";
    o.notes = "Ran in a child process. Requested {2^31,2^31} = 2^62 float32 elements (2^64 bytes); "
              "checked size arithmetic rejects it before any allocation is attempted";
    return o;
}

// ------------------------------------------------------------ functional

std::vector<Row> functional_tests(const std::string& self, const std::string& scratch)
{
    std::vector<Row> rows;
    const auto add_row = [&](const std::string& id, const std::string& op,
                             const std::string& expected, const std::function<Outcome()>& body) {
        Row row{id, op, expected, {}};
        try {
            row.outcome = body();
        } catch (const std::exception& e) {
            row.outcome.status = "FAIL";
            row.outcome.actual = std::string("unexpected exception: ") + e.what();
        }
        rows.push_back(row);
        std::cout << id << ' ' << row.outcome.status << '\n';
    };

    // ---- A. Tensor creation, shape and storage
    add_row("TC01", "create", "Rank 0; numel=1; value=5", [] {
        FloatTensor t(std::vector<std::size_t>{}, 5.0f);
        Outcome o;
        o.actual = "rank " + std::to_string(t.rank()) + "; numel " + std::to_string(t.size())
            + "; value " + format_number(t.at(std::vector<std::size_t>{}), 9);
        o.status = (t.rank() == 0 && t.size() == 1 && t.at(std::vector<std::size_t>{}) == 5.0f)
            ? "PASS" : "FAIL";
        o.notes = "Rank-0 tensor: FloatTensor(std::vector<std::size_t>{}, 5.0f)";
        return o;
    });

    const auto creation = [&](const std::string& id, const std::vector<std::size_t>& dims,
                              const std::vector<float>& values, const std::string& expected) {
        add_row(id, "create", expected, [=] {
            const FloatTensor t = make(dims, values);
            Outcome o;
            o.actual = "shape " + shape_text(t) + "; numel " + std::to_string(t.size())
                + "; values " + values_text(t);
            o.status = equals(t, dims, values) ? "PASS" : "FAIL";
            o.notes = "MiniTensor has no constructor taking data; values were written to the "
                      "contiguous buffer after construction";
            return o;
        });
    };

    creation("TC02", {5}, {1, 2, 3, 4, 5}, "Shape {5}; numel=5; values unchanged");
    creation("TC03", {2, 3}, {1, 2, 3, 4, 5, 6}, "Shape {2,3}; numel=6; rows [[1,2,3],[4,5,6]]");
    creation("TC04", {2, 2, 3}, iota_values(12, 0), "Shape {2,2,3}; numel=12; flattened order unchanged");
    creation("TC05", {2, 3, 2, 2}, iota_values(24, 0), "Shape {2,3,2,2}; numel=24; flattened order unchanged");

    add_row("TC06", "create", "All six values are 0", [] {
        const FloatTensor t({2, 3});
        Outcome o;
        o.actual = "shape " + shape_text(t) + "; values " + values_text(t);
        o.status = equals(t, {2, 3}, std::vector<float>(6, 0.0f)) ? "PASS" : "FAIL";
        o.notes = "The API promises initialization: the constructor's initial_value defaults to 0";
        return o;
    });

    add_row("TC07", "memory", "address(i+1)-address(i)=sizeof(float) for all adjacent pairs", [] {
        FloatTensor t = make({2, 3}, {0, 1, 2, 3, 4, 5});
        Outcome o;
        std::string steps;
        bool ok = true;
        for (std::size_t i = 0; i + 1 < t.size(); ++i) {
            const auto here = reinterpret_cast<std::uintptr_t>(&t.flat_at(i));
            const auto next = reinterpret_cast<std::uintptr_t>(&t.flat_at(i + 1));
            steps += (i == 0 ? "" : ",") + std::to_string(next - here);
            ok = ok && next - here == sizeof(float);
        }
        ok = ok && &t.flat_at(0) == t.data();
        o.actual = "address steps (bytes) [" + steps + "]; sizeof(float)=" + std::to_string(sizeof(float))
            + "; base address 64-byte aligned: "
            + (reinterpret_cast<std::uintptr_t>(t.data()) % 64 == 0 ? "yes" : "no");
        o.status = ok ? "PASS" : "FAIL";
        return o;
    });

    add_row("TC08", "create", "numel=1,000,000; sum=2,000,000", [] {
        const FloatTensor t({1000, 1000}, 2.0f);
        const double total = sum(t);
        Outcome o;
        o.actual = "numel " + std::to_string(t.size()) + "; sum " + format_number(total, 12);
        o.max_abs_error = format_number(std::fabs(total - 2000000.0), 6);
        o.status = (t.size() == 1000000 && within_tolerance(total, 2000000.0)) ? "PASS" : "FAIL";
        o.notes = "minitensor::sum accumulates in double";
        return o;
    });

    // ---- B. Indexing and element modification
    const auto table_t = [] { return make({2, 3}, {1, 2, 3, 4, 5, 6}); };

    const auto read = [&](const std::string& id, std::size_t r, std::size_t c, float expected) {
        add_row(id, "index", format_number(expected), [=] {
            const FloatTensor t = table_t();
            const float value = t(std::vector<std::size_t>{r, c});
            Outcome o;
            o.actual = format_number(value, 9);
            o.status = value == expected ? "PASS" : "FAIL";
            return o;
        });
    };

    read("TC09", 0, 0, 1);
    read("TC10", 1, 2, 6);
    read("TC11", 1, 1, 5);

    add_row("TC12", "index", "Read-back value is 20; all other elements remain [1,3,4,5,6] in their positions", [=] {
        FloatTensor t = table_t();
        t(std::vector<std::size_t>{0, 1}) = 20.0f;
        Outcome o;
        o.actual = "T(0,1)=" + format_number(t(std::vector<std::size_t>{0, 1}), 9) + "; all values " + values_text(t);
        o.status = equals(t, {2, 3}, {1, 20, 3, 4, 5, 6}) ? "PASS" : "FAIL";
        return o;
    });

    add_row("TC13", "index", "Documented error; no crash or memory corruption", [=] {
        FloatTensor t = table_t();
        return expect_error([&] { (void)t(std::vector<std::size_t>{2, 0}); },
                            "Tensor unchanged", [&] { return equals(t, {2, 3}, {1, 2, 3, 4, 5, 6}); });
    });

    add_row("TC14", "index", "Documented error; no crash or memory corruption", [=] {
        FloatTensor t = table_t();
        return expect_error([&] { (void)t(std::vector<std::size_t>{0, 3}); },
                            "Tensor unchanged", [&] { return equals(t, {2, 3}, {1, 2, 3, 4, 5, 6}); });
    });

    add_row("TC15", "index", "Documented error; negative index must not access unrelated memory", [=] {
        FloatTensor t = table_t();
        const long long row = -1;
        Outcome o = expect_error([&] {
            (void)t(std::vector<std::size_t>{static_cast<std::size_t>(row), 0});
        }, "Tensor unchanged", [&] { return equals(t, {2, 3}, {1, 2, 3, 4, 5, 6}); });
        o.notes = "The API takes unsigned std::size_t indices, so a literal T(-1,0) does not compile. "
                  "A signed -1 converted at run time becomes SIZE_MAX and is rejected by the bounds check. "
                  + o.notes;
        return o;
    });

    add_row("TC16", "index", "Value 23; flattened offset = 1x12 + 2x4 + 3 = 23", [] {
        const FloatTensor t = make({2, 3, 4}, iota_values(24, 0));
        const float value = t(std::vector<std::size_t>{1, 2, 3});
        const auto offset = &t(std::vector<std::size_t>{1, 2, 3}) - t.data();
        Outcome o;
        o.actual = "value " + format_number(value, 9) + "; offset " + std::to_string(offset);
        o.status = (value == 23.0f && offset == 23) ? "PASS" : "FAIL";
        return o;
    });

    // ---- C. Addition and subtraction
    const auto A = [] { return make({2, 2}, {1, 2, 3, 4}); };
    const auto B = [] { return make({2, 2}, {5, 6, 7, 8}); };
    const auto S = [] { return make({2, 2}, {-2, 0.5f, 3, -4}); };
    const auto R = [] { return make({2, 2}, {1, -0.5f, -3, 2}); };
    const auto Z = [] { return FloatTensor({2, 2}, 0.0f); };

    add_row("TC17", "add", "Shape {2,2}; [[6,8],[10,12]]",
            [=] { return numeric_result(add(A(), B()), {2, 2}, {6, 8, 10, 12}); });
    add_row("TC18", "subtract", "Shape {2,2}; [[-4,-4],[-4,-4]]",
            [=] { return numeric_result(subtract(A(), B()), {2, 2}, {-4, -4, -4, -4}); });

    add_row("TC19", "add", "C equals A exactly", [=] {
        const FloatTensor c = add(A(), Z());
        Outcome o;
        o.actual = "shape " + shape_text(c) + "; values " + values_text(c);
        o.status = equals(c, {2, 2}, {1, 2, 3, 4}) ? "PASS" : "FAIL";
        o.max_abs_error = "0";
        return o;
    });

    add_row("TC20", "subtract", "C equals A exactly", [=] {
        const FloatTensor c = subtract(A(), Z());
        Outcome o;
        o.actual = "shape " + shape_text(c) + "; values " + values_text(c);
        o.status = equals(c, {2, 2}, {1, 2, 3, 4}) ? "PASS" : "FAIL";
        o.max_abs_error = "0";
        return o;
    });

    add_row("TC21", "add", "[[-1,0],[0,-2]]",
            [=] { return numeric_result(add(S(), R()), {2, 2}, {-1, 0, 0, -2}); });
    add_row("TC22", "subtract", "[[-3,1],[6,-6]]",
            [=] { return numeric_result(subtract(S(), R()), {2, 2}, {-3, 1, 6, -6}); });

    add_row("TC23", "add", "Documented error; no implicit broadcasting or truncation", [] {
        const FloatTensor a = make({2, 3}, iota_values(6, 1));
        const FloatTensor b = make({3, 2}, iota_values(6, 1));
        return expect_error([&] { (void)add(a, b); });
    });

    add_row("TC24", "subtract", "Documented error; no implicit reshape or truncation", [] {
        const FloatTensor a = make({2, 3}, iota_values(6, 1));
        const FloatTensor b = make({6}, iota_values(6, 1));
        return expect_error([&] { (void)subtract(a, b); });
    });

    // ---- D. Element-wise multiplication
    add_row("TC25", "elementwise_multiply", "[[5,12],[21,32]]",
            [=] { return numeric_result(multiply(A(), B()), {2, 2}, {5, 12, 21, 32}); });
    add_row("TC26", "elementwise_multiply", "[[0,0],[0,0]]",
            [=] { return numeric_result(multiply(A(), Z()), {2, 2}, {0, 0, 0, 0}); });

    add_row("TC27", "elementwise_multiply", "C equals A", [=] {
        const FloatTensor c = multiply(A(), FloatTensor({2, 2}, 1.0f));
        Outcome o;
        o.actual = "shape " + shape_text(c) + "; values " + values_text(c);
        o.status = equals(c, {2, 2}, {1, 2, 3, 4}) ? "PASS" : "FAIL";
        o.max_abs_error = "0";
        return o;
    });

    add_row("TC28", "elementwise_multiply", "[[-2,-0.25],[-9,-8]]",
            [=] { return numeric_result(multiply(S(), R()), {2, 2}, {-2, -0.25, -9, -8}); });

    add_row("TC29", "elementwise_multiply", "[[0.02,0.06],[0.12,0.20]] within stated tolerance", [] {
        const FloatTensor a = make({2, 2}, {0.1f, 0.2f, 0.3f, 0.4f});
        const FloatTensor b = make({2, 2}, {0.2f, 0.3f, 0.4f, 0.5f});
        return numeric_result(multiply(a, b), {2, 2}, {0.02, 0.06, 0.12, 0.20});
    });

    add_row("TC30", "elementwise_multiply", "Documented error; no implicit broadcasting", [] {
        const FloatTensor a = make({2, 3}, iota_values(6, 1));
        const FloatTensor b = make({3, 2}, iota_values(6, 1));
        return expect_error([&] { (void)multiply(a, b); });
    });

    // ---- E. Matrix multiplication
    add_row("TC31", "matmul", "Shape {2,2}; [[58,64],[139,154]]", [] {
        return numeric_result(matmul(make({2, 3}, {1, 2, 3, 4, 5, 6}),
                                     make({3, 2}, {7, 8, 9, 10, 11, 12})),
                              {2, 2}, {58, 64, 139, 154});
    });

    add_row("TC32", "matmul", "[[30,24,18],[84,69,54],[138,114,90]]", [] {
        return numeric_result(matmul(make({3, 3}, iota_values(9, 1)),
                                     make({3, 3}, {9, 8, 7, 6, 5, 4, 3, 2, 1})),
                              {3, 3}, {30, 24, 18, 84, 69, 54, 138, 114, 90});
    });

    add_row("TC33", "matmul", "C equals A", [] {
        return numeric_result(matmul(make({2, 3}, {1, 2, 3, 4, 5, 6}),
                                     make({3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1})),
                              {2, 3}, {1, 2, 3, 4, 5, 6});
    });

    add_row("TC34", "matmul", "Shape {2,3}; all six elements zero", [] {
        return numeric_result(matmul(make({2, 2}, {1, 2, 3, 4}), FloatTensor({2, 3}, 0.0f)),
                              {2, 3}, {0, 0, 0, 0, 0, 0});
    });

    add_row("TC35", "matmul", "Shape {2,4}; [[190,200,210,220],[440,475,510,545]]", [] {
        // Compared with the spec's expected values unchanged, as the spec requires
        Outcome o = numeric_result(matmul(make({2, 5}, iota_values(10, 1)), make({5, 4}, iota_values(20, 1))),
                                   {2, 4}, {190, 200, 210, 220, 440, 475, 510, 545});
        o.notes = "The specification's expected values do not match its stated inputs: A{2,5}=1..10 times "
                  "B{5,4}=1..20 (row-major) is [[175,190,205,220],[400,440,480,520]] (e.g. C(0,0) = "
                  "1*1+2*5+3*9+4*13+5*17 = 175), confirmed with NumPy float32. MiniTensor's output equals "
                  "that product exactly.";
        return o;
    });

    add_row("TC36", "matmul", "Shape {1,1}; value 7.0", [] {
        return numeric_result(matmul(make({1, 1}, {3.5f}), make({1, 1}, {2.0f})), {1, 1}, {7.0});
    });

    add_row("TC37", "matmul", "Documented error; no output presented as valid", [] {
        const FloatTensor a({2, 3}, 1.0f);
        const FloatTensor b({4, 2}, 1.0f);
        return expect_error([&] { (void)matmul(a, b); });
    });

    add_row("TC38", "matmul", "[[11,14,17,20],[23,30,37,44],[35,46,57,68]]", [] {
        return numeric_result(matmul(make({3, 2}, {1, 2, 3, 4, 5, 6}), make({2, 4}, iota_values(8, 1))),
                              {3, 4}, {11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68});
    });

    // ---- F. Reshape (MiniTensor reshapes in place: only the shape metadata
    //      changes, and the element buffer is neither copied nor reallocated)
    const std::string reshape_note = "Reshape is in place: shape metadata changes, the element buffer "
                                     "is not copied (same data pointer before and after)";
    const auto V = [] { return make({12}, iota_values(12, 0)); };

    const auto reshape_case = [&](const std::string& id, const std::vector<std::size_t>& start,
                                  const std::vector<std::size_t>& target, const std::string& expected) {
        add_row(id, "reshape", expected, [=] {
            FloatTensor t = V();
            if (!start.empty()) {
                t.reshape(start);
            }
            const float* before = t.data();
            t.reshape(target);
            Outcome o;
            o.actual = "shape " + shape_text(t) + "; numel " + std::to_string(t.size())
                + "; values " + values_text(t) + "; data pointer unchanged: "
                + (t.data() == before ? "yes" : "no");
            o.status = (equals(t, target, iota_values(12, 0)) && t.data() == before) ? "PASS" : "FAIL";
            o.notes = reshape_note;
            return o;
        });
    };

    reshape_case("TC39", {}, {3, 4}, "Shape {3,4}; rows [[0,1,2,3],[4,5,6,7],[8,9,10,11]]");
    reshape_case("TC40", {3, 4}, {2, 2, 3}, "Shape {2,2,3}; flattened values remain 0..11 in same order");
    reshape_case("TC41", {2, 2, 3}, {12}, "Shape {12}; values [0,1,...,11]");

    add_row("TC42", "reshape", "Shape {3,2}; [[1,2],[3,4],[5,6]]", [&] {
        FloatTensor t = make({2, 3}, {1, 2, 3, 4, 5, 6});
        t.reshape({3, 2});
        Outcome o;
        o.actual = "shape " + shape_text(t) + "; values " + values_text(t);
        o.status = equals(t, {3, 2}, {1, 2, 3, 4, 5, 6}) ? "PASS" : "FAIL";
        o.notes = reshape_note;
        return o;
    });

    add_row("TC43", "reshape", "Documented error; original tensor remains valid and unchanged", [] {
        FloatTensor t = make({2, 3}, {1, 2, 3, 4, 5, 6});
        return expect_error([&] { t.reshape({4, 4}); }, "Original shape {2,3} and values",
                            [&] { return equals(t, {2, 3}, {1, 2, 3, 4, 5, 6}); });
    });

    reshape_case("TC44", {}, {1, 12, 1}, "Shape {1,12,1}; numel=12; flattened order unchanged");

    add_row("TC45", "reshape", "After every round trip, shape {12} and values 0..11; no corruption/leak", [&] {
        const long long live_before = live_allocations.load();
        bool ok = true;
        {
            FloatTensor t = V();
            const float* buffer = t.data();
            for (int round = 0; round < 100; ++round) {
                t.reshape({3, 4});
                t.reshape({2, 2, 3});
                t.reshape({12});
                ok = ok && equals(t, {12}, iota_values(12, 0)) && t.data() == buffer;
            }
        }
        const long long leaked = live_allocations.load() - live_before;
        Outcome o;
        o.actual = std::string("100 round trips; every round shape {12} and values 0..11: ")
            + (ok ? "yes" : "no");
#ifdef MT_COUNT_ALLOCATIONS
        o.actual += "; live allocations after minus before: " + std::to_string(leaked);
        ok = ok && leaked == 0;
#else
        (void)leaked;
#endif
        o.status = ok ? "PASS" : "FAIL";
        o.notes = reshape_note + ". Checker: " + checker_name();
        return o;
    });

    // ---- G. Ownership, copying and memory robustness
    add_row("TC46", "memory", "No invalid read/write, double free or reported leak attributable to the library", [&] {
        const long long live_before = live_allocations.load();
        bool ok = true;
        for (int i = 0; i < 10000; ++i) {
            FloatTensor t({100, 100});
            t.fill(1.0f);
            ok = ok && t.data()[0] == 1.0f && t.data()[t.size() - 1] == 1.0f;
        }
        const long long leaked = live_allocations.load() - live_before;
        Outcome o;
        o.actual = std::string("10,000 create/fill/destroy cycles completed; values correct: ")
            + (ok ? "yes" : "no");
#ifdef MT_COUNT_ALLOCATIONS
        o.actual += "; live allocations after minus before: " + std::to_string(leaked);
        ok = ok && leaked == 0;
#else
        (void)leaked;
#endif
        o.status = ok ? "PASS" : "FAIL";
        o.notes = "Checker: " + checker_name();
        return o;
    });

    // TC47 and TC48 copy what they observed into fixed-size arrays, so that no
    // string or vector built for the report is still alive when allocations
    // are counted.
    add_row("TC47", "memory", "A remains [1,2,3,4]; B becomes [99,2,3,4]", [&] {
        const long long live_before = live_allocations.load();
        bool ok = false;
        bool separate = false;
        float a_seen[4] = {};
        float b_seen[4] = {};
        {
            const FloatTensor a = make({2, 2}, {1, 2, 3, 4});
            FloatTensor b(a);
            b(std::vector<std::size_t>{0, 0}) = 99.0f;
            ok = equals(a, {2, 2}, {1, 2, 3, 4}) && equals(b, {2, 2}, {99, 2, 3, 4});
            separate = a.data() != b.data();
            std::memcpy(a_seen, a.data(), sizeof(a_seen));
            std::memcpy(b_seen, b.data(), sizeof(b_seen));
        }
        const long long leaked = live_allocations.load() - live_before;
        ok = ok && separate;
        std::string actual = "A [";
        for (int i = 0; i < 4; ++i) {
            actual += (i == 0 ? "" : ",") + format_number(a_seen[i]);
        }
        actual += "]; B [";
        for (int i = 0; i < 4; ++i) {
            actual += (i == 0 ? "" : ",") + format_number(b_seen[i]);
        }
        actual += std::string("]; separate buffers: ") + (separate ? "yes" : "no");
#ifdef MT_COUNT_ALLOCATIONS
        actual += "; live allocations after minus before: " + std::to_string(leaked);
        ok = ok && leaked == 0;
#else
        (void)leaked;
#endif
        Outcome o;
        o.actual = actual;
        o.status = ok ? "PASS" : "FAIL";
        o.notes = "Copy construction is a deep copy. Checker: " + checker_name();
        return o;
    });

    add_row("TC48", "memory", "B has shape {2,2}; A unchanged under deep-copy contract; no leak or double free", [&] {
        const long long live_before = live_allocations.load();
        bool ok = false;
        bool b_is_2x2 = false;
        float a_seen[4] = {};
        float b_seen[4] = {};
        {
            const FloatTensor a = make({2, 2}, {1, 2, 3, 4});
            FloatTensor b = make({1}, {9});
            b = a;
            b(std::vector<std::size_t>{1, 1}) = 77.0f;
            ok = equals(a, {2, 2}, {1, 2, 3, 4}) && equals(b, {2, 2}, {1, 2, 3, 77});
            b_is_2x2 = b.rank() == 2 && b.shape().dimensions()[0] == 2 && b.shape().dimensions()[1] == 2;
            std::memcpy(a_seen, a.data(), sizeof(a_seen));
            std::memcpy(b_seen, b.data(), b.size() >= 4 ? sizeof(b_seen) : 0);
        }
        const long long leaked = live_allocations.load() - live_before;
        std::string actual = "A shape {2,2} [";
        for (int i = 0; i < 4; ++i) {
            actual += (i == 0 ? "" : ",") + format_number(a_seen[i]);
        }
        actual += std::string("]; B shape ") + (b_is_2x2 ? "{2,2}" : "not {2,2}") + " [";
        for (int i = 0; i < 4; ++i) {
            actual += (i == 0 ? "" : ",") + format_number(b_seen[i]);
        }
        actual += "]";
#ifdef MT_COUNT_ALLOCATIONS
        actual += "; live allocations after minus before: " + std::to_string(leaked);
        ok = ok && leaked == 0;
#else
        (void)leaked;
#endif
        Outcome o;
        o.actual = actual;
        o.status = ok ? "PASS" : "FAIL";
        o.notes = "Copy assignment is a deep copy. Checker: " + checker_name();
        return o;
    });

    add_row("TC49", "memory", "No crash, invalid access or leak; each sum equals 250,000 x iteration", [&] {
        const long long live_before = live_allocations.load();
        bool ok = true;
        int first_bad = 0;
        for (int iteration = 1; iteration <= 100; ++iteration) {
            FloatTensor t({500, 500});
            t.fill(static_cast<float>(iteration));
            const double total = sum(t);
            if (!within_tolerance(total, 250000.0 * iteration)) {
                ok = false;
                if (first_bad == 0) {
                    first_bad = iteration;
                }
            }
        }
        const long long leaked = live_allocations.load() - live_before;
        Outcome o;
        o.actual = std::string("100 iterations; every sum correct: ") + (ok ? "yes" : "no");
        if (first_bad != 0) {
            o.actual += " (first wrong iteration " + std::to_string(first_bad) + ")";
        }
#ifdef MT_COUNT_ALLOCATIONS
        o.actual += "; live allocations after minus before: " + std::to_string(leaked);
        ok = ok && leaked == 0;
#else
        (void)leaked;
#endif
        o.actual += "; peak RSS so far " + format_number(peak_memory_mb(), 4) + " MB";
        o.status = ok ? "PASS" : "FAIL";
        o.notes = "Iteration numbers 1..100. Peak RSS is for the whole test process. Checker: "
            + checker_name();
        return o;
    });

    add_row("TC50", "memory", "Library reports overflow/allocation failure safely; no integer overflow, hang or memory exhaustion",
            [&] { return run_tc50(self, scratch); });

    return rows;
}

// `expected_failures` lists tests whose FAIL is already explained (TC35's
// expected values are wrong in the specification). They are still written as
// FAIL; they just don't make the exit code non-zero, so ctest can run this.
int run_functional(const std::string& self, const std::string& path, const std::string& team,
                   const std::vector<std::string>& expected_failures)
{
    const std::vector<Row> rows = functional_tests(self, path);

    std::ofstream out(path, std::ios::binary);
    out << "Team_ID,Test_ID,Operation,Status,Expected,Actual,Max_Abs_Error,Student_Median_ms,"
           "Reference_Median_ms,Speedup,Peak_Memory_MB,Notes\n";

    int failed = 0;
    int unexpected = 0;
    for (const Row& row : rows) {
        out << csv_field(team) << ',' << csv_field(row.id) << ',' << csv_field(row.operation) << ','
            << row.outcome.status << ',' << csv_field(row.expected) << ','
            << csv_field(row.outcome.actual) << ',' << csv_field(row.outcome.max_abs_error)
            << ",,,,," << csv_field(row.outcome.notes) << '\n';
        if (row.outcome.status == "FAIL") {
            ++failed;
            if (std::find(expected_failures.begin(), expected_failures.end(), row.id) == expected_failures.end()) {
                ++unexpected;
            }
        }
    }

    std::cout << "functional: " << rows.size() << " tests, " << failed << " failed ("
              << unexpected << " unexpected); " << checker_name() << '\n';
    return unexpected == 0 ? 0 : 1;
}

// ------------------------------------------------------------ benchmarks

FloatTensor filled_with_index(std::size_t rows, std::size_t cols, float first)
{
    FloatTensor t({rows, cols});
    for (std::size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = first + static_cast<float>(i);
    }
    return t;
}

FloatTensor formula_matrix(std::size_t rows, std::size_t cols, bool is_a)
{
    FloatTensor t({rows, cols});
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            t.data()[i * cols + j] = is_a ? matrix_a(i, j) : matrix_b(i, j);
        }
    }
    return t;
}

double tensor_sum(const FloatTensor& t)
{
    double total = 0.0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        total += t.data()[i];
    }
    return total;
}

void creation_bench(BenchRow& row, std::size_t n, float value, bool zero)
{
    std::optional<FloatTensor> holder;
    bool values_ok = true;

    const Measurement m = measure(
        [&] {
            if (zero) {
                holder.emplace(std::vector<std::size_t>{n, n});
            } else {
                holder.emplace(std::vector<std::size_t>{n, n}, value);
            }
        },
        [&] {
            const double total = tensor_sum(*holder);
            values_ok = values_ok && holder->data()[0] == value && holder->data()[holder->size() - 1] == value;
            holder.reset();  // destroy after each run, outside the timer
            return total;
        });

    row.samples_ms = m.samples_ms;
    row.checksum = m.checksum;
    row.expected_checksum = static_cast<double>(n) * static_cast<double>(n) * value;
    row.status = (values_ok && row.checksum == row.expected_checksum) ? "PASS" : "FAIL";
    row.notes = zero ? "FloatTensor({N,N}) zero-initializes" : "FloatTensor({N,N}, 1.0f)";
}

void elementwise_bench(BenchRow& row, std::size_t n, char op)
{
    const float a_value = op == '+' ? 1.25f : op == '-' ? 5.0f : 1.25f;
    const float b_value = op == '+' ? 2.5f : op == '-' ? 1.5f : 2.0f;
    const float expected = op == '+' ? 3.75f : op == '-' ? 3.5f : 2.5f;

    const FloatTensor a({n, n}, a_value);
    const FloatTensor b({n, n}, b_value);
    std::optional<FloatTensor> c;
    bool values_ok = true;

    const Measurement m = measure(
        [&] {
            if (op == '+') {
                c.emplace(add(a, b));
            } else if (op == '-') {
                c.emplace(subtract(a, b));
            } else {
                c.emplace(multiply(a, b));
            }
        },
        [&] {
            for (std::size_t i = 0; i < c->size(); ++i) {
                values_ok = values_ok && c->data()[i] == expected;
            }
            const double total = tensor_sum(*c);
            c.reset();
            return total;
        });

    row.samples_ms = m.samples_ms;
    row.checksum = m.checksum;
    row.expected_checksum = static_cast<double>(n) * static_cast<double>(n) * expected;
    row.status = (values_ok && row.checksum == row.expected_checksum) ? "PASS" : "FAIL";
    row.notes = std::string("Every element checked against ") + format_number(expected)
        + ". Element-wise operations are single-threaded in MiniTensor";
}

void matmul_bench(BenchRow& row, const MatmulShape& s)
{
    const FloatTensor a = formula_matrix(s.m, s.k, true);
    const FloatTensor b = formula_matrix(s.k, s.n, false);
    const std::vector<double> reference = reference_matmul(s.m, s.k, s.n);

    std::optional<FloatTensor> c;
    Accuracy accuracy;
    double expected_sum = 0.0;
    for (const double v : reference) {
        expected_sum += v;
    }

    const Measurement m = measure(
        [&] { c.emplace(matmul(a, b)); },
        [&] {
            Accuracy this_run;
            for (std::size_t i = 0; i < reference.size(); ++i) {
                accumulate_accuracy(this_run, c->data()[i], reference[i]);
            }
            accuracy.ok = accuracy.ok && this_run.ok;
            accuracy.max_abs_error = std::max(accuracy.max_abs_error, this_run.max_abs_error);
            accuracy.max_rel_error = std::max(accuracy.max_rel_error, this_run.max_rel_error);
            const double total = tensor_sum(*c);
            c.reset();
            return total;
        });

    row.samples_ms = m.samples_ms;
    row.checksum = m.checksum;
    row.expected_checksum = expected_sum;
    row.max_abs_error = accuracy.max_abs_error;
    row.max_rel_error = accuracy.max_rel_error;
    row.status = accuracy.ok ? "PASS" : "FAIL";
    const bool parallel = s.m >= 128
        && static_cast<long double>(s.m) * s.k * s.n >= 256.0L * 256.0L * 256.0L;
    row.notes = std::string("Full output compared with a double-precision reference. matmul() used its ")
        + (parallel ? "OpenMP parallel path" : "serial path") + " for this shape";
}

int run_bench(const std::string& test_id, const std::string& config, int run, const std::string& path)
{
    BenchRow row;
    row.library = "MiniTensor " + std::string(version());
    row.test_id = test_id;
    row.config = config;
    row.run = run;

    try {
        if (test_id == "PB01") {
            creation_bench(row, 100, 0.0f, true);
        } else if (test_id == "PB02") {
            creation_bench(row, 1000, 1.0f, false);
        } else if (test_id == "PB03") {
            creation_bench(row, 4000, 1.0f, false);
        } else if (test_id == "PB04") {
            const FloatTensor t = filled_with_index(1000, 1000, 1.0f);
            double result = 0.0;
            const Measurement m = measure(
                [&] {
                    double total = 0.0;
                    const float* data = t.data();
                    for (std::size_t i = 0; i < t.size(); ++i) {
                        total += data[i];
                    }
                    result = total;
                },
                [&] { return result; });
            row.samples_ms = m.samples_ms;
            row.checksum = m.checksum;
            row.expected_checksum = 500000500000.0;
            row.status = row.checksum == row.expected_checksum ? "PASS" : "FAIL";
            row.notes = "Contiguous row-major traversal through data(); float64 accumulator";
        } else if (test_id == "PB05") {
            const FloatTensor t = filled_with_index(1000, 1000, 0.0f);
            const auto indices = random_indices();
            double expected = 0.0;
            for (const auto& index : indices) {
                expected += static_cast<double>(index.first * 1000 + index.second);
            }
            double result = 0.0;
            const Measurement m = measure(
                [&] {
                    double total = 0.0;
                    for (const auto& index : indices) {
                        total += t({index.first, index.second});
                    }
                    result = total;
                },
                [&] { return result; });
            row.samples_ms = m.samples_ms;
            row.checksum = m.checksum;
            row.expected_checksum = expected;
            row.status = row.checksum == row.expected_checksum ? "PASS" : "FAIL";
            row.notes = "Bounds-checked t({row, col}) access. Indices: std::mt19937(20261117), "
                        "row = engine() % 1000 then col = engine() % 1000, 100,000 pairs";
        } else if (test_id == "PB06" || test_id == "PB07" || test_id == "PB08") {
            const char op = test_id == "PB06" ? '+' : test_id == "PB07" ? '-' : '*';
            elementwise_bench(row, static_cast<std::size_t>(std::stoul(config)), op);
        } else if (test_id == "PB09" || test_id == "PB10" || test_id == "PB11" || test_id == "MATMUL") {
            matmul_bench(row, parse_matmul(config));
        } else if (test_id == "PB12") {
            FloatTensor t = filled_with_index(1000, 1000, 0.0f);
            const float* buffer = t.data();
            constexpr int batch = 1000;
            bool ok = true;
            const Measurement m = measure(
                [&] {
                    for (int i = 0; i < batch / 2; ++i) {
                        t.reshape({100, 10000});
                        t.reshape({1000, 1000});
                    }
                },
                [&] {
                    ok = ok && t.data() == buffer && t.shape().dimensions() == std::vector<std::size_t>{1000, 1000};
                    return 0.0;
                });
            t.reshape({100, 10000});
            ok = ok && t.data() == buffer
                && t({37, 4567}) == 374567.0f && t({99, 9999}) == 999999.0f;
            row.samples_ms = m.samples_ms;
            row.checksum = tensor_sum(t);
            row.expected_checksum = 499999500000.0;
            row.status = (ok && row.checksum == row.expected_checksum) ? "PASS" : "FAIL";
            row.notes = "In-place reshape (metadata only; the element buffer is shared, not copied). "
                        "Each sample is one batch of 1000 reshape calls alternating {100,10000} and {1000,1000}, "
                        "because one call is near timer resolution";
        } else {
            std::cerr << "Unknown benchmark " << test_id << '\n';
            return 2;
        }
    } catch (const std::bad_alloc&) {
        row.status = "OUT_OF_MEMORY";
        row.notes = "std::bad_alloc";
    }

    if (!row.samples_ms.empty()) {
        row.stats = summarize(row.samples_ms);
    }
    row.peak_mb = peak_memory_mb();
    append_bench_row(path, row);

    std::cout << test_id << ' ' << config << " run " << run << ": " << row.status << ", median "
              << row.stats.median << " ms, threads " << thread_count() << '\n';
    return row.status == "PASS" ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "functional" && argc >= 3) {
            std::string team = "MiniTensor";
            std::vector<std::string> expected_failures;
            for (int i = 3; i < argc; ++i) {
                const std::string arg = argv[i];
                if (arg.rfind("--expect-fail=", 0) == 0) {
                    expected_failures.push_back(arg.substr(14));
                } else {
                    team = arg;
                }
            }
            return run_functional(argv[0], argv[2], team, expected_failures);
        }
        if (mode == "tc50-child") {
            return tc50_child();
        }
        if (mode == "indices" && argc == 3) {
            // PB05 index list as little-endian uint32 (row, col) pairs, so
            // that non-C++ references read exactly the same list
            std::ofstream out(argv[2], std::ios::binary);
            for (const auto& index : random_indices()) {
                const std::uint32_t pair[2] = {static_cast<std::uint32_t>(index.first),
                                               static_cast<std::uint32_t>(index.second)};
                out.write(reinterpret_cast<const char*>(pair), sizeof(pair));
            }
            return out ? 0 : 2;
        }
        if (mode == "bench" && argc == 6) {
            return run_bench(argv[2], argv[3], std::atoi(argv[4]), argv[5]);
        }
        std::cerr << "Usage:\n"
                     "  common_test_driver functional <results.csv> [team_id] [--expect-fail=TCxx ...]\n"
                     "  common_test_driver bench <test_id> <config> <run> <raw.csv>\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 2;
    }
}
