#!/usr/bin/env python3
"""Writes seeded test inputs and NumPy's answers to a binary file.

verify_numerical.cpp reads this file, runs the same operations through
MiniTensor and compares the results. Whether a shape pair counts as
incompatible is decided by NumPy actually raising, not by a hand-written list.

File format (little-endian): a sequence of cases, each made of
    str group, str name, str op, u32 expect_throw, u32 input_count,
    inputs..., and the expected tensor unless expect_throw is set
where str is u32 length + bytes, and tensor is u32 rank + u64 dims + doubles.
"""

import struct
import sys

import numpy as np

SEED = 20261103
FORMAT_MAGIC = b"MTREF1\n"


def write_str(out, text):
    data = text.encode()
    out.write(struct.pack("<I", len(data)))
    out.write(data)


def write_tensor(out, array):
    array = np.asarray(array, dtype="<f8")
    out.write(struct.pack("<I", array.ndim))
    out.write(struct.pack(f"<{array.ndim}Q", *array.shape))
    out.write(array.tobytes(order="C"))


def write_case(out, group, name, op, inputs, expected=None):
    write_str(out, group)
    write_str(out, name)
    write_str(out, op)
    out.write(struct.pack("<II", 1 if expected is None else 0, len(inputs)))
    for tensor in inputs:
        write_tensor(out, tensor)
    if expected is not None:
        write_tensor(out, expected)


def shape_name(*shapes):
    return "_".join("x".join(str(d) for d in shape) for shape in shapes)


def main():
    rng = np.random.default_rng(SEED)

    def uniform(shape):
        return rng.uniform(-100.0, 100.0, size=shape)

    def wide_range(shape):
        # Twelve orders of magnitude, to stress rounding as well as indexing.
        signs = rng.choice([-1.0, 1.0], size=shape)
        return signs * 10.0 ** rng.uniform(-6.0, 6.0, size=shape)

    with open(sys.argv[1], "wb") as out:
        out.write(FORMAT_MAGIC)

        # Matrix multiplication. Covers vector-like shapes, non-square shapes,
        # and both sides of the serial/OpenMP switch in matmul() (rows >= 128
        # and rows*inner*cols >= 256^3).
        for m, k, n in [
            (1, 1, 1), (1, 7, 1), (7, 1, 7), (2, 3, 4), (5, 2, 9), (16, 16, 16),
            (64, 128, 32), (127, 512, 512), (128, 256, 512), (128, 64, 256),
            (100, 100, 100), (255, 256, 257), (256, 256, 256), (300, 260, 280),
            (512, 512, 512), (1024, 1024, 1024),
        ]:
            a, b = uniform((m, k)), uniform((k, n))
            name = shape_name((m, k), (k, n))
            expected = np.matmul(a, b)
            write_case(out, "matmul", name, "matmul", [a, b], expected)
            write_case(out, "matmul", name, "matmul_parallel", [a, b], expected)

        a, b = wide_range((200, 150)), wide_range((150, 100))
        write_case(out, "matmul", "wide_range_200x150_150x100", "matmul", [a, b], np.matmul(a, b))

        for a_shape, b_shape in [((2, 3), (2, 3)), ((3, 4), (5, 6)), ((1, 2), (1, 2))]:
            a, b = uniform(a_shape), uniform(b_shape)
            try:
                np.matmul(a, b)
            except ValueError:
                write_case(out, "matmul_incompatible", shape_name(a_shape, b_shape), "matmul", [a, b])
            else:
                raise SystemExit(f"NumPy accepted {a_shape} @ {b_shape}")

        # Broadcasting: every pair is run through add, subtract and multiply.
        operations = {
            "add_broadcast": np.add,
            "subtract_broadcast": np.subtract,
            "multiply_broadcast": np.multiply,
        }
        for a_shape, b_shape in [
            ((3, 4), (3, 4)), ((3, 4), (4,)), ((4,), (3, 4)), ((3, 1), (1, 4)),
            ((1,), (5, 5)), ((5,), (1,)), ((1, 1), (1, 1)), ((2, 3, 4), (3, 1)),
            ((4, 1, 3), (2, 3)), ((2, 1, 3, 1), (5, 1, 4)), ((1, 6, 1), (7, 1, 5)),
            ((200, 300), (300,)), ((64, 1, 32), (1, 128, 1)),
            # Incompatible
            ((3,), (4,)), ((2, 3), (3, 2)), ((2, 3, 4), (3, 3)), ((3, 4), (2, 3, 5)),
            ((2, 2), (4,)), ((5, 1, 2), (5, 3)),
        ]:
            a, b = uniform(a_shape), uniform(b_shape)
            name = shape_name(a_shape, b_shape)
            for op, function in operations.items():
                try:
                    expected = function(a, b)
                except ValueError:
                    write_case(out, "broadcast_incompatible", name, op, [a, b])
                else:
                    write_case(out, "broadcast", name, op, [a, b], expected)

        # Transpose
        for shape in [(1, 1), (1, 6), (6, 1), (3, 5), (64, 128), (300, 200), (1024, 1024)]:
            a = uniform(shape)
            write_case(out, "transpose", shape_name(shape), "transpose", [a], a.T)

        # Same-shape element-wise operations
        for shape in [(1,), (9,), (4, 6), (2, 3, 4), (256, 256)]:
            a, b = uniform(shape), uniform(shape)
            name = shape_name(shape)
            write_case(out, "elementwise", name, "add", [a, b], a + b)
            write_case(out, "elementwise", name, "subtract", [a, b], a - b)
            write_case(out, "elementwise", name, "multiply", [a, b], a * b)

        # Reductions (the answer is stored as a one-element tensor)
        for shape in [(1,), (17,), (40, 25), (1000, 1000)]:
            a = uniform(shape)
            name = shape_name(shape)
            write_case(out, "reduction", name, "sum", [a], [np.sum(a)])
            write_case(out, "reduction", name, "mean", [a], [np.mean(a)])
            write_case(out, "reduction", name, "min", [a], [np.min(a)])
            write_case(out, "reduction", name, "max", [a], [np.max(a)])

        # Reshape. The second input holds the target shape.
        for shape, target in [((6,), (2, 3)), ((2, 3), (3, 2)), ((2, 3, 4), (4, 6)), ((12,), (2, 2, 3))]:
            a = uniform(shape)
            write_case(out, "reshape", shape_name(shape, target), "reshape", [a, target], a.reshape(target))
        for shape, target in [((6,), (4, 2)), ((2, 3), (7,))]:
            a = uniform(shape)
            try:
                a.reshape(target)
            except ValueError:
                write_case(out, "reshape_incompatible", shape_name(shape, target), "reshape", [a, target])
            else:
                raise SystemExit(f"NumPy accepted reshape {shape} -> {target}")

        # Deliberate scope limits: NumPy accepts these and MiniTensor rejects
        # them, because its matmul and transpose are defined for 2-D tensors.
        write_case(out, "scope_limit", "matmul_rank1_3_3x2", "matmul", [uniform((3,)), uniform((3, 2))])
        write_case(out, "scope_limit", "transpose_rank3_2x3x4", "transpose", [uniform((2, 3, 4))])


if __name__ == "__main__":
    main()
