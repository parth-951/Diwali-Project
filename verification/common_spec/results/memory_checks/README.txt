Memory and undefined-behaviour checks for TC46-TC50 (all 50 functional tests were run in each build)

1. MSVC AddressSanitizer: Microsoft (R) C/C++ Optimizing Compiler Version 19.44.35229 for x64
   cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=cl -DMINITENSOR_USE_OPENMP=OFF "-DCMAKE_CXX_FLAGS=/fsanitize=address /EHsc"
   Result: functional_msvc_asan.csv / .log. Exit code 0, no AddressSanitizer report. Detects invalid reads/writes and double frees.
   MSVC's AddressSanitizer has no leak detection.

2. GCC UndefinedBehaviorSanitizer (trap mode): g++ 16.1.0 MSYS2 UCRT64
   -std=c++17 -O1 -g -fsanitize=undefined -fsanitize-undefined-trap-on-error -fno-sanitize-recover=all
   Result: functional_gcc_ubsan.csv. Exit code 0; any undefined behaviour would have aborted the process.

3. Leak check: the normal (non-sanitizer) driver build counts every operator new/delete in the process.
   TC45-TC49 check that live allocations return to their starting count (see the Actual column of ../functional.csv).

Not available on this machine (Windows): LeakSanitizer and Valgrind. The CommonSpecFunctional ctest target runs the same
driver in the Linux CI sanitizer job (ASan + UBSan, with LeakSanitizer on by default) once the branch is pushed.
