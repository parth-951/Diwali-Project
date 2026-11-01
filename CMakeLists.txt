cmake_minimum_required(VERSION 3.16)

project(MiniTensor
    VERSION 1.0.0
    DESCRIPTION "A small C++17 multidimensional tensor library"
    HOMEPAGE_URL "https://github.com/parth-951/Diwali-Project"
    LANGUAGES CXX
)

# Phase 7: Work both as a standalone project and as a dependency pulled in
# with add_subdirectory() or FetchContent. As a dependency, tests, examples,
# benchmarks and install rules are off by default.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(MINITENSOR_IS_TOP_LEVEL ON)
else()
    set(MINITENSOR_IS_TOP_LEVEL OFF)
endif()

option(MINITENSOR_USE_OPENMP "Use OpenMP for parallel matrix multiplication" ON)
option(MINITENSOR_BUILD_TESTS "Build the test suite" ${MINITENSOR_IS_TOP_LEVEL})
option(MINITENSOR_BUILD_EXAMPLES "Build the example applications" ${MINITENSOR_IS_TOP_LEVEL})
option(MINITENSOR_BUILD_BENCHMARKS "Build the benchmarks" ${MINITENSOR_IS_TOP_LEVEL})
option(MINITENSOR_INSTALL "Generate install and packaging rules" ${MINITENSOR_IS_TOP_LEVEL})

# Execution time is the evaluation criterion, so default to an optimized build.
if(MINITENSOR_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

function(enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()
endfunction()

# ---------------------------------------------------------------- library

configure_file(
    cmake/Version.h.in
    ${PROJECT_BINARY_DIR}/include/minitensor/Version.h
    @ONLY
)

add_library(minitensor
    src/Shape.cpp
    src/Storage.cpp
    src/Tensor.cpp
    src/Ops.cpp
    src/IO.cpp
    src/Version.cpp
)
add_library(minitensor::minitensor ALIAS minitensor)

target_include_directories(minitensor
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(minitensor PUBLIC cxx_std_17)

set_target_properties(minitensor PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

enable_warnings(minitensor)

# OpenMP is optional (the challenge lists it as such). Without it,
# matmul_parallel() falls back to sequential execution.
set(MINITENSOR_WITH_OPENMP OFF)

if(MINITENSOR_USE_OPENMP)
    find_package(OpenMP COMPONENTS CXX)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(minitensor PRIVATE OpenMP::OpenMP_CXX)
        set(MINITENSOR_WITH_OPENMP ON)
    else()
        message(WARNING "OpenMP not found; matmul_parallel() will run sequentially")
    endif()
endif()

message(STATUS "MiniTensor ${PROJECT_VERSION}: OpenMP ${MINITENSOR_WITH_OPENMP}")

# ------------------------------------------------------------------ tests

if(MINITENSOR_BUILD_TESTS)
    enable_testing()

    foreach(phase 1 2 3 4 5 6)
        set(test_target test_phase${phase})

        add_executable(${test_target} tests/${test_target}.cpp)
        target_link_libraries(${test_target} PRIVATE minitensor)
        enable_warnings(${test_target})

        # Keep assertions enabled in every test build, including Release.
        if(MSVC)
            target_compile_options(${test_target} PRIVATE /UNDEBUG)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${test_target} PRIVATE -UNDEBUG)
        endif()

        add_test(NAME Phase${phase}Tests COMMAND ${test_target})
    endforeach()
endif()

# --------------------------------------------------------------- examples

if(MINITENSOR_BUILD_EXAMPLES)
    add_executable(basic_example examples/main.cpp)
    target_link_libraries(basic_example PRIVATE minitensor)
    enable_warnings(basic_example)

    foreach(example quickstart linear_regression xor_neural_network image_processing)
        add_executable(${example} examples/${example}.cpp)
        target_link_libraries(${example} PRIVATE minitensor)
        enable_warnings(${example})

        # Phase 6: each application checks its own result and exits non-zero
        # on failure, so they double as end-to-end tests.
        if(MINITENSOR_BUILD_TESTS)
            add_test(NAME Example_${example} COMMAND ${example})
        endif()
    endforeach()
endif()

# ------------------------------------------------------------- benchmarks

if(MINITENSOR_BUILD_BENCHMARKS)
    foreach(benchmark benchmark_phase1 benchmark_v2 benchmark_matmul)
        add_executable(${benchmark} benchmarks/${benchmark}.cpp)
        target_link_libraries(${benchmark} PRIVATE minitensor)
        enable_warnings(${benchmark})
    endforeach()
endif()

# ------------------------------------------------- install and packaging

if(MINITENSOR_INSTALL)
    include(CMakePackageConfigHelpers)

    set(MINITENSOR_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/MiniTensor)

    install(TARGETS minitensor
        EXPORT MiniTensorTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(DIRECTORY include/minitensor DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${PROJECT_BINARY_DIR}/include/minitensor/Version.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/minitensor
    )

    # find_package(MiniTensor) support
    install(EXPORT MiniTensorTargets
        NAMESPACE minitensor::
        DESTINATION ${MINITENSOR_CMAKE_DIR}
    )

    configure_package_config_file(
        cmake/MiniTensorConfig.cmake.in
        ${PROJECT_BINARY_DIR}/MiniTensorConfig.cmake
        INSTALL_DESTINATION ${MINITENSOR_CMAKE_DIR}
    )

    write_basic_package_version_file(
        ${PROJECT_BINARY_DIR}/MiniTensorConfigVersion.cmake
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
        ${PROJECT_BINARY_DIR}/MiniTensorConfig.cmake
        ${PROJECT_BINARY_DIR}/MiniTensorConfigVersion.cmake
        DESTINATION ${MINITENSOR_CMAKE_DIR}
    )

    # pkg-config support for non-CMake builds
    set(MINITENSOR_PC_OPENMP_FLAGS "")

    if(MINITENSOR_WITH_OPENMP)
        set(MINITENSOR_PC_OPENMP_FLAGS "${OpenMP_CXX_FLAGS}")
    endif()

    configure_file(cmake/minitensor.pc.in ${PROJECT_BINARY_DIR}/minitensor.pc @ONLY)
    install(FILES ${PROJECT_BINARY_DIR}/minitensor.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    )

    install(FILES LICENSE README.md CHANGELOG.md DESTINATION ${CMAKE_INSTALL_DOCDIR})
    install(DIRECTORY docs/ DESTINATION ${CMAKE_INSTALL_DOCDIR})

    # Binary packages: `cpack` in the build directory
    set(CPACK_PACKAGE_NAME minitensor)
    set(CPACK_PACKAGE_VENDOR "MiniTensor contributors")
    set(CPACK_PACKAGE_CONTACT "${PROJECT_HOMEPAGE_URL}")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
    set(CPACK_PACKAGE_HOMEPAGE_URL "${PROJECT_HOMEPAGE_URL}")
    set(CPACK_RESOURCE_FILE_LICENSE ${PROJECT_SOURCE_DIR}/LICENSE)
    set(CPACK_RESOURCE_FILE_README ${PROJECT_SOURCE_DIR}/README.md)
    set(CPACK_PACKAGE_FILE_NAME
        "minitensor-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}"
    )

    if(WIN32)
        set(CPACK_GENERATOR ZIP)
    elseif(APPLE)
        set(CPACK_GENERATOR TGZ)
    else()
        set(CPACK_GENERATOR "TGZ;DEB")
    endif()

    set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
    set(CPACK_DEBIAN_PACKAGE_SECTION libdevel)

    if(MINITENSOR_WITH_OPENMP AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "libgomp1")
    endif()

    set(CPACK_SOURCE_GENERATOR TGZ)
    set(CPACK_SOURCE_IGNORE_FILES "/\\\\.git/" "/build.*/")

    include(CPack)
endif()
