# A separate project that uses an *installed* MiniTensor, exactly as an
# outside developer would. CI builds it to verify the public release.
#
#   cmake -S examples/consumer -B build-consumer -DCMAKE_PREFIX_PATH=/path/to/install
#   cmake --build build-consumer
#   ./build-consumer/consumer

cmake_minimum_required(VERSION 3.16)

project(MiniTensorConsumer LANGUAGES CXX)

find_package(MiniTensor 1.0 REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE minitensor::minitensor)
