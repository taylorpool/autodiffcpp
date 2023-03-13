cmake_minimum_required(VERSION 3.10)

project(AutoDiff)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)


add_library(autodiff
    INTERFACE
)
target_include_directories(autodiff
    INTERFACE
    include
)

option(BUILD_AUTODIFF_TESTS OFF)

if(BUILD_AUTODIFF_TESTS)

include(FetchContent)

find_package(GTest REQUIRED)

enable_testing()

add_executable(
    tests
    test/test_autodiff.cpp
)
target_link_libraries(
    tests
    autodiff
    GTest::GTest
    GTest::Main
)

include(GoogleTest)
gtest_discover_tests(tests)

endif()