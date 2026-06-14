cmake_minimum_required(VERSION 3.15)
project(dks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Header-only core library.
add_library(dks INTERFACE)
target_include_directories(dks INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(dks INTERFACE cxx_std_17)

# Command-line tool.
add_executable(dks_cli cli/dks_cli.cpp)
target_link_libraries(dks_cli PRIVATE dks)
set_target_properties(dks_cli PROPERTIES OUTPUT_NAME dks)

# Tests.
enable_testing()
add_executable(test_dks tests/test_dks.cpp)
target_link_libraries(test_dks PRIVATE dks)
add_test(NAME test_dks COMMAND test_dks)
