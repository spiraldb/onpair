cmake_minimum_required(VERSION 3.20)
project(cpp_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Produces a target named `onpair`. Replace with:
#   add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../onpair-sys/cmake onpair-sys)
# once the upstream wrapper exists; the same `onpair` target name is expected.
add_subdirectory(mock)

add_executable(cpp_bench main.cpp)
target_link_libraries(cpp_bench PRIVATE onpair)
target_compile_options(cpp_bench PRIVATE -O3 -DNDEBUG)
