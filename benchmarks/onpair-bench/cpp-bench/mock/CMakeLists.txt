# Mock onpair static lib. Drop this subdir and point cpp-bench/CMakeLists.txt
# at `../../onpair-sys/cmake` to consume the real upstream target instead.
add_library(onpair STATIC src/onpair_mock.cpp)
target_include_directories(onpair PUBLIC include)
target_compile_features(onpair PUBLIC cxx_std_17)
