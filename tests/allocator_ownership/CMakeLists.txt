cmake_minimum_required(VERSION 3.16)
project(allocator_ownership_tests LANGUAGES CXX)

# Standalone: no GPU, ORT library, or backend SDKs are needed, only ORT headers.
set(ORT_INCLUDE_DIR "" CACHE PATH "Directory containing onnxruntime_c_api.h")
if(NOT EXISTS "${ORT_INCLUDE_DIR}/onnxruntime_c_api.h")
    message(FATAL_ERROR "Set ORT_INCLUDE_DIR to the ONNX Runtime include directory")
endif()
find_package(Threads REQUIRED)
add_executable(allocator_ownership_test allocator_ownership_test.cc)
target_compile_features(allocator_ownership_test PRIVATE cxx_std_17)
target_include_directories(allocator_ownership_test PRIVATE
    "${ORT_INCLUDE_DIR}" "${CMAKE_CURRENT_LIST_DIR}/../../src/amdgpu")
target_link_libraries(allocator_ownership_test PRIVATE Threads::Threads)
enable_testing()
add_test(NAME allocator_ownership COMMAND allocator_ownership_test)
set_tests_properties(allocator_ownership PROPERTIES TIMEOUT 30)
