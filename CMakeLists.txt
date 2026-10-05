cmake_minimum_required(VERSION 3.18)
project(MicroTensor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 静态库会被链进 Python 扩展(.so)，必须位置无关
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

# 核心库：Tensor 抽象 + 存储底座 + CUDA kernel
add_library(microtensor STATIC
    src/Tensor.cpp
    src/StorageImpl.cpp
    src/ops_elementwise_cpu.cpp
    src/add.cu
    src/matmul.cu
    src/contiguous.cu
    src/elementwise.cu
)
target_include_directories(microtensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(microtensor PUBLIC CUDA::cudart)

# 测试（assert 驱动，无外部框架依赖；ctest 运行）
option(MT_BUILD_TESTS "Build test executables" ON)
if(MT_BUILD_TESTS)
    enable_testing()
    foreach(t IN ITEMS test_tensor test_matmul test_ops test_gpu_matmul)
        add_executable(${t} tests/${t}.cpp)
        target_link_libraries(${t} PRIVATE microtensor)
        add_test(NAME ${t} COMMAND ${t})
    endforeach()
endif()

# Python 扩展模块：构建后 PYTHONPATH=<build 目录> 即可 import mlspirit
option(MT_BUILD_PYTHON "Build Python extension module (requires pybind11)" ON)
if(MT_BUILD_PYTHON)
    execute_process(
        COMMAND python3 -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    find_package(pybind11 CONFIG QUIET)
    if(pybind11_FOUND)
        # 扩展核心编译为包私有模块 _mlspirit，直接落进 Python 包目录
        #（对标 torch 的 torch._C；PYTHONPATH=python 即可 import mlspirit）
        pybind11_add_module(_mlspirit MODULE bindings/mlspirit_py.cpp)
        target_link_libraries(_mlspirit PRIVATE microtensor)
        set_target_properties(_mlspirit PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/mlspirit)
        if(MT_BUILD_TESTS)
            add_test(NAME python_bindings COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mlspirit.py)
            set_tests_properties(python_bindings PROPERTIES
                ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python")
            add_test(NAME python_autograd COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_autograd.py)
            set_tests_properties(python_autograd PROPERTIES
                ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python")
        endif()
    else()
        message(WARNING "pybind11 not found (pip install pybind11); skipping Python bindings")
    endif()
endif()
