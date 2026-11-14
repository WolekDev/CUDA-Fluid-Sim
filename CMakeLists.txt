cmake_minimum_required(VERSION 3.20)

# Enable cmake compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Specify where NVCC is located
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.4/bin/nvcc")

# Target the RTX 4060 Mobile architecture (Ada Lovelace, Compute Capability 8.9)
set(CMAKE_CUDA_ARCHITECTURES 89)

project(FluidSim LANGUAGES CXX CUDA)

# Enforce C++20 standard for both host (CPU) and device (GPU) code
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA toolkit explicitly
find_package(CUDAToolkit 13.4 REQUIRED)

# Find required windowing and graphics libraries
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

# Tell CMake where to look for header files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libraries/glad/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libraries/nlohmann/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Automatically copy the config directory to the build output folder
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/config DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# 1. Define the main application target (pure production code)
add_executable(FluidSim 
    src/main.cpp
    src/FluidSimulation.cu
    src/Kernels.cu
    libraries/glad/src/glad.c
)

target_link_libraries(FluidSim PRIVATE 
    glfw 
    OpenGL::GL 
    CUDA::cudart
    CUDA::cuda_driver
)

# 2. Define the isolated test executable target in the tests/ directory
add_executable(FluidSimTest
    tests/tests.cu
    src/FluidSimulation.cu
    src/Kernels.cu
)

target_link_libraries(FluidSimTest PRIVATE 
    CUDA::cudart
    CUDA::cuda_driver
)

# Automatically copy the config folder to the build directory next to the executable
file(COPY ${CMAKE_SOURCE_DIR}/config DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# 3. Enable CTest and register the test binary
enable_testing()
add_test(NAME PipelineTestSuite COMMAND FluidSimTest)

# Ensure proper compilation flags for CUDA release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()