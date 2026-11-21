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

# -----------------------------------------------------------------------------
# Configuration Sync Setup
# -----------------------------------------------------------------------------
# Automatically copy the config directory to the build output folder on every build
add_custom_target(CopyConfigs ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/config
    ${CMAKE_CURRENT_BINARY_DIR}/config
    COMMENT "Copying latest configuration files to build directory..."
)

# Define the main application target
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

# Define the test executable target in the tests/ directory
add_executable(FluidSimTest
    tests/tests.cu
    src/FluidSimulation.cu
    src/Kernels.cu
)

target_link_libraries(FluidSimTest PRIVATE 
    CUDA::cudart
    CUDA::cuda_driver
)

# Force both targets to depend on CopyConfigs so they are always synchronized
add_dependencies(FluidSim CopyConfigs)
add_dependencies(FluidSimTest CopyConfigs)

# Each test is a separate CTest target: ctest -R <name>
enable_testing()

# CMake Tools launches tests with `ctest -T test`. That dashboard mode looks
# for DartConfiguration.tcl, a leftover name from the old CTest dashboard
# client. Writing the file here removes the missing-file notice without
# adding the Experimental, Nightly, and Continuous targets from include(CTest).
if(NOT DEFINED DART_TESTING_TIMEOUT)
    set(DART_TESTING_TIMEOUT 1500)
endif()
if(NOT SITE)
    site_name(SITE)
endif()
if(NOT BUILDNAME)
    set(BUILDNAME "${CMAKE_SYSTEM_NAME}-${CMAKE_CXX_COMPILER_ID}")
endif()
configure_file(
    ${CMAKE_ROOT}/Modules/DartConfiguration.tcl.in
    ${CMAKE_BINARY_DIR}/DartConfiguration.tcl
    @ONLY
)
set(FLUIDSIM_TESTS
    testInitializationAndMemory
    testSpatialHashingAndSort
    testPoly6Kernel
    testSpikyKernel
    testDensityKernel
    testAdvectionAndGradient
    testViscosityKernel
    testViscosityForce
    testInitPressureSolver
    testPressureSolve
    testPredictedDensity
    testIntegration
    testPreviousPressureCopy
)
foreach(test_name IN LISTS FLUIDSIM_TESTS)
    add_test(NAME ${test_name} COMMAND FluidSimTest ${test_name})
    set_tests_properties(${test_name} PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endforeach()

# Ensure proper compilation flags for CUDA release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()