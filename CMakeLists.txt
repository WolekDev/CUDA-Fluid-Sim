cmake_minimum_required(VERSION 3.20)

# Enable cmake compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Specify where NVCC is located
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.4/bin/nvcc")

# Target the RTX 4060 Mobile architecture (Ada Lovelace, Compute Capability 8.9)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Link the CUDA runtime into the executable. Must be set before project() so
# nvcc does not also pull in the shared libcudart.
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

project(FluidSim LANGUAGES C CXX CUDA)

# Enforce C++20 standard for both host (CPU) and device (GPU) code
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 13's CCCL headers reject MSVC's traditional preprocessor. nvcc does not
# pass /Zc:preprocessor on its own, so the host compiler has to be told here.
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif()

# Find CUDA toolkit explicitly
find_package(CUDAToolkit 13.4 REQUIRED)

# Build the vendored GLFW. Docs, tests, and examples are not needed, and
# Wayland pulls in extra system packages the X11 backend does not.
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
add_subdirectory(libraries/glfw-3.5.1)

find_package(OpenGL REQUIRED)

# Tell CMake where to look for header files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libraries/glad/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libraries/nlohmann/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libraries/imgui-1.93.0)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libraries/imgui-1.93.0/backends)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libraries/glfw-3.5.1/include)
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
    src/PipeGenerator.cpp
    src/RenderPack.cu
    src/FluidView.cpp
    libraries/glad/src/glad.c
    libraries/imgui-1.93.0/imgui.cpp
    libraries/imgui-1.93.0/imgui_draw.cpp
    libraries/imgui-1.93.0/imgui_tables.cpp
    libraries/imgui-1.93.0/imgui_widgets.cpp
    libraries/imgui-1.93.0/backends/imgui_impl_glfw.cpp
    libraries/imgui-1.93.0/backends/imgui_impl_opengl3.cpp
)

target_link_libraries(FluidSim PRIVATE 
    glfw 
    OpenGL::GL 
    CUDA::cudart_static
    CUDA::cuda_driver
)

# Define the test executable target in the tests/ directory
add_executable(FluidSimTest
    tests/tests.cu
    src/FluidSimulation.cu
    src/Kernels.cu
    src/PipeGenerator.cpp
)

target_link_libraries(FluidSimTest PRIVATE 
    CUDA::cudart_static
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
    testBoundaryUpload
    testBoundaryPsi
    testBoundaryDensity
    testBoundaryAdvectionAndDiagonal
    testBoundaryPressureSolve
    testBoundaryIntegration
    testBoundaryContainment
    testPipeConfigParse
    testPipeCircleWall
    testPipePolygonWall
    testPipeTaperSpacing
    testPipeCrossSectionFill
    testPipeConfigLoading
    testPipeInflowEmission
    testPipeInflowZone
    testPipeOutflowRemoval
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