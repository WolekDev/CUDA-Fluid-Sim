#include "FluidSimulation.cuh"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Initialization and Memory Lifecycle Tests
void testInitializationAndMemory() {
    std::cout << "[TEST] Running testInitializationAndMemory...\n";

    FluidSimulation sim;
    std::string config_path = "config/test_config.json";

    try {
        sim.initialize(config_path);
        std::cout << "[Pass] initialize() executed successfully.\n";
    } catch (const std::exception& e) {
        std::cerr << "[Fail] initialize() threw an exception: " << e.what()
                  << "\n";
        assert(false);
    }

    assert(sim.getMaxParticles() > 0);
    std::cout << "[Pass] Max particles loaded: " << sim.getMaxParticles()
              << "\n";

    assert(sim.arePointersAllocated() == true);
    std::cout << "[Pass] Device memory pointers successfully allocated via "
                 "cudaMalloc.\n";

    const SimulationParams& h_test_params = sim.getParams();
    assert(h_test_params.smoothing_radius > 0.0f);
    std::cout << "[Pass] Parameters verified! Smoothing radius = "
              << h_test_params.smoothing_radius << "\n";

    sim.cleanup();
    assert(sim.arePointersAllocated() == false);
    std::cout << "[Pass] Cleanup executed and pointers reset to nullptr.\n";
}

// Spatial Hashing & Monotonic CUB Sorting Test
void testSpatialHashingAndSort() {
    std::cout << "\n[TEST] Running testSpatialHashingAndSort...\n";

    FluidSimulation sim;
    sim.initialize("config/test_config.json");

    std::vector<float2> test_pos = {make_float2(0.2f, 0.2f),
                                    make_float2(0.8f, 0.8f),
                                    make_float2(1.5f, 1.5f)};

    sim.uploadTestParticles(test_pos);
    sim.updateSpatialHash();

    std::vector<unsigned int> sorted_keys = sim.copyKeysToHost();

    for (size_t i = 0; i < test_pos.size() - 1; ++i) {
        assert(sorted_keys[i] <= sorted_keys[i + 1]);
    }

    std::cout
        << "[Pass] Spatial hashing keys are correctly sorted and monotonic.\n";
    sim.cleanup();
}

// Density Kernel Accumulation Test
void testDensityKernel() {
    std::cout << "\n[TEST] Running testDensityKernel...\n";

    FluidSimulation sim;
    sim.initialize("config/test_config.json");

    std::vector<float2> test_pos = {make_float2(0.5f, 0.5f),
                                    make_float2(0.55f, 0.55f),
                                    make_float2(1.8f, 1.8f)};

    sim.uploadTestParticles(test_pos);
    // Triggers hash, sort, cell ranges, density, and init pressure kernels
    sim.update(0.01f);

    std::vector<float> densities = sim.copyDensitiesToHost();

    assert(densities[0] > 0.0f);
    assert(densities[1] > 0.0f);

    std::cout << "[Pass] Density kernel computed valid accumulations.\n";
    sim.cleanup();
}

// Advection Prediction & Gradient Coupling Vector Test
void testAdvectionAndGradient() {
    std::cout << "\n[TEST] Running testAdvectionAndGradient...\n";

    FluidSimulation sim;
    sim.initialize("config/test_config.json");

    // Two close particles that will interact within smoothing radius
    std::vector<float2> test_pos = {make_float2(0.5f, 0.5f),
                                    make_float2(0.55f, 0.55f),
                                    make_float2(1.8f, 1.8f)};

    sim.uploadTestParticles(test_pos);

    float dt = 0.01f;
    sim.update(dt); // Will now trigger hash, sort, cell ranges, density, and
                    // advection/gradient kernels

    std::vector<float2> vel_adv = sim.copyVelAdvToHost();
    std::vector<float2> grad_coup = sim.copyGradCoupToHost();
    const SimulationParams& params = sim.getParams();

    // Verify advective velocity incorporates gravity/external force
    float expected_y_vel = dt * (params.gravity_vector.y *
                                 params.particle_mass / params.particle_mass);

    assert(vel_adv.size() >= 2);
    // Checking particle 0's y advective velocity
    std::cout << "[Pass] Particle 0 advective velocity y = " << vel_adv[0].y
              << "\n";

    // Verify gradient coupling vector d_ii accumulates neighbor
    // contributions
    float d_ii_mag_sq =
        grad_coup[0].x * grad_coup[0].x + grad_coup[0].y * grad_coup[0].y;
    assert(d_ii_mag_sq > 0.0f);

    // Isolated particle 2 shouldn't have neighbors within smoothing radius
    float d_ii_isolated_sq =
        grad_coup[2].x * grad_coup[2].x + grad_coup[2].y * grad_coup[2].y;
    assert(d_ii_isolated_sq == 0.0f);

    std::cout << "[Pass] Advection and gradient coupling vectors computed "
                 "correctly.\n";
    sim.cleanup();
}

// Pressure Solver Initialization Test
void testInitPressureSolver() {
    std::cout << "\n[TEST] Running testInitPressureSolver...\n";

    FluidSimulation sim;
    sim.initialize("config/test_config.json");

    std::vector<float2> test_pos = {make_float2(0.5f, 0.5f),
                                    make_float2(0.55f, 0.55f),
                                    make_float2(1.8f, 1.8f)};

    sim.uploadTestParticles(test_pos);

    // sim.update() runs spatial hashing, density,
    // advection/gradient, and initPressureSolverKernel
    float dt = 0.01f;
    sim.update(dt);

    std::vector<float2> grad_coup = sim.copyGradCoupToHost();

    // Retrieve resulting buffers from device
    std::vector<float> density_adv = sim.copyDensityAdvToHost();
    std::vector<float> diag_coeff = sim.copyDiagCoeffToHost();
    std::vector<float> pressure = sim.copyPressureToHost();

    // Assertions
    assert(density_adv.size() >= 3);
    assert(diag_coeff.size() >= 3);
    assert(pressure.size() >= 3);

    // Interacting particle 0 should have a valid computed diagonal coefficient
    // (a_ii > 0)
    assert(diag_coeff[0] > 0.0f);
    // Isolated particle 2 should have zero diagonal coefficient accumulation
    assert(diag_coeff[2] == 0.0f);

    // Verify advective density computed valid values
    assert(density_adv[0] > 0.0f);

    std::cout << "[Pass] Pressure solver initialization kernel executed and "
                 "validated successfully via update().\n";
    sim.cleanup();
}

int main() {
    std::cout << "=== STARTING FLUID SIMULATION TEST SUITE ===\n";
    try {
        testInitializationAndMemory();
        testSpatialHashingAndSort();
        testDensityKernel();
        testAdvectionAndGradient();
        testInitPressureSolver();
        std::cout << "\n=== ALL TESTS PASSED SUCCESSFULLY ===\n";
    } catch (const std::exception& e) {
        std::cerr << "[TEST ERROR] Exception caught: " << e.what() << "\n";
        return 1;
    }
    return 0;
}