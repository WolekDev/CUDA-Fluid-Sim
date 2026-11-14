#pragma once

#include "SimulationParams.cuh"
#include <cuda_runtime.h>
#include <string>
#include <vector>

class FluidSimulation {
private:
    // Host Configuration & Control Variables
    int h_max_particles = 0;
    int h_max_iterations = 0;
    float h_density_error_threshold = 0.0f;
    int h_active_particle_count = 0;
    float2 h_start_vel = make_float2(0.0f, 0.0f);

    // Grid sizing metadata calculated on host during initialization
    int h_grid_width = 0;
    int h_grid_height = 0;
    int h_max_grid_cells = 0;

    // Hard coded value for block size
    static constexpr int kBlockSize = 256;

    // Simulation parameters (passed by value to GPU kernels)
    SimulationParams h_params;

    // Device Pointers
    float2* d_pos = nullptr;
    float2* d_vel = nullptr;
    float2* d_vel_adv = nullptr;
    float* d_density = nullptr;
    float* d_density_adv = nullptr;
    float* d_pressure = nullptr;
    float* d_pressure_prev = nullptr;
    float2* d_grad_coup_vect = nullptr; // d_ii
    float* d_diag_coeff = nullptr;      // a_ii

    // Spatial Hashing & CUB Sort Buffers
    unsigned int* d_keys_in = nullptr;
    unsigned int* d_keys_out = nullptr;
    unsigned int* d_values_in = nullptr;
    unsigned int* d_values_out = nullptr;
    unsigned int* d_cell_start_indices = nullptr;
    unsigned int* d_cell_end_indices = nullptr;

    // CUB Temporary Storage Workspaces
    void* d_cub_temp_storage = nullptr;
    size_t h_cub_temp_storage_bytes = 0;

    bool pointers_allocated = false;

public:
    FluidSimulation() = default;
    ~FluidSimulation();

    // Function headers
    void initialize(const std::string& config_path);
    void update(float delta_time);
    void updateSpatialHash();
    void cleanup();

    // Debug Functions
    int getMaxParticles() const { return h_max_particles; }
    int getMaxIterations() const { return h_max_iterations; }
    float getDensityErrorThreshold() const { return h_density_error_threshold; }
    const SimulationParams& getParams() const { return h_params; }

    bool arePointersAllocated() const {
        return d_pos != nullptr && d_vel != nullptr && d_density != nullptr &&
               d_density_adv != nullptr && d_pressure != nullptr &&
               d_keys_in != nullptr && d_keys_out != nullptr &&
               pointers_allocated;
    }

    // Test Helper Methods
    void uploadTestParticles(const std::vector<float2>& pts);
    std::vector<unsigned int> copyKeysToHost();
    std::vector<float> copyDensitiesToHost();
    std::vector<float2> copyVelAdvToHost() const;
    std::vector<float2> copyGradCoupToHost() const;
    std::vector<float> copyDensityAdvToHost() const;
    std::vector<float> copyDiagCoeffToHost() const;
    std::vector<float> copyPressureToHost() const;
};