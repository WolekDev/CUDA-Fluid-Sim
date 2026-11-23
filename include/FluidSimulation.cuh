#pragma once

#include "Kernels.cuh"
#include "PipeGenerator.hpp"
#include "SimulationParams.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <optional>
#include <string>
#include <vector>

class FluidSimulation {
private:
    // Host Configuration & Control Variables
    int h_max_particles = 0;
    int h_max_iterations = 0;
    float h_density_error_threshold = 0.0f;
    int h_active_particle_count = 0;
    float3 h_start_vel = make_float3(0.0f, 0.0f, 0.0f);

    // Static boundary particles (pipe walls). Ihmsen et al., p. 5, Section 4.
    int h_max_boundary_particles = 0;
    int h_boundary_particle_count = 0;
    bool h_boundary_psi_dirty = false;

    // Optional pipe from the "pipe" config block, with its inlet and outlet
    std::optional<PipeSpec> h_pipe;
    float h_fluid_spacing = 0.0f; // rest spacing, cbrt(m / rho_0)
    float3 h_domain_max = make_float3(0.0f, 0.0f, 0.0f);
    std::vector<float3> h_inflow_layer; // one cross-section of inlet fluid
    float h_inflow_distance = 0.0f;     // inlet travel since the last layer
    bool h_capacity_warned = false;

    // Grid sizing metadata calculated on host during initialization
    int h_grid_width = 0;
    int h_grid_height = 0;
    int h_grid_depth = 0;
    int h_max_grid_cells = 0;

    // Hard coded value for block size
    static constexpr int kBlockSize = 256;

    // Simulation parameters (passed by value to GPU kernels)
    SimulationParams h_params;

    // Device Pointers
    float3* d_pos = nullptr;
    float3* d_pos_next = nullptr;
    float3* d_vel = nullptr;
    float3* d_vel_adv = nullptr;
    float* d_density = nullptr;
    float* d_density_adv = nullptr;
    float* d_density_predicted = nullptr;
    float* d_pressure = nullptr;
    float* d_pressure_prev = nullptr;
    float* d_pressure_next = nullptr;
    float3* d_grad_coup_vect = nullptr;        // d_ii
    float* d_diag_coeff = nullptr;             // a_ii
    float3* d_sum_movement_pressure = nullptr; // SUM_j(d_ij * p_j^l)

    // Boundary particle buffers
    float3* d_boundary_pos = nullptr; // x_b, never integrated
    float* d_boundary_psi = nullptr;  // Psi_b(rho_0) = rho_0 / delta_b

    // Outlet compaction (CUB DeviceSelect::Flagged)
    unsigned char* d_keep_flags = nullptr;
    int* d_num_selected = nullptr;
    void* d_select_temp_storage = nullptr;
    size_t h_select_temp_storage_bytes = 0;

    // Spatial Hashing & CUB Sort Buffers
    unsigned int* d_keys_in = nullptr;
    unsigned int* d_keys_out = nullptr;
    unsigned int* d_values_in = nullptr;
    unsigned int* d_values_out = nullptr;
    unsigned int* d_cell_start_indices = nullptr;
    unsigned int* d_cell_end_indices = nullptr;
    float* d_density_sum_output = nullptr;

    // CUB Temporary Storage Workspaces
    void* d_cub_temp_storage = nullptr;
    void* d_cub_reduction_storage = nullptr;
    size_t h_cub_temp_storage_bytes = 0;
    size_t h_cub_reduction_storage_bytes = 0;

    bool pointers_allocated = false;

    int activeFluidCount() const {
        return std::min(h_active_particle_count, h_max_particles);
    }
    BoundaryParticles boundaryView() const {
        return BoundaryParticles{d_boundary_pos, d_boundary_psi};
    }

    void loadPipe(const PipeSpec& spec);
    bool appendFluidParticles(const std::vector<float3>& positions,
                              const std::vector<float3>& velocities);
    void emitInflow(float delta_time);
    void removeOutflowParticles();

public:
    FluidSimulation() = default;
    ~FluidSimulation();

    // Function headers
    void initialize(const std::string& config_path);
    void update(float delta_time);
    void updateSpatialHash();
    void cleanup();

    /* Replaces the static boundary particles, e.g. samples of the pipe walls.
     * Positions must lie inside the domain. Psi_b is recomputed on the next
     * spatial hash. */
    void uploadBoundaryParticles(const std::vector<float3>& positions);

    // Debug Functions
    int getMaxParticles() const { return h_max_particles; }
    int getMaxBoundaryParticles() const { return h_max_boundary_particles; }
    int getBoundaryParticleCount() const { return h_boundary_particle_count; }
    int getActiveParticleCount() const { return activeFluidCount(); }
    float getFluidSpacing() const { return h_fluid_spacing; }
    const PipeSpec* getPipe() const { return h_pipe ? &*h_pipe : nullptr; }
    int getInflowLayerSize() const {
        return static_cast<int>(h_inflow_layer.size());
    }
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
    void uploadTestParticles(const std::vector<float3>& pts);
    void uploadTestVelocities(const std::vector<float3>& velocities);
    std::vector<unsigned int> copyKeysToHost();
    std::vector<float> copyDensitiesToHost();
    std::vector<float3> copyPositionsToHost() const;
    std::vector<float3> copyVelocitiesToHost() const;
    std::vector<float3> copyVelAdvToHost() const;
    std::vector<float3> copyGradCoupToHost() const;
    std::vector<float> copyDensityAdvToHost() const;
    std::vector<float> copyDiagCoeffToHost() const;
    std::vector<float> copyPressureToHost() const;
    std::vector<float> copyPredictedDensitiesToHost() const;
    std::vector<float> copyPrevPressureToHost() const;
    std::vector<float3> copyBoundaryPositionsToHost() const;
    std::vector<float> copyBoundaryPsiToHost() const;
};