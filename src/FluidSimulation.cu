#include "FluidSimulation.cuh"
#include "Kernels.cuh"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;

/* Destructor to release all allocated memory. */
FluidSimulation::~FluidSimulation() { cleanup(); }

/* Init function. Allocates memory and prepares variables from config. */
void FluidSimulation::initialize(const std::string& config_file) {
    // Open config file and create JSON object
    std::ifstream conf(config_file);
    if (!conf.is_open()) {
        throw std::runtime_error("Failed to open configuration file: " +
                                 config_file);
    }
    json data = json::parse(conf);

    // Parse config & simulation parameters
    h_max_iterations = data["solver"]["max_iterations"];
    h_density_error_threshold = data["solver"]["density_error_threshold"];
    h_max_particles = data["simulation"]["max_particles"];
    h_active_particle_count = h_max_particles; // Swap to 0 for dynamic spawning
    h_start_vel = make_float2(data["fluid"]["start_vel"][0],
                              data["fluid"]["start_vel"][1]);

    auto grav = data["simulation"]["gravity"];
    h_params.gravity_vector = make_float2(grav[0], grav[1]);
    h_params.smoothing_radius = data["fluid"]["smoothing_radius"];
    h_params.rest_density = data["fluid"]["rest_density"];
    h_params.time_step = data["simulation"]["time_step"];
    h_params.particle_mass = data["fluid"]["particle_mass"];
    h_params.viscosity = data["fluid"]["viscosity"];

    // Parse domain boundaries & store in h_params for kernel access
    h_params.domain_min_x = data["domain"]["min_x"];
    h_params.domain_min_y = data["domain"]["min_y"];
    float domain_max_x = data["domain"]["max_x"];
    float domain_max_y = data["domain"]["max_y"];

    // Compute grid dimensions
    float h = h_params.smoothing_radius;
    h_grid_width =
        static_cast<int>(ceilf((domain_max_x - h_params.domain_min_x) / h));
    h_grid_height =
        static_cast<int>(ceilf((domain_max_y - h_params.domain_min_y) / h));
    h_max_grid_cells = h_grid_width * h_grid_height;

    h_params.grid_width = h_grid_width;
    h_params.grid_height = h_grid_height;

    // Allocate particle buffers
    cudaMalloc(&d_pos, h_max_particles * sizeof(float2));    // Position array
    cudaMalloc(&d_density, h_max_particles * sizeof(float)); // Density array
    cudaMalloc(&d_predicted_density,
               h_max_particles * sizeof(float)); // Predicted density array
    cudaMalloc(&d_pressure, h_max_particles * sizeof(float)); // Pressure array
    cudaMalloc(&d_vel, h_max_particles * sizeof(float2));     // Velocity array
    cudaMalloc(&d_vel_adv,
               h_max_particles * sizeof(float2)); // Advection velocity array
    cudaMalloc(&d_grad_coup_vect,
               h_max_particles *
                   sizeof(float2)); // Gradient Coupling Vector (dii) array

    // Allocate hash key & value arrays
    cudaMalloc(&d_keys_in,
               h_max_particles * sizeof(unsigned int)); // Keys before sorting
    cudaMalloc(&d_keys_out,
               h_max_particles * sizeof(unsigned int)); // Keys after sorting
    cudaMalloc(&d_values_in,
               h_max_particles * sizeof(unsigned int)); // Values before sorting
    cudaMalloc(&d_values_out,
               h_max_particles * sizeof(unsigned int)); // Values after sorting

    // Allocate cell range lookup tables
    cudaMalloc(&d_cell_start_indices, h_max_grid_cells * sizeof(unsigned int));
    cudaMalloc(&d_cell_end_indices, h_max_grid_cells * sizeof(unsigned int));

    // Pre-allocate CUB Radix sort temp storage
    cub::DeviceRadixSort::SortPairs(nullptr, h_cub_temp_storage_bytes,
                                    d_keys_in, d_keys_out, d_values_in,
                                    d_values_out, h_max_particles);
    cudaMalloc(&d_cub_temp_storage, h_cub_temp_storage_bytes);

    // Set flag that all pointers are allocated
    pointers_allocated = true;
}

/* Update function for fluid simulation. */
void FluidSimulation::update(float delta_time) {
    // Calculate grid size based on current particle count
    int current_particles = std::min(h_active_particle_count, h_max_particles);
    if (current_particles <= 0)
        return;

    int gridSize = (current_particles + kBlockSize - 1) / kBlockSize;

    // Get latest spatial hashes for the particles.
    // Sorts the keys and values using CUB Radix sort.
    updateSpatialHash();

    // Compute density
    computeDensityKernel<<<gridSize, kBlockSize>>>(
        d_values_out, d_pos, d_cell_start_indices, d_cell_end_indices,
        d_density, current_particles, h_params);

    // Compute Advection and Gradient
    computeAdvectionAndGradientKernel<<<gridSize, kBlockSize>>>(
        d_pos, d_vel, nullptr, d_vel_adv, d_density, d_grad_coup_vect,
        d_values_out, d_cell_start_indices, d_cell_end_indices,
        current_particles, h_params, delta_time);

    // Check if an error occured while launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err)
                  << "\n";
    }
}

/* Get new spatial hashes for the particles. */
void FluidSimulation::updateSpatialHash() {

    // Calculate grid size based on current particle count
    int current_particles = std::min(h_active_particle_count, h_max_particles);
    if (current_particles <= 0)
        return;

    int gridSize = (current_particles + kBlockSize - 1) / kBlockSize;

    // Compute hash for each particle. Updates keys and values arrays.
    // This gets two arrays. One for keys, which are the cell codes of each
    // particle. The other for values, which are the index numbers.
    computeHashKernel<<<gridSize, kBlockSize>>>(d_pos, d_keys_in, d_values_in,
                                                current_particles, h_params);

    // Sort using CUB Radix sort. This ensures that the key value pairs
    // stay together after sort. End goal is to have all identical cell
    // keys grouped together.
    cub::DeviceRadixSort::SortPairs(
        d_cub_temp_storage, h_cub_temp_storage_bytes, d_keys_in, d_keys_out,
        d_values_in, d_values_out, current_particles);

    // Clear values in index arrays
    cudaMemset(d_cell_start_indices, 0xFF,
               h_max_grid_cells * sizeof(unsigned int));
    cudaMemset(d_cell_end_indices, 0xFF,
               h_max_grid_cells * sizeof(unsigned int));

    // Get new start and end index for each cell.
    findCellRangesKernel<<<gridSize, kBlockSize>>>(
        d_keys_out, d_cell_start_indices, d_cell_end_indices,
        current_particles);
}

/* Deallocate pointers before finishing program. */
void FluidSimulation::cleanup() {
    if (!pointers_allocated)
        return;

    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_density);
    cudaFree(d_predicted_density);
    cudaFree(d_pressure);
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_in);
    cudaFree(d_values_out);
    cudaFree(d_cell_start_indices);
    cudaFree(d_cell_end_indices);
    cudaFree(d_cub_temp_storage);
    cudaFree(d_vel_adv);
    cudaFree(d_grad_coup_vect);

    d_pos = d_vel = d_vel_adv = d_grad_coup_vect = nullptr;
    d_density = d_predicted_density = d_pressure = nullptr;
    d_keys_in = d_keys_out = nullptr;
    d_values_in = d_values_out = nullptr;
    d_cell_start_indices = d_cell_end_indices = nullptr;
    d_cub_temp_storage = nullptr;
    pointers_allocated = false;
}

/* Debug function: Setter for test particles */
void FluidSimulation::uploadTestParticles(const std::vector<float2>& pts) {
    if (pts.size() > static_cast<size_t>(h_max_particles)) {
        throw std::runtime_error(
            "Test particle count exceeds max_particles defined in config.");
    }

    cudaError_t err = cudaMemcpy(d_pos, pts.data(), pts.size() * sizeof(float2),
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to upload test particles: ") +
            cudaGetErrorString(err));
    }
}

/* Debug function: Getter for keys from host. */
std::vector<unsigned int> FluidSimulation::copyKeysToHost() {
    std::vector<unsigned int> host_keys(h_max_particles);
    cudaError_t err = cudaMemcpy(host_keys.data(), d_keys_out,
                                 h_max_particles * sizeof(unsigned int),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to copy keys to host: ") +
                                 cudaGetErrorString(err));
    }
    return host_keys;
}

/* Debug function: Getter for densities. */
std::vector<float> FluidSimulation::copyDensitiesToHost() {
    std::vector<float> host_densities(h_max_particles);
    cudaError_t err =
        cudaMemcpy(host_densities.data(), d_density,
                   h_max_particles * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to copy densities to host: ") +
            cudaGetErrorString(err));
    }
    return host_densities;
}

/* Debug function: Getter for Velocity Advection. */
std::vector<float2> FluidSimulation::copyVelAdvToHost() const {
    std::vector<float2> h_vel_adv(h_active_particle_count);
    if (d_vel_adv != nullptr && h_active_particle_count > 0) {
        cudaMemcpy(h_vel_adv.data(), d_vel_adv,
                   h_active_particle_count * sizeof(float2),
                   cudaMemcpyDeviceToHost);
    }
    return h_vel_adv;
}

/* Debug function: Getter for Gradient Coupling Vector. */
std::vector<float2> FluidSimulation::copyGradCoupToHost() const {
    std::vector<float2> h_grad_coup(h_active_particle_count);
    if (d_grad_coup_vect != nullptr && h_active_particle_count > 0) {
        cudaMemcpy(h_grad_coup.data(), d_grad_coup_vect,
                   h_active_particle_count * sizeof(float2),
                   cudaMemcpyDeviceToHost);
    }
    return h_grad_coup;
}