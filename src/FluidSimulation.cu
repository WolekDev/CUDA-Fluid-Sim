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
        throw std::runtime_error("Failed to open configuration file at path: " +
                                 config_file);
    }

    // Ensure config file is not empty
    conf.seekg(0, std::ios::end);
    std::streamsize len = conf.tellg();
    conf.seekg(0, std::ios::beg);
    if (len == 0) {
        throw std::runtime_error("Config file is empty (0 bytes): " +
                                 config_file);
    }

    json data;
    try {
        conf >> data;
    } catch (const json::parse_error& e) {
        throw std::runtime_error("JSON parse error in " + config_file + ": " +
                                 std::string(e.what()));
    }

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
    h_params.relaxation_factor = data["fluid"]["relaxation_factor"];

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

    // Position
    cudaMalloc(&d_pos, h_max_particles * sizeof(float2));

    // Density & Density Adv & Density Predicted
    cudaMalloc(&d_density, h_max_particles * sizeof(float));
    cudaMalloc(&d_density_adv, h_max_particles * sizeof(float));
    cudaMalloc(&d_density_predicted, h_max_particles * sizeof(float));

    // Pressure & Previous Pressure
    cudaMalloc(&d_pressure, h_max_particles * sizeof(float));
    cudaMalloc(&d_pressure_prev, h_max_particles * sizeof(float));
    cudaMemset(d_pressure_prev, 0.0f, h_max_particles * sizeof(float));

    // Velocity & Velocity Advection
    cudaMalloc(&d_vel, h_max_particles * sizeof(float2));
    cudaMalloc(&d_vel_adv, h_max_particles * sizeof(float2));

    // Gradient Coupling Vector (dii)
    cudaMalloc(&d_grad_coup_vect, h_max_particles * sizeof(float2));

    // Diagonal Coefficient for Pressure Matrix (aii)
    cudaMalloc(&d_diag_coeff, h_max_particles * sizeof(float));

    // SUM_j(d_ij * p_j^l) buffer
    cudaMalloc(&d_sum_movement_pressure, h_max_particles * sizeof(float2));

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

    // Pre-allocate CUB Radix sort temp storage based on max particles
    cub::DeviceRadixSort::SortPairs(nullptr, h_cub_temp_storage_bytes,
                                    d_keys_in, d_keys_out, d_values_in,
                                    d_values_out, h_max_particles);
    cudaMalloc(&d_cub_temp_storage, h_cub_temp_storage_bytes);

    // Pre-allocate CUB Reduction temp storage and output scalar buffer for
    // convergence check
    cub::DeviceReduce::Sum(nullptr, h_cub_reduction_storage_bytes,
                           d_density_predicted, d_density_sum_output,
                           h_max_particles);
    cudaMalloc(&d_cub_reduction_storage, h_cub_reduction_storage_bytes);
    cudaMalloc(&d_density_sum_output, sizeof(float));

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

    // Initialize Pressure Solver
    // Calculates density adv, initial pressure, and a_ii
    initPressureSolverKernel<<<gridSize, kBlockSize>>>(
        d_pos, d_vel_adv, d_density, d_density_adv, d_grad_coup_vect,
        d_values_out, d_cell_start_indices, d_cell_end_indices, d_pressure,
        d_pressure_prev, d_diag_coeff, current_particles, h_params, delta_time);

    // Pressure loop
    int l = 0;
    float density_avg_l = 0.0f;

    while ((density_avg_l - h_params.rest_density > h_density_error_threshold ||
            l < 2) &&
           l < h_max_iterations) {
        // Calculate SUM_j(d_ij * p_j^l)
        calculateSumMovementFromPressureKernel<<<gridSize, kBlockSize>>>(
            d_pos, d_density, d_pressure, d_values_out, d_cell_start_indices,
            d_cell_end_indices, d_sum_movement_pressure, current_particles,
            h_params, delta_time);

        // Update pressure
        updatePressureKernel<<<gridSize, kBlockSize>>>(
            d_pos, d_density_adv, d_diag_coeff, d_pressure, d_values_out,
            d_cell_start_indices, d_cell_end_indices, d_sum_movement_pressure,
            d_grad_coup_vect, current_particles, h_params);

        // Update density
        updateDensityKernel<<<gridSize, kBlockSize>>>(
            d_pos, d_density_adv, d_diag_coeff, d_pressure, d_density_predicted,
            d_values_out, d_cell_start_indices, d_cell_end_indices,
            d_sum_movement_pressure, d_grad_coup_vect, current_particles,
            h_params);

        // Compute average predicted density via CUB parallel reduction
        cub::DeviceReduce::Sum(
            d_cub_reduction_storage, h_cub_reduction_storage_bytes,
            d_density_predicted, d_density_sum_output, current_particles);

        float h_density_sum = 0.0f;
        cudaMemcpy(&h_density_sum, d_density_sum_output, sizeof(float),
                   cudaMemcpyDeviceToHost);
        density_avg_l = h_density_sum / static_cast<float>(current_particles);

        l++;
    }

    // Save current pressure as previous pressure for the next frame
    cudaMemcpy(d_pressure_prev, d_pressure, h_max_particles * sizeof(float),
               cudaMemcpyDeviceToDevice);

    // Check if an error occurred while launching the kernel
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
    cudaFree(d_vel_adv);
    cudaFree(d_density);
    cudaFree(d_density_adv);
    cudaFree(d_density_predicted);
    cudaFree(d_pressure);
    cudaFree(d_pressure_prev);
    cudaFree(d_grad_coup_vect);
    cudaFree(d_diag_coeff);
    cudaFree(d_sum_movement_pressure);
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_in);
    cudaFree(d_values_out);
    cudaFree(d_cell_start_indices);
    cudaFree(d_cell_end_indices);
    cudaFree(d_cub_temp_storage);
    cudaFree(d_cub_reduction_storage);
    cudaFree(d_density_sum_output);

    // Nullify all pointers
    d_pos = d_vel = d_vel_adv = d_grad_coup_vect = nullptr;
    d_density = d_density_adv = d_density_predicted = nullptr;
    d_pressure = d_pressure_prev = d_diag_coeff = nullptr;
    d_sum_movement_pressure = nullptr;
    d_keys_in = d_keys_out = d_values_in = d_values_out = nullptr;
    d_cell_start_indices = d_cell_end_indices = nullptr;
    d_cub_temp_storage = d_cub_reduction_storage = nullptr;
    d_density_sum_output = nullptr;

    pointers_allocated = false;
}

/* Debug function: Uploads test particles for the simulation. */
void FluidSimulation::uploadTestParticles(const std::vector<float2>& pts) {
    if (pts.size() > static_cast<size_t>(h_max_particles)) {
        throw std::runtime_error(
            "Test particle count exceeds max_particles defined in config.");
    }

    // Set active particle count to match the test batch size
    h_active_particle_count = static_cast<int>(pts.size());

    cudaError_t err = cudaMemcpy(d_pos, pts.data(), pts.size() * sizeof(float2),
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to upload test particles: ") +
            cudaGetErrorString(err));
    }
}

/* Debug function: Copies keys array from device to host. */
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

/* Debug function: Copies the density array from device to host. */
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

/* Debug function: Copies the Velocity Adv array from device to host. */
std::vector<float2> FluidSimulation::copyVelAdvToHost() const {
    std::vector<float2> h_vel_adv(h_active_particle_count);
    if (d_vel_adv != nullptr && h_active_particle_count > 0) {
        cudaMemcpy(h_vel_adv.data(), d_vel_adv,
                   h_active_particle_count * sizeof(float2),
                   cudaMemcpyDeviceToHost);
    }
    return h_vel_adv;
}

/* Debug function: Copies the Gradient Coupling Vector from device to host. */
std::vector<float2> FluidSimulation::copyGradCoupToHost() const {
    std::vector<float2> h_grad_coup(h_active_particle_count);
    if (d_grad_coup_vect != nullptr && h_active_particle_count > 0) {
        cudaMemcpy(h_grad_coup.data(), d_grad_coup_vect,
                   h_active_particle_count * sizeof(float2),
                   cudaMemcpyDeviceToHost);
    }
    return h_grad_coup;
}

/* Debug function: Copies the advective density buffer from device to host. */
std::vector<float> FluidSimulation::copyDensityAdvToHost() const {
    int current_particles = std::min(h_active_particle_count, h_max_particles);
    std::vector<float> h_density_adv(current_particles);
    if (current_particles > 0 && d_density_adv) {
        cudaMemcpy(h_density_adv.data(), d_density_adv,
                   current_particles * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return h_density_adv;
}

/* Debug function: Copies the pressure diagonal coefficient buffer from device
 * to host. */
std::vector<float> FluidSimulation::copyDiagCoeffToHost() const {
    int current_particles = std::min(h_active_particle_count, h_max_particles);
    std::vector<float> h_diag_coeff(current_particles);
    if (current_particles > 0 && d_diag_coeff) {
        cudaMemcpy(h_diag_coeff.data(), d_diag_coeff,
                   current_particles * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return h_diag_coeff;
}

/* Debug function: Copies the pressure buffer from device to host. */
std::vector<float> FluidSimulation::copyPressureToHost() const {
    int current_particles = std::min(h_active_particle_count, h_max_particles);
    std::vector<float> h_pressure(current_particles);
    if (current_particles > 0 && d_pressure) {
        cudaMemcpy(h_pressure.data(), d_pressure,
                   current_particles * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return h_pressure;
}

/* Debug function: Copies the predicted density buffer from device to host. */
std::vector<float> FluidSimulation::copyPredictedDensitiesToHost() const {
    int current_particles = std::min(h_active_particle_count, h_max_particles);
    std::vector<float> h_pred_densities(current_particles);
    if (current_particles > 0 && d_density_predicted) {
        cudaMemcpy(h_pred_densities.data(), d_density_predicted,
                   current_particles * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return h_pred_densities;
}

/* Debug function: Copies the previous pressure buffer from device to host. */
std::vector<float> FluidSimulation::copyPrevPressureToHost() const {
    int current_particles = std::min(h_active_particle_count, h_max_particles);
    std::vector<float> h_prev_pressure(current_particles);
    if (current_particles > 0 && d_pressure_prev) {
        cudaMemcpy(h_prev_pressure.data(), d_pressure_prev,
                   current_particles * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return h_prev_pressure;
}