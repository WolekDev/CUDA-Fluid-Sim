#include "FluidSimulation.cuh"
#include "Kernels.cuh"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;

/* Destructor to release all allocated memory. */
FluidSimulation::~FluidSimulation() { cleanup(); }

FluidSimulation::DeviceFields FluidSimulation::deviceFields() const {
    return {d_pos,
            d_vel,
            d_pressure,
            d_density,
            activeFluidCount(),
            d_boundary_pos,
            h_boundary_particle_count};
}

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
    h_max_boundary_particles =
        data["simulation"].value("max_boundary_particles", 0);
    h_boundary_particle_count = 0;
    h_boundary_psi_dirty = false;
    h_start_vel = make_float3(data["fluid"]["start_vel"][0],
                              data["fluid"]["start_vel"][1],
                              data["fluid"]["start_vel"][2]);

    auto grav = data["simulation"]["gravity"];
    h_params.gravity_vector = make_float3(grav[0], grav[1], grav[2]);
    h_params.smoothing_radius = data["fluid"]["smoothing_radius"];
    h_params.rest_density = data["fluid"]["rest_density"];
    h_params.time_step = data["simulation"]["time_step"];
    h_params.particle_mass = data["fluid"]["particle_mass"];
    h_params.viscosity = data["fluid"]["viscosity"];
    h_params.relaxation_factor = data["fluid"]["relaxation_factor"];

    // Parse domain boundaries & store in h_params for kernel access
    h_params.domain_min_x = data["domain"]["min_x"];
    h_params.domain_min_y = data["domain"]["min_y"];
    h_params.domain_min_z = data["domain"]["min_z"];
    float domain_max_x = data["domain"]["max_x"];
    float domain_max_y = data["domain"]["max_y"];
    float domain_max_z = data["domain"]["max_z"];

    // Compute grid dimensions. Cell size is the kernel support radius.
    float h = h_params.smoothing_radius;
    h_grid_width =
        static_cast<int>(ceilf((domain_max_x - h_params.domain_min_x) / h));
    h_grid_height =
        static_cast<int>(ceilf((domain_max_y - h_params.domain_min_y) / h));
    h_grid_depth =
        static_cast<int>(ceilf((domain_max_z - h_params.domain_min_z) / h));
    h_max_grid_cells = h_grid_width * h_grid_height * h_grid_depth;

    h_params.grid_width = h_grid_width;
    h_params.grid_height = h_grid_height;
    h_params.grid_depth = h_grid_depth;

    // Allocate particle buffers

    // Position
    cudaMalloc(&d_pos, h_max_particles * sizeof(float3));
    cudaMalloc(&d_pos_next, h_max_particles * sizeof(float3));
    cudaMemset(d_pos, 0, h_max_particles * sizeof(float3));
    cudaMemset(d_pos_next, 0, h_max_particles * sizeof(float3));

    // Density & Density Adv & Density Predicted
    cudaMalloc(&d_density, h_max_particles * sizeof(float));
    cudaMalloc(&d_density_adv, h_max_particles * sizeof(float));
    cudaMalloc(&d_density_predicted, h_max_particles * sizeof(float));
    cudaMemset(d_density, 0, h_max_particles * sizeof(float));
    cudaMemset(d_density_adv, 0, h_max_particles * sizeof(float));
    cudaMemset(d_density_predicted, 0, h_max_particles * sizeof(float));

    // Pressure, previous pressure, and the Jacobi output for p^{l+1}
    cudaMalloc(&d_pressure, h_max_particles * sizeof(float));
    cudaMalloc(&d_pressure_prev, h_max_particles * sizeof(float));
    cudaMalloc(&d_pressure_next, h_max_particles * sizeof(float));
    cudaMemset(d_pressure, 0, h_max_particles * sizeof(float));
    cudaMemset(d_pressure_prev, 0, h_max_particles * sizeof(float));
    cudaMemset(d_pressure_next, 0, h_max_particles * sizeof(float));

    // Velocity & Velocity Advection
    cudaMalloc(&d_vel, h_max_particles * sizeof(float3));
    cudaMalloc(&d_vel_adv, h_max_particles * sizeof(float3));
    cudaMemset(d_vel, 0, h_max_particles * sizeof(float3));
    cudaMemset(d_vel_adv, 0, h_max_particles * sizeof(float3));

    // Displacement factor d_ii
    cudaMalloc(&d_grad_coup_vect, h_max_particles * sizeof(float3));
    cudaMemset(d_grad_coup_vect, 0, h_max_particles * sizeof(float3));

    // Diagonal Coefficient for Pressure Matrix (aii)
    cudaMalloc(&d_diag_coeff, h_max_particles * sizeof(float));
    cudaMemset(d_diag_coeff, 0, h_max_particles * sizeof(float));

    // SUM_j(d_ij * p_j^l) buffer
    cudaMalloc(&d_sum_movement_pressure, h_max_particles * sizeof(float3));
    cudaMemset(d_sum_movement_pressure, 0, h_max_particles * sizeof(float3));

    // Static boundary positions and Psi_b. Allocated with at least one entry
    // so the kernels always receive valid pointers.
    const int boundary_capacity = std::max(h_max_boundary_particles, 1);
    cudaMalloc(&d_boundary_pos, boundary_capacity * sizeof(float3));
    cudaMalloc(&d_boundary_psi, boundary_capacity * sizeof(float));
    cudaMemset(d_boundary_pos, 0, boundary_capacity * sizeof(float3));
    cudaMemset(d_boundary_psi, 0, boundary_capacity * sizeof(float));

    // Fluid and boundary particles share the grid, so the hash buffers hold
    // both.
    const int hash_capacity = h_max_particles + h_max_boundary_particles;

    // Allocate hash key & value arrays
    cudaMalloc(&d_keys_in,
               hash_capacity * sizeof(unsigned int)); // Keys before sorting
    cudaMalloc(&d_keys_out,
               hash_capacity * sizeof(unsigned int)); // Keys after sorting
    cudaMalloc(&d_values_in,
               hash_capacity * sizeof(unsigned int)); // Values before sorting
    cudaMalloc(&d_values_out,
               hash_capacity * sizeof(unsigned int)); // Values after sorting

    // Allocate cell range lookup tables
    cudaMalloc(&d_cell_start_indices, h_max_grid_cells * sizeof(unsigned int));
    cudaMalloc(&d_cell_end_indices, h_max_grid_cells * sizeof(unsigned int));

    // Pre-allocate CUB Radix sort temp storage for fluid plus boundary
    cub::DeviceRadixSort::SortPairs(nullptr, h_cub_temp_storage_bytes,
                                    d_keys_in, d_keys_out, d_values_in,
                                    d_values_out, hash_capacity);
    cudaMalloc(&d_cub_temp_storage, h_cub_temp_storage_bytes);

    // Pre-allocate CUB Reduction temp storage and output scalar buffer for
    // convergence check
    cub::DeviceReduce::Sum(nullptr, h_cub_reduction_storage_bytes,
                           d_density_predicted, d_density_sum_output,
                           h_max_particles);
    cudaMalloc(&d_cub_reduction_storage, h_cub_reduction_storage_bytes);
    cudaMalloc(&d_density_sum_output, sizeof(float));

    // Outlet compaction buffers. Temp storage covers the float3 and float
    // arrays that are compacted.
    const int fluid_capacity = std::max(h_max_particles, 1);
    cudaMalloc(&d_keep_flags, fluid_capacity * sizeof(unsigned char));
    cudaMalloc(&d_num_selected, sizeof(int));
    size_t select_vec_bytes = 0;
    size_t select_scalar_bytes = 0;
    cub::DeviceSelect::Flagged(nullptr, select_vec_bytes, d_pos, d_keep_flags,
                               d_pos_next, d_num_selected, fluid_capacity);
    cub::DeviceSelect::Flagged(nullptr, select_scalar_bytes, d_pressure,
                               d_keep_flags, d_pressure_next, d_num_selected,
                               fluid_capacity);
    h_select_temp_storage_bytes = std::max(select_vec_bytes, select_scalar_bytes);
    cudaMalloc(&d_select_temp_storage, h_select_temp_storage_bytes);

    // Set flag that all pointers are allocated
    pointers_allocated = true;

    // Rest spacing from the particle volume V = m / rho_0.
    h_fluid_spacing = std::cbrt(h_params.particle_mass / h_params.rest_density);
    h_domain_max = make_float3(domain_max_x, domain_max_y, domain_max_z);
    h_pipe.reset();
    h_inflow_layer.clear();
    h_inflow_distance = 0.0f;
    h_capacity_warned = false;
    if (data.contains("pipe"))
        loadPipe(parsePipeSpec(data["pipe"]));
}

/* Builds the pipe wall as boundary particles and prepares the inlet. */
void FluidSimulation::loadPipe(const PipeSpec& spec) {
    const PipeWall wall = generatePipeWall(spec);
    for (const float3& p : wall.particles) {
        if (p.x < h_params.domain_min_x || p.x > h_domain_max.x ||
            p.y < h_params.domain_min_y || p.y > h_domain_max.y ||
            p.z < h_params.domain_min_z || p.z > h_domain_max.z) {
            throw std::runtime_error(
                "Pipe wall leaves the domain at (" + std::to_string(p.x) + ", " +
                std::to_string(p.y) + ", " + std::to_string(p.z) +
                "). Enlarge the domain, move the origin, or shrink the pipe.");
        }
    }
    uploadBoundaryParticles(wall.particles);

    h_pipe = spec;
    if (h_pipe->inflow.zone_length <= 0.0f)
        h_pipe->inflow.zone_length = 4.0f * h_fluid_spacing;

    // Keep fluid one rest spacing from the wall particle centers.
    const float inset = h_fluid_spacing;
    h_inflow_layer =
        generateCrossSection(spec, 0.5f * h_fluid_spacing, h_fluid_spacing, inset);
    h_active_particle_count = 0;

    if (spec.prefill) {
        const PipeFill fill = generatePipeFill(spec, h_fluid_spacing, inset);
        if (!appendFluidParticles(fill.positions, fill.velocities)) {
            throw std::runtime_error(
                "Prefilling the pipe needs " + std::to_string(fill.positions.size()) +
                " fluid particles, more than max_particles.");
        }
    }
}

/* Appends fluid particles after the active ones. Returns false if they do
 * not fit. */
bool FluidSimulation::appendFluidParticles(const std::vector<float3>& positions,
                                           const std::vector<float3>& velocities) {
    const int count = activeFluidCount();
    const int added = static_cast<int>(positions.size());
    if (added == 0)
        return true;
    if (count + added > h_max_particles)
        return false;

    cudaMemcpy(d_pos + count, positions.data(), added * sizeof(float3),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel + count, velocities.data(), added * sizeof(float3),
               cudaMemcpyHostToDevice);
    cudaMemset(d_pressure_prev + count, 0, added * sizeof(float));
    h_active_particle_count = count + added;
    return true;
}

/* Spawns a new inlet layer each time the inlet fluid has moved one rest
 * spacing, so emitted layers stay one spacing apart. */
void FluidSimulation::emitInflow(float delta_time) {
    if (!h_pipe || h_pipe->inflow.speed <= 0.0f || h_inflow_layer.empty())
        return;

    const float speed = h_pipe->inflow.speed;
    const float3 velocity = make_float3(h_pipe->axis.x * speed,
                                        h_pipe->axis.y * speed,
                                        h_pipe->axis.z * speed);
    const std::vector<float3> velocities(h_inflow_layer.size(), velocity);

    h_inflow_distance += speed * delta_time;
    while (h_inflow_distance >= h_fluid_spacing) {
        h_inflow_distance -= h_fluid_spacing;
        if (!appendFluidParticles(h_inflow_layer, velocities) && !h_capacity_warned) {
            std::cerr << "Inlet paused: max_particles reached.\n";
            h_capacity_warned = true;
        }
    }
}

/* Removes fluid past the outlet or outside the domain, keeping the order of
 * the remaining particles. Scratch buffers are swapped in as the outputs. */
void FluidSimulation::removeOutflowParticles() {
    const int count = activeFluidCount();
    if (!h_pipe || count <= 0)
        return;

    const int gridSize = (count + kBlockSize - 1) / kBlockSize;
    markOutflowKernel<<<gridSize, kBlockSize>>>(
        d_pos, d_keep_flags, count, h_pipe->origin, h_pipe->axis,
        pipeLength(*h_pipe), h_domain_max, h_params);

    const auto compact = [&](auto*& data, auto*& scratch) {
        cub::DeviceSelect::Flagged(d_select_temp_storage,
                                   h_select_temp_storage_bytes, data, d_keep_flags,
                                   scratch, d_num_selected, count);
        std::swap(data, scratch);
    };
    // d_pos_next, d_vel_adv, d_density_adv, and d_pressure_next are all
    // rewritten before they are read in the next step.
    compact(d_pos, d_pos_next);
    compact(d_vel, d_vel_adv);
    compact(d_density, d_density_adv);
    compact(d_pressure, d_pressure_next);
    compact(d_pressure_prev, d_pressure_next);

    int kept = 0;
    cudaMemcpy(&kept, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost);
    h_active_particle_count = kept;
}

/* Update function for fluid simulation. */
void FluidSimulation::update(float delta_time) {
    // New inlet layers join this step.
    emitInflow(delta_time);

    // Calculate grid size based on current fluid particle count. Boundary
    // particles are only neighbors: they are never integrated.
    const int current_particles = activeFluidCount();
    if (current_particles <= 0)
        return;

    int gridSize = (current_particles + kBlockSize - 1) / kBlockSize;
    const BoundaryParticles boundary = boundaryView();

    // Get latest spatial hashes for fluid and boundary particles.
    // Sorts the keys and values using CUB Radix sort.
    updateSpatialHash();

    // Compute density, including boundary contributions (Eq. (14))
    computeDensityKernel<<<gridSize, kBlockSize>>>(
        d_values_out, d_pos, boundary, d_cell_start_indices, d_cell_end_indices,
        d_density, current_particles, h_params);

    // Compute Advection and Gradient
    computeAdvectionAndGradientKernel<<<gridSize, kBlockSize>>>(
        d_pos, boundary, d_vel, nullptr, d_vel_adv, d_density, d_grad_coup_vect,
        d_values_out, d_cell_start_indices, d_cell_end_indices,
        current_particles, h_params, delta_time);

    // Initialize Pressure Solver
    // Calculates density adv, initial pressure, and a_ii
    initPressureSolverKernel<<<gridSize, kBlockSize>>>(
        d_pos, boundary, d_vel_adv, d_density, d_density_adv, d_grad_coup_vect,
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

        // Update pressure into a separate buffer so every particle reads p^l.
        updatePressureKernel<<<gridSize, kBlockSize>>>(
            d_pos, boundary, d_density, d_density_adv, d_diag_coeff, d_pressure,
            d_pressure_next, d_values_out, d_cell_start_indices,
            d_cell_end_indices, d_sum_movement_pressure, d_grad_coup_vect,
            current_particles, h_params, delta_time);
        cudaMemcpy(d_pressure, d_pressure_next,
                   current_particles * sizeof(float), cudaMemcpyDeviceToDevice);

        // Update density from the pressure assigned for this iteration.
        updateDensityKernel<<<gridSize, kBlockSize>>>(
            d_pos, boundary, d_density, d_density_adv, d_diag_coeff, d_pressure,
            d_density_predicted, d_values_out, d_cell_start_indices,
            d_cell_end_indices, d_sum_movement_pressure, d_grad_coup_vect,
            current_particles, h_params, delta_time);

        // Compute average predicted fluid density via CUB parallel reduction
        cub::DeviceReduce::Sum(
            d_cub_reduction_storage, h_cub_reduction_storage_bytes,
            d_density_predicted, d_density_sum_output, current_particles);

        float h_density_sum = 0.0f;
        cudaMemcpy(&h_density_sum, d_density_sum_output, sizeof(float),
                   cudaMemcpyDeviceToHost);
        density_avg_l = h_density_sum / static_cast<float>(current_particles);

        l++;
    }

    // v(t+dt) = v_adv + dt * F_pressure / m, then x(t+dt) = x + dt * v(t+dt).
    // Thesis p. 14, Eq. (4.12), Ihmsen Eq. (15), and Algorithm 1 lines 21–22.
    // Only fluid positions are swapped; boundary positions stay fixed.
    integrateKernel<<<gridSize, kBlockSize>>>(
        d_pos, boundary, d_pos_next, d_vel, d_vel_adv, d_density, d_pressure,
        d_values_out, d_cell_start_indices, d_cell_end_indices,
        current_particles, h_params, delta_time);
    float3* integrated_positions = d_pos;
    d_pos = d_pos_next;
    d_pos_next = integrated_positions;

    // Pipe inlet: fluid that started in the zone moves at the inflow speed.
    // d_pos_next now holds the positions from before integration.
    if (h_pipe && h_pipe->inflow.speed > 0.0f) {
        applyInflowKernel<<<gridSize, kBlockSize>>>(
            d_pos, d_pos_next, d_vel, current_particles, h_pipe->origin,
            h_pipe->axis, h_pipe->inflow.zone_length, h_pipe->inflow.speed,
            delta_time);
    }

    // Save current pressure as previous pressure for the next frame
    cudaMemcpy(d_pressure_prev, d_pressure, h_max_particles * sizeof(float),
               cudaMemcpyDeviceToDevice);

    // Pipe outlet: drop fluid that has left the pipe or the domain.
    removeOutflowParticles();

    // Check if an error occurred while launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err)
                  << "\n";
    }
}

/* Get new spatial hashes for fluid and boundary particles. */
void FluidSimulation::updateSpatialHash() {

    // Fluid particles use ids [0, fluid) and boundary particles
    // [fluid, fluid + boundary), so one sorted grid serves both.
    const int fluid_particles = activeFluidCount();
    const int boundary_particles = h_boundary_particle_count;
    const int total_particles = fluid_particles + boundary_particles;
    if (total_particles <= 0)
        return;

    int gridSize = (total_particles + kBlockSize - 1) / kBlockSize;

    // Compute hash for each particle. Updates keys and values arrays.
    // This gets two arrays. One for keys, which are the cell codes of each
    // particle. The other for values, which are the particle ids.
    computeHashKernel<<<gridSize, kBlockSize>>>(
        d_pos, d_boundary_pos, d_keys_in, d_values_in, fluid_particles,
        boundary_particles, h_params);

    // Sort using CUB Radix sort. This ensures that the key value pairs
    // stay together after sort. End goal is to have all identical cell
    // keys grouped together.
    cub::DeviceRadixSort::SortPairs(
        d_cub_temp_storage, h_cub_temp_storage_bytes, d_keys_in, d_keys_out,
        d_values_in, d_values_out, total_particles);

    // Clear values in index arrays
    cudaMemset(d_cell_start_indices, 0xFF,
               h_max_grid_cells * sizeof(unsigned int));
    cudaMemset(d_cell_end_indices, 0xFF,
               h_max_grid_cells * sizeof(unsigned int));

    // Get new start and end index for each cell.
    findCellRangesKernel<<<gridSize, kBlockSize>>>(
        d_keys_out, d_cell_start_indices, d_cell_end_indices, total_particles);

    // Boundary particles are static, so Psi_b only changes when they are
    // re-uploaded. Ihmsen et al., p. 5, Section 4.
    if (h_boundary_psi_dirty && boundary_particles > 0) {
        const int boundaryGrid = (boundary_particles + kBlockSize - 1) / kBlockSize;
        computeBoundaryPsiKernel<<<boundaryGrid, kBlockSize>>>(
            d_boundary_pos, d_boundary_psi, d_values_out, d_cell_start_indices,
            d_cell_end_indices, fluid_particles, boundary_particles, h_params);
    }
    h_boundary_psi_dirty = false;
}

/* Replaces the static boundary particles. */
void FluidSimulation::uploadBoundaryParticles(
    const std::vector<float3>& positions) {
    if (positions.size() > static_cast<size_t>(h_max_boundary_particles)) {
        throw std::runtime_error(
            "Boundary particle count exceeds max_boundary_particles defined in "
            "config.");
    }

    h_boundary_particle_count = static_cast<int>(positions.size());
    h_boundary_psi_dirty = true;
    if (positions.empty())
        return;

    cudaError_t err =
        cudaMemcpy(d_boundary_pos, positions.data(),
                   positions.size() * sizeof(float3), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to upload boundary particles: ") +
            cudaGetErrorString(err));
    }
}

/* Deallocate pointers before finishing program. */
void FluidSimulation::cleanup() {
    if (!pointers_allocated)
        return;

    // Finish in-flight kernels before the buffers they read are released.
    cudaDeviceSynchronize();

    cudaFree(d_pos);
    cudaFree(d_pos_next);
    cudaFree(d_vel);
    cudaFree(d_vel_adv);
    cudaFree(d_density);
    cudaFree(d_density_adv);
    cudaFree(d_density_predicted);
    cudaFree(d_pressure);
    cudaFree(d_pressure_prev);
    cudaFree(d_pressure_next);
    cudaFree(d_grad_coup_vect);
    cudaFree(d_diag_coeff);
    cudaFree(d_sum_movement_pressure);
    cudaFree(d_boundary_pos);
    cudaFree(d_boundary_psi);
    cudaFree(d_keep_flags);
    cudaFree(d_num_selected);
    cudaFree(d_select_temp_storage);
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
    d_pos = d_pos_next = d_vel = d_vel_adv = d_grad_coup_vect = nullptr;
    d_density = d_density_adv = d_density_predicted = nullptr;
    d_pressure = d_pressure_prev = d_pressure_next = d_diag_coeff = nullptr;
    d_sum_movement_pressure = nullptr;
    d_boundary_pos = nullptr;
    d_boundary_psi = nullptr;
    h_boundary_particle_count = 0;
    h_boundary_psi_dirty = false;
    d_keep_flags = nullptr;
    d_num_selected = nullptr;
    d_select_temp_storage = nullptr;
    h_pipe.reset();
    h_inflow_layer.clear();
    d_keys_in = d_keys_out = d_values_in = d_values_out = nullptr;
    d_cell_start_indices = d_cell_end_indices = nullptr;
    d_cub_temp_storage = d_cub_reduction_storage = nullptr;
    d_density_sum_output = nullptr;

    pointers_allocated = false;
}

/* Debug function: Uploads test particles for the simulation. */
void FluidSimulation::uploadTestParticles(const std::vector<float3>& pts) {
    if (pts.size() > static_cast<size_t>(h_max_particles)) {
        throw std::runtime_error(
            "Test particle count exceeds max_particles defined in config.");
    }

    // Set active particle count to match the test batch size
    h_active_particle_count = static_cast<int>(pts.size());

    cudaError_t err = cudaMemcpy(d_pos, pts.data(), pts.size() * sizeof(float3),
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to upload test particles: ") +
            cudaGetErrorString(err));
    }

    // Rest velocities unless a test uploads its own.
    cudaMemset(d_vel, 0, pts.size() * sizeof(float3));
}

/* Debug function: Uploads velocities for the active test particles. */
void FluidSimulation::uploadTestVelocities(const std::vector<float3>& velocities) {
    if (velocities.size() != static_cast<size_t>(h_active_particle_count)) {
        throw std::runtime_error(
            "Test velocity count does not match the uploaded particle count.");
    }

    cudaError_t err =
        cudaMemcpy(d_vel, velocities.data(), velocities.size() * sizeof(float3),
                   cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to upload test velocities: ") +
            cudaGetErrorString(err));
    }
}

/* Debug function: Copies the sorted keys of fluid and boundary particles. */
std::vector<unsigned int> FluidSimulation::copyKeysToHost() {
    const int total = activeFluidCount() + h_boundary_particle_count;
    std::vector<unsigned int> host_keys(total);
    if (total == 0)
        return host_keys;
    cudaError_t err = cudaMemcpy(host_keys.data(), d_keys_out,
                                 total * sizeof(unsigned int),
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

/* Debug function: Copies integrated positions from device to host. */
std::vector<float3> FluidSimulation::copyPositionsToHost() const {
    std::vector<float3> h_pos(h_active_particle_count);
    if (d_pos != nullptr && h_active_particle_count > 0) {
        cudaMemcpy(h_pos.data(), d_pos, h_active_particle_count * sizeof(float3),
                   cudaMemcpyDeviceToHost);
    }
    return h_pos;
}

/* Debug function: Copies integrated velocities from device to host. */
std::vector<float3> FluidSimulation::copyVelocitiesToHost() const {
    std::vector<float3> h_vel(h_active_particle_count);
    if (d_vel != nullptr && h_active_particle_count > 0) {
        cudaMemcpy(h_vel.data(), d_vel, h_active_particle_count * sizeof(float3),
                   cudaMemcpyDeviceToHost);
    }
    return h_vel;
}

/* Debug function: Copies the Velocity Adv array from device to host. */
std::vector<float3> FluidSimulation::copyVelAdvToHost() const {
    std::vector<float3> h_vel_adv(h_active_particle_count);
    if (d_vel_adv != nullptr && h_active_particle_count > 0) {
        cudaMemcpy(h_vel_adv.data(), d_vel_adv,
                   h_active_particle_count * sizeof(float3),
                   cudaMemcpyDeviceToHost);
    }
    return h_vel_adv;
}

/* Debug function: Copies the Gradient Coupling Vector from device to host. */
std::vector<float3> FluidSimulation::copyGradCoupToHost() const {
    std::vector<float3> h_grad_coup(h_active_particle_count);
    if (d_grad_coup_vect != nullptr && h_active_particle_count > 0) {
        cudaMemcpy(h_grad_coup.data(), d_grad_coup_vect,
                   h_active_particle_count * sizeof(float3),
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

/* Debug function: Copies the static boundary positions from device to host. */
std::vector<float3> FluidSimulation::copyBoundaryPositionsToHost() const {
    std::vector<float3> h_boundary_pos(h_boundary_particle_count);
    if (h_boundary_particle_count > 0 && d_boundary_pos) {
        cudaMemcpy(h_boundary_pos.data(), d_boundary_pos,
                   h_boundary_particle_count * sizeof(float3),
                   cudaMemcpyDeviceToHost);
    }
    return h_boundary_pos;
}

/* Debug function: Copies Psi_b(rho_0) from device to host. */
std::vector<float> FluidSimulation::copyBoundaryPsiToHost() const {
    std::vector<float> h_boundary_psi(h_boundary_particle_count);
    if (h_boundary_particle_count > 0 && d_boundary_psi) {
        cudaMemcpy(h_boundary_psi.data(), d_boundary_psi,
                   h_boundary_particle_count * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    return h_boundary_psi;
}