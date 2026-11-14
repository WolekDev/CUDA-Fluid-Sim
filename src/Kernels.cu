#include "Kernels.cuh"
#include <math_constants.h>

/* Compute hashes for the particles. Store in key value arrays. */
__global__ void computeHashKernel(const float2* __restrict__ d_pos,
                                  unsigned int* __restrict__ d_keys,
                                  unsigned int* __restrict__ d_values,
                                  int num_particles, SimulationParams params) {
    // Get particle's unique index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles)
        return;

    // Extract particle position
    const float2 pos = d_pos[idx];

    // Find the cell coordinates of the particle
    const int cx = static_cast<int>(
        floorf((pos.x - params.domain_min_x) / params.smoothing_radius));
    const int cy = static_cast<int>(
        floorf((pos.y - params.domain_min_y) / params.smoothing_radius));

    // Create hash key for the particle
    const unsigned int hx = static_cast<unsigned int>(cx);
    const unsigned int hy = static_cast<unsigned int>(cy);
    const unsigned int key = hy * params.grid_width + hx;

    // Update keys and values arrays with new data
    d_keys[idx] = key;
    d_values[idx] = static_cast<unsigned int>(idx);
}

/* Find start and end indexes of each cell. */
__global__ void
findCellRangesKernel(const unsigned int* __restrict__ d_sorted_keys,
                     unsigned int* __restrict__ d_cell_start_indices,
                     unsigned int* __restrict__ d_cell_end_indices,
                     int num_particles) {
    // Get particle's unique index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles)
        return;

    // Get paricle's cell key
    const unsigned int current_key = d_sorted_keys[idx];

    // Find the point in the keys array where a different cell is located.
    // Update start and end position of the range.
    if (idx == 0) {
        d_cell_start_indices[current_key] = 0;
    } else {
        const unsigned int prev_key = d_sorted_keys[idx - 1];
        if (current_key != prev_key) {
            d_cell_start_indices[current_key] = idx;
            d_cell_end_indices[prev_key] = idx;
        }
    }

    if (idx == num_particles - 1) {
        d_cell_end_indices[current_key] = num_particles;
    }
}

/* Compute density for SPH  */
__global__ void
computeDensityKernel(const unsigned int* __restrict__ d_sorted_values,
                     const float2* __restrict__ d_pos,
                     const unsigned int* __restrict__ d_cell_start_indices,
                     const unsigned int* __restrict__ d_cell_end_indices,
                     float* __restrict__ d_density, int num_particles,
                     SimulationParams params) {
    // Get particle's unique index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles)
        return;

    // Extract particle's position
    const float2 pos = d_pos[idx];

    // Get cell coordinates for particle
    const int cx = static_cast<int>(
        floorf((pos.x - params.domain_min_x) / params.smoothing_radius));
    const int cy = static_cast<int>(
        floorf((pos.y - params.domain_min_y) / params.smoothing_radius));

    // Smoothing radius for furture calculations
    const float h = params.smoothing_radius;
    const float h2 = h * h;

    float density = 0.0f;

    // Loop through neighboring cells on x-axis
    for (int i = cx - 1; i <= cx + 1; i++) {
        // Ensure x bounds
        if (i < 0 || i >= params.grid_width)
            continue;

        // Loop through neighboring cells on y-axis
        for (int j = cy - 1; j <= cy + 1; j++) {
            // Ensure y bounds
            if (j < 0 || j >= params.grid_height)
                continue;

            // Get hash key for the target cell
            const unsigned int hi = static_cast<unsigned int>(i);
            const unsigned int hj = static_cast<unsigned int>(j);
            const unsigned int key = hj * params.grid_width + hi;

            // Get start and end index for particles in this cell
            const unsigned int start_index = d_cell_start_indices[key];
            const unsigned int end_index = d_cell_end_indices[key];

            // Ensure that the start index is before the end and that the
            // indexes are not default value (unmodified)
            if (start_index > end_index || start_index == 0xFFFFFFFFu ||
                end_index == 0xFFFFFFFFu)
                continue;

            // Loop through all particles in cell range
            for (unsigned int k = start_index; k < end_index; ++k) {
                // Get neighbor particle index and position
                const unsigned int neighbor_particle_idx = d_sorted_values[k];
                const float2 neighbor_pos = d_pos[neighbor_particle_idx];

                // Calculate distance to main particle
                const float2 rij =
                    make_float2(pos.x - neighbor_pos.x, pos.y - neighbor_pos.y);
                const float r2 = rij.x * rij.x + rij.y * rij.y;

                /* Check if neighbor is in range of particle based on smoothing
                   radius. If neighbor is in radius, Use 2D Poly6 smoothing
                   function to calculate impact of neighbor on particle. */
                // h = smoothing radius,
                // r = distance between particle and neighbor,
                // q = ratio of how far neighbor is from particle,
                // w = Poly6 kernel weight value
                if (r2 <= h2) {
                    const float r = sqrtf(r2);
                    const float q = (h - r) / h;
                    const float w =
                        315.0f / (64.0f * CUDART_PI_F * h * h * h) * q * q * q;
                    density += params.particle_mass * w;
                }
            }
        }
    }

    // Update particle's density
    d_density[idx] = density;
}

/* Helper kernel for calculating the Spiky gradient */
__device__ inline float2 computeSpikyGradient(float2 r_ij, float r, float h) {
    if (r <= 0.0f || r >= h)
        return make_float2(0.0f, 0.0f);

    // Gradient formula: -45 / (pi * h^6) * (h - r)^2 * (r_ij / r)
    const float factor =
        -45.0f / (CUDART_PI_F * powf(h, 6.0f)) * powf(h - r, 2.0f);
    const float2 normalized_r = make_float2(r_ij.x / r, r_ij.y / r);

    return make_float2(factor * normalized_r.x, factor * normalized_r.y);
}

/* Compute Advection and Gradient */
__global__ void computeAdvectionAndGradientKernel(
    const float2* __restrict__ d_pos, const float2* __restrict__ d_vel,
    const float2* __restrict__ d_force_ext, float2* __restrict__ d_vel_adv,
    const float* __restrict__ d_density, float2* __restrict__ d_grad_coup_vect,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices, int num_particles,
    SimulationParams params, float delta_time) {

    // Get particle's unique index
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_particles)
        return;

    // Extract pos and vel of particle
    const unsigned int p_idx = d_sorted_values[idx];
    const float2 pos_i = d_pos[p_idx];
    const float2 vel_i = d_vel[p_idx];

    // Compute advective velocity:
    // v_i^adv = v_i + dt * (F_ext / mass)
    // Assuming d_force_ext exists, or compute from gravity:
    // F = mass * g
    const float2 f_ext =
        d_force_ext
            ? d_force_ext[p_idx]
            : make_float2(0.0f, params.gravity_vector.y * params.particle_mass);
    const float2 vel_adv =
        make_float2(vel_i.x + delta_time * (f_ext.x / params.particle_mass),
                    vel_i.y + delta_time * (f_ext.y / params.particle_mass));
    d_vel_adv[p_idx] = vel_adv;

    // Calculate gradient coupling vector d_ii
    // d_ii = - (dt^2) * sum_j ( (m_j / rho_i^2) * grad_W_ij )
    float2 d_ii_sum = make_float2(0.0f, 0.0f);
    const float density_i = d_density[p_idx];

    // Get cell coordinates for particle
    const int cx = static_cast<int>(
        floorf((pos_i.x - params.domain_min_x) / params.smoothing_radius));
    const int cy = static_cast<int>(
        floorf((pos_i.y - params.domain_min_y) / params.smoothing_radius));

    // Smothing radius for future calculations
    const float h = params.smoothing_radius;
    const float h2 = h * h;

    // Loop through neighboring cells on x-axis
    for (int i = cx - 1; i <= cx + 1; i++) {
        // Ensure x bounds
        if (i < 0 || i >= params.grid_width)
            continue;

        // Loop through neighboring cells on y-axis
        for (int j = cy - 1; j <= cy + 1; j++) {
            // Ensure y bounds
            if (j < 0 || j >= params.grid_height)
                continue;

            // Get hash key for the target cell
            const unsigned int hi = static_cast<unsigned int>(i);
            const unsigned int hj = static_cast<unsigned int>(j);
            const unsigned int key = hj * params.grid_width + hi;

            // Get start and end index for particles in this cell
            const unsigned int start_index = d_cell_start_indices[key];
            const unsigned int end_index = d_cell_end_indices[key];

            // Ensure that the start index is before the end and that the
            // indexes are not default value (unmodified)
            if (start_index > end_index || start_index == 0xFFFFFFFFu ||
                end_index == 0xFFFFFFFFu)
                continue;

            // Loop through all particles in cell range
            for (unsigned int k = start_index; k < end_index; ++k) {

                // Get neighbor position
                const unsigned int neighbor_idx = d_sorted_values[k];
                const float2 pos_j = d_pos[neighbor_idx];

                // Calculate distance between main particle and neighbor
                const float2 r_ij =
                    make_float2(pos_i.x - pos_j.x, pos_i.y - pos_j.y);
                const float r2 = r_ij.x * r_ij.x + r_ij.y * r_ij.y;

                // If neighbor in range of main, add its contribution to d_ii,
                // the gradient coupling vector.
                // Spiky kernel gradient is used here as Poly6 does not have
                // repulsion as r approaches 0. Spiky prevents particle overlap.
                // r = distance between neighbor and main particle
                // h = smoothing radius
                if (r2 > 0.0f && r2 < h2) {
                    const float r = sqrtf(r2);
                    const float2 grad_w_ij = computeSpikyGradient(r_ij, r, h);

                    // Calculate the gradient coupling vector
                    // d_ii += m_j / rho_i^2 * grad_W_ij
                    const float scalar =
                        params.particle_mass / (density_i * density_i);
                    d_ii_sum.x += scalar * grad_w_ij.x;
                    d_ii_sum.y += scalar * grad_w_ij.y;
                }
            }
        }
    }

    // Finalize d_ii: multiply by -(dt^2)
    float dt_sq = delta_time * delta_time;
    d_grad_coup_vect[p_idx] =
        make_float2(-dt_sq * d_ii_sum.x, -dt_sq * d_ii_sum.y);
}
