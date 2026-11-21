#include "Kernels.cuh"

namespace {

__device__ inline int cellCoordinate(float position, float domain_min,
                                     float cell_size) {
    return static_cast<int>(floorf((position - domain_min) / cell_size));
}

__device__ inline int clampCell(int cell, int count) {
    if (cell < 0)
        return 0;
    if (cell >= count)
        return count - 1;
    return cell;
}

__device__ inline bool cellRange(const unsigned int* __restrict__ starts,
                                 const unsigned int* __restrict__ ends,
                                 unsigned int key, unsigned int& start,
                                 unsigned int& end) {
    start = starts[key];
    end = ends[key];
    return start <= end && start != 0xFFFFFFFFu && end != 0xFFFFFFFFu;
}

/* d_ji = -dt^2 * (m_i / rho_i^2) * grad W_ji
 * Thesis p. 13, Eq. (4.11); Ihmsen et al., p. 4, Eq. (9) with i and j swapped. */
__device__ inline float3 pairwiseDisplacementFromI(float3 grad_w_ji, float mass,
                                                   float density_i, float dt2) {
    const float scale = -dt2 * mass / (density_i * density_i);
    return make_float3(scale * grad_w_ji.x, scale * grad_w_ji.y,
                       scale * grad_w_ji.z);
}

} // namespace

/* Compute hashes for the particles. Store in key value arrays. */
__global__ void computeHashKernel(const float3* __restrict__ d_pos,
                                  unsigned int* __restrict__ d_keys,
                                  unsigned int* __restrict__ d_values,
                                  int num_particles, SimulationParams params) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles)
        return;

    const float3 pos = d_pos[idx];
    const float cell_size = params.smoothing_radius;

    int cx = cellCoordinate(pos.x, params.domain_min_x, cell_size);
    int cy = cellCoordinate(pos.y, params.domain_min_y, cell_size);
    int cz = cellCoordinate(pos.z, params.domain_min_z, cell_size);
    cx = clampCell(cx, params.grid_width);
    cy = clampCell(cy, params.grid_height);
    cz = clampCell(cz, params.grid_depth);

    d_keys[idx] = computeCellKey(cx, cy, cz, params.grid_width, params.grid_height);
    d_values[idx] = static_cast<unsigned int>(idx);
}

/* Find start and end indexes of each cell. */
__global__ void
findCellRangesKernel(const unsigned int* __restrict__ d_sorted_keys,
                     unsigned int* __restrict__ d_cell_start_indices,
                     unsigned int* __restrict__ d_cell_end_indices,
                     int num_particles) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles)
        return;

    const unsigned int current_key = d_sorted_keys[idx];

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

/* rho_i(t) = sum_j m_j W_poly6(|x_i - x_j|, R)
 * Ihmsen et al., IISPH, TVCG author's version, p. 2;
 * thesis p. 8, Algorithm 1 line 3. W_poly6 is thesis p. 10, Eq. (4.1). */
__global__ void
computeDensityKernel(const unsigned int* __restrict__ d_sorted_values,
                     const float3* __restrict__ d_pos,
                     const unsigned int* __restrict__ d_cell_start_indices,
                     const unsigned int* __restrict__ d_cell_end_indices,
                     float* __restrict__ d_density, int num_particles,
                     SimulationParams params) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles)
        return;

    const float3 pos = d_pos[idx];
    const float support = params.smoothing_radius;
    const float support2 = support * support;
    const int cx = cellCoordinate(pos.x, params.domain_min_x, support);
    const int cy = cellCoordinate(pos.y, params.domain_min_y, support);
    const int cz = cellCoordinate(pos.z, params.domain_min_z, support);

    float density = 0.0f;

    // Support radius equals the cell size, so only the 3x3x3 block can contribute.
    for (int i = cx - 1; i <= cx + 1; ++i) {
        if (i < 0 || i >= params.grid_width)
            continue;
        for (int j = cy - 1; j <= cy + 1; ++j) {
            if (j < 0 || j >= params.grid_height)
                continue;
            for (int k = cz - 1; k <= cz + 1; ++k) {
                if (k < 0 || k >= params.grid_depth)
                    continue;

                const unsigned int key = computeCellKey(i, j, k, params.grid_width,
                                                        params.grid_height);
                unsigned int start_index = 0;
                unsigned int end_index = 0;
                if (!cellRange(d_cell_start_indices, d_cell_end_indices, key,
                               start_index, end_index))
                    continue;

                for (unsigned int n = start_index; n < end_index; ++n) {
                    const unsigned int neighbor_idx = d_sorted_values[n];
                    const float3 neighbor_pos = d_pos[neighbor_idx];
                    const float3 rij =
                        make_float3(pos.x - neighbor_pos.x, pos.y - neighbor_pos.y,
                                    pos.z - neighbor_pos.z);
                    const float r2 = dot3(rij, rij);
                    if (r2 <= support2) {
                        // Eq. (4.1). Self term is W(0), included when r = 0.
                        density += params.particle_mass *
                                   computePoly6(sqrtf(r2), support);
                    }
                }
            }
        }
    }

    d_density[idx] = density;
}

/* F_i^adv = m_i g + F_i^visc, then v_i^adv = v_i + dt * F_i^adv / m_i.
 * F_i^visc: thesis p. 11, Eq. (4.4). d_ii: Ihmsen Algorithm 1, p. 3. */
__global__ void computeAdvectionAndGradientKernel(
    const float3* __restrict__ d_pos, const float3* __restrict__ d_vel,
    const float3* __restrict__ d_force_ext, float3* __restrict__ d_vel_adv,
    const float* __restrict__ d_density, float3* __restrict__ d_grad_coup_vect,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices, int num_particles,
    SimulationParams params, float delta_time) {

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_particles)
        return;

    const unsigned int p_idx = d_sorted_values[idx];
    const float3 pos_i = d_pos[p_idx];
    const float3 vel_i = d_vel[p_idx];
    const float mass = params.particle_mass;

    const float density_i = d_density[p_idx];
    float3 d_ii_sum = make_float3(0.0f, 0.0f, 0.0f);
    float3 f_visc = make_float3(0.0f, 0.0f, 0.0f);
    if (density_i > 0.0f) {
        const float support = params.smoothing_radius;
        const float support2 = support * support;
        const int cx = cellCoordinate(pos_i.x, params.domain_min_x, support);
        const int cy = cellCoordinate(pos_i.y, params.domain_min_y, support);
        const int cz = cellCoordinate(pos_i.z, params.domain_min_z, support);
        const float density_scale = mass / (density_i * density_i);
        // Eq. (4.4): (m_j / rho_i)^2 * mu, with mu in Pa·s.
        const float mass_over_density = mass / density_i;
        const float visc_scale = mass_over_density * mass_over_density *
                                 dynamicViscosityPascalSeconds(params.viscosity);

        for (int i = cx - 1; i <= cx + 1; ++i) {
            if (i < 0 || i >= params.grid_width)
                continue;
            for (int j = cy - 1; j <= cy + 1; ++j) {
                if (j < 0 || j >= params.grid_height)
                    continue;
                for (int k = cz - 1; k <= cz + 1; ++k) {
                    if (k < 0 || k >= params.grid_depth)
                        continue;

                    const unsigned int key = computeCellKey(
                        i, j, k, params.grid_width, params.grid_height);
                    unsigned int start_index = 0;
                    unsigned int end_index = 0;
                    if (!cellRange(d_cell_start_indices, d_cell_end_indices, key,
                                   start_index, end_index))
                        continue;

                    for (unsigned int n = start_index; n < end_index; ++n) {
                        const unsigned int neighbor_idx = d_sorted_values[n];
                        const float3 pos_j = d_pos[neighbor_idx];
                        const float3 r_ij =
                            make_float3(pos_i.x - pos_j.x, pos_i.y - pos_j.y,
                                        pos_i.z - pos_j.z);
                        const float r2 = dot3(r_ij, r_ij);
                        if (r2 >= support2)
                            continue;

                        const float distance = sqrtf(r2);
                        if (distance > 0.0f) {
                            const float3 grad_w_ij =
                                computeSpikyGradient(r_ij, distance, support);
                            d_ii_sum.x += density_scale * grad_w_ij.x;
                            d_ii_sum.y += density_scale * grad_w_ij.y;
                            d_ii_sum.z += density_scale * grad_w_ij.z;
                        }

                        // Eq. (4.4): -(v_i - v_j) * (m_j / rho_i)^2 * Const_visc * W
                        const float3 vel_j = d_vel[neighbor_idx];
                        const float weight = computeViscosity(distance, support);
                        const float coeff = visc_scale * weight;
                        f_visc.x += -(vel_i.x - vel_j.x) * coeff;
                        f_visc.y += -(vel_i.y - vel_j.y) * coeff;
                        f_visc.z += -(vel_i.z - vel_j.z) * coeff;
                    }
                }
            }
        }
    }

    // F_adv = F_ext + F_visc, then v_adv = v + dt * F_adv / m. Thesis p. 11.
    const float3 f_ext =
        d_force_ext ? d_force_ext[p_idx]
                    : make_float3(params.gravity_vector.x * mass,
                                  params.gravity_vector.y * mass,
                                  params.gravity_vector.z * mass);
    const float inv_mass = 1.0f / mass;
    d_vel_adv[p_idx] = make_float3(
        vel_i.x + delta_time * (f_ext.x + f_visc.x) * inv_mass,
        vel_i.y + delta_time * (f_ext.y + f_visc.y) * inv_mass,
        vel_i.z + delta_time * (f_ext.z + f_visc.z) * inv_mass);

    const float dt2 = delta_time * delta_time;
    d_grad_coup_vect[p_idx] =
        make_float3(-dt2 * d_ii_sum.x, -dt2 * d_ii_sum.y, -dt2 * d_ii_sum.z);
}

/* rho_i^adv from Ihmsen Eq. (4), p_i^0 = 0.5 p_i(t - dt), and
 * a_ii from Ihmsen Eq. (12) using d_ji from thesis Eq. (4.11). */
__global__ void initPressureSolverKernel(
    const float3* __restrict__ d_pos, const float3* __restrict__ d_vel_adv,
    const float* __restrict__ d_density, float* __restrict__ d_density_adv,
    const float3* __restrict__ d_grad_coup_vect,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices,
    float* __restrict__ d_pressure, const float* __restrict__ d_pressure_prev,
    float* __restrict__ d_diag_coeff, int num_particles,
    SimulationParams params, float delta_time) {

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_particles)
        return;

    const unsigned int p_idx = d_sorted_values[idx];
    const float3 pos_i = d_pos[p_idx];
    const float mass = params.particle_mass;
    const float support = params.smoothing_radius;
    const float support2 = support * support;
    const float density_i = d_density[p_idx];
    const float3 d_ii = d_grad_coup_vect[p_idx];
    const float3 vel_adv_i = d_vel_adv[p_idx];
    const float dt2 = delta_time * delta_time;

    // p_i^0 = 0.5 * p_i(t - dt). Ihmsen et al., p. 4; thesis p. 12.
    d_pressure[p_idx] = 0.5f * d_pressure_prev[p_idx];

    float density_adv_sum = 0.0f;
    float a_ii_sum = 0.0f;

    const int cx = cellCoordinate(pos_i.x, params.domain_min_x, support);
    const int cy = cellCoordinate(pos_i.y, params.domain_min_y, support);
    const int cz = cellCoordinate(pos_i.z, params.domain_min_z, support);

    for (int i = cx - 1; i <= cx + 1; ++i) {
        if (i < 0 || i >= params.grid_width)
            continue;
        for (int j = cy - 1; j <= cy + 1; ++j) {
            if (j < 0 || j >= params.grid_height)
                continue;
            for (int k = cz - 1; k <= cz + 1; ++k) {
                if (k < 0 || k >= params.grid_depth)
                    continue;

                const unsigned int key = computeCellKey(i, j, k, params.grid_width,
                                                        params.grid_height);
                unsigned int start_index = 0;
                unsigned int end_index = 0;
                if (!cellRange(d_cell_start_indices, d_cell_end_indices, key,
                               start_index, end_index))
                    continue;

                for (unsigned int n = start_index; n < end_index; ++n) {
                    const unsigned int neighbor_idx = d_sorted_values[n];
                    const float3 pos_j = d_pos[neighbor_idx];
                    const float3 r_ij = make_float3(pos_i.x - pos_j.x,
                                                    pos_i.y - pos_j.y,
                                                    pos_i.z - pos_j.z);
                    const float r2 = dot3(r_ij, r_ij);
                    if (r2 <= 0.0f || r2 >= support2 || density_i <= 0.0f)
                        continue;

                    const float distance = sqrtf(r2);
                    const float3 grad_w_ij =
                        computeSpikyGradient(r_ij, distance, support);
                    const float3 r_ji =
                        make_float3(-r_ij.x, -r_ij.y, -r_ij.z);
                    const float3 grad_w_ji =
                        computeSpikyGradient(r_ji, distance, support);

                    // Eq. (4): m_j (v_i^adv - v_j^adv) · grad W_ij
                    const float3 vel_adv_j = d_vel_adv[neighbor_idx];
                    const float3 relative_vel =
                        make_float3(vel_adv_i.x - vel_adv_j.x,
                                    vel_adv_i.y - vel_adv_j.y,
                                    vel_adv_i.z - vel_adv_j.z);
                    density_adv_sum += mass * dot3(relative_vel, grad_w_ij);

                    // Eq. (12): m_j (d_ii - d_ji) · grad W_ij
                    const float3 d_ji = pairwiseDisplacementFromI(
                        grad_w_ji, mass, density_i, dt2);
                    const float3 difference =
                        make_float3(d_ii.x - d_ji.x, d_ii.y - d_ji.y,
                                    d_ii.z - d_ji.z);
                    a_ii_sum += mass * dot3(difference, grad_w_ij);
                }
            }
        }
    }

    d_density_adv[p_idx] = d_density[p_idx] + delta_time * density_adv_sum;
    d_diag_coeff[p_idx] = a_ii_sum;
}

/* sum_j d_ij p_j^l = dt^2 * sum_j (-m_j / rho_j^2) p_j^l grad W_ij
 * Ihmsen et al., p. 3, Algorithm 1; thesis p. 8 line 14. */
__global__ void calculateSumMovementFromPressureKernel(
    const float3* __restrict__ d_pos, const float* __restrict__ d_density,
    const float* __restrict__ d_pressure,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices,
    float3* __restrict__ d_sum_movement_pressure, int num_particles,
    SimulationParams params, float delta_time) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles)
        return;

    const float3 pos = d_pos[idx];
    const float support = params.smoothing_radius;
    const float support2 = support * support;
    const float mass = params.particle_mass;
    const int cx = cellCoordinate(pos.x, params.domain_min_x, support);
    const int cy = cellCoordinate(pos.y, params.domain_min_y, support);
    const int cz = cellCoordinate(pos.z, params.domain_min_z, support);

    float3 sum = make_float3(0.0f, 0.0f, 0.0f);

    for (int i = cx - 1; i <= cx + 1; ++i) {
        if (i < 0 || i >= params.grid_width)
            continue;
        for (int j = cy - 1; j <= cy + 1; ++j) {
            if (j < 0 || j >= params.grid_height)
                continue;
            for (int k = cz - 1; k <= cz + 1; ++k) {
                if (k < 0 || k >= params.grid_depth)
                    continue;

                const unsigned int key = computeCellKey(i, j, k, params.grid_width,
                                                        params.grid_height);
                unsigned int start_index = 0;
                unsigned int end_index = 0;
                if (!cellRange(d_cell_start_indices, d_cell_end_indices, key,
                               start_index, end_index))
                    continue;

                for (unsigned int n = start_index; n < end_index; ++n) {
                    const unsigned int neighbor_idx = d_sorted_values[n];
                    const float3 neighbor_pos = d_pos[neighbor_idx];
                    const float3 r_ij =
                        make_float3(pos.x - neighbor_pos.x, pos.y - neighbor_pos.y,
                                    pos.z - neighbor_pos.z);
                    const float r2 = dot3(r_ij, r_ij);
                    if (r2 <= 0.0f || r2 >= support2)
                        continue;

                    const float density_j = d_density[neighbor_idx];
                    if (density_j <= 0.0f)
                        continue;

                    // -p_j * m_j / rho_j^2 * grad W_ij
                    const float coeff = -d_pressure[neighbor_idx] * mass /
                                        (density_j * density_j);
                    const float3 grad_w_ij =
                        computeSpikyGradient(r_ij, sqrtf(r2), support);
                    sum.x += coeff * grad_w_ij.x;
                    sum.y += coeff * grad_w_ij.y;
                    sum.z += coeff * grad_w_ij.z;
                }
            }
        }
    }

    const float dt2 = delta_time * delta_time;
    d_sum_movement_pressure[idx] =
        make_float3(dt2 * sum.x, dt2 * sum.y, dt2 * sum.z);
}

/* Eq. (13) / thesis Eq. (4.8). The result is written to d_pressure_next so
 * every particle reads the same p^l. Negative pressure is clamped to 0. */
__global__ void updatePressureKernel(
    const float3* __restrict__ d_pos, const float* __restrict__ d_density,
    const float* __restrict__ d_density_adv,
    const float* __restrict__ d_diag_coeff,
    const float* __restrict__ d_pressure, float* __restrict__ d_pressure_next,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices,
    const float3* __restrict__ d_sum_movement_pressure,
    const float3* __restrict__ d_grad_coup_vect, int num_particles,
    SimulationParams params, float delta_time) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles)
        return;

    const float3 pos = d_pos[idx];
    const float support = params.smoothing_radius;
    const float support2 = support * support;
    const float mass = params.particle_mass;
    const float omega = params.relaxation_factor;
    const float density_i = d_density[idx];
    const float rho_i_adv = d_density_adv[idx];
    const float p_i_l = d_pressure[idx];
    const float a_ii = d_diag_coeff[idx];
    const float3 sum_dij_pj = d_sum_movement_pressure[idx];
    const float dt2 = delta_time * delta_time;
    const int cx = cellCoordinate(pos.x, params.domain_min_x, support);
    const int cy = cellCoordinate(pos.y, params.domain_min_y, support);
    const int cz = cellCoordinate(pos.z, params.domain_min_z, support);

    float neighbor_sum = 0.0f;

    if (density_i > 0.0f) {
        for (int i = cx - 1; i <= cx + 1; ++i) {
            if (i < 0 || i >= params.grid_width)
                continue;
            for (int j = cy - 1; j <= cy + 1; ++j) {
                if (j < 0 || j >= params.grid_height)
                    continue;
                for (int k = cz - 1; k <= cz + 1; ++k) {
                    if (k < 0 || k >= params.grid_depth)
                        continue;

                    const unsigned int key = computeCellKey(
                        i, j, k, params.grid_width, params.grid_height);
                    unsigned int start_index = 0;
                    unsigned int end_index = 0;
                    if (!cellRange(d_cell_start_indices, d_cell_end_indices, key,
                                   start_index, end_index))
                        continue;

                    for (unsigned int n = start_index; n < end_index; ++n) {
                        const unsigned int neighbor_idx = d_sorted_values[n];
                        if (neighbor_idx == static_cast<unsigned int>(idx))
                            continue;

                        const float3 neighbor_pos = d_pos[neighbor_idx];
                        const float3 r_ij = make_float3(pos.x - neighbor_pos.x,
                                                        pos.y - neighbor_pos.y,
                                                        pos.z - neighbor_pos.z);
                        const float r2 = dot3(r_ij, r_ij);
                        if (r2 <= 0.0f || r2 >= support2)
                            continue;

                        const float distance = sqrtf(r2);
                        const float3 grad_w_ij =
                            computeSpikyGradient(r_ij, distance, support);
                        const float3 grad_w_ji = computeSpikyGradient(
                            make_float3(-r_ij.x, -r_ij.y, -r_ij.z), distance,
                            support);

                        // sum_{k != i} d_jk p_k = sum_k d_jk p_k - d_ji p_i
                        // Ihmsen et al., p. 4, Eq. (11); thesis p. 13, Eq. (4.10).
                        const float3 d_ji = pairwiseDisplacementFromI(
                            grad_w_ji, mass, density_i, dt2);
                        const float3 sum_djk_pk = d_sum_movement_pressure[neighbor_idx];
                        const float3 d_jj = d_grad_coup_vect[neighbor_idx];
                        const float p_j = d_pressure[neighbor_idx];
                        const float3 excluded = make_float3(
                            sum_djk_pk.x - d_ji.x * p_i_l,
                            sum_djk_pk.y - d_ji.y * p_i_l,
                            sum_djk_pk.z - d_ji.z * p_i_l);
                        const float3 inner = make_float3(
                            sum_dij_pj.x - d_jj.x * p_j - excluded.x,
                            sum_dij_pj.y - d_jj.y * p_j - excluded.y,
                            sum_dij_pj.z - d_jj.z * p_j - excluded.z);
                        neighbor_sum += mass * dot3(inner, grad_w_ij);
                    }
                }
            }
        }
    }

    float updated_pressure = p_i_l;
    if (a_ii > 1e-12f || a_ii < -1e-12f) {
        // Eq. (13) / Eq. (4.8), then clamp negative pressure to 0 (p. 4 / p. 14).
        updated_pressure = (1.0f - omega) * p_i_l +
                           (omega / a_ii) * (params.rest_density - rho_i_adv -
                                             neighbor_sum);
        updated_pressure = fmaxf(0.0f, updated_pressure);
    }
    d_pressure_next[idx] = updated_pressure;
}

/* Eq. (4.7), using the pressure assigned for this iteration. */
__global__ void
updateDensityKernel(const float3* __restrict__ d_pos,
                    const float* __restrict__ d_density,
                    const float* __restrict__ d_density_adv,
                    const float* __restrict__ d_diag_coeff,
                    const float* __restrict__ d_pressure,
                    float* __restrict__ d_density_predicted,
                    const unsigned int* __restrict__ d_sorted_values,
                    const unsigned int* __restrict__ d_cell_start_indices,
                    const unsigned int* __restrict__ d_cell_end_indices,
                    const float3* __restrict__ d_sum_movement_pressure,
                    const float3* __restrict__ d_grad_coup_vect,
                    int num_particles, SimulationParams params,
                    float delta_time) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles)
        return;

    const float3 pos = d_pos[idx];
    const float support = params.smoothing_radius;
    const float support2 = support * support;
    const float mass = params.particle_mass;
    const float density_i = d_density[idx];
    const float rho_i_adv = d_density_adv[idx];
    const float p_i = d_pressure[idx];
    const float a_ii = d_diag_coeff[idx];
    const float3 sum_dij_pj = d_sum_movement_pressure[idx];
    const float dt2 = delta_time * delta_time;
    const int cx = cellCoordinate(pos.x, params.domain_min_x, support);
    const int cy = cellCoordinate(pos.y, params.domain_min_y, support);
    const int cz = cellCoordinate(pos.z, params.domain_min_z, support);

    float neighbor_sum = 0.0f;

    if (density_i > 0.0f) {
        for (int i = cx - 1; i <= cx + 1; ++i) {
            if (i < 0 || i >= params.grid_width)
                continue;
            for (int j = cy - 1; j <= cy + 1; ++j) {
                if (j < 0 || j >= params.grid_height)
                    continue;
                for (int k = cz - 1; k <= cz + 1; ++k) {
                    if (k < 0 || k >= params.grid_depth)
                        continue;

                    const unsigned int key = computeCellKey(
                        i, j, k, params.grid_width, params.grid_height);
                    unsigned int start_index = 0;
                    unsigned int end_index = 0;
                    if (!cellRange(d_cell_start_indices, d_cell_end_indices, key,
                                   start_index, end_index))
                        continue;

                    for (unsigned int n = start_index; n < end_index; ++n) {
                        const unsigned int neighbor_idx = d_sorted_values[n];
                        if (neighbor_idx == static_cast<unsigned int>(idx))
                            continue;

                        const float3 neighbor_pos = d_pos[neighbor_idx];
                        const float3 r_ij = make_float3(pos.x - neighbor_pos.x,
                                                        pos.y - neighbor_pos.y,
                                                        pos.z - neighbor_pos.z);
                        const float r2 = dot3(r_ij, r_ij);
                        if (r2 <= 0.0f || r2 >= support2)
                            continue;

                        const float distance = sqrtf(r2);
                        const float3 grad_w_ij =
                            computeSpikyGradient(r_ij, distance, support);
                        const float3 grad_w_ji = computeSpikyGradient(
                            make_float3(-r_ij.x, -r_ij.y, -r_ij.z), distance,
                            support);
                        const float3 d_ji = pairwiseDisplacementFromI(
                            grad_w_ji, mass, density_i, dt2);
                        const float3 sum_djk_pk = d_sum_movement_pressure[neighbor_idx];
                        const float3 d_jj = d_grad_coup_vect[neighbor_idx];
                        const float p_j = d_pressure[neighbor_idx];
                        const float3 excluded = make_float3(
                            sum_djk_pk.x - d_ji.x * p_i,
                            sum_djk_pk.y - d_ji.y * p_i,
                            sum_djk_pk.z - d_ji.z * p_i);
                        const float3 inner = make_float3(
                            sum_dij_pj.x - d_jj.x * p_j - excluded.x,
                            sum_dij_pj.y - d_jj.y * p_j - excluded.y,
                            sum_dij_pj.z - d_jj.z * p_j - excluded.z);
                        neighbor_sum += mass * dot3(inner, grad_w_ij);
                    }
                }
            }
        }
    }

    // rho^{l+1} = rho^adv + a_ii * p + neighbor sum. Thesis Eq. (4.7).
    d_density_predicted[idx] = rho_i_adv + a_ii * p_i + neighbor_sum;
}

/* F_i^p / m_i = -sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) grad W_ij
 * Ihmsen et al., p. 2, Eq. (2); thesis p. 14, Eq. (4.12).
 * v and x: Algorithm 1, Ihmsen p. 3 and thesis p. 14. */
__global__ void integrateKernel(
    const float3* __restrict__ d_pos, float3* __restrict__ d_pos_next,
    float3* __restrict__ d_vel, const float3* __restrict__ d_vel_adv,
    const float* __restrict__ d_density, const float* __restrict__ d_pressure,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices, int num_particles,
    SimulationParams params, float delta_time) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles)
        return;

    const float3 pos = d_pos[idx];
    const float support = params.smoothing_radius;
    const float support2 = support * support;
    const float mass = params.particle_mass;
    const float density_i = d_density[idx];
    const float pressure_i = d_pressure[idx];
    const int cx = cellCoordinate(pos.x, params.domain_min_x, support);
    const int cy = cellCoordinate(pos.y, params.domain_min_y, support);
    const int cz = cellCoordinate(pos.z, params.domain_min_z, support);

    float3 pressure_sum = make_float3(0.0f, 0.0f, 0.0f);
    if (density_i > 0.0f) {
        const float inv_density_i2 = 1.0f / (density_i * density_i);
        for (int i = cx - 1; i <= cx + 1; ++i) {
            if (i < 0 || i >= params.grid_width)
                continue;
            for (int j = cy - 1; j <= cy + 1; ++j) {
                if (j < 0 || j >= params.grid_height)
                    continue;
                for (int k = cz - 1; k <= cz + 1; ++k) {
                    if (k < 0 || k >= params.grid_depth)
                        continue;

                    const unsigned int key = computeCellKey(
                        i, j, k, params.grid_width, params.grid_height);
                    unsigned int start_index = 0;
                    unsigned int end_index = 0;
                    if (!cellRange(d_cell_start_indices, d_cell_end_indices, key,
                                   start_index, end_index))
                        continue;

                    for (unsigned int n = start_index; n < end_index; ++n) {
                        const unsigned int neighbor_idx = d_sorted_values[n];
                        const float3 neighbor_pos = d_pos[neighbor_idx];
                        const float3 r_ij = make_float3(pos.x - neighbor_pos.x,
                                                        pos.y - neighbor_pos.y,
                                                        pos.z - neighbor_pos.z);
                        const float r2 = dot3(r_ij, r_ij);
                        if (r2 <= 0.0f || r2 >= support2)
                            continue;

                        const float density_j = d_density[neighbor_idx];
                        if (density_j <= 0.0f)
                            continue;

                        // Eq. (4.12): m_j (p_i/rho_i^2 + p_j/rho_j^2) grad W_ij
                        const float coeff =
                            mass * (pressure_i * inv_density_i2 +
                                    d_pressure[neighbor_idx] /
                                        (density_j * density_j));
                        const float3 grad_w_ij =
                            computeSpikyGradient(r_ij, sqrtf(r2), support);
                        pressure_sum.x += coeff * grad_w_ij.x;
                        pressure_sum.y += coeff * grad_w_ij.y;
                        pressure_sum.z += coeff * grad_w_ij.z;
                    }
                }
            }
        }
    }

    const float3 vel_adv = d_vel_adv[idx];
    const float3 vel_new =
        make_float3(vel_adv.x - delta_time * pressure_sum.x,
                    vel_adv.y - delta_time * pressure_sum.y,
                    vel_adv.z - delta_time * pressure_sum.z);
    d_vel[idx] = vel_new;
    d_pos_next[idx] = make_float3(pos.x + delta_time * vel_new.x,
                                  pos.y + delta_time * vel_new.y,
                                  pos.z + delta_time * vel_new.z);
}
