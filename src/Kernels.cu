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

__device__ inline float3 subtract3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline bool isBoundaryId(unsigned int id, int num_fluid) {
    return id >= static_cast<unsigned int>(num_fluid);
}

__device__ inline unsigned int boundaryIndex(unsigned int id, int num_fluid) {
    return id - static_cast<unsigned int>(num_fluid);
}

/* Visits every particle id stored in the 3x3x3 block of cells around pos.
 * Support radius equals the cell size, so no other cell can contribute. */
template <typename Visitor>
__device__ inline void forEachNeighbor(float3 pos, const SimulationParams& params,
                                       const unsigned int* __restrict__ sorted,
                                       const unsigned int* __restrict__ starts,
                                       const unsigned int* __restrict__ ends,
                                       Visitor visit) {
    const float support = params.smoothing_radius;
    const int cx = cellCoordinate(pos.x, params.domain_min_x, support);
    const int cy = cellCoordinate(pos.y, params.domain_min_y, support);
    const int cz = cellCoordinate(pos.z, params.domain_min_z, support);

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
                if (!cellRange(starts, ends, key, start_index, end_index))
                    continue;

                for (unsigned int n = start_index; n < end_index; ++n)
                    visit(sorted[n]);
            }
        }
    }
}

/* d_ji = -dt^2 * (m_i / rho_i^2) * grad W_ji
 * Thesis p. 13, Eq. (4.11); Ihmsen et al., p. 4, Eq. (9) with i and j swapped. */
__device__ inline float3 pairwiseDisplacementFromI(float3 grad_w_ji, float mass,
                                                   float density_i, float dt2) {
    const float scale = -dt2 * mass / (density_i * density_i);
    return make_float3(scale * grad_w_ji.x, scale * grad_w_ji.y,
                       scale * grad_w_ji.z);
}

/* Neighbor part of Eq. (16) and of the density prediction:
 *   sum_j m_j (sum_k d_ik p_k - d_jj p_j - sum_{k != i} d_jk p_k) · grad W_ij
 *   + sum_b Psi_b (sum_j d_ij p_j) · grad W_ib
 * Fluid term: Ihmsen et al., p. 4, Eq. (13); thesis Eqs. (4.7), (4.8), (4.10).
 * Boundary term: Ihmsen et al., p. 6, Eq. (16). */
__device__ float pressureNeighborTerm(
    int idx, const float3* __restrict__ d_pos, BoundaryParticles boundary,
    const float* __restrict__ d_density, const float* __restrict__ d_pressure,
    const float3* __restrict__ d_sum_movement_pressure,
    const float3* __restrict__ d_grad_coup_vect,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices, int num_fluid,
    const SimulationParams& params, float delta_time) {

    const float density_i = d_density[idx];
    if (density_i <= 0.0f)
        return 0.0f;

    const float3 pos = d_pos[idx];
    const float support = params.smoothing_radius;
    const float support2 = support * support;
    const float mass = params.particle_mass;
    const float p_i = d_pressure[idx];
    const float3 sum_dij_pj = d_sum_movement_pressure[idx];
    const float dt2 = delta_time * delta_time;

    float neighbor_sum = 0.0f;
    forEachNeighbor(
        pos, params, d_sorted_values, d_cell_start_indices, d_cell_end_indices,
        [&](unsigned int neighbor_idx) {
            if (isBoundaryId(neighbor_idx, num_fluid)) {
                const unsigned int b = boundaryIndex(neighbor_idx, num_fluid);
                const float3 r_ib = subtract3(pos, boundary.positions[b]);
                const float r2 = dot3(r_ib, r_ib);
                if (r2 <= 0.0f || r2 >= support2)
                    return;
                // Eq. (16): Psi_b (sum_j d_ij p_j) · grad W_ib
                const float3 grad_w_ib =
                    computeSpikyGradient(r_ib, sqrtf(r2), support);
                neighbor_sum += boundary.psi[b] * dot3(sum_dij_pj, grad_w_ib);
                return;
            }

            if (neighbor_idx == static_cast<unsigned int>(idx))
                return;
            const float3 r_ij = subtract3(pos, d_pos[neighbor_idx]);
            const float r2 = dot3(r_ij, r_ij);
            if (r2 <= 0.0f || r2 >= support2)
                return;

            const float distance = sqrtf(r2);
            const float3 grad_w_ij = computeSpikyGradient(r_ij, distance, support);
            const float3 grad_w_ji = computeSpikyGradient(
                make_float3(-r_ij.x, -r_ij.y, -r_ij.z), distance, support);

            // sum_{k != i} d_jk p_k = sum_k d_jk p_k - d_ji p_i
            // Ihmsen et al., p. 4, Eq. (11); thesis p. 13, Eq. (4.10).
            const float3 d_ji =
                pairwiseDisplacementFromI(grad_w_ji, mass, density_i, dt2);
            const float3 sum_djk_pk = d_sum_movement_pressure[neighbor_idx];
            const float3 d_jj = d_grad_coup_vect[neighbor_idx];
            const float p_j = d_pressure[neighbor_idx];
            const float3 excluded = make_float3(sum_djk_pk.x - d_ji.x * p_i,
                                                sum_djk_pk.y - d_ji.y * p_i,
                                                sum_djk_pk.z - d_ji.z * p_i);
            const float3 inner =
                make_float3(sum_dij_pj.x - d_jj.x * p_j - excluded.x,
                            sum_dij_pj.y - d_jj.y * p_j - excluded.y,
                            sum_dij_pj.z - d_jj.z * p_j - excluded.z);
            neighbor_sum += mass * dot3(inner, grad_w_ij);
        });
    return neighbor_sum;
}

} // namespace

/* Compute hashes for fluid and boundary particles. Store in key value arrays. */
__global__ void computeHashKernel(const float3* __restrict__ d_pos,
                                  const float3* __restrict__ d_boundary_pos,
                                  unsigned int* __restrict__ d_keys,
                                  unsigned int* __restrict__ d_values,
                                  int num_fluid, int num_boundary,
                                  SimulationParams params) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_fluid + num_boundary)
        return;

    const float3 pos =
        idx < num_fluid ? d_pos[idx] : d_boundary_pos[idx - num_fluid];
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

/* delta_b = sum_{b_j} W_{b b_j}, Psi_b(rho_0) = rho_0 / delta_b
 * Ihmsen et al., p. 5, Section 4. W is thesis p. 10, Eq. (4.1). */
__global__ void computeBoundaryPsiKernel(
    const float3* __restrict__ d_boundary_pos, float* __restrict__ d_boundary_psi,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices, int num_fluid,
    int num_boundary, SimulationParams params) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= num_boundary)
        return;

    const float3 pos_b = d_boundary_pos[b];
    const float support = params.smoothing_radius;
    const float support2 = support * support;

    float delta_b = 0.0f;
    forEachNeighbor(pos_b, params, d_sorted_values, d_cell_start_indices,
                    d_cell_end_indices, [&](unsigned int neighbor_idx) {
                        if (!isBoundaryId(neighbor_idx, num_fluid))
                            return;
                        const float3 r = subtract3(
                            pos_b, d_boundary_pos[boundaryIndex(neighbor_idx,
                                                                num_fluid)]);
                        const float r2 = dot3(r, r);
                        // Self term W(0) is included when r = 0.
                        if (r2 <= support2)
                            delta_b += computePoly6(sqrtf(r2), support);
                    });

    // delta_b >= W(0) > 0 because b is its own neighbor.
    d_boundary_psi[b] = params.rest_density / delta_b;
}

/* rho_i(t) = sum_j m_j W_ij + sum_b Psi_b(rho_0) W_ib
 * Ihmsen et al., p. 2 and p. 5, first line of Eq. (14);
 * thesis p. 8, Algorithm 1 line 3. W is thesis p. 10, Eq. (4.1). */
__global__ void
computeDensityKernel(const unsigned int* __restrict__ d_sorted_values,
                     const float3* __restrict__ d_pos, BoundaryParticles boundary,
                     const unsigned int* __restrict__ d_cell_start_indices,
                     const unsigned int* __restrict__ d_cell_end_indices,
                     float* __restrict__ d_density, int num_fluid,
                     SimulationParams params) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_fluid)
        return;

    const float3 pos = d_pos[idx];
    const float support = params.smoothing_radius;
    const float support2 = support * support;

    float density = 0.0f;
    forEachNeighbor(
        pos, params, d_sorted_values, d_cell_start_indices, d_cell_end_indices,
        [&](unsigned int neighbor_idx) {
            const bool is_boundary = isBoundaryId(neighbor_idx, num_fluid);
            const unsigned int b = boundaryIndex(neighbor_idx, num_fluid);
            const float3 neighbor_pos =
                is_boundary ? boundary.positions[b] : d_pos[neighbor_idx];
            const float3 r = subtract3(pos, neighbor_pos);
            const float r2 = dot3(r, r);
            if (r2 > support2)
                return;
            // Eq. (14): Psi_b replaces m_j for boundary neighbors.
            // Eq. (4.1): the fluid self term is W(0), included when r = 0.
            const float weight = is_boundary ? boundary.psi[b] : params.particle_mass;
            density += weight * computePoly6(sqrtf(r2), support);
        });

    d_density[idx] = density;
}

/* F_i^adv = m_i g + F_i^visc, then v_i^adv = v_i + dt * F_i^adv / m_i.
 * F_i^visc: thesis p. 11, Eq. (4.4). d_ii: Ihmsen Algorithm 1, p. 3, with the
 * boundary term from p. 6. */
__global__ void computeAdvectionAndGradientKernel(
    const float3* __restrict__ d_pos, BoundaryParticles boundary,
    const float3* __restrict__ d_vel, const float3* __restrict__ d_force_ext,
    float3* __restrict__ d_vel_adv, const float* __restrict__ d_density,
    float3* __restrict__ d_grad_coup_vect,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices, int num_fluid,
    SimulationParams params, float delta_time) {

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_fluid)
        return;

    const float3 pos_i = d_pos[idx];
    const float3 vel_i = d_vel[idx];
    const float mass = params.particle_mass;

    const float density_i = d_density[idx];
    float3 d_ii_sum = make_float3(0.0f, 0.0f, 0.0f);
    float3 f_visc = make_float3(0.0f, 0.0f, 0.0f);
    if (density_i > 0.0f) {
        const float support = params.smoothing_radius;
        const float support2 = support * support;
        const float inv_density_i2 = 1.0f / (density_i * density_i);
        // Eq. (4.4): (m_j / rho_i)^2 * mu, with mu in Pa·s.
        const float mass_over_density = mass / density_i;
        const float visc_scale = mass_over_density * mass_over_density *
                                 dynamicViscosityPascalSeconds(params.viscosity);

        forEachNeighbor(
            pos_i, params, d_sorted_values, d_cell_start_indices,
            d_cell_end_indices, [&](unsigned int neighbor_idx) {
                const bool is_boundary = isBoundaryId(neighbor_idx, num_fluid);
                const unsigned int b = boundaryIndex(neighbor_idx, num_fluid);
                const float3 r_ij = subtract3(
                    pos_i, is_boundary ? boundary.positions[b] : d_pos[neighbor_idx]);
                const float r2 = dot3(r_ij, r_ij);
                if (r2 >= support2)
                    return;

                const float distance = sqrtf(r2);
                if (distance > 0.0f) {
                    // Ihmsen p. 6: m_j / rho_i^2 grad W_ij for fluid,
                    // Psi_b / rho_i^2 grad W_ib for boundary.
                    const float3 grad_w = computeSpikyGradient(r_ij, distance, support);
                    const float scale =
                        (is_boundary ? boundary.psi[b] : mass) * inv_density_i2;
                    d_ii_sum.x += scale * grad_w.x;
                    d_ii_sum.y += scale * grad_w.y;
                    d_ii_sum.z += scale * grad_w.z;
                }

                if (is_boundary)
                    return;

                // Eq. (4.4): -(v_i - v_j) * (m_j / rho_i)^2 * Const_visc * W
                const float3 vel_j = d_vel[neighbor_idx];
                const float coeff = visc_scale * computeViscosity(distance, support);
                f_visc.x += -(vel_i.x - vel_j.x) * coeff;
                f_visc.y += -(vel_i.y - vel_j.y) * coeff;
                f_visc.z += -(vel_i.z - vel_j.z) * coeff;
            });
    }

    // F_adv = F_ext + F_visc, then v_adv = v + dt * F_adv / m. Thesis p. 11.
    const float3 f_ext =
        d_force_ext ? d_force_ext[idx]
                    : make_float3(params.gravity_vector.x * mass,
                                  params.gravity_vector.y * mass,
                                  params.gravity_vector.z * mass);
    const float inv_mass = 1.0f / mass;
    d_vel_adv[idx] = make_float3(
        vel_i.x + delta_time * (f_ext.x + f_visc.x) * inv_mass,
        vel_i.y + delta_time * (f_ext.y + f_visc.y) * inv_mass,
        vel_i.z + delta_time * (f_ext.z + f_visc.z) * inv_mass);

    const float dt2 = delta_time * delta_time;
    d_grad_coup_vect[idx] =
        make_float3(-dt2 * d_ii_sum.x, -dt2 * d_ii_sum.y, -dt2 * d_ii_sum.z);
}

/* rho_i^adv from Ihmsen Eq. (4) with the boundary term from p. 6,
 * p_i^0 = 0.5 p_i(t - dt), and a_ii from Eq. (12) with the boundary term
 * from the p. 6 density correction. d_ji is thesis Eq. (4.11). */
__global__ void initPressureSolverKernel(
    const float3* __restrict__ d_pos, BoundaryParticles boundary,
    const float3* __restrict__ d_vel_adv, const float* __restrict__ d_density,
    float* __restrict__ d_density_adv, const float3* __restrict__ d_grad_coup_vect,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices,
    float* __restrict__ d_pressure, const float* __restrict__ d_pressure_prev,
    float* __restrict__ d_diag_coeff, int num_fluid, SimulationParams params,
    float delta_time) {

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_fluid)
        return;

    const float3 pos_i = d_pos[idx];
    const float mass = params.particle_mass;
    const float support = params.smoothing_radius;
    const float support2 = support * support;
    const float density_i = d_density[idx];
    const float3 d_ii = d_grad_coup_vect[idx];
    const float3 vel_adv_i = d_vel_adv[idx];
    const float dt2 = delta_time * delta_time;

    // p_i^0 = 0.5 * p_i(t - dt). Ihmsen et al., p. 4; thesis p. 12.
    d_pressure[idx] = 0.5f * d_pressure_prev[idx];

    float density_adv_sum = 0.0f;
    float a_ii_sum = 0.0f;

    if (density_i > 0.0f) {
        forEachNeighbor(
            pos_i, params, d_sorted_values, d_cell_start_indices,
            d_cell_end_indices, [&](unsigned int neighbor_idx) {
                if (isBoundaryId(neighbor_idx, num_fluid)) {
                    const unsigned int b = boundaryIndex(neighbor_idx, num_fluid);
                    const float3 r_ib = subtract3(pos_i, boundary.positions[b]);
                    const float r2 = dot3(r_ib, r_ib);
                    if (r2 <= 0.0f || r2 >= support2)
                        return;
                    const float3 grad_w_ib =
                        computeSpikyGradient(r_ib, sqrtf(r2), support);
                    const float psi_b = boundary.psi[b];

                    // p. 6: Psi_b (v_i^adv - v_b) · grad W_ib, with v_b = 0.
                    density_adv_sum += psi_b * dot3(vel_adv_i, grad_w_ib);
                    // p. 6: Psi_b d_ii · grad W_ib
                    a_ii_sum += psi_b * dot3(d_ii, grad_w_ib);
                    return;
                }

                const float3 r_ij = subtract3(pos_i, d_pos[neighbor_idx]);
                const float r2 = dot3(r_ij, r_ij);
                if (r2 <= 0.0f || r2 >= support2)
                    return;

                const float distance = sqrtf(r2);
                const float3 grad_w_ij = computeSpikyGradient(r_ij, distance, support);
                const float3 grad_w_ji = computeSpikyGradient(
                    make_float3(-r_ij.x, -r_ij.y, -r_ij.z), distance, support);

                // Eq. (4): m_j (v_i^adv - v_j^adv) · grad W_ij
                const float3 relative_vel =
                    subtract3(vel_adv_i, d_vel_adv[neighbor_idx]);
                density_adv_sum += mass * dot3(relative_vel, grad_w_ij);

                // Eq. (12): m_j (d_ii - d_ji) · grad W_ij
                const float3 d_ji =
                    pairwiseDisplacementFromI(grad_w_ji, mass, density_i, dt2);
                a_ii_sum += mass * dot3(subtract3(d_ii, d_ji), grad_w_ij);
            });
    }

    d_density_adv[idx] = density_i + delta_time * density_adv_sum;
    d_diag_coeff[idx] = a_ii_sum;
}

/* sum_j d_ij p_j^l = dt^2 * sum_j (-m_j / rho_j^2) p_j^l grad W_ij
 * Ihmsen et al., p. 3, Algorithm 1; thesis p. 8 line 14.
 * Boundary particles have no pressure (Ihmsen p. 6), so only fluid j count. */
__global__ void calculateSumMovementFromPressureKernel(
    const float3* __restrict__ d_pos, const float* __restrict__ d_density,
    const float* __restrict__ d_pressure,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices,
    float3* __restrict__ d_sum_movement_pressure, int num_fluid,
    SimulationParams params, float delta_time) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_fluid)
        return;

    const float3 pos = d_pos[idx];
    const float support = params.smoothing_radius;
    const float support2 = support * support;
    const float mass = params.particle_mass;

    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    forEachNeighbor(
        pos, params, d_sorted_values, d_cell_start_indices, d_cell_end_indices,
        [&](unsigned int neighbor_idx) {
            if (isBoundaryId(neighbor_idx, num_fluid))
                return;
            const float3 r_ij = subtract3(pos, d_pos[neighbor_idx]);
            const float r2 = dot3(r_ij, r_ij);
            if (r2 <= 0.0f || r2 >= support2)
                return;

            const float density_j = d_density[neighbor_idx];
            if (density_j <= 0.0f)
                return;

            // -p_j * m_j / rho_j^2 * grad W_ij
            const float coeff =
                -d_pressure[neighbor_idx] * mass / (density_j * density_j);
            const float3 grad_w_ij = computeSpikyGradient(r_ij, sqrtf(r2), support);
            sum.x += coeff * grad_w_ij.x;
            sum.y += coeff * grad_w_ij.y;
            sum.z += coeff * grad_w_ij.z;
        });

    const float dt2 = delta_time * delta_time;
    d_sum_movement_pressure[idx] =
        make_float3(dt2 * sum.x, dt2 * sum.y, dt2 * sum.z);
}

/* Eq. (16), p. 6 (Eq. (13) / thesis Eq. (4.8) plus the boundary term).
 * The result is written to d_pressure_next so every particle reads the same
 * p^l. Negative pressure is clamped to 0. */
__global__ void updatePressureKernel(
    const float3* __restrict__ d_pos, BoundaryParticles boundary,
    const float* __restrict__ d_density, const float* __restrict__ d_density_adv,
    const float* __restrict__ d_diag_coeff,
    const float* __restrict__ d_pressure, float* __restrict__ d_pressure_next,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices,
    const float3* __restrict__ d_sum_movement_pressure,
    const float3* __restrict__ d_grad_coup_vect, int num_fluid,
    SimulationParams params, float delta_time) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_fluid)
        return;

    const float omega = params.relaxation_factor;
    const float rho_i_adv = d_density_adv[idx];
    const float p_i_l = d_pressure[idx];
    const float a_ii = d_diag_coeff[idx];
    const float neighbor_sum = pressureNeighborTerm(
        idx, d_pos, boundary, d_density, d_pressure, d_sum_movement_pressure,
        d_grad_coup_vect, d_sorted_values, d_cell_start_indices,
        d_cell_end_indices, num_fluid, params, delta_time);

    float updated_pressure = p_i_l;
    if (a_ii > 1e-12f || a_ii < -1e-12f) {
        // Eq. (16), then clamp negative pressure to 0 (p. 4 / thesis p. 14).
        updated_pressure = (1.0f - omega) * p_i_l +
                           (omega / a_ii) * (params.rest_density - rho_i_adv -
                                             neighbor_sum);
        updated_pressure = fmaxf(0.0f, updated_pressure);
    }
    d_pressure_next[idx] = updated_pressure;
}

/* Thesis Eq. (4.7) with the boundary term from Ihmsen p. 6, using the
 * pressure assigned for this iteration. */
__global__ void
updateDensityKernel(const float3* __restrict__ d_pos, BoundaryParticles boundary,
                    const float* __restrict__ d_density,
                    const float* __restrict__ d_density_adv,
                    const float* __restrict__ d_diag_coeff,
                    const float* __restrict__ d_pressure,
                    float* __restrict__ d_density_predicted,
                    const unsigned int* __restrict__ d_sorted_values,
                    const unsigned int* __restrict__ d_cell_start_indices,
                    const unsigned int* __restrict__ d_cell_end_indices,
                    const float3* __restrict__ d_sum_movement_pressure,
                    const float3* __restrict__ d_grad_coup_vect, int num_fluid,
                    SimulationParams params, float delta_time) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_fluid)
        return;

    const float neighbor_sum = pressureNeighborTerm(
        idx, d_pos, boundary, d_density, d_pressure, d_sum_movement_pressure,
        d_grad_coup_vect, d_sorted_values, d_cell_start_indices,
        d_cell_end_indices, num_fluid, params, delta_time);

    // rho^{l+1} = rho^adv + a_ii * p + neighbor sum. Thesis Eq. (4.7).
    d_density_predicted[idx] =
        d_density_adv[idx] + d_diag_coeff[idx] * d_pressure[idx] + neighbor_sum;
}

/* F_i^p / m_i = -sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) grad W_ij
 *             - sum_b Psi_b p_i/rho_i^2 grad W_ib
 * Ihmsen et al., p. 2, Eq. (2) and p. 6, Eq. (15); thesis p. 14, Eq. (4.12).
 * v and x: Algorithm 1, Ihmsen p. 3 and thesis p. 14. Threads cover only fluid
 * particles, so boundary particles never move (thesis pp. 19 and 25). */
__global__ void integrateKernel(
    const float3* __restrict__ d_pos, BoundaryParticles boundary,
    float3* __restrict__ d_pos_next, float3* __restrict__ d_vel,
    const float3* __restrict__ d_vel_adv, const float* __restrict__ d_density,
    const float* __restrict__ d_pressure,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices, int num_fluid,
    SimulationParams params, float delta_time) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_fluid)
        return;

    const float3 pos = d_pos[idx];
    const float support = params.smoothing_radius;
    const float support2 = support * support;
    const float mass = params.particle_mass;
    const float density_i = d_density[idx];
    const float pressure_i = d_pressure[idx];

    float3 pressure_sum = make_float3(0.0f, 0.0f, 0.0f);
    if (density_i > 0.0f) {
        const float pressure_term_i = pressure_i / (density_i * density_i);
        forEachNeighbor(
            pos, params, d_sorted_values, d_cell_start_indices, d_cell_end_indices,
            [&](unsigned int neighbor_idx) {
                const bool is_boundary = isBoundaryId(neighbor_idx, num_fluid);
                const unsigned int b = boundaryIndex(neighbor_idx, num_fluid);
                const float3 r_ij = subtract3(
                    pos, is_boundary ? boundary.positions[b] : d_pos[neighbor_idx]);
                const float r2 = dot3(r_ij, r_ij);
                if (r2 <= 0.0f || r2 >= support2)
                    return;

                float coeff = 0.0f;
                if (is_boundary) {
                    // Eq. (15) / m_i: Psi_b p_i / rho_i^2 grad W_ib
                    coeff = boundary.psi[b] * pressure_term_i;
                } else {
                    const float density_j = d_density[neighbor_idx];
                    if (density_j <= 0.0f)
                        return;
                    // Eq. (4.12): m_j (p_i/rho_i^2 + p_j/rho_j^2) grad W_ij
                    coeff = mass * (pressure_term_i + d_pressure[neighbor_idx] /
                                                          (density_j * density_j));
                }
                const float3 grad_w = computeSpikyGradient(r_ij, sqrtf(r2), support);
                pressure_sum.x += coeff * grad_w.x;
                pressure_sum.y += coeff * grad_w.y;
                pressure_sum.z += coeff * grad_w.z;
            });
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

/* Pipe inlet zone. Overrides the integrated velocity and
 * position of fluid that started the step inside the zone. */
__global__ void applyInflowKernel(float3* __restrict__ d_pos,
                                  const float3* __restrict__ d_pos_before,
                                  float3* __restrict__ d_vel, int num_fluid,
                                  float3 origin, float3 axis, float zone_length,
                                  float speed, float delta_time) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_fluid)
        return;

    const float3 before = d_pos_before[idx];
    if (dot3(subtract3(before, origin), axis) >= zone_length)
        return;

    const float3 vel = make_float3(speed * axis.x, speed * axis.y, speed * axis.z);
    d_vel[idx] = vel;
    d_pos[idx] = make_float3(before.x + delta_time * vel.x,
                             before.y + delta_time * vel.y,
                             before.z + delta_time * vel.z);
}

/* Pipe outlet */
__global__ void markOutflowKernel(const float3* __restrict__ d_pos,
                                  unsigned char* __restrict__ d_keep, int num_fluid,
                                  float3 origin, float3 axis, float outlet_s,
                                  float3 domain_max, SimulationParams params) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_fluid)
        return;

    const float3 pos = d_pos[idx];
    const bool inside_domain =
        pos.x >= params.domain_min_x && pos.x <= domain_max.x &&
        pos.y >= params.domain_min_y && pos.y <= domain_max.y &&
        pos.z >= params.domain_min_z && pos.z <= domain_max.z;
    const bool before_outlet = dot3(subtract3(pos, origin), axis) <= outlet_s;
    d_keep[idx] = inside_domain && before_outlet ? 1 : 0;
}
