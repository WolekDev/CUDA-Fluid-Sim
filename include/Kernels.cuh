#pragma once

#include "SimulationParams.cuh"
#include <cuda_runtime.h>

/* pi used by the 3D kernel normalizations. */
constexpr float kPi = 3.14159265358979323846f;

/* Cell key for a uniform grid with cell size equal to the support radius.
 * x changes fastest, then y, then z. */
__host__ __device__ inline unsigned int computeCellKey(int cx, int cy, int cz,
                                                       int grid_width,
                                                       int grid_height) {
    const auto ux = static_cast<unsigned int>(cx);
    const auto uy = static_cast<unsigned int>(cy);
    const auto uz = static_cast<unsigned int>(cz);
    const auto width = static_cast<unsigned int>(grid_width);
    const auto height = static_cast<unsigned int>(grid_height);
    return (uz * height + uy) * width + ux;
}

__host__ __device__ inline float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/* 3D cubic-spline kernel, called W_poly6 in the thesis.
 * Eliasson & Franzen, Accelerating IISPH, p. 10, Eq. (4.1).
 * s = d / R. Defined at d = 0, which supplies the SPH self contribution. */
__host__ __device__ inline float computePoly6(float distance, float support) {
    if (distance < 0.0f || distance >= support || support <= 0.0f)
        return 0.0f;

    const float s = distance / support;
    const float support3 = support * support * support;
    if (s < 0.5f) {
        const float radial = 1.0f - 6.0f * s * s + 6.0f * s * s * s;
        return 8.0f * radial / (kPi * support3);
    }

    const float one_minus_s = 1.0f - s;
    return 16.0f * one_minus_s * one_minus_s * one_minus_s / (kPi * support3);
}

/* Scalar W_spiky.
 * Eliasson & Franzen, Accelerating IISPH, p. 10, Eq. (4.2).
 * Distance 0 is the special case that returns 0 (same page): a particle has
 * no directional interaction with itself. */
__host__ __device__ inline float computeSpiky(float distance, float support) {
    if (distance <= 0.0f || distance >= support || support <= 0.0f)
        return 0.0f;

    const float s2 = (distance * distance) / (support * support);
    const float radial = 1.0f - s2;
    const float support2 = support * support;
    const float support5 = support2 * support2 * support;
    const float coeff = -8.0f * 3465.0f / (512.0f * kPi * support5);
    return coeff * radial * radial * radial;
}

/* W_viscosity.
 * Eliasson & Franzen, Accelerating IISPH, p. 10, Eq. (4.3).
 * Defined at d = 0. A particle's own contribution still vanishes because
 * the viscosity force multiplies by (v_i - v_i). */
__host__ __device__ inline float computeViscosity(float distance, float support) {
    if (distance < 0.0f || distance >= support || support <= 0.0f)
        return 0.0f;

    const float support2 = support * support;
    const float support3 = support2 * support;
    const float support6 = support3 * support3;
    return 45.0f * (support - distance) / (kPi * support6);
}

/* Dynamic viscosity for Eq. (4.4). The configured value is in mPa·s
 * (water at 25 C is 0.89). The force uses Pa·s = kg/(m·s), and
 * 1 mPa·s = 0.001 Pa·s. */
__host__ __device__ inline float dynamicViscosityPascalSeconds(float viscosity_mpa_s) {
    return viscosity_mpa_s * 0.001f;
}

/* ∇W_ij = (x_i - x_j) * W_spiky(|x_i - x_j|, R).
 * The thesis multiplies the non-normalized distance by W_spiky wherever
 * Ihmsen uses ∇W. Eliasson & Franzen, pp. 10–11, Eq. (4.2);
 * Ihmsen et al., IISPH, TVCG author's version, p. 3, Algorithm 1. */
__host__ __device__ inline float3 computeSpikyGradient(float3 r_ij, float distance,
                                                       float support) {
    const float weight = computeSpiky(distance, support);
    return make_float3(r_ij.x * weight, r_ij.y * weight, r_ij.z * weight);
}

/* Compute hashes for the particles. Store in key value arrays. */
__global__ void computeHashKernel(const float3* __restrict__ d_pos,
                                  unsigned int* __restrict__ d_keys,
                                  unsigned int* __restrict__ d_values,
                                  int num_particles, SimulationParams params);

/* Find start and end indexes of each cell. */
__global__ void
findCellRangesKernel(const unsigned int* __restrict__ d_sorted_keys,
                     unsigned int* __restrict__ d_cell_start_indices,
                     unsigned int* __restrict__ d_cell_end_indices,
                     int num_particles);

/* Compute density for SPH.
 * rho_i(t) = sum_j m_j W_ij(t).
 * Ihmsen et al., p. 2 and Algorithm 1; thesis p. 8, Algorithm 1 line 3.
 * W_ij is Eq. (4.1). */
__global__ void
computeDensityKernel(const unsigned int* __restrict__ d_sorted_values,
                     const float3* __restrict__ d_pos,
                     const unsigned int* __restrict__ d_cell_start_indices,
                     const unsigned int* __restrict__ d_cell_end_indices,
                     float* __restrict__ d_density, int num_particles,
                     SimulationParams params);

/* Computes velocity advection and the displacement factor d_ii.
 * F_i^adv = F_i^ext + F_i^visc. F_i^ext = m_i g unless a force buffer is supplied.
 * F_i^visc = -sum_j (v_i - v_j) (m_j / rho_i)^2 * mu * W_viscosity
 *   Thesis p. 11, Eq. (4.4). W_viscosity is Eq. (4.3).
 *   mu is the dynamic viscosity in Pa·s, from the configured mPa·s value.
 * v_i^adv = v_i + dt * F_i^adv / m_i
 * d_ii = dt^2 * sum_j -m_j / rho_i^2 * grad W_ij
 * Ihmsen et al., p. 3, Algorithm 1; thesis pp. 8 and 11. */
__global__ void computeAdvectionAndGradientKernel(
    const float3* __restrict__ d_pos, const float3* __restrict__ d_vel,
    const float3* __restrict__ d_force_ext, float3* __restrict__ d_vel_adv,
    const float* __restrict__ d_density, float3* __restrict__ d_grad_coup_vect,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices, int num_particles,
    SimulationParams params, float delta_time);

/* Initialize pressure solver. Calculates density advection, initial pressure,
 * and the diagonal coefficient a_ii.
 * rho_i^adv = rho_i + dt * sum_j m_j (v_i^adv - v_j^adv) · grad W_ij
 *   Ihmsen et al., p. 2, Eq. (4); thesis p. 8, Algorithm 1 line 7.
 * p_i^0 = 0.5 * p_i(t - dt)
 *   Ihmsen et al., p. 4; thesis p. 12.
 * a_ii = sum_j m_j (d_ii - d_ji) · grad W_ij
 *   Ihmsen et al., p. 4, Eq. (12); thesis p. 12, Eq. (4.5).
 * d_ji = -dt^2 * (m_i / rho_i^2) * grad W_ji
 *   thesis p. 13, Eq. (4.11), consistent with Ihmsen Eq. (9). */
__global__ void initPressureSolverKernel(
    const float3* __restrict__ d_pos, const float3* __restrict__ d_vel_adv,
    const float* __restrict__ d_density, float* __restrict__ d_density_adv,
    const float3* __restrict__ d_grad_coup_vect,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices,
    float* __restrict__ d_pressure, const float* __restrict__ d_pressure_prev,
    float* __restrict__ d_diag_coeff, int num_particles,
    SimulationParams params, float delta_time);

/* Sum of pressure displacements, sum_j d_ij p_j^l.
 * sum_j d_ij p_j^l = dt^2 * sum_j -m_j / rho_j^2 * p_j^l * grad W_ij
 * Ihmsen et al., p. 3, Algorithm 1; thesis p. 8 line 14 and p. 12. */
__global__ void calculateSumMovementFromPressureKernel(
    const float3* __restrict__ d_pos, const float* __restrict__ d_density,
    const float* __restrict__ d_pressure,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices,
    float3* __restrict__ d_sum_movement_pressure, int num_particles,
    SimulationParams params, float delta_time);

/* Relaxed Jacobi pressure update. Writes p^{l+1} without reading the
 * in-progress output, so every particle uses the same p^l.
 * p_i^{l+1} = (1 - omega) p_i^l + (omega / a_ii) *
 *   (rho_0 - rho_i^adv - sum_j m_j (sum_j d_ij p_j^l - d_jj p_j^l
 *   - sum_{k != i} d_jk p_k^l) · grad W_ij)
 * Ihmsen et al., p. 4, Eq. (13); thesis p. 13, Eq. (4.8).
 * Negative results are clamped to 0. Ihmsen et al., p. 4; thesis p. 14. */
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
    SimulationParams params, float delta_time);

/* Predicted density after the pressure update.
 * rho_i^{l+1} = rho_i^adv + a_ii p_i + sum_j m_j (sum d_ij p_j - d_jj p_j
 *   - sum_{k != i} d_jk p_k) · grad W_ij
 * Ihmsen et al., p. 4 (rearrangement of Eq. (13));
 * thesis p. 12, Eq. (4.7), evaluated after p^{l+1} is assigned
 * (thesis p. 24, Algorithm 6). */
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
                    float delta_time);

/* Integration. Pressure force, then semi-implicit Euler.
 * F_i^p = -m_i * sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) grad W_ij
 *   Ihmsen et al., p. 2, Eq. (2); thesis p. 14, Eq. (4.12).
 * v_i(t+dt) = v_i^adv + dt * F_i^p / m_i
 * x_i(t+dt) = x_i + dt * v_i(t+dt)
 *   Ihmsen et al., p. 3, Algorithm 1; thesis p. 8 lines 21–22 and p. 14.
 * New positions are written separately so every thread still reads the
 * positions from the pressure solve. */
__global__ void integrateKernel(
    const float3* __restrict__ d_pos, float3* __restrict__ d_pos_next,
    float3* __restrict__ d_vel, const float3* __restrict__ d_vel_adv,
    const float* __restrict__ d_density, const float* __restrict__ d_pressure,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices, int num_particles,
    SimulationParams params, float delta_time);
