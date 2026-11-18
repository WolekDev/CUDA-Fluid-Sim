#pragma once

#include "SimulationParams.cuh"
#include <cuda_runtime.h>

/* Compute hashes for the particles. Store in key value arrays. */
__global__ void computeHashKernel(const float2* __restrict__ d_pos,
                                  unsigned int* __restrict__ d_keys,
                                  unsigned int* __restrict__ d_values,
                                  int num_particles, SimulationParams params);

/* Find start and end indexes of each cell. */
__global__ void
findCellRangesKernel(const unsigned int* __restrict__ d_sorted_keys,
                     unsigned int* __restrict__ d_cell_start_indices,
                     unsigned int* __restrict__ d_cell_end_indices,
                     int num_particles);

/* Helper kernel for calculating the 2D Poly6 kernel weight */
__device__ inline float computePoly6(float r2, float h2, float h);

/* Compute density for SPH  */
__global__ void
computeDensityKernel(const unsigned int* __restrict__ d_sorted_values,
                     const float2* __restrict__ d_pos,
                     const unsigned int* __restrict__ d_cell_start_indices,
                     const unsigned int* __restrict__ d_cell_end_indices,
                     float* __restrict__ d_density, int num_particles,
                     SimulationParams params);

/* Helper kernel for calculating the Spiky gradient */
__device__ inline float2 computeSpikyGradient(float2 r_ij, float r, float h);

/* Computes Velocity Advection and Gradient Coupling Vector */
__global__ void computeAdvectionAndGradientKernel(
    const float2* __restrict__ d_pos, const float2* __restrict__ d_vel,
    const float2* __restrict__ d_force_ext, float2* __restrict__ d_vel_adv,
    const float* __restrict__ d_density, float2* __restrict__ d_grad_coup_vect,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices, int num_particles,
    SimulationParams params, float delta_time);

/* Initialize pressure solver. Calculates density advection, initial pressure,
 * and the diagonal coefficient for the pressure matrix. */
__global__ void initPressureSolverKernel(
    const float2* __restrict__ d_pos, const float2* __restrict__ d_vel_adv,
    const float* __restrict__ d_density, float* __restrict__ d_density_adv,
    const float2* __restrict__ d_grad_coup_vect,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices,
    float* __restrict__ d_pressure, const float* __restrict__ d_pressure_prev,
    float* __restrict__ d_diag_coeff, int num_particles,
    SimulationParams params, float delta_time);

/* This is the kernel for solving SUM(dij * Pj^l). This equation refers to the
 * total movement based on pressure forces for each particle. This is needed for
 * the main pressure solver kernel.
 * Equation: SUM(d_ji * P_j^l) = dt^2 * SUM_j( p_j * (-mj / (rho_j(t)^2) ) dot
 * Wij)*/
__global__ void calculateSumMovementFromPressureKernel(
    const float2* __restrict__ d_pos, const float* __restrict__ d_density,
    const float* __restrict__ d_pressure,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices,
    float2* __restrict__ d_sum_movement_pressure, int num_particles,
    SimulationParams params, float delta_time);

/* Main pressure solver kernel. Computes p_i^(l+1) for the next iteration of
 * the loop.
 *
 * Equation Pressure: p_i^(l+1) = (1-omega)*p_i^l + (omega/a_ii) *
 * (rest_density
 * - rho_i^adv - SUM_j(mass_j * (SUM_j(d_ij * p_j^l) - d_jj*p_j -
 * SUM_k!=i(d_jk*p_k^l))) dot W_ij) */
__global__ void updatePressureKernel(
    const float2* __restrict__ d_pos, const float* __restrict__ d_density_adv,
    const float* __restrict__ d_diag_coeff, float* __restrict__ d_pressure,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices,
    const float2* __restrict__ d_sum_movement_pressure,
    const float2* __restrict__ d_grad_coup_vect, int num_particles,
    SimulationParams params);

/* Computes rho_i^l for the next iteration of the loop.
 *
 * Equation Density: rho_i^l = rho_i^adv + a_ii * p_i^l + SUM_j(d_ij * p_j^l) */
__global__ void
updateDensityKernel(const float2* __restrict__ d_pos,
                    const float* __restrict__ d_density_adv,
                    const float* __restrict__ d_diag_coeff,
                    const float* __restrict__ d_pressure,
                    float* __restrict__ d_density_predicted,
                    const unsigned int* __restrict__ d_sorted_values,
                    const unsigned int* __restrict__ d_cell_start_indices,
                    const unsigned int* __restrict__ d_cell_end_indices,
                    const float2* __restrict__ d_sum_movement_pressure,
                    const float2* __restrict__ d_grad_coup_vect,
                    int num_particles, SimulationParams params);