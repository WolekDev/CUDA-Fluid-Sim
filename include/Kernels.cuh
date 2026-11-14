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

/* Compute Advection and Gradient */
__global__ void computeAdvectionAndGradientKernel(
    const float2* __restrict__ d_pos, const float2* __restrict__ d_vel,
    const float2* __restrict__ d_force_ext, float2* __restrict__ d_vel_adv,
    const float* __restrict__ d_density, float2* __restrict__ d_grad_coup_vect,
    const unsigned int* __restrict__ d_sorted_values,
    const unsigned int* __restrict__ d_cell_start_indices,
    const unsigned int* __restrict__ d_cell_end_indices, int num_particles,
    SimulationParams params, float delta_time);