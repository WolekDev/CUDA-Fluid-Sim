#ifndef SIMULATION_PARAMS_CUH
#define SIMULATION_PARAMS_CUH

#include <cuda_runtime.h>

/* A struct to combine several useful paramaters together for use in kernels. */
struct SimulationParams {
    // Physics & Simulation Constants
    float2 gravity_vector;
    float smoothing_radius;
    float rest_density;
    float time_step;
    float particle_mass;
    float viscosity;

    // Spatial Domain & Grid Parameters for GPU Hashing
    float domain_min_x;
    float domain_min_y;
    int grid_width;
    int grid_height;
};

#endif // SIMULATION_PARAMS_CUH