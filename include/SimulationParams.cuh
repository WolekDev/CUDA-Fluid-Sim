#ifndef SIMULATION_PARAMS_CUH
#define SIMULATION_PARAMS_CUH

#include <cuda_runtime.h>

/* A struct to combine several useful parameters together for use in kernels. */
struct SimulationParams {
    // Physics & Simulation Constants
    float3 gravity_vector;
    float smoothing_radius;
    float rest_density;
    float time_step;
    float particle_mass;
    // Dynamic viscosity in mPa·s. Water at 25 C is 0.89.
    // Eq. (4.4) evaluates this as Pa·s after multiplying by 0.001.
    float viscosity;
    float relaxation_factor;

    // Spatial Domain & Grid Parameters for GPU Hashing
    float domain_min_x;
    float domain_min_y;
    float domain_min_z;
    int grid_width;
    int grid_height;
    int grid_depth;
};

#endif // SIMULATION_PARAMS_CUH
