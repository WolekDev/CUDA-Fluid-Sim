#pragma once

#include <cuda_runtime.h>
#include <vector>

/* Scalar stored in the packed particle buffer and used by the color ramp. */
enum class ParticleColorMode { Velocity = 0, Pressure = 1 };

struct ScalarRange {
    float min_value = 0.0f;
    float max_value = 1.0f;
};

/* Averages over fluid particles inside an axial slab. */
struct ProbeStats {
    int count = 0;
    float mean_speed = 0.0f;
    float mean_axial_speed = 0.0f;
    float mean_pressure = 0.0f;
    float mean_density = 0.0f;
};

/* Packs fluid particles to float4(x, y, z, scalar) and reduces a probe slab.
 * Scratch stays on the device and grows to the simulation capacity. */
class RenderPacker {
public:
    RenderPacker() = default;
    ~RenderPacker();

    RenderPacker(const RenderPacker&) = delete;
    RenderPacker& operator=(const RenderPacker&) = delete;

    void ensureCapacity(int max_particles);

    /* dst holds count packed particles. color selects the scalar in w. */
    void pack(const float3* positions, const float3* velocities,
              const float* pressures, float4* dst, int count,
              ParticleColorMode color);

    /* Min and max of the packed scalar. Empty input yields [0, 1]. */
    ScalarRange scalarRange(const float4* packed, int count);

    /* Particles whose axial coordinate is within half_width of station. */
    ProbeStats probe(const float3* positions, const float3* velocities,
                     const float* pressures, const float* densities, int count,
                     float3 origin, float3 axis, float station, float half_width);

private:
    int capacity_ = 0;
    float* d_block_min_ = nullptr;
    float* d_block_max_ = nullptr;
    int* d_probe_count_ = nullptr;
    float* d_probe_speed_ = nullptr;
    float* d_probe_axial_ = nullptr;
    float* d_probe_pressure_ = nullptr;
    float* d_probe_density_ = nullptr;

    std::vector<float> h_block_min_;
    std::vector<float> h_block_max_;
    std::vector<int> h_probe_count_;
    std::vector<float> h_probe_speed_;
    std::vector<float> h_probe_axial_;
    std::vector<float> h_probe_pressure_;
    std::vector<float> h_probe_density_;

    void release();
};
