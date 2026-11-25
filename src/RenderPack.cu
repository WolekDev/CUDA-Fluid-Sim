#include "RenderPack.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

constexpr int kBlockSize = 256;

void checkCuda(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(error));
    }
}

__device__ float velocityMagnitude(float3 velocity) {
    return sqrtf(velocity.x * velocity.x + velocity.y * velocity.y +
                 velocity.z * velocity.z);
}

__global__ void packParticlesKernel(const float3* positions,
                                    const float3* velocities,
                                    const float* pressures, float4* packed,
                                    int count, int color_mode) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;

    const float3 position = positions[index];
    const float scalar = color_mode == 0 ? velocityMagnitude(velocities[index])
                                          : pressures[index];
    packed[index] = make_float4(position.x, position.y, position.z, scalar);
}

__global__ void scalarRangeKernel(const float4* packed, int count,
                                  float* block_min, float* block_max) {
    __shared__ float shared_min[kBlockSize];
    __shared__ float shared_max[kBlockSize];

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    float lo = 1.0e30f;
    float hi = -1.0e30f;
    if (index < count) {
        const float scalar = packed[index].w;
        lo = scalar;
        hi = scalar;
    }
    shared_min[threadIdx.x] = lo;
    shared_max[threadIdx.x] = hi;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_min[threadIdx.x] =
                fminf(shared_min[threadIdx.x], shared_min[threadIdx.x + stride]);
            shared_max[threadIdx.x] =
                fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_min[blockIdx.x] = shared_min[0];
        block_max[blockIdx.x] = shared_max[0];
    }
}

__global__ void probeKernel(const float3* positions, const float3* velocities,
                            const float* pressures, const float* densities,
                            int count, float3 origin, float3 axis, float station,
                            float half_width, int* block_count, float* block_speed,
                            float* block_axial, float* block_pressure,
                            float* block_density) {
    __shared__ int shared_count[kBlockSize];
    __shared__ float shared_speed[kBlockSize];
    __shared__ float shared_axial[kBlockSize];
    __shared__ float shared_pressure[kBlockSize];
    __shared__ float shared_density[kBlockSize];

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    int local_count = 0;
    float local_speed = 0.0f;
    float local_axial = 0.0f;
    float local_pressure = 0.0f;
    float local_density = 0.0f;
    if (index < count) {
        const float3 position = positions[index];
        const float axial_position =
            (position.x - origin.x) * axis.x + (position.y - origin.y) * axis.y +
            (position.z - origin.z) * axis.z;
        if (fabsf(axial_position - station) <= half_width) {
            const float3 velocity = velocities[index];
            local_count = 1;
            local_speed = velocityMagnitude(velocity);
            local_axial = velocity.x * axis.x + velocity.y * axis.y +
                          velocity.z * axis.z;
            local_pressure = pressures[index];
            local_density = densities[index];
        }
    }

    shared_count[threadIdx.x] = local_count;
    shared_speed[threadIdx.x] = local_speed;
    shared_axial[threadIdx.x] = local_axial;
    shared_pressure[threadIdx.x] = local_pressure;
    shared_density[threadIdx.x] = local_density;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_count[threadIdx.x] += shared_count[threadIdx.x + stride];
            shared_speed[threadIdx.x] += shared_speed[threadIdx.x + stride];
            shared_axial[threadIdx.x] += shared_axial[threadIdx.x + stride];
            shared_pressure[threadIdx.x] += shared_pressure[threadIdx.x + stride];
            shared_density[threadIdx.x] += shared_density[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_count[blockIdx.x] = shared_count[0];
        block_speed[blockIdx.x] = shared_speed[0];
        block_axial[blockIdx.x] = shared_axial[0];
        block_pressure[blockIdx.x] = shared_pressure[0];
        block_density[blockIdx.x] = shared_density[0];
    }
}

} // namespace

RenderPacker::~RenderPacker() { release(); }

void RenderPacker::release() {
    cudaFree(d_block_min_);
    cudaFree(d_block_max_);
    cudaFree(d_probe_count_);
    cudaFree(d_probe_speed_);
    cudaFree(d_probe_axial_);
    cudaFree(d_probe_pressure_);
    cudaFree(d_probe_density_);
    d_block_min_ = nullptr;
    d_block_max_ = nullptr;
    d_probe_count_ = nullptr;
    d_probe_speed_ = nullptr;
    d_probe_axial_ = nullptr;
    d_probe_pressure_ = nullptr;
    d_probe_density_ = nullptr;
    capacity_ = 0;
}

void RenderPacker::ensureCapacity(int max_particles) {
    if (max_particles <= capacity_)
        return;
    release();
    const int blocks = (std::max(max_particles, 1) + kBlockSize - 1) / kBlockSize;
    checkCuda(cudaMalloc(&d_block_min_, blocks * sizeof(float)),
              "range min alloc");
    checkCuda(cudaMalloc(&d_block_max_, blocks * sizeof(float)),
              "range max alloc");
    checkCuda(cudaMalloc(&d_probe_count_, blocks * sizeof(int)), "probe count alloc");
    checkCuda(cudaMalloc(&d_probe_speed_, blocks * sizeof(float)),
              "probe speed alloc");
    checkCuda(cudaMalloc(&d_probe_axial_, blocks * sizeof(float)),
              "probe axial alloc");
    checkCuda(cudaMalloc(&d_probe_pressure_, blocks * sizeof(float)),
              "probe pressure alloc");
    checkCuda(cudaMalloc(&d_probe_density_, blocks * sizeof(float)),
              "probe density alloc");
    h_block_min_.resize(blocks);
    h_block_max_.resize(blocks);
    h_probe_count_.resize(blocks);
    h_probe_speed_.resize(blocks);
    h_probe_axial_.resize(blocks);
    h_probe_pressure_.resize(blocks);
    h_probe_density_.resize(blocks);
    capacity_ = max_particles;
}

void RenderPacker::pack(const float3* positions, const float3* velocities,
                        const float* pressures, float4* dst, int count,
                        ParticleColorMode color) {
    if (count <= 0)
        return;
    const int blocks = (count + kBlockSize - 1) / kBlockSize;
    packParticlesKernel<<<blocks, kBlockSize>>>(
        positions, velocities, pressures, dst, count, static_cast<int>(color));
    checkCuda(cudaGetLastError(), "pack particles");
}

ScalarRange RenderPacker::scalarRange(const float4* packed, int count) {
    if (count <= 0)
        return {};
    const int blocks = (count + kBlockSize - 1) / kBlockSize;
    scalarRangeKernel<<<blocks, kBlockSize>>>(packed, count, d_block_min_,
                                              d_block_max_);
    checkCuda(cudaGetLastError(), "scalar range");
    checkCuda(cudaMemcpy(h_block_min_.data(), d_block_min_, blocks * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "scalar range min copy");
    checkCuda(cudaMemcpy(h_block_max_.data(), d_block_max_, blocks * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "scalar range max copy");

    float lo = h_block_min_[0];
    float hi = h_block_max_[0];
    for (int block = 1; block < blocks; ++block) {
        lo = std::min(lo, h_block_min_[block]);
        hi = std::max(hi, h_block_max_[block]);
    }
    if (!(hi > lo)) {
        lo -= 0.5f;
        hi += 0.5f;
    }
    return {lo, hi};
}

ProbeStats RenderPacker::probe(const float3* positions, const float3* velocities,
                               const float* pressures, const float* densities,
                               int count, float3 origin, float3 axis, float station,
                               float half_width) {
    ProbeStats stats;
    if (count <= 0)
        return stats;

    const int blocks = (count + kBlockSize - 1) / kBlockSize;
    probeKernel<<<blocks, kBlockSize>>>(
        positions, velocities, pressures, densities, count, origin, axis, station,
        half_width, d_probe_count_, d_probe_speed_, d_probe_axial_, d_probe_pressure_,
        d_probe_density_);
    checkCuda(cudaGetLastError(), "probe slab");
    checkCuda(cudaMemcpy(h_probe_count_.data(), d_probe_count_, blocks * sizeof(int),
                         cudaMemcpyDeviceToHost),
              "probe count copy");
    checkCuda(cudaMemcpy(h_probe_speed_.data(), d_probe_speed_, blocks * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "probe speed copy");
    checkCuda(cudaMemcpy(h_probe_axial_.data(), d_probe_axial_, blocks * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "probe axial copy");
    checkCuda(cudaMemcpy(h_probe_pressure_.data(), d_probe_pressure_,
                         blocks * sizeof(float), cudaMemcpyDeviceToHost),
              "probe pressure copy");
    checkCuda(cudaMemcpy(h_probe_density_.data(), d_probe_density_,
                         blocks * sizeof(float), cudaMemcpyDeviceToHost),
              "probe density copy");

    double speed = 0.0;
    double axial = 0.0;
    double pressure = 0.0;
    double density = 0.0;
    for (int block = 0; block < blocks; ++block) {
        stats.count += h_probe_count_[block];
        speed += h_probe_speed_[block];
        axial += h_probe_axial_[block];
        pressure += h_probe_pressure_[block];
        density += h_probe_density_[block];
    }
    if (stats.count > 0) {
        const double samples = stats.count;
        stats.mean_speed = static_cast<float>(speed / samples);
        stats.mean_axial_speed = static_cast<float>(axial / samples);
        stats.mean_pressure = static_cast<float>(pressure / samples);
        stats.mean_density = static_cast<float>(density / samples);
    }
    return stats;
}
