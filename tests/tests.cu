#include "FluidSimulation.cuh"
#include "Kernels.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

bool near(float actual, float expected, float relative = 1.0e-4f,
          float absolute = 1.0e-4f) {
    return std::fabs(actual - expected) <=
           absolute + relative * std::max(std::fabs(actual), std::fabs(expected));
}

void expectNear(float actual, float expected, const std::string& message) {
    if (!near(actual, expected)) {
        throw std::runtime_error(message + " actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
}

void expectNear3(float3 actual, float3 expected, const std::string& message) {
    expectNear(actual.x, expected.x, message + ".x");
    expectNear(actual.y, expected.y, message + ".y");
    expectNear(actual.z, expected.z, message + ".z");
}

/* Independent transcription of thesis Eq. (4.1), p. 10. */
double poly6Reference(double distance, double support) {
    if (distance < 0.0 || distance >= support || support <= 0.0)
        return 0.0;
    const double s = distance / support;
    const double support3 = support * support * support;
    if (s < 0.5) {
        const double radial = 1.0 - 6.0 * s * s + 6.0 * s * s * s;
        return 8.0 * radial / (M_PI * support3);
    }
    const double one_minus_s = 1.0 - s;
    return 16.0 * one_minus_s * one_minus_s * one_minus_s / (M_PI * support3);
}

/* Independent transcription of thesis Eq. (4.3), p. 10. */
double viscosityReference(double distance, double support) {
    if (distance < 0.0 || distance >= support || support <= 0.0)
        return 0.0;
    const double support2 = support * support;
    const double support3 = support2 * support;
    const double support6 = support3 * support3;
    return 45.0 * (support - distance) / (M_PI * support6);
}

/* Independent transcription of thesis Eq. (4.2), p. 10. */
double spikyReference(double distance, double support) {
    if (distance <= 0.0 || distance >= support || support <= 0.0)
        return 0.0;
    const double s2 = (distance * distance) / (support * support);
    const double radial = 1.0 - s2;
    const double support2 = support * support;
    const double support5 = support2 * support2 * support;
    const double coeff = -8.0 * 3465.0 / (512.0 * M_PI * support5);
    return coeff * radial * radial * radial;
}

int cellCoordinate(float position, float domain_min, float cell_size) {
    return static_cast<int>(std::floor((position - domain_min) / cell_size));
}

int clampCell(int cell, int count) {
    return std::max(0, std::min(cell, count - 1));
}

struct ReferenceState {
    std::vector<float3> vel_adv;
    std::vector<float3> dii;
    std::vector<float3> sum_movement;
    std::vector<float> density;
    std::vector<float> density_adv;
    std::vector<float> density_predicted;
    std::vector<float> pressure;
    std::vector<float> diag;
    std::vector<float3> position;
    std::vector<float3> velocity;
};

/* Same 3x3x3 cell walk as the kernels, with particles kept in index order.
 * That matches a stable sort of (cell key, particle index). */
template <typename Visitor>
void forNeighbors(const std::vector<float3>& pos, const SimulationParams& params,
                  const std::vector<std::vector<int>>& cells, int particle,
                  Visitor visitor) {
    const float support = params.smoothing_radius;
    const int cx = cellCoordinate(pos[particle].x, params.domain_min_x, support);
    const int cy = cellCoordinate(pos[particle].y, params.domain_min_y, support);
    const int cz = cellCoordinate(pos[particle].z, params.domain_min_z, support);
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
                for (int neighbor : cells[key])
                    visitor(neighbor);
            }
        }
    }
}

std::vector<std::vector<int>> buildCells(const std::vector<float3>& pos,
                                         const SimulationParams& params) {
    const int count =
        params.grid_width * params.grid_height * params.grid_depth;
    std::vector<std::vector<int>> cells(count);
    const float support = params.smoothing_radius;
    for (int index = 0; index < static_cast<int>(pos.size()); ++index) {
        int cx = cellCoordinate(pos[index].x, params.domain_min_x, support);
        int cy = cellCoordinate(pos[index].y, params.domain_min_y, support);
        int cz = cellCoordinate(pos[index].z, params.domain_min_z, support);
        cx = clampCell(cx, params.grid_width);
        cy = clampCell(cy, params.grid_height);
        cz = clampCell(cz, params.grid_depth);
        cells[computeCellKey(cx, cy, cz, params.grid_width, params.grid_height)]
            .push_back(index);
    }
    return cells;
}

float3 pairwiseDji(float3 grad_w_ji, float mass, float density_i, float dt2) {
    const float scale = -dt2 * mass / (density_i * density_i);
    return make_float3(scale * grad_w_ji.x, scale * grad_w_ji.y,
                       scale * grad_w_ji.z);
}

/* Host transcription of one IISPH step.
 * Density: Ihmsen p. 2. Advection and d_ii: Algorithm 1, p. 3.
 * rho_adv: Eq. (4). a_ii: Eq. (12). d_ji: thesis Eq. (4.11).
 * Pressure: Eq. (13) / thesis Eq. (4.8). Density prediction: thesis Eq. (4.7). */
ReferenceState solveReference(const std::vector<float3>& pos,
                              const SimulationParams& params, float dt,
                              int max_iterations, float density_threshold,
                              const std::vector<float3>& initial_velocity = {}) {
    const int count = static_cast<int>(pos.size());
    const float support = params.smoothing_radius;
    const float support2 = support * support;
    const float mass = params.particle_mass;
    const float dt2 = dt * dt;
    const auto cells = buildCells(pos, params);

    ReferenceState state;
    state.density.assign(count, 0.0f);
    state.vel_adv.assign(count, make_float3(0.0f, 0.0f, 0.0f));
    state.dii.assign(count, make_float3(0.0f, 0.0f, 0.0f));
    state.density_adv.assign(count, 0.0f);
    state.diag.assign(count, 0.0f);
    state.pressure.assign(count, 0.0f);
    state.density_predicted.assign(count, 0.0f);
    state.sum_movement.assign(count, make_float3(0.0f, 0.0f, 0.0f));
    std::vector<float> pressure_prev(count, 0.0f);
    std::vector<float3> vel(count, make_float3(0.0f, 0.0f, 0.0f));
    if (!initial_velocity.empty()) {
        expect(initial_velocity.size() == pos.size(),
               "reference velocities do not match the particle count");
        vel = initial_velocity;
    }

    for (int i = 0; i < count; ++i) {
        forNeighbors(pos, params, cells, i, [&](int j) {
            const float3 rij = make_float3(pos[i].x - pos[j].x, pos[i].y - pos[j].y,
                                           pos[i].z - pos[j].z);
            const float r2 = dot3(rij, rij);
            if (r2 <= support2)
                state.density[i] += mass * computePoly6(std::sqrt(r2), support);
        });
    }

    for (int i = 0; i < count; ++i) {
        float3 accumulated = make_float3(0.0f, 0.0f, 0.0f);
        float3 f_visc = make_float3(0.0f, 0.0f, 0.0f);
        if (state.density[i] > 0.0f) {
            const float scale = mass / (state.density[i] * state.density[i]);
            const float mass_over_density = mass / state.density[i];
            const float visc_scale =
                mass_over_density * mass_over_density *
                dynamicViscosityPascalSeconds(params.viscosity);
            forNeighbors(pos, params, cells, i, [&](int j) {
                const float3 rij =
                    make_float3(pos[i].x - pos[j].x, pos[i].y - pos[j].y,
                                pos[i].z - pos[j].z);
                const float r2 = dot3(rij, rij);
                if (r2 >= support2)
                    return;
                const float distance = std::sqrt(r2);
                if (distance > 0.0f) {
                    const float3 grad =
                        computeSpikyGradient(rij, distance, support);
                    accumulated.x += scale * grad.x;
                    accumulated.y += scale * grad.y;
                    accumulated.z += scale * grad.z;
                }
                const float coeff = visc_scale * computeViscosity(distance, support);
                f_visc.x += -(vel[i].x - vel[j].x) * coeff;
                f_visc.y += -(vel[i].y - vel[j].y) * coeff;
                f_visc.z += -(vel[i].z - vel[j].z) * coeff;
            });
        }
        state.vel_adv[i] = make_float3(
            vel[i].x + dt * (params.gravity_vector.x + f_visc.x / mass),
            vel[i].y + dt * (params.gravity_vector.y + f_visc.y / mass),
            vel[i].z + dt * (params.gravity_vector.z + f_visc.z / mass));
        state.dii[i] =
            make_float3(-dt2 * accumulated.x, -dt2 * accumulated.y,
                        -dt2 * accumulated.z);
    }

    for (int i = 0; i < count; ++i) {
        state.pressure[i] = 0.5f * pressure_prev[i];
        float density_sum = 0.0f;
        float a_ii = 0.0f;
        if (state.density[i] > 0.0f) {
            forNeighbors(pos, params, cells, i, [&](int j) {
                const float3 rij =
                    make_float3(pos[i].x - pos[j].x, pos[i].y - pos[j].y,
                                pos[i].z - pos[j].z);
                const float r2 = dot3(rij, rij);
                if (r2 <= 0.0f || r2 >= support2)
                    return;
                const float distance = std::sqrt(r2);
                const float3 grad_ij =
                    computeSpikyGradient(rij, distance, support);
                const float3 grad_ji = computeSpikyGradient(
                    make_float3(-rij.x, -rij.y, -rij.z), distance, support);
                const float3 relative = make_float3(
                    state.vel_adv[i].x - state.vel_adv[j].x,
                    state.vel_adv[i].y - state.vel_adv[j].y,
                    state.vel_adv[i].z - state.vel_adv[j].z);
                density_sum += mass * dot3(relative, grad_ij);
                const float3 d_ji =
                    pairwiseDji(grad_ji, mass, state.density[i], dt2);
                const float3 difference =
                    make_float3(state.dii[i].x - d_ji.x, state.dii[i].y - d_ji.y,
                                state.dii[i].z - d_ji.z);
                a_ii += mass * dot3(difference, grad_ij);
            });
        }
        state.density_adv[i] = state.density[i] + dt * density_sum;
        state.diag[i] = a_ii;
    }

    int iteration = 0;
    float density_average = 0.0f;
    while ((density_average - params.rest_density > density_threshold ||
            iteration < 2) &&
           iteration < max_iterations) {
        for (int i = 0; i < count; ++i) {
            float3 sum = make_float3(0.0f, 0.0f, 0.0f);
            forNeighbors(pos, params, cells, i, [&](int j) {
                const float3 rij =
                    make_float3(pos[i].x - pos[j].x, pos[i].y - pos[j].y,
                                pos[i].z - pos[j].z);
                const float r2 = dot3(rij, rij);
                if (r2 <= 0.0f || r2 >= support2 || state.density[j] <= 0.0f)
                    return;
                const float coeff = -state.pressure[j] * mass /
                                    (state.density[j] * state.density[j]);
                const float3 grad =
                    computeSpikyGradient(rij, std::sqrt(r2), support);
                sum.x += coeff * grad.x;
                sum.y += coeff * grad.y;
                sum.z += coeff * grad.z;
            });
            state.sum_movement[i] = make_float3(dt2 * sum.x, dt2 * sum.y, dt2 * sum.z);
        }

        std::vector<float> pressure_next = state.pressure;
        for (int i = 0; i < count; ++i) {
            float neighbor_sum = 0.0f;
            if (state.density[i] > 0.0f) {
                forNeighbors(pos, params, cells, i, [&](int j) {
                    if (j == i)
                        return;
                    const float3 rij =
                        make_float3(pos[i].x - pos[j].x, pos[i].y - pos[j].y,
                                    pos[i].z - pos[j].z);
                    const float r2 = dot3(rij, rij);
                    if (r2 <= 0.0f || r2 >= support2)
                        return;
                    const float distance = std::sqrt(r2);
                    const float3 grad_ij =
                        computeSpikyGradient(rij, distance, support);
                    const float3 grad_ji = computeSpikyGradient(
                        make_float3(-rij.x, -rij.y, -rij.z), distance, support);
                    const float3 d_ji =
                        pairwiseDji(grad_ji, mass, state.density[i], dt2);
                    const float3 excluded = make_float3(
                        state.sum_movement[j].x - d_ji.x * state.pressure[i],
                        state.sum_movement[j].y - d_ji.y * state.pressure[i],
                        state.sum_movement[j].z - d_ji.z * state.pressure[i]);
                    const float3 inner = make_float3(
                        state.sum_movement[i].x - state.dii[j].x * state.pressure[j] -
                            excluded.x,
                        state.sum_movement[i].y - state.dii[j].y * state.pressure[j] -
                            excluded.y,
                        state.sum_movement[i].z - state.dii[j].z * state.pressure[j] -
                            excluded.z);
                    neighbor_sum += mass * dot3(inner, grad_ij);
                });
            }
            float updated = state.pressure[i];
            if (state.diag[i] > 1.0e-12f || state.diag[i] < -1.0e-12f) {
                updated = (1.0f - params.relaxation_factor) * state.pressure[i] +
                          (params.relaxation_factor / state.diag[i]) *
                              (params.rest_density - state.density_adv[i] -
                               neighbor_sum);
                updated = std::max(0.0f, updated);
            }
            pressure_next[i] = updated;
        }
        state.pressure.swap(pressure_next);

        float density_sum = 0.0f;
        for (int i = 0; i < count; ++i) {
            float neighbor_sum = 0.0f;
            if (state.density[i] > 0.0f) {
                forNeighbors(pos, params, cells, i, [&](int j) {
                    if (j == i)
                        return;
                    const float3 rij =
                        make_float3(pos[i].x - pos[j].x, pos[i].y - pos[j].y,
                                    pos[i].z - pos[j].z);
                    const float r2 = dot3(rij, rij);
                    if (r2 <= 0.0f || r2 >= support2)
                        return;
                    const float distance = std::sqrt(r2);
                    const float3 grad_ij =
                        computeSpikyGradient(rij, distance, support);
                    const float3 grad_ji = computeSpikyGradient(
                        make_float3(-rij.x, -rij.y, -rij.z), distance, support);
                    const float3 d_ji =
                        pairwiseDji(grad_ji, mass, state.density[i], dt2);
                    const float3 excluded = make_float3(
                        state.sum_movement[j].x - d_ji.x * state.pressure[i],
                        state.sum_movement[j].y - d_ji.y * state.pressure[i],
                        state.sum_movement[j].z - d_ji.z * state.pressure[i]);
                    const float3 inner = make_float3(
                        state.sum_movement[i].x - state.dii[j].x * state.pressure[j] -
                            excluded.x,
                        state.sum_movement[i].y - state.dii[j].y * state.pressure[j] -
                            excluded.y,
                        state.sum_movement[i].z - state.dii[j].z * state.pressure[j] -
                            excluded.z);
                    neighbor_sum += mass * dot3(inner, grad_ij);
                });
            }
            state.density_predicted[i] =
                state.density_adv[i] + state.diag[i] * state.pressure[i] +
                neighbor_sum;
            density_sum += state.density_predicted[i];
        }
        density_average = density_sum / static_cast<float>(count);
        ++iteration;
    }

    // Eq. (4.12) / Ihmsen Eq. (2), then Algorithm 1 lines 21–22.
    state.position.resize(count);
    state.velocity.resize(count);
    for (int i = 0; i < count; ++i) {
        float3 pressure_sum = make_float3(0.0f, 0.0f, 0.0f);
        if (state.density[i] > 0.0f) {
            const float inv_density_i2 =
                1.0f / (state.density[i] * state.density[i]);
            forNeighbors(pos, params, cells, i, [&](int j) {
                const float3 rij =
                    make_float3(pos[i].x - pos[j].x, pos[i].y - pos[j].y,
                                pos[i].z - pos[j].z);
                const float r2 = dot3(rij, rij);
                if (r2 <= 0.0f || r2 >= support2 || state.density[j] <= 0.0f)
                    return;
                const float coeff =
                    mass * (state.pressure[i] * inv_density_i2 +
                            state.pressure[j] /
                                (state.density[j] * state.density[j]));
                const float3 grad =
                    computeSpikyGradient(rij, std::sqrt(r2), support);
                pressure_sum.x += coeff * grad.x;
                pressure_sum.y += coeff * grad.y;
                pressure_sum.z += coeff * grad.z;
            });
        }
        state.velocity[i] = make_float3(
            state.vel_adv[i].x - dt * pressure_sum.x,
            state.vel_adv[i].y - dt * pressure_sum.y,
            state.vel_adv[i].z - dt * pressure_sum.z);
        state.position[i] = make_float3(
            pos[i].x + dt * state.velocity[i].x,
            pos[i].y + dt * state.velocity[i].y,
            pos[i].z + dt * state.velocity[i].z);
    }

    return state;
}

const char* kConfig = "config/test_config.json";

std::vector<float3> sampleParticles() {
    // Two particles share a support radius along x. The third is isolated.
    return {make_float3(0.50f, 0.50f, 0.50f), make_float3(0.51f, 0.50f, 0.50f),
            make_float3(0.90f, 0.90f, 0.90f)};
}

} // namespace

void testInitializationAndMemory() {
    std::cout << "[TEST] testInitializationAndMemory\n";
    FluidSimulation sim;
    sim.initialize(kConfig);

    expect(sim.getMaxParticles() == 1000, "max_particles was not loaded");
    expect(sim.getMaxIterations() == 5, "max_iterations was not loaded");
    expect(sim.arePointersAllocated(), "device buffers were not allocated");

    const SimulationParams& params = sim.getParams();
    expect(params.smoothing_radius == 0.04f, "smoothing radius mismatch");
    expect(params.grid_width == 25 && params.grid_height == 25 &&
               params.grid_depth == 25,
           "3D grid dimensions are wrong for a 1m domain and h = 0.04");
    expect(params.gravity_vector.x == 0.0f && params.gravity_vector.y == -9.81f &&
               params.gravity_vector.z == 0.0f,
           "gravity is not a 3D vector");
    expect(params.domain_min_z == 0.0f, "domain_min_z was not loaded");
    expect(params.viscosity == 0.89f,
           "viscosity should load as 0.89 mPa*s for water at 25 C");

    sim.cleanup();
    expect(!sim.arePointersAllocated(), "cleanup did not release the buffers");
    std::cout << "[Pass] initialization allocates a 3D grid and releases it\n";
}

void testSpatialHashingAndSort() {
    std::cout << "\n[TEST] testSpatialHashingAndSort\n";
    FluidSimulation sim;
    sim.initialize(kConfig);
    const SimulationParams& params = sim.getParams();

    const std::vector<float3> positions = {make_float3(0.20f, 0.20f, 0.20f),
                                           make_float3(0.80f, 0.20f, 0.20f),
                                           make_float3(0.20f, 0.80f, 0.80f)};
    sim.uploadTestParticles(positions);
    sim.updateSpatialHash();
    const std::vector<unsigned int> keys = sim.copyKeysToHost();

    std::vector<unsigned int> expected;
    for (const float3& position : positions) {
        const int cx =
            clampCell(cellCoordinate(position.x, params.domain_min_x,
                                     params.smoothing_radius),
                      params.grid_width);
        const int cy =
            clampCell(cellCoordinate(position.y, params.domain_min_y,
                                     params.smoothing_radius),
                      params.grid_height);
        const int cz =
            clampCell(cellCoordinate(position.z, params.domain_min_z,
                                     params.smoothing_radius),
                      params.grid_depth);
        expected.push_back(computeCellKey(cx, cy, cz, params.grid_width,
                                           params.grid_height));
    }
    std::sort(expected.begin(), expected.end());

    expect(expected[0] < expected[1] && expected[1] < expected[2],
           "fixture cells do not exercise x and z");
    for (size_t i = 0; i < expected.size(); ++i) {
        expect(keys[i] == expected[i], "sorted 3D cell keys do not match");
        if (i + 1 < expected.size())
            expect(keys[i] <= keys[i + 1], "cell keys are not monotonic");
    }
    sim.cleanup();
    std::cout << "[Pass] 3D hashes sort by x, then y, then z\n";
}

void testPoly6Kernel() {
    std::cout << "\n[TEST] testPoly6Kernel\n";
    const float support = 0.04f;
    const float samples[] = {0.0f, 0.25f * support, 0.5f * support,
                             0.75f * support, support, support * 1.25f};

    for (float distance : samples) {
        const float actual = computePoly6(distance, support);
        const float expected =
            static_cast<float>(poly6Reference(distance, support));
        expectNear(actual, expected, "W_poly6 disagrees with Eq. (4.1)");
    }

    const float at_center = computePoly6(0.0f, support);
    const float at_half = computePoly6(0.5f * support, support);
    const double half = 0.5 * static_cast<double>(support);
    const double support3 = static_cast<double>(support) * support * support;
    const double first_branch = 8.0 * (1.0 - 6.0 * 0.25 + 6.0 * 0.125) / (M_PI * support3);
    const double second_branch = 16.0 * std::pow(1.0 - 0.5, 3.0) / (M_PI * support3);
    expect(at_center > at_half && at_half > 0.0f,
           "W_poly6 should decrease from the center to s = 0.5");
    expect(std::fabs(first_branch - second_branch) <= 1.0e-9 * first_branch,
           "Eq. (4.1) is discontinuous at s = 0.5");
    expectNear(at_half, static_cast<float>(poly6Reference(half, support)),
               "s = 0.5 should use the second piece of Eq. (4.1)");
    expect(computePoly6(support, support) == 0.0f, "W_poly6 must be 0 at d = R");
    std::cout << "[Pass] W_poly6 matches thesis Eq. (4.1)\n";
}

void testSpikyKernel() {
    std::cout << "\n[TEST] testSpikyKernel\n";
    const float support = 0.04f;
    expect(computeSpiky(0.0f, support) == 0.0f, "W_spiky must be 0 at d = 0");
    expect(computeSpiky(support, support) == 0.0f, "W_spiky must be 0 at d = R");
    expect(computeSpiky(support * 1.1f, support) == 0.0f,
           "W_spiky must be 0 outside the support");

    const float distance = 0.5f * support;
    const float actual = computeSpiky(distance, support);
    const float expected = static_cast<float>(spikyReference(distance, support));
    expectNear(actual, expected, "W_spiky disagrees with Eq. (4.2)");
    expect(actual < 0.0f, "Eq. (4.2) is negative inside the support");

    const float3 offset = make_float3(0.01f, -0.004f, 0.006f);
    const float length = std::sqrt(dot3(offset, offset));
    const float3 gradient = computeSpikyGradient(offset, length, support);
    const float weight = computeSpiky(length, support);
    expectNear3(gradient,
                make_float3(offset.x * weight, offset.y * weight, offset.z * weight),
                "grad W is not the non-normalized distance times W_spiky");
    expect(gradient.x < 0.0f && gradient.y > 0.0f && gradient.z < 0.0f,
           "grad W should point opposite the positive offset because W_spiky < 0");
    std::cout << "[Pass] W_spiky and grad W match thesis Eq. (4.2)\n";
}

void testDensityKernel() {
    std::cout << "\n[TEST] testDensityKernel\n";
    FluidSimulation sim;
    sim.initialize(kConfig);
    const auto positions = sampleParticles();
    sim.uploadTestParticles(positions);
    sim.update(0.01f);

    const SimulationParams& params = sim.getParams();
    const auto densities = sim.copyDensitiesToHost();
    const double support = params.smoothing_radius;
    const double mass = params.particle_mass;
    const double self = poly6Reference(0.0, support);
    const double pair = poly6Reference(0.01, support);
    const float interacting =
        static_cast<float>(mass * (self + pair));
    const float isolated = static_cast<float>(mass * self);

    expect(densities.size() >= 3, "density buffer is too small");
    expectNear(densities[0], interacting, "particle 0 density");
    expectNear(densities[1], interacting, "particle 1 density");
    expectNear(densities[2], isolated, "isolated particle must keep only W(0)");
    expect(densities[0] > params.rest_density,
           "the close pair should be over the rest density");
    expect(densities[2] < params.rest_density,
           "one particle should be under the rest density");
    sim.cleanup();
    std::cout << "[Pass] density is sum_j m_j W_poly6, including the self term\n";
}

void testAdvectionAndGradient() {
    std::cout << "\n[TEST] testAdvectionAndGradient\n";
    FluidSimulation sim;
    sim.initialize(kConfig);
    const auto positions = sampleParticles();
    sim.uploadTestParticles(positions);
    const float dt = 0.01f;
    sim.update(dt);

    const SimulationParams& params = sim.getParams();
    const ReferenceState expected = solveReference(
        positions, params, dt, sim.getMaxIterations(),
        sim.getDensityErrorThreshold());
    const auto vel_adv = sim.copyVelAdvToHost();
    const auto dii = sim.copyGradCoupToHost();

    for (size_t i = 0; i < positions.size(); ++i) {
        expectNear3(vel_adv[i], expected.vel_adv[i],
                    "advected velocity " + std::to_string(i));
        expectNear3(dii[i], expected.dii[i],
                    "displacement d_ii " + std::to_string(i));
    }
    expect(vel_adv[0].y < 0.0f && vel_adv[0].x == 0.0f && vel_adv[0].z == 0.0f,
           "gravity should change only the y component of velocity");
    expect(dii[0].x != 0.0f && dii[0].y == 0.0f && dii[0].z == 0.0f,
           "a pair separated on x should displace only on x");
    expect(dii[2].x == 0.0f && dii[2].y == 0.0f && dii[2].z == 0.0f,
           "an isolated particle has no displacement factor");
    sim.cleanup();
    std::cout << "[Pass] advection and d_ii match Algorithm 1\n";
}

void testViscosityKernel() {
    std::cout << "\n[TEST] testViscosityKernel\n";
    const float support = 0.04f;
    const float samples[] = {0.0f, 0.01f, 0.5f * support, support, support * 1.2f};
    for (float distance : samples) {
        const float actual = computeViscosity(distance, support);
        const float expected =
            static_cast<float>(viscosityReference(distance, support));
        expectNear(actual, expected, "W_viscosity disagrees with Eq. (4.3)");
    }

    expect(computeViscosity(0.0f, support) > computeViscosity(0.5f * support, support),
           "W_viscosity should be largest at d = 0");
    expect(computeViscosity(support, support) == 0.0f,
           "W_viscosity must be 0 at d = R");
    expectNear(dynamicViscosityPascalSeconds(0.89f), 0.00089f,
               "0.89 mPa*s must enter Eq. (4.4) as 0.00089 Pa*s");
    std::cout << "[Pass] W_viscosity matches thesis Eq. (4.3)\n";
}

void testViscosityForce() {
    std::cout << "\n[TEST] testViscosityForce\n";
    FluidSimulation sim;
    sim.initialize(kConfig);
    const auto positions = sampleParticles();
    const std::vector<float3> velocities = {make_float3(1.0f, 0.0f, 0.0f),
                                            make_float3(0.0f, 0.0f, 0.0f),
                                            make_float3(0.4f, 0.0f, 0.0f)};
    sim.uploadTestParticles(positions);
    sim.uploadTestVelocities(velocities);
    const float dt = 0.01f;
    sim.update(dt);

    const SimulationParams& params = sim.getParams();
    const ReferenceState expected = solveReference(
        positions, params, dt, sim.getMaxIterations(),
        sim.getDensityErrorThreshold(), velocities);
    const auto vel_adv = sim.copyVelAdvToHost();
    for (size_t i = 0; i < positions.size(); ++i) {
        expectNear3(vel_adv[i], expected.vel_adv[i],
                    "viscous advected velocity " + std::to_string(i));
    }

    // Direct Eq. (4.4) check for the moving particle against its one neighbor.
    const double support = params.smoothing_radius;
    const double mass = params.particle_mass;
    const double rho =
        mass * (poly6Reference(0.0, support) + poly6Reference(0.01, support));
    const double weight = viscosityReference(0.01, support);
    const double const_visc = static_cast<double>(params.viscosity) * 0.001;
    const double rel_vx = 1.0 - 0.0;
    const double force_x =
        -rel_vx * (mass / rho) * (mass / rho) * const_visc * weight;
    const float expected_vx =
        static_cast<float>(1.0 + dt * (params.gravity_vector.x + force_x / mass));
    expectNear(vel_adv[0].x, expected_vx, "Eq. (4.4) x-force on particle 0");
    expect(vel_adv[0].x < 1.0f, "viscosity should slow the faster particle");
    expect(vel_adv[1].x > 0.0f, "viscosity should drag the resting neighbor along");
    expectNear(vel_adv[2].x, velocities[2].x + dt * params.gravity_vector.x,
               "isolated particle has no viscous neighbor");
    sim.cleanup();
    std::cout << "[Pass] viscosity force matches thesis Eq. (4.4)\n";
}

void testInitPressureSolver() {
    std::cout << "\n[TEST] testInitPressureSolver\n";
    FluidSimulation sim;
    sim.initialize(kConfig);
    const auto positions = sampleParticles();
    sim.uploadTestParticles(positions);
    const float dt = 0.01f;
    sim.update(dt);

    const ReferenceState expected = solveReference(
        positions, sim.getParams(), dt, sim.getMaxIterations(),
        sim.getDensityErrorThreshold());
    const auto density_adv = sim.copyDensityAdvToHost();
    const auto diag = sim.copyDiagCoeffToHost();
    const auto density = sim.copyDensitiesToHost();

    for (size_t i = 0; i < positions.size(); ++i) {
        expectNear(density_adv[i], expected.density_adv[i],
                   "advected density " + std::to_string(i));
        expectNear(diag[i], expected.diag[i],
                   "diagonal a_ii " + std::to_string(i));
    }
    // Equal gravity on equal masses leaves the relative velocity at 0, so
    // Eq. (4) does not change density before the pressure loop.
    expectNear(density_adv[0], density[0], "relative velocity should be zero");
    expect(diag[0] != 0.0f, "interacting particle should have a nonzero a_ii");
    expect(diag[2] == 0.0f, "isolated particle should have a_ii = 0");
    sim.cleanup();
    std::cout << "[Pass] rho_adv, p^0, and a_ii match Eqs. (4) and (12)\n";
}

void testPressureSolve() {
    std::cout << "\n[TEST] testPressureSolve\n";
    FluidSimulation sim;
    sim.initialize(kConfig);
    const auto positions = sampleParticles();
    sim.uploadTestParticles(positions);
    const float dt = 0.01f;
    sim.update(dt);

    const ReferenceState expected = solveReference(
        positions, sim.getParams(), dt, sim.getMaxIterations(),
        sim.getDensityErrorThreshold());
    const auto pressure = sim.copyPressureToHost();

    for (size_t i = 0; i < positions.size(); ++i) {
        expect(pressure[i] >= 0.0f, "pressure was not clamped at 0");
        expectNear(pressure[i], expected.pressure[i],
                   "pressure " + std::to_string(i));
    }
    expect(pressure[0] > 0.0f && pressure[1] > 0.0f,
           "the over-dense pair should receive positive pressure");
    expect(pressure[2] == 0.0f,
           "the isolated particle has a_ii = 0, so its pressure stays 0");
    sim.cleanup();
    std::cout << "[Pass] relaxed Jacobi pressure matches Eq. (13) / Eq. (4.8)\n";
}

void testPredictedDensity() {
    std::cout << "\n[TEST] testPredictedDensity\n";
    FluidSimulation sim;
    sim.initialize(kConfig);
    const auto positions = sampleParticles();
    sim.uploadTestParticles(positions);
    const float dt = 0.01f;
    sim.update(dt);

    const ReferenceState expected = solveReference(
        positions, sim.getParams(), dt, sim.getMaxIterations(),
        sim.getDensityErrorThreshold());
    const auto predicted = sim.copyPredictedDensitiesToHost();
    const auto density_adv = sim.copyDensityAdvToHost();

    for (size_t i = 0; i < positions.size(); ++i) {
        expectNear(predicted[i], expected.density_predicted[i],
                   "predicted density " + std::to_string(i));
    }
    expect(predicted[0] > 0.0f && predicted[1] > 0.0f,
           "predicted density should stay positive");
    expect(std::fabs(predicted[0] - density_adv[0]) > 0.0f,
           "pressure should change the predicted density of the close pair");
    sim.cleanup();
    std::cout << "[Pass] predicted density matches thesis Eq. (4.7)\n";
}

void testIntegration() {
    std::cout << "[TEST] testIntegration\n";
    FluidSimulation sim;
    sim.initialize(kConfig);
    const auto positions = sampleParticles();
    sim.uploadTestParticles(positions);
    const float dt = 0.01f;
    sim.update(dt);

    const ReferenceState expected = solveReference(
        positions, sim.getParams(), dt, sim.getMaxIterations(),
        sim.getDensityErrorThreshold());
    const auto integrated_pos = sim.copyPositionsToHost();
    const auto integrated_vel = sim.copyVelocitiesToHost();

    expect(integrated_pos.size() == positions.size(), "position count changed");
    expect(integrated_vel.size() == positions.size(), "velocity count changed");
    for (size_t i = 0; i < positions.size(); ++i) {
        expectNear3(integrated_pos[i], expected.position[i],
                    "integrated position " + std::to_string(i));
        expectNear3(integrated_vel[i], expected.velocity[i],
                    "integrated velocity " + std::to_string(i));
    }

    expect(integrated_pos[0].x < positions[0].x,
           "positive pressure should push particle 0 away along -x");
    expect(integrated_pos[1].x > positions[1].x,
           "positive pressure should push particle 1 away along +x");
    expectNear(integrated_pos[2].x, positions[2].x,
               "isolated particle has no pressure force in x");
    expect(integrated_pos[2].y < positions[2].y,
           "gravity should move the isolated particle downward");
    sim.cleanup();
    std::cout << "[Pass] integration matches Eq. (4.12) and Algorithm 1\n";
}

void testPreviousPressureCopy() {
    std::cout << "\n[TEST] testPreviousPressureCopy\n";
    FluidSimulation sim;
    sim.initialize(kConfig);
    sim.uploadTestParticles(sampleParticles());
    sim.update(0.01f);

    const auto pressure = sim.copyPressureToHost();
    const auto previous = sim.copyPrevPressureToHost();
    expect(pressure.size() == previous.size(), "pressure buffers differ in size");
    for (size_t i = 0; i < pressure.size(); ++i)
        expect(pressure[i] == previous[i],
               "p(t) was not stored for the next frame's p^0 = 0.5 p(t - dt)");
    sim.cleanup();
    std::cout << "[Pass] end-of-step pressure is copied to the previous buffer\n";
}

int main(int argc, char** argv) {
    const std::pair<const char*, void (*)()> tests[] = {
        {"testInitializationAndMemory", testInitializationAndMemory},
        {"testSpatialHashingAndSort", testSpatialHashingAndSort},
        {"testPoly6Kernel", testPoly6Kernel},
        {"testSpikyKernel", testSpikyKernel},
        {"testDensityKernel", testDensityKernel},
        {"testAdvectionAndGradient", testAdvectionAndGradient},
        {"testViscosityKernel", testViscosityKernel},
        {"testViscosityForce", testViscosityForce},
        {"testInitPressureSolver", testInitPressureSolver},
        {"testPressureSolve", testPressureSolve},
        {"testPredictedDensity", testPredictedDensity},
        {"testIntegration", testIntegration},
        {"testPreviousPressureCopy", testPreviousPressureCopy},
    };

    try {
        if (argc > 1) {
            const std::string requested = argv[1];
            for (const auto& test : tests) {
                if (requested == test.first) {
                    test.second();
                    return 0;
                }
            }
            std::cerr << "Unknown test '" << requested << "'. Known tests:\n";
            for (const auto& test : tests)
                std::cerr << "  " << test.first << "\n";
            return 1;
        }

        for (const auto& test : tests)
            test.second();
        std::cout << "\n=== ALL TESTS PASSED ===\n";
    } catch (const std::exception& error) {
        std::cerr << "[Fail] " << error.what() << "\n";
        return 1;
    }
    return 0;
}
