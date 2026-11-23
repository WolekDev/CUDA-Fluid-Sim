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
    std::vector<float> boundary_psi;
};

/* Same 3x3x3 cell walk as the kernels, with particles kept in index order.
 * That matches a stable sort of (cell key, particle id). */
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

float3 difference3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

/* Host transcription of one IISPH step with static boundary particles.
 * Fluid particle i has id i, and boundary particle b has id count + b.
 * Psi_b and boundary density: Ihmsen p. 5, Eq. (14). Density: Ihmsen p. 2.
 * Advection and d_ii: Algorithm 1, p. 3, and p. 6. rho_adv: Eq. (4) and p. 6.
 * a_ii: Eq. (12) plus the p. 6 boundary term. d_ji: thesis Eq. (4.11).
 * Pressure: Eq. (16). Density prediction: thesis Eq. (4.7) plus p. 6.
 * Integration: Eq. (4.12), Eq. (15), and Algorithm 1 lines 21–22. */
ReferenceState solveReference(const std::vector<float3>& pos,
                              const SimulationParams& params, float dt,
                              int max_iterations, float density_threshold,
                              const std::vector<float3>& initial_velocity = {},
                              const std::vector<float3>& boundary = {}) {
    const int count = static_cast<int>(pos.size());
    const int boundary_count = static_cast<int>(boundary.size());
    const float support = params.smoothing_radius;
    const float support2 = support * support;
    const float mass = params.particle_mass;
    const float dt2 = dt * dt;

    std::vector<float3> all = pos;
    all.insert(all.end(), boundary.begin(), boundary.end());
    const auto cells = buildCells(all, params);
    const auto is_boundary = [count](int id) { return id >= count; };

    ReferenceState state;
    state.density.assign(count, 0.0f);
    state.vel_adv.assign(count, make_float3(0.0f, 0.0f, 0.0f));
    state.dii.assign(count, make_float3(0.0f, 0.0f, 0.0f));
    state.density_adv.assign(count, 0.0f);
    state.diag.assign(count, 0.0f);
    state.pressure.assign(count, 0.0f);
    state.density_predicted.assign(count, 0.0f);
    state.sum_movement.assign(count, make_float3(0.0f, 0.0f, 0.0f));
    state.boundary_psi.assign(boundary_count, 0.0f);
    std::vector<float> pressure_prev(count, 0.0f);
    std::vector<float3> vel(count, make_float3(0.0f, 0.0f, 0.0f));
    if (!initial_velocity.empty()) {
        expect(initial_velocity.size() == pos.size(),
               "reference velocities do not match the particle count");
        vel = initial_velocity;
    }

    // delta_b = sum_{b_j} W_{b b_j}, Psi_b = rho_0 / delta_b. Ihmsen p. 5.
    for (int b = 0; b < boundary_count; ++b) {
        float delta = 0.0f;
        forNeighbors(all, params, cells, count + b, [&](int j) {
            if (!is_boundary(j))
                return;
            const float3 r = difference3(all[count + b], all[j]);
            const float r2 = dot3(r, r);
            if (r2 <= support2)
                delta += computePoly6(std::sqrt(r2), support);
        });
        state.boundary_psi[b] = params.rest_density / delta;
    }
    const auto weight_of = [&](int id) {
        return is_boundary(id) ? state.boundary_psi[id - count] : mass;
    };

    for (int i = 0; i < count; ++i) {
        forNeighbors(all, params, cells, i, [&](int j) {
            const float3 rij = difference3(all[i], all[j]);
            const float r2 = dot3(rij, rij);
            if (r2 <= support2)
                state.density[i] += weight_of(j) * computePoly6(std::sqrt(r2), support);
        });
    }

    for (int i = 0; i < count; ++i) {
        float3 accumulated = make_float3(0.0f, 0.0f, 0.0f);
        float3 f_visc = make_float3(0.0f, 0.0f, 0.0f);
        if (state.density[i] > 0.0f) {
            const float inv_density2 = 1.0f / (state.density[i] * state.density[i]);
            const float mass_over_density = mass / state.density[i];
            const float visc_scale =
                mass_over_density * mass_over_density *
                dynamicViscosityPascalSeconds(params.viscosity);
            forNeighbors(all, params, cells, i, [&](int j) {
                const float3 rij = difference3(all[i], all[j]);
                const float r2 = dot3(rij, rij);
                if (r2 >= support2)
                    return;
                const float distance = std::sqrt(r2);
                if (distance > 0.0f) {
                    const float3 grad =
                        computeSpikyGradient(rij, distance, support);
                    const float scale = weight_of(j) * inv_density2;
                    accumulated.x += scale * grad.x;
                    accumulated.y += scale * grad.y;
                    accumulated.z += scale * grad.z;
                }
                if (is_boundary(j))
                    return;
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
            forNeighbors(all, params, cells, i, [&](int j) {
                const float3 rij = difference3(all[i], all[j]);
                const float r2 = dot3(rij, rij);
                if (r2 <= 0.0f || r2 >= support2)
                    return;
                const float distance = std::sqrt(r2);
                const float3 grad_ij =
                    computeSpikyGradient(rij, distance, support);
                if (is_boundary(j)) {
                    // p. 6: Psi_b (v_i^adv - 0) · grad W_ib and Psi_b d_ii · grad W_ib
                    const float psi = state.boundary_psi[j - count];
                    density_sum += psi * dot3(state.vel_adv[i], grad_ij);
                    a_ii += psi * dot3(state.dii[i], grad_ij);
                    return;
                }
                const float3 grad_ji = computeSpikyGradient(
                    make_float3(-rij.x, -rij.y, -rij.z), distance, support);
                density_sum +=
                    mass * dot3(difference3(state.vel_adv[i], state.vel_adv[j]),
                                grad_ij);
                const float3 d_ji =
                    pairwiseDji(grad_ji, mass, state.density[i], dt2);
                a_ii += mass * dot3(difference3(state.dii[i], d_ji), grad_ij);
            });
        }
        state.density_adv[i] = state.density[i] + dt * density_sum;
        state.diag[i] = a_ii;
    }

    // Eq. (16) neighbor terms: fluid part from Eq. (13), boundary part p. 6.
    const auto neighbor_term = [&](int i) {
        float neighbor_sum = 0.0f;
        if (state.density[i] <= 0.0f)
            return neighbor_sum;
        forNeighbors(all, params, cells, i, [&](int j) {
            const float3 rij = difference3(all[i], all[j]);
            const float r2 = dot3(rij, rij);
            if (is_boundary(j)) {
                if (r2 <= 0.0f || r2 >= support2)
                    return;
                const float3 grad_ib =
                    computeSpikyGradient(rij, std::sqrt(r2), support);
                neighbor_sum += state.boundary_psi[j - count] *
                                dot3(state.sum_movement[i], grad_ib);
                return;
            }
            if (j == i || r2 <= 0.0f || r2 >= support2)
                return;
            const float distance = std::sqrt(r2);
            const float3 grad_ij = computeSpikyGradient(rij, distance, support);
            const float3 grad_ji = computeSpikyGradient(
                make_float3(-rij.x, -rij.y, -rij.z), distance, support);
            const float3 d_ji = pairwiseDji(grad_ji, mass, state.density[i], dt2);
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
        return neighbor_sum;
    };

    int iteration = 0;
    float density_average = 0.0f;
    while ((density_average - params.rest_density > density_threshold ||
            iteration < 2) &&
           iteration < max_iterations) {
        for (int i = 0; i < count; ++i) {
            float3 sum = make_float3(0.0f, 0.0f, 0.0f);
            forNeighbors(all, params, cells, i, [&](int j) {
                if (is_boundary(j))
                    return;
                const float3 rij = difference3(all[i], all[j]);
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
            const float neighbor_sum = neighbor_term(i);
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
            state.density_predicted[i] = state.density_adv[i] +
                                         state.diag[i] * state.pressure[i] +
                                         neighbor_term(i);
            density_sum += state.density_predicted[i];
        }
        density_average = density_sum / static_cast<float>(count);
        ++iteration;
    }

    state.position.resize(count);
    state.velocity.resize(count);
    for (int i = 0; i < count; ++i) {
        float3 pressure_sum = make_float3(0.0f, 0.0f, 0.0f);
        if (state.density[i] > 0.0f) {
            const float pressure_term_i =
                state.pressure[i] / (state.density[i] * state.density[i]);
            forNeighbors(all, params, cells, i, [&](int j) {
                const float3 rij = difference3(all[i], all[j]);
                const float r2 = dot3(rij, rij);
                if (r2 <= 0.0f || r2 >= support2)
                    return;
                float coeff = 0.0f;
                if (is_boundary(j)) {
                    coeff = state.boundary_psi[j - count] * pressure_term_i;
                } else {
                    if (state.density[j] <= 0.0f)
                        return;
                    coeff = mass * (pressure_term_i +
                                    state.pressure[j] /
                                        (state.density[j] * state.density[j]));
                }
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

/* Square wall of static boundary particles in the plane y = height, centered
 * at x = z = 0.5, like the open-top box walls of the thesis scenes (p. 29). */
std::vector<float3> boundaryFloor(int per_side, float spacing, float height) {
    std::vector<float3> floor;
    const float offset = 0.5f * static_cast<float>(per_side - 1) * spacing;
    for (int ix = 0; ix < per_side; ++ix)
        for (int iz = 0; iz < per_side; ++iz)
            floor.push_back(make_float3(0.5f - offset + ix * spacing, height,
                                        0.5f - offset + iz * spacing));
    return floor;
}

constexpr float kFloorHeight = 0.5f;
constexpr float kFloorSpacing = 0.02f;

std::vector<float3> wallFloor() {
    return boundaryFloor(11, kFloorSpacing, kFloorHeight);
}

/* Particles 0 and 1 interact with each other and with the floor. Particle 2
 * is isolated. Particle 3 touches only the floor. */
std::vector<float3> fluidNearWall() {
    return {make_float3(0.50f, 0.52f, 0.50f), make_float3(0.51f, 0.525f, 0.505f),
            make_float3(0.90f, 0.90f, 0.90f), make_float3(0.56f, 0.51f, 0.56f)};
}

/* Particles 0 and 1 move toward the floor together, so only boundary terms
 * change their advected density. */
std::vector<float3> fluidNearWallVelocities() {
    return {make_float3(0.0f, -0.5f, 0.0f), make_float3(0.0f, -0.5f, 0.0f),
            make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f)};
}

/* Brute-force Psi_b = rho_0 / sum_{b_j} W_{b b_j}, Ihmsen p. 5, in double. */
std::vector<double> psiReference(const std::vector<float3>& boundary,
                                 double rest_density, double support) {
    std::vector<double> psi;
    for (const float3& b : boundary) {
        double delta = 0.0;
        for (const float3& other : boundary) {
            const double dx = b.x - other.x;
            const double dy = b.y - other.y;
            const double dz = b.z - other.z;
            delta += poly6Reference(std::sqrt(dx * dx + dy * dy + dz * dz), support);
        }
        psi.push_back(rest_density / delta);
    }
    return psi;
}

double distance3(float3 a, float3 b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/* One boundary step on the GPU, plus the matching host reference. */
struct BoundaryStep {
    std::vector<float3> fluid = fluidNearWall();
    std::vector<float3> velocities = fluidNearWallVelocities();
    std::vector<float3> boundary = wallFloor();
    float dt = 0.005f;
    ReferenceState expected;

    void run(FluidSimulation& sim) {
        sim.initialize(kConfig);
        sim.uploadTestParticles(fluid);
        sim.uploadTestVelocities(velocities);
        sim.uploadBoundaryParticles(boundary);
        sim.update(dt);
        expected = solveReference(fluid, sim.getParams(), dt, sim.getMaxIterations(),
                                  sim.getDensityErrorThreshold(), velocities,
                                  boundary);
    }
};

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

void testBoundaryUpload() {
    std::cout << "\n[TEST] testBoundaryUpload\n";
    FluidSimulation sim;
    sim.initialize(kConfig);
    expect(sim.getMaxBoundaryParticles() == 1000,
           "max_boundary_particles was not loaded");
    expect(sim.getBoundaryParticleCount() == 0,
           "a new simulation should start without boundary particles");

    const auto floor = wallFloor();
    sim.uploadBoundaryParticles(floor);
    expect(sim.getBoundaryParticleCount() == static_cast<int>(floor.size()),
           "boundary count does not match the upload");
    const auto stored = sim.copyBoundaryPositionsToHost();
    for (size_t b = 0; b < floor.size(); ++b)
        expectNear3(stored[b], floor[b], "boundary position " + std::to_string(b));

    bool threw = false;
    try {
        sim.uploadBoundaryParticles(
            std::vector<float3>(sim.getMaxBoundaryParticles() + 1,
                                make_float3(0.5f, 0.5f, 0.5f)));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    expect(threw, "uploading more than max_boundary_particles should throw");

    // Fluid and boundary share one sorted grid.
    sim.uploadBoundaryParticles(floor);
    sim.uploadTestParticles(fluidNearWall());
    sim.updateSpatialHash();
    const auto keys = sim.copyKeysToHost();
    expect(keys.size() == fluidNearWall().size() + floor.size(),
           "the grid should hold fluid and boundary particles");
    for (size_t i = 0; i + 1 < keys.size(); ++i)
        expect(keys[i] <= keys[i + 1], "combined cell keys are not sorted");
    sim.cleanup();
    std::cout << "[Pass] boundary particles upload and share the fluid grid\n";
}

void testBoundaryPsi() {
    std::cout << "\n[TEST] testBoundaryPsi\n";
    FluidSimulation sim;
    sim.initialize(kConfig);
    auto boundary = wallFloor();
    boundary.push_back(make_float3(0.1f, 0.1f, 0.1f)); // isolated
    sim.uploadTestParticles(fluidNearWall());
    sim.uploadBoundaryParticles(boundary);
    sim.updateSpatialHash();

    const SimulationParams& params = sim.getParams();
    const auto psi = sim.copyBoundaryPsiToHost();
    const auto expected =
        psiReference(boundary, params.rest_density, params.smoothing_radius);
    expect(psi.size() == boundary.size(), "Psi_b buffer has the wrong size");
    for (size_t b = 0; b < boundary.size(); ++b)
        expectNear(psi[b], static_cast<float>(expected[b]),
                   "Psi_b " + std::to_string(b));

    const size_t corner = 0;
    const size_t center = 5 * 11 + 5;
    expect(psi[center] < psi[corner],
           "a densely sampled wall particle should get a smaller Psi_b");
    expectNear(psi.back(),
               static_cast<float>(params.rest_density /
                                  poly6Reference(0.0, params.smoothing_radius)),
               "an isolated boundary particle has delta_b = W(0)");
    sim.cleanup();
    std::cout << "[Pass] Psi_b = rho_0 / delta_b matches Ihmsen p. 5\n";
}

void testBoundaryDensity() {
    std::cout << "\n[TEST] testBoundaryDensity\n";
    FluidSimulation sim;
    BoundaryStep step;
    step.run(sim);

    const SimulationParams& params = sim.getParams();
    const double support = params.smoothing_radius;
    const double mass = params.particle_mass;
    const auto psi = psiReference(step.boundary, params.rest_density, support);
    const auto density = sim.copyDensitiesToHost();

    // Brute-force Eq. (14), first line.
    for (size_t i = 0; i < step.fluid.size(); ++i) {
        double rho = 0.0;
        for (const float3& other : step.fluid)
            rho += mass * poly6Reference(distance3(step.fluid[i], other), support);
        for (size_t b = 0; b < step.boundary.size(); ++b)
            rho += psi[b] *
                   poly6Reference(distance3(step.fluid[i], step.boundary[b]), support);
        expectNear(density[i], static_cast<float>(rho),
                   "boundary density " + std::to_string(i));
        expectNear(density[i], step.expected.density[i],
                   "reference density " + std::to_string(i));
    }

    const float lone = static_cast<float>(mass * poly6Reference(0.0, support));
    expectNear(density[2], lone, "the isolated particle sees no boundary");
    expect(density[3] > lone,
           "a particle next to the wall should gain density from it");
    expect(density[3] > params.rest_density,
           "the wall alone should push particle 3 over the rest density");
    sim.cleanup();
    std::cout << "[Pass] density includes sum_b Psi_b W_ib (Eq. (14))\n";
}

void testBoundaryAdvectionAndDiagonal() {
    std::cout << "\n[TEST] testBoundaryAdvectionAndDiagonal\n";
    FluidSimulation sim;
    BoundaryStep step;
    step.run(sim);

    const auto vel_adv = sim.copyVelAdvToHost();
    const auto dii = sim.copyGradCoupToHost();
    const auto density = sim.copyDensitiesToHost();
    const auto density_adv = sim.copyDensityAdvToHost();
    const auto diag = sim.copyDiagCoeffToHost();
    for (size_t i = 0; i < step.fluid.size(); ++i) {
        const std::string id = std::to_string(i);
        expectNear3(vel_adv[i], step.expected.vel_adv[i], "advected velocity " + id);
        expectNear3(dii[i], step.expected.dii[i], "boundary d_ii " + id);
        expectNear(density_adv[i], step.expected.density_adv[i],
                   "boundary rho_adv " + id);
        expectNear(diag[i], step.expected.diag[i], "boundary a_ii " + id);
    }

    expect(dii[3].y > 0.0f,
           "the wall's d_ii term should displace particle 3 away from the wall");
    expect(density_adv[0] > density[0] && density_adv[1] > density[1],
           "moving toward a static wall should raise rho_adv (v_b = 0)");
    expect(diag[3] != 0.0f,
           "a particle touching only the wall should still get a_ii != 0");
    expect(diag[2] == 0.0f, "the isolated particle should keep a_ii = 0");
    sim.cleanup();
    std::cout << "[Pass] d_ii, rho_adv, and a_ii include the p. 6 boundary terms\n";
}

void testBoundaryPressureSolve() {
    std::cout << "\n[TEST] testBoundaryPressureSolve\n";
    FluidSimulation sim;
    BoundaryStep step;
    step.run(sim);

    const auto pressure = sim.copyPressureToHost();
    const auto predicted = sim.copyPredictedDensitiesToHost();
    for (size_t i = 0; i < step.fluid.size(); ++i) {
        const std::string id = std::to_string(i);
        expect(pressure[i] >= 0.0f, "pressure was not clamped at 0");
        expectNear(pressure[i], step.expected.pressure[i], "Eq. (16) pressure " + id);
        expectNear(predicted[i], step.expected.density_predicted[i],
                   "predicted density " + id);
    }
    expect(pressure[3] > 0.0f,
           "the wall alone should give particle 3 a positive pressure");
    expect(pressure[2] == 0.0f, "the isolated particle should keep zero pressure");
    sim.cleanup();
    std::cout << "[Pass] relaxed Jacobi pressure matches Eq. (16)\n";
}

void testBoundaryIntegration() {
    std::cout << "\n[TEST] testBoundaryIntegration\n";
    FluidSimulation sim;
    BoundaryStep step;
    step.run(sim);

    const auto position = sim.copyPositionsToHost();
    const auto velocity = sim.copyVelocitiesToHost();
    for (size_t i = 0; i < step.fluid.size(); ++i) {
        const std::string id = std::to_string(i);
        expectNear3(position[i], step.expected.position[i], "position " + id);
        expectNear3(velocity[i], step.expected.velocity[i], "velocity " + id);
    }

    const SimulationParams& params = sim.getParams();
    expect(velocity[3].y > step.dt * params.gravity_vector.y,
           "Eq. (15) should push particle 3 away from the wall");

    const auto boundary = sim.copyBoundaryPositionsToHost();
    for (size_t b = 0; b < step.boundary.size(); ++b) {
        expect(boundary[b].x == step.boundary[b].x &&
                   boundary[b].y == step.boundary[b].y &&
                   boundary[b].z == step.boundary[b].z,
               "boundary particle " + std::to_string(b) + " moved");
    }
    sim.cleanup();
    std::cout << "[Pass] integration applies Eq. (15) and keeps boundaries fixed\n";
}

void testBoundaryContainment() {
    std::cout << "\n[TEST] testBoundaryContainment\n";
    FluidSimulation sim;
    sim.initialize(kConfig);
    const auto floor = boundaryFloor(21, kFloorSpacing, kFloorHeight);
    sim.uploadBoundaryParticles(floor);
    sim.uploadTestParticles({make_float3(0.5f, kFloorHeight + 0.06f, 0.5f)});

    const float dt = 0.002f;
    float lowest = 1.0f;
    for (int step = 0; step < 400; ++step) {
        sim.update(dt);
        lowest = std::min(lowest, sim.copyPositionsToHost()[0].y);
    }

    const float3 final_position = sim.copyPositionsToHost()[0];
    expect(lowest > kFloorHeight,
           "the fluid particle passed through the boundary wall, lowest y=" +
               std::to_string(lowest));
    expect(final_position.y < kFloorHeight + 0.06f,
           "the particle should have fallen toward the wall");
    const auto boundary = sim.copyBoundaryPositionsToHost();
    expect(std::equal(floor.begin(), floor.end(), boundary.begin(),
                      [](float3 a, float3 b) {
                          return a.x == b.x && a.y == b.y && a.z == b.z;
                      }),
           "boundary particles moved during the simulation");
    std::cout << "  lowest y=" << lowest << ", final y=" << final_position.y << "\n";
    sim.cleanup();
    std::cout << "[Pass] a falling particle rests on the boundary wall\n";
}

namespace {

const char* kPipeConfig = "config/test_pipe_config.json";

/* Straight pipe along +x through (0.5, 0.5, 0.5). For that axis the
 * cross-section frame is u = -z and w = +y. */
PipeSpec straightPipe(PipeShape shape, int sides, float area, float length) {
    PipeSpec spec;
    spec.shape = shape;
    spec.sides = sides;
    spec.origin = make_float3(0.5f, 0.5f, 0.5f);
    spec.axis = make_float3(1.0f, 0.0f, 0.0f);
    spec.wall_spacing = 0.02f;
    spec.wall_layers = 2;
    spec.segments.push_back({length, area, area, PipeRamp::Linear});
    return spec;
}

struct Local {
    double s, x, y;
};

Local localCoordinates(const PipeSpec& spec, float3 p) {
    return {p.x - spec.origin.x, -(p.z - spec.origin.z), p.y - spec.origin.y};
}

double areaFactorReference(const PipeSpec& spec) {
    if (spec.shape == PipeShape::Circle)
        return M_PI;
    return 0.5 * spec.sides * std::sin(2.0 * M_PI / spec.sides);
}

/* Largest edge-normal projection. Equals the apothem on a polygon outline. */
double polygonSupport(const PipeSpec& spec, double x, double y) {
    double support = -1.0e30;
    for (int i = 0; i < spec.sides; ++i) {
        const double phi = -0.5 * M_PI + 2.0 * M_PI * i / spec.sides;
        support = std::max(support, x * std::cos(phi) + y * std::sin(phi));
    }
    return support;
}

} // namespace

void testPipeConfigParse() {
    std::cout << "\n[TEST] testPipeConfigParse\n";
    const auto config = nlohmann::json::parse(R"({
        "shape": "square", "rotation_deg": 90.0,
        "origin": [0.1, 0.2, 0.3], "axis": [2.0, 0.0, 0.0],
        "wall_spacing": 0.02,
        "segments": [
            {"length": 0.5, "start_area": 0.04},
            {"length": 0.25, "start_area": 0.04, "end_area": 0.01, "ramp": "cosine"}
        ],
        "inflow": {"speed": 1.5, "zone_length": 0.2},
        "prefill": true
    })");
    const PipeSpec spec = parsePipeSpec(config);
    expect(spec.shape == PipeShape::RegularPolygon && spec.sides == 4,
           "square should parse as a 4-sided polygon");
    expectNear(spec.rotation_radians, static_cast<float>(M_PI / 2.0), "rotation_deg");
    expectNear3(spec.axis, make_float3(1.0f, 0.0f, 0.0f), "axis is normalized");
    expect(spec.wall_layers == 2, "wall_layers should default to 2");
    expect(spec.segments.size() == 2, "segment count");
    expectNear(spec.segments[0].end_area, 0.04f, "end_area defaults to start_area");
    expect(spec.segments[1].ramp == PipeRamp::Cosine, "ramp should parse");
    expectNear(spec.inflow.speed, 1.5f, "inflow speed");
    expect(spec.prefill, "prefill should parse");
    expectNear(pipeLength(spec), 0.75f, "pipe length is the sum of segments");

    const auto throws = [](const char* text) {
        try {
            parsePipeSpec(nlohmann::json::parse(text));
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    expect(throws(R"({"shape": "star", "origin": [0,0,0], "wall_spacing": 0.02,
                      "segments": [{"length": 1, "start_area": 1}]})"),
           "unknown shapes should be rejected");
    expect(throws(R"({"shape": "polygon", "sides": 2, "origin": [0,0,0],
                      "wall_spacing": 0.02,
                      "segments": [{"length": 1, "start_area": 1}]})"),
           "polygons need at least 3 sides");
    expect(throws(R"({"origin": [0,0,0], "wall_spacing": 0.02,
                      "segments": [{"length": 1, "start_area": -1}]})"),
           "negative areas should be rejected");
    std::cout << "[Pass] pipe config parses and validates\n";
}

void testPipeCircleWall() {
    std::cout << "\n[TEST] testPipeCircleWall\n";
    const PipeSpec spec = straightPipe(PipeShape::Circle, 0, 0.02f, 0.2f);
    const PipeWall wall = generatePipeWall(spec);
    const double radius = std::sqrt(0.02 / M_PI);
    const double spacing = spec.wall_spacing;

    expect(wall.ring_positions.size() == 11, "a 0.2 m pipe needs 11 rings at 0.02 m");
    for (size_t k = 0; k + 1 < wall.ring_positions.size(); ++k)
        expectNear(wall.ring_positions[k + 1] - wall.ring_positions[k], 0.02f, "ring spacing");

    const int per_ring = static_cast<int>(std::ceil(2.0 * M_PI * radius / spacing));
    int inner = 0;
    for (const float3& p : wall.particles) {
        const Local l = localCoordinates(spec, p);
        const double r = std::sqrt(l.x * l.x + l.y * l.y);
        const bool on_inner = std::fabs(r - radius) < 1.0e-5;
        const bool on_outer = std::fabs(r - (radius + spacing)) < 1.0e-5;
        expect(on_inner || on_outer, "wall particle is not on a layer circle");
        expect(l.s >= -1.0e-6 && l.s <= 0.2 + 1.0e-6, "wall particle off the pipe");
        inner += on_inner ? 1 : 0;
    }
    expect(inner == per_ring * static_cast<int>(wall.ring_positions.size()),
           "each inner ring should have ceil(2 pi R / spacing) points");
    expectNear(pipeAreaAt(spec, 0.1f), 0.02f, "area should match the request");
    std::cout << "[Pass] circular wall rings are spaced by wall_spacing\n";
}

void testPipePolygonWall() {
    std::cout << "\n[TEST] testPipePolygonWall\n";
    for (int sides : {3, 4, 6}) {
        const std::string name = std::to_string(sides) + "-gon ";
        const PipeSpec spec =
            straightPipe(PipeShape::RegularPolygon, sides, 0.02f, 0.1f);
        const PipeWall wall = generatePipeWall(spec);
        const double radius = std::sqrt(0.02 / areaFactorReference(spec));
        const double apothem = radius * std::cos(M_PI / sides);

        for (const float3& p : wall.particles) {
            const Local l = localCoordinates(spec, p);
            const double support = polygonSupport(spec, l.x, l.y);
            expect(std::fabs(support - apothem) < 1.0e-5 ||
                       std::fabs(support - apothem - spec.wall_spacing) < 1.0e-5,
                   name + "wall particle is not on a layer outline");
        }

        // Every corner of the inner outline is sampled.
        for (int i = 0; i < sides; ++i) {
            const double theta = -0.5 * M_PI - M_PI / sides + 2.0 * M_PI * i / sides;
            const double cx = radius * std::cos(theta);
            const double cy = radius * std::sin(theta);
            bool found = false;
            for (const float3& p : wall.particles) {
                const Local l = localCoordinates(spec, p);
                found = found || (std::fabs(l.s) < 1.0e-6 &&
                                  std::hypot(l.x - cx, l.y - cy) < 1.0e-5);
            }
            expect(found, name + "corner " + std::to_string(i) + " is missing");
        }

        // rotation 0 puts a flat side at the bottom (-w).
        int bottom = 0;
        for (const float3& p : wall.particles) {
            const Local l = localCoordinates(spec, p);
            bottom += std::fabs(l.s) < 1.0e-6 && std::fabs(l.y + apothem) < 1.0e-5;
        }
        expect(bottom >= 2, name + "should have a flat bottom side");
        expectNear(pipeAreaAt(spec, 0.05f), 0.02f, name + "area");
    }
    std::cout << "[Pass] polygon walls follow the outline and keep corners\n";
}

void testPipeTaperSpacing() {
    std::cout << "\n[TEST] testPipeTaperSpacing\n";
    PipeSpec spec = straightPipe(PipeShape::Circle, 0, 0.02f, 0.1f);
    spec.segments.push_back({0.2f, 0.02f, 0.005f, PipeRamp::Linear});
    const PipeWall wall = generatePipeWall(spec);

    expectNear(wall.ring_positions.front(), 0.0f, "first ring at the inlet");
    expectNear(wall.ring_positions.back(), 0.3f, "last ring at the outlet");
    bool has_joint = false;
    for (size_t k = 0; k + 1 < wall.ring_positions.size(); ++k) {
        expect(wall.ring_positions[k + 1] > wall.ring_positions[k] + 1.0e-6,
               "rings must not repeat at a segment joint");
        has_joint = has_joint || std::fabs(wall.ring_positions[k] - 0.1f) < 1.0e-6;

        // Distance between rings measured along the sloped wall.
        const double ds = wall.ring_positions[k + 1] - wall.ring_positions[k];
        const double dr = pipeCircumradiusAt(spec, wall.ring_positions[k + 1]) -
                          pipeCircumradiusAt(spec, wall.ring_positions[k]);
        const double along_wall = std::sqrt(ds * ds + dr * dr);
        expect(along_wall <= spec.wall_spacing + 1.0e-5,
               "rings are further apart than wall_spacing along the taper");
        expect(along_wall >= 0.5 * spec.wall_spacing,
               "rings are much closer than wall_spacing");
    }
    expect(has_joint, "the joint between segments needs a ring");
    expectNear(pipeCircumradiusAt(spec, 0.3f),
               static_cast<float>(std::sqrt(0.005 / M_PI)), "outlet radius");
    expectNear(pipeAreaAt(spec, 0.3f), 0.005f, "outlet area");

    PipeSpec cosine = spec;
    cosine.segments[1].ramp = PipeRamp::Cosine;
    const float joint_slope =
        (pipeCircumradiusAt(cosine, 0.1001f) - pipeCircumradiusAt(cosine, 0.1f)) /
        0.0001f;
    expect(std::fabs(joint_slope) < 1.0e-2f,
           "a cosine ramp should start with zero slope");
    std::cout << "[Pass] taper rings are wall_spacing apart along the wall\n";
}

void testPipeCrossSectionFill() {
    std::cout << "\n[TEST] testPipeCrossSectionFill\n";
    // Square with area 0.04 has apothem 0.1. Inset 0.03 keeps |x|, |y| <= 0.07,
    // so a 0.02 lattice gives 7 x 7 points.
    const PipeSpec square =
        straightPipe(PipeShape::RegularPolygon, 4, 0.04f, 0.5f);
    const auto layer = generateCrossSection(square, 0.25f, 0.02f, 0.03f);
    expect(layer.size() == 49, "square cross-section should hold 7 x 7 points, got " +
                                   std::to_string(layer.size()));
    for (const float3& p : layer) {
        const Local l = localCoordinates(square, p);
        expectNear(static_cast<float>(l.s), 0.25f, "fill point off its station");
        expect(polygonSupport(square, l.x, l.y) <= 0.07 + 1.0e-6,
               "fill point inside the inset");
    }

    PipeSpec venturi = straightPipe(PipeShape::Circle, 0, 0.04f, 0.2f);
    venturi.segments.push_back({0.2f, 0.04f, 0.01f, PipeRamp::Cosine});
    venturi.inflow.speed = 1.0f;
    const float spacing = 0.02f;
    const float inset = 0.02f;
    const PipeFill fill = generatePipeFill(venturi, spacing, inset);
    expect(!fill.positions.empty(), "fill should not be empty");
    for (size_t i = 0; i < fill.positions.size(); ++i) {
        const Local l = localCoordinates(venturi, fill.positions[i]);
        const double radius = pipeCircumradiusAt(venturi, static_cast<float>(l.s));
        expect(std::hypot(l.x, l.y) <= radius - inset + 1.0e-6,
               "fill point too close to the wall");
        // Continuity: v(s) = v_in A(0) / A(s).
        const float expected = 0.04f / pipeAreaAt(venturi, static_cast<float>(l.s));
        expectNear(fill.velocities[i].x, expected, "continuity speed");
        expect(fill.velocities[i].y == 0.0f && fill.velocities[i].z == 0.0f,
               "fill velocity should be along the axis");
    }
    std::cout << "[Pass] cross-section fill stays inside and obeys continuity\n";
}

void testPipeConfigLoading() {
    std::cout << "\n[TEST] testPipeConfigLoading\n";
    FluidSimulation sim;
    sim.initialize(kPipeConfig);
    const PipeSpec* spec = sim.getPipe();
    expect(spec != nullptr, "the pipe block was not loaded");
    expect(sim.getBoundaryParticleCount() ==
               static_cast<int>(generatePipeWall(*spec).particles.size()),
           "the pipe wall should be uploaded as boundary particles");
    expect(sim.getActiveParticleCount() == 0,
           "without prefill the pipe should start empty");
    expectNear(sim.getFluidSpacing(), std::cbrt(0.02f / 1000.0f),
               "rest spacing is cbrt(m / rho_0)");
    expectNear(spec->inflow.zone_length, 4.0f * sim.getFluidSpacing(),
               "zone_length should default to four rest spacings");
    expect(sim.getInflowLayerSize() == 9,
           "the inlet layer should hold 3 x 3 points, got " +
               std::to_string(sim.getInflowLayerSize()));

    sim.updateSpatialHash();
    const auto psi = sim.copyBoundaryPsiToHost();
    expect(std::all_of(psi.begin(), psi.end(), [](float v) { return v > 0.0f; }),
           "Psi_b should be computed for every pipe wall particle");
    sim.cleanup();
    std::cout << "[Pass] the pipe config builds the wall and the inlet\n";
}

void testPipeInflowEmission() {
    std::cout << "\n[TEST] testPipeInflowEmission\n";
    FluidSimulation sim;
    sim.initialize(kPipeConfig);
    const PipeSpec& spec = *sim.getPipe();
    const int layer = sim.getInflowLayerSize();
    const float dt = 0.005f;
    const int steps_per_layer = static_cast<int>(
        std::ceil(sim.getFluidSpacing() / (spec.inflow.speed * dt)));

    for (int step = 1; step < steps_per_layer; ++step) {
        sim.update(dt);
        expect(sim.getActiveParticleCount() == 0,
               "a layer was emitted before the inlet moved one spacing");
    }
    sim.update(dt);
    expect(sim.getActiveParticleCount() == layer,
           "one layer should be emitted after one rest spacing of inflow");
    for (int step = 0; step < steps_per_layer; ++step)
        sim.update(dt);
    expect(sim.getActiveParticleCount() == 2 * layer,
           "a second layer should follow one spacing later");
    sim.cleanup();
    std::cout << "[Pass] the inlet emits one layer per rest spacing\n";
}

void testPipeInflowZone() {
    std::cout << "\n[TEST] testPipeInflowZone\n";
    FluidSimulation sim;
    sim.initialize(kPipeConfig);
    const PipeSpec& spec = *sim.getPipe();
    const int layer = sim.getInflowLayerSize();
    const float dt = 0.005f;
    for (int step = 0; step < 25; ++step)
        sim.update(dt);

    const auto positions = sim.copyPositionsToHost();
    const auto velocities = sim.copyVelocitiesToHost();
    expect(static_cast<int>(positions.size()) == 2 * layer, "expected two layers");
    for (size_t i = 0; i < positions.size(); ++i) {
        const float s = pipeAxialCoordinate(spec, positions[i]);
        expect(s < spec.inflow.zone_length, "fluid left the zone too early");
        expectNear3(velocities[i], make_float3(spec.inflow.speed, 0.0f, 0.0f),
                    "zone velocity " + std::to_string(i));
    }

    const float first = pipeAxialCoordinate(spec, positions[0]);
    const float second = pipeAxialCoordinate(spec, positions[layer]);
    expect(std::fabs((first - second) - sim.getFluidSpacing()) <=
               spec.inflow.speed * dt + 1.0e-5f,
           "inlet layers should stay one rest spacing apart");
    sim.cleanup();
    std::cout << "[Pass] fluid in the inlet zone moves at the inflow speed\n";
}

void testPipeOutflowRemoval() {
    std::cout << "\n[TEST] testPipeOutflowRemoval\n";
    FluidSimulation sim;
    sim.initialize(kPipeConfig);
    // Pipe runs from x = 0.1 to x = 0.8 along y = z = 0.5.
    const std::vector<float3> particles = {
        make_float3(0.40f, 0.5f, 0.5f),  // inside
        make_float3(0.85f, 0.5f, 0.5f),  // past the outlet
        make_float3(0.46f, 0.5f, 0.5f),  // inside
        make_float3(0.50f, 1.20f, 0.5f), // outside the domain
    };
    sim.uploadTestParticles(particles);
    sim.update(0.001f);

    expect(sim.getActiveParticleCount() == 2,
           "particles past the outlet or outside the domain should be removed");
    const auto positions = sim.copyPositionsToHost();
    expect(std::fabs(positions[0].x - 0.40f) < 0.01f &&
               std::fabs(positions[1].x - 0.46f) < 0.01f,
           "remaining particles should keep their order");
    expect(sim.copyPressureToHost().size() == 2 &&
               sim.copyPrevPressureToHost().size() == 2 &&
               sim.copyVelocitiesToHost().size() == 2,
           "per-particle buffers should shrink with the count");
    sim.cleanup();
    std::cout << "[Pass] the outlet removes fluid and compacts the buffers\n";
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
        {"testBoundaryUpload", testBoundaryUpload},
        {"testBoundaryPsi", testBoundaryPsi},
        {"testBoundaryDensity", testBoundaryDensity},
        {"testBoundaryAdvectionAndDiagonal", testBoundaryAdvectionAndDiagonal},
        {"testBoundaryPressureSolve", testBoundaryPressureSolve},
        {"testBoundaryIntegration", testBoundaryIntegration},
        {"testBoundaryContainment", testBoundaryContainment},
        {"testPipeConfigParse", testPipeConfigParse},
        {"testPipeCircleWall", testPipeCircleWall},
        {"testPipePolygonWall", testPipePolygonWall},
        {"testPipeTaperSpacing", testPipeTaperSpacing},
        {"testPipeCrossSectionFill", testPipeCrossSectionFill},
        {"testPipeConfigLoading", testPipeConfigLoading},
        {"testPipeInflowEmission", testPipeInflowEmission},
        {"testPipeInflowZone", testPipeInflowZone},
        {"testPipeOutflowRemoval", testPipeOutflowRemoval},
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
