#pragma once

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <vector>

/* Pipe geometry for static boundary particles. Neither PDF describes pipe
 * geometry or inflow/outflow, so this is plain geometry, not a cited formula.
 *
 * A pipe is a cross-section outline swept along a straight axis. The outline is
 * a circle or a regular n-gon (n = 3 triangle, n = 4 square, ...). Each segment
 * sets the cross-sectional area at its two ends, so different shapes can be
 * compared directly with continuity, A_1 v_1 = A_2 v_2. */

enum class PipeShape { Circle, RegularPolygon };

/* How the size changes along a segment. Both interpolate the circumradius, so
 * Linear gives a straight conical (or pyramidal) wall. Cosine has zero slope at
 * both ends, which avoids sharp corners at segment joints. */
enum class PipeRamp { Linear, Cosine };

struct PipeSegment {
    float length = 0.0f;     // m, along the axis
    float start_area = 0.0f; // m^2, enclosed by the innermost wall layer
    float end_area = 0.0f;   // m^2
    PipeRamp ramp = PipeRamp::Linear;
};

struct PipeInflow {
    float speed = 0.0f;       // m/s along the axis. 0 disables the inlet.
    float zone_length = 0.0f; // m. Fluid with s < zone_length moves at speed.
                              // 0 means four fluid rest spacings.
};

struct PipeSpec {
    PipeShape shape = PipeShape::Circle;
    int sides = 0; // number of polygon sides, only for RegularPolygon (>= 3)
    float rotation_radians = 0.0f; // about the axis. 0 puts a flat side down.
    float3 origin = make_float3(0.0f, 0.0f, 0.0f); // center of the inlet
    float3 axis = make_float3(1.0f, 0.0f, 0.0f);   // unit flow direction
    float wall_spacing = 0.0f;                     // m between wall particles
    int wall_layers = 1; // wall thickness in particles
    std::vector<PipeSegment> segments;
    PipeInflow inflow;
    bool prefill = false; // start with the pipe full of fluid
};

struct PipeWall {
    std::vector<float3> particles;
    std::vector<float> ring_positions; // axial position of every wall ring
};

/* Fluid start state. velocities[i] belongs to positions[i]. */
struct PipeFill {
    std::vector<float3> positions;
    std::vector<float3> velocities;
};

/* In the functions below, axial_position is the distance from the inlet along
 * the axis, s = (x - origin) · axis. Circumradius is the center-to-corner
 * distance of the outline (the radius for a circle). */

/* Total pipe length along the axis. */
float pipeLength(const PipeSpec& spec);

/* Area enclosed by the innermost wall layer. */
float pipeAreaAt(const PipeSpec& spec, float axial_position);

/* Circumradius of the innermost wall layer. */
float pipeCircumradiusAt(const PipeSpec& spec, float axial_position);

/* Axial position of a world-space point. */
float pipeAxialCoordinate(const PipeSpec& spec, float3 position);

/* Wall particles for every layer. Rings are spaced wall_spacing apart,
 * measured along the (possibly sloped) wall. Points in each ring are spaced
 * at most wall_spacing apart, and every polygon corner gets a point. */
PipeWall generatePipeWall(const PipeSpec& spec);

/* Square lattice of fluid positions, lattice_spacing apart, in the
 * cross-section at axial_position. Points stay at least wall_clearance inside
 * the innermost wall layer. */
std::vector<float3> generateCrossSection(const PipeSpec& spec, float axial_position,
                                         float lattice_spacing, float wall_clearance);

/* Fluid filling the whole pipe in cross-section layers lattice_spacing apart.
 * Each layer moves along the axis at inflow.speed * A(0) / A(s), the steady
 * continuity speed. */
PipeFill generatePipeFill(const PipeSpec& spec, float lattice_spacing,
                          float wall_clearance);

/* Reads a "pipe" config block. Throws std::runtime_error on invalid input. */
PipeSpec parsePipeSpec(const nlohmann::json& pipe);
