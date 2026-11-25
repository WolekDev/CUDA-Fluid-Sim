#include "PipeGenerator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

constexpr double kPi = 3.14159265358979323846;

constexpr double kCeilTolerance = 1e-6;

struct Vec3 {
    double x, y, z;
};

Vec3 toVec3(float3 v) { return {v.x, v.y, v.z}; }
double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 scale(Vec3 v, double factor) { return {v.x * factor, v.y * factor, v.z * factor}; }
Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
Vec3 normalized(Vec3 v) { return scale(v, 1.0 / std::sqrt(dot(v, v))); }

/* Coordinate frame of the pipe. A point is
 *   origin + axial_position * axis + u_offset * u + w_offset * w
 * u and w are unit vectors spanning the cross-section plane. w is as close to
 * world +y as possible, so angle -pi/2 in the (u, w) plane points down. */
struct CrossSectionFrame {
    Vec3 origin; // center of the inlet cross-section
    Vec3 axis;   // unit flow direction
    Vec3 u;      // horizontal cross-section direction
    Vec3 w;      // cross-section direction closest to world up
};

CrossSectionFrame makeCrossSectionFrame(const PipeSpec& spec) {
    const Vec3 axis = normalized(toVec3(spec.axis));
    Vec3 world_up = {0.0, 1.0, 0.0};
    if (std::fabs(dot(world_up, axis)) > 0.99)
        world_up = {0.0, 0.0, 1.0};
    const Vec3 w = normalized(add(world_up, scale(axis, -dot(world_up, axis))));
    const Vec3 u = cross(w, axis);
    return {toVec3(spec.origin), axis, u, w};
}

float3 toWorld(const CrossSectionFrame& frame, double axial_position,
               double u_offset, double w_offset) {
    const Vec3 point =
        add(add(frame.origin, scale(frame.axis, axial_position)),
            add(scale(frame.u, u_offset), scale(frame.w, w_offset)));
    return make_float3(static_cast<float>(point.x), static_cast<float>(point.y),
                       static_cast<float>(point.z));
}

/* Area enclosed by the outline divided by its circumradius squared.
 * Circle: pi. Regular n-gon: (n / 2) sin(2 pi / n). */
double areaPerCircumradiusSquared(const PipeSpec& spec) {
    if (spec.shape == PipeShape::Circle)
        return kPi;
    const double sides = spec.sides;
    return 0.5 * sides * std::sin(2.0 * kPi / sides);
}

/* Circumradius (center-to-corner distance; the radius for a circle) of the
 * outline that encloses the given area. */
double circumradiusForArea(const PipeSpec& spec, double area) {
    return std::sqrt(area / areaPerCircumradiusSquared(spec));
}

/* Apothem (center-to-side distance) / circumradius of a regular n-gon. */
double apothemToCircumradiusRatio(const PipeSpec& spec) {
    return std::cos(kPi / spec.sides);
}

/* Fraction of the size change reached at segment_fraction in [0, 1]. */
double rampFraction(PipeRamp ramp, double segment_fraction) {
    return ramp == PipeRamp::Linear
               ? segment_fraction
               : 0.5 * (1.0 - std::cos(kPi * segment_fraction));
}

/* d rampFraction / d segment_fraction. */
double rampFractionDerivative(PipeRamp ramp, double segment_fraction) {
    return ramp == PipeRamp::Linear ? 1.0
                                    : 0.5 * kPi * std::sin(kPi * segment_fraction);
}

/* Inner-wall circumradius along one segment. Positions are measured from the
 * start of the segment. */
struct SegmentProfile {
    double start_circumradius;
    double end_circumradius;
    double length;
    PipeRamp ramp;

    double circumradiusAt(double position_in_segment) const {
        return start_circumradius +
               (end_circumradius - start_circumradius) *
                   rampFraction(ramp, position_in_segment / length);
    }
    /* d circumradius / d axial position. */
    double circumradiusSlope(double position_in_segment) const {
        return (end_circumradius - start_circumradius) *
               rampFractionDerivative(ramp, position_in_segment / length) / length;
    }
};

SegmentProfile segmentProfile(const PipeSpec& spec, const PipeSegment& segment) {
    return {circumradiusForArea(spec, segment.start_area),
            circumradiusForArea(spec, segment.end_area), segment.length,
            segment.ramp};
}

double innerCircumradiusAt(const PipeSpec& spec, double axial_position) {
    if (spec.segments.empty())
        return 0.0;
    double segment_start = 0.0;
    for (const PipeSegment& segment : spec.segments) {
        if (axial_position <= segment_start + segment.length) {
            const double position_in_segment =
                std::clamp(axial_position - segment_start, 0.0,
                           static_cast<double>(segment.length));
            return segmentProfile(spec, segment).circumradiusAt(position_in_segment);
        }
        segment_start += segment.length;
    }
    const PipeSegment& last = spec.segments.back();
    return segmentProfile(spec, last).circumradiusAt(last.length);
}

/* Axial positions of the wall rings in one segment, measured from the segment
 * start. Rings are equally spaced in wall arc length,
 *   d(arc) = sqrt(1 + (dR/ds)^2) ds,
 * and no more than ring_spacing apart along the wall. */
std::vector<double> segmentRingPositions(const SegmentProfile& profile,
                                         double ring_spacing) {
    // Trapezoid-rule table of arc length at kArcSamples + 1 axial samples.
    constexpr int kArcSamples = 1024;
    std::vector<double> arc_length_at_sample(kArcSamples + 1, 0.0);
    const double sample_step = profile.length / kArcSamples;
    for (int sample = 1; sample <= kArcSamples; ++sample) {
        const double slope_before = profile.circumradiusSlope((sample - 1) * sample_step);
        const double slope_after = profile.circumradiusSlope(sample * sample_step);
        arc_length_at_sample[sample] =
            arc_length_at_sample[sample - 1] +
            0.5 * sample_step *
                (std::sqrt(1.0 + slope_before * slope_before) +
                 std::sqrt(1.0 + slope_after * slope_after));
    }
    const double total_arc_length = arc_length_at_sample.back();

    const int ring_intervals = std::max(
        1, static_cast<int>(std::ceil(total_arc_length / ring_spacing - kCeilTolerance)));
    std::vector<double> ring_positions;
    int sample = 1;
    for (int ring = 0; ring <= ring_intervals; ++ring) {
        // Invert the table: find the axial position with this arc length.
        const double target_arc_length = total_arc_length * ring / ring_intervals;
        while (sample < kArcSamples && arc_length_at_sample[sample] < target_arc_length)
            ++sample;
        const double sample_arc_length =
            arc_length_at_sample[sample] - arc_length_at_sample[sample - 1];
        const double fraction_in_sample =
            sample_arc_length > 0.0
                ? (target_arc_length - arc_length_at_sample[sample - 1]) /
                      sample_arc_length
                : 0.0;
        ring_positions.push_back(std::clamp(
            (sample - 1 + fraction_in_sample) * sample_step, 0.0, profile.length));
    }
    ring_positions.front() = 0.0;
    ring_positions.back() = profile.length;
    return ring_positions;
}

/* One ring of wall particles at axial_position for the given wall layer.
 * Layer k is the inner outline pushed outward by k * wall_spacing. */
void appendWallRing(const PipeSpec& spec, const CrossSectionFrame& frame,
                    double axial_position, double inner_circumradius, int layer,
                    std::vector<float3>& wall_particles) {
    const double wall_spacing = spec.wall_spacing;
    if (spec.shape == PipeShape::Circle) {
        const double layer_radius = inner_circumradius + layer * wall_spacing;
        const int points_in_ring = std::max(
            6, static_cast<int>(std::ceil(2.0 * kPi * layer_radius / wall_spacing -
                                          kCeilTolerance)));
        for (int point = 0; point < points_in_ring; ++point) {
            const double angle =
                spec.rotation_radians + 2.0 * kPi * point / points_in_ring;
            wall_particles.push_back(toWorld(frame, axial_position,
                                             layer_radius * std::cos(angle),
                                             layer_radius * std::sin(angle)));
        }
        return;
    }

    // Offsetting every side outward by one spacing grows the apothem by the
    // spacing; the corners move out along their diagonals.
    const int sides = spec.sides;
    const double layer_apothem =
        inner_circumradius * apothemToCircumradiusRatio(spec) + layer * wall_spacing;
    const double layer_circumradius = layer_apothem / apothemToCircumradiusRatio(spec);
    const double side_length = 2.0 * layer_circumradius * std::sin(kPi / sides);
    const int points_per_side = std::max(
        1, static_cast<int>(std::ceil(side_length / wall_spacing - kCeilTolerance)));
    // Corner angles that put the outward normal of side 0 at -pi/2, so a flat
    // side faces down when rotation_radians is 0.
    const double first_corner_angle = spec.rotation_radians - 0.5 * kPi - kPi / sides;
    for (int side = 0; side < sides; ++side) {
        const double start_angle = first_corner_angle + 2.0 * kPi * side / sides;
        const double end_angle = first_corner_angle + 2.0 * kPi * (side + 1) / sides;
        const double start_u = layer_circumradius * std::cos(start_angle);
        const double start_w = layer_circumradius * std::sin(start_angle);
        const double end_u = layer_circumradius * std::cos(end_angle);
        const double end_w = layer_circumradius * std::sin(end_angle);
        // Points run from this corner up to, but not including, the next one,
        // which starts the next side.
        for (int point = 0; point < points_per_side; ++point) {
            const double fraction_along_side =
                static_cast<double>(point) / points_per_side;
            wall_particles.push_back(toWorld(
                frame, axial_position,
                start_u + (end_u - start_u) * fraction_along_side,
                start_w + (end_w - start_w) * fraction_along_side));
        }
    }
}

/* True if (u_offset, w_offset) is inside the inner outline and at least
 * wall_clearance away from it. */
bool insideOutlineWithClearance(const PipeSpec& spec, double inner_circumradius,
                                double u_offset, double w_offset,
                                double wall_clearance) {
    if (spec.shape == PipeShape::Circle) {
        const double max_radius = inner_circumradius - wall_clearance;
        return max_radius > 0.0 &&
               u_offset * u_offset + w_offset * w_offset <= max_radius * max_radius;
    }
    // Inside a regular polygon means within max_apothem along every side's
    // outward normal.
    const double max_apothem =
        inner_circumradius * apothemToCircumradiusRatio(spec) - wall_clearance;
    if (max_apothem <= 0.0)
        return false;
    for (int side = 0; side < spec.sides; ++side) {
        const double normal_angle =
            spec.rotation_radians - 0.5 * kPi + 2.0 * kPi * side / spec.sides;
        if (u_offset * std::cos(normal_angle) + w_offset * std::sin(normal_angle) >
            max_apothem)
            return false;
    }
    return true;
}

float3 readVector3(const nlohmann::json& json_value, const char* key) {
    if (!json_value.is_array() || json_value.size() != 3)
        throw std::runtime_error(std::string("pipe.") + key +
                                 " must be an array of 3 numbers");
    return make_float3(json_value[0].get<float>(), json_value[1].get<float>(),
                       json_value[2].get<float>());
}

} // namespace

float pipeLength(const PipeSpec& spec) {
    double length = 0.0;
    for (const PipeSegment& segment : spec.segments)
        length += segment.length;
    return static_cast<float>(length);
}

float pipeCircumradiusAt(const PipeSpec& spec, float axial_position) {
    return static_cast<float>(innerCircumradiusAt(spec, axial_position));
}

float pipeAreaAt(const PipeSpec& spec, float axial_position) {
    const double circumradius = innerCircumradiusAt(spec, axial_position);
    return static_cast<float>(areaPerCircumradiusSquared(spec) * circumradius *
                              circumradius);
}

float pipeAxialCoordinate(const PipeSpec& spec, float3 position) {
    const Vec3 axis = normalized(toVec3(spec.axis));
    const Vec3 offset_from_inlet =
        add(toVec3(position), scale(toVec3(spec.origin), -1.0));
    return static_cast<float>(dot(offset_from_inlet, axis));
}

PipeCrossSectionFrame pipeCrossSectionFrame(const PipeSpec& spec) {
    const CrossSectionFrame frame = makeCrossSectionFrame(spec);
    auto asFloat3 = [](Vec3 v) {
        return make_float3(static_cast<float>(v.x), static_cast<float>(v.y),
                           static_cast<float>(v.z));
    };
    return {asFloat3(frame.origin), asFloat3(frame.axis), asFloat3(frame.u),
            asFloat3(frame.w)};
}

PipeWall generatePipeWall(const PipeSpec& spec) {
    if (spec.wall_spacing <= 0.0f)
        throw std::runtime_error("pipe wall_spacing must be positive");
    if (spec.segments.empty())
        throw std::runtime_error("pipe needs at least one segment");

    const CrossSectionFrame frame = makeCrossSectionFrame(spec);
    PipeWall wall;
    double segment_start = 0.0;
    for (size_t segment_index = 0; segment_index < spec.segments.size();
         ++segment_index) {
        const SegmentProfile profile =
            segmentProfile(spec, spec.segments[segment_index]);
        const auto ring_positions = segmentRingPositions(profile, spec.wall_spacing);
        // The ring at a joint belongs to the previous segment.
        const size_t first_ring = segment_index == 0 ? 0 : 1;
        for (size_t ring = first_ring; ring < ring_positions.size(); ++ring) {
            const double axial_position = segment_start + ring_positions[ring];
            wall.ring_positions.push_back(static_cast<float>(axial_position));
            for (int layer = 0; layer < spec.wall_layers; ++layer)
                appendWallRing(spec, frame, axial_position,
                               profile.circumradiusAt(ring_positions[ring]), layer,
                               wall.particles);
        }
        segment_start += profile.length;
    }
    return wall;
}

std::vector<float3> generateCrossSection(const PipeSpec& spec, float axial_position,
                                         float lattice_spacing,
                                         float wall_clearance) {
    const CrossSectionFrame frame = makeCrossSectionFrame(spec);
    const double inner_circumradius = innerCircumradiusAt(spec, axial_position);
    // The outline fits inside the square |u|, |w| <= circumradius.
    const int max_lattice_index =
        static_cast<int>(std::floor(inner_circumradius / lattice_spacing));
    std::vector<float3> lattice_points;
    for (int u_index = -max_lattice_index; u_index <= max_lattice_index; ++u_index) {
        for (int w_index = -max_lattice_index; w_index <= max_lattice_index;
             ++w_index) {
            const double u_offset = u_index * static_cast<double>(lattice_spacing);
            const double w_offset = w_index * static_cast<double>(lattice_spacing);
            if (insideOutlineWithClearance(spec, inner_circumradius, u_offset,
                                           w_offset, wall_clearance))
                lattice_points.push_back(
                    toWorld(frame, axial_position, u_offset, w_offset));
        }
    }
    return lattice_points;
}

PipeFill generatePipeFill(const PipeSpec& spec, float lattice_spacing,
                          float wall_clearance) {
    PipeFill fill;
    const float length = pipeLength(spec);
    const float inlet_area = pipeAreaAt(spec, 0.0f);
    const float3 axis = [&] {
        const Vec3 unit_axis = normalized(toVec3(spec.axis));
        return make_float3(static_cast<float>(unit_axis.x),
                           static_cast<float>(unit_axis.y),
                           static_cast<float>(unit_axis.z));
    }();
    for (float axial_position = 0.5f * lattice_spacing; axial_position < length;
         axial_position += lattice_spacing) {
        // Continuity: v(s) = v_inlet * A(0) / A(s).
        const float layer_speed =
            spec.inflow.speed * inlet_area / pipeAreaAt(spec, axial_position);
        const float3 layer_velocity =
            make_float3(axis.x * layer_speed, axis.y * layer_speed,
                        axis.z * layer_speed);
        for (const float3& position :
             generateCrossSection(spec, axial_position, lattice_spacing,
                                  wall_clearance)) {
            fill.positions.push_back(position);
            fill.velocities.push_back(layer_velocity);
        }
    }
    return fill;
}

PipeSpec parsePipeSpec(const nlohmann::json& pipe) {
    PipeSpec spec;

    const std::string shape_name = pipe.value("shape", std::string("circle"));
    if (shape_name == "circle") {
        spec.shape = PipeShape::Circle;
    } else {
        spec.shape = PipeShape::RegularPolygon;
        if (shape_name == "triangle")
            spec.sides = 3;
        else if (shape_name == "square")
            spec.sides = 4;
        else if (shape_name == "hexagon")
            spec.sides = 6;
        else if (shape_name == "polygon")
            spec.sides = pipe.value("sides", 0);
        else
            throw std::runtime_error("unknown pipe shape '" + shape_name +
                                     "' (use circle, triangle, square, "
                                     "hexagon, or polygon)");
        if (spec.sides < 3)
            throw std::runtime_error("a polygon pipe needs sides >= 3");
    }

    spec.rotation_radians =
        pipe.value("rotation_deg", 0.0f) * static_cast<float>(kPi / 180.0);
    spec.origin = readVector3(pipe.at("origin"), "origin");
    const float3 axis_as_given = readVector3(
        pipe.value("axis", nlohmann::json::array({1, 0, 0})), "axis");
    const double axis_length =
        std::sqrt(axis_as_given.x * axis_as_given.x + axis_as_given.y * axis_as_given.y +
                  axis_as_given.z * axis_as_given.z);
    if (axis_length <= 0.0)
        throw std::runtime_error("pipe.axis must be nonzero");
    spec.axis = make_float3(static_cast<float>(axis_as_given.x / axis_length),
                            static_cast<float>(axis_as_given.y / axis_length),
                            static_cast<float>(axis_as_given.z / axis_length));

    spec.wall_spacing = pipe.at("wall_spacing").get<float>();
    spec.wall_layers = pipe.value("wall_layers", 2);
    if (spec.wall_spacing <= 0.0f || spec.wall_layers < 1)
        throw std::runtime_error("pipe needs wall_spacing > 0 and wall_layers >= 1");

    for (const auto& segment_json : pipe.at("segments")) {
        PipeSegment segment;
        segment.length = segment_json.at("length").get<float>();
        segment.start_area = segment_json.at("start_area").get<float>();
        segment.end_area = segment_json.value("end_area", segment.start_area);
        const std::string ramp_name =
            segment_json.value("ramp", std::string("linear"));
        if (ramp_name == "linear")
            segment.ramp = PipeRamp::Linear;
        else if (ramp_name == "cosine")
            segment.ramp = PipeRamp::Cosine;
        else
            throw std::runtime_error("unknown segment ramp '" + ramp_name +
                                     "' (use linear or cosine)");
        if (segment.length <= 0.0f || segment.start_area <= 0.0f ||
            segment.end_area <= 0.0f)
            throw std::runtime_error("pipe segments need positive length and areas");
        spec.segments.push_back(segment);
    }
    if (spec.segments.empty())
        throw std::runtime_error("pipe needs at least one segment");

    if (pipe.contains("inflow")) {
        const auto& inflow_json = pipe.at("inflow");
        spec.inflow.speed = inflow_json.value("speed", 0.0f);
        spec.inflow.zone_length = inflow_json.value("zone_length", 0.0f);
        if (spec.inflow.speed < 0.0f || spec.inflow.zone_length < 0.0f)
            throw std::runtime_error("pipe inflow speed and zone_length must be >= 0");
    }
    spec.prefill = pipe.value("prefill", false);
    return spec;
}
