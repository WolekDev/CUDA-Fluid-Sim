#include "FluidView.hpp"

#include "FluidSimulation.cuh"
#include "PipeGenerator.hpp"

#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr int kPreviewMargin = 16;

struct Ortho {
    float min_x = -1.0f;
    float min_y = -1.0f;
    float max_x = 1.0f;
    float max_y = 1.0f;
};

struct Viewport {
    int x = 0;
    int y = 0;
    int width = 1;
    int height = 1;
};

void checkCuda(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(error));
    }
}

/* Anton Mikhailov's turbo polynomial. Same coefficients as the fragment shader. */
ImVec4 turboColor(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    const float x2 = t * t;
    const float x3 = x2 * t;
    const float x4 = x3 * t;
    const float x5 = x4 * t;
    const auto channel = [&](float c0, float c1, float c2, float c3, float c4,
                             float c5) {
        return std::clamp(c0 + c1 * t + c2 * x2 + c3 * x3 + c4 * x4 + c5 * x5, 0.0f,
                          1.0f);
    };
    return ImVec4(channel(0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f,
                          -152.94239396f, 59.28637943f),
                  channel(0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f,
                          4.27729857f, 2.82956604f),
                  channel(0.10667330f, 12.64194608f, -60.58204836f, 110.36276771f,
                          -89.90310912f, 27.34824973f),
                  1.0f);
}

GLuint compileShader(GLenum type, const char* source) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        glDeleteShader(shader);
        throw std::runtime_error(std::string("shader compile: ") + log);
    }
    return shader;
}

GLuint linkProgram(const char* vertex_source, const char* fragment_source) {
    const GLuint vertex = compileShader(GL_VERTEX_SHADER, vertex_source);
    const GLuint fragment = compileShader(GL_FRAGMENT_SHADER, fragment_source);
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        glDeleteProgram(program);
        throw std::runtime_error(std::string("shader link: ") + log);
    }
    return program;
}

Ortho fitAspect(float center_x, float center_y, float half_width, float half_height,
                float aspect) {
    half_width = std::max(half_width, 1.0e-4f);
    half_height = std::max(half_height, 1.0e-4f);
    const float world_aspect = half_width / half_height;
    if (world_aspect < aspect)
        half_width = half_height * aspect;
    else
        half_height = half_width / aspect;
    return {center_x - half_width, center_y - half_height, center_x + half_width,
            center_y + half_height};
}

/* Inner outline in the cross-section plane, matching PipeGenerator's angles. */
std::vector<float> outlineOffsets(const PipeSpec& spec, float circumradius) {
    std::vector<float> offsets;
    if (spec.shape == PipeShape::Circle) {
        constexpr int kSamples = 64;
        offsets.reserve(kSamples * 2);
        for (int sample = 0; sample < kSamples; ++sample) {
            const float angle =
                spec.rotation_radians + 2.0f * kPi * sample / kSamples;
            offsets.push_back(circumradius * std::cos(angle));
            offsets.push_back(circumradius * std::sin(angle));
        }
        return offsets;
    }

    const int sides = std::max(spec.sides, 3);
    offsets.reserve(sides * 2);
    const float first_corner =
        spec.rotation_radians - 0.5f * kPi - kPi / static_cast<float>(sides);
    for (int side = 0; side < sides; ++side) {
        const float angle =
            first_corner + 2.0f * kPi * side / static_cast<float>(sides);
        offsets.push_back(circumradius * std::cos(angle));
        offsets.push_back(circumradius * std::sin(angle));
    }
    return offsets;
}

float outlineSupport(const std::vector<float>& offsets) {
    float support = 0.0f;
    for (size_t index = 1; index < offsets.size(); index += 2)
        support = std::max(support, std::fabs(offsets[index]));
    return support;
}

const char* kParticleVertex = R"glsl(
#version 330 core
layout(location = 0) in vec4 particle;
uniform vec3 u_origin;
uniform vec3 u_axis;
uniform vec3 u_u;
uniform vec3 u_w;
uniform vec2 u_view_min;
uniform vec2 u_view_max;
uniform int u_mode;
uniform float u_probe_s;
uniform float u_slab;
uniform float u_fade;
uniform float u_point_size;
out float v_scalar;
out float v_alpha;
void main() {
    vec3 rel = particle.xyz - u_origin;
    float s = dot(rel, u_axis);
    float uo = dot(rel, u_u);
    float wo = dot(rel, u_w);
    vec2 p;
    float alpha = 1.0;
    if (u_mode == 0) {
        p = vec2(s, wo);
        alpha = 1.0 - smoothstep(u_fade * 0.35, u_fade, abs(uo));
    } else {
        p = vec2(uo, wo);
        if (abs(s - u_probe_s) > u_slab)
            alpha = 0.0;
    }
    vec2 span = max(u_view_max - u_view_min, vec2(1e-6));
    vec2 ndc = ((p - u_view_min) / span) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = u_point_size;
    v_scalar = particle.w;
    v_alpha = alpha;
}
)glsl";

const char* kParticleFragment = R"glsl(
#version 330 core
in float v_scalar;
in float v_alpha;
uniform vec2 u_scalar_range;
out vec4 frag_color;
vec3 turbo(float x) {
    const vec4 kRed4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kGreen4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kBlue4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kRed2 = vec2(-152.94239396, 59.28637943);
    const vec2 kGreen2 = vec2(4.27729857, 2.82956604);
    const vec2 kBlue2 = vec2(-89.90310912, 27.34824973);
    x = clamp(x, 0.0, 1.0);
    vec4 v4 = vec4(1.0, x, x * x, x * x * x);
    vec2 v2 = v4.zw * v4.z;
    return clamp(vec3(dot(v4, kRed4) + dot(v2, kRed2),
                      dot(v4, kGreen4) + dot(v2, kGreen2),
                      dot(v4, kBlue4) + dot(v2, kBlue2)), 0.0, 1.0);
}
void main() {
    vec2 point = gl_PointCoord * 2.0 - 1.0;
    if (dot(point, point) > 1.0 || v_alpha < 0.02)
        discard;
    float span = max(u_scalar_range.y - u_scalar_range.x, 1e-6);
    float t = clamp((v_scalar - u_scalar_range.x) / span, 0.0, 1.0);
    frag_color = vec4(turbo(t), v_alpha);
}
)glsl";

const char* kLineVertex = R"glsl(
#version 330 core
layout(location = 0) in vec2 point;
uniform vec2 u_view_min;
uniform vec2 u_view_max;
void main() {
    vec2 span = max(u_view_max - u_view_min, vec2(1e-6));
    vec2 ndc = ((point - u_view_min) / span) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)glsl";

const char* kLineFragment = R"glsl(
#version 330 core
uniform vec4 u_color;
out vec4 frag_color;
void main() { frag_color = u_color; }
)glsl";

} // namespace

struct FluidViewState {
    GLFWwindow* window = nullptr;
    RenderPacker packer;
    GLuint particle_program = 0;
    GLuint line_program = 0;
    GLuint particle_vao = 0;
    GLuint particle_vbo = 0;
    GLuint line_vao = 0;
    GLuint line_vbo = 0;
    cudaGraphicsResource* graphics = nullptr;
    float4* d_packed = nullptr;
    int packed_capacity = 0;
    bool interop = false;
    bool capture_mouse = false;
    bool imgui_ready = false;
    bool buffer_mapped = false;
    int panel_width = 380;
    bool panel_ready = false;
    std::vector<float4> h_packed;
    std::vector<float> line_vertices;

    ProbeStats stats;
    ScalarRange range;
    int particle_count = 0;
    float pipe_length = 1.0f;
    float max_radius = 0.1f;
    float fade = 0.1f;
    PipeCrossSectionFrame frame;
    bool has_pipe = false;
    Ortho longitudinal;
    Ortho transverse;
    Viewport longitudinal_viewport;
    Viewport transverse_viewport;
    int preview_x = 0;
    int preview_y = 0;
    int preview_size = 0;
};

FluidView::FluidView() = default;

FluidView::~FluidView() {
    if (state_ == nullptr)
        return;
    if (state_->window != nullptr)
        glfwMakeContextCurrent(state_->window);
    // Unmap before unregistering. A close during a frame can leave the
    // buffer mapped, and unregistering a mapped resource fails.
    if (state_->buffer_mapped && state_->graphics != nullptr) {
        cudaGraphicsUnmapResources(1, &state_->graphics, 0);
        state_->buffer_mapped = false;
    }
    if (state_->graphics != nullptr) {
        cudaGraphicsUnregisterResource(state_->graphics);
        state_->graphics = nullptr;
    }
    cudaDeviceSynchronize();
    cudaFree(state_->d_packed);
    state_->d_packed = nullptr;
    if (state_->imgui_ready) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        state_->imgui_ready = false;
    }
    if (state_->particle_program != 0)
        glDeleteProgram(state_->particle_program);
    if (state_->line_program != 0)
        glDeleteProgram(state_->line_program);
    if (state_->particle_vbo != 0)
        glDeleteBuffers(1, &state_->particle_vbo);
    if (state_->line_vbo != 0)
        glDeleteBuffers(1, &state_->line_vbo);
    if (state_->particle_vao != 0)
        glDeleteVertexArrays(1, &state_->particle_vao);
    if (state_->line_vao != 0)
        glDeleteVertexArrays(1, &state_->line_vao);
    state_->particle_program = state_->line_program = 0;
    state_->particle_vbo = state_->line_vbo = 0;
    state_->particle_vao = state_->line_vao = 0;
    if (state_->window != nullptr) {
        glfwDestroyWindow(state_->window);
        state_->window = nullptr;
        glfwTerminate();
    }
    delete state_;
    state_ = nullptr;
}

void FluidView::open(int width, int height) {
    state_ = new FluidViewState();
    glfwSetErrorCallback([](int error, const char* description) {
        std::cerr << "GLFW " << error << ": " << description << '\n';
    });
    if (!glfwInit())
        throw std::runtime_error("GLFW init failed");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    state_->window = glfwCreateWindow(width, height, "FluidSim", nullptr, nullptr);
    if (state_->window == nullptr) {
        glfwTerminate();
        throw std::runtime_error("Failed to create an OpenGL 3.3 window");
    }
    glfwMakeContextCurrent(state_->window);
    glfwSwapInterval(1);
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        throw std::runtime_error("GLAD failed to load OpenGL");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui::GetStyle().FontScaleMain = 1.6f;
    ImGui_ImplGlfw_InitForOpenGL(state_->window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    state_->imgui_ready = true;

    state_->particle_program = linkProgram(kParticleVertex, kParticleFragment);
    state_->line_program = linkProgram(kLineVertex, kLineFragment);

    glGenVertexArrays(1, &state_->particle_vao);
    glGenBuffers(1, &state_->particle_vbo);
    glBindVertexArray(state_->particle_vao);
    glBindBuffer(GL_ARRAY_BUFFER, state_->particle_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float4), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float4), nullptr);

    glGenVertexArrays(1, &state_->line_vao);
    glGenBuffers(1, &state_->line_vbo);
    glBindVertexArray(state_->line_vao);
    glBindBuffer(GL_ARRAY_BUFFER, state_->line_vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, nullptr);
    glBindVertexArray(0);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    cudaError_t registered = cudaGraphicsGLRegisterBuffer(
        &state_->graphics, state_->particle_vbo, cudaGraphicsRegisterFlagsWriteDiscard);
    if (registered == cudaSuccess) {
        state_->interop = true;
        std::cout << "Particle upload: CUDA-OpenGL interop\n";
    } else {
        cudaGetLastError();
        state_->graphics = nullptr;
        state_->interop = false;
        std::cout << "Particle upload: host copy (" << cudaGetErrorString(registered)
                  << ")\n";
    }
}

bool FluidView::shouldClose() const {
    return state_ == nullptr || state_->window == nullptr ||
           glfwWindowShouldClose(state_->window);
}

namespace {

void ensurePackedCapacity(FluidViewState& gpu, int max_particles) {
    if (max_particles <= gpu.packed_capacity)
        return;
    const int count = std::max(max_particles, 1);
    if (gpu.interop) {
        if (gpu.graphics != nullptr)
            checkCuda(cudaGraphicsUnregisterResource(gpu.graphics),
                      "unregister particle buffer");
        gpu.graphics = nullptr;
        glBindBuffer(GL_ARRAY_BUFFER, gpu.particle_vbo);
        glBufferData(GL_ARRAY_BUFFER, count * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);
        checkCuda(cudaGraphicsGLRegisterBuffer(&gpu.graphics, gpu.particle_vbo,
                                              cudaGraphicsRegisterFlagsWriteDiscard),
                  "register particle buffer");
    } else {
        cudaFree(gpu.d_packed);
        gpu.d_packed = nullptr;
        checkCuda(cudaMalloc(&gpu.d_packed, count * sizeof(float4)),
                  "particle upload buffer");
        glBindBuffer(GL_ARRAY_BUFFER, gpu.particle_vbo);
        glBufferData(GL_ARRAY_BUFFER, count * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);
        gpu.h_packed.resize(count);
    }
    gpu.packed_capacity = count;
    gpu.packer.ensureCapacity(count);
}

float4* mapPacked(FluidViewState& gpu, int count, bool& mapped) {
    mapped = false;
    gpu.buffer_mapped = false;
    if (count <= 0 || !gpu.interop)
        return gpu.d_packed;
    checkCuda(cudaGraphicsMapResources(1, &gpu.graphics, 0), "map particle buffer");
    mapped = true;
    gpu.buffer_mapped = true;
    void* pointer = nullptr;
    size_t bytes = 0;
    checkCuda(cudaGraphicsResourceGetMappedPointer(&pointer, &bytes, gpu.graphics),
              "particle buffer pointer");
    return static_cast<float4*>(pointer);
}

void unmapPacked(FluidViewState& gpu, bool mapped) {
    if (!mapped)
        return;
    checkCuda(cudaGraphicsUnmapResources(1, &gpu.graphics, 0),
              "unmap particle buffer");
    gpu.buffer_mapped = false;
}

void uploadPacked(FluidViewState& gpu, int count) {
    if (gpu.interop || count <= 0)
        return;
    checkCuda(cudaMemcpy(gpu.h_packed.data(), gpu.d_packed, count * sizeof(float4),
                         cudaMemcpyDeviceToHost),
              "particle upload");
    glBindBuffer(GL_ARRAY_BUFFER, gpu.particle_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, count * sizeof(float4), gpu.h_packed.data());
}

void drawLines(FluidViewState& gpu, const std::vector<float>& vertices, GLenum mode,
               const Ortho& ortho, ImVec4 color) {
    if (vertices.size() < 4)
        return;
    glUseProgram(gpu.line_program);
    glUniform2f(glGetUniformLocation(gpu.line_program, "u_view_min"), ortho.min_x,
                ortho.min_y);
    glUniform2f(glGetUniformLocation(gpu.line_program, "u_view_max"), ortho.max_x,
                ortho.max_y);
    glUniform4f(glGetUniformLocation(gpu.line_program, "u_color"), color.x, color.y,
                color.z, color.w);
    glBindVertexArray(gpu.line_vao);
    glBindBuffer(GL_ARRAY_BUFFER, gpu.line_vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(),
                 GL_DYNAMIC_DRAW);
    glDrawArrays(mode, 0, static_cast<GLsizei>(vertices.size() / 2));
}

void setParticleUniforms(FluidViewState& gpu, const Ortho& ortho, int mode,
                         float point_size) {
    const GLuint program = gpu.particle_program;
    glUseProgram(program);
    glUniform3f(glGetUniformLocation(program, "u_origin"), gpu.frame.origin.x,
                gpu.frame.origin.y, gpu.frame.origin.z);
    glUniform3f(glGetUniformLocation(program, "u_axis"), gpu.frame.axis.x,
                gpu.frame.axis.y, gpu.frame.axis.z);
    glUniform3f(glGetUniformLocation(program, "u_u"), gpu.frame.u.x, gpu.frame.u.y,
                gpu.frame.u.z);
    glUniform3f(glGetUniformLocation(program, "u_w"), gpu.frame.w.x, gpu.frame.w.y,
                gpu.frame.w.z);
    glUniform2f(glGetUniformLocation(program, "u_view_min"), ortho.min_x, ortho.min_y);
    glUniform2f(glGetUniformLocation(program, "u_view_max"), ortho.max_x, ortho.max_y);
    glUniform1i(glGetUniformLocation(program, "u_mode"), mode);
    glUniform1f(glGetUniformLocation(program, "u_fade"), gpu.fade);
    glUniform1f(glGetUniformLocation(program, "u_point_size"), point_size);
    glUniform2f(glGetUniformLocation(program, "u_scalar_range"), gpu.range.min_value,
                gpu.range.max_value);
}

void drawParticles(FluidViewState& gpu, const Ortho& ortho, int mode, float point_size) {
    if (gpu.particle_count <= 0)
        return;
    setParticleUniforms(gpu, ortho, mode, point_size);
    glBindVertexArray(gpu.particle_vao);
    glDrawArrays(GL_POINTS, 0, gpu.particle_count);
}

float pointSizeFor(const Ortho& ortho, const Viewport& viewport, float spacing) {
    const float world_per_pixel =
        (ortho.max_y - ortho.min_y) / std::max(viewport.height, 1);
    return std::clamp(spacing / std::max(world_per_pixel, 1.0e-8f), 2.0f, 28.0f);
}

std::vector<float> longitudinalWalls(const PipeSpec& spec, float length) {
    constexpr int kStations = 128;
    std::vector<float> upper;
    std::vector<float> lower;
    upper.reserve(kStations * 2);
    lower.reserve(kStations * 2);
    for (int station = 0; station < kStations; ++station) {
        const float s = length * station / (kStations - 1);
        const float support =
            outlineSupport(outlineOffsets(spec, pipeCircumradiusAt(spec, s)));
        upper.push_back(s);
        upper.push_back(support);
        lower.push_back(s);
        lower.push_back(-support);
    }
    upper.insert(upper.end(), lower.begin(), lower.end());
    return upper;
}

bool mouseInRect(GLFWwindow* window, int x, int y, int width, int height, float& fx,
                 float& fy) {
    double mouse_x = 0.0;
    double mouse_y = 0.0;
    glfwGetCursorPos(window, &mouse_x, &mouse_y);
    int window_w = 1;
    int window_h = 1;
    int fb_w = 1;
    int fb_h = 1;
    glfwGetWindowSize(window, &window_w, &window_h);
    glfwGetFramebufferSize(window, &fb_w, &fb_h);
    if (window_w <= 0 || window_h <= 0)
        return false;
    fx = static_cast<float>(mouse_x * fb_w / window_w);
    fy = static_cast<float>((window_h - mouse_y) * fb_h / window_h);
    return fx >= x && fx < x + width && fy >= y && fy < y + height;
}

void layout(FluidViewState& gpu, const PipeSpec* pipe, float spacing) {
    int window_w = 1;
    int window_h = 1;
    int fb_w = 1;
    int fb_h = 1;
    glfwGetWindowSize(gpu.window, &window_w, &window_h);
    glfwGetFramebufferSize(gpu.window, &fb_w, &fb_h);
    window_w = std::max(window_w, 1);
    window_h = std::max(window_h, 1);
    const float scale_x = static_cast<float>(fb_w) / window_w;
    const float scale_y = static_cast<float>(fb_h) / window_h;

    const int panel_min = 180;
    const int panel_max = std::max(panel_min, window_w - 160);
    if (!gpu.panel_ready)
        gpu.panel_width = std::clamp(window_w * 30 / 100, panel_min, panel_max);
    else
        gpu.panel_width = std::clamp(gpu.panel_width, panel_min, panel_max);
    gpu.panel_ready = true;
    const int view_w = std::max(window_w - gpu.panel_width, 1);
    gpu.longitudinal_viewport = {0, 0, static_cast<int>(view_w * scale_x), fb_h};

    const int preview = std::max(
        80, std::min({260, view_w / 3, window_h / 3}));
    gpu.preview_size = preview;
    gpu.preview_x = kPreviewMargin;
    gpu.preview_y = kPreviewMargin + 18;
    gpu.transverse_viewport = {
        static_cast<int>(gpu.preview_x * scale_x),
        static_cast<int>((window_h - gpu.preview_y - preview) * scale_y),
        static_cast<int>(preview * scale_x), static_cast<int>(preview * scale_y)};

    gpu.pipe_length = pipe != nullptr ? std::max(pipeLength(*pipe), 1.0e-3f) : 1.0f;
    gpu.max_radius = 0.05f;
    if (pipe != nullptr) {
        const int samples = 32;
        for (int sample = 0; sample <= samples; ++sample) {
            const float s = gpu.pipe_length * sample / samples;
            gpu.max_radius =
                std::max(gpu.max_radius, pipeCircumradiusAt(*pipe, s));
        }
        gpu.frame = pipeCrossSectionFrame(*pipe);
    } else {
        gpu.frame = {};
    }
    gpu.has_pipe = pipe != nullptr;
    gpu.fade = std::max(6.0f * spacing, 0.35f * gpu.max_radius);

    const float aspect =
        static_cast<float>(gpu.longitudinal_viewport.width) /
        std::max(gpu.longitudinal_viewport.height, 1);
    gpu.longitudinal =
        fitAspect(0.5f * gpu.pipe_length, 0.0f, 0.54f * gpu.pipe_length,
                  gpu.max_radius * 1.45f, aspect);
    gpu.transverse =
        fitAspect(0.0f, 0.0f, gpu.max_radius * 1.25f, gpu.max_radius * 1.25f, 1.0f);
}

void prepareFrame(FluidViewState& gpu, const FluidSimulation& simulation,
                  const ViewUi& ui) {
    const auto fields = simulation.deviceFields();
    const PipeSpec* pipe = simulation.getPipe();
    const float spacing = std::max(simulation.getFluidSpacing(), 1.0e-4f);
    layout(gpu, pipe, spacing);
    ensurePackedCapacity(gpu, std::max(simulation.getMaxParticles(), fields.count));

    gpu.particle_count = fields.count;
    bool mapped = false;
    float4* packed = mapPacked(gpu, fields.count, mapped);
    gpu.packer.pack(fields.positions, fields.velocities, fields.pressures, packed,
                    fields.count, ui.color);
    gpu.range = gpu.packer.scalarRange(packed, fields.count);
    unmapPacked(gpu, mapped);
    uploadPacked(gpu, fields.count);

    const float half_width = std::max(ui.slab_spacings, 0.25f) * spacing;
    gpu.stats = gpu.packer.probe(fields.positions, fields.velocities, fields.pressures,
                                 fields.densities, fields.count, gpu.frame.origin,
                                 gpu.frame.axis, ui.probe_s, half_width);
    glUseProgram(gpu.particle_program);
    glUniform1f(glGetUniformLocation(gpu.particle_program, "u_probe_s"), ui.probe_s);
    glUniform1f(glGetUniformLocation(gpu.particle_program, "u_slab"), half_width);
}

void renderScene(FluidViewState& gpu, const FluidSimulation& simulation,
                 const ViewUi& ui) {
    const float spacing = std::max(simulation.getFluidSpacing(), 1.0e-4f);
    int fb_w = 1;
    int fb_h = 1;
    glfwGetFramebufferSize(gpu.window, &fb_w, &fb_h);
    glViewport(0, 0, fb_w, fb_h);
    glClearColor(0.07f, 0.08f, 0.10f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glViewport(gpu.longitudinal_viewport.x, gpu.longitudinal_viewport.y,
               gpu.longitudinal_viewport.width, gpu.longitudinal_viewport.height);
    if (gpu.has_pipe) {
        const std::vector<float> walls =
            longitudinalWalls(*simulation.getPipe(), gpu.pipe_length);
        const auto mid = walls.begin() + static_cast<std::ptrdiff_t>(walls.size() / 2);
        drawLines(gpu, std::vector<float>(walls.begin(), mid), GL_LINE_STRIP,
                  gpu.longitudinal, ImVec4(0.78f, 0.82f, 0.86f, 1.0f));
        drawLines(gpu, std::vector<float>(mid, walls.end()), GL_LINE_STRIP,
                  gpu.longitudinal, ImVec4(0.78f, 0.82f, 0.86f, 1.0f));
        const std::vector<float> probe = {ui.probe_s, gpu.longitudinal.min_y, ui.probe_s,
                                          gpu.longitudinal.max_y};
        drawLines(gpu, probe, GL_LINES, gpu.longitudinal,
                  ImVec4(1.0f, 1.0f, 1.0f, 0.85f));
    }
    drawParticles(gpu, gpu.longitudinal, 0,
                  pointSizeFor(gpu.longitudinal, gpu.longitudinal_viewport, spacing));

    glEnable(GL_SCISSOR_TEST);
    glScissor(gpu.transverse_viewport.x, gpu.transverse_viewport.y,
              gpu.transverse_viewport.width, gpu.transverse_viewport.height);
    glClearColor(0.11f, 0.12f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);

    glViewport(gpu.transverse_viewport.x, gpu.transverse_viewport.y,
               gpu.transverse_viewport.width, gpu.transverse_viewport.height);
    if (gpu.has_pipe) {
        const PipeSpec& spec = *simulation.getPipe();
        const float radius = pipeCircumradiusAt(spec, ui.probe_s);
        const std::vector<float> loop = outlineOffsets(spec, radius);
        drawLines(gpu, loop, GL_LINE_LOOP, gpu.transverse,
                  ImVec4(0.78f, 0.82f, 0.86f, 1.0f));
    }
    drawParticles(gpu, gpu.transverse, 1,
                  pointSizeFor(gpu.transverse, gpu.transverse_viewport, spacing));
}

void drawLegend(const ScalarRange& range, ParticleColorMode color) {
    ImGui::Separator();
    ImGui::TextUnformatted(color == ParticleColorMode::Velocity ? "Speed" : "Pressure");
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float bar_h = 140.0f;
    const float bar_w = 18.0f;
    ImDrawList* draw = ImGui::GetWindowDrawList();
    constexpr int kBands = 48;
    for (int band = 0; band < kBands; ++band) {
        const float t = 1.0f - (band + 0.5f) / kBands;
        const ImVec4 rgba = turboColor(t);
        const ImU32 pixel = ImGui::ColorConvertFloat4ToU32(rgba);
        const float y0 = origin.y + bar_h * band / kBands;
        const float y1 = origin.y + bar_h * (band + 1) / kBands;
        draw->AddRectFilled(ImVec2(origin.x, y0), ImVec2(origin.x + bar_w, y1), pixel);
    }
    char high_label[32];
    char low_label[32];
    std::snprintf(high_label, sizeof(high_label), "%.3g", range.max_value);
    std::snprintf(low_label, sizeof(low_label), "%.3g", range.min_value);
    draw->AddText(ImVec2(origin.x + bar_w + 8.0f, origin.y),
                  IM_COL32(230, 230, 230, 255), high_label);
    draw->AddText(ImVec2(origin.x + bar_w + 8.0f,
                         origin.y + bar_h - ImGui::GetTextLineHeight()),
                  IM_COL32(230, 230, 230, 255), low_label);
    ImGui::Dummy(ImVec2(bar_w + 120.0f, bar_h));
}

void drawPanel(FluidViewState& gpu, const FluidSimulation& simulation, ViewUi& ui) {
    int window_w = 1;
    int window_h = 1;
    glfwGetWindowSize(gpu.window, &window_w, &window_h);
    const int minimum = 180;
    const int maximum = std::max(minimum, window_w - 160);
    gpu.panel_width = std::clamp(gpu.panel_width, minimum, maximum);
    // Position is pinned to the right edge. Size is not forced after the first
    // frame, so dragging the left border is kept.
    ImGui::SetNextWindowPos(
        ImVec2(static_cast<float>(window_w - gpu.panel_width), 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(
        ImVec2(static_cast<float>(gpu.panel_width), static_cast<float>(window_h)),
        ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(static_cast<float>(minimum), static_cast<float>(window_h)),
        ImVec2(static_cast<float>(maximum), static_cast<float>(window_h)));
    ImGui::Begin("Pipe", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    const int width =
        std::clamp(static_cast<int>(ImGui::GetWindowWidth()), minimum, maximum);
    ImGui::SetWindowSize(ImVec2(static_cast<float>(width), static_cast<float>(window_h)));
    gpu.panel_width = width;

    ImGui::PushTextWrapPos(0.0f);
    ImGui::SliderFloat("Text size", &ui.text_scale, 1.0f, 2.5f, "%.1f");
    ImGui::Checkbox("Pause", &ui.paused);
    ImGui::SameLine();
    if (ImGui::Button("Step"))
        ui.step_once = true;
    ImGui::SliderInt("Steps / frame", &ui.steps_per_frame, 1, 40);
    ImGui::Text("%.0f fps", ImGui::GetIO().Framerate);
    ImGui::Text("Particles %d", gpu.particle_count);
    ImGui::TextUnformatted(gpu.interop ? "Upload: CUDA-GL" : "Upload: host copy");

    ImGui::Separator();
    int color = static_cast<int>(ui.color);
    ImGui::RadioButton("Velocity", &color, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Pressure", &color, 1);
    ui.color = static_cast<ParticleColorMode>(color);
    drawLegend(gpu.range, ui.color);

    ImGui::Separator();
    ImGui::TextUnformatted("Axial probe");
    ImGui::SliderFloat("Station s", &ui.probe_s, 0.0f, gpu.pipe_length, "%.3f m");
    ImGui::SliderFloat("Slab", &ui.slab_spacings, 0.5f, 8.0f, "%.1f spacings");
    ImGui::Text("In slab %d", gpu.stats.count);
    ImGui::Text("Mean speed %.4f m/s", gpu.stats.mean_speed);
    ImGui::Text("Mean axial %.4f m/s", gpu.stats.mean_axial_speed);
    ImGui::Text("Mean pressure %.3f", gpu.stats.mean_pressure);
    ImGui::Text("Mean density %.2f", gpu.stats.mean_density);

    if (gpu.has_pipe) {
        const PipeSpec& spec = *simulation.getPipe();
        const float area = pipeAreaAt(spec, ui.probe_s);
        const float inlet_area = pipeAreaAt(spec, 0.0f);
        ImGui::Text("Area %.5f m^2", area);
        if (area > 0.0f) {
            const float continuity = spec.inflow.speed * inlet_area / area;
            ImGui::Text("Continuity %.4f m/s", continuity);
        }
    } else {
        ImGui::TextUnformatted("No pipe in this config.");
    }
    ImGui::TextUnformatted("Drag the slice to move the probe. Drag the panel's left edge to resize it.");
    ImGui::PopTextWrapPos();
    ImGui::End();

    ImGui::GetForegroundDrawList()->AddText(
        ImVec2(static_cast<float>(gpu.preview_x),
               static_cast<float>(gpu.preview_y - 18)),
        IM_COL32(220, 224, 230, 255), "Cross-section");
}

void dragProbe(FluidViewState& gpu, ViewUi& ui) {
    if (gpu.capture_mouse || !gpu.has_pipe)
        return;
    if (glfwGetMouseButton(gpu.window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS)
        return;
    float fx = 0.0f;
    float fy = 0.0f;
    if (!mouseInRect(gpu.window, gpu.longitudinal_viewport.x, gpu.longitudinal_viewport.y,
                     gpu.longitudinal_viewport.width, gpu.longitudinal_viewport.height, fx,
                     fy))
        return;
    if (mouseInRect(gpu.window, gpu.transverse_viewport.x, gpu.transverse_viewport.y,
                    gpu.transverse_viewport.width, gpu.transverse_viewport.height, fx, fy))
        return;
    const float t =
        (fx - gpu.longitudinal_viewport.x) /
        std::max(gpu.longitudinal_viewport.width, 1);
    const float s = gpu.longitudinal.min_x +
                    t * (gpu.longitudinal.max_x - gpu.longitudinal.min_x);
    ui.probe_s = std::clamp(s, 0.0f, gpu.pipe_length);
}

} // namespace

void FluidView::frame(FluidSimulation& simulation, ViewUi& ui) {
    glfwPollEvents();
    ui.text_scale = std::clamp(ui.text_scale, 1.0f, 2.5f);
    ImGui::GetStyle().FontScaleMain = ui.text_scale;
    const float spacing = std::max(simulation.getFluidSpacing(), 1.0e-4f);
    layout(*state_, simulation.getPipe(), spacing);
    dragProbe(*state_, ui);

    const int steps = ui.step_once ? 1 : (ui.paused ? 0 : ui.steps_per_frame);
    ui.step_once = false;
    const float dt = simulation.getParams().time_step;
    for (int step = 0; step < steps; ++step)
        simulation.update(dt);

    ui.probe_s = std::clamp(ui.probe_s, 0.0f, state_->pipe_length);
    prepareFrame(*state_, simulation, ui);
    renderScene(*state_, simulation, ui);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    drawPanel(*state_, simulation, ui);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    state_->capture_mouse = ImGui::GetIO().WantCaptureMouse;
    glfwSwapBuffers(state_->window);
}
