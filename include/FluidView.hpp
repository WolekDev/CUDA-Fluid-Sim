#pragma once

#include "RenderPack.cuh"

struct FluidSimulation;
struct FluidViewState;

/* Controls edited by the side panel and the axial drag. */
struct ViewUi {
    bool paused = false;
    bool step_once = false;
    int steps_per_frame = 4;
    ParticleColorMode color = ParticleColorMode::Velocity;
    float probe_s = 0.0f;
    float slab_spacings = 2.0f;
    float text_scale = 1.6f;
};

/* GLFW window: longitudinal slice, cross-section preview, and probe readout. */
class FluidView {
public:
    FluidView();
    ~FluidView();

    FluidView(const FluidView&) = delete;
    FluidView& operator=(const FluidView&) = delete;

    void open(int width, int height);
    bool shouldClose() const;

    /* Advances the solver from the controls chosen on the previous frame,
     * then draws particles and the panel. */
    void frame(FluidSimulation& simulation, ViewUi& ui);

private:
    FluidViewState* state_ = nullptr;
};
