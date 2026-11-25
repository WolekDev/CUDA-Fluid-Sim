#include "FluidSimulation.cuh"
#include "FluidView.hpp"
#include "PipeGenerator.hpp"

#include <csignal>
#include <iostream>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace {

volatile std::sig_atomic_t g_quit = 0;
volatile std::sig_atomic_t g_cleanup_done = 0;

void requestQuit(int) { g_quit = 1; }

#ifdef _WIN32
BOOL WINAPI consoleCloseHandler(DWORD event) {
    switch (event) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        g_quit = 1;
        break;
    default:
        return FALSE;
    }
    // Closing the console terminates the process when this handler returns,
    // so wait until main has destroyed the simulation and the window.
    if (event == CTRL_CLOSE_EVENT || event == CTRL_LOGOFF_EVENT ||
        event == CTRL_SHUTDOWN_EVENT) {
        for (int wait = 0; wait < 200 && g_cleanup_done == 0; ++wait)
            Sleep(50);
    }
    return TRUE;
}
#endif

void installQuitHandlers() {
    std::signal(SIGINT, requestQuit);
    std::signal(SIGTERM, requestQuit);
#ifdef _WIN32
    SetConsoleCtrlHandler(consoleCloseHandler, TRUE);
#endif
}

} // namespace

int main(int argc, char** argv) {
    const std::string config = argc > 1 ? argv[1] : "config/settings.json";
    installQuitHandlers();
    try {
        FluidSimulation simulation;
        simulation.initialize(config);

        FluidView view;
        view.open(1280, 800);

        ViewUi ui;
        if (const PipeSpec* pipe = simulation.getPipe())
            ui.probe_s = 0.5f * pipeLength(*pipe);

        while (g_quit == 0 && !view.shouldClose())
            view.frame(simulation, ui);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        g_cleanup_done = 1;
        return 1;
    }
    g_cleanup_done = 1;
    return 0;
}
