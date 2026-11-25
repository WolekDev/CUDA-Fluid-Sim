# Requirements

GLAD, Dear ImGui, nlohmann/json, and GLFW are in `libraries/`. Install the items below for your system, then build from the repository root. The viewer needs an NVIDIA GPU and a display. Tests need only the compiler and the CUDA toolkit.

`CMakeLists.txt` requires CUDA toolkit 13.4, looks for `nvcc` at `/usr/local/cuda-13.4/bin/nvcc`, and compiles for compute capability 8.9 (RTX 40-series, including the RTX 4060). A different toolkit path or a different GPU means editing those two lines before configuring.

## Linux and WSL

Use a C++20 compiler, CMake 3.20 or newer, and the X11 and OpenGL headers GLFW compiles against. On WSL, install a current NVIDIA driver in Windows and do not install a Linux NVIDIA driver. The CUDA toolkit alone is enough inside WSL.

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config \
    libgl1-mesa-dev libx11-dev libxrandr-dev libxinerama-dev libxi-dev libxcursor-dev
```

Install the CUDA 13.4 toolkit so that `nvcc` lands at `/usr/local/cuda-13.4/bin/nvcc`. The Ubuntu 24.04 packages are:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-4
```

On native Linux, the matching driver package is `cuda-drivers` from that same repository. Skip it on WSL.

```bash
cmake -S . -B build
cmake --build build -j
./build/FluidSim config/settings.json
./build/FluidSimTest
```

## Windows

Windows uses the Win32 backend of the bundled GLFW. It does not need X11. OpenGL is `opengl32`, which comes with Windows. `nvcc` compiles the CPU code with MSVC, so install the Visual Studio 2022 Build Tools rather than the Visual Studio IDE. VS Code can open the folder and run the build from its terminal. You also need CMake and the CUDA 13.4 toolkit.

```bat
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
winget install Kitware.CMake
winget install Nvidia.CUDA
```

`winget install Nvidia.CUDA` tracks NVIDIA's current toolkit. If that is not 13.4, install 13.4 from https://developer.nvidia.com/cuda-13-4-0-download and leave the integration with the Build Tools selected.

Before configuring, point `CMAKE_CUDA_COMPILER` in `CMakeLists.txt` at the Windows `nvcc`, for example `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe`. Start "Developer PowerShell for VS 2022" from the Start menu so the MSVC compiler is on `PATH`. VS Code can be opened from that shell.

```bat
cmake -S . -B build
cmake --build build --config Release
build\Release\FluidSim.exe config\settings.json
build\Release\FluidSimTest.exe
```
