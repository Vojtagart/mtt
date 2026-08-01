# mtt (Multiple Target Tracking)

A Multiple Target Tracking library implementing advanced filters such as PMBM, PMB, and PHD. The project features a high-performance, header-only C++20 backend and a Python frontend. 

The Python package exposes the C++ tracking filters via `pybind11` and includes built-in tools for trajectory generation, radar simulation, evaluation metrics (GOSPA), and real-time visualization using Matplotlib.

## Project Structure

* `include/mtt/`: Core C++20 header-only backend containing tracking filters (`pmbm/`, `phd/`) and core math/state utilities.
* `bindings/`: C++ code utilizing `pybind11` to expose the backend to Python.
* `src/mtt/`: Python package containing the compiled extension, pure Python wrappers, simulation scenarios, and visualization tools.
* `examples/`: Standalone Jupyter notebooks and scripts demonstrating typical tracking pipelines.
* `tests/python/`: Unit tests validating both the C++ bindings and Python utilities.
* `scripts/`: Shell scripts for automating builds and testing.

## Requirements

### System Dependencies
* CMake >= 3.15
* A C++20 compatible compiler
* **Eigen3**: Must be installed and discoverable by CMake via `find_package`.

*Note: `nanoflann` (v1.8.0) and `murty` are automatically fetched by CMake during the build process.*

### Python Dependencies

**Core Library:**
* Python >= 3.8
* `numpy` >= 1.20.0
* `matplotlib`
* `imageio`
* `murty` (Automatically fetched via Git in `pyproject.toml`)

**Build System:**
* `pybind11`
* `scikit-build-core`

**Testing & Evaluation:**
* `pytest`
* `scipy`

## Installation

### Python (via pip)
The library is built and installed using standard Python packaging tools. `scikit-build-core` will automatically invoke CMake to compile the C++ backend and link the dependencies.

```bash
# Install in the current environment
pip install .

# Or install in editable mode for development
pip install -e .
```

By default, `pyproject.toml` is configured to build in Release mode. This applies standard optimizations (`-O3` or `/O2`) for optimal performance.

### Advanced Python Builds (Debug & Profiling)

By default, the package builds in `Release` mode. For development and performance analysis, you can change the CMake build type.

Helper scripts are provided in the `scripts/` directory to automate this:

**1. Debug Mode**
Compiles without optimizations (`-O0`), adds debug symbols (`-g`), and enables C++ assertions (removes `-DNDEBUG`).

```bash
./scripts/install_debug.sh
# Equivalently: SKBUILD_CMAKE_BUILD_TYPE=Debug pip install -e .
```

**Profiling Mode (RelWithDebInfo)**
Compiles with release optimizations (`-O3`) but includes debug symbols (`-g`). This is the recommended mode for running performance benchmarks with tools like `perf` or `valgrind`.

```bash
./scripts/install_profile.sh
# Equivalently: SKBUILD_CMAKE_BUILD_TYPE=RelWithDebInfo pip install -e .
```

### C++ (via CMake FetchContent)

The core tracking algorithms are header-only. You can integrate them directly into an existing CMake project without building the Python bindings.

```cmake
include(FetchContent)

# Fetch MTT (which will recursively fetch murty and nanoflann)
FetchContent_Declare(
    mtt
    GIT_REPOSITORY https://github.com/Vojtagart/mtt.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(mtt)

# Ensure Eigen3 is available in your project
find_package(Eigen3 REQUIRED)

target_link_libraries(your_target PRIVATE mtt Eigen3::Eigen)
```

## Usage

For a comprehensive look at how to interact with the mtt API, refer to the examples/ directory. The examples cover setting up Trajectory, Radar, and Tracker instances, as well as visualizing the output using the Plotter class.

## Testing

The project uses `pytest` to validate the correctness of the C++ trackers through their Python bindings, as well as the pure Python simulation components. Ensure the package is installed in editable mode first.

```bash
pip install -e .
pytest tests/python/
```
