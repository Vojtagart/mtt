cmake_minimum_required(VERSION 3.15)
project(MultipleTargetTracking LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
    murty
    GIT_REPOSITORY https://github.com/Vojtagart/murty.git
    GIT_TAG        master
)
FetchContent_Declare(
    nanoflann
    GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
    GIT_TAG        v1.8.0
)
FetchContent_MakeAvailable(murty nanoflann)

find_package(Eigen3 REQUIRED)

add_library(mtt INTERFACE)

target_include_directories(mtt INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(mtt INTERFACE cxx_std_20)
target_link_libraries(mtt INTERFACE 
    murty 
    nanoflann::nanoflann 
    Eigen3::Eigen
)

option(MTT_BUILD_PYTHON_BINDINGS "Build Pybind11 bindings" ON)

if(MTT_BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()
