#!/bin/bash
set -e

echo "Building and installing mtt in Debug mode (-O0 -g, asserts enabled)..."
SKBUILD_CMAKE_BUILD_TYPE=Debug pip install -e .
echo "Debug installation complete!"