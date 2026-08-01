#!/bin/bash
set -e

echo "Building and installing mtt in RelWithDebInfo mode (-O3 -g)..."
SKBUILD_CMAKE_BUILD_TYPE=RelWithDebInfo pip install -e .
echo "Profile installation complete!"