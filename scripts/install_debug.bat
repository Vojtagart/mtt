@echo off
echo Building and installing mtt in Debug mode (/Od /Zi, asserts enabled)...

set SKBUILD_CMAKE_BUILD_TYPE=Debug
pip install -e .

echo Debug installation complete!