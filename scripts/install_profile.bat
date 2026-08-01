@echo off
echo Building and installing mtt in RelWithDebInfo mode (/O2 /Zi)...

set SKBUILD_CMAKE_BUILD_TYPE=RelWithDebInfo
pip install -e .

echo Profile installation complete!