find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core bindings.cpp)

target_link_libraries(_core PRIVATE mtt)

if(MSVC)
    target_compile_options(_core PRIVATE /W4) 
else()
    target_compile_options(_core PRIVATE -Wall -Wextra -pedantic)
endif()

install(TARGETS _core DESTINATION mtt)