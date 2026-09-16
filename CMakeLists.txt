cmake_minimum_required(VERSION 3.14)
project(hea_planner VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -Wall -Wextra -Wno-unused-parameter)
endif()

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Declare header-only interface library
add_library(hea_planner INTERFACE)
target_include_directories(hea_planner INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(hea_planner INTERFACE Eigen3::Eigen)



# Installation rules
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS hea_planner EXPORT hea_plannerTargets)
