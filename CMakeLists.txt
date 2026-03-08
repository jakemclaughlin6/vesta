cmake_minimum_required(VERSION 3.20)
project(fuse LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_TESTING "Build tests" ON)

# Find dependencies
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLOG REQUIRED libglog)

# SuiteSparse (for CCOLAMD in fuse_constraints)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/fuse_constraints/cmake)
find_package(SuiteSparse REQUIRED COMPONENTS CCOLAMD)

# Testing - must be before add_subdirectory for tests to register
if(BUILD_TESTING)
  enable_testing()
endif()

# Add sub-packages in dependency order
add_subdirectory(fuse_core)
add_subdirectory(fuse_variables)
add_subdirectory(fuse_loss)
add_subdirectory(fuse_graphs)
add_subdirectory(fuse_constraints)
add_subdirectory(fuse_optimizers)
