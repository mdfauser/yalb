cmake_minimum_required(VERSION 3.16)
project(MyProject CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Kokkos REQUIRED)

add_library(lib STATIC hello.cpp)
target_include_directories(lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lib PUBLIC Kokkos::kokkos MPI::MPI_CXX Eigen3::Eigen)

add_executable(LBM main.cpp)
target_link_libraries(LBM PRIVATE lib)
