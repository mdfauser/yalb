add_library(lib STATIC hello.cpp)
target_include_directories(lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lib PUBLIC Kokkos::kokkos MPI::MPI_CXX Eigen3::Eigen)

add_executable(LBM main.cpp)
target_link_libraries(LBM PRIVATE lib)
