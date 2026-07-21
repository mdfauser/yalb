add_executable(LBM main.cpp)
target_include_directories(LBM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(LBM PUBLIC Kokkos::kokkos MPI::MPI_CXX Eigen3::Eigen)
