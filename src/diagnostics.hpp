#pragma once
#include <iostream>
#include <iomanip>
#include <cmath>
#include <mpi.h>
#include "sim_state.hpp"
#include "lattice.hpp"
#include "config.hpp"

namespace lbm {

inline double compute_mass(const SimState& s, const Config& cfg) {
    auto f = s.f;
    const int Nx_local = cfg.Nx_local - 2;   // interior tile size
    const int Ny_local = cfg.Ny_local - 2;
    double local_mass = 0.0;

    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {Nx_local + 1, Ny_local + 1}),
        KOKKOS_LAMBDA(int x, int y, double& m) {
            for (int q = 0; q < 9; q++) m += f(x, y, q);
        }, local_mass);

    double global_mass = 0.0;
    MPI_Allreduce(&local_mass, &global_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_mass;
}

inline double compute_local_mass(const SimState& s, const Config& cfg, const Decomp& dec) {
    auto f = s.f;
    const int Nx_local = dec.Nx_local;   // interior tile size
    const int Ny_local = dec.Ny_local;
    double mass = 0.0;

    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {Nx_local + 1, Ny_local + 1}),
        KOKKOS_LAMBDA(int x, int y, double& m) {
            for (int q = 0; q < 9; q++) m += f(x, y, q);
        }, mass);
    return mass;
}

struct Momentum { double x = 0.0, y = 0.0; };

inline Momentum compute_momentum(const SimState& s, const Lattice& lat,
                                  const Config& cfg, bool fluid_only = true) {
    auto f    = s.f;
    auto mask = s.mask;
    auto cx   = lat.cx;
    auto cy   = lat.cy;
    const int Nx_local = cfg.Nx_local - 2;
    const int Ny_local = cfg.Ny_local - 2;
    double mx = 0.0, my = 0.0;

    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {Nx_local + 1, Ny_local + 1}),
        KOKKOS_LAMBDA(int x, int y, double& pmx, double& pmy) {
            if (fluid_only && mask(x, y) == 0) return;
            for (int q = 0; q < 9; q++) {
                pmx += f(x, y, q) * cx[q];
                pmy += f(x, y, q) * cy[q];
            }
        }, mx, my);

    double global[2] = {0.0, 0.0};
    double local[2]  = {mx, my};
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return {global[0], global[1]};
}

inline double check_steady_state(const SimState& s, const Config& cfg) {
    auto u     = s.u;
    auto u_old = s.u_old;
    const int Nx_local = cfg.Nx_local - 2;
    const int Ny_local = cfg.Ny_local - 2;
    double local_max = 0.0;

    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {Nx_local + 1, Ny_local + 1}),
        KOKKOS_LAMBDA(int x, int y, double& md) {
            double dx = u(x, y, 0) - u_old(x, y, 0);
            double dy = u(x, y, 1) - u_old(x, y, 1);
            double d  = Kokkos::sqrt(dx * dx + dy * dy);
            if (d > md) md = d;
        }, Kokkos::Max<double>(local_max));

    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global_max;
}

inline void print_momentum_check(const Momentum& pre, const Momentum& post, int step) {
    std::cout << std::setprecision(15)
              << "step " << step
              << " px: " << pre.x << " -> " << post.x
              << " diff: " << std::abs(post.x - pre.x)
              << " py: " << pre.y << " -> " << post.y
              << " diff: " << std::abs(post.y - pre.y) << "\n";
}

} // namespace lbm
