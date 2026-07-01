#pragma once
#include <iostream>
#include <iomanip>
#include <cmath>
#include "sim_state.hpp"
#include "lattice.hpp"
#include "config.hpp"

namespace lbm {

inline double compute_mass(const SimState& s, const Config& cfg) {
    auto f = s.f;
    int Nx_global = cfg.Nx_global, Ny = cfg.Ny;
    double mass = 0.0;

    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx_global, Ny}),
        KOKKOS_LAMBDA(int x, int y, double& m) {
            for (int q = 0; q < 9; q++) m += f(x, y, q);
        }, mass);

    return mass;
}

inline double compute_local_mass(const SimState& s, const Config& cfg, const Decomp& dec) {
    auto f = s.f;
    int Nx_local = dec.Nx_local;
    int Ny = f.extent(1);   // ← use View's actual size, not cfg
    double mass = 0.0;

    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {Nx_local+1, Ny}),
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
    int Nx_global = cfg.Nx_global, Ny = cfg.Ny;
    double mx = 0.0, my = 0.0;

    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx_global, Ny}),
        KOKKOS_LAMBDA(int x, int y, double& pmx, double& pmy) {
            if (fluid_only && mask(x, y) == 0) return;
            for (int q = 0; q < 9; q++) {
                pmx += f(x, y, q) * cx[q];
                pmy += f(x, y, q) * cy[q];
            }
        }, mx, my);

    return {mx, my};
}

inline double check_steady_state(const SimState& s, const Config& cfg) {
    auto u     = s.u;
    auto u_old = s.u_old;
    int Nx_global = cfg.Nx_global, Ny = cfg.Ny;
    double max_diff = 0.0;

    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx_global, Ny}),
        KOKKOS_LAMBDA(int x, int y, double& md) {
            double dx = u(x, y, 0) - u_old(x, y, 0);
            double dy = u(x, y, 1) - u_old(x, y, 1);
            double d  = Kokkos::sqrt(dx * dx + dy * dy);
            if (d > md) md = d;
        }, Kokkos::Max<double>(max_diff));

    return max_diff;
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
