#pragma once
#include "sim_state.hpp"
#include "lattice.hpp"
#include "config.hpp"

namespace lbm {

inline void compute_density(SimState& s, const Config& cfg) {
    auto f    = s.f;
    auto rho  = s.rho;
    auto mask = s.mask;
    const int Nx = cfg.Nx, Ny = cfg.Ny;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
        KOKKOS_LAMBDA(int x, int y) {
            if (mask(x, y) == 0) return;
            double r = 0.0;
            for (int q = 0; q < 9; q++) r += f(x, y, q);
            rho(x, y) = r;
        });
}

inline void compute_velocity(SimState& s, const Lattice& lat, const Config& cfg) {
    auto f    = s.f;
    auto rho  = s.rho;
    auto u    = s.u;
    auto mask = s.mask;
    auto cx   = lat.cx;
    auto cy   = lat.cy;
    const int Nx = cfg.Nx, Ny = cfg.Ny;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
        KOKKOS_LAMBDA(int x, int y) {
            if (mask(x, y) == 0) return;
            double ux = 0.0, uy = 0.0;
            for (int q = 0; q < 9; q++) {
                ux += f(x, y, q) * cx[q];
                uy += f(x, y, q) * cy[q];
            }
            u(x, y, 0) = ux / rho(x, y);
            u(x, y, 1) = uy / rho(x, y);
        });
}

inline void collide_stream(SimState& s, const Lattice& lat, const Config& cfg) {
    auto f    = s.f;
    auto rho  = s.rho;
    auto u    = s.u;
    auto f_new  = s.f_new;
    auto dx     = s.dest_x;
    auto dy     = s.dest_y;
    auto dq     = s.dest_q;
    auto mask = s.mask;
    auto bc = s.bounce_corr;
    auto w    = lat.w;
    auto cx   = lat.cx;
    auto cy   = lat.cy;
    const int Nx = cfg.Nx, Ny = cfg.Ny;
    const double tau = cfg.tau;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
        KOKKOS_LAMBDA(int x, int y) {
            if (mask(x, y) == 0) return;
            double ux    = u(x, y, 0);
            double uy    = u(x, y, 1);
            double rho_v = rho(x, y);
            double udotu = ux * ux + uy * uy;
            for (int q = 0; q < 9; q++) {
                double cu   = cx[q] * ux + cy[q] * uy;
                double f_eq = w[q] * rho_v * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * udotu);
                double f_col = f(x, y, q) - (f(x, y, q) - f_eq) / tau;
                f_new(dx(x, y, q), dy(x, y, q), dq(x, y, q)) = f_col + bc(x, y, q);
            }
        });
}

inline void stream_bounce_back(SimState& s, const Config& cfg) {
    auto f      = s.f;
    auto f_new  = s.f_new;
    auto dx     = s.dest_x;
    auto dy     = s.dest_y;
    auto dq     = s.dest_q;
    auto mask   = s.mask;
    auto bc = s.bounce_corr;
    const int Nx = cfg.Nx, Ny = cfg.Ny;

    Kokkos::deep_copy(f_new, 0.0);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
        KOKKOS_LAMBDA(int x, int y) {
            if (mask(x, y) == 0) return;
            for (int q = 0; q < 9; q++) {
                f_new(dx(x, y, q), dy(x, y, q), dq(x, y, q)) = f(x, y, q) + bc(x, y, q);
            }
        });
}

// Periodic streaming (kept for validation / shear-wave tests)
inline void stream_periodic(SimState& s, const Lattice& lat, const Config& cfg) {
    auto f     = s.f;
    auto f_new = s.f_new;
    auto cx    = lat.cx;
    auto cy    = lat.cy;
    const int Nx = cfg.Nx, Ny = cfg.Ny;

    Kokkos::deep_copy(f_new, 0.0);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
        KOKKOS_LAMBDA(int x, int y) {
            for (int q = 0; q < 9; q++) {
                int xn = (x + cx[q] + Nx) % Nx;
                int yn = (y + cy[q] + Ny) % Ny;
                f_new(xn, yn, q) = f(x, y, q);
            }
        });
}

} // namespace lbm
