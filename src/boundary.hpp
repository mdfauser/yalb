#pragma once
#include "sim_state.hpp"
#include "lattice.hpp"
#include "config.hpp"

namespace lbm {

inline void initialize_mask(SimState& s, const Config& cfg, const Decomp& dec) {
    auto mask_loc = s.mask;
    int x_start = dec.x_start;
    int Nx_local = dec.Nx_local;
    int Nx_global = cfg.Nx_global, Ny = cfg.Ny;

    Kokkos::parallel_for(
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx_local + 2, Ny}),
    KOKKOS_LAMBDA(int x_local, int y) {
        int x_global = x_start + x_local - 1;
        mask_loc(x_local, y) = (x_global == 0
                             || x_global == Nx_global - 1
                             || y == 0
                             || y == Ny - 1) ? 0 : 1;
    });
}

inline void initialize_wall_velocity(SimState& s, const Config& cfg, const Decomp& dec) {
    auto wux = s.wall_ux;
    auto wuy = s.wall_uy;
    int Nx_local = dec.Nx_local, Ny = cfg.Ny;
    const double lid = cfg.lid_speed;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx_local + 2, Ny}),
        KOKKOS_LAMBDA(int x, int y) {
            wux(x, y) = (y == Ny - 1) ? lid : 0.0;
            wuy(x, y) = 0.0;
        });
}

inline void setup_streaming_targets(SimState& s, const Lattice& lat, const Config& cfg, const Decomp& dec) {
    auto dx   = s.dest_x;
    auto dy   = s.dest_y;
    auto dq   = s.dest_q;
    auto mask = s.mask;
    auto bc   = s.bounce_corr;
    auto wux  = s.wall_ux;
    auto wuy  = s.wall_uy;
    auto cx   = lat.cx;
    auto cy   = lat.cy;
    auto opp  = lat.opp;
    auto w    = lat.w;
    int Nx_local = dec.Nx_local, Ny = cfg.Ny;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx_local + 2, Ny}),
        KOKKOS_LAMBDA(int x, int y) {
            for (int q = 0; q < 9; q++) {
                int xn = (x + cx[q] + Nx_local) % Nx_local;
                int yn = (y + cy[q] + Ny) % Ny;
                bool is_wall = (mask(xn, yn) == 0);

                dx(x, y, q) = is_wall ? x : xn;
                dy(x, y, q) = is_wall ? y : yn;
                dq(x, y, q) = is_wall ? opp[q] : q;

                double cs2     = 1.0 / 3.0;
                double cu_wall = cx[q] * wux(xn, yn) + cy[q] * wuy(xn, yn);
                bc(x, y, q)   = is_wall ? -2.0 * w[q] * 1.0 * (cu_wall / cs2) : 0.0;
            }
        });
}

void init_at_rest(SimState& s, const Lattice& lat, const Config& cfg) {
    auto f_loc    = s.f;
    auto rho_loc  = s.rho;
    auto u_loc    = s.u;
    auto mask_loc = s.mask;
    auto w_loc    = lat.w;
    int Nx_local  = cfg.Nx_local;   // assumes you stored it in cfg

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx_local + 2, cfg.Ny}),
        KOKKOS_LAMBDA(int x_local, int y) {
            if (mask_loc(x_local, y) == 0) {
                rho_loc(x_local, y)  = 0.0;
                u_loc(x_local, y, 0) = 0.0;
                u_loc(x_local, y, 1) = 0.0;
                for (int q = 0; q < 9; q++) f_loc(x_local, y, q) = 0.0;
                return;
            }
            rho_loc(x_local, y)  = 1.0;
            u_loc(x_local, y, 0) = 0.0;
            u_loc(x_local, y, 1) = 0.0;
            for (int q = 0; q < 9; q++) f_loc(x_local, y, q) = w_loc(q);
        });
}

inline void init_shear_wave(SimState& s, const Lattice& lat, const Config& cfg) {
    auto f   = s.f;
    auto rho = s.rho;
    auto u   = s.u;
    auto w   = lat.w;
    auto cx  = lat.cx;
    auto cy  = lat.cy;
    int Nx_local = cfg.Nx_local, Ny = cfg.Ny;
    const double u0 = cfg.u0;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {Nx_local + 1, Ny}),
        KOKKOS_LAMBDA(int x, int y) {
            double r  = 1.0;
            double ux = u0 * Kokkos::sin(2.0 * M_PI * y / Ny);
            double uy = 0.0;

            u(x, y, 0)  = ux;
            u(x, y, 1)  = uy;
            rho(x, y)   = r;

            double udotu = ux * ux + uy * uy;
            for (int q = 0; q < 9; q++) {
                double cu = cx[q] * ux + cy[q] * uy;
                f(x, y, q) = w[q] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * udotu);
            }
        });
}

inline void init_density_bump(SimState& s, const Lattice& lat, const Config& cfg) {
    auto f   = s.f;
    auto rho = s.rho;
    auto u   = s.u;
    auto w   = lat.w;
    int Nx_local = cfg.Nx_local, Ny = cfg.Ny;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {Nx_local + 1, Ny}),
        KOKKOS_LAMBDA(int x, int y) {
            double r = 1.0;
            if (x == Nx_local / 4 && y == Ny / 2) r = 1.1;
            if (x == 3 * Nx_local / 4 && y == Ny / 2) r = 0.9;

            u(x, y, 0) = 0.0;
            u(x, y, 1) = 0.0;
            for (int q = 0; q < 9; q++) f(x, y, q) = w[q] * r;
            rho(x, y) = r;
        });
}

}
