#pragma once
#include "sim_state.hpp"
#include "lattice.hpp"
#include "config.hpp"

namespace lbm {

inline void initialize_mask(SimState& s, const Config& cfg, const Decomp& dec) {
    auto mask_loc = s.mask;
    const int x_start   = dec.x_start;
    const int y_start   = dec.y_start;
    const int Nx_local  = dec.Nx_local;
    const int Ny_local  = dec.Ny_local;
    const int Nx_global = cfg.Nx_global;
    const int Ny_global = cfg.Ny_global;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx_local + 2, Ny_local + 2}),
        KOKKOS_LAMBDA(int x_local, int y_local) {
            int x_global = x_start + x_local - 1;
            int y_global = y_start + y_local - 1;
            // 0 for wall cells and 1 for fluid cells
            mask_loc(x_local, y_local) = (x_global == 0
                                       || x_global == Nx_global - 1
                                       || y_global == 0
                                       || y_global == Ny_global - 1) ? 0 : 1;
        });
}

inline void initialize_wall_velocity(SimState& s, const Config& cfg, const Decomp& dec) {
    auto wux = s.wall_ux;
    auto wuy = s.wall_uy;
    const int y_start   = dec.y_start;
    const int Nx_local  = dec.Nx_local;
    const int Ny_local  = dec.Ny_local;
    const int Ny_global = cfg.Ny_global;
    const double lid    = cfg.lid_speed;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx_local + 2, Ny_local + 2}),
        KOKKOS_LAMBDA(int x_local, int y_local) {
            int y_global = y_start + y_local - 1; // true global index
            wux(x_local, y_local) = (y_global == Ny_global - 1) ? lid : 0.0;
            wuy(x_local, y_local) = 0.0;
        });
}


void init_at_rest(SimState& s, const Lattice& lat, const Config& cfg) {
    auto f_loc    = s.f;
    auto rho_loc  = s.rho;
    auto u_loc    = s.u;
    auto mask_loc = s.mask;
    auto w_loc    = lat.w;
    // cfg.Nx_local / cfg.Ny_local are halo-inclusive tile sizes
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {cfg.Nx_local, cfg.Ny_local}),
        KOKKOS_LAMBDA(int x_local, int y_local) {
            if (mask_loc(x_local, y_local) == 0) {
                rho_loc(x_local, y_local)  = 0.0;
                u_loc(x_local, y_local, 0) = 0.0;
                u_loc(x_local, y_local, 1) = 0.0;
                for (int q = 0; q < 9; q++) f_loc(x_local, y_local, q) = 0.0;
                return;
            }
            rho_loc(x_local, y_local)  = 1.0;
            u_loc(x_local, y_local, 0) = 0.0;
            u_loc(x_local, y_local, 1) = 0.0;
            for (int q = 0; q < 9; q++) f_loc(x_local, y_local, q) = w_loc(q);
        });
}

inline void init_shear_wave(SimState& s, const Lattice& lat, const Config& cfg, const Decomp& dec) {
    auto f   = s.f;
    auto rho = s.rho;
    auto u   = s.u;
    auto w   = lat.w;
    auto cx  = lat.cx;
    auto cy  = lat.cy;
    const int y_start   = dec.y_start;
    const int Nx_local  = dec.Nx_local;
    const int Ny_local  = dec.Ny_local;
    const int Ny_global = cfg.Ny_global;
    const double u0     = cfg.u0;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {Nx_local + 1, Ny_local + 1}),
        KOKKOS_LAMBDA(int x_local, int y_local) {
            int y_global = y_start + y_local - 1;
            double r  = 1.0;
            double ux = u0 * Kokkos::sin(2.0 * M_PI * y_global / Ny_global);
            double uy = 0.0;

            u(x_local, y_local, 0) = ux;
            u(x_local, y_local, 1) = uy;
            rho(x_local, y_local)  = r;

            double udotu = ux * ux + uy * uy;
            for (int q = 0; q < 9; q++) {
                double cu = cx[q] * ux + cy[q] * uy;
                f(x_local, y_local, q) = w[q] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * udotu);
            }
        });
}

inline void init_density_bump(SimState& s, const Lattice& lat, const Config& cfg, const Decomp& dec) {
    auto f   = s.f;
    auto rho = s.rho;
    auto u   = s.u;
    auto w   = lat.w;
    const int x_start   = dec.x_start;
    const int y_start   = dec.y_start;
    const int Nx_local  = dec.Nx_local;
    const int Ny_local  = dec.Ny_local;
    const int Nx_global = cfg.Nx_global;
    const int Ny_global = cfg.Ny_global;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {Nx_local + 1, Ny_local + 1}),
        KOKKOS_LAMBDA(int x_local, int y_local) {
            int x_global = x_start + x_local - 1;
            int y_global = y_start + y_local - 1;
            double r = 1.0;
            if (x_global == Nx_global / 4     && y_global == Ny_global / 2) r = 1.1;
            if (x_global == 3 * Nx_global / 4 && y_global == Ny_global / 2) r = 0.9;

            u(x_local, y_local, 0) = 0.0;
            u(x_local, y_local, 1) = 0.0;
            for (int q = 0; q < 9; q++) f(x_local, y_local, q) = w[q] * r;
            rho(x_local, y_local) = r;
        });
}

}
