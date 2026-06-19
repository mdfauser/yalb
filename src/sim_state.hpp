#pragma once
#include <Kokkos_Core.hpp>
#include "config.hpp"

// All simulation state in one place.
// Functions take `SimState&` instead of a dozen individual Views.
struct SimState {
    Kokkos::View<double***> f;
    Kokkos::View<double***> f_new;
    Kokkos::View<double**>  rho;
    Kokkos::View<double***> u;
    Kokkos::View<double***> u_old;

    // boundary
    Kokkos::View<int**>     mask;
    Kokkos::View<int***>    dest_x;
    Kokkos::View<int***>    dest_y;
    Kokkos::View<int***>    dest_q;
    Kokkos::View<double***> bounce_corr;
    Kokkos::View<double**>  wall_ux;
    Kokkos::View<double**>  wall_uy;

    explicit SimState(const Config& cfg) :
        f         ("f",          cfg.Nx, cfg.Ny, 9),
        f_new     ("f_new",      cfg.Nx, cfg.Ny, 9),
        rho       ("rho",        cfg.Nx, cfg.Ny),
        u         ("u",          cfg.Nx, cfg.Ny, 2),
        u_old     ("u_old",      cfg.Nx, cfg.Ny, 2),
        mask      ("mask",       cfg.Nx, cfg.Ny),
        dest_x    ("dest_x",     cfg.Nx, cfg.Ny, 9),
        dest_y    ("dest_y",     cfg.Nx, cfg.Ny, 9),
        dest_q    ("dest_q",     cfg.Nx, cfg.Ny, 9),
        bounce_corr("bounce_corr", cfg.Nx, cfg.Ny, 9),
        wall_ux   ("wall_ux",    cfg.Nx, cfg.Ny),
        wall_uy   ("wall_uy",    cfg.Nx, cfg.Ny)
    {}

    // Swap f and f_new after streaming
    void swap_distributions() {
        auto temp = f;
        f     = f_new;
        f_new = temp;
    }
};
