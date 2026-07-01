#pragma once
#include <Kokkos_Core.hpp>
#include "config.hpp"

struct SimState {
    const int Nx_with_ghosts;
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
    Kokkos::View<double*> send_left_buf;
    Kokkos::View<double*> send_right_buf;
    Kokkos::View<double*> recv_left_buf;
    Kokkos::View<double*> recv_right_buf;


    explicit SimState(const Config &cfg)
    : Nx_with_ghosts(cfg.Nx_local + 2),
        f         ("f",           Nx_with_ghosts, cfg.Ny, 9),
        f_new     ("f_new",       Nx_with_ghosts, cfg.Ny, 9),
        rho       ("rho",         Nx_with_ghosts, cfg.Ny),
        u         ("u",           Nx_with_ghosts, cfg.Ny, 2),
        u_old     ("u_old",       Nx_with_ghosts, cfg.Ny, 2),
        mask      ("mask",        Nx_with_ghosts, cfg.Ny),
        dest_x    ("dest_x",      Nx_with_ghosts, cfg.Ny, 9),
        dest_y    ("dest_y",      Nx_with_ghosts, cfg.Ny, 9),
        dest_q    ("dest_q",      Nx_with_ghosts, cfg.Ny, 9),
        bounce_corr("bounce_corr", Nx_with_ghosts, cfg.Ny, 9),
        wall_ux   ("wall_ux",     Nx_with_ghosts, cfg.Ny),
        wall_uy   ("wall_uy",     Nx_with_ghosts, cfg.Ny),
        send_left_buf ("send_l",  cfg.Ny * 9),
        send_right_buf("send_r",  cfg.Ny * 9),
        recv_left_buf ("recv_l",  cfg.Ny * 9),
        recv_right_buf("recv_r",  cfg.Ny * 9){
    }
    // Swap f and f_new after streaming
    void swap_distributions() {
        auto temp = f;
        f     = f_new;
        f_new = temp;
    }
};
