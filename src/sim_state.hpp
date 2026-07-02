#pragma once
#include <Kokkos_Core.hpp>
#include "config.hpp"

struct SimState {
    // cfg.Nx_local / cfg.Ny_local already include the 1-cell halo on each side
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
    Kokkos::View<double*> send_top_buf;
    Kokkos::View<double*> send_bottom_buf;
    Kokkos::View<double*> recv_top_buf;
    Kokkos::View<double*> recv_bottom_buf;

    explicit SimState(const Config &cfg)
        : f           ("f",           cfg.Nx_local, cfg.Ny_local, 9),
          f_new       ("f_new",       cfg.Nx_local, cfg.Ny_local, 9),
          rho         ("rho",         cfg.Nx_local, cfg.Ny_local),
          u           ("u",           cfg.Nx_local, cfg.Ny_local, 2),
          u_old       ("u_old",       cfg.Nx_local, cfg.Ny_local, 2),
          mask        ("mask",        cfg.Nx_local, cfg.Ny_local),
          dest_x      ("dest_x",      cfg.Nx_local, cfg.Ny_local, 9),
          dest_y      ("dest_y",      cfg.Nx_local, cfg.Ny_local, 9),
          dest_q      ("dest_q",      cfg.Nx_local, cfg.Ny_local, 9),
          bounce_corr ("bounce_corr", cfg.Nx_local, cfg.Ny_local, 9),
          wall_ux     ("wall_ux",     cfg.Nx_local, cfg.Ny_local),
          wall_uy     ("wall_uy",     cfg.Nx_local, cfg.Ny_local),
          send_left_buf  ("send_l",   cfg.Ny_local * 9),
          send_right_buf ("send_r",   cfg.Ny_local * 9),
          recv_left_buf  ("recv_l",   cfg.Ny_local * 9),
          recv_right_buf ("recv_r",   cfg.Ny_local * 9),
          send_top_buf   ("send_t",   cfg.Nx_local * 9),
          send_bottom_buf("send_b",   cfg.Nx_local * 9),
          recv_top_buf   ("recv_t",   cfg.Nx_local * 9),
          recv_bottom_buf("recv_b",   cfg.Nx_local * 9)
    {}

    void swap_distributions() {
        auto temp = f;
        f     = f_new;
        f_new = temp;
    }
};
