#pragma once

struct Config {
    int    Nx_global = 1024;
    int    Ny_global = 1024;
    int    Nx_local  = 1024 + 2;  // includes left/right halos
    int    Ny_local  = 1024 + 2;  // includes top/bottom halos
    int    N_steps   = 5000;
    double tau       = 0.8;
    double omega = 1.0 / tau;
    double u0        = 0.07;
    double lid_speed = 0.1;
    int    x_start;
    int    y_start;

    int    output_interval       = 10;
    int    steady_check_interval = 100;
    int    diag_interval         = 1000;
    double steady_threshold      = 1e-7;

    const char* output_dir = "./output";
};

struct Decomp {
    int rank, size;
    int px, py;
    int rank_x, rank_y;
    int Nx_global, Ny_global;
    int Nx_local, Ny_local;
    int x_start, y_start;
    int left_rank, right_rank;
    int top_rank,  bottom_rank;
};

