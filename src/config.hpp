#pragma once

// Runtime configuration for the simulation.
// No more global constexpr Nx/Ny — pass this around instead.
struct Config {
    int    Nx_global = 1024;
    int    Ny        = 1024;
    int    Nx_local  = 1024 +2;
    int    N_steps   = 5000;
    double tau       = 0.8;
    double u0        = 0.07;
    double lid_speed = 0.1;
    int x_start;

    int    output_interval      = 10;
    int    steady_check_interval = 100;
    int    diag_interval        = 1000;
    double steady_threshold     = 1e-7;

    // output path (no trailing slash)
    const char* output_dir = "./output";
};

struct Decomp {
    int rank, size;
    int Nx_global;
    int Nx_local;
    int x_start;
    int left_rank;
    int right_rank;
};

