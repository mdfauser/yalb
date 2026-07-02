#include "boundary.hpp"
#include "config.hpp"
#include "diagnostics.hpp"
#include "halo.hpp"
#include "io.hpp"
#include "kernels.hpp"
#include "lattice.hpp"
#include "sim_state.hpp"
#include "halo.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <string>

Decomp init_decomp(int Nx_global, int Ny_global) {
    Decomp d;
    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.size);

    d.Nx_global = Nx_global;
    d.Ny_global = Ny_global;

    int dims[2] = {0, 0};
    MPI_Dims_create(d.size, 2, dims);
    d.px = dims[0];
    d.py = dims[1];

    d.rank_x = d.rank % d.px;
    d.rank_y = d.rank / d.px;

    d.Nx_local = Nx_global / d.px;
    d.Ny_local = Ny_global / d.py;

    d.x_start = d.rank_x * d.Nx_local;
    d.y_start = d.rank_y * d.Ny_local;

    d.left_rank   = (d.rank_x == 0)        ? MPI_PROC_NULL : d.rank - 1;
    d.right_rank  = (d.rank_x == d.px - 1) ? MPI_PROC_NULL : d.rank + 1;
    d.bottom_rank = (d.rank_y == 0)        ? MPI_PROC_NULL : d.rank - d.px;
    d.top_rank    = (d.rank_y == d.py - 1) ? MPI_PROC_NULL : d.rank + d.px;

    return d;
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "MPI process " << rank <<  " of " << size << " starting up.\n";

    MPI_Barrier(MPI_COMM_WORLD);

    Config cfg;
    Decomp dec = init_decomp(cfg.Nx_global, cfg.Ny_global);
    if (cfg.Nx_global % dec.px != 0 || cfg.Ny_global % dec.py != 0) {
        if (rank == 0)
            std::cerr << "Nx_global must be divisible by px=" << dec.px
                      << " and Ny_global by py=" << dec.py << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cfg.Nx_local = dec.Nx_local + 2;
    cfg.Ny_local = dec.Ny_local + 2;
    cfg.x_start  = dec.x_start;
    cfg.y_start  = dec.y_start;
    std::cout << "[rank " << dec.rank << "] "
              << "grid=" << dec.px << "x" << dec.py
              << " coord=(" << dec.rank_x << "," << dec.rank_y << ")"
              << " tile=" << dec.Nx_local << "x" << dec.Ny_local
              << " start=(" << dec.x_start << "," << dec.y_start << ")"
              << " neighbors: L=" << dec.left_rank << " R=" << dec.right_rank
              << " B=" << dec.bottom_rank << " T=" << dec.top_rank << "\n";

    bool benchmark_mode = (argc > 1 && std::string(argv[1]) == "bench");
    Kokkos::initialize(argc, argv);
    {
        if (dec.rank == 0) {
            std::cout << "Backend: " << Kokkos::DefaultExecutionSpace::name() << "\n";
            std::cout << "size,Nx,Ny,steps,runtime_s,mlups\n";
        }

        if (benchmark_mode) {
            const int sizes[]  = {64, 128, 256, 512, 1024, 2048, 4096};
            const int n_warmup = 50;

            for (int N : sizes) {
                Config bcfg = cfg;
                bcfg.Nx_global = N;
                bcfg.Ny_global = N;
                bcfg.N_steps = std::max(200, 10000 / (N / 64));

                Decomp dec_bench = init_decomp(N, N);
                bcfg.Nx_local = dec_bench.Nx_local + 2;
                bcfg.Ny_local = dec_bench.Ny_local + 2;
                bcfg.x_start  = dec_bench.x_start;
                bcfg.y_start  = dec_bench.y_start;

                Lattice  lat;
                SimState s(bcfg);

                lbm::initialize_mask(s, bcfg, dec_bench);
                lbm::initialize_wall_velocity(s, bcfg, dec_bench);
                lbm::setup_streaming_targets(s, lat, bcfg, dec_bench);
                lbm::init_at_rest(s, lat, bcfg);

                // Baseline mass (post-init, pre-warmup) for the conservation check.
                double initial_mass = lbm::compute_mass(s, bcfg);
                if (dec.rank == 0) {
                    std::cout << std::setprecision(15)
                              << "[N=" << N << "] initial global mass: "
                              << initial_mass << "\n";
                }

                for (int step = 0; step < n_warmup; step++) {
                    lbm::halo_exchange(s, bcfg, dec_bench);
                    lbm::compute_density(s, bcfg);
                    lbm::compute_velocity(s, lat, bcfg);
                    lbm::collide_stream(s, lat, bcfg);
                    s.swap_distributions();
                }
                Kokkos::fence();

                double post_warmup_mass = lbm::compute_mass(s, bcfg);
                if (dec.rank == 0) {
                    std::cout << std::setprecision(15)
                              << "[N=" << N << "] post-warmup global mass: "
                              << post_warmup_mass
                              << "  drift: " << (post_warmup_mass - initial_mass)
                              << "\n";
                }

                auto start = std::chrono::high_resolution_clock::now();
                for (int step = 0; step < bcfg.N_steps; step++) {
                    lbm::halo_exchange(s, bcfg, dec_bench);
                    lbm::compute_density(s, bcfg);
                    lbm::compute_velocity(s, lat, bcfg);
                    lbm::collide_stream(s, lat, bcfg);
                    s.swap_distributions();
                }
                Kokkos::fence();
                auto end = std::chrono::high_resolution_clock::now();

                double local_secs  = std::chrono::duration<double>(end - start).count();
                double max_secs;
                MPI_Allreduce(&local_secs, &max_secs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

                // Per-rank mass to spot rank-imbalance or a single bad tile.
                double local_mass = lbm::compute_local_mass(s, bcfg, dec_bench);
                std::cout << "[rank " << dec.rank << "] local mass: " << local_mass << "\n";

                // Global mass conservation summary (rank 0 only).
                double final_mass = lbm::compute_mass(s, bcfg);
                if (dec.rank == 0) {
                    double abs_drift = final_mass - initial_mass;
                    double rel_drift = abs_drift / initial_mass;
                    std::cout << std::setprecision(15)
                              << "[N=" << N << "] final global mass:   "
                              << final_mass << "\n"
                              << "[N=" << N << "] mass drift: abs="
                              << abs_drift << " rel=" << rel_drift << "\n";
                }

                double cells = double(bcfg.Nx_global) * double(bcfg.Ny_global);
                double mlups = (cells * bcfg.N_steps) / (max_secs * 1e6);

                if (rank == 0) {
                    std::cout << N << "," << bcfg.Nx_global << "," << bcfg.Ny_global << ","
                              << bcfg.N_steps << "," << max_secs << "," << mlups << "\n";
                }
            }
        }
        else {
            Lattice  lat;
            SimState s(cfg);

            lbm::initialize_mask(s, cfg, dec);
            lbm::initialize_wall_velocity(s, cfg, dec);
            lbm::setup_streaming_targets(s, lat, cfg, dec);
            lbm::init_at_rest(s, lat, cfg);


            double initial_mass = lbm::compute_mass(s, cfg);
            if (dec.rank == 0)
                std::cout << std::setprecision(15)
                          << "Initial mass: " << initial_mass << "\n";

            for (int step = 0; step <= cfg.N_steps; step++) {
                lbm::halo_exchange(s, cfg, dec);
                lbm::compute_density(s, cfg);
                lbm::compute_velocity(s, lat, cfg);

                if (step % cfg.diag_interval == 0) {
                    auto pre  = lbm::compute_momentum(s, lat, cfg);
                    lbm::collide_stream(s, lat, cfg);
                    auto post = lbm::compute_momentum(s, lat, cfg);
                    if (dec.rank == 0) lbm::print_momentum_check(pre, post, step);
                } else {
                    lbm::collide_stream(s, lat, cfg);
                }

                s.swap_distributions();

                if (step % cfg.steady_check_interval == 0 && step > 0) {
                    double diff = lbm::check_steady_state(s, cfg);
                    if (dec.rank == 0)
                        std::cout << "step " << step << " max change: " << diff << "\n";
                    if (diff < cfg.steady_threshold) {
                        if (dec.rank == 0)
                            std::cout << "Steady state reached at step " << step << "\n";
                        break;
                    }
                    Kokkos::deep_copy(s.u_old, s.u);
                }

                // if (step % cfg.output_interval == 0) {
                //     lbm::write_csv(s, cfg, step);
                // }

                if (step % cfg.diag_interval == 0) {
                    double mass = lbm::compute_mass(s, cfg);
                    if (dec.rank == 0)
                        std::cout << std::setprecision(15)
                                  << "Current mass: " << mass << "\n";
                }
            }
        }
    }
    Kokkos::finalize();

    MPI_Finalize();
    return 0;
}
