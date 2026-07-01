#include <Kokkos_Core.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include "config.hpp"
#include "lattice.hpp"
#include "sim_state.hpp"
#include "boundary.hpp"
#include "kernels.hpp"
#include "diagnostics.hpp"
#include "io.hpp"
#include <mpi.h>


Decomp init_decomp(int Nx_global) {
    Decomp d;
    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.size);

    d.Nx_global = Nx_global;
    d.Nx_local = Nx_global / d.size;
    d.x_start  = d.rank * d.Nx_local;

    d.left_rank  = (d.rank == 0)          ? MPI_PROC_NULL : d.rank - 1;
    d.right_rank = (d.rank == d.size - 1) ? MPI_PROC_NULL : d.rank + 1;

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
    if (cfg.Nx_global % size != 0) {
        if (rank == 0) std::cerr << "Nx_global must be divisible by number of processes\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    Decomp dec = init_decomp(cfg.Nx_global);  // pretend our global grid is 1024
    std::cout << "[rank " << dec.rank << "] guard check: rank==0 is "
          << (dec.rank == 0) << "\n";
    cfg.Nx_local =dec.Nx_local;
    cfg.x_start = dec.x_start;
    std::cout << "[rank " << dec.rank << "] "
              << "Nx_local=" << dec.Nx_local
              << " x_start=" << dec.x_start
              << " neighbors: left=" << dec.left_rank
              << " right=" << dec.right_rank << "\n";

    bool benchmark_mode = (argc > 1 && std::string(argv[1]) == "bench");
    Kokkos::initialize(argc, argv);
    {
        if (dec.rank == 0) {
            std::cout << "Backend: " << Kokkos::DefaultExecutionSpace::name() << "\n";
            std::cout << "size,Nx,Ny,steps,runtime_s,mlups\n";
        }

        if (benchmark_mode) {
            const int sizes[]  = {64}; // 64, 128, 256, 512, 1024, 2048, 4096
            const int n_warmup = 50;

            for (int N : sizes) {
                Config bcfg = cfg;
                bcfg.Nx_global = N;
                bcfg.Ny = N;
                bcfg.N_steps = std::max(200, 10000 / (N / 64));

                Decomp dec_bench = init_decomp(N);
                bcfg.Nx_local = dec_bench.Nx_local;
                bcfg.x_start  = dec_bench.x_start;

                Lattice  lat;
                SimState s(bcfg);

            //    lbm::initialize_mask(s, bcfg, dec);
                // lbm::initialize_wall_velocity(s, bcfg, dec);
                // lbm::setup_streaming_targets(s, lat, bcfg, dec);
                // lbm::init_at_rest(s, lat, bcfg);

                if (dec.rank == 0) std::cout << "SimState built\n" << std::flush;

                lbm::initialize_mask(s, bcfg, dec_bench);
                if (dec.rank == 0) std::cout << "mask initialized\n" << std::flush;

                lbm::initialize_wall_velocity(s, bcfg, dec_bench);
                if (dec.rank == 0) std::cout << "wall velocity initialized\n" << std::flush;

                lbm::setup_streaming_targets(s, lat, bcfg, dec_bench);
                if (dec.rank == 0) std::cout << "streaming targets set\n" << std::flush;

                lbm::init_at_rest(s, lat, bcfg);
                if (dec.rank == 0) std::cout << "init at rest done\n" << std::flush;

                if (dec.rank == 0) {
                    auto f_host = Kokkos::create_mirror_view(s.f);
                    Kokkos::deep_copy(f_host, s.f);

                    int counted_fluid = 0;
                    double naive_sum = 0;
                    double sum_at_one = 0;
                    int weird_cells = 0;

                    for (int x = 1; x < cfg.Nx_local + 1; x++) {
                        for (int y = 0; y < cfg.Ny; y++) {
                            double cell_mass = 0;
                            for (int q = 0; q < 9; q++) cell_mass += f_host(x, y, q);
                            naive_sum += cell_mass;
                            if (cell_mass > 0.99 && cell_mass < 1.01) {
                                sum_at_one += cell_mass;
                                counted_fluid++;
                            } else if (cell_mass != 0.0) {
                                weird_cells++;
                                if (weird_cells < 5) {
                                    std::cout << "[rank 0] weird cell at x=" << x << " y=" << y
                                              << " mass=" << cell_mass << "\n";
                                }
                            }
                        }
                    }
                    std::cout << "[rank 0] fluid cells with mass~1: " << counted_fluid
                              << "  weird cells: " << weird_cells
                              << "  naive_sum: " << naive_sum << "\n";

                }
                if (dec.rank == 0) std::cout << "debug block done\n" << std::flush;

                for (int step = 0; step < n_warmup; step++) {
                    if (dec.rank == 0 && step == 0) std::cout << "warmup starting\n" << std::flush;
                    lbm::compute_density(s, bcfg);
                    lbm::compute_velocity(s, lat, bcfg);
                    lbm::collide_stream(s, lat, bcfg);
                    s.swap_distributions();
                }
                if (dec.rank == 0) std::cout << "warmup done\n" << std::flush;
                Kokkos::fence();

                auto start = std::chrono::high_resolution_clock::now();
                for (int step = 0; step < bcfg.N_steps; step++) {
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

                double local_mass = lbm::compute_local_mass(s, bcfg, dec);
                std::cout << "[rank " << dec.rank << "] local mass: " << local_mass << "\n";

                double cells = double(bcfg.Nx_global) * double(bcfg.Ny);
                double mlups = (cells * bcfg.N_steps) / (max_secs * 1e6);

                if (rank == 0) {
                    std::cout << N << "," << bcfg.Nx_global << "," << bcfg.Ny << ","
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
            std::cout << std::setprecision(15)
                      << "Initial mass: " << initial_mass << "\n";

            for (int step = 0; step <= cfg.N_steps; step++) {
                lbm::compute_density(s, cfg);
                lbm::compute_velocity(s, lat, cfg);

                if (step % cfg.diag_interval == 0) {
                    auto pre  = lbm::compute_momentum(s, lat, cfg);
                    lbm::collide_stream(s, lat, cfg);
                    auto post = lbm::compute_momentum(s, lat, cfg);
                    lbm::print_momentum_check(pre, post, step);
                } else {
                    lbm::collide_stream(s, lat, cfg);
                }

                s.swap_distributions();

                if (step % cfg.steady_check_interval == 0 && step > 0) {
                    double diff = lbm::check_steady_state(s, cfg);
                    std::cout << "step " << step << " max change: " << diff << "\n";
                    if (diff < cfg.steady_threshold) {
                        std::cout << "Steady state reached at step " << step << "\n";
                        break;
                    }
                    Kokkos::deep_copy(s.u_old, s.u);
                }

                if (step % cfg.output_interval == 0) {
                    lbm::write_csv(s, cfg, step);
                }

                if (step % cfg.diag_interval == 0) {
                    double mass = lbm::compute_mass(s, cfg);
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
