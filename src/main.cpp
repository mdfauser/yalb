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
    d.px = dims[0]; // number of processors in X-direction
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
    bool shear_wave_mode = (argc > 1 && std::string(argv[1]) == "shear");

    Kokkos::initialize(argc, argv);
    {
        // if (dec.rank == 0) {
        //     std::cout << "Backend: " << Kokkos::DefaultExecutionSpace::name() << "\n";
        //     std::cout << "size,Nx,Ny,steps,runtime_s,mlups\n";
        // }
        if (shear_wave_mode) {
            Config scfg = cfg;
            scfg.Nx_global = 128;
            scfg.Ny_global = 32;
            scfg.tau = 0.8;
            scfg.N_steps = 3000;

            Decomp dec_shear = init_decomp(scfg.Nx_global, scfg.Ny_global);
            scfg.Nx_local = dec_shear.Nx_local;
            scfg.x_start = dec_shear.x_start;

            Lattice lat;
            SimState s(scfg);

            // Set mask to all-fluid (no walls)
            Kokkos::deep_copy(s.mask, 1);
            // Set wall_ux, wall_uy to 0 (not used, but be safe)
            Kokkos::deep_copy(s.wall_ux, 0.0);
            Kokkos::deep_copy(s.wall_uy, 0.0);

            // Initialize f = f_eq with uy = u0*sin(k*x), ux = 0
            const double u0 = 0.01;
            const double k = 2.0 * M_PI / scfg.Nx_global;

            auto f = s.f;
            int Nx_local = dec_shear.Nx_local;
            int Ny = scfg.Ny_global;
            int x_start = dec_shear.x_start;

            Kokkos::parallel_for(
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {Nx_local+1, Ny+1}),
                KOKKOS_LAMBDA(int x, int y) {
                    int x_global = x_start + x - 1;
                    double uy = u0 * sin(k * x_global);
                    double ux = 0.0;
                    double rho = 1.0;
                    double udotu = ux*ux + uy*uy;
                    for (int q = 0; q < 9; q++) {
                        double cu = D2Q9::cx(q) * ux + D2Q9::cy(q) * uy;
                        f(x, y, q) = D2Q9::w(q) * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*udotu);
                    }
                });
            Kokkos::fence();

            // Time loop with sampling
            std::ofstream out;
            if (dec_shear.rank == 0) {
                out.open("shear_wave.csv");
                out << "step,amplitude\n";
            }

            for (int step = 0; step <= scfg.N_steps; step++) {
                // sample the fundamental mode amplitude every 25 steps
                if (step % 25 == 0) {
                    // project uy onto sin(k*x): A = (2/Nx) * sum uy(x) * sin(k*x)
                    // First compute local sum, then MPI_Allreduce for global
                    double local_sum = 0.0;
                    auto f_local = s.f;
                    int Nx_l = dec_shear.Nx_local;
                    int Ny_l = scfg.Ny_global;
                    int x_start_l = dec_shear.x_start;
                    double k_l = k;

                    Kokkos::parallel_reduce(
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {Nx_l+1, Ny_l+1}),
                        KOKKOS_LAMBDA(int x, int y, double& sum) {
                            int x_global = x_start_l + x - 1;
                            // uy = sum over q of cy[q]*f[q] / rho
                            double rho = 0.0, uy = 0.0;
                            for (int q = 0; q < 9; q++) {
                                rho += f_local(x, y, q);
                                uy += D2Q9::cy(q) * f_local(x, y, q);
                            }
                            uy /= rho;
                            sum += uy * sin(k_l * x_global);
                        }, local_sum);

                    double global_sum;
                    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    double amplitude = std::abs(2.0 * global_sum / (scfg.Nx_global * Ny));

                    if (dec_shear.rank == 0) {
                        out << step << "," << std::setprecision(15) << amplitude << "\n";
                    }
                }

                lbm::halo_exchange(s, scfg, dec_shear);
                lbm::collide_stream_bench(s, lat, scfg);
                s.swap_distributions();
            }

            if (dec_shear.rank == 0) out.close();

            if (dec_shear.rank == 0) std::cout << "shear_wave.csv written\n";
        }
        if (benchmark_mode) {
            const int sizes[]  = {64, 256, 512, 1024, 2048, 4096, 8192};
            const int n_warmup = 50;

            for (int N : sizes) {
                Config bcfg = cfg;
                bcfg.Nx_global = N;
                bcfg.Ny_global = N;
                bcfg.N_steps = std::max(200, 10000 / (N / 64));

                Decomp dec_bench = init_decomp(N, N);
                bcfg.Nx_local = dec_bench.Nx_local + 2; // add because of ghost cells
                bcfg.Ny_local = dec_bench.Ny_local + 2;
                bcfg.x_start  = dec_bench.x_start;
                bcfg.y_start  = dec_bench.y_start;

                Lattice  lat;
                SimState s(bcfg);

                lbm::initialize_mask(s, bcfg, dec_bench);
                lbm::initialize_wall_velocity(s, bcfg, dec_bench);

                lbm::init_at_rest(s, lat, bcfg);

                // Baseline mass (post-init, pre-warmup) for the conservation check.
                double initial_mass = lbm::compute_mass(s, bcfg);
                // if (dec.rank == 0) {
                //     std::cout << std::setprecision(15)
                //               << "[N=" << N << "] initial global mass: "
                //               << initial_mass << "\n";
                // }

                for (int step = 0; step < n_warmup; step++) {
                    lbm::halo_exchange(s, bcfg, dec_bench);
                    lbm::collide_stream_bench(s, lat, bcfg);
                    s.swap_distributions();
                }
                Kokkos::fence();

                // double post_warmup_mass = lbm::compute_mass(s, bcfg);
                // if (dec.rank == 0) {
                //     std::cout << std::setprecision(15)
                //               << "[N=" << N << "] post-warmup global mass: "
                //               << post_warmup_mass
                //               << "  drift: " << (post_warmup_mass - initial_mass)
                //               << "\n";
                // }

                auto start = std::chrono::high_resolution_clock::now();
                for (int step = 0; step < bcfg.N_steps; step++) {
                    lbm::halo_exchange(s, bcfg, dec_bench);
                    lbm::collide_stream_bench(s, lat, bcfg);
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

            lbm::init_at_rest(s, lat, cfg);


            double initial_mass = lbm::compute_mass(s, cfg);
            if (dec.rank == 0)
                std::cout << std::setprecision(15)
                          << "Initial mass: " << initial_mass << "\n";

            for (int step = 0; step <= cfg.N_steps; step++) {
                lbm::halo_exchange(s, cfg, dec);

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
