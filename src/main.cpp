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

int main(int argc, char** argv) {
    Config cfg;
    bool benchmark_mode = (argc > 1 && std::string(argv[1]) == "bench");

    Kokkos::initialize(argc, argv);
    {
        if (benchmark_mode) {
            std::cout << "Backend: " << Kokkos::DefaultExecutionSpace::name() << "\n";
            std::cout << "size,Nx,Ny,steps,runtime_s,mlups\n";

            const int sizes[]  = {64, 128, 256, 512, 1024, 2048, 4096};
            const int n_warmup = 50;

            for (int N : sizes) {
                Config bcfg = cfg;
                bcfg.Nx = N;
                bcfg.Ny = N;
                bcfg.N_steps = std::max(200, 10000 / (N / 64));

                Lattice  lat;
                SimState s(bcfg);

                lbm::initialize_mask(s, bcfg);
                lbm::initialize_wall_velocity(s, bcfg);
                lbm::setup_streaming_targets(s, lat, bcfg);

                for (int step = 0; step < n_warmup; step++) {
                    lbm::compute_density(s, bcfg);
                    lbm::compute_velocity(s, lat, bcfg);
                    lbm::collide_stream(s, lat, bcfg);
                    s.swap_distributions();
                }
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

                double secs  = std::chrono::duration<double>(end - start).count();
                double cells = double(bcfg.Nx) * double(bcfg.Ny);
                double mlups = (cells * bcfg.N_steps) / (secs * 1e6);

                std::cout << N << "," << bcfg.Nx << "," << bcfg.Ny << ","
                          << bcfg.N_steps << "," << secs << "," << mlups << "\n";
            }
        }
        else {
            Lattice  lat;
            SimState s(cfg);

            lbm::initialize_mask(s, cfg);
            lbm::initialize_wall_velocity(s, cfg);
            lbm::setup_streaming_targets(s, lat, cfg);

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
    return 0;
}
