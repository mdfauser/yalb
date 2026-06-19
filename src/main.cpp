#include <Kokkos_Core.hpp>
#include <iostream>
#include <iomanip>

#include "config.hpp"
#include "lattice.hpp"
#include "sim_state.hpp"
#include "boundary.hpp"
#include "kernels.hpp"
#include "diagnostics.hpp"
#include "io.hpp"

int main(int argc, char** argv) {
    Config cfg;
    // Override defaults here or parse from argv later
    // cfg.Nx = 400;  cfg.tau = 0.6;  etc.

    Kokkos::initialize(argc, argv);
    {
        Lattice  lat;
        SimState s(cfg);

        // --- Setup ---
        lbm::initialize_mask(s, cfg);
        lbm::initialize_wall_velocity(s, cfg);
        lbm::setup_streaming_targets(s, lat, cfg);
        lbm::init_shear_wave(s, lat, cfg);

        double initial_mass = lbm::compute_mass(s, cfg);
        std::cout << std::setprecision(15)
                  << "Initial mass: " << initial_mass << "\n";

        // --- Time loop ---
        for (int step = 0; step <= cfg.N_steps; step++) {
            lbm::compute_density(s, cfg);
            lbm::compute_velocity(s, lat, cfg);

            // Momentum conservation check (diagnostic steps only)
            if (step % cfg.diag_interval == 0) {
                auto pre = lbm::compute_momentum(s, lat, cfg);
                lbm::collide(s, lat, cfg);
                auto post = lbm::compute_momentum(s, lat, cfg);
                lbm::print_momentum_check(pre, post, step);
            } else {
                lbm::collide(s, lat, cfg);
            }

            lbm::stream_bounce_back(s, cfg);
            s.swap_distributions();

            // Steady-state check
            if (step % cfg.steady_check_interval == 0 && step > 0) {
                double diff = lbm::check_steady_state(s, cfg);
                std::cout << "step " << step << " max change: " << diff << "\n";
                if (diff < cfg.steady_threshold) {
                    std::cout << "Steady state reached at step " << step << "\n";
                    break;
                }
                Kokkos::deep_copy(s.u_old, s.u);
            }

            // Output
            if (step % cfg.output_interval == 0) {
                lbm::write_csv(s, cfg, step);
            }

            // Mass conservation check
            if (step % cfg.diag_interval == 0) {
                double mass = lbm::compute_mass(s, cfg);
                std::cout << std::setprecision(15)
                          << "Current mass: " << mass << "\n";
            }
        }
    }
    Kokkos::finalize();
    return 0;
}
