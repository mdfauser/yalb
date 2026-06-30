#pragma once
#include <fstream>
#include <string>
#include "sim_state.hpp"
#include "config.hpp"

namespace lbm {

inline void write_csv(const SimState& s, const Config& cfg, int step) {
    auto rho_h = Kokkos::create_mirror_view(s.rho);
    auto u_h   = Kokkos::create_mirror_view(s.u);
    Kokkos::deep_copy(rho_h, s.rho);
    Kokkos::deep_copy(u_h, s.u);

    std::string filename = std::string(cfg.output_dir) + "/output_"
                         + std::to_string(step) + ".csv";
    std::ofstream file(filename);
    file << "x,y,rho,ux,uy\n";

    for (int x = 0; x < cfg.Nx_global; x++) {
        for (int y = 0; y < cfg.Ny; y++) {
            file << x << ","
                 << y << ","
                 << rho_h(x, y) << ","
                 << u_h(x, y, 0) << ","
                 << u_h(x, y, 1) << "\n";
        }
    }
}

} // namespace lbm
