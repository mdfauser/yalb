#pragma once
#include <fstream>
#include <string>
#include "sim_state.hpp"
#include "config.hpp"

namespace lbm {

// Each rank writes its own interior tile with global (x, y) coordinates.
// A post-process step can concatenate all rank files into a single grid.
inline void write_csv(const SimState& s, const Config& cfg, const Decomp& dec, int step) {
    auto rho_h = Kokkos::create_mirror_view(s.rho);
    auto u_h   = Kokkos::create_mirror_view(s.u);
    Kokkos::deep_copy(rho_h, s.rho);
    Kokkos::deep_copy(u_h,   s.u);

    std::string filename = std::string(cfg.output_dir) + "/output_"
                         + std::to_string(step) + "_rank"
                         + std::to_string(dec.rank) + ".csv";
    std::ofstream file(filename);
    file << "x,y,rho,ux,uy\n";

    // Interior tile: local [1, Nx_local] x [1, Ny_local]  →  global via x_start/y_start.
    for (int x_local = 1; x_local <= dec.Nx_local; x_local++) {
        for (int y_local = 1; y_local <= dec.Ny_local; y_local++) {
            int x_global = dec.x_start + x_local - 1;
            int y_global = dec.y_start + y_local - 1;
            file << x_global << ","
                 << y_global << ","
                 << rho_h(x_local, y_local) << ","
                 << u_h(x_local, y_local, 0) << ","
                 << u_h(x_local, y_local, 1) << "\n";
        }
    }
}

} // namespace lbm
