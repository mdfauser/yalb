#pragma once
#include "sim_state.hpp"
#include "lattice.hpp"
#include "config.hpp"

namespace lbm {

inline void compute_density(SimState& s, const Config& cfg) {
    auto f    = s.f;
    auto rho  = s.rho;
    auto mask = s.mask;
    const int Nx_local = cfg.Nx_local - 2;  // interior tile x-size
    const int Ny_local = cfg.Ny_local - 2;  // interior tile y-size

    // Loop over INTERIOR cells: (x,y) in [1, Nx_local] x [1, Ny_local], skip halos
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {Nx_local + 1, Ny_local + 1}),
        KOKKOS_LAMBDA(int x, int y) {
            if (mask(x, y) == 0) return;
            double r = 0.0;
            for (int q = 0; q < 9; q++) r += f(x, y, q);
            rho(x, y) = r;
        });
}

inline void compute_velocity(SimState& s, const Lattice& lat, const Config& cfg) {
    auto f    = s.f;
    auto rho  = s.rho;
    auto u    = s.u;
    auto mask = s.mask;
    const int Nx_local = cfg.Nx_local - 2;
    const int Ny_local = cfg.Ny_local - 2;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {Nx_local + 1, Ny_local + 1}),
        KOKKOS_LAMBDA(int x, int y) {
            if (mask(x, y) == 0) return;
            double ux = 0.0, uy = 0.0;
            for (int q = 0; q < 9; q++) {
                ux += f(x, y, q) * D2Q9::cx(q);
                uy += f(x, y, q) * D2Q9::cy(q);
            }
            u(x, y, 0) = ux / rho(x, y);
            u(x, y, 1) = uy / rho(x, y);
        });
}

inline void collide_stream(SimState& s, const Lattice& lat, const Config& cfg) {
    auto f      = s.f;
    auto rho_v  = s.rho;
    auto u_v    = s.u;
    auto f_new  = s.f_new;
    auto mask   = s.mask;
    auto wux    = s.wall_ux;
    auto wuy    = s.wall_uy;
    const int Nx_local = cfg.Nx_local - 2;
    const int Ny_local = cfg.Ny_local - 2;
    const double tau = cfg.tau;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {Nx_local + 1, Ny_local + 1}),
        KOKKOS_LAMBDA(int x, int y) {
        if (mask(x, y) == 0) return;
        double rho = 0.0, ux = 0.0, uy = 0.0;
        double f_local[9];

        for (int q = 0; q < 9; q++) {
            int xn = x - D2Q9::cx(q);
            int yn = y - D2Q9::cy(q);

            if (mask(xn, yn) == 0) {
                int qo = D2Q9::opp(q);
                double cu_wall = D2Q9::cx(q) * wux(xn, yn) + D2Q9::cy(q) * wuy(xn, yn);
                f_local[q] = f(x, y, qo) + 2.0 * D2Q9::w(qo) * (cu_wall / (1.0/3.0));
            } else {
                f_local[q] = f(xn, yn, q);
            }
        }
        for (int q = 0; q < 9; q++) {
            rho += f_local[q];
            ux  += D2Q9::cx(q) * f_local[q];
            uy  += D2Q9::cy(q) * f_local[q];
        }
        ux /= rho;
        uy /= rho;
        rho_v(x, y)    = rho;
        u_v(x, y, 0)   = ux;
        u_v(x, y, 1)   = uy;
        double udotu = ux * ux + uy * uy;

        for (int q = 0; q < 9; q++) {
            double cu = D2Q9::cx(q) * ux + D2Q9::cy(q) * uy;
            double f_eq = D2Q9::w(q) * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*udotu);
            f_new(x, y, q) = f_local[q] - (f_local[q] - f_eq) / tau;
        }

        });
}

// deprecated
inline void stream_bounce_back(SimState& s, const Config& cfg) {
    auto f      = s.f;
    auto f_new  = s.f_new;
    auto dx     = s.dest_x;
    auto dy     = s.dest_y;
    auto dq     = s.dest_q;
    auto mask   = s.mask;
    auto bc = s.bounce_corr;
    const int Nx_local = cfg.Nx_local - 2;
    const int Ny_local = cfg.Ny_local - 2;

    Kokkos::deep_copy(f_new, 0.0);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {Nx_local + 1, Ny_local + 1}),
        KOKKOS_LAMBDA(int x, int y) {
            if (mask(x, y) == 0) return;
            for (int q = 0; q < 9; q++) {
                f_new(dx(x, y, q), dy(x, y, q), dq(x, y, q)) = f(x, y, q) + bc(x, y, q);
            }
        });
}

// Periodic streaming
inline void stream_periodic(SimState& s, const Lattice& lat, const Config& cfg) {
    auto f     = s.f;
    auto f_new = s.f_new;
    const int Nx_local = cfg.Nx_local - 2;
    const int Ny_local = cfg.Ny_local - 2;

    Kokkos::deep_copy(f_new, 0.0);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {Nx_local + 1, Ny_local + 1}),
        KOKKOS_LAMBDA(int x, int y) {
            for (int q = 0; q < 9; q++) {
                int xn = (x + D2Q9::cx(q) + Nx_local) % Nx_local;
                int yn = (y + D2Q9::cy(q) + Ny_local) % Ny_local;
                f_new(xn, yn, q) = f(x, y, q);
            }
        });
}

}
