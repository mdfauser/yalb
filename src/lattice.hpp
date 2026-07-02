#pragma once
#include <Kokkos_Core.hpp>

// Compile-time D2Q9 constants — safe to use in device lambdas.
// nvcc places namespace-scope constexpr arrays in constant memory for device
// code, giving broadcast L1-cached reads (all warp lanes hit the same q each
// iteration, so this is a pure broadcast access — optimal for constant cache).
namespace D2Q9 {
    constexpr int    cx[9]  = { 0, 1, 0,-1, 0, 1,-1,-1, 1};
    constexpr int    cy[9]  = { 0, 0, 1, 0,-1, 1, 1,-1,-1};
    constexpr int    opp[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6};
    constexpr double w[9]   = {4./9, 1./9, 1./9, 1./9, 1./9,
                               1./36, 1./36, 1./36, 1./36};
}

// D2Q9 lattice velocities, weights, and opposite directions — all in one struct.
// Pass `const Lattice&` instead of 5 separate Views.
struct Lattice {
    Kokkos::View<int*>    cx;
    Kokkos::View<int*>    cy;
    Kokkos::View<int*>    opp;
    Kokkos::View<double*> w;

    Lattice()
        : cx("cx", 9), cy("cy", 9), opp("opp", 9), w("w", 9)
    {
        auto cx_h  = Kokkos::create_mirror_view(cx);
        auto cy_h  = Kokkos::create_mirror_view(cy);
        auto opp_h = Kokkos::create_mirror_view(opp);
        auto w_h   = Kokkos::create_mirror_view(w);

        constexpr int    cx_v[9]  = { 0, 1, 0,-1, 0, 1,-1,-1, 1};
        constexpr int    cy_v[9]  = { 0, 0, 1, 0,-1, 1, 1,-1,-1};
        constexpr int    opp_v[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6};
        constexpr double w_v[9]   = {4./9, 1./9, 1./9, 1./9, 1./9,
                                     1./36, 1./36, 1./36, 1./36};

        for (int q = 0; q < 9; q++) {
            cx_h(q)  = cx_v[q];
            cy_h(q)  = cy_v[q];
            opp_h(q) = opp_v[q];
            w_h(q)   = w_v[q];
        }

        Kokkos::deep_copy(cx,  cx_h);
        Kokkos::deep_copy(cy,  cy_h);
        Kokkos::deep_copy(opp, opp_h);
        Kokkos::deep_copy(w,   w_h);
    }
};
