#pragma once
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include "sim_state.hpp"
#include "config.hpp"

namespace lbm {

void halo_exchange(SimState& s, const Config& cfg, const Decomp& dec) {
    auto f       = s.f;
    auto sl_buf  = s.send_left_buf;
    auto sr_buf  = s.send_right_buf;
    auto rl_buf  = s.recv_left_buf;
    auto rr_buf  = s.recv_right_buf;
    int Nx_local = dec.Nx_local;
    int Ny = f.extent(1);
    
    // 1. Pack
    Kokkos::parallel_for(Ny, KOKKOS_LAMBDA(int y) {
            for (int q = 0; q < 9; q++) {
                sl_buf(y * 9 + q) = f(1, y, q);
                sr_buf(y * 9 + q) = f(Nx_local, y, q);
            }
        });
    Kokkos::fence();

    // 2. communicate
    // copy on CPU
    auto sl_host = Kokkos::create_mirror_view(sl_buf);
    auto sr_host = Kokkos::create_mirror_view(sr_buf);
    auto rl_host = Kokkos::create_mirror_view(rl_buf);
    auto rr_host = Kokkos::create_mirror_view(rr_buf);
    // GPU -> CPU transfer
    Kokkos::deep_copy(sl_host, sl_buf);
    Kokkos::deep_copy(sr_host, sr_buf);

    int count = Ny * 9;
    // send right, receive left (reading from CPU)
    MPI_Sendrecv(
        sr_host.data(), count, MPI_DOUBLE, dec.right_rank, 0,
        rl_host.data(), count, MPI_DOUBLE, dec.left_rank,  0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
    // send left, receive right
    MPI_Sendrecv(
        sl_host.data(), count, MPI_DOUBLE, dec.left_rank,  1,
        rr_host.data(), count, MPI_DOUBLE, dec.right_rank, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    Kokkos::deep_copy(rl_buf, rl_host);
    Kokkos::deep_copy(rr_buf, rr_host);

    // 3. unpack
    Kokkos::parallel_for(Ny, KOKKOS_LAMBDA(int y) {
        for (int q = 0; q < 9; q++) {
            f(0, y, q)            = rl_buf(y * 9 + q);
            f(Nx_local + 1, y, q) = rr_buf(y * 9 + q);
        }
    });
    Kokkos::fence();

}
}