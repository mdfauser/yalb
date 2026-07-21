#pragma once
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include "sim_state.hpp"
#include "config.hpp"

namespace lbm {

void halo_exchange(SimState& s, const Config& cfg, const Decomp& dec) {
    // Skip phases with no neighbors (all MPI_PROC_NULL).
    const bool has_x = dec.left_rank != MPI_PROC_NULL || dec.right_rank  != MPI_PROC_NULL;
    const bool has_y = dec.top_rank  != MPI_PROC_NULL || dec.bottom_rank != MPI_PROC_NULL;
    if (!has_x && !has_y) return;

    auto f = s.f;

    const int Nx_local = dec.Nx_local;   // interior tile size
    const int Ny_local = dec.Ny_local;
    const int Nx_full  = Nx_local + 2;   // halo-inclusive x-range

    // blocking Kokkos::deep_copy fences the execution space itself, and
    // device kernels are stream-ordered, so no explicit fences are needed.
    if (has_x) {
        auto sl_buf = s.send_left_buf;
        auto sr_buf = s.send_right_buf;
        auto rl_buf = s.recv_left_buf;
        auto rr_buf = s.recv_right_buf;

        // Pack column x=1 (send left) and x=Nx_local (send right), interior rows only.
        Kokkos::parallel_for("halo_pack_x", Ny_local, KOKKOS_LAMBDA(int j) {
            int y = j + 1;
            for (int q = 0; q < 9; q++) {
                sl_buf(j * 9 + q) = f(1,        y, q);
                sr_buf(j * 9 + q) = f(Nx_local, y, q);
            }
        });

        Kokkos::deep_copy(s.send_left_host,  sl_buf);
        Kokkos::deep_copy(s.send_right_host, sr_buf);

        const int count_x = Ny_local * 9;
        MPI_Sendrecv(
            s.send_right_host.data(), count_x, MPI_DOUBLE, dec.right_rank, 0,
            s.recv_left_host.data(),  count_x, MPI_DOUBLE, dec.left_rank,  0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(
            s.send_left_host.data(),  count_x, MPI_DOUBLE, dec.left_rank,  1,
            s.recv_right_host.data(), count_x, MPI_DOUBLE, dec.right_rank, 1,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        Kokkos::deep_copy(rl_buf, s.recv_left_host);
        Kokkos::deep_copy(rr_buf, s.recv_right_host);

        // Unpack into halo columns x=0 (left) and x=Nx_local+1 (right).
        Kokkos::parallel_for("halo_unpack_x", Ny_local, KOKKOS_LAMBDA(int j) {
            int y = j + 1;
            for (int q = 0; q < 9; q++) {
                f(0,            y, q) = rl_buf(j * 9 + q);
                f(Nx_local + 1, y, q) = rr_buf(j * 9 + q);
            }
        });
    }

    if (has_y) {
        auto st_buf = s.send_top_buf;
        auto sb_buf = s.send_bottom_buf;
        auto rt_buf = s.recv_top_buf;
        auto rb_buf = s.recv_bottom_buf;

        // Pack row y=1 (send bottom) and y=Ny_local (send top) across the FULL
        // x-range so corner cells are transferred diagonally through the neighbor.
        Kokkos::parallel_for("halo_pack_y", Nx_full, KOKKOS_LAMBDA(int x) {
            for (int q = 0; q < 9; q++) {
                sb_buf(x * 9 + q) = f(x, 1,        q);
                st_buf(x * 9 + q) = f(x, Ny_local, q);
            }
        });

        Kokkos::deep_copy(s.send_top_host,    st_buf);
        Kokkos::deep_copy(s.send_bottom_host, sb_buf);

        const int count_y = Nx_full * 9;
        MPI_Sendrecv(
            s.send_top_host.data(),    count_y, MPI_DOUBLE, dec.top_rank,    2,
            s.recv_bottom_host.data(), count_y, MPI_DOUBLE, dec.bottom_rank, 2,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(
            s.send_bottom_host.data(), count_y, MPI_DOUBLE, dec.bottom_rank, 3,
            s.recv_top_host.data(),    count_y, MPI_DOUBLE, dec.top_rank,    3,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        Kokkos::deep_copy(rt_buf, s.recv_top_host);
        Kokkos::deep_copy(rb_buf, s.recv_bottom_host);

        // Unpack into halo rows y=0 (bottom) and y=Ny_local+1 (top).
        Kokkos::parallel_for("halo_unpack_y", Nx_full, KOKKOS_LAMBDA(int x) {
            for (int q = 0; q < 9; q++) {
                f(x, 0,            q) = rb_buf(x * 9 + q);
                f(x, Ny_local + 1, q) = rt_buf(x * 9 + q);
            }
        });
    }
}

} // namespace lbm
