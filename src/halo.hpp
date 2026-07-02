#pragma once
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include "sim_state.hpp"
#include "config.hpp"

namespace lbm {

// Two-phase halo exchange for a 2D-decomposed D2Q9 tile.
//   Phase 1 (X): swap columns x=1 / x=Nx_local with left/right neighbors, using
//                interior rows only. Fills halo columns x=0 and x=Nx_local+1.
//   Phase 2 (Y): swap rows y=1 / y=Ny_local with bottom/top neighbors, packing
//                the FULL x-range (including halo columns just filled). This
//                propagates corner data via the diagonal-through-cardinal path,
//                so we get 8-neighbor connectivity with only 4 sendrecv pairs.
void halo_exchange(SimState& s, const Config& cfg, const Decomp& dec) {
    auto f      = s.f;
    auto sl_buf = s.send_left_buf;
    auto sr_buf = s.send_right_buf;
    auto rl_buf = s.recv_left_buf;
    auto rr_buf = s.recv_right_buf;
    auto st_buf = s.send_top_buf;
    auto sb_buf = s.send_bottom_buf;
    auto rt_buf = s.recv_top_buf;
    auto rb_buf = s.recv_bottom_buf;

    const int Nx_local = dec.Nx_local;   // interior tile size
    const int Ny_local = dec.Ny_local;
    const int Nx_full  = Nx_local + 2;   // halo-inclusive x-range

    // ================== Phase 1: X exchange ==================
    // Pack column x=1 (send left) and x=Nx_local (send right), interior rows only.
    Kokkos::parallel_for("halo_pack_x", Ny_local, KOKKOS_LAMBDA(int j) {
        int y = j + 1;
        for (int q = 0; q < 9; q++) {
            sl_buf(j * 9 + q) = f(1,        y, q);
            sr_buf(j * 9 + q) = f(Nx_local, y, q);
        }
    });
    Kokkos::fence();

    auto sl_host = Kokkos::create_mirror_view(sl_buf);
    auto sr_host = Kokkos::create_mirror_view(sr_buf);
    auto rl_host = Kokkos::create_mirror_view(rl_buf);
    auto rr_host = Kokkos::create_mirror_view(rr_buf);
    Kokkos::deep_copy(sl_host, sl_buf);
    Kokkos::deep_copy(sr_host, sr_buf);

    const int count_x = Ny_local * 9;
    MPI_Sendrecv(
        sr_host.data(), count_x, MPI_DOUBLE, dec.right_rank, 0,
        rl_host.data(), count_x, MPI_DOUBLE, dec.left_rank,  0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(
        sl_host.data(), count_x, MPI_DOUBLE, dec.left_rank,  1,
        rr_host.data(), count_x, MPI_DOUBLE, dec.right_rank, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    Kokkos::deep_copy(rl_buf, rl_host);
    Kokkos::deep_copy(rr_buf, rr_host);

    // Unpack into halo columns x=0 (left) and x=Nx_local+1 (right).
    Kokkos::parallel_for("halo_unpack_x", Ny_local, KOKKOS_LAMBDA(int j) {
        int y = j + 1;
        for (int q = 0; q < 9; q++) {
            f(0,            y, q) = rl_buf(j * 9 + q);
            f(Nx_local + 1, y, q) = rr_buf(j * 9 + q);
        }
    });
    Kokkos::fence();

    // ================== Phase 2: Y exchange ==================
    // Pack row y=1 (send bottom) and y=Ny_local (send top) across the FULL
    // x-range so corner cells are transferred diagonally through the neighbor.
    Kokkos::parallel_for("halo_pack_y", Nx_full, KOKKOS_LAMBDA(int x) {
        for (int q = 0; q < 9; q++) {
            sb_buf(x * 9 + q) = f(x, 1,        q);
            st_buf(x * 9 + q) = f(x, Ny_local, q);
        }
    });
    Kokkos::fence();

    auto st_host = Kokkos::create_mirror_view(st_buf);
    auto sb_host = Kokkos::create_mirror_view(sb_buf);
    auto rt_host = Kokkos::create_mirror_view(rt_buf);
    auto rb_host = Kokkos::create_mirror_view(rb_buf);
    Kokkos::deep_copy(st_host, st_buf);
    Kokkos::deep_copy(sb_host, sb_buf);

    const int count_y = Nx_full * 9;
    MPI_Sendrecv(
        st_host.data(), count_y, MPI_DOUBLE, dec.top_rank,    2,
        rb_host.data(), count_y, MPI_DOUBLE, dec.bottom_rank, 2,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(
        sb_host.data(), count_y, MPI_DOUBLE, dec.bottom_rank, 3,
        rt_host.data(), count_y, MPI_DOUBLE, dec.top_rank,    3,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    Kokkos::deep_copy(rt_buf, rt_host);
    Kokkos::deep_copy(rb_buf, rb_host);

    // Unpack into halo rows y=0 (bottom) and y=Ny_local+1 (top).
    Kokkos::parallel_for("halo_unpack_y", Nx_full, KOKKOS_LAMBDA(int x) {
        for (int q = 0; q < 9; q++) {
            f(x, 0,            q) = rb_buf(x * 9 + q);
            f(x, Ny_local + 1, q) = rt_buf(x * 9 + q);
        }
    });
    Kokkos::fence();
}

} // namespace lbm
