#include <Kokkos_Core.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
// hardcoding
const int cx[9] = { 0, 1,-1, 0, 0, 1,-1,-1, 1}; // for cuda change the "const" to "constexpr"
const int cy[9] = { 0, 0, 0, 1,-1, 1, 1,-1,-1};
const double w[9] = {4./9, 1./9, 1./9, 1./9, 1./9,
                     1./36,1./36,1./36,1./36};

const int opp[9] = {0, 2, 1, 4, 3, 6, 5, 8, 7};

const int Nx = 20;
const int Ny = 20;

void computeDensity(Kokkos::View<double***> f, Kokkos::View<double**> rho){
        auto f_loc = f; // behaves like pointer so the real f is modified
        auto rho_loc = rho;
        Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx,Ny}),
            KOKKOS_LAMBDA(int x, int y) {
        double r = 0.0;
        for (int q = 0; q<9; q++) {
            r += f_loc(x, y, q);
        }
        rho_loc(x,y) = r;
    });
}

void computeVelocity(Kokkos::View<double***> f, Kokkos::View<double**> rho,
                     Kokkos::View<double***> u){
        auto f_loc = f;
        auto rho_loc = rho;
        auto u_loc = u;
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx,Ny}),
        KOKKOS_LAMBDA(int x, int y) {
        double ux = 0.0;
        double uy = 0.0;
        for (int q = 0; q<9; q++) {
            ux += (f_loc(x, y, q) * cx[q]);
            uy += (f_loc(x, y, q) * cy[q]);
        }
        u_loc(x,y, 0) = ux / rho_loc(x, y);
        u_loc(x,y, 1) = uy / rho_loc(x, y);
    });
}

void streaming(Kokkos::View<double***> f, Kokkos::View<double***> f_new) {
    auto f_loc     = f;
    auto f_new_loc = f_new;

    Kokkos::deep_copy(f_new_loc, 0.0);  // clear stale values

    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{Nx,Ny}),
    KOKKOS_LAMBDA(int x, int y) {
        for (int q = 0; q < 9; q++) {
            int x_new = (x + cx[q] + Nx) % Nx;
            int y_new = (y + cy[q] + Ny) % Ny;
            f_new_loc(x_new, y_new, q) = f_loc(x, y, q);
        }
    });
}

void collision(Kokkos::View<double***> f, Kokkos::View<double**> rho, Kokkos::View<double***> u, double tau) {
    auto f_loc   = f;
    auto rho_loc = rho;
    auto u_loc   = u;
    Kokkos::parallel_for(Kokkos::MDRangePolicy({0,0},{Nx,Ny}),
        KOKKOS_LAMBDA(int x, int y) {
        double ux = u_loc(x, y, 0);
        double uy = u_loc(x, y, 1);
        double rho  = rho_loc(x, y);
        double udotu = ux*ux + uy*uy;
        for (int q = 0; q<9; q++) {
            double cu   = cx[q]*ux + cy[q]*uy;
            double f_eq = w[q] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*udotu);
            f_loc(x, y, q) += -(f_loc(x, y, q) - f_eq) / tau;
        }
    });
}

void writeOutput(Kokkos::View<double**> rho, Kokkos::View<double***> u,
                 int Nx, int Ny, int step) {

    // copy to host first
    auto rho_host = Kokkos::create_mirror_view(rho);
    auto u_host   = Kokkos::create_mirror_view(u);
    Kokkos::deep_copy(rho_host, rho);
    Kokkos::deep_copy(u_host, u);

    // filename includes step number
    std::string filename = "/home/mdfauser/Computer Science/HPC GPU Parallelization/yalb/src/output/output_" + std::to_string(step) + ".csv";
    std::ofstream file(filename);

    file << "x,y,rho,ux,uy\n";  // header
    for (int x = 0; x < Nx; x++) {
        for (int y = 0; y < Ny; y++) {
            file << x << ","
                 << y << ","
                 << rho_host(x, y) << ","
                 << u_host(x, y, 0) << ","
                 << u_host(x, y, 1) << "\n";
        }
    }
}

void initialize(Kokkos::View<double***> f, Kokkos::View<double**> rho,
                Kokkos::View<double***> u) {
    auto f_loc   = f;
    auto rho_loc = rho;
    auto u_loc   = u;

    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{Nx,Ny}),
    KOKKOS_LAMBDA(int x, int y) {

        // uniform density with a small bump at the center
        double r = 1.0;
        if (x == Nx/4 && y == Ny/2) r = 1.1;  // 10% higher at 1/4
        if (x == 3*Nx/4 && y == Ny/2) r = 0.9;

        // zero velocity everywhere
        u_loc(x, y, 0) = 0.0;
        u_loc(x, y, 1) = 0.0;

        // initialize f to equilibrium for this rho and u
        // since u=0, the formula simplifies a lot: cu=0, udotu=0
        for (int q = 0; q < 9; q++) {
            f_loc(x, y, q) = w[q] * r;  // simplified f_eq at rest
        }
        rho_loc(x, y) = r;
    });
}

void initializeShearWave(Kokkos::View<double***> f, Kokkos::View<double**> rho,
                         Kokkos::View<double***> u, double u0) {
    auto f_loc   = f;
    auto rho_loc = rho;
    auto u_loc   = u;

    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{Nx,Ny}),
    KOKKOS_LAMBDA(int x, int y) {
        double r  = 1.0;
        double ux = u0 * Kokkos::sin(2.0 * M_PI * y / Ny);
        double uy = 0.0;

        u_loc(x, y, 0) = ux;
        u_loc(x, y, 1) = uy;
        rho_loc(x, y)  = r;

        // full f_eq since u != 0 here
        double udotu = ux * ux + uy * uy;
        for (int q = 0; q < 9; q++) {
            double cu = cx[q]*ux + cy[q]*uy;
            f_loc(x, y, q) = w[q] * r * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*udotu);
        }
    });
}

int main(int argc, char** argv){
    const int N_steps = 1000;
    const double tau = 0.8;
    const double u0 = 0.07;
    Kokkos::initialize(argc, argv);
    {
        // distribution function
        Kokkos::View<double***>f("f", Nx, Ny, 9);
        // for streaming
        Kokkos::View<double***>f_new("f_new", Nx, Ny, 9);
        // equilibrium
        Kokkos::View<double***>f_eq("f_eq", Nx, Ny, 9);
        // density
        Kokkos::View<double**>rho("rho", Nx, Ny);
        // velocity
        Kokkos::View<double***>u("u", Nx, Ny, 2);

        initializeShearWave(f, rho, u, u0);

        // checking the total mass is conserved
        double initial_mass = 0.0;
        auto f_loc = f;
        Kokkos::parallel_reduce(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{Nx,Ny}),
       KOKKOS_LAMBDA(int x, int y, double& mass) {
           for (int q = 0; q < 9; q++) {
               mass += f_loc(x, y, q);
           }
        }, initial_mass);
        std::cout << std::setprecision(15) << "Initial mass: " << initial_mass << std::endl;

        for (int step = 0; step <= N_steps; step++) {
            computeDensity(f, rho);
            computeVelocity(f, rho, u);
            // checking for the momentum conservation
            if (step % 1000 == 0) {
                auto f_loc = f;

                // compute momentum BEFORE collision
                double mom_x_pre = 0.0, mom_y_pre = 0.0;
                Kokkos::parallel_reduce(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{Nx,Ny}),
                KOKKOS_LAMBDA(int x, int y, double& mx, double& my) {
                    for (int q = 0; q < 9; q++) {
                        mx += f_loc(x, y, q) * cx[q];
                        my += f_loc(x, y, q) * cy[q];
                    }
                }, mom_x_pre, mom_y_pre);

                collision(f, rho, u, tau);

                // compute momentum AFTER collision
                double mom_x_post = 0.0, mom_y_post = 0.0;
                Kokkos::parallel_reduce(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{Nx,Ny}),
                KOKKOS_LAMBDA(int x, int y, double& mx, double& my) {
                    for (int q = 0; q < 9; q++) {
                        mx += f_loc(x, y, q) * cx[q];
                        my += f_loc(x, y, q) * cy[q];
                    }
                }, mom_x_post, mom_y_post);

                std::cout << std::setprecision(15)
                          << "step " << step
                          << " px: " << mom_x_pre << " -> " << mom_x_post
                          << " diff: " << std::abs(mom_x_post - mom_x_pre)
                          << " py: " << mom_y_pre << " -> " << mom_y_post
                          << " diff: " << std::abs(mom_y_post - mom_y_pre)
                          << std::endl;
            } else {
                collision(f, rho, u, tau);
            }
            streaming(f, f_new);
            // swapping
            auto temp = f;
            f = f_new;
            f_new = temp;

            if (step % 10 == 0) writeOutput(rho,u,Nx,Ny,step);


            // checking the total mass
            if (step % 1000 == 0) {
                double current_mass = 0.0;
                auto f_loc = f;
                Kokkos::parallel_reduce(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{Nx,Ny}),
       KOKKOS_LAMBDA(int x, int y, double& mass) {
               for (int q = 0; q < 9; q++) {
                    mass += f_loc(x, y, q);
                   }
                }, current_mass);
                std::cout << std::setprecision(15) << "Current mass: " << current_mass << std::endl;
            }
        }


    }
    Kokkos::finalize();
    return 0;
}
