#include <Kokkos_Core.hpp>
#include <iostream>
#include <iomanip>
// hardcoding
const int cx[9] = { 0, 1,-1, 0, 0, 1,-1,-1, 1}; // for cuda chang the "const" to "constexpr"
const int cy[9] = { 0, 0, 0, 1,-1, 1, 1,-1,-1};
const double w[9] = {4./9, 1./9, 1./9, 1./9, 1./9,
                     1./36,1./36,1./36,1./36};

const int opp[9] = {0, 2, 1, 4, 3, 6, 5, 8, 7};

const int Nx = 15;
const int Ny = 10;

int main(int argc, char** argv){

    Kokkos::initialize(argc, argv);
    {
        // distribution function
        Kokkos::View<double***>f("f", Nx, Ny, 9);
        // for streaming
        Kokkos::View<double***>f_new("f_new", Nx, Ny, 9);
        // density
        Kokkos::View<double**>rho("rho", Nx, Ny);
        // velocity
        Kokkos::View<double***>u("u", Nx, Ny, 2);
    }
    Kokkos::finalize();
    return 0;
}

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
    auto f_loc = f;
    auto f_new_loc = f_new;
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx,Ny}),
        KOKKOS_LAMBDA(int x, int y) {
        for (int q = 0; q<9; q++) {
            int x_new = (x + cx[q] + Nx) % Nx;
            int y_new = (y + cy[q] + Ny) % Ny;
            f_new(x_new, y_new, q) = f(x, y, q);
        }
    });
    auto temp = f_loc;
    f_loc = f_new;
    f_new = temp;
}


// calculating Pi with the Newton Method
int newtonPi(int argc, char** argv, int N) {
    double sum = 0.0;

    Kokkos::parallel_reduce(N,
    KOKKOS_LAMBDA(int i, double& s){
        double sign = (i % 2 == 0) ? 1.0 : -1.0;
        s += sign / (2.0 * i + 1.0);
    }, sum);

    double pi = 4.0 * sum;
    std::cout << std::setprecision(15) << "pi ≈ " << pi << std::endl;
}