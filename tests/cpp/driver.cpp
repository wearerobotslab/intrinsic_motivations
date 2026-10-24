// Test driver for cpp/homeokinesis/eigen. Reads a job from stdin, prints numbers to stdout.
//
// Input (whitespace separated):
//   mode ns nm
//   nconf  {key value} * nconf          (config overrides, see apply_conf)
//   mode == "update": T, then T*ns sensor values  -> prints T*nm motor values (Homeokinesis::update)
//   mode == "sox":    T, then T*ns sensor values  -> same, but drives step()/learn() in the
//                                                    order of the original Sox::step
//   mode == "learn":  t, A, C, S, b, h, v_avg, x_buffer, y_buffer (row-major)
//                     -> calls learn() once, prints A C S b h v_avg
//
// Private members are opened up so the tests can set and read the state.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/QR>
#define private public
#include "homeokinesis_controller.hpp"
#undef private

using are::hk::Homeokinesis;
using Eigen::MatrixXd;

static void read_mat(MatrixXd& M, int r, int c) {
    M.resize(r, c);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) std::cin >> M(i, j);
}

static void print_mat(const MatrixXd& M) {
    for (int i = 0; i < M.rows(); i++)
        for (int j = 0; j < M.cols(); j++) std::cout << M(i, j) << ' ';
    std::cout << '\n';
}

static void apply_conf(Homeokinesis& hk, const std::string& k, double v) {
    if (k == "epsA") hk._conf.epsA = v;
    else if (k == "epsC") hk._conf.epsC = v;
    else if (k == "initFeedbackStrength") hk._conf.initFeedbackStrength = v;
    else if (k == "useExtendedModel") hk._conf.useExtendedModel = v != 0;
    else if (k == "steps4Averaging") hk._conf.steps4Averaging = static_cast<size_t>(v);
    else if (k == "steps4Delay") hk._conf.steps4Delay = static_cast<size_t>(v);
    else if (k == "creativity") hk.creativity = v;
    else if (k == "harmony") hk.harmony = v;
    else if (k == "causeaware") hk.causeaware = v;
    else if (k == "sense") hk.sense = v;
    else if (k == "damping") hk.damping = v;
    else if (k == "loga") hk.loga = v != 0;
    else { std::cerr << "unknown conf key " << k << '\n'; std::exit(2); }
}

int main() {
    std::cout << std::setprecision(17);
    std::string mode;
    int ns, nm, nconf;
    std::cin >> mode >> ns >> nm >> nconf;
    Homeokinesis hk;
    for (int i = 0; i < nconf; i++) {
        std::string k; double v;
        std::cin >> k >> v;
        apply_conf(hk, k, v);
    }
    hk.init(ns, nm);
    const int bs = static_cast<int>(hk._conf.buffersize);

    if (mode == "update" || mode == "sox") {
        int T; std::cin >> T;
        std::vector<double> x(static_cast<size_t>(ns));
        for (int s = 0; s < T; s++) {
            for (auto& e : x) std::cin >> e;
            std::vector<double> y;
            if (mode == "update") {
                y = hk.update(x);
            } else {
                MatrixXd xm, ym;
                are::misc::stdvect_to_eigenmat(x, xm);
                hk.step(xm, ym);               // stepNoLearning (t++)
                if (hk.t > bs) {               // Sox::step: no learning while the buffer fills
                    hk.t--; hk.learn(); hk.t++;
                }
                are::misc::eigenvect_to_stdvect(ym, y);
            }
            for (double e : y) std::cout << e << ' ';
            std::cout << '\n';
        }
    } else if (mode == "learn") {
        std::cin >> hk.t;
        read_mat(hk.A, ns, nm); read_mat(hk.C, nm, ns); read_mat(hk.S, ns, ns);
        read_mat(hk.b, ns, 1); read_mat(hk.h, nm, 1); read_mat(hk.v_avg, ns, 1);
        for (int k = 0; k < bs; k++) read_mat(hk.x_buffer[k], ns, 1);
        for (int k = 0; k < bs; k++) read_mat(hk.y_buffer[k], nm, 1);
        hk.learn();
        print_mat(hk.A); print_mat(hk.C); print_mat(hk.S);
        print_mat(hk.b); print_mat(hk.h); print_mat(hk.v_avg);
    } else if (mode == "noise") {
        // add_noise(e): prints e, then max |change| of A S C b h
        double e; std::cin >> e;
        MatrixXd A0 = hk.A, S0 = hk.S, C0 = hk.C, b0 = hk.b, h0 = hk.h;
        hk.add_noise(e);
        std::cout << e << ' ' << (hk.A - A0).cwiseAbs().maxCoeff() << ' ' << (hk.S - S0).cwiseAbs().maxCoeff()
                  << ' ' << (hk.C - C0).cwiseAbs().maxCoeff() << ' ' << (hk.b - b0).cwiseAbs().maxCoeff()
                  << ' ' << (hk.h - h0).cwiseAbs().maxCoeff() << '\n';
    } else if (mode == "clone") {
        // T sensor vectors: run half of them, clone(), feed the rest to both; print both outputs per step
        int T; std::cin >> T;
        std::vector<std::vector<double>> xs(static_cast<size_t>(T), std::vector<double>(static_cast<size_t>(ns)));
        for (auto& x : xs) for (auto& e : x) std::cin >> e;
        for (int s = 0; s < T / 2; s++) hk.update(xs[static_cast<size_t>(s)]);
        are::Control::Ptr copy = hk.clone();
        for (int s = T / 2; s < T; s++) {
            for (double e : hk.update(xs[static_cast<size_t>(s)])) std::cout << e << ' ';
            for (double e : copy->update(xs[static_cast<size_t>(s)])) std::cout << e << ' ';
            std::cout << '\n';
        }
    } else {
        std::cerr << "unknown mode " << mode << '\n';
        return 2;
    }
    return 0;
}
