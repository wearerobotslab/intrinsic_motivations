// Checks the portable controller against a straightforward reference implementation of the same
// equations (dynamic double matrices, written like cpp/homeokinesis/eigen, called in LpzRobots' order).
//   g++ -std=c++17 -O2 -isystem /usr/include/eigen3 test_homeokinesis.cpp -o test_homeokinesis && ./test_homeokinesis
#include <cmath>
#include <cstdio>
#include <vector>

#include <Eigen/Dense>

#include "homeokinesis.hpp"

namespace ref {
using Matrix = Eigen::MatrixXd;

// Reference: same equations as Homeokinesis::learn() in cpp/homeokinesis/eigen, with the C++ defaults
struct Sox {
  int ns, nm, bs = 10, t = 0;
  Matrix A, S, C, b, h, A_native, C_native, v_avg, x_smooth;
  std::vector<Matrix> xb, yb;
  double epsA = 0.1, epsC = 0.05, causeaware = 0.01, sense = 1, creativity = 0.1, harmony = 0.1,
         damping = 0.00001;
  bool loga = true;
  int s4avg = 10, s4delay = 10;

  Sox(int ns_, int nm_) : ns(ns_), nm(nm_) {
    A = Matrix::Identity(ns, nm);
    S = Matrix::Identity(ns, ns) * 0.05;
    C = Matrix::Identity(nm, ns) * 1.1;
    b = Matrix::Zero(ns, 1);
    h = Matrix::Zero(nm, 1);
    A_native = Matrix::Identity(ns, nm);
    C_native = Matrix::Identity(nm, ns) * 1.2;
    v_avg = Matrix::Zero(ns, 1);
    x_smooth = Matrix::Zero(ns, 1);
    for (int k = 0; k < bs; k++) {
      xb.push_back(Matrix::Zero(ns, 1));
      yb.push_back(Matrix::Zero(nm, 1));
    }
  }

  static double clip(double r, double x) { return std::min(std::max(x, -r), r); }
  static Matrix map_clip(const Matrix& m, double r) {
    Matrix out = m;
    for (int i = 0; i < out.size(); i++) out(i) = clip(r, out(i));
    return out;
  }
  static Matrix cube(const Matrix& m) { return m.array().cube().matrix(); }

  Matrix step(const Matrix& x) {
    int sa = std::min(std::max(s4avg, 1), bs - 1);
    x_smooth += (x - x_smooth) * (1.0 / sa);
    xb[t % bs] = x_smooth;
    Matrix y = (C * (x_smooth + v_avg * creativity) + h).array().tanh().matrix();
    yb[t % bs] = y;
    t++;
    if (t <= bs) return y;
    t--;
    learn();
    t++;
    return y;
  }

  void learn() {
    int sd = std::min(std::max(s4delay, 1), bs - 1);
    const Matrix x = xb[(t - sd + bs) % bs];
    const Matrix y_creat = yb[(t - sd + bs) % bs];
    const Matrix x_fut = xb[t % bs];
    const Matrix xi = x_fut - (A * y_creat + b + S * x);
    const Matrix z = C * x + h;
    const Matrix y = z.array().tanh().matrix();
    Matrix g_prime = (1.0 - y.array().square()).matrix();
    const Matrix L = A * (g_prime.asDiagonal() * C) + S;
    const Matrix eta = A.completeOrthogonalDecomposition().pseudoInverse() * xi;
    const Matrix y_hat = y + eta * causeaware;
    const Matrix Lplus = L.completeOrthogonalDecomposition().pseudoInverse();
    const Matrix v = Lplus * xi;
    const Matrix chi = Lplus.transpose() * v;
    const Matrix mu = (A * g_prime.asDiagonal()).transpose() * chi;
    const Matrix epsrel = ((C * v).asDiagonal() * mu) * (sense * 2);
    const Matrix v_hat = v + x * harmony;
    v_avg += (v - v_avg) * .1;
    double EE = loga ? .1 / (v.squaredNorm() + .001) : 1.0;
    A += map_clip(xi * y_hat.transpose() * epsA, 0.1);
    A += map_clip(cube(A_native - A) * damping, 0.1);
    b += map_clip(xi * epsA + b * -damping, 0.1);
    C += map_clip(((mu * v_hat.transpose()) - (y.asDiagonal() * epsrel) * x.transpose()) * (EE * epsC), 0.1);
    C += map_clip(cube(C_native - C) * damping, 0.05);
    h += map_clip(((mu * harmony) - (y.asDiagonal() * epsrel)) * (EE * epsC), 0.05);
  }
};
}  // namespace ref

constexpr int N = 12;

// A simple "robot": each sensor follows its motor with a lag, with some coupling between neighbours
template <typename V>
V plant(const V& state, const V& y) {
  V next = state * 0.6 + y * 0.35;
  for (int i = 0; i < N; i++) next(i) += 0.05 * y((i + 1) % N);
  return next;
}

template <typename Scalar>
double compare(const char* name, int steps, double tolerance) {
  im::hk::Homeokinesis<Scalar, N, N> hk;
  ref::Sox sox(N, N);
  Eigen::VectorXd state = Eigen::VectorXd::LinSpaced(N, -0.3, 0.3);  // same start for both
  Eigen::Matrix<Scalar, N, 1> state_hk = state.cast<Scalar>();
  double max_diff = 0, max_y = 0;
  for (int t = 0; t < steps; t++) {
    Eigen::VectorXd y_ref = sox.step(state);
    Eigen::Matrix<Scalar, N, 1> y_hk = hk.step(state_hk);
    max_diff = std::max(max_diff, (y_ref - y_hk.template cast<double>()).cwiseAbs().maxCoeff());
    max_y = std::max(max_y, y_ref.cwiseAbs().maxCoeff());
    state = plant(state, y_ref);
    state_hk = plant(state_hk, y_hk);
  }
  double c_diff = (sox.C - hk.C().template cast<double>()).cwiseAbs().maxCoeff();
  bool ok = max_diff <= tolerance && std::isfinite(max_diff);
  std::printf("%-7s %4d steps: max |y - y_ref| = %.3g, max |C - C_ref| = %.3g (largest |y| %.2f)  %s\n", name,
              steps, max_diff, c_diff, max_y, ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

int main() {
  int failures = 0;
  failures += compare<double>("double", 2000, 1e-9);
  failures += compare<float>("float", 200, 1e-3);
  failures += compare<float>("float", 2000, 5e-2);
  std::printf(failures ? "FAILED\n" : "all checks passed\n");
  return failures;
}
