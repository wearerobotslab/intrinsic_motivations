// Test driver for homeokinesis/cpp/portable. Reads a job from stdin, prints numbers to stdout.
// Same input format as driver.cpp (the Eigen version), so the Python side can share its helpers:
//
//   mode ns nm
//   nconf  {key value} * nconf                    (config overrides, see apply_conf)
//   mode == "sox" / "sox_float": T, then T*ns sensor values -> prints T*nm motor values (step(), double / float)
//   mode == "learn":  t, A, C, S, b, h, v_avg, x_buffer, y_buffer (row-major)
//                     -> calls learn() once, prints A C S b h v_avg
//   mode == "noise":  e -> add_noise(e), prints e and the largest change of A S C b h
//
// The controller has fixed sizes, so the sizes the tests use are instantiated here: (3, 3) and (4, 2).
// Private members are opened up so the tests can set and read the state.
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#define private public
#include "homeokinesis.hpp"
#undef private

template <typename HK>
static void apply_conf(HK& hk, const std::string& k, double v) {
  auto& p = hk.params();
  using S = typename HK::SensorVec::Scalar;
  if (k == "epsA") p.epsA = S(v);
  else if (k == "epsC") p.epsC = S(v);
  else if (k == "initFeedbackStrength") p.initFeedbackStrength = S(v);
  else if (k == "useExtendedModel") p.useExtendedModel = v != 0;
  else if (k == "steps4Averaging") p.steps4Averaging = static_cast<int>(v);
  else if (k == "steps4Delay") p.steps4Delay = static_cast<int>(v);
  else if (k == "creativity") p.creativity = S(v);
  else if (k == "harmony") p.harmony = S(v);
  else if (k == "causeaware") p.causeaware = S(v);
  else if (k == "sense") p.sense = S(v);
  else if (k == "damping") p.damping = S(v);
  else if (k == "loga") p.loga = v != 0;
  else { std::cerr << "unknown conf key " << k << '\n'; std::exit(2); }
}

template <typename M>
static void read_mat(M& m) {
  for (int i = 0; i < m.rows(); i++)
    for (int j = 0; j < m.cols(); j++) {
      double v; std::cin >> v; m(i, j) = typename M::Scalar(v);
    }
}

template <typename M>
static void print_mat(const M& m) {
  for (int i = 0; i < m.rows(); i++)
    for (int j = 0; j < m.cols(); j++) std::cout << static_cast<double>(m(i, j)) << ' ';
  std::cout << '\n';
}

template <typename Scalar, int NS, int NM>
static int run(const std::string& mode, int nconf) {
  using HK = im::hk::Homeokinesis<Scalar, NS, NM>;
  HK hk;
  for (int i = 0; i < nconf; i++) {
    std::string k; double v;
    std::cin >> k >> v;
    apply_conf(hk, k, v);
  }
  hk.reset();  // initFeedbackStrength is used by reset()
  constexpr int bs = 10;  // the default buffer size, as in the reference

  if (mode == "sox" || mode == "sox_float") {
    int T; std::cin >> T;
    typename HK::SensorVec x;
    for (int s = 0; s < T; s++) {
      read_mat(x);
      print_mat(hk.step(x));
    }
  } else if (mode == "learn") {
    std::cin >> hk.t_;
    read_mat(hk.A_); read_mat(hk.C_); read_mat(hk.S_);
    read_mat(hk.b_); read_mat(hk.h_); read_mat(hk.v_avg_);
    for (int k = 0; k < bs; k++) read_mat(hk.x_buffer_[k]);
    for (int k = 0; k < bs; k++) read_mat(hk.y_buffer_[k]);
    hk.learn();
    print_mat(hk.A_); print_mat(hk.C_); print_mat(hk.S_);
    print_mat(hk.b_); print_mat(hk.h_); print_mat(hk.v_avg_);
  } else if (mode == "noise") {
    double e; std::cin >> e;
    const auto A0 = hk.A_; const auto S0 = hk.S_; const auto C0 = hk.C_; const auto b0 = hk.b_; const auto h0 = hk.h_;
    hk.add_noise(Scalar(e));
    std::cout << e << ' ' << (hk.A_ - A0).cwiseAbs().maxCoeff() << ' ' << (hk.S_ - S0).cwiseAbs().maxCoeff() << ' '
              << (hk.C_ - C0).cwiseAbs().maxCoeff() << ' ' << (hk.b_ - b0).cwiseAbs().maxCoeff() << ' '
              << (hk.h_ - h0).cwiseAbs().maxCoeff() << '\n';
  } else {
    std::cerr << "unknown mode " << mode << '\n';
    return 2;
  }
  return 0;
}

int main() {
  std::cout << std::setprecision(17);
  std::string mode;
  int ns, nm, nconf;
  std::cin >> mode >> ns >> nm >> nconf;
  const bool use_float = mode == "sox_float";
  if (ns == 3 && nm == 3) return use_float ? run<float, 3, 3>(mode, nconf) : run<double, 3, 3>(mode, nconf);
  if (ns == 4 && nm == 2) return use_float ? run<float, 4, 2>(mode, nconf) : run<double, 4, 2>(mode, nconf);
  std::cerr << "sizes " << ns << 'x' << nm << " not instantiated in portable_driver.cpp\n";
  return 2;
}
