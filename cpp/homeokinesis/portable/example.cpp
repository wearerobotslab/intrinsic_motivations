// Runs the portable homeokinesis controller on a simple simulated system and times a step.
//   g++ -std=c++17 -O2 -isystem /usr/include/eigen3 example.cpp -o example && ./example
#include <chrono>
#include <cstdio>

#include "homeokinesis.hpp"

constexpr int N = 12;  // e.g. the Qutee: 12 joint positions in, 12 joint targets out
using HK = im::hk::Homeokinesis<float, N, N>;

int main() {
  HK hk;
  hk.params().epsA = 0.1f;
  hk.params().epsC = 0.05f;

  // A simple "robot": each sensor follows its motor with a lag, plus coupling to the next motor
  HK::SensorVec x = HK::SensorVec::LinSpaced(-0.3f, 0.3f);
  double total_us = 0;
  const int steps = 3000;
  for (int t = 1; t <= steps; t++) {
    auto start = std::chrono::steady_clock::now();
    const HK::MotorVec& y = hk.step(x);
    total_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();

    HK::SensorVec next = x * 0.6f + y * 0.35f;
    for (int i = 0; i < N; i++) next(i) += 0.05f * y((i + 1) % N);
    x = next;

    if (t % 500 == 0)
      std::printf("step %4d: |y| %.3f  |C| %.3f  |A| %.3f  prediction error %.2e\n", t, y.norm(), hk.C().norm(),
                  hk.A().norm(), hk.prediction_error().norm());
  }
  std::printf("average step time on this computer: %.1f us\n", total_us / steps);
  return 0;
}
