#ifndef IM_HOMEOKINESIS_PORTABLE_HPP
#define IM_HOMEOKINESIS_PORTABLE_HPP

// Homeokinesis: the self-organising "Sox" controller from The Playful Machine (Der and Martius, 2012),
// following the LpzRobots implementation.
//
// Portable version: header-only, depends only on Eigen, fixed-size matrices (no allocation while
// running) and float or double. Suitable for microcontrollers such as the ESP32-S3, where float is
// much faster than double.
//
//   im::hk::Homeokinesis<float, 12, 12> hk;   // 12 sensors, 12 motors
//   hk.params().epsC = 0.05f;
//   y = hk.step(x);                           // act on x, then learn
//
// Order of operations: like LpzRobots' Sox::step, each step stores the new sensor values and computes
// the motor values, then learns from the newest sensor values (x_fut) and the sensors and motors from
// steps4Delay steps earlier. Learning starts once the buffer has been filled.

#include <Eigen/Core>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>

namespace im {
namespace hk {

template <typename Scalar>
struct Params {
  // Learning rates of the forward model (A, with S and b) and of the controller (C and h)
  Scalar epsA = Scalar(0.1);
  Scalar epsC = Scalar(0.05);
  Scalar factorS = Scalar(1);  // learning rate of S relative to epsA
  Scalar factorb = Scalar(1);  // learning rate of b relative to epsA
  Scalar factorh = Scalar(1);  // learning rate of h relative to epsC

  Scalar initFeedbackStrength = Scalar(1.1);  // initial C = identity * this (used by reset())
  bool useExtendedModel = false;              // learn S (forward model also uses the current state)
  int steps4Averaging = 10;                   // sensor smoothing, in steps (1 = none), < buffer size
  int steps4Delay = 10;                       // delay between action and its effect, in steps, < buffer size

  Scalar causeaware = Scalar(0.01);
  Scalar sense = Scalar(1);
  Scalar creativity = Scalar(0.1);
  Scalar harmony = Scalar(0.1);
  Scalar damping = Scalar(0.00001);  // pulls A and C back towards their native values
  bool loga = true;                  // logarithmic error: learning step scaled by 0.1 / (|v|^2 + 0.001)

  // Per-step limits of each matrix element's update
  Scalar clipLarge = Scalar(0.1);   // A, S, b and C updates, and A damping
  Scalar clipSmall = Scalar(0.05);  // C damping and h updates
};

template <typename Scalar, int NS, int NM, int BufferSize = 10>
class Homeokinesis {
 public:
  using SensorVec = Eigen::Matrix<Scalar, NS, 1>;
  using MotorVec = Eigen::Matrix<Scalar, NM, 1>;
  using MatA = Eigen::Matrix<Scalar, NS, NM>;  // forward model: motors -> sensors
  using MatS = Eigen::Matrix<Scalar, NS, NS>;  // extended model: sensors -> sensors
  using MatC = Eigen::Matrix<Scalar, NM, NS>;  // controller: sensors -> motors

  Homeokinesis() { reset(); }
  explicit Homeokinesis(const Params<Scalar>& p) : params_(p) { reset(); }

  Params<Scalar>& params() { return params_; }
  const Params<Scalar>& params() const { return params_; }

  // Back to the initial state: identity models, empty buffers, step counter at 0
  void reset() {
    A_.setIdentity();
    S_ = MatS::Identity() * Scalar(0.05);
    C_ = MatC::Identity() * params_.initFeedbackStrength;
    b_.setZero();
    h_.setZero();
    A_native_.setIdentity();
    C_native_ = MatC::Identity() * Scalar(1.2);
    v_avg_.setZero();
    x_smooth_.setZero();
    for (int k = 0; k < BufferSize; k++) {
      x_buffer_[k].setZero();
      y_buffer_[k].setZero();
    }
    xi_.setZero();
    t_ = 0;
  }

  // Compute the motor values for sensor values x, then learn
  const MotorVec& step(const SensorVec& x) {
    step_no_learning(x);
    if (t_ <= BufferSize) return y_;  // learn once the buffer is full
    t_--;  // learn() uses the step that step_no_learning() has just stored
    if (params_.epsA != Scalar(0) || params_.epsC != Scalar(0)) learn();
    t_++;
    return y_;
  }

  // Compute the motor values for sensor values x, without learning
  const MotorVec& step_no_learning(const SensorVec& x) {
    const int s4avg = std::clamp(params_.steps4Averaging, 1, BufferSize - 1);
    if (s4avg > 1)
      x_smooth_ += (x - x_smooth_) / Scalar(s4avg);
    else
      x_smooth_ = x;
    x_buffer_[t_ % BufferSize] = x_smooth_;
    y_ = (C_ * (x_smooth_ + v_avg_ * params_.creativity) + h_).array().tanh().matrix();
    y_buffer_[t_ % BufferSize] = y_;
    t_++;
    return y_;
  }

  // Add uniform noise in [-e, e] to the models and controller (and in [-1, 1] to the biases)
  void add_noise(Scalar e) {
    A_ += MatA::Random() * e;
    S_ += MatS::Random() * e;
    C_ += MatC::Random() * e;
    b_ += SensorVec::Random();
    h_ += MotorVec::Random();
  }

  const MatA& A() const { return A_; }
  const MatS& S() const { return S_; }
  const MatC& C() const { return C_; }
  const SensorVec& b() const { return b_; }
  const MotorVec& h() const { return h_; }
  const MotorVec& y() const { return y_; }
  // Prediction error of the forward model at the last learning step
  const SensorVec& prediction_error() const { return xi_; }
  long steps() const { return t_; }

  void set_A(const MatA& m) { A_ = m; }
  void set_C(const MatC& m) { C_ = m; }
  void set_h(const MotorVec& v) { h_ = v; }

 private:
  template <typename M>
  static M clipped(const M& m, Scalar limit) {
    return m.cwiseMax(-limit).cwiseMin(limit);
  }

  void learn() {
    const Params<Scalar>& p = params_;
    const int s4delay = std::clamp(p.steps4Delay, 1, BufferSize - 1);
    const int past = (t_ - s4delay + BufferSize) % BufferSize;
    const SensorVec& x = x_buffer_[past];
    const MotorVec& y_creat = y_buffer_[past];
    const SensorVec& x_fut = x_buffer_[t_ % BufferSize];  // future sensor (with respect to x, y)

    xi_ = x_fut - (A_ * y_creat + b_ + S_ * x);

    const MotorVec z = C_ * x + h_;
    const MotorVec y = z.array().tanh().matrix();
    const MotorVec g_prime = (Scalar(1) - y.array().square()).matrix();

    const MatS L = A_ * (g_prime.asDiagonal() * C_) + S_;

    const MotorVec eta = A_.completeOrthogonalDecomposition().pseudoInverse() * xi_;
    const MotorVec y_hat = y + eta * p.causeaware;

    const MatS Lplus = L.completeOrthogonalDecomposition().pseudoInverse();
    const SensorVec v = Lplus * xi_;
    const SensorVec chi = Lplus.transpose() * v;

    const MotorVec mu = (A_ * g_prime.asDiagonal()).transpose() * chi;
    const MotorVec epsrel = ((C_ * v).asDiagonal() * mu) * (p.sense * Scalar(2));

    const SensorVec v_hat = v + x * p.harmony;

    v_avg_ += (v - v_avg_) * Scalar(0.1);

    Scalar EE = Scalar(1);
    if (p.loga) EE = Scalar(0.1) / (v.squaredNorm() + Scalar(0.001));

    if (p.epsA > Scalar(0)) {
      const Scalar epsS = p.epsA * p.factorS;
      const Scalar epsb = p.epsA * p.factorb;
      A_ += clipped<MatA>(xi_ * y_hat.transpose() * p.epsA, p.clipLarge);
      if (p.damping != Scalar(0))
        A_ += clipped<MatA>((A_native_ - A_).array().cube().matrix() * p.damping, p.clipLarge);
      if (p.useExtendedModel)
        S_ += clipped<MatS>(xi_ * x.transpose() * epsS + S_ * (-p.damping * Scalar(10)), p.clipLarge);
      b_ += clipped<SensorVec>(xi_ * epsb + b_ * (-p.damping), p.clipLarge);
    }
    if (p.epsC > Scalar(0)) {
      C_ += clipped<MatC>((mu * v_hat.transpose() - (y.asDiagonal() * epsrel) * x.transpose()) * (EE * p.epsC),
                          p.clipLarge);
      if (p.damping != Scalar(0))
        C_ += clipped<MatC>((C_native_ - C_).array().cube().matrix() * p.damping, p.clipSmall);
      h_ += clipped<MotorVec>((mu * p.harmony - y.asDiagonal() * epsrel) * (EE * p.epsC * p.factorh),
                              p.clipSmall);
    }
  }

  Params<Scalar> params_;

  MatA A_, A_native_;
  MatS S_;
  MatC C_, C_native_;
  SensorVec b_, v_avg_, x_smooth_, xi_;
  MotorVec h_, y_ = MotorVec::Zero();
  SensorVec x_buffer_[BufferSize];
  MotorVec y_buffer_[BufferSize];
  long t_ = 0;
};

}  // namespace hk
}  // namespace im

#endif  // IM_HOMEOKINESIS_PORTABLE_HPP
