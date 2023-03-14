#pragma once

#include <cmath>
#include <cstdint>
#include <functional>

namespace autodiff {

template <typename T> struct Jet {
  T x;
  T x_prime;
};

using Jetd = Jet<double>;

constexpr Jetd operator+(const Jetd &jet, double increment) {
  return Jetd{jet.x + increment, jet.x_prime};
}

constexpr Jetd operator+(const Jetd &left, const Jetd &right) {
  return {left.x + right.x, left.x_prime + right.x_prime};
}

constexpr Jetd operator+(double increment, const Jetd &jet) {
  return Jetd{jet.x + increment, jet.x_prime};
}

constexpr void operator+=(Jetd &jet, double increment) { jet.x += increment; }

constexpr Jetd operator-(const Jetd &jet, double x) {
  return Jetd{jet.x - x, jet.x_prime};
}

constexpr Jetd operator-(double temp, const Jetd &jet) {
  return Jetd{temp + jet.x, jet.x_prime};
}

constexpr Jetd operator-(const Jetd &left, const Jetd &right) {
  return Jetd{left.x - right.x, left.x_prime - right.x_prime};
}

constexpr Jetd operator-(const Jetd &jet) { return Jetd{-jet.x, -jet.x_prime}; }

constexpr void operator-=(Jetd &x, double y) { x.x -= y; }

constexpr Jetd operator*(double factor, const Jetd &jet) {
  return Jetd{factor * jet.x, factor * jet.x_prime};
}

constexpr Jetd operator*(const Jetd &jet, double factor) {
  return Jetd{jet.x * factor, jet.x_prime * factor};
}

constexpr Jetd operator*(const Jetd &left, const Jetd &right) {
  return Jetd{left.x * right.x,
              left.x * right.x_prime + left.x_prime * right.x};
}

constexpr void operator*=(Jetd &x, double y) {
  x.x *= y;
  x.x_prime *= y;
}

constexpr Jetd operator/(const Jetd &x, double y) {
  return Jetd{x.x / y, x.x_prime / y};
}

constexpr Jetd operator/(double x, const Jetd &y) {
  return Jetd{x / y.x, x / y.x_prime};
}

constexpr Jetd operator/(const Jetd &high, const Jetd &low) {
  return Jetd{high.x / low.x,
              (low.x * high.x_prime - high.x * low.x_prime) / (low.x * low.x)};
}

constexpr void operator/=(Jetd &x, double y) {
  x.x /= y;
  x.x_prime /= y;
}

constexpr Jetd exp(const Jetd &jet) {
  return Jetd{std::exp(jet.x), std::exp(jet.x) * jet.x_prime};
}

constexpr Jetd log(const Jetd &jet) {
  return Jetd{std::log(jet.x), 1 / jet.x * jet.x_prime};
}

constexpr Jetd sin(const Jetd &jet) {
  return Jetd{std::sin(jet.x), std::cos(jet.x) * jet.x_prime};
}

constexpr Jetd cos(const Jetd &jet) {
  return Jetd{std::cos(jet.x), -std::sin(jet.x) * jet.x_prime};
}

constexpr Jetd tan(const Jetd &jet) {
  const auto cosX = std::cos(jet.x);
  return Jetd{std::tan(jet.x), 1.0 / (cosX * cosX) * jet.x_prime};
}

constexpr Jetd pow(const Jetd &base, double exponent) {
  return Jetd{std::pow(base.x, exponent),
              exponent * std::pow(base.x, exponent - 1.0) * base.x_prime};
}

} // namespace autodiff

namespace root_finding {

struct NewtonParams {
  uint64_t maximumIterations;
  double absoluteTolerance;
};

struct OptimizationResult {
  uint64_t numIterations;
  double x;
  double y;
};

constexpr OptimizationResult
newton(const std::function<autodiff::Jetd(const autodiff::Jetd &)> &f,
       const double &x0, const NewtonParams &params) {
  autodiff::Jetd jet{x0, 1.0};
  auto value = f(jet);
  OptimizationResult result;
  result.numIterations = 0;
  while (result.numIterations < params.maximumIterations &&
         std::abs(value.x) > params.absoluteTolerance) {
    jet.x -= value.x / value.x_prime;
    value = f(jet);
    ++result.numIterations;
  }
  result.x = jet.x;
  result.y = value.x;
  return result;
}

} // namespace root_finding
