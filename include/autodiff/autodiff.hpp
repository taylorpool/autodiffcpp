#pragma once

#include <cstdint>

namespace autodiff {

template <typename T> struct Jet {
  T x;
  T x_prime;
};

using Jetd = Jet<double>;

constexpr Jetd operator+(const Jetd &jet, double increment) {
  return Jetd{jet.x + increment, jet.x_prime};
}

} // namespace autodiff
