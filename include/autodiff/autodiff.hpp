#pragma once

#include "autodiff/forward_mode.hpp"

#include <cmath>
#include <cstdint>
#include <functional>

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
newton(const std::function<autodiff::forward_mode::Jetd(
           const autodiff::forward_mode::Jetd &)> &f,
       const double &x0, const NewtonParams &params) {
  autodiff::forward_mode::Jetd jet{x0, 1.0};
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
