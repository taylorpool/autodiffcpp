#include "autodiff/autodiff.hpp"

#include <cstdint>
#include <gtest/gtest.h>

#include <cmath>

template <typename T> T add1(const T &x) { return x + 1; }

using autodiff::Jetd;

TEST(ForwardDiff, Add1) {
  Jetd jet{1, 1};
  Jetd result = add1(jet);
  ASSERT_NEAR(result.x_prime, 1.0, 1e-6);
  ASSERT_NEAR(result.x, jet.x + 1, 1e-6);
}

template <typename T> T add2(const T &x) { return x + 2; }

TEST(ForwardDiff, Add2) {
  Jetd jet{1, 1};
  Jetd result = add2(jet);
  ASSERT_NEAR(result.x_prime, 1.0, 1e-6);
  ASSERT_NEAR(result.x, jet.x + 2, 1e-6);
}

TEST(ForwardDiff, Subtract1) {
  Jetd jet{1, 1};
  double subtraction = 1;
  Jetd result = jet - subtraction;
  ASSERT_NEAR(result.x, jet.x - subtraction, 1e-6);
  ASSERT_NEAR(result.x_prime, 1, 1e-6);
}

TEST(ForwardDiff, Negate) {
  Jetd jet{1, 1};
  Jetd result = -jet;
  ASSERT_NEAR(result.x, -jet.x, 1e-6);
  ASSERT_NEAR(result.x_prime, -1, 1e-6);
}

TEST(ForwardDiff, MultiplyPositive) {
  Jetd jet{1, 1};
  double factor = 3;
  Jetd result = factor * jet;
  ASSERT_NEAR(result.x, factor * jet.x, 1e-6);
  ASSERT_NEAR(result.x_prime, factor, 1e-6);
}

TEST(ForwardDiff, MultiplyNegative) {
  Jetd jet{1, 1};
  double factor = -3;
  Jetd result = factor * jet;
  ASSERT_NEAR(result.x, factor * jet.x, 1e-6);
  ASSERT_NEAR(result.x_prime, factor, 1e-6);
}

TEST(ForwardDiff, AddToItself) {
  Jetd jet{1, 1};
  Jetd result = jet + jet;
  ASSERT_NEAR(result.x, jet.x + jet.x, 1e-6);
  ASSERT_NEAR(result.x_prime, jet.x_prime + jet.x_prime, 1e-6);
}

TEST(ForwardDiff, SubtractFromItself) {
  Jetd jet1{1, 1};
  Jetd jet2{2, 2};
  Jetd result = jet1 - jet2;
  ASSERT_NEAR(result.x, jet1.x - jet2.x, 1e-6);
  ASSERT_NEAR(result.x_prime, jet1.x_prime - jet2.x_prime, 1e-6);
}

TEST(ForwardDiff, MultiplyByJetd) {
  Jetd jet1{1, 2};
  Jetd jet2{3, 4};
  Jetd result = jet1 * jet2;
  ASSERT_NEAR(result.x, jet1.x * jet2.x, 1e-6);
  ASSERT_NEAR(result.x_prime, jet1.x * jet2.x_prime + jet1.x_prime * jet2.x,
              1e-6);
}

TEST(ForwardDiff, DivideByJetd) {
  Jetd jet1{1, 2};
  Jetd jet2{3, 4};
  Jetd result = jet1 / jet2;
  ASSERT_NEAR(result.x, jet1.x / jet2.x, 1e-6);
  ASSERT_NEAR(result.x_prime,
              (jet2.x * jet1.x_prime - jet1.x * jet2.x_prime) /
                  (jet2.x * jet2.x),
              1e-6);
}

TEST(ForwardDiff, Exponential) {
  Jetd jet{1, 1};
  Jetd result = exp(jet);
  ASSERT_NEAR(result.x, std::exp(jet.x), 1e-6);
  ASSERT_NEAR(result.x_prime, std::exp(jet.x), 1e-6);
}

TEST(ForwardDiff, NaturalLogarithm) {
  Jetd jet{1, 1};
  Jetd result = log(jet);
  ASSERT_NEAR(result.x, std::log(jet.x), 1e-6);
  ASSERT_NEAR(result.x_prime, 1.0 / jet.x, 1e-6);
}

TEST(ForwardDiff, Sin0) {
  Jetd jet{0, 1};
  auto result = sin(jet);
  ASSERT_NEAR(result.x, std::sin(jet.x), 1e-6);
  ASSERT_NEAR(result.x_prime, std::cos(jet.x), 1e-6);
}

TEST(ForwardDiff, SinPI) {
  Jetd jet{M_PI, 1};
  auto result = sin(jet);
  ASSERT_NEAR(result.x, std::sin(jet.x), 1e-6);
  ASSERT_NEAR(result.x_prime, std::cos(jet.x), 1e-6);
}

TEST(ForwardDiff, Cos0) {
  Jetd jet{0, 1};
  auto result = cos(jet);
  ASSERT_NEAR(result.x, std::cos(jet.x), 1e-6);
  ASSERT_NEAR(result.x_prime, -std::sin(jet.x), 1e-6);
}

TEST(ForwardDiff, CosPI) {
  Jetd jet{M_PI, 1};
  auto result = cos(jet);
  ASSERT_NEAR(result.x, std::cos(jet.x), 1e-6);
  ASSERT_NEAR(result.x_prime, -std::sin(jet.x), 1e-6);
}

TEST(ForwardDiff, Tan0) {
  Jetd jet{0, 1};
  auto result = tan(jet);
  ASSERT_NEAR(result.x, std::tan(jet.x), 1e-6);
  ASSERT_NEAR(result.x_prime, 1 / std::cos(jet.x) / std::cos(jet.x), 1e-6);
}

TEST(ForwardDiff, TanPI) {
  Jetd jet{M_PI, 1};
  auto result = tan(jet);
  ASSERT_NEAR(result.x, std::tan(jet.x), 1e-6);
  ASSERT_NEAR(result.x_prime, 1 / std::cos(jet.x) / std::cos(jet.x), 1e-6);
}

TEST(ForwardDiff, Chained1) {
  Jetd jet{1, 1};
  Jetd result = 2 * jet + 1;
  ASSERT_NEAR(result.x, 2 * jet.x + 1, 1e-6);
  ASSERT_NEAR(result.x_prime, 2, 1e-6);
}

TEST(ForwardDiff, Chained2) {
  Jetd jet{2, 1};
  Jetd result = exp(2 * jet);
  ASSERT_NEAR(result.x, std::exp(2 * jet.x), 1e-6);
  ASSERT_NEAR(result.x_prime, 2 * std::exp(2 * jet.x), 1e-6);
}

TEST(ForwardDiff, Chained3) {
  Jetd x1{2, 1};
  Jetd x2{5, 0};
  Jetd result = log(x1) + x1 * x2 - sin(x2);
  ASSERT_NEAR(result.x, std::log(x1.x) + x1.x * x2.x - std::sin(x2.x), 1e-3);
  ASSERT_NEAR(result.x_prime, 1.0 / x1.x + 1.0 * x2.x, 1e-2);
}

TEST(ForwardDiff, Chained4) {
  Jetd x1{2, 0};
  Jetd x2{5, 1};
  Jetd result = log(x1) + x1 * x2 - sin(x2);
  ASSERT_NEAR(result.x, std::log(x1.x) + x1.x * x2.x - std::sin(x2.x), 1e-3);
  ASSERT_NEAR(result.x_prime, x1.x * 1.0 - std::cos(x2.x), 1e-2);
}

TEST(ForwardDiff, NewtonX_PowerOf2) {
  const auto costFunction = [](const Jetd &val) { return pow(val, 2); };
  autodiff::NewtonParams params;
  params.maximumIterations = 50;
  params.tolerance = 1e-14;
  const double x0 = 10.0;
  const auto result = autodiff::newton<double>(costFunction, x0, params);
  ASSERT_NEAR(result.x, 0.0, 1e-4);
  ASSERT_NEAR(result.y, 0.0, 1e-4);
}

TEST(ForwardDiff, NewtonPowerOf4) {
  const auto costFunction = [](const Jetd &val) { return pow(val, 4); };
  autodiff::NewtonParams params;
  params.maximumIterations = 50;
  params.tolerance = 1e-14;
  const double x0 = 10.0;
  const auto result = autodiff::newton<double>(costFunction, x0, params);
  ASSERT_NEAR(result.x, 0.0, 1e-3);
  ASSERT_NEAR(result.y, 0.0, 1e-3);
}

Jetd my_cost_function(const Jetd &x) { return x * x; }

TEST(Newton, x_Squared) {
  const double x0 = 10.0;
  autodiff::NewtonParams params;
  params.maximumIterations = 50;
  params.tolerance = 1e-14;
  auto result = autodiff::newton<double>(my_cost_function, x0, params);
  ASSERT_NEAR(result.x, 0.0, 1e-4);
  ASSERT_NEAR(result.y, 0.0, 1e-4);
}
