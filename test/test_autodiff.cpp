#include <cstdint>
#include <gtest/gtest.h>

#include <cmath>

template <typename T> T add1(const T &x) { return x + 1; }

template <typename T> struct Jet {
  T x;
  T x_prime;
};

using Jetd = Jet<double>;

Jetd operator+(const Jetd &jet, double increment) {
  return Jetd{jet.x + increment, jet.x_prime};
}

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

Jetd operator-(const Jetd &jet, double x) {
  return Jetd{jet.x - x, jet.x_prime};
}

TEST(ForwardDiff, Subtract1) {
  Jetd jet{1, 1};
  double subtraction = 1;
  Jetd result = jet - subtraction;
  ASSERT_NEAR(result.x, jet.x - subtraction, 1e-6);
  ASSERT_NEAR(result.x_prime, 1, 1e-6);
}

Jetd operator-(const Jetd &jet) { return Jetd{-jet.x, -jet.x_prime}; }

TEST(ForwardDiff, Negate) {
  Jetd jet{1, 1};
  Jetd result = -jet;
  ASSERT_NEAR(result.x, -jet.x, 1e-6);
  ASSERT_NEAR(result.x_prime, -1, 1e-6);
}

Jetd operator*(double factor, const Jetd &jet) {
  return Jetd{factor * jet.x, factor * jet.x_prime};
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

Jetd operator+(const Jetd &left, const Jetd &right) {
  return {left.x + right.x, left.x_prime + right.x_prime};
}

TEST(ForwardDiff, AddToItself) {
  Jetd jet{1, 1};
  Jetd result = jet + jet;
  ASSERT_NEAR(result.x, jet.x + jet.x, 1e-6);
  ASSERT_NEAR(result.x_prime, jet.x_prime + jet.x_prime, 1e-6);
}

Jetd operator-(const Jetd &left, const Jetd &right) {
  return Jetd{left.x - right.x, left.x_prime - right.x_prime};
}

TEST(ForwardDiff, SubtractFromItself) {
  Jetd jet1{1, 1};
  Jetd jet2{2, 2};
  Jetd result = jet1 - jet2;
  ASSERT_NEAR(result.x, jet1.x - jet2.x, 1e-6);
  ASSERT_NEAR(result.x_prime, jet1.x_prime - jet2.x_prime, 1e-6);
}

Jetd operator*(const Jetd &left, const Jetd &right) {
  return Jetd{left.x * right.x,
              left.x * right.x_prime + left.x_prime * right.x};
}

TEST(ForwardDiff, MultiplyByJetd) {
  Jetd jet1{1, 2};
  Jetd jet2{3, 4};
  Jetd result = jet1 * jet2;
  ASSERT_NEAR(result.x, jet1.x * jet2.x, 1e-6);
  ASSERT_NEAR(result.x_prime, jet1.x * jet2.x_prime + jet1.x_prime * jet2.x,
              1e-6);
}

Jetd operator/(const Jetd &high, const Jetd &low) {
  return Jetd{high.x / low.x,
              (low.x * high.x_prime - high.x * low.x_prime) / (low.x * low.x)};
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

Jetd exp(const Jetd &jet) {
  return Jetd{std::exp(jet.x), std::exp(jet.x) * jet.x_prime};
}

TEST(ForwardDiff, Exponential) {
  Jetd jet{1, 1};
  Jetd result = exp(jet);
  ASSERT_NEAR(result.x, std::exp(jet.x), 1e-6);
  ASSERT_NEAR(result.x_prime, std::exp(jet.x), 1e-6);
}

Jetd log(const Jetd &jet) {
  return Jetd{std::log(jet.x), 1 / jet.x * jet.x_prime};
}

TEST(ForwardDiff, NaturalLogarithm) {
  Jetd jet{1, 1};
  Jetd result = log(jet);
  ASSERT_NEAR(result.x, std::log(jet.x), 1e-6);
  ASSERT_NEAR(result.x_prime, 1.0 / jet.x, 1e-6);
}

Jetd sin(const Jetd &jet) {
  return Jetd{std::sin(jet.x), std::cos(jet.x) * jet.x_prime};
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

Jetd cos(const Jetd &jet) {
  return Jetd{std::cos(jet.x), -std::sin(jet.x) * jet.x_prime};
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

Jetd tan(const Jetd &jet) {
  auto cosX = std::cos(jet.x);
  return Jetd{std::tan(jet.x), 1.0 / (cosX * cosX) * jet.x_prime};
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

Jetd pow(const Jetd &base, double exponent) {
  return Jetd{std::pow(base.x, exponent),
              exponent * std::pow(base.x, exponent - 1) * base.x_prime};
}

Jetd operator^(const Jetd &base, double exponent) {
  return Jetd{std::pow(base.x, exponent),
              exponent * std::pow(base.x, exponent - 1) * base.x_prime};
}

void operator-=(Jetd &x, double y) { x.x -= y; }

TEST(ForwardDiff, GradientDescentX2) {
  Jetd x{10.0, 1.0};
  auto costFunction = [](const Jetd &val) { return pow(val, 2); };
  auto cost = costFunction(x);
  for (int index = 0; index < 10; ++index) {
    x -= 0.5 * cost.x_prime;
    cost = costFunction(x);
    std::cout << "iteration: " << index << " x: " << x.x << " cost: " << cost.x
              << std::endl;
  }
}

TEST(ForwardDiff, GradientDescentX4) {
  Jetd x{10.0, 1.0};
  auto costFunction = [](const Jetd &val) { return pow(val, 4); };
  auto cost = costFunction(x);
  std::cout << "iteration: " << 0 << " x: " << x.x << " cost: " << cost.x
            << " gradient: " << cost.x_prime << std::endl;
  for (int index = 0; index < 10; ++index) {
    x -= 1e-3 * cost.x_prime;
    cost = costFunction(x);
    std::cout << "iteration: " << 0 << " x: " << x.x << " cost: " << cost.x
              << " gradient: " << cost.x_prime << std::endl;
  }
}

template <uint64_t N> using JetVec = Jet<double[N]>;

template <uint64_t N> JetVec<N> operator+(const JetVec<N> &x, double y) {
  JetVec<N> jet;
  for (uint64_t index = 0; index < N; ++index) {
    jet.x[index] = x.x[index] + y;
    jet.x_prime[index] = x.x_prime[index];
  }
  return jet;
}

TEST(ForwardDiff, JetVecAddScalar) {
  JetVec<2> vec;
  vec.x[0] = 0.0;
  vec.x[1] = 1.0;
  vec.x_prime[0] = 1.0;
  vec.x_prime[1] = 1.0;

  double value = 2.0;

  auto result = vec + value;
  ASSERT_NEAR(result.x[0], vec.x[0] + value, 1e-6);
  ASSERT_NEAR(result.x[1], vec.x[1] + value, 1e-6);
  ASSERT_NEAR(result.x_prime[0], vec.x_prime[0], 1e-6);
  ASSERT_NEAR(result.x_prime[1], vec.x_prime[1], 1e-6);
}

template <uint64_t N>
JetVec<N> operator+(const JetVec<N> &x, const JetVec<N> &y) {
  JetVec<N> jet;
  for (uint64_t index = 0; index < N; ++index) {
    jet.x[index] = x.x[index] + y.x[index];
    jet.x_prime[index] = x.x_prime[index] + y.x_prime[index];
  }
  return jet;
}

TEST(ForwardDiff, JetVecAddJetVec) {
  JetVec<2> a;
  JetVec<2> b;
  a.x[0] = 0.0;
  a.x[1] = 1.0;
  a.x_prime[0] = 1.0;
  a.x_prime[1] = 1.0;
  b.x[0] = 2.0;
  b.x[1] = 3.0;
  b.x_prime[0] = 1.0;
  b.x_prime[1] = 1.0;

  auto result = a + b;
  ASSERT_NEAR(result.x[0], a.x[0] + b.x[0], 1e-6);
  ASSERT_NEAR(result.x[1], a.x[1] + b.x[1], 1e-6);
  ASSERT_NEAR(result.x_prime[0], a.x_prime[0] + b.x_prime[0], 1e-6);
  ASSERT_NEAR(result.x_prime[1], a.x_prime[1] + b.x_prime[1], 1e-6);
}

template <uint64_t N>
JetVec<N> operator*(const JetVec<N> &x, const JetVec<N> &y) {
  JetVec<N> result;
  for (uint64_t index = 0; index < N; ++index) {
    result.x[index] = x.x[index] * y.x[index];
    result.x_prime[index] =
        x.x_prime[index] * y.x[index] + x.x[index] * y.x_prime[index];
  }
  return result;
}

TEST(ForwardDiff, JetVecMultiplyJetVec) {
  JetVec<2> a;
  JetVec<2> b;
  a.x[0] = 0.0;
  a.x[1] = 1.0;
  a.x_prime[0] = 1.0;
  a.x_prime[1] = 1.0;
  b.x[0] = 2.0;
  b.x[1] = 3.0;
  b.x_prime[0] = 1.0;
  b.x_prime[1] = 1.0;

  auto result = a * b;
  ASSERT_NEAR(result.x[0], a.x[0] * b.x[0], 1e-6);
  ASSERT_NEAR(result.x[1], a.x[1] * b.x[1], 1e-6);
  ASSERT_NEAR(result.x_prime[0], a.x_prime[0] * b.x[0] + a.x[0] * b.x_prime[0],
              1e-6);
  ASSERT_NEAR(result.x_prime[1], a.x_prime[1] * b.x[1] + a.x[1] * b.x_prime[1],
              1e-6);
}
