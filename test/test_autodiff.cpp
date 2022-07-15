#include <gtest/gtest.h>

#include <cmath>

template <typename T>
T add1(const T& x)
{
    return x+1;
}

struct Jet
{
    double x;
    double x_prime;
};

Jet operator+(const Jet& jet, double increment)
{
    return Jet {jet.x+increment, jet.x_prime};
}

TEST(ForwardDiff, Add1)
{
    Jet jet {1,1};
    Jet result = add1(jet);
    ASSERT_NEAR(result.x_prime, 1.0, 1e-6);
    ASSERT_NEAR(result.x, jet.x+1, 1e-6);
}

template <typename T>
T add2(const T& x)
{
    return x+2;
}

TEST(ForwardDiff, Add2)
{
    Jet jet {1,1};
    Jet result = add2(jet);
    ASSERT_NEAR(result.x_prime, 1.0, 1e-6);
    ASSERT_NEAR(result.x, jet.x+2, 1e-6);
}

Jet operator-(const Jet& jet, double x)
{
    return Jet {jet.x-x, jet.x_prime};
}

TEST(ForwardDiff, Subtract1)
{
    Jet jet {1,1};
    double subtraction = 1;
    Jet result = jet-subtraction;
    ASSERT_NEAR(result.x, jet.x-subtraction, 1e-6);
    ASSERT_NEAR(result.x_prime, 1, 1e-6);
}

Jet operator-(const Jet& jet)
{
    return Jet {-jet.x, -1};
}

TEST(ForwardDiff, Negate)
{
    Jet jet {1,1};
    Jet result = -jet;
    ASSERT_NEAR(result.x, -jet.x, 1e-6);
    ASSERT_NEAR(result.x_prime, -1, 1e-6);
}

Jet operator*(double factor, const Jet& jet)
{
    return Jet {factor*jet.x, factor};
}

TEST(ForwardDiff, MultiplyPositive)
{
    Jet jet {1,1};
    double factor = 3;
    Jet result = factor*jet;
    ASSERT_NEAR(result.x, factor*jet.x, 1e-6);
    ASSERT_NEAR(result.x_prime, factor, 1e-6);
}

TEST(ForwardDiff, MultiplyNegative)
{
    Jet jet {1,1};
    double factor = -3;
    Jet result = factor*jet;
    ASSERT_NEAR(result.x, factor*jet.x, 1e-6);
    ASSERT_NEAR(result.x_prime, factor, 1e-6);
}

Jet operator+(const Jet& left, const Jet& right)
{
    return {left.x+right.x, left.x_prime+right.x_prime};
}

TEST(ForwardDiff, AddToItself)
{
    Jet jet {1,1};
    Jet result = jet + jet;
    ASSERT_NEAR(result.x, jet.x+jet.x, 1e-6);
    ASSERT_NEAR(result.x_prime, jet.x_prime+jet.x_prime, 1e-6);
}

Jet operator-(const Jet& left, const Jet& right)
{
    return Jet {left.x-right.x, left.x_prime-right.x_prime};
}

TEST(ForwardDiff, SubtractFromItself)
{
    Jet jet1 {1,1};
    Jet jet2 {2,2};
    Jet result = jet1 - jet2;
    ASSERT_NEAR(result.x, jet1.x-jet2.x, 1e-6);
    ASSERT_NEAR(result.x_prime, jet1.x_prime-jet2.x_prime, 1e-6);
}

Jet operator*(const Jet& left, const Jet& right)
{
    return Jet {left.x*right.x, left.x*right.x_prime+left.x_prime*right.x};
}

TEST(ForwardDiff, MultiplyByJet)
{
    Jet jet1 {1,2};
    Jet jet2 {3,4};
    Jet result = jet1*jet2;
    ASSERT_NEAR(result.x, jet1.x*jet2.x, 1e-6);
    ASSERT_NEAR(result.x_prime, jet1.x*jet2.x_prime+jet1.x_prime*jet2.x, 1e-6);
}

Jet operator/(const Jet& high, const Jet& low)
{
    return Jet {
        high.x/low.x,
        (low.x*high.x_prime-high.x*low.x_prime)/(low.x*low.x)
    };
}
TEST(ForwardDiff, DivideByJet)
{
    Jet jet1 {1,2};
    Jet jet2 {3,4};
    Jet result = jet1/jet2;
    ASSERT_NEAR(result.x, jet1.x/jet2.x, 1e-6);
    ASSERT_NEAR(result.x_prime, (jet2.x*jet1.x_prime-jet1.x*jet2.x_prime)/(jet2.x*jet2.x), 1e-6);
}

Jet exp(const Jet& jet)
{
    return Jet {std::exp(jet.x), std::exp(jet.x)};
}

TEST(ForwardDiff, Exponential)
{
    Jet jet {1,1};
    Jet result = exp(jet);
    ASSERT_NEAR(result.x, std::exp(jet.x), 1e-6);
    ASSERT_NEAR(result.x_prime, std::exp(jet.x), 1e-6);
}

Jet log(const Jet& jet)
{
    return Jet {std::log(jet.x), 1/jet.x};
}

TEST(ForwardDiff, NaturalLogarithm)
{
    Jet jet {1,1};
    Jet result = log(jet);
    ASSERT_NEAR(result.x, std::log(jet.x), 1e-6);
    ASSERT_NEAR(result.x_prime, 1.0/jet.x, 1e-6);
}