#pragma once

#include <algorithm>
#include <concepts>
#include <span>

namespace mc {

template <typename Payoff>
concept PathPayoff =
    std::copy_constructible<Payoff> &&
    requires(const Payoff& payoff, std::span<const double> path) {
        { payoff(path) } noexcept -> std::convertible_to<double>;
    };

class EuropeanCallPayoff final {
public:
    explicit constexpr EuropeanCallPayoff(const double strike) noexcept
        : strike_{strike} {}

    [[nodiscard]] double operator()(std::span<const double> path) const noexcept {
        return std::max(path.back() - strike_, 0.0);
    }

private:
    double strike_;
};

class ArithmeticAsianCallPayoff final {
public:
    explicit constexpr ArithmeticAsianCallPayoff(const double strike) noexcept
        : strike_{strike} {}

    [[nodiscard]] double operator()(std::span<const double> path) const noexcept {
        double running_sum = 0.0;
        for (const double spot : path) {
            running_sum += spot;
        }

        const double average = running_sum / static_cast<double>(path.size());
        return std::max(average - strike_, 0.0);
    }

private:
    double strike_;
};

}  // namespace mc

