#pragma once

#include "mc/MarketParameters.hpp"
#include "mc/ThreadLocalRng.hpp"

#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace mc {

struct GbmModelConfig final {
    double spot{100.0};
    double risk_free_rate{0.05};
    double dividend_yield{0.0};
    double drift{0.05};
    double volatility{0.20};
    double maturity_years{1.0};
    bool use_explicit_drift{false};

    void validate() const {
        if (!std::isfinite(spot) || spot <= 0.0) {
            throw std::invalid_argument{"spot must be finite and positive"};
        }
        if (!std::isfinite(risk_free_rate)) {
            throw std::invalid_argument{"risk_free_rate must be finite"};
        }
        if (!std::isfinite(dividend_yield)) {
            throw std::invalid_argument{"dividend_yield must be finite"};
        }
        if (!std::isfinite(drift)) {
            throw std::invalid_argument{"drift must be finite"};
        }
        if (!std::isfinite(volatility) || volatility < 0.0) {
            throw std::invalid_argument{"volatility must be finite and non-negative"};
        }
        if (!std::isfinite(maturity_years) || maturity_years <= 0.0) {
            throw std::invalid_argument{"maturity_years must be finite and positive"};
        }
    }

    void apply_market_parameters(const MarketParameters parameters) {
        validate_market_parameters(parameters);
        drift = parameters.drift;
        volatility = parameters.volatility;
        use_explicit_drift = true;
    }

    [[nodiscard]] constexpr double effective_drift() const noexcept {
        return use_explicit_drift ? drift : risk_free_rate - dividend_yield;
    }
};

class GbmModel final {
public:
    class Kernel final {
    public:
        constexpr Kernel(
            const double initial_price,
            const double discount_factor,
            const double drift_per_step,
            const double diffusion_per_step) noexcept
            : initial_price_{initial_price},
              discount_factor_{discount_factor},
              drift_per_step_{drift_per_step},
              diffusion_per_step_{diffusion_per_step} {}

        void generate_path(std::span<double> path) const noexcept {
            double spot = initial_price_;
            path[0] = spot;

            for (std::size_t step = 1U; step < path.size(); ++step) {
                const double z = ThreadLocalRng::standard_normal();
                const double log_return = std::fma(diffusion_per_step_, z, drift_per_step_);
                spot *= std::exp(log_return);
                path[step] = spot;
            }
        }

        [[nodiscard]] constexpr double discount_factor() const noexcept {
            return discount_factor_;
        }

    private:
        double initial_price_;
        double discount_factor_;
        double drift_per_step_;
        double diffusion_per_step_;
    };

    explicit GbmModel(GbmModelConfig config)
        : config_{config} {
        config_.validate();
    }

    [[nodiscard]] Kernel compile(const std::uint32_t time_steps) const noexcept {
        const double dt = config_.maturity_years / static_cast<double>(time_steps);
        const double variance = config_.volatility * config_.volatility;
        const double drift_per_step = (config_.effective_drift() - 0.5 * variance) * dt;
        const double diffusion_per_step = config_.volatility * std::sqrt(dt);
        const double discount_factor = std::exp(-config_.risk_free_rate * config_.maturity_years);

        return Kernel{
            config_.spot,
            discount_factor,
            drift_per_step,
            diffusion_per_step,
        };
    }

private:
    GbmModelConfig config_;
};

}  // namespace mc
