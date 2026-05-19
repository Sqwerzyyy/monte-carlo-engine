#pragma once

#include "mc/AlignedAllocator.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace mc {

struct MarketParameters final {
    double drift{0.05};
    double volatility{0.20};
};

inline void validate_market_parameters(const MarketParameters& parameters) {
    if (!std::isfinite(parameters.drift)) {
        throw std::invalid_argument{"market drift must be finite"};
    }
    if (!std::isfinite(parameters.volatility) || parameters.volatility <= 0.0) {
        throw std::invalid_argument{"market volatility must be finite and positive"};
    }
}

class alignas(kCacheLineSize) AtomicMarketParameters final {
public:
    AtomicMarketParameters() noexcept = default;

    explicit AtomicMarketParameters(const MarketParameters initial_parameters) noexcept
        : drift_{initial_parameters.drift},
          volatility_{initial_parameters.volatility} {}

    void store(const MarketParameters parameters) noexcept {
        sequence_.fetch_add(1U, std::memory_order_acq_rel);
        drift_.store(parameters.drift, std::memory_order_relaxed);
        volatility_.store(parameters.volatility, std::memory_order_relaxed);
        sequence_.fetch_add(1U, std::memory_order_release);
    }

    void store_checked(const MarketParameters parameters) {
        validate_market_parameters(parameters);
        store(parameters);
    }

    [[nodiscard]] MarketParameters load() const noexcept {
        MarketParameters snapshot{};

        for (;;) {
            const std::uint64_t observed_before = sequence_.load(std::memory_order_acquire);
            if ((observed_before & 1U) != 0U) {
                continue;
            }

            snapshot.drift = drift_.load(std::memory_order_relaxed);
            snapshot.volatility = volatility_.load(std::memory_order_relaxed);

            const std::uint64_t observed_after = sequence_.load(std::memory_order_acquire);
            if (observed_before == observed_after) {
                return snapshot;
            }
        }
    }

    [[nodiscard]] bool is_lock_free() const noexcept {
        return drift_.is_lock_free() && volatility_.is_lock_free() && sequence_.is_lock_free();
    }

private:
    std::atomic<std::uint64_t> sequence_{0U};
    std::atomic<double> drift_{0.05};
    std::atomic<double> volatility_{0.20};
};

static_assert(alignof(AtomicMarketParameters) >= kCacheLineSize);
static_assert(std::atomic<double>::is_always_lock_free,
              "AtomicMarketParameters requires lock-free double atomics");

}  // namespace mc

