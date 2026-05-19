#pragma once

#include <cstdint>
#include <random>

namespace mc {
namespace detail {

[[nodiscard]] inline std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

}  // namespace detail

class ThreadLocalRng final {
public:
    static void seed(const std::uint64_t base_seed, const std::uint32_t worker_id) noexcept {
        const auto worker_seed = detail::splitmix64(
            base_seed + (static_cast<std::uint64_t>(worker_id) + 1ULL) * 0xD1B54A32D192ED03ULL);
        engine().seed(worker_seed);
        standard_normal_distribution().reset();
    }

    [[nodiscard]] static double standard_normal() noexcept {
        return standard_normal_distribution()(engine());
    }

private:
    [[nodiscard]] static std::mt19937_64& engine() noexcept {
        thread_local std::mt19937_64 rng{0x853C49E6748FEA9BULL};
        return rng;
    }

    [[nodiscard]] static std::normal_distribution<double>& standard_normal_distribution() noexcept {
        thread_local std::normal_distribution<double> distribution{0.0, 1.0};
        return distribution;
    }
};

}  // namespace mc

