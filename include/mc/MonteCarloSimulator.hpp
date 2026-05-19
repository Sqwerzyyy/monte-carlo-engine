#pragma once

#include "mc/AlignedAllocator.hpp"
#include "mc/GbmModel.hpp"
#include "mc/IMarketModel.hpp"
#include "mc/MarketParameters.hpp"
#include "mc/Payoff.hpp"
#include "mc/SimulationParameters.hpp"
#include "mc/ThreadLocalRng.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace mc {

struct SimulationResult final {
    double price{0.0};
    double standard_error{0.0};
    double confidence_95_radius{0.0};
    double elapsed_seconds{0.0};
    double throughput_mpaths_per_second{0.0};
    std::uint64_t simulated_paths{0U};
    std::uint32_t worker_threads{0U};
};

namespace detail {

struct alignas(kCacheLineSize) AlignedAccumulator final {
    double payoff_sum{0.0};
    double payoff_square_sum{0.0};
    std::uint64_t path_count{0U};
};

struct alignas(kCacheLineSize) AtomicPathCursor final {
    std::atomic<std::uint64_t> next_path{0U};
};

[[nodiscard]] constexpr std::size_t round_up_to_multiple(
    const std::size_t value,
    const std::size_t multiple) noexcept {
    return ((value + multiple - 1U) / multiple) * multiple;
}

static_assert(alignof(AlignedAccumulator) >= kCacheLineSize);
static_assert(alignof(AtomicPathCursor) >= kCacheLineSize);

}  // namespace detail

class MonteCarloSimulator final {
public:
    explicit MonteCarloSimulator(SimulationParameters parameters)
        : parameters_{parameters} {
        parameters_.validate();
    }

    template <IMarketModel Model, PathPayoff Payoff>
    [[nodiscard]] SimulationResult price(const Model& model, const Payoff& payoff) const {
        using AccumulatorVector =
            std::vector<detail::AlignedAccumulator,
                        AlignedAllocator<detail::AlignedAccumulator, kCacheLineSize>>;
        using PathStorage =
            std::vector<double, AlignedAllocator<double, kCacheLineSize>>;

        const auto kernel = model.compile(parameters_.time_steps);
        const std::uint32_t worker_count = parameters_.effective_threads();
        const std::size_t path_size = static_cast<std::size_t>(parameters_.time_steps) + 1U;
        const std::size_t doubles_per_cache_line = kCacheLineSize / sizeof(double);
        const std::size_t path_stride =
            detail::round_up_to_multiple(path_size, doubles_per_cache_line);

        if (path_stride > std::numeric_limits<std::size_t>::max() /
                              static_cast<std::size_t>(worker_count)) {
            throw std::length_error{"path buffer allocation size overflow"};
        }

        AccumulatorVector accumulators(worker_count);
        PathStorage path_storage(static_cast<std::size_t>(worker_count) * path_stride);
        std::vector<std::jthread> workers;
        workers.reserve(worker_count);
        detail::AtomicPathCursor cursor{};

        const auto started_at = std::chrono::high_resolution_clock::now();

        for (std::uint32_t worker_id = 0U; worker_id < worker_count; ++worker_id) {
            workers.emplace_back(
                [&, worker_id](std::stop_token stop_token) noexcept {
                    ThreadLocalRng::seed(parameters_.seed, worker_id);

                    double local_sum = 0.0;
                    double local_square_sum = 0.0;
                    std::uint64_t local_path_count = 0U;

                    double* const path_base =
                        path_storage.data() + static_cast<std::size_t>(worker_id) * path_stride;
                    std::span<double> path{path_base, path_size};
                    const std::span<const double> payoff_path{path.data(), path.size()};

                    for (;;) {
                        if (stop_token.stop_requested()) {
                            break;
                        }

                        const std::uint64_t begin =
                            cursor.next_path.fetch_add(parameters_.chunk_size, std::memory_order_relaxed);
                        if (begin >= parameters_.path_count) {
                            break;
                        }

                        const std::uint64_t remaining = parameters_.path_count - begin;
                        const std::uint64_t batch_size = std::min(parameters_.chunk_size, remaining);

                        for (std::uint64_t offset = 0U; offset < batch_size; ++offset) {
                            kernel.generate_path(path);
                            const double payoff_value = payoff(payoff_path);
                            local_sum += payoff_value;
                            local_square_sum = std::fma(payoff_value, payoff_value, local_square_sum);
                            ++local_path_count;
                        }
                    }

                    detail::AlignedAccumulator& accumulator = accumulators[worker_id];
                    accumulator.payoff_sum = local_sum;
                    accumulator.payoff_square_sum = local_square_sum;
                    accumulator.path_count = local_path_count;
                });
        }

        for (std::jthread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        const auto finished_at = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsed = finished_at - started_at;

        double payoff_sum = 0.0;
        double payoff_square_sum = 0.0;
        std::uint64_t simulated_paths = 0U;

        for (const detail::AlignedAccumulator& accumulator : accumulators) {
            payoff_sum += accumulator.payoff_sum;
            payoff_square_sum += accumulator.payoff_square_sum;
            simulated_paths += accumulator.path_count;
        }

        const double n = static_cast<double>(simulated_paths);
        const double mean_payoff = payoff_sum / n;
        const double second_moment = payoff_square_sum / n;
        const double payoff_variance = std::max(second_moment - mean_payoff * mean_payoff, 0.0);
        const double discount_factor = kernel.discount_factor();
        const double standard_error = discount_factor * std::sqrt(payoff_variance / n);
        const double elapsed_seconds = elapsed.count();
        const double throughput =
            (static_cast<double>(simulated_paths) / elapsed_seconds) / 1'000'000.0;

        return SimulationResult{
            .price = discount_factor * mean_payoff,
            .standard_error = standard_error,
            .confidence_95_radius = 1.959963984540054 * standard_error,
            .elapsed_seconds = elapsed_seconds,
            .throughput_mpaths_per_second = throughput,
            .simulated_paths = simulated_paths,
            .worker_threads = worker_count,
        };
    }

    template <PathPayoff Payoff>
    [[nodiscard]] SimulationResult price_live_gbm(
        const GbmModelConfig& base_config,
        const AtomicMarketParameters& market_parameters,
        const Payoff& payoff) const {
        using AccumulatorVector =
            std::vector<detail::AlignedAccumulator,
                        AlignedAllocator<detail::AlignedAccumulator, kCacheLineSize>>;
        using PathStorage =
            std::vector<double, AlignedAllocator<double, kCacheLineSize>>;

        base_config.validate();

        const std::uint32_t worker_count = parameters_.effective_threads();
        const std::size_t path_size = static_cast<std::size_t>(parameters_.time_steps) + 1U;
        const std::size_t doubles_per_cache_line = kCacheLineSize / sizeof(double);
        const std::size_t path_stride =
            detail::round_up_to_multiple(path_size, doubles_per_cache_line);

        if (path_stride > std::numeric_limits<std::size_t>::max() /
                              static_cast<std::size_t>(worker_count)) {
            throw std::length_error{"path buffer allocation size overflow"};
        }

        AccumulatorVector accumulators(worker_count);
        PathStorage path_storage(static_cast<std::size_t>(worker_count) * path_stride);
        std::vector<std::jthread> workers;
        workers.reserve(worker_count);
        detail::AtomicPathCursor cursor{};

        const double dt = base_config.maturity_years / static_cast<double>(parameters_.time_steps);
        const double sqrt_dt = std::sqrt(dt);
        const double discount_factor =
            std::exp(-base_config.risk_free_rate * base_config.maturity_years);

        const auto started_at = std::chrono::high_resolution_clock::now();

        for (std::uint32_t worker_id = 0U; worker_id < worker_count; ++worker_id) {
            workers.emplace_back(
                [&, worker_id](std::stop_token stop_token) noexcept {
                    ThreadLocalRng::seed(parameters_.seed, worker_id);

                    double local_sum = 0.0;
                    double local_square_sum = 0.0;
                    std::uint64_t local_path_count = 0U;

                    double* const path_base =
                        path_storage.data() + static_cast<std::size_t>(worker_id) * path_stride;
                    std::span<double> path{path_base, path_size};
                    const std::span<const double> payoff_path{path.data(), path.size()};

                    for (;;) {
                        if (stop_token.stop_requested()) {
                            break;
                        }

                        const std::uint64_t begin =
                            cursor.next_path.fetch_add(parameters_.chunk_size, std::memory_order_relaxed);
                        if (begin >= parameters_.path_count) {
                            break;
                        }

                        const MarketParameters snapshot = market_parameters.load();
                        const double variance = snapshot.volatility * snapshot.volatility;
                        const GbmModel::Kernel kernel{
                            base_config.spot,
                            discount_factor,
                            (snapshot.drift - 0.5 * variance) * dt,
                            snapshot.volatility * sqrt_dt,
                        };

                        const std::uint64_t remaining = parameters_.path_count - begin;
                        const std::uint64_t batch_size = std::min(parameters_.chunk_size, remaining);

                        for (std::uint64_t offset = 0U; offset < batch_size; ++offset) {
                            kernel.generate_path(path);
                            const double discounted_payoff = kernel.discount_factor() * payoff(payoff_path);
                            local_sum += discounted_payoff;
                            local_square_sum =
                                std::fma(discounted_payoff, discounted_payoff, local_square_sum);
                            ++local_path_count;
                        }
                    }

                    detail::AlignedAccumulator& accumulator = accumulators[worker_id];
                    accumulator.payoff_sum = local_sum;
                    accumulator.payoff_square_sum = local_square_sum;
                    accumulator.path_count = local_path_count;
                });
        }

        for (std::jthread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        const auto finished_at = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsed = finished_at - started_at;

        double discounted_payoff_sum = 0.0;
        double discounted_payoff_square_sum = 0.0;
        std::uint64_t simulated_paths = 0U;

        for (const detail::AlignedAccumulator& accumulator : accumulators) {
            discounted_payoff_sum += accumulator.payoff_sum;
            discounted_payoff_square_sum += accumulator.payoff_square_sum;
            simulated_paths += accumulator.path_count;
        }

        const double n = static_cast<double>(simulated_paths);
        const double mean_value = discounted_payoff_sum / n;
        const double second_moment = discounted_payoff_square_sum / n;
        const double value_variance = std::max(second_moment - mean_value * mean_value, 0.0);
        const double standard_error = std::sqrt(value_variance / n);
        const double elapsed_seconds = elapsed.count();
        const double throughput =
            (static_cast<double>(simulated_paths) / elapsed_seconds) / 1'000'000.0;

        return SimulationResult{
            .price = mean_value,
            .standard_error = standard_error,
            .confidence_95_radius = 1.959963984540054 * standard_error,
            .elapsed_seconds = elapsed_seconds,
            .throughput_mpaths_per_second = throughput,
            .simulated_paths = simulated_paths,
            .worker_threads = worker_count,
        };
    }

private:
    SimulationParameters parameters_;
};

}  // namespace mc
