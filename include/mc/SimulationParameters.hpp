#pragma once

#include <cstdint>
#include <stdexcept>
#include <thread>

namespace mc {

struct SimulationParameters final {
    std::uint64_t path_count{1'000'000ULL};
    std::uint32_t time_steps{252U};
    std::uint32_t thread_count{default_thread_count()};
    std::uint64_t chunk_size{4096ULL};
    std::uint64_t seed{0x9E3779B97F4A7C15ULL};

    [[nodiscard]] static std::uint32_t default_thread_count() noexcept {
        const auto detected = std::thread::hardware_concurrency();
        return detected == 0U ? 1U : detected;
    }

    void validate() const {
        if (path_count == 0U) {
            throw std::invalid_argument{"path_count must be positive"};
        }
        if (time_steps == 0U) {
            throw std::invalid_argument{"time_steps must be positive"};
        }
        if (thread_count == 0U) {
            throw std::invalid_argument{"thread_count must be positive"};
        }
        if (chunk_size == 0U) {
            throw std::invalid_argument{"chunk_size must be positive"};
        }
    }

    [[nodiscard]] std::uint32_t effective_threads() const noexcept {
        if (path_count < static_cast<std::uint64_t>(thread_count)) {
            return static_cast<std::uint32_t>(path_count);
        }
        return thread_count;
    }
};

}  // namespace mc

