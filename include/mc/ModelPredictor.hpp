#pragma once

#include "mc/AlignedAllocator.hpp"
#include "mc/MarketParameters.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef MC_ENABLE_ONNX_RUNTIME
#define MC_ENABLE_ONNX_RUNTIME 0
#endif

#if MC_ENABLE_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace mc {

using FeatureBuffer = std::vector<float, AlignedAllocator<float, kCacheLineSize>>;

struct OnnxModelPredictorConfig final {
    std::string model_path;
    std::string input_name{"features"};
    std::string output_name{"parameters"};
    std::size_t feature_count{0U};
    std::uint32_t intra_op_threads{1U};
    std::uint32_t warmup_iterations{0U};
};

class StaticModelPredictor final {
public:
    explicit constexpr StaticModelPredictor(const MarketParameters parameters) noexcept
        : parameters_{parameters} {}

    [[nodiscard]] MarketParameters predict(std::span<const float>) const noexcept {
        return parameters_;
    }

private:
    MarketParameters parameters_;
};

#if MC_ENABLE_ONNX_RUNTIME

class OnnxModelPredictor final {
public:
    explicit OnnxModelPredictor(OnnxModelPredictorConfig config)
        : config_{std::move(config)},
          env_{ORT_LOGGING_LEVEL_WARNING, "mc_model_predictor"},
          session_options_{},
          session_{nullptr},
          memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)},
          input_buffer_(config_.feature_count),
          output_buffer_{0.0F, 0.0F},
          input_shape_{1, static_cast<std::int64_t>(config_.feature_count)},
          output_shape_{1, static_cast<std::int64_t>(output_buffer_.size())},
          input_tensor_{nullptr},
          output_tensor_{nullptr},
          input_names_{config_.input_name.c_str()},
          output_names_{config_.output_name.c_str()} {
        if (config_.model_path.empty()) {
            throw std::invalid_argument{"ONNX model path must not be empty"};
        }
        if (config_.feature_count == 0U) {
            throw std::invalid_argument{"ONNX feature_count must be positive"};
        }
        if (config_.intra_op_threads == 0U) {
            throw std::invalid_argument{"ONNX intra_op_threads must be positive"};
        }

        session_options_.SetIntraOpNumThreads(static_cast<int>(config_.intra_op_threads));
        session_options_.SetInterOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = Ort::Session{env_, config_.model_path.c_str(), session_options_};

        input_tensor_ = Ort::Value::CreateTensor<float>(
            memory_info_,
            input_buffer_.data(),
            input_buffer_.size(),
            input_shape_.data(),
            input_shape_.size());

        output_tensor_ = Ort::Value::CreateTensor<float>(
            memory_info_,
            output_buffer_.data(),
            output_buffer_.size(),
            output_shape_.data(),
            output_shape_.size());

        warm_up();
    }

    OnnxModelPredictor(const OnnxModelPredictor&) = delete;
    OnnxModelPredictor& operator=(const OnnxModelPredictor&) = delete;
    OnnxModelPredictor(OnnxModelPredictor&&) noexcept = default;
    OnnxModelPredictor& operator=(OnnxModelPredictor&&) noexcept = default;

    [[nodiscard]] MarketParameters predict(std::span<const float> features) {
        if (features.size() != input_buffer_.size()) {
            throw std::invalid_argument{"ONNX feature vector size does not match feature_count"};
        }

        std::copy(features.begin(), features.end(), input_buffer_.begin());
        return predict_in_place();
    }

    [[nodiscard]] MarketParameters predict_in_place() {
        session_.Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor_,
            1U,
            output_names_.data(),
            &output_tensor_,
            1U);

        const MarketParameters parameters{
            .drift = static_cast<double>(output_buffer_[0]),
            .volatility = static_cast<double>(output_buffer_[1]),
        };
        validate_market_parameters(parameters);
        return parameters;
    }

    [[nodiscard]] std::span<float> mutable_feature_buffer() noexcept {
        return input_buffer_;
    }

private:
    void warm_up() {
        if (config_.warmup_iterations == 0U) {
            return;
        }

        std::fill(input_buffer_.begin(), input_buffer_.end(), 0.0F);
        for (std::uint32_t iteration = 0U; iteration < config_.warmup_iterations; ++iteration) {
            session_.Run(
                Ort::RunOptions{nullptr},
                input_names_.data(),
                &input_tensor_,
                1U,
                output_names_.data(),
                &output_tensor_,
                1U);
        }
    }

    OnnxModelPredictorConfig config_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;
    FeatureBuffer input_buffer_;
    std::array<float, 2U> output_buffer_;
    std::array<std::int64_t, 2U> input_shape_;
    std::array<std::int64_t, 2U> output_shape_;
    Ort::Value input_tensor_;
    Ort::Value output_tensor_;
    std::array<const char*, 1U> input_names_;
    std::array<const char*, 1U> output_names_;
};

using ModelPredictor = OnnxModelPredictor;

#endif

template <typename Predictor>
class AsyncMarketParameterUpdater final {
public:
    AsyncMarketParameterUpdater(
        Predictor& predictor,
        AtomicMarketParameters& destination,
        std::span<const float> feature_view,
        const std::chrono::nanoseconds refresh_interval) noexcept
        : predictor_{predictor},
          destination_{destination},
          feature_view_{feature_view},
          refresh_interval_{refresh_interval} {}

    AsyncMarketParameterUpdater(const AsyncMarketParameterUpdater&) = delete;
    AsyncMarketParameterUpdater& operator=(const AsyncMarketParameterUpdater&) = delete;

    void start() {
        if (worker_.joinable()) {
            return;
        }

        worker_ = std::jthread{
            [this](std::stop_token stop_token) {
                while (!stop_token.stop_requested()) {
                    try {
                        destination_.store_checked(predictor_.predict(feature_view_));
                    } catch (const std::exception&) {
                        // Keep the last known-good parameters. Production code should route this
                        // to the fund's telemetry channel, not stderr.
                    }

                    if (refresh_interval_.count() > 0) {
                        std::this_thread::sleep_for(refresh_interval_);
                    }
                }
            }};
    }

    void request_stop() noexcept {
        worker_.request_stop();
    }

private:
    Predictor& predictor_;
    AtomicMarketParameters& destination_;
    std::span<const float> feature_view_;
    std::chrono::nanoseconds refresh_interval_;
    std::jthread worker_{};
};

}  // namespace mc
