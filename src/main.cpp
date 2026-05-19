#include "mc/GbmModel.hpp"
#include "mc/MarketParameters.hpp"
#include "mc/ModelPredictor.hpp"
#include "mc/MonteCarloSimulator.hpp"
#include "mc/Payoff.hpp"
#include "mc/SimulationParameters.hpp"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

enum class PayoffKind {
    EuropeanCall,
    ArithmeticAsianCall,
};

struct CliOptions final {
    mc::SimulationParameters simulation{};
    mc::GbmModelConfig model{};
    double strike{100.0};
    std::optional<double> calibration_drift{};
    std::optional<double> calibration_volatility{};
    std::string onnx_model_path{};
    std::string onnx_input_name{"features"};
    std::string onnx_output_name{"parameters"};
    std::vector<float> features{};
    std::uint32_t onnx_intra_op_threads{1U};
    std::uint32_t onnx_warmup_iterations{0U};
    PayoffKind payoff_kind{PayoffKind::EuropeanCall};
    bool show_help{false};
};

[[nodiscard]] const char* require_value(
    const int argc,
    char** argv,
    int& index,
    const std::string_view option_name) {
    if (index + 1 >= argc) {
        throw std::invalid_argument{"missing value for " + std::string{option_name}};
    }
    ++index;
    return argv[index];
}

[[nodiscard]] std::uint64_t parse_u64(const char* text, const std::string_view option_name) {
    if (text[0] == '-') {
        throw std::invalid_argument{"negative value for " + std::string{option_name}};
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0') {
        throw std::invalid_argument{"invalid unsigned integer for " + std::string{option_name}};
    }

    return static_cast<std::uint64_t>(parsed);
}

[[nodiscard]] std::uint32_t parse_u32(const char* text, const std::string_view option_name) {
    const std::uint64_t parsed = parse_u64(text, option_name);
    if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument{"value too large for " + std::string{option_name}};
    }
    return static_cast<std::uint32_t>(parsed);
}

[[nodiscard]] double parse_double(const char* text, const std::string_view option_name) {
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(text, &end);

    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(parsed)) {
        throw std::invalid_argument{"invalid floating-point value for " + std::string{option_name}};
    }

    return parsed;
}

[[nodiscard]] float parse_float(const char* text, const std::string_view option_name) {
    return static_cast<float>(parse_double(text, option_name));
}

void print_usage() {
    std::cout
        << "Usage: mc_pricer [options]\n"
        << "\n"
        << "Options:\n"
        << "  --paths N          Monte Carlo paths, default 1000000\n"
        << "  --steps N          Time steps per path, default 252\n"
        << "  --threads N        Worker threads, default hardware_concurrency\n"
        << "  --chunk N          Atomic fetch chunk size, default 4096\n"
        << "  --seed N           Base RNG seed\n"
        << "  --spot X           Initial spot, default 100\n"
        << "  --strike X         Strike, default 100\n"
        << "  --rate X           Risk-free rate, default 0.05\n"
        << "  --dividend X       Continuous dividend yield, default 0\n"
        << "  --drift X          Explicit GBM drift for static pricing\n"
        << "  --vol X            Volatility, default 0.20\n"
        << "  --maturity X       Maturity in years, default 1\n"
        << "  --payoff NAME      european-call or asian-call\n"
        << "  --calibration-drift X\n"
        << "                     Live calibrated drift for aligned atomic market state\n"
        << "  --calibration-vol X\n"
        << "                     Live calibrated volatility for aligned atomic market state\n"
        << "  --onnx-model PATH  Enable ONNX Runtime predictor when built with MC_ENABLE_ONNX_RUNTIME=ON\n"
        << "  --onnx-input NAME  ONNX input tensor name, default features\n"
        << "  --onnx-output NAME ONNX output tensor name, default parameters\n"
        << "  --feature X        Add one float feature; repeat for the ONNX input vector\n"
        << "  --onnx-intra-op N  ONNX intra-op threads, default 1\n"
        << "  --onnx-warmup N    ONNX warmup inference calls before pricing, default 0\n"
        << "  --help             Show this help\n";
}

[[nodiscard]] CliOptions parse_cli(const int argc, char** argv) {
    CliOptions options{};

    for (int index = 1; index < argc; ++index) {
        const std::string_view arg{argv[index]};

        if (arg == "--help" || arg == "-h") {
            options.show_help = true;
        } else if (arg == "--paths") {
            options.simulation.path_count = parse_u64(require_value(argc, argv, index, arg), arg);
        } else if (arg == "--steps") {
            options.simulation.time_steps = parse_u32(require_value(argc, argv, index, arg), arg);
        } else if (arg == "--threads") {
            options.simulation.thread_count = parse_u32(require_value(argc, argv, index, arg), arg);
        } else if (arg == "--chunk") {
            options.simulation.chunk_size = parse_u64(require_value(argc, argv, index, arg), arg);
        } else if (arg == "--seed") {
            options.simulation.seed = parse_u64(require_value(argc, argv, index, arg), arg);
        } else if (arg == "--spot") {
            options.model.spot = parse_double(require_value(argc, argv, index, arg), arg);
        } else if (arg == "--strike") {
            options.strike = parse_double(require_value(argc, argv, index, arg), arg);
        } else if (arg == "--rate") {
            options.model.risk_free_rate = parse_double(require_value(argc, argv, index, arg), arg);
        } else if (arg == "--dividend") {
            options.model.dividend_yield = parse_double(require_value(argc, argv, index, arg), arg);
        } else if (arg == "--drift") {
            options.model.drift = parse_double(require_value(argc, argv, index, arg), arg);
            options.model.use_explicit_drift = true;
        } else if (arg == "--vol") {
            options.model.volatility = parse_double(require_value(argc, argv, index, arg), arg);
        } else if (arg == "--maturity") {
            options.model.maturity_years = parse_double(require_value(argc, argv, index, arg), arg);
        } else if (arg == "--calibration-drift") {
            options.calibration_drift = parse_double(require_value(argc, argv, index, arg), arg);
        } else if (arg == "--calibration-vol") {
            options.calibration_volatility = parse_double(require_value(argc, argv, index, arg), arg);
        } else if (arg == "--onnx-model") {
            options.onnx_model_path = require_value(argc, argv, index, arg);
        } else if (arg == "--onnx-input") {
            options.onnx_input_name = require_value(argc, argv, index, arg);
        } else if (arg == "--onnx-output") {
            options.onnx_output_name = require_value(argc, argv, index, arg);
        } else if (arg == "--feature") {
            options.features.push_back(parse_float(require_value(argc, argv, index, arg), arg));
        } else if (arg == "--onnx-intra-op") {
            options.onnx_intra_op_threads = parse_u32(require_value(argc, argv, index, arg), arg);
        } else if (arg == "--onnx-warmup") {
            options.onnx_warmup_iterations = parse_u32(require_value(argc, argv, index, arg), arg);
        } else if (arg == "--payoff") {
            const std::string_view value{require_value(argc, argv, index, arg)};
            if (value == "european-call") {
                options.payoff_kind = PayoffKind::EuropeanCall;
            } else if (value == "asian-call") {
                options.payoff_kind = PayoffKind::ArithmeticAsianCall;
            } else {
                throw std::invalid_argument{"unsupported payoff: " + std::string{value}};
            }
        } else {
            throw std::invalid_argument{"unknown option: " + std::string{arg}};
        }
    }

    return options;
}

[[nodiscard]] bool has_manual_calibration(const CliOptions& options) noexcept {
    return options.calibration_drift.has_value() || options.calibration_volatility.has_value();
}

[[nodiscard]] bool has_live_calibration(const CliOptions& options) noexcept {
    return has_manual_calibration(options) || !options.onnx_model_path.empty();
}

[[nodiscard]] mc::MarketParameters static_market_parameters(const CliOptions& options) noexcept {
    return mc::MarketParameters{
        .drift = options.model.effective_drift(),
        .volatility = options.model.volatility,
    };
}

[[nodiscard]] mc::MarketParameters resolve_live_market_parameters(const CliOptions& options) {
    if (!options.onnx_model_path.empty()) {
        if (options.features.empty()) {
            throw std::invalid_argument{"--onnx-model requires at least one --feature value"};
        }

#if MC_ENABLE_ONNX_RUNTIME
        mc::OnnxModelPredictor predictor{
            mc::OnnxModelPredictorConfig{
                .model_path = options.onnx_model_path,
                .input_name = options.onnx_input_name,
                .output_name = options.onnx_output_name,
                .feature_count = options.features.size(),
                .intra_op_threads = options.onnx_intra_op_threads,
                .warmup_iterations = options.onnx_warmup_iterations,
            }};
        return predictor.predict(options.features);
#else
        throw std::invalid_argument{
            "ONNX Runtime support is disabled. Rebuild with -DMC_ENABLE_ONNX_RUNTIME=ON"};
#endif
    }

    if (has_manual_calibration(options)) {
        if (!options.calibration_drift.has_value() || !options.calibration_volatility.has_value()) {
            throw std::invalid_argument{
                "--calibration-drift and --calibration-vol must be provided together"};
        }

        const mc::MarketParameters parameters{
            .drift = *options.calibration_drift,
            .volatility = *options.calibration_volatility,
        };
        mc::validate_market_parameters(parameters);
        return parameters;
    }

    const mc::MarketParameters parameters = static_market_parameters(options);
    mc::validate_market_parameters(parameters);
    return parameters;
}

void print_result(
    const CliOptions& options,
    const mc::SimulationResult& result,
    const std::string_view payoff_name,
    const std::string_view calibration_mode,
    const mc::MarketParameters market_parameters) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Monte Carlo GBM benchmark\n";
    std::cout << "payoff: " << payoff_name << '\n';
    std::cout << "calibration_mode: " << calibration_mode << '\n';
    std::cout << "paths: " << result.simulated_paths << '\n';
    std::cout << "steps: " << options.simulation.time_steps << '\n';
    std::cout << "threads: " << result.worker_threads << '\n';
    std::cout << "chunk_size: " << options.simulation.chunk_size << '\n';
    std::cout << "spot: " << options.model.spot << '\n';
    std::cout << "strike: " << options.strike << '\n';
    std::cout << "rate: " << options.model.risk_free_rate << '\n';
    std::cout << "dividend: " << options.model.dividend_yield << '\n';
    std::cout << "gbm_drift: " << market_parameters.drift << '\n';
    std::cout << "volatility: " << options.model.volatility << '\n';
    std::cout << "calibrated_volatility: " << market_parameters.volatility << '\n';
    std::cout << "maturity_years: " << options.model.maturity_years << '\n';
    std::cout << "price: " << result.price << '\n';
    std::cout << "standard_error: " << result.standard_error << '\n';
    std::cout << "confidence_95_radius: " << result.confidence_95_radius << '\n';
    std::cout << "elapsed_seconds: " << result.elapsed_seconds << '\n';
    std::cout << "throughput_mpaths_per_second: " << result.throughput_mpaths_per_second << '\n';
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const CliOptions options = parse_cli(argc, argv);
        if (options.show_help) {
            print_usage();
            return EXIT_SUCCESS;
        }

        options.simulation.validate();
        const mc::MonteCarloSimulator simulator{options.simulation};
        const bool use_live_calibration = has_live_calibration(options);
        const mc::MarketParameters market_parameters = resolve_live_market_parameters(options);
        const std::string_view calibration_mode = use_live_calibration ? "live-atomic" : "static";
        const mc::AtomicMarketParameters live_market_parameters{market_parameters};

        if (options.payoff_kind == PayoffKind::EuropeanCall) {
            const mc::EuropeanCallPayoff payoff{options.strike};
            const mc::SimulationResult result =
                use_live_calibration
                    ? simulator.price_live_gbm(options.model, live_market_parameters, payoff)
                    : simulator.price(mc::GbmModel{options.model}, payoff);
            print_result(options, result, "european-call", calibration_mode, market_parameters);
        } else {
            const mc::ArithmeticAsianCallPayoff payoff{options.strike};
            const mc::SimulationResult result =
                use_live_calibration
                    ? simulator.price_live_gbm(options.model, live_market_parameters, payoff)
                    : simulator.price(mc::GbmModel{options.model}, payoff);
            print_result(options, result, "asian-call", calibration_mode, market_parameters);
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
