# Monte Carlo Pricing Engine

Production-oriented C++20 Monte Carlo pricing engine for GBM asset simulation.

The project is intentionally terminal-first: no IDE is required. Build it with
CMake, run the `mc_pricer` binary, and tune paths, threads, chunk size, payoff,
and live market calibration from the command line.

## Highlights

- C++20 implementation with `std::jthread`, concepts, spans, and atomics.
- GBM path simulation with precomputed drift, diffusion, and discount factors.
- Dynamic work distribution through atomic chunk fetching.
- `thread_local std::mt19937_64` and `thread_local std::normal_distribution`.
- Cache-line aligned per-worker buffers and accumulators to reduce false sharing.
- Zero allocation inside the per-path simulation loop.
- Built-in latency and throughput benchmark output.
- Optional live drift/volatility calibration through aligned lock-free atomics.
- Optional ONNX Runtime inference wrapper for model-driven market parameters.

## Requirements

| Dependency | Version |
| --- | --- |
| CMake | 3.20+ |
| C++ compiler | C++20 capable compiler |
| Threads | pthreads / platform thread library |
| ONNX Runtime | Optional, only for `MC_ENABLE_ONNX_RUNTIME=ON` |

The project is tested with AppleClang 17 on macOS. It should also build with
recent Clang or GCC on Linux.

## Quick Start

Clone and build in Release mode:

```sh
git clone <your-repo-url> monte-carlo-engine
cd monte-carlo-engine

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Run a European call benchmark:

```sh
./build/mc_pricer --paths 1000000 --steps 252 --threads 8 --payoff european-call
```

Run an arithmetic Asian call benchmark:

```sh
./build/mc_pricer --paths 1000000 --steps 252 --threads 8 --payoff asian-call
```

Example output:

```text
Monte Carlo GBM benchmark
payoff: european-call
calibration_mode: static
paths: 1000000
steps: 252
threads: 8
chunk_size: 4096
spot: 100.000000
strike: 100.000000
rate: 0.050000
dividend: 0.000000
gbm_drift: 0.050000
volatility: 0.200000
calibrated_volatility: 0.200000
maturity_years: 1.000000
price: 10.45
standard_error: 0.0147
confidence_95_radius: 0.0288
elapsed_seconds: 1.92
throughput_mpaths_per_second: 0.52
```

Exact numbers vary by CPU, compiler, flags, random seed, and workload.

## Command-Line Options

| Option | Default | Description |
| --- | ---: | --- |
| `--paths N` | `1000000` | Number of Monte Carlo paths. |
| `--steps N` | `252` | Time steps per path. |
| `--threads N` | hardware concurrency | Worker thread count. |
| `--chunk N` | `4096` | Paths fetched per atomic work chunk. |
| `--seed N` | fixed seed | Base RNG seed. |
| `--spot X` | `100` | Initial asset price. |
| `--strike X` | `100` | Option strike. |
| `--rate X` | `0.05` | Risk-free rate. |
| `--dividend X` | `0` | Continuous dividend yield. |
| `--drift X` | `rate - dividend` | Explicit GBM drift for static pricing. |
| `--vol X` | `0.20` | Static volatility. |
| `--maturity X` | `1` | Maturity in years. |
| `--payoff NAME` | `european-call` | `european-call` or `asian-call`. |
| `--calibration-drift X` | unset | Live calibrated drift. |
| `--calibration-vol X` | unset | Live calibrated volatility. |
| `--onnx-model PATH` | unset | ONNX model path, if ONNX support is enabled. |
| `--onnx-input NAME` | `features` | ONNX input tensor name. |
| `--onnx-output NAME` | `parameters` | ONNX output tensor name. |
| `--feature X` | unset | Add one ONNX float feature. Repeat per feature. |
| `--onnx-intra-op N` | `1` | ONNX Runtime intra-op threads. |
| `--onnx-warmup N` | `0` | Warmup inference calls before pricing. |

## Live Calibration

The simulator can run with market parameters stored in `alignas(64)`
`AtomicMarketParameters`. Worker threads read a consistent drift/volatility
snapshot at chunk boundaries, avoiding locks and avoiding false sharing.

Run with manually calibrated parameters:

```sh
./build/mc_pricer --paths 1000000 --steps 252 --threads 8 \
  --payoff european-call \
  --calibration-drift 0.045 \
  --calibration-vol 0.23
```

The output should show:

```text
calibration_mode: live-atomic
gbm_drift: 0.045000
calibrated_volatility: 0.230000
```

This path exercises the same atomic market-parameter pipeline that the ONNX
predictor uses. The only difference is that values come from CLI arguments
instead of an inference model.

## Optional ONNX Runtime Build

ONNX support is disabled by default so the base engine builds without external
ML dependencies.

Expected model contract:

| Tensor | Shape | Type | Meaning |
| --- | --- | --- | --- |
| input | `[1, feature_count]` | `float32` | Alternative data / market features. |
| output | `[1, 2]` | `float32` | `[drift, volatility]`. |

Build with ONNX Runtime:

```sh
cmake -S . -B build-onnx -DCMAKE_BUILD_TYPE=Release \
  -DMC_ENABLE_ONNX_RUNTIME=ON \
  -DMC_ONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime

cmake --build build-onnx --config Release
```

Run with an ONNX calibrator:

```sh
./build-onnx/mc_pricer --paths 1000000 --steps 252 --threads 8 \
  --payoff european-call \
  --onnx-model ./models/calibrator.onnx \
  --onnx-input features \
  --onnx-output parameters \
  --feature 0.12 \
  --feature -0.03 \
  --feature 1.70
```

The ONNX wrapper preallocates input/output tensors. For true zero-copy feature
ingestion, write market features directly into `OnnxModelPredictor`'s
`mutable_feature_buffer()` and call `predict_in_place()`.

## Architecture

```text
.
|-- CMakeLists.txt
|-- include
|   `-- mc
|       |-- AlignedAllocator.hpp
|       |-- GbmModel.hpp
|       |-- IMarketModel.hpp
|       |-- MarketParameters.hpp
|       |-- ModelPredictor.hpp
|       |-- MonteCarloSimulator.hpp
|       |-- Payoff.hpp
|       |-- SimulationParameters.hpp
|       `-- ThreadLocalRng.hpp
`-- src
    `-- main.cpp
```

Core components:

| File | Role |
| --- | --- |
| `GbmModel.hpp` | GBM kernel generation and static calibrated drift support. |
| `MonteCarloSimulator.hpp` | Parallel path simulation, timing, throughput, live GBM path. |
| `MarketParameters.hpp` | Lock-free aligned market parameter storage. |
| `ModelPredictor.hpp` | Optional ONNX Runtime predictor and async updater scaffold. |
| `Payoff.hpp` | European call and arithmetic Asian call payoffs. |
| `ThreadLocalRng.hpp` | Per-thread RNG and normal distribution. |
| `AlignedAllocator.hpp` | Cache-line aligned vector allocator. |

## Performance Notes

Release builds use aggressive optimization flags:

```text
-O3 -march=native -ffast-math -Wall -Wextra -Wpedantic
```

The hot path avoids allocation by allocating worker path storage before threads
start. Each worker writes to its own aligned path slice and local accumulator.
Global coordination is limited to relaxed atomic work fetching.

Useful tuning knobs:

```sh
./build/mc_pricer --paths 1000000 --steps 252 --threads 4 --chunk 2048
./build/mc_pricer --paths 1000000 --steps 252 --threads 8 --chunk 4096
./build/mc_pricer --paths 5000000 --steps 252 --threads 8 --chunk 8192
```

Smaller chunks can react faster to live parameter changes. Larger chunks reduce
atomic scheduling overhead.

## Troubleshooting

`zsh: no such file or directory: ./build-onnx/mc_pricer`

You have not built the ONNX target yet. Use the normal binary first:

```sh
./build/mc_pricer --paths 100000 --steps 252 --threads 4
```

Then build `build-onnx` only after ONNX Runtime is installed.

`ONNX Runtime support is disabled`

Reconfigure with:

```sh
cmake -S . -B build-onnx -DCMAKE_BUILD_TYPE=Release \
  -DMC_ENABLE_ONNX_RUNTIME=ON \
  -DMC_ONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime
```

`Cannot find onnxruntime_cxx_api.h`

Check that `MC_ONNXRUNTIME_ROOT` points to a directory with:

```text
include/onnxruntime_cxx_api.h
lib/libonnxruntime.*
```

CLion shows red includes but terminal build works

Select the `mc_pricer` CMake run configuration instead of `main.cpp`, then reload
or invalidate IDE caches. Terminal builds are authoritative.

## License

See [LICENSE](LICENSE).
