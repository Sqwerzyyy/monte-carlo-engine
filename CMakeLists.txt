cmake_minimum_required(VERSION 3.20)

project(MonteCarloPricingEngine
    VERSION 1.0.0
    LANGUAGES CXX
)

find_package(Threads REQUIRED)

option(MC_ENABLE_ONNX_RUNTIME "Build ONNX Runtime-backed ModelPredictor" OFF)
set(MC_ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime installation root containing include/ and lib/")

add_executable(mc_pricer
    src/main.cpp
    include/mc/AlignedAllocator.hpp
    include/mc/GbmModel.hpp
    include/mc/IMarketModel.hpp
    include/mc/MarketParameters.hpp
    include/mc/ModelPredictor.hpp
    include/mc/MonteCarloSimulator.hpp
    include/mc/Payoff.hpp
    include/mc/SimulationParameters.hpp
    include/mc/ThreadLocalRng.hpp
)

target_include_directories(mc_pricer
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(mc_pricer PRIVATE cxx_std_20)

set_target_properties(mc_pricer PROPERTIES
    CXX_EXTENSIONS OFF
)

target_link_libraries(mc_pricer
    PRIVATE
        Threads::Threads
)

if(MC_ENABLE_ONNX_RUNTIME)
    target_compile_definitions(mc_pricer PRIVATE MC_ENABLE_ONNX_RUNTIME=1)

    find_package(onnxruntime CONFIG QUIET)
    if(onnxruntime_FOUND)
        if(TARGET onnxruntime::onnxruntime)
            target_link_libraries(mc_pricer PRIVATE onnxruntime::onnxruntime)
        elseif(TARGET onnxruntime)
            target_link_libraries(mc_pricer PRIVATE onnxruntime)
        else()
            message(FATAL_ERROR "onnxruntime package found, but no known CMake target is exported")
        endif()
    else()
        if(NOT MC_ONNXRUNTIME_ROOT)
            message(FATAL_ERROR
                "MC_ENABLE_ONNX_RUNTIME=ON requires either find_package(onnxruntime) "
                "or -DMC_ONNXRUNTIME_ROOT=/path/to/onnxruntime")
        endif()

        find_path(ONNXRUNTIME_INCLUDE_DIR
            NAMES onnxruntime_cxx_api.h
            HINTS ${MC_ONNXRUNTIME_ROOT}/include
            REQUIRED
        )
        find_library(ONNXRUNTIME_LIBRARY
            NAMES onnxruntime
            HINTS ${MC_ONNXRUNTIME_ROOT}/lib
            REQUIRED
        )

        target_include_directories(mc_pricer PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
        target_link_libraries(mc_pricer PRIVATE ${ONNXRUNTIME_LIBRARY})
    endif()
else()
    target_compile_definitions(mc_pricer PRIVATE MC_ENABLE_ONNX_RUNTIME=0)
endif()

if(MSVC)
    target_compile_options(mc_pricer PRIVATE
        /O2
        /W4
        /fp:fast
        /arch:AVX2
    )
else()
    target_compile_options(mc_pricer PRIVATE
        -O3
        -march=native
        -ffast-math
        -Wall
        -Wextra
        -Wpedantic
    )
endif()
