#pragma once

#include <concepts>
#include <cstdint>
#include <span>

namespace mc {

template <typename Kernel>
concept IPathKernel =
    std::copy_constructible<Kernel> &&
    requires(const Kernel& kernel, std::span<double> path) {
        { kernel.generate_path(path) } noexcept -> std::same_as<void>;
        { kernel.discount_factor() } noexcept -> std::convertible_to<double>;
    };

template <typename Model>
concept IMarketModel =
    std::copy_constructible<Model> &&
    requires(const Model& model, std::uint32_t time_steps) {
        typename Model::Kernel;
        { model.compile(time_steps) } noexcept -> std::same_as<typename Model::Kernel>;
    } &&
    IPathKernel<typename Model::Kernel>;

}  // namespace mc

