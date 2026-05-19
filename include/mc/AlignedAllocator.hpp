#pragma once

#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>

namespace mc {

#if defined(__cpp_lib_hardware_interference_size)
inline constexpr std::size_t kCacheLineSize = std::hardware_destructive_interference_size;
#else
inline constexpr std::size_t kCacheLineSize = 64U;
#endif

template <typename T, std::size_t Alignment>
class AlignedAllocator final {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    static_assert(Alignment >= alignof(T));

    constexpr AlignedAllocator() noexcept = default;

    template <typename U>
    constexpr explicit AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(const std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length{};
        }

        void* memory = ::operator new(count * sizeof(T), std::align_val_t{Alignment});
        return static_cast<T*>(memory);
    }

    void deallocate(T* pointer, std::size_t) noexcept {
        ::operator delete(pointer, std::align_val_t{Alignment});
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template <typename T, std::size_t AlignmentT, typename U, std::size_t AlignmentU>
[[nodiscard]] constexpr bool operator==(
    const AlignedAllocator<T, AlignmentT>&,
    const AlignedAllocator<U, AlignmentU>&) noexcept {
    return AlignmentT == AlignmentU;
}

template <typename T, std::size_t AlignmentT, typename U, std::size_t AlignmentU>
[[nodiscard]] constexpr bool operator!=(
    const AlignedAllocator<T, AlignmentT>& left,
    const AlignedAllocator<U, AlignmentU>& right) noexcept {
    return !(left == right);
}

}  // namespace mc

