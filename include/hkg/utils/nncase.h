#pragma once
#include "bfloat16.h"
#include "small_vector.h"
#include <numeric>
namespace nncase
{
struct padding
{
    int32_t before;
    int32_t after;

    int32_t sum() const noexcept { return before + after; }

    static padding zero() noexcept { return {}; }
};

template <class T>
struct value_range
{
    T min;
    T max;

    static constexpr value_range<T> full() noexcept
    {
        if (std::is_floating_point<T>::value || std::is_same<T, bfloat16>::value)
            return { -std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity() };
        else
            return { std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max() };
    }

    static constexpr value_range<T> nonnegative() noexcept
    {
        return { 0, std::numeric_limits<T>::max() };
    }

    constexpr T length() const noexcept { return max - min; }
};

using runtime_shape_t = itlib::small_vector<size_t, 4>;
using runtime_axis_t = itlib::small_vector<int32_t, 4>;

namespace runtime
{
    namespace detail
    {
        template <class shape_type, class strides_type>
        inline void adapt_strides(const shape_type &shape, strides_type &strides,
            std::nullptr_t, typename strides_type::size_type i) noexcept
        {
            if (shape[i] == 1)
            {
                strides[i] = 0;
            }
        }

        template <class shape_type, class strides_type, class bs_ptr>
        inline std::size_t compute_strides(const shape_type &shape,
            strides_type &strides, bs_ptr bs)
        {
            using strides_value_type = typename std::decay_t<strides_type>::value_type;
            strides_value_type data_size = 1;
            for (std::size_t i = shape.size(); i != 0; --i)
            {
                strides[i - 1] = data_size;
                data_size = strides[i - 1] * static_cast<strides_value_type>(shape[i - 1]);
                adapt_strides(shape, strides, bs, i - 1);
            }
            return static_cast<std::size_t>(data_size);
        }
    }

    template <class shape_type, class strides_type>
    inline std::size_t compute_strides(const shape_type &shape, strides_type &strides)
    {
        return detail::compute_strides(shape, strides, nullptr);
    }

    inline runtime_shape_t get_default_strides(const runtime_shape_t &shape)
    {
        runtime_shape_t strides(shape.size());
        compute_strides(shape, strides);
        return strides;
    }

    inline size_t compute_size(const runtime_shape_t &shape, const runtime_shape_t &strides)
    {
        size_t max_stride = 0, max_shape = 0;
        for (size_t i = 0; i < shape.size(); i++)
        {
            if ((shape[i] == 1 ? 0 : strides[i]) > max_stride)
            {
                max_stride = strides[i];
                max_shape = shape[i];
            }
        }
        size_t size = max_stride * max_shape;
        return size ? size : 1;
    }

    inline size_t compute_size(const runtime_shape_t &shape)
    {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    }
}

namespace kernels
{
    template <class offset_type, class S, class It>
    inline offset_type element_offset(const S &strides, It first, It last) noexcept
    {
        using difference_type = typename std::iterator_traits<It>::difference_type;
        auto size = static_cast<difference_type>((std::min)(static_cast<typename S::size_type>(std::distance(first, last)), strides.size()));
        return std::inner_product(last - size, last, strides.cend() - size, offset_type(0));
    }

    template <class TShape>
    size_t offset(const TShape &strides, const TShape &index)
    {
        assert(strides.size() == index.size());
        return element_offset<size_t>(strides, index.begin(), index.end());
    }

    namespace detail
    {

        inline size_t get_windowed_output_size(size_t size, int32_t filter, int32_t stride, int32_t dilation, const padding &padding)
        {
            auto effective_filter_size = (filter - 1) * dilation + 1;
            return (size_t)((int32_t)size + padding.before + padding.after - effective_filter_size + stride) / stride;
        }

        template <class T>
        inline T clamp(T value, T min, T max)
        {
            return std::max(std::min(value, max), min);
        }

        template <class T>
        inline T apply_activation(T value, value_range<T> activation)
        {
            return clamp(value, activation.min, activation.max);
        }

    } // namespace detail

} // namespace kernels

}
