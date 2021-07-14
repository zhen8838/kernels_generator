#pragma once

#include "hkg/export/HalideBuffer.h"
#include "nncase.h"
#include "prng.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

template <>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<nncase::bfloat16>()
{
    return halide_type_t(halide_type_bfloat, 16);
}

static struct prng_rand_t g_prng_rand_state;
#define SRAND(seed) prng_srand(seed, &g_prng_rand_state)
#define RAND() prng_rand(&g_prng_rand_state)

enum class init_method
{
    rand = 0,
    sequence,
    zero,
};

template <typename T>
static T random_value(float a = -1.2f, float b = 1.2f)
{
    float random = ((float)RAND()) / (float)uint64_t(-1); //RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    T rv;
    if constexpr (std::is_same<T, nncase::bfloat16>::value)
        rv = nncase::bfloat16::round_to_bfloat16(r);
    else
        rv = static_cast<T>(a + r);
    return rv;
}

void traversal_blob(std::function<void(size_t)> f, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        f(i);
    }
}

template <typename T>
void init_blob(T *&m, init_method method, nncase::runtime_shape_t shape)
{
    init_blob<T>(m, method, shape, T(0.));
}

template <typename Iterator, typename dest_t>
void init_blob(Iterator src, dest_t *&dest, nncase::runtime_shape_t shape)
{
    assert(src != nullptr);
    auto size = nncase::runtime::compute_size(shape);
    if (dest == nullptr)
    {
        dest = new dest_t[size];
    }
    auto func = [&src, &dest](size_t i)
    {
        if constexpr (std::is_same_v<dest_t, nncase::bfloat16>)
        {
            dest[i] = nncase::bfloat16::round_to_bfloat16(*src++);
        }
        else
        {
            dest[i] = static_cast<dest_t>(*src++);
        }
    };
    traversal_blob(func, size);
}

template <typename T>
void init_blob(T *&m, init_method method, nncase::runtime_shape_t shape, T base)
{
    size_t size = nncase::runtime::compute_size(shape);
    if (m == nullptr)
    {
        m = new T[size];
    }

    if (method == init_method::rand)
    {
        SRAND((size_t)std::chrono::system_clock::now().time_since_epoch().count());
    }

    auto func = [&m, &method, &base](size_t i)
    {
        T v = static_cast<T>(0);
        switch (method)
        {
        case init_method::rand:
            v = random_value<T>();
            break;
        case init_method::sequence:
            v = static_cast<T>((int)i);
            break;
        case init_method::zero:
            v = static_cast<T>(0);
            break;
        default:
            break;
        };
        m[i] = v + base;
    };
    traversal_blob(func, size);
}

template <typename T>
void write_blob(T *&m, const nncase::runtime_shape_t &shape, std::string &path)
{
    std::ofstream ofs;
    ofs.open(path, std::ios::binary);
    traversal_blob([&m, &ofs](size_t i)
        { ofs << m[i]; },
        nncase::runtime::compute_size(shape));
    ofs.close();
}

template <typename T>
void print_blob(T *&m, const nncase::runtime_shape_t &shape, bool cast = true)
{
    traversal_blob(
        [&m, &shape, &cast](size_t i)
        {
            if (cast)
            {
                std::cout << (float)m[i] << " ";
            }
            else
            {
                std::cout << m[i] << " ";
            }
        },
        nncase::runtime::compute_size(shape));
    std::cout << std::endl;
}

template <typename T1, typename T2>
size_t compare_blob(T1 *&a, T2 *&b, const nncase::runtime_shape_t &shape, float tolerance = .03f, bool print = true, std::ostream &os = std::cout)
{
    size_t count = 0;
    traversal_blob(
        [&a, &b, &count, &print, &os, tolerance](size_t i)
        {
            char str[200];
            if (abs((float)a[i] - (float)b[i]) > tolerance)
            {
                count++;
                sprintf(str, "error in:  %10ld\t%10.5f\t%10.5f\n", i, (float)a[i], (float)b[i]);
                os << str;
            };
        },
        nncase::runtime::compute_size(shape));
    return count;
}

template <typename ref_t, typename T1, typename T2>
size_t compare_blob(ref_t *&ref, T1 *&a, T2 *&b, const nncase::runtime_shape_t &shape, float tolerance = .03f, bool print = true, std::ostream &os = std::cout)
{
    size_t count = 0;
    traversal_blob(
        [&ref, &a, &b, &count, &print, &os, tolerance](size_t i)
        {
            char str[200];
            // NOTE when the diff(a,b) > tolerance , log all
            if (abs((float)a[i] - (float)b[i]) > tolerance)
            {
                count++;
                sprintf(str, "error in:  %10ld\t%10.5f\t%10.5f\t%10.5f\n", i, ref[i], (float)a[i], (float)b[i]);
                os << str;
            };
        },
        nncase::runtime::compute_size(shape));
    return count;
}

enum class range_t
{
    full,
    noneg
};

template <typename NNCASE_T, typename HALIDE_T>
struct Tensor_t
{
    nncase::runtime_shape_t shape;
    float *raw = nullptr;
    NNCASE_T *nraw = nullptr;
    NNCASE_T *hraw = nullptr;
    Halide::Runtime::Buffer<HALIDE_T> hbuf;
    std::string name;
    Tensor_t()
        : raw(nullptr), nraw(nullptr), hraw(nullptr), name("") {};

    Tensor_t(Tensor_t &&other)
        : shape(other.shape), hbuf(other.hbuf), name(other.name)
    {
        if (nraw)
        {
            delete[] nraw;
            delete[] hraw;
            delete[] raw;
            nraw = nullptr;
            hraw = nullptr;
            raw = nullptr;
        }
        raw = std::exchange(other.raw, nullptr);
        nraw = std::exchange(other.nraw, nullptr);
        hraw = std::exchange(other.hraw, nullptr);
    };

    Tensor_t &operator=(Tensor_t &&other) noexcept
    {
        if (this == &other)
            return *this;
        if (nraw)
        {
            delete[] hraw;
            delete[] nraw;
            delete[] raw;
            nraw = nullptr;
            hraw = nullptr;
            raw = nullptr;
            hbuf.deallocate();
        }
        raw = std::exchange(other.raw, nullptr);
        nraw = std::exchange(other.nraw, nullptr);
        hraw = std::exchange(other.hraw, nullptr);
        name = std::exchange(other.name, "");
        shape = other.shape;
        hbuf = other.hbuf;
        return *this;
    }
    Tensor_t(nncase::runtime_shape_t _shape, std::string name = "")
        : shape(_shape), raw(nullptr), nraw(nullptr), hraw(nullptr), name(name) {};

    void allocate(init_method method = init_method::rand)
    {
        init_blob(raw, method, shape);
        init_blob(raw, nraw, shape);
        init_blob(raw, hraw, shape);
        hbuf = Halide::Runtime::Buffer<HALIDE_T>((HALIDE_T *)hraw,
            std::vector<int>(shape.rbegin(), shape.rend()));
    }

    void allocate(std::initializer_list<float> l)
    {
        assert(l.size() == *shape.begin() and shape.size() == 1);
        init_blob(l.begin(), raw, shape);
        init_blob(raw, nraw, shape);
        init_blob(raw, hraw, shape);
        hbuf = Halide::Runtime::Buffer<HALIDE_T>((HALIDE_T *)hraw,
            std::vector<int>(shape.rbegin(), shape.rend()));
    }

    void as_value_range(range_t range_type)
    {
        assert(shape.size() == 1 and shape[0] == 2);
        if (range_type == range_t::full)
        {
            raw[0] = nncase::value_range<float>::full().min;
            raw[1] = nncase::value_range<float>::full().max;
            hraw[0] = nraw[0] = nncase::value_range<NNCASE_T>::full().min;
            hraw[1] = nraw[1] = nncase::value_range<NNCASE_T>::full().max;
        }
    }

    void release_buf()
    {
        if (nraw)
        {
            delete[] nraw;
            delete[] hraw;
            delete[] raw;
            nraw = nullptr;
            hraw = nullptr;
            raw = nullptr;
            hbuf.deallocate();
        }
    }

    ~Tensor_t()
    {
        // std::cout << "release shape: ";
        // for (auto &&i : shape)
        // {
        //     std::cout << i << ",";
        // }
        // std::cout << "; addr = " << raw << " , " << nraw << " , " << hraw << std::endl;
        release_buf();
    }
};

template <typename NNCASE_T, typename HALIDE_T>
struct Scalar_t
{
    NNCASE_T raw, nraw, hraw;
    HALIDE_T hbuf;

    Scalar_t()
    {
    }

    Scalar_t(NNCASE_T value)
        : raw(value), nraw(raw), hraw(raw), hbuf((HALIDE_T)raw)
    {
    }

    Scalar_t &operator=(Scalar_t &&other) noexcept
    {
        raw = other.raw;
        nraw = other.nraw;
        hraw = other.hraw;
        hbuf = other.hbuf;
        return *this;
    }

    Scalar_t(Scalar_t &other)
    {
        raw = other.raw;
        nraw = other.nraw;
        hraw = other.hraw;
        hbuf = other.hbuf;
    }
};

template <typename T1, typename T2>
void print_blob(Tensor_t<T1, T2> &t)
{
    std::cout << t.name << std::endl;
    print_blob(t.nraw, t.shape);
    print_blob(t.hraw, t.shape);
}

template <typename T1, typename T2>
size_t check_error(Tensor_t<T1, T2> &output, float tol = 0.01f)
{
    size_t err_count = 0, total_count = 0;
    err_count += compare_blob(output.nraw, output.hraw, output.shape, tol);
    total_count += nncase::runtime::compute_size(output.shape);

    std::cout << "err :" << err_count << ", total :" << total_count << std::endl;
    return err_count;
}

template <typename T1, typename T2>
size_t check_error_with_float(Tensor_t<T1, T2> &output, float tol = 0.01f)
{
    size_t err_count = 0, total_count = 0;
    err_count += compare_blob(output.raw, output.nraw, output.hraw, output.shape, tol);
    total_count += nncase::runtime::compute_size(output.shape);

    std::cout << "err :" << err_count << ", total :" << total_count << std::endl;
    return err_count;
}

std::tuple<size_t, size_t, nncase::padding, nncase::padding> calc_padded_shape(size_t height, size_t width, size_t k_height, size_t k_width, bool pad_same)
{
    nncase::padding padding_h = nncase::padding::zero(), padding_w = nncase::padding::zero();
    size_t out_h = 0, out_w = 0;
    if (pad_same)
    {
        out_h = int(ceilf(float(height) / float(k_height)));
        out_w = int(ceilf(float(width) / float(k_width)));
        size_t pad_height = std::max(k_height - (height % k_height ? (height % k_height) : k_height), size_t(0));
        size_t pad_width = std::max(k_width - (width % k_width ? (width % k_width) : k_width), size_t(0));
        padding_h.before = pad_height / 2;
        padding_h.after = pad_height - padding_h.before;

        padding_w.before = pad_width / 2;
        padding_w.after = pad_width - padding_w.before;
    }
    return std::make_tuple(out_h, out_w, padding_h, padding_w);
}