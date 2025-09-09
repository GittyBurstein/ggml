#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <random>

// ==== helpers: absolute/relative tolerance check ====
static inline bool almost_equal(double a, double b, double abs_tol, double rel_tol) {
    const double diff  = std::abs(a - b);
    const double scale = std::max(std::abs(a), std::abs(b));
    return diff <= std::max(abs_tol, rel_tol * scale);
}

static inline int64_t cpu_arange_count(double start, double stop, double step) {
    const float s  = (float) start;
    const float e  = (float) stop;
    const float st = (float) step;

    if (st <= 0.0f) return 0;
    if (e <= s)     return 0;

    const float span = e - s;
    float n_exact = span / st;

    n_exact = std::nextafterf(n_exact, -INFINITY);

    const int64_t n = (int64_t) std::ceil(n_exact);
    return n;
}

// -------- Expected builders (CPU reference) --------
template<typename T>
static std::vector<T> expected_arange(double start, double stop, double step) {
    const int64_t steps = cpu_arange_count(start, stop, step);

    const float s  = (float) start;
    const float st = (float) step;

    std::vector<T> v;
    v.reserve((size_t) std::max<int64_t>(0, steps));

    for (int64_t i = 0; i < steps; ++i) {
        const float valf = s + st * (float)i;

        if constexpr (std::is_same<T, int8_t>::value) {
            long r = lrint((double)valf);
            if (r < std::numeric_limits<int8_t>::min()) r = std::numeric_limits<int8_t>::min();
            if (r > std::numeric_limits<int8_t>::max()) r = std::numeric_limits<int8_t>::max();
            v.push_back((int8_t) r);
        } else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
            v.push_back(ggml_fp32_to_fp16(valf));
        } else {
            v.push_back((T) valf);
        }
    }
    return v;
}

// -------- Run arange on SYCL (compute in F32), then host-cast to T --------
template<typename T>
static std::vector<T> run_arange(double start, double stop, double step) {
    ggml_init_params ip = { 64 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    assert(ctx);

    ggml_tensor * t_f32 = ggml_arange(ctx, (float)start, (float)stop, (float)step);
    assert(t_f32);

    const int64_t steps = ggml_nelements(t_f32);
    assert(steps >= 0);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_f32);

    ggml_backend_t be = ggml_backend_sycl_init(0);
    assert(be);

    ggml_backend_graph_compute(be, gf);

    std::vector<float> host_f32((size_t)steps);
    if (steps > 0) {
        if (!t_f32->buffer || ggml_backend_buffer_is_host(t_f32->buffer)) {
            std::memcpy(host_f32.data(), t_f32->data, (size_t)steps * sizeof(float));
        } else {
            ggml_backend_tensor_get(t_f32, host_f32.data(), 0, (size_t)steps * sizeof(float));
        }
    }

    std::vector<T> host_T((size_t)steps);
    if constexpr (std::is_same<T, int8_t>::value) {
        for (size_t i = 0; i < (size_t)steps; ++i) {
            long r = lrint(host_f32[i]);
            if (r < std::numeric_limits<int8_t>::min()) r = std::numeric_limits<int8_t>::min();
            if (r > std::numeric_limits<int8_t>::max()) r = std::numeric_limits<int8_t>::max();
            host_T[i] = (int8_t) r;
        }
    } else if constexpr (std::is_same<T, ggml_fp16_t>::value) {
        for (size_t i = 0; i < (size_t)steps; ++i) {
            host_T[i] = ggml_fp32_to_fp16(host_f32[i]);
        }
    } else {
        for (size_t i = 0; i < (size_t)steps; ++i) {
            host_T[i] = (T) host_f32[i];
        }
    }

    ggml_backend_free(be);
    ggml_free(ctx);

    return host_T;
}

// -------- Test harness --------
template<typename T>
static void test_case(const char *name, double start, double stop, double step,
                      double abs_tol, double rel_tol) {
    std::printf("[TEST] %s: start=%g stop=%g step=%g\n", name, start, stop, step);

    auto got = run_arange<T>(start, stop, step);
    auto exp = expected_arange<T>(start, stop, step);

    if (got.size() != exp.size()) {
        std::printf("❌ FAIL (%s): size mismatch got=%zu exp=%zu\n",
                    name, got.size(), exp.size());
        std::exit(1);
    }

    size_t mism = 0;
    for (size_t i = 0; i < exp.size(); ++i) {
        bool bad = false;

        if constexpr (std::is_same<T, ggml_fp16_t>::value) {
            const float g = ggml_fp16_to_fp32(got[i]);
            const float e = ggml_fp16_to_fp32(exp[i]);
            const bool both_nan = std::isnan(g) && std::isnan(e);
            const bool ok = both_nan || almost_equal(g, e, abs_tol, rel_tol);
            bad = !ok;
        } else if constexpr (std::is_same<T, int8_t>::value) {
            bad = (got[i] != exp[i]); // exact for int8
        } else {
            const double g = (double) got[i];
            const double e = (double) exp[i];
            const bool both_nan = std::isnan(g) && std::isnan(e);
            const bool ok = both_nan || almost_equal(g, e, abs_tol, rel_tol);
            bad = !ok;
        }

        if (bad) {
            if (mism < 12) {
                if constexpr (std::is_same<T, ggml_fp16_t>::value) {
                    std::printf("Mismatch @%zu: got=%g exp=%g\n",
                                i,
                                (double)ggml_fp16_to_fp32(got[i]),
                                (double)ggml_fp16_to_fp32(exp[i]));
                } else {
                    std::printf("Mismatch @%zu: got=%g exp=%g\n",
                                i, (double)got[i], (double)exp[i]);
                }
            }
            mism++;
        }
    }

    if (mism) {
        std::printf("❌ FAIL (%s): %zu mismatches out of %zu\n", name, mism, exp.size());
        std::exit(1);
    } else {
        std::printf("✅ OK (%s): all %zu values matched\n", name, exp.size());
    }
}

int main() {
    test_case<float>("F32-basic",    0.0, 5.0, 1.0,   1e-6, 1e-6);
    test_case<float>("F32-decimal",  0.3, 1.3, 0.2,   1e-6, 1e-6);

    test_case<double>("F64-basic",       0.0, 3.0, 1.0,    1e-7, 1e-7);
    test_case<double>("F64-smallstep",   0.0, 1.0, 0.1,    1e-7, 1e-7);

    test_case<ggml_fp16_t>("F16-basic",  0.0, 5.0, 1.0,    1e-3, 1e-3);

    test_case<int8_t>("I8-basic",        -2.0, 3.0, 1.0,   0.0,  0.0);
    test_case<int8_t>("I8-decimal",      -1.5, 2.5, 0.5,   0.0,  0.0);

    test_case<float>("non_divisible_f32", 0.0, 1.0, 0.3,   1e-6, 1e-6);

    test_case<int8_t>("i8_clamp_extremes", -200.0, 200.0, 50.0, 0.0, 0.0);

    test_case<float>("tiny_step_acc", 1.0, 1.0 + 1e-4, 1e-6, 1e-6, 1e-6);

    test_case<float>("long_vector_smoke", -1000.0, 1000.0, 0.5, 1e-6, 1e-6);

    test_case<float>("multidim_contiguous_shape", -1.0, 1.0, 0.125, 1e-6, 1e-6);

    test_case<ggml_fp16_t>("f16_extremes", 3.0e4, 3.0e4 * 5.0, 3.0e4, 1e-2, 1e-3);

    {
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dstart(-10.0f, 10.0f);
        std::uniform_real_distribution<float> dstep(1e-4f, 2.0f); // חיובי
        std::uniform_real_distribution<float> dspan(0.001f, 50.0f);

        int added = 0;
        for (int i = 0; i < 60 && added < 40; ++i) {
            float start = dstart(rng);
            float step  = dstep(rng);
            float stop  = start + dspan(rng);

            int64_t n = cpu_arange_count(start, stop, step);
            if (n == 0 || n > 200000) continue;

            char name[64];
            std::snprintf(name, sizeof(name), "fuzz_%02d", added);
            test_case<float>(name, start, stop, step, 1e-6, 1e-6);
            added++;
        }
    }

    std::printf("ALL TESTS PASSED ✅\n");
    return 0;
}
