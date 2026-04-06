
#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <math.h>
#include <stddef.h>

#ifndef ACTIVATIONS_SQRT3
#define ACTIVATIONS_SQRT3 1.7320508075688772935274463415059
#endif
#ifndef ACTIVATIONS_PI
#define ACTIVATIONS_PI 3.1415926535897932384626433832795
#endif
#ifndef ACTIVATIONS_SILULIKE_C
#define ACTIVATIONS_SILULIKE_C 0.004409
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Numerically stable sigmoid for scalars (matches typical torch.sigmoid behavior). */
static inline double activations_sigmoid(double x)
{
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    }
    const double z = exp(x);
    return z / (1.0 + z);
}

static inline float activations_sigmoidf(float x)
{
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    }
    const float z = expf(x);
    return z / (1.0f + z);
}

/* CDF of Student's t with 3 degrees of freedom. */
static inline double student_t3_cdf(double x)
{
    const double sx = x * x + 3.0;
    return 0.5 + (1.0 / ACTIVATIONS_PI) * (atan(x / ACTIVATIONS_SQRT3) + (ACTIVATIONS_SQRT3 * x) / sx);
}

static inline float student_t3_cdff(float x)
{
    const float sx = x * x + 3.0f;
    return 0.5f + (1.0f / (float)ACTIVATIONS_PI) * (atanf(x / (float)ACTIVATIONS_SQRT3) + ((float)ACTIVATIONS_SQRT3 * x) / sx);
}

static inline double tgelu3(double x)
{
    return x * student_t3_cdf(x);
}

static inline float tgelu3f(float x)
{
    return x * student_t3_cdff(x);
}

/* siluLike from func_025.py / func_05.py */
static inline double silu_like(double x)
{
    const double x2 = x * x;
    const double part2 = 1.0 + ACTIVATIONS_SILULIKE_C * log1p(x2);
    return x * activations_sigmoid(x) * part2;
}

static inline float silu_likef(float x)
{
    const float x2 = x * x;
    const float part2 = 1.0f + (float)ACTIVATIONS_SILULIKE_C * log1pf(x2);
    return x * activations_sigmoidf(x) * part2;
}

/* hybrid_0.25: 0.25 * (siluLike(x) + tgelu3(x)) */
static inline double silu_tgelu3_hybrid_like_025(double x)
{
    return 0.25 * (silu_like(x) + tgelu3(x));
}

static inline float silu_tgelu3_hybrid_like_025f(float x)
{
    return 0.25f * (silu_likef(x) + tgelu3f(x));
}

/* hybrid_0.5: 0.5 * (siluLike(x) + tgelu3(x)) */
static inline double silu_tgelu3_hybrid_like_05(double x)
{
    return 0.5 * (silu_like(x) + tgelu3(x));
}

static inline float silu_tgelu3_hybrid_like_05f(float x)
{
    return 0.5f * (silu_likef(x) + tgelu3f(x));
}

/* --- Contiguous 1D buffers (typical: tensor.data_ptr, tensor.numel when is_contiguous) --- */

static inline void tgelu3_inplace_d(double *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = tgelu3(x[i]);
    }
}

static inline void tgelu3_inplace_f(float *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = tgelu3f(x[i]);
    }
}

static inline void tgelu3_apply_d(const double *src, double *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        dst[i] = tgelu3(src[i]);
    }
}

static inline void tgelu3_apply_f(const float *src, float *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        dst[i] = tgelu3f(src[i]);
    }
}

static inline void silu_tgelu3_hybrid_like_025_inplace_d(double *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = silu_tgelu3_hybrid_like_025(x[i]);
    }
}

static inline void silu_tgelu3_hybrid_like_025_inplace_f(float *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = silu_tgelu3_hybrid_like_025f(x[i]);
    }
}

static inline void silu_tgelu3_hybrid_like_05_inplace_d(double *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = silu_tgelu3_hybrid_like_05(x[i]);
    }
}

static inline void silu_tgelu3_hybrid_like_05_inplace_f(float *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = silu_tgelu3_hybrid_like_05f(x[i]);
    }
}

/*
 * Arbitrary byte stride between elements (e.g. numpy/PyTorch stride in bytes).
 * data must point at the first element; n = number of elements to update.
 */
static inline void tgelu3_inplace_strided_f(float *data, size_t n, ptrdiff_t byte_stride)
{
    unsigned char *p = (unsigned char *)data;
    for (size_t i = 0; i < n; ++i) {
        float *fp = (float *)(p + (ptrdiff_t)i * byte_stride);
        *fp = tgelu3f(*fp);
    }
}

static inline void silu_tgelu3_hybrid_like_025_inplace_strided_f(float *data, size_t n, ptrdiff_t byte_stride)
{
    unsigned char *p = (unsigned char *)data;
    for (size_t i = 0; i < n; ++i) {
        float *fp = (float *)(p + (ptrdiff_t)i * byte_stride);
        *fp = silu_tgelu3_hybrid_like_025f(*fp);
    }
}

static inline void silu_tgelu3_hybrid_like_05_inplace_strided_f(float *data, size_t n, ptrdiff_t byte_stride)
{
    unsigned char *p = (unsigned char *)data;
    for (size_t i = 0; i < n; ++i) {
        float *fp = (float *)(p + (ptrdiff_t)i * byte_stride);
        *fp = silu_tgelu3_hybrid_like_05f(*fp);
    }
}

#ifdef __cplusplus
}
#endif

#endif 
