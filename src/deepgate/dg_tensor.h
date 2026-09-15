/* Convert float features to int8 input and int8 output to probabilities.
 * Use the scale and zero point from the generated model header. */
#ifndef DG_TENSOR_H
#define DG_TENSOR_H

#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Quantize finite features with a positive scale. Clamp to the int8 range.
 * The default rounding mode rounds halfway values to the nearest even integer. */
static inline void dg_quantize_int8(const float *x, int n, float scale, int zero_point, int8_t *out)
{
    for (int i = 0; i < n; i++) {
        long q = lrintf(x[i] / scale) + zero_point;
        out[i] = (int8_t)(q < -128 ? -128 : q > 127 ? 127 : q);
    }
}

static inline void dg_dequantize_int8(const int8_t *q, int n, float scale, int zero_point, float *out)
{
    for (int i = 0; i < n; i++) out[i] = (q[i] - zero_point) * scale;
}

/* Convert n finite logits to probabilities in place. n must be positive. */
static inline void dg_softmax(float *x, int n)
{
    float max = x[0], sum = 0.0f;
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

#ifdef __cplusplus
}
#endif
#endif
