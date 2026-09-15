/* MFCC features for complete recordings or streaming audio.
 * Configuration and preprocessing details: docs/preprocessing.md. */
#ifndef DG_MFCC_H
#define DG_MFCC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

/* Apply the generated model settings to both the application and SDK. */
#ifndef DG_MFCC_SAMPLE_RATE
#define DG_MFCC_SAMPLE_RATE 16000
#endif
#ifndef DG_MFCC_N_FFT
#define DG_MFCC_N_FFT 1024
#endif
#define DG_MFCC_N_BINS (DG_MFCC_N_FFT / 2 + 1)
#ifndef DG_MFCC_WIN
#define DG_MFCC_WIN 640
#endif
#ifndef DG_MFCC_HOP
#define DG_MFCC_HOP 320
#endif
#ifndef DG_MFCC_N_MELS
#define DG_MFCC_N_MELS 40
#endif
#ifndef DG_MFCC_N_COEFFS
#define DG_MFCC_N_COEFFS 10
#endif
#ifndef DG_MFCC_CLIP_HOPS
#define DG_MFCC_CLIP_HOPS 50
#endif
#define DG_MFCC_CLIP_SAMPLES (DG_MFCC_CLIP_HOPS * DG_MFCC_HOP)
#define DG_MFCC_FRAMES (DG_MFCC_CLIP_HOPS - 1) /* excludes the two padded edge frames */
#ifndef DG_MFCC_TOP_DB
#define DG_MFCC_TOP_DB 80.0f
#endif

#if DG_MFCC_HOP < 2 || DG_MFCC_WIN != 2 * DG_MFCC_HOP
#error "MFCC requires a window of two hops, with at least two samples per hop"
#endif
#if DG_MFCC_N_FFT < DG_MFCC_WIN || (DG_MFCC_N_FFT & (DG_MFCC_N_FFT - 1)) != 0
#error "MFCC FFT size must be a power of two at least as large as the window"
#endif
#if DG_MFCC_SAMPLE_RATE < 1 || DG_MFCC_CLIP_HOPS < 2 || DG_MFCC_N_MELS < 1 || DG_MFCC_N_MELS > 254 || DG_MFCC_N_COEFFS < 1 || DG_MFCC_N_COEFFS > DG_MFCC_N_MELS
#error "Invalid MFCC sample rate, clip length, mel count, or coefficient count"
#endif

typedef struct {
    /* Precomputed analysis tables. */
    float hann_window[DG_MFCC_WIN];
    float fft_twiddle_real[DG_MFCC_N_FFT / 2];
    float fft_twiddle_imag[DG_MFCC_N_FFT / 2];
    float mel_rising_weight[DG_MFCC_N_BINS];
    uint8_t mel_rising_band[DG_MFCC_N_BINS];
    float dct_basis[DG_MFCC_N_COEFFS * DG_MFCC_N_MELS];
    /* Reused for windowed samples and the in-place FFT. */
    float fft_real[DG_MFCC_N_FFT];
    float fft_imag[DG_MFCC_N_FFT];
} dg_mfcc_t;

/* Initialize the window, FFT, mel, and DCT tables.
 * Build the application and SDK with the same DG_MFCC_* definitions. */
void dg_mfcc_init(dg_mfcc_t *mfcc);

/* Convert DG_MFCC_CLIP_SAMPLES PCM samples to
 * DG_MFCC_FRAMES * DG_MFCC_N_COEFFS features in chronological frame order.
 * Uses about 8 KB of stack at the default configuration. */
void dg_mfcc_clip(dg_mfcc_t *mfcc, const int16_t pcm[DG_MFCC_CLIP_SAMPLES], float *features);

/* Check that the application and SDK use the same MFCC settings. */
#define DG_MFCC_CONFIG_ID                                                                   \
    ((uint32_t)(DG_MFCC_SAMPLE_RATE) * 31u ^ (uint32_t)(DG_MFCC_N_FFT) * 131u ^             \
     (uint32_t)(DG_MFCC_WIN) * 1031u ^ (uint32_t)(DG_MFCC_HOP) * 7u ^                       \
     (uint32_t)(DG_MFCC_N_MELS) * 10007u ^ (uint32_t)(DG_MFCC_N_COEFFS) * 100003u ^         \
     (uint32_t)(DG_MFCC_CLIP_HOPS) * 1000003u ^ (uint32_t)((DG_MFCC_TOP_DB) * 10.0f) * 3u)
bool dg_mfcc_config_matches(uint32_t config_id);

/* Reuse log-mel frames between predictions with these streaming functions.
 * Call them from the main loop. They share the scratch buffers in dg_mfcc_t. */

/* Log-mel (dB, before the clip floor) of two consecutive audio hops. Samples are
 * int16 PCM, scaled to [-1, 1) before analysis. */
void dg_mfcc_log_mel(dg_mfcc_t *mfcc,
                     const int16_t first[DG_MFCC_HOP], const int16_t second[DG_MFCC_HOP],
                     float log_mel[DG_MFCC_N_MELS]);

/* Reflect-padded edge frames are cropped from the model input, but their
 * maxima still contribute to the clip-relative dB floor. `first` is the
 * first hop; `last` is the final hop and `before_last` the preceding sample. */
float dg_mfcc_start_edge_max(dg_mfcc_t *mfcc, const int16_t first[DG_MFCC_HOP]);
float dg_mfcc_end_edge_max(dg_mfcc_t *mfcc, int16_t before_last,
                           const int16_t last[DG_MFCC_HOP]);

/* Apply the clip-relative dB floor and DCT. log_mel is a ring of
 * DG_MFCC_FRAMES rows, each DG_MFCC_N_MELS wide. first_frame is the oldest
 * row (0 <= first_frame < DG_MFCC_FRAMES). Output rows are chronological,
 * each containing DG_MFCC_N_COEFFS values. */
void dg_mfcc_finish(const dg_mfcc_t *mfcc, const float *log_mel, int first_frame,
                    float edge_max_db, float *features);

#ifdef __cplusplus
}
#endif

#endif
