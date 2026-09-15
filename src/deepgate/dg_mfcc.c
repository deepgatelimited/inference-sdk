/* Keep preprocessing fast when the application is built for code size. */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("O3")
#endif

#include "dg_mfcc.h"

#include <math.h>
#include <string.h>

#define PI 3.14159265358979323846

static double hz_to_mel(double hz) { return 2595.0 * log10(1.0 + hz / 700.0); }
static double mel_to_hz(double mel) { return 700.0 * (pow(10.0, mel / 2595.0) - 1.0); }

void dg_mfcc_init(dg_mfcc_t *mfcc)
{
    /* A periodic Hann window reduces discontinuities at each frame's edges. */
    for (int sample = 0; sample < DG_MFCC_WIN; sample++) {
        mfcc->hann_window[sample] = (float)(0.5 - 0.5 * cos(2.0 * PI * sample / DG_MFCC_WIN));
    }
    for (int bin = 0; bin < DG_MFCC_N_FFT / 2; bin++) {
        mfcc->fft_twiddle_real[bin] = (float)cos(-2.0 * PI * bin / DG_MFCC_N_FFT);
        mfcc->fft_twiddle_imag[bin] = (float)sin(-2.0 * PI * bin / DG_MFCC_N_FFT);
    }

    /* Triangular filters spaced on the HTK mel scale, from 0 Hz to Nyquist. */
    float mel_points_hz[DG_MFCC_N_MELS + 2];
    double max_mel = hz_to_mel(DG_MFCC_SAMPLE_RATE / 2.0);
    for (int point = 0; point < DG_MFCC_N_MELS + 2; point++) {
        mel_points_hz[point] = (float)mel_to_hz(max_mel * point / (DG_MFCC_N_MELS + 1));
    }
    /* Each FFT bin feeds at most two bands: a rising triangle and the
     * preceding falling triangle. Store their shared weight for each bin. */
    for (int bin = 0, band = 0; bin < DG_MFCC_N_BINS; bin++) {
        float hz = bin * ((float)DG_MFCC_SAMPLE_RATE / DG_MFCC_N_FFT);
        while (band + 1 < DG_MFCC_N_MELS + 2 && hz >= mel_points_hz[band + 1]) band++;
        mfcc->mel_rising_band[bin] = (uint8_t)band;
        mfcc->mel_rising_weight[bin] = band + 1 < DG_MFCC_N_MELS + 2
            ? (hz - mel_points_hz[band]) / (mel_points_hz[band + 1] - mel_points_hz[band])
            : 0.0f;
    }

    /* Orthonormal DCT-II turns log-mel bands into MFCC coefficients. */
    for (int coefficient = 0; coefficient < DG_MFCC_N_COEFFS; coefficient++) {
        double scale = sqrt(2.0 / DG_MFCC_N_MELS) * (coefficient == 0 ? 1.0 / sqrt(2.0) : 1.0);
        for (int band = 0; band < DG_MFCC_N_MELS; band++) {
            mfcc->dct_basis[coefficient * DG_MFCC_N_MELS + band] =
                (float)(scale * cos(PI / DG_MFCC_N_MELS * (band + 0.5) * coefficient));
        }
    }
}

/* In-place radix-2 FFT: reorder samples, then combine successively larger pairs. */
static void fft(dg_mfcc_t *mfcc)
{
    float *real = mfcc->fft_real;
    float *imag = mfcc->fft_imag;
    for (unsigned index = 1, reversed = 0; index < DG_MFCC_N_FFT; index++) {
        unsigned bit = DG_MFCC_N_FFT >> 1;
        for (; reversed & bit; bit >>= 1) reversed ^= bit;
        reversed ^= bit;
        if (index < reversed) {
            float swap = real[index];
            real[index] = real[reversed];
            real[reversed] = swap;
            swap = imag[index];
            imag[index] = imag[reversed];
            imag[reversed] = swap;
        }
    }
    for (unsigned length = 2; length <= DG_MFCC_N_FFT; length <<= 1) {
        unsigned twiddle_step = DG_MFCC_N_FFT / length;
        for (unsigned start = 0; start < DG_MFCC_N_FFT; start += length) {
            for (unsigned offset = 0; offset < length / 2; offset++) {
                float rotation_real = mfcc->fft_twiddle_real[offset * twiddle_step];
                float rotation_imag = mfcc->fft_twiddle_imag[offset * twiddle_step];
                unsigned even = start + offset;
                unsigned odd = even + length / 2;
                float rotated_real = real[odd] * rotation_real - imag[odd] * rotation_imag;
                float rotated_imag = real[odd] * rotation_imag + imag[odd] * rotation_real;
                real[odd] = real[even] - rotated_real;
                imag[odd] = imag[even] - rotated_imag;
                real[even] += rotated_real;
                imag[even] += rotated_imag;
            }
        }
    }
}

/* Convert normalized samples already in fft_real to power in each mel band. */
static void compute_mel_power(dg_mfcc_t *mfcc, float mel_power[DG_MFCC_N_MELS])
{
    for (int sample = 0; sample < DG_MFCC_WIN; sample++) {
        mfcc->fft_real[sample] *= mfcc->hann_window[sample];
    }
    memset(mfcc->fft_real + DG_MFCC_WIN, 0, (DG_MFCC_N_FFT - DG_MFCC_WIN) * sizeof(float));
    memset(mfcc->fft_imag, 0, sizeof mfcc->fft_imag);
    fft(mfcc);
    for (int band = 0; band < DG_MFCC_N_MELS; band++) mel_power[band] = 0.0f;
    for (int bin = 0; bin < DG_MFCC_N_BINS; bin++) {
        float power = mfcc->fft_real[bin] * mfcc->fft_real[bin] + mfcc->fft_imag[bin] * mfcc->fft_imag[bin];
        int band = mfcc->mel_rising_band[bin];
        float weight = mfcc->mel_rising_weight[bin];
        if (band < DG_MFCC_N_MELS) mel_power[band] += power * weight;
        if (band >= 1 && band - 1 < DG_MFCC_N_MELS) mel_power[band - 1] += power * (1.0f - weight);
    }
}

static float power_to_db(float power)
{
    return 10.0f * log10f(power < 1e-10f ? 1e-10f : power);
}

static void normalize_pcm(const int16_t *pcm, float *normalized, int count)
{
    for (int sample = 0; sample < count; sample++) {
        normalized[sample] = (float)pcm[sample] * (1.0f / 32768.0f);
    }
}

void dg_mfcc_log_mel(dg_mfcc_t *mfcc,
                     const int16_t first[DG_MFCC_HOP], const int16_t second[DG_MFCC_HOP],
                     float log_mel[DG_MFCC_N_MELS])
{
    normalize_pcm(first, mfcc->fft_real, DG_MFCC_HOP);
    normalize_pcm(second, mfcc->fft_real + DG_MFCC_HOP, DG_MFCC_HOP);
    compute_mel_power(mfcc, log_mel);
    for (int band = 0; band < DG_MFCC_N_MELS; band++) log_mel[band] = power_to_db(log_mel[band]);
}

static float maximum(const float *values, int count)
{
    float max_value = values[0];
    for (int index = 1; index < count; index++) {
        if (values[index] > max_value) max_value = values[index];
    }
    return max_value;
}

/* Reflect around the first sample: pcm[-i] = pcm[i]. The first window
 * position has zero Hann weight, so it needs no sample from the next hop. */
float dg_mfcc_start_edge_max(dg_mfcc_t *mfcc, const int16_t first[DG_MFCC_HOP])
{
    float *frame = mfcc->fft_real;
    float mel_power[DG_MFCC_N_MELS];
    for (int sample = 0; sample < DG_MFCC_HOP; sample++) {
        frame[sample] = sample == 0 ? 0.0f : (float)first[DG_MFCC_HOP - sample] * (1.0f / 32768.0f);
    }
    normalize_pcm(first, frame + DG_MFCC_HOP, DG_MFCC_HOP);
    compute_mel_power(mfcc, mel_power);
    return power_to_db(maximum(mel_power, DG_MFCC_N_MELS));
}

/* Reflect around the final sample without repeating it: the padded half
 * starts at the second-last sample and ends one sample before the last hop. */
float dg_mfcc_end_edge_max(dg_mfcc_t *mfcc, int16_t before_last,
                           const int16_t last[DG_MFCC_HOP])
{
    float *frame = mfcc->fft_real;
    float mel_power[DG_MFCC_N_MELS];
    normalize_pcm(last, frame, DG_MFCC_HOP);
    for (int sample = 0; sample < DG_MFCC_HOP; sample++) {
        int reflected_index = DG_MFCC_HOP - 2 - sample;
        int16_t reflected_sample = reflected_index >= 0 ? last[reflected_index] : before_last;
        frame[DG_MFCC_HOP + sample] = (float)reflected_sample * (1.0f / 32768.0f);
    }
    compute_mel_power(mfcc, mel_power);
    return power_to_db(maximum(mel_power, DG_MFCC_N_MELS));
}

void dg_mfcc_finish(const dg_mfcc_t *mfcc, const float *log_mel, int first_frame,
                    float edge_max_db, float *features)
{
    /* The cropped edge frames still set the floor for the whole recording. */
    float floor_db = maximum(log_mel, DG_MFCC_FRAMES * DG_MFCC_N_MELS);
    if (edge_max_db > floor_db) floor_db = edge_max_db;
    floor_db -= DG_MFCC_TOP_DB;
    for (int frame = 0; frame < DG_MFCC_FRAMES; frame++) {
        const float *frame_db = log_mel + ((first_frame + frame) % DG_MFCC_FRAMES) * DG_MFCC_N_MELS;
        for (int coefficient = 0; coefficient < DG_MFCC_N_COEFFS; coefficient++) {
            const float *basis = mfcc->dct_basis + coefficient * DG_MFCC_N_MELS;
            float sum = 0.0f;
            for (int band = 0; band < DG_MFCC_N_MELS; band++) {
                float clipped_db = frame_db[band] < floor_db ? floor_db : frame_db[band];
                sum += clipped_db * basis[band];
            }
            features[frame * DG_MFCC_N_COEFFS + coefficient] = sum;
        }
    }
}

void dg_mfcc_clip(dg_mfcc_t *mfcc, const int16_t pcm[DG_MFCC_CLIP_SAMPLES], float *features)
{
    float log_mel[DG_MFCC_FRAMES * DG_MFCC_N_MELS];
    for (int frame = 0; frame < DG_MFCC_FRAMES; frame++) {
        dg_mfcc_log_mel(mfcc, pcm + frame * DG_MFCC_HOP, pcm + (frame + 1) * DG_MFCC_HOP,
                       log_mel + frame * DG_MFCC_N_MELS);
    }
    float start_edge_db = dg_mfcc_start_edge_max(mfcc, pcm);
    float end_edge_db = dg_mfcc_end_edge_max(mfcc, pcm[DG_MFCC_CLIP_SAMPLES - DG_MFCC_HOP - 1],
                                           pcm + DG_MFCC_CLIP_SAMPLES - DG_MFCC_HOP);
    dg_mfcc_finish(mfcc, log_mel, 0, start_edge_db > end_edge_db ? start_edge_db : end_edge_db, features);
}

bool dg_mfcc_config_matches(uint32_t config_id)
{
    return config_id == DG_MFCC_CONFIG_ID;
}
