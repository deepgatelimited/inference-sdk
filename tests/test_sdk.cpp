#include "KeywordSpotting.ino"
static constexpr int chunks_per_prediction =
    (prediction_interval_samples + chunk_samples - 1) / chunk_samples;
#include "vectors.h"
#include <cstdio>

// The fake model overwrites its input to check buffer ownership.
static int calls, input_changed, fail_run;
static bool check_pipeline_input;
static dg_mfcc_t reference;
static int16_t history[DG_MFCC_CLIP_SAMPLES];
static int mismatches;
static int8_t next_output[MODEL_OUTPUT0_ELEMENTS] = {-100, 100, -100};
static int8_t arena[MODEL_INPUT0_ELEMENTS];
static int8_t first_input[MODEL_INPUT0_ELEMENTS];
int8_t *model_get_input0(void) { return arena; }
const int8_t *model_get_output0(void) { return arena; }
int run_model(void)
{
    if (check_pipeline_input) {
        float expected_features[MODEL_INPUT0_ELEMENTS];
        int8_t expected[MODEL_INPUT0_ELEMENTS];
        dg_mfcc_clip(&reference, history, expected_features);
        dg_quantize_int8(expected_features, MODEL_INPUT0_ELEMENTS,
                         DG_MODEL_INPUT_SCALE, DG_MODEL_INPUT_ZERO_POINT, expected);
        for (int i = 0; i < MODEL_INPUT0_ELEMENTS; i++) {
            if (expected[i] != arena[i]) mismatches++;
        }
    }
    if (calls == 0) memcpy(first_input, arena, sizeof first_input);
    else if (memcmp(first_input, arena, sizeof first_input) != 0) input_changed = 1;
    calls++;
    memset(arena, 0x5a, sizeof arena);
    memcpy(arena, next_output, sizeof next_output);
    return fail_run ? -1 : 0;
}

static void feed_chunks(int count)
{
    int16_t chunk[chunk_samples] = {};
    for (int i = 0; i < count; i++) {
        memcpy(dg_audio_ring_reserve(&audio_ring), chunk, sizeof chunk);
        dg_audio_ring_commit(&audio_ring);
        test_millis += 20;
        loop();
    }
}

static int test_keywords()
{
    setup();
    if (Serial.output.find("listening for: lights\n") == std::string::npos) return 1;
    feed_chunks(DG_MFCC_CLIP_HOPS - 1);
    if (calls) return 1;
    feed_chunks(1);
    if (calls != 1 || test_led != LOW || scores[1] < 0.99f ||
        Serial.output.find("heard lights (100%, threshold 40%)") == std::string::npos) return 1;
    test_millis += 199;
    loop();
    if (test_led != LOW) return 1;
    test_millis++;
    loop();
    if (test_led != HIGH) return 1;

    Serial.output.clear();
    test_millis += 400;  // Let the reporting cooldown expire before background wins.
    next_output[0] = 0;
    next_output[1] = -1;
    feed_chunks(chunks_per_prediction);
    // Background wins normally, even when the runner-up exceeds its own threshold.
    if (calls != 2 || scores[0] <= scores[1] || scores[1] < DG_MODEL_THRESHOLDS[1] ||
        input_changed || !Serial.output.empty() || test_led != HIGH) return 1;
    next_output[0] = -100;
    next_output[1] = 100;
    feed_chunks(chunks_per_prediction);
    if (calls != 3 || test_led != LOW || Serial.output.find("heard lights") == std::string::npos) return 1;
    Serial.output.clear();
    feed_chunks(chunks_per_prediction);
    if (!Serial.output.empty()) return 1;  // Still in the one-second cooldown.

    // A foreground winner below its threshold must not trigger an action.
    test_millis += detection_cooldown_ms;
    next_output[0] = -1;
    next_output[1] = 0;
    next_output[2] = -1;
    feed_chunks(chunks_per_prediction);
    if (calls != 5 || scores[1] <= scores[0] || scores[1] <= scores[2] ||
        scores[1] >= DG_MODEL_THRESHOLDS[1] || !Serial.output.empty() || test_led != HIGH) return 1;

    // The demo hides the user unknown class without changing its background flag.
    next_output[0] = next_output[1] = -100;
    next_output[2] = 100;
    feed_chunks(chunks_per_prediction);
    if (calls != 6 || DG_MODEL_IS_BACKGROUND[2] || scores[2] < DG_MODEL_THRESHOLDS[2] ||
        test_led != HIGH || !Serial.output.empty()) return 1;

    // A hidden result must not start the reporting cooldown.
    next_output[1] = 100;
    next_output[2] = -100;
    feed_chunks(chunks_per_prediction);
    if (calls != 7 || test_led != LOW || Serial.output.find("heard lights") == std::string::npos) return 1;
    Serial.output.clear();

    fail_run = 1;
    feed_chunks(chunks_per_prediction);
    if (input_changed || Serial.output.find("model inference failed") == std::string::npos) return 1;
    // A steady ADC voltage is bias. A voltage change must retain its sign.
    uint32_t adc[chunk_samples];
    int16_t pcm[chunk_samples];
    for (int i = 0; i < chunk_samples; i++) adc[i] = 2048;
    adc_to_pcm(adc, pcm, chunk_samples);
    for (int i = 0; i < chunk_samples; i++) if (pcm[i] != 0) return 1;
    for (int i = 0; i < chunk_samples; i++) adc[i] = 3072;
    adc_to_pcm(adc, pcm, chunk_samples);
    if (pcm[0] <= 0) return 1;
    for (int repeat = 0; repeat < 5; repeat++) adc_to_pcm(adc, pcm, chunk_samples);
    if (abs(pcm[chunk_samples - 1]) > 1) return 1;
    for (int i = 0; i < chunk_samples; i++) adc[i] = 1024;
    adc_to_pcm(adc, pcm, chunk_samples);
    if (pcm[0] >= 0) return 1;
    puts("Model scores, hidden labels, thresholds, cooldown, and LED timing passed");
    return 0;
}

// Audio ring ownership and overflow.
static dg_audio_ring_t ring;

static void write_value(int value)
{
    int16_t *chunk = dg_audio_ring_reserve(&ring);
    for (int i = 0; i < DG_AUDIO_CHUNK_SAMPLES; i++) chunk[i] = (int16_t)value;
    dg_audio_ring_commit(&ring);
}

static int take_value(int value)
{
    const int16_t *chunk = dg_audio_ring_peek(&ring);
    if (!chunk) return 0;
    for (int i = 0; i < DG_AUDIO_CHUNK_SAMPLES; i++) {
        if (chunk[i] != value) return 0;
    }
    dg_audio_ring_release(&ring);
    return 1;
}

static int test_audio()
{
    dg_audio_ring_init(&ring);
    for (int cycle = 0; cycle < 3; cycle++) {
        for (int i = 0; i < DG_AUDIO_RING_SLOTS - 1; i++) write_value(i + 1);
        const int16_t *held = dg_audio_ring_peek(&ring);
        write_value(99);
        if (ring.dropped_chunks != (uint32_t)(cycle + 1) || held[0] != 1) {
            puts("A full ring must drop the new chunk and preserve the oldest chunk");
            return 1;
        }
        for (int i = 0; i < DG_AUDIO_RING_SLOTS - 1; i++) {
            if (!take_value(i + 1)) return 1;
        }
        dg_audio_ring_release(&ring);
        if (dg_audio_ring_peek(&ring)) return 1;
    }

    // The consumer can free a slot while the producer fills its reserved chunk.
    dg_audio_ring_init(&ring);
    for (int i = 0; i < DG_AUDIO_RING_SLOTS - 1; i++) write_value(i + 1);
    int16_t *reserved = dg_audio_ring_reserve(&ring);
    for (int i = 0; i < DG_AUDIO_CHUNK_SAMPLES; i++) reserved[i] = 77;
    if (!take_value(1)) return 1;
    dg_audio_ring_commit(&ring);
    for (int i = 1; i < DG_AUDIO_RING_SLOTS - 1; i++) {
        if (!take_value(i + 1)) return 1;
    }
    if (!take_value(77) || ring.dropped_chunks) return 1;
    puts("Audio ring wraparound, overflow, and ownership passed");
    return 0;
}

// MFCC features against independent torchaudio results.
#define TOLERANCE 0.05f /* Maximum error allowed against the independent reference features. */

static int failures;

static void compare(const char *what, const float *got, const float *want)
{
    float worst = 0.0f;
    int at = 0;
    for (int i = 0; i < DG_MFCC_FRAMES * DG_MFCC_N_COEFFS; i++) {
        float err = fabsf(got[i] - want[i]);
        if (!isfinite(err)) { worst = INFINITY; at = i; break; }
        if (err > worst) { worst = err; at = i; }
    }
    int ok = worst <= TOLERANCE;
    printf("%-28s max |err| = %.4f at frame %d coeff %d  %s\n", what, worst, at / DG_MFCC_N_COEFFS, at % DG_MFCC_N_COEFFS, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

static int test_mfcc()
{
    static dg_mfcc_t mfcc;
    static float features[DG_MFCC_FRAMES * DG_MFCC_N_COEFFS];
    if (!dg_mfcc_config_matches(DG_MFCC_CONFIG_ID) ||
        dg_mfcc_config_matches(DG_MFCC_CONFIG_ID ^ 1u)) return 1;
    dg_mfcc_init(&mfcc);
    for (int v = 0; v < N_VECTORS; v++) {
        dg_mfcc_clip(&mfcc, VECTORS[v].pcm, features);
        char label[64];
        snprintf(label, sizeof label, "clip/%s", VECTORS[v].name);
        compare(label, features, VECTORS[v].mfcc);
    }

    /* Feed one complete chunk at a time. Reuse each hop for the next MFCC frame. */
    static dg_audio_ring_t ring;
    static int16_t audio[DG_MFCC_CLIP_HOPS][DG_MFCC_HOP];
    static float log_mel[DG_MFCC_FRAMES * DG_MFCC_N_MELS];
    for (int v = 0; v < N_VECTORS; v++) {
        dg_audio_ring_init(&ring);
        const int16_t *pcm = VECTORS[v].pcm;
        int fed = 0, hops = 0;
        while (fed < DG_MFCC_CLIP_SAMPLES) {
            memcpy(dg_audio_ring_reserve(&ring), pcm + fed, DG_AUDIO_CHUNK_SAMPLES * sizeof *pcm);
            dg_audio_ring_commit(&ring);
            fed += DG_AUDIO_CHUNK_SAMPLES;
            const int16_t *hop;
            while ((hop = dg_audio_ring_peek(&ring)) != NULL) {
                memcpy(audio[hops], hop, sizeof audio[hops]);
                if (hops > 0) { /* frame ending with this hop = previous hop + this hop */
                    dg_mfcc_log_mel(&mfcc, audio[hops - 1], hop, log_mel + (hops - 1) * DG_MFCC_N_MELS);
                }
                hops++;
                dg_audio_ring_release(&ring);
            }
        }
        char label[64];
        snprintf(label, sizeof label, "stream/%s", VECTORS[v].name);
        if (hops != DG_MFCC_CLIP_HOPS || ring.dropped_chunks) { printf("%-28s %d hops, dropped %u  FAIL\n", label, hops, ring.dropped_chunks); failures++; continue; }
        float edge = dg_mfcc_start_edge_max(&mfcc, audio[0]);
        float end = dg_mfcc_end_edge_max(&mfcc, audio[DG_MFCC_CLIP_HOPS - 2][DG_MFCC_HOP - 1], audio[DG_MFCC_CLIP_HOPS - 1]);
        dg_mfcc_finish(&mfcc, log_mel, 0, edge > end ? edge : end, features);
        compare(label, features, VECTORS[v].mfcc);
    }
    printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}

// Overlapping recordings and model input ownership.
static int test_pipeline()
{
    check_pipeline_input = true;
    dg_mfcc_init(&reference);
    dg_mfcc_init(&mfcc);
    for (int pattern = 0; pattern < 2; pattern++) {
        dg_audio_ring_init(&audio_ring);
        samples_recorded = new_samples = calls = mismatches = 0;
        memset(recording, 0, sizeof recording);
        memset(history, 0, sizeof history);
        for (int h = 0; h < 330; h++) {
            const int16_t *pcm = VECTORS[(h / DG_MFCC_CLIP_HOPS) % N_VECTORS].pcm +
                                 (h % DG_MFCC_CLIP_HOPS) * chunk_samples;
            int16_t boundary[chunk_samples];
            if (pattern) {
                for (int i = 0; i < chunk_samples; i++) boundary[i] = (int16_t)(i % 7 - 3);
                if (h % 47 == 0) boundary[0] = 32767;
                if (h % 37 == 0) boundary[chunk_samples - 1] = -32768;
                pcm = boundary;
            }
            memmove(history, history + chunk_samples, (recording_samples - chunk_samples) * sizeof *history);
            memcpy(history + recording_samples - chunk_samples, pcm, chunk_samples * sizeof *pcm);
            memcpy(dg_audio_ring_reserve(&audio_ring), pcm, chunk_samples * sizeof *pcm);
            dg_audio_ring_commit(&audio_ring);
            int before = calls;
            test_millis += 20;
            loop();
            bool expected = h + 1 >= DG_MFCC_CLIP_HOPS &&
                            (h + 1 - DG_MFCC_CLIP_HOPS) % chunks_per_prediction == 0;
            if (calls - before != expected || memcmp(history, recording, sizeof history)) return 1;
        }
        if (mismatches) {
            fprintf(stderr, "%d quantized feature mismatches\n", mismatches);
            return 1;
        }
        if (calls != (330 - DG_MFCC_CLIP_HOPS) / chunks_per_prediction + 1) return 1;
    }
    puts("Overlapping recordings, prediction timing, and model inputs match whole-clip references");
    return 0;
}

// CTest runs each group in a separate process, with a fresh sketch state.
int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    int result;
    if (strcmp(argv[1], "audio") == 0) result = test_audio();
    else if (strcmp(argv[1], "mfcc") == 0) result = test_mfcc();
    else if (strcmp(argv[1], "keywords") == 0) result = test_keywords();
    else if (strcmp(argv[1], "pipeline") == 0) result = test_pipeline();
    else return 2;
    if (result) fprintf(stderr, "%s tests failed\n", argv[1]);
    return result;
}
