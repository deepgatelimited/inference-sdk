// Keyword spotting on the XIAO MG24 Sense. Setup: README.md.

#include <Arduino.h>
#include "deepgate_model.h"
#include <string.h>
extern "C" {
#include "model.h"
}

#include "dmadrv.h"
#include "em_cmu.h"
#include "em_gpio.h"
#include "em_iadc.h"
#include "em_prs.h"
#include "em_timer.h"

// A chunk is a short microphone block. A recording holds the model's full input.
static constexpr int chunk_samples = DG_AUDIO_CHUNK_SAMPLES;  // 320 samples = 20 ms at 16 kHz.
static constexpr int recording_samples = DG_MFCC_CLIP_SAMPLES;  // One second for this model.
static constexpr int prediction_interval_ms = 500;
static constexpr int detection_cooldown_ms = 1000;  // Minimum time between reports. Capture and inference continue.
static constexpr float mic_gain = 5.9f;  // ADC counts to int16 PCM.
static constexpr int prediction_interval_samples = DG_MFCC_SAMPLE_RATE * prediction_interval_ms / 1000;

static_assert(recording_samples >= chunk_samples, "Recording must hold at least one audio chunk");
static_assert(prediction_interval_samples > 0, "Prediction interval must be positive");
static_assert(MODEL_INPUT0_ELEMENTS == DG_MFCC_FRAMES * DG_MFCC_N_COEFFS, "Model input size must match MFCC features");
static_assert(MODEL_OUTPUT0_ELEMENTS == DG_MODEL_N_LABELS, "Model output size must match labels");

// Keep the recording and processing buffers off the task stack.
static dg_audio_ring_t audio_ring;
static dg_mfcc_t mfcc;
static int16_t recording[recording_samples];
static float features[MODEL_INPUT0_ELEMENTS];
static float scores[MODEL_OUTPUT0_ELEMENTS];
static int samples_recorded = 0;
static int new_samples = 0;
static volatile uint32_t adc_overflows = 0;

static bool update_recording();
static const char *start_microphone();

static bool is_demo_keyword(int label)
{
    // Hide unknown words in this demo. Keep the model's background flags unchanged.
    return !DG_MODEL_IS_BACKGROUND[label] && strcmp(DG_MODEL_LABELS[label], "unknown") != 0;
}

void setup()
{
    // Run the CPU and sample timer from the same 78 MHz clock.
    setCPUClock(CPU_78MHZ);
    CMU_ClockSelectSet(cmuClock_EM01GRPACLK, cmuSelect_HFRCODPLL);
    // Keep peripheral clocks running when the Arduino core idles.
    sl_power_manager_init();
    sl_power_manager_add_em_requirement(SL_POWER_MANAGER_EM1);

    // Connect the serial monitor and leave the active-low LED off.
    Serial.begin(921600);
    delay(1000);  // Give the serial monitor time to connect.
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);

    // Prepare audio processing, then start continuous microphone capture.
    dg_audio_ring_init(&audio_ring);
    dg_mfcc_init(&mfcc);
    const char *error = start_microphone();
    if (error) {
        Serial.println(error);
        while (true) {}
    }
    Serial.print("listening for:");
    for (int i = 0; i < DG_MODEL_N_LABELS; i++) {
        if (!is_demo_keyword(i)) continue;
        Serial.print(' ');
        Serial.print(DG_MODEL_LABELS[i]);
    }
    Serial.println();
}

void loop()
{
    static bool led_on = false;
    static uint32_t led_on_at_ms = 0;
    static uint32_t last_detection_ms = millis() - detection_cooldown_ms;
    static uint32_t last_dropped_chunks = 0;
    static uint32_t last_overflows = 0;

    // Turn the LED off after a short flash without pausing audio capture.
    if (led_on && millis() - led_on_at_ms >= 200) {
        digitalWrite(LED_BUILTIN, HIGH);
        led_on = false;
    }
    if (audio_ring.dropped_chunks != last_dropped_chunks) {
        last_dropped_chunks = audio_ring.dropped_chunks;
        Serial.println("warning: audio ring buffer full; processing cannot keep up");
    }
    if (adc_overflows != last_overflows) {
        last_overflows = adc_overflows;
        Serial.println("warning: microphone overflow");
    }

    // 1. Add the next audio chunk to the recording. Wait until a prediction is due.
    if (!update_recording()) return;

    // 2. Convert the recording into MFCC features, then into the model's int8 input.
    dg_mfcc_clip(&mfcc, recording, features);
    dg_quantize_int8(features, MODEL_INPUT0_ELEMENTS,
                     DG_MODEL_INPUT_SCALE, DG_MODEL_INPUT_ZERO_POINT, model_get_input0());

    // 3. Run the model and convert its output into a probability for each label.
    if (run_model() != 0) {
        Serial.println("model inference failed");
        return;
    }
    dg_dequantize_int8(model_get_output0(), MODEL_OUTPUT0_ELEMENTS,
                       DG_MODEL_OUTPUT_SCALE, DG_MODEL_OUTPUT_ZERO_POINT, scores);
    dg_softmax(scores, MODEL_OUTPUT0_ELEMENTS);
    int label = 0;
    for (int i = 1; i < MODEL_OUTPUT0_ELEMENTS; i++) {
        if (scores[i] > scores[label]) label = i;
    }

    // Report only demo keywords. Do not select a lower-scoring label.
    if (!is_demo_keyword(label)) return;

    // 4. Check the model's threshold, report the keyword, and flash the LED.
    if (scores[label] >= DG_MODEL_THRESHOLDS[label] &&
        millis() - last_detection_ms >= detection_cooldown_ms) {
        last_detection_ms = millis();
        Serial.print("heard ");
        Serial.print(DG_MODEL_LABELS[label]);
        Serial.print(" (");
        Serial.print((int)(scores[label] * 100));
        Serial.print("%, threshold ");
        Serial.print((int)(DG_MODEL_THRESHOLDS[label] * 100));
        Serial.println("%)");
        // Your action: change this block to respond to the detected word.
        digitalWrite(LED_BUILTIN, LOW);
        led_on = true;
        led_on_at_ms = millis();
    }
}

// Add each new chunk to the rolling recording. Consecutive predictions overlap.
static bool update_recording()
{
    const int16_t *chunk = dg_audio_ring_peek(&audio_ring);
    if (!chunk) return false;
    memmove(recording, recording + chunk_samples,
            (recording_samples - chunk_samples) * sizeof recording[0]);
    memcpy(recording + recording_samples - chunk_samples, chunk, chunk_samples * sizeof chunk[0]);
    dg_audio_ring_release(&audio_ring);

    if (samples_recorded < recording_samples) samples_recorded += chunk_samples;
    new_samples += chunk_samples;
    if (samples_recorded < recording_samples || new_samples < prediction_interval_samples) return false;
    new_samples = 0;
    return true;
}

// TIMER0 triggers the ADC. DMA alternates buffers while inference runs.
static uint32_t adc_buffers[2][chunk_samples];
static LDMA_Descriptor_t adc_descriptors[2];
static unsigned int adc_dma_channel;
static volatile unsigned int completed_buffers = 0;

static float microphone_bias = 0;
static bool bias_initialized = false;

// Subtract the microphone's steady voltage bias, apply gain, and clamp to int16 PCM.
static void adc_to_pcm(const uint32_t *adc_samples, int16_t *pcm, int sample_count)
{
    for (int i = 0; i < sample_count; i++) {
        float sample = adc_samples[i] & 0xfff;  // ADC measurements are 12-bit values.
        if (!bias_initialized) {
            microphone_bias = sample;
            bias_initialized = true;
        }
        // Track slow bias drift while leaving rapid sound changes in the signal.
        microphone_bias += (sample - microphone_bias) / 128.0f;
        int32_t value = (int32_t)lroundf((sample - microphone_bias) * mic_gain);
        if (value > 32767) value = 32767;
        if (value < -32768) value = -32768;
        pcm[i] = (int16_t)value;
    }
}

// Publish the completed DMA buffer to the audio ring buffer from the capture interrupt.
static bool on_samples_ready(unsigned int, unsigned int, void *)
{
    int16_t *pcm = dg_audio_ring_reserve(&audio_ring);
    adc_to_pcm(adc_buffers[completed_buffers++ % 2], pcm, chunk_samples);
    dg_audio_ring_commit(&audio_ring);
    if (IADC_getInt(IADC0) & IADC_IF_SCANFIFOOF) {
        adc_overflows++;
        IADC_clearInt(IADC0, IADC_IF_SCANFIFOOF);
    }
    return true;
}

// Power the microphone on PC8 and capture PC9 at the model's sample rate.
static const char *start_microphone()
{
    pinMode(PC9, INPUT);
    pinMode(PC8, OUTPUT);
    digitalWrite(PC8, HIGH);
    delay(200);

    CMU_ClockEnable(cmuClock_IADC0, true);
    CMU_ClockEnable(cmuClock_GPIO, true);
    CMU_ClockEnable(cmuClock_TIMER0, true);
    CMU_ClockEnable(cmuClock_PRS, true);
    CMU_ClockSelectSet(cmuClock_IADCCLK, cmuSelect_EM01GRPACLK);

    IADC_reset(IADC0);
    IADC_Init_t adc_init = IADC_INIT_DEFAULT;
    IADC_AllConfigs_t adc_configs = IADC_ALLCONFIGS_DEFAULT;
    adc_init.warmup = iadcWarmupKeepWarm;
    adc_init.srcClkPrescale = IADC_calcSrcClkPrescale(IADC0, 20000000, 0);
    adc_configs.configs[0].reference = iadcCfgReferenceInt1V2;
    adc_configs.configs[0].vRef = 1210;
    adc_configs.configs[0].analogGain = iadcCfgAnalogGain1x;
    adc_configs.configs[0].osrHighSpeed = iadcCfgOsrHighSpeed32x;
    adc_configs.configs[0].adcClkPrescale =
        IADC_calcAdcClkPrescale(IADC0, 2500000, 0, iadcCfgModeNormal, adc_init.srcClkPrescale);
    IADC_init(IADC0, &adc_init, &adc_configs);

    GPIO->CDBUSALLOC |= GPIO_CDBUSALLOC_CDODD0_ADC0;
    IADC_InitScan_t scan_init = IADC_INITSCAN_DEFAULT;
    IADC_ScanTable_t scan_table = IADC_SCANTABLE_DEFAULT;
    scan_init.triggerSelect = iadcTriggerSelPrs0PosEdge;
    scan_init.triggerAction = iadcTriggerActionOnce;
    scan_init.dataValidLevel = iadcFifoCfgDvl1;
    scan_init.fifoDmaWakeup = true;
    scan_table.entries[0].posInput = iadcPosInputPortCPin9;
    scan_table.entries[0].negInput = iadcNegInputGnd;
    scan_table.entries[0].includeInScan = true;
    IADC_initScan(IADC0, &scan_init, &scan_table);

    // Each DMA descriptor fills one buffer and links to the other, so
    // conversion of one chunk overlaps capture of the next.
    DMADRV_Init();
    if (DMADRV_AllocateChannel(&adc_dma_channel, nullptr) != ECODE_EMDRV_DMADRV_OK) {
        return "microphone: no DMA channel";
    }
    adc_descriptors[0] =
        (LDMA_Descriptor_t)LDMA_DESCRIPTOR_LINKREL_P2M_WORD(&IADC0->SCANFIFODATA, adc_buffers[0], chunk_samples, 1);
    adc_descriptors[1] =
        (LDMA_Descriptor_t)LDMA_DESCRIPTOR_LINKREL_P2M_WORD(&IADC0->SCANFIFODATA, adc_buffers[1], chunk_samples, -1);
    LDMA_TransferCfg_t adc_transfer = LDMA_TRANSFER_CFG_PERIPHERAL(ldmaPeripheralSignal_IADC0_IADC_SCAN);
    if (DMADRV_LdmaStartTransfer(adc_dma_channel, &adc_transfer, adc_descriptors, on_samples_ready, nullptr) !=
        ECODE_EMDRV_DMADRV_OK) {
        return "microphone: DMA transfer refused";
    }
    IADC_command(IADC0, iadcCmdStartScan);

    // One timer overflow triggers one sample: 78 MHz / 16 kHz = 4875 ticks.
    const uint32_t timer_hz = CMU_ClockFreqGet(cmuClock_TIMER0);
    if (timer_hz % DG_MFCC_SAMPLE_RATE != 0) {
        return "microphone: timer clock must divide evenly by the model sample rate";
    }
    TIMER_Init_TypeDef timer_init = TIMER_INIT_DEFAULT;
    timer_init.enable = false;
    TIMER_Init(TIMER0, &timer_init);
    TIMER_TopSet(TIMER0, timer_hz / DG_MFCC_SAMPLE_RATE - 1);
    PRS_ConnectSignal(0, prsTypeAsync, prsSignalTIMER0_OF);
    PRS_ConnectConsumer(0, prsTypeAsync, prsConsumerIADC0_SCANTRIGGER);
    TIMER_Enable(TIMER0, true);
    return nullptr;
}
