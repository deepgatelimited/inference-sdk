# Audio preprocessing and model settings

## Supported models

The header generator accepts a version 1 `model_spec.json` with:

- Mono int16 audio.
- Scaling, MFCC calculation, and frame cropping on the device.
- One int8 input with axes `[batch, channel, frame, coefficient]`.
- One int8 output with axes `[batch, class]` and softmax classification.
- Labels, quantization values, and detection thresholds.

The model output contains logits. These are scores before softmax converts
them to probabilities. The generator rejects unsupported output meanings and
checks that tensor sizes match the features and labels.

The default configuration uses 16 kHz audio and a 640-sample periodic Hann
window. Windows advance by 320 samples. A 1024-point FFT supplies 40 HTK mel
bands from 0 to 8 kHz. The DCT produces 10 coefficients per frame.

A one-second recording produces 49 × 10 features. The minimum power value is
80 dB below the recording's maximum. Two reflect-padded edge frames affect
that maximum even though they do not appear in the model input.

## Apply model settings

Generate `deepgate_model.h` from the model spec. The Arduino build helper
applies its settings to both C and C++ sources. For CMake, set
`DEEPGATE_MODEL_HEADER` before adding the SDK. Linked targets inherit the settings.

For another build system, pass the generated `DG_MFCC_*` values as `-D`
definitions to every application and SDK source. These settings control both
calculations and buffer sizes. They must match throughout the build.

`DG_AUDIO_CHUNK_SAMPLES` defaults to `DG_MFCC_HOP`. Each capture callback must
fill one complete chunk. The window must span two hops. The FFT size must be
a power of two and at least as large as the window. A recording must contain
at least two hops. The maximum mel-band count is 254.

Call `dg_mfcc_init()` once before calculating features. An application can call
`dg_mfcc_config_matches(DG_MFCC_CONFIG_ID)` before initialization to check its
settings against the compiled SDK. This check covers MFCC settings, not audio
ring capacity.

## Thresholds and background labels

`DG_MODEL_THRESHOLDS` contains one threshold per output, in model label order.
The example reports the winning label only when its score meets that threshold.
It also checks `DG_MODEL_IS_BACKGROUND`. If background wins, the example does
not report a detection or try the next label.

Background behavior comes from `background_labels` in the spec. It does not
come from the spelling of a label. A user class named `unknown` and a background
class named `_unknown_` are different outputs. The keyword demo also hides the
user `unknown` class from its label list and actions. This is a choice in the
sketch, separate from the model's background flags.

A background label does not need a threshold because the example never reports
it. If its threshold is absent, the generated array contains zero in that slot.
All supplied threshold values are preserved.

Use the spec's default preset, or select another one when generating the header:

```sh
python3 tools/gen_model_meta.py model_spec.json -o deepgate_model.h --preset high_precision
```

`--scheme` selects how thresholds were calculated. Names and defaults come from
the spec. Validation measurements apply to individual recordings. Test detection
behavior with continuous audio from the intended microphone and environment.

## Overlap and processing time

The example adds each incoming chunk to a rolling one-second recording. It
predicts every 500 ms, so recordings overlap by half a second. The detection
cooldown only limits reports. Capture and inference continue during the cooldown.

`dg_mfcc_clip()` recalculates features for each recording. To reuse work between
predictions, retain log-mel frames from `dg_mfcc_log_mel()` and call
`dg_mfcc_finish()`. Supply the maximum of both edge-frame values so the result
uses the same power floor as whole-recording processing.

The ring must hold all chunks that arrive during a single preprocessing and
inference call. A larger ring absorbs short delays. It cannot correct a
processing rate that remains slower than capture. A longer prediction interval
reduces average work but does not shorten a single processing call.

## Memory

The SDK uses caller-owned buffers and no heap allocation. Keep large arrays
and `dg_mfcc_t` in static storage. The model has its own buffers and also needs
stack space. `dg_mfcc_clip()` uses about 8 KB of stack with the default settings.

The MG24 build helper reserves 8192 stack words, or 32 KB. Check memory use for
the model and settings you select.

`DG_AUDIO_RING_SLOTS` includes one reserved writer slot. The default is 32 slots,
which holds 31 queued chunks. Apply any capacity change to both the application
and SDK. The capture interrupt owns the write index. The main loop owns the read
index and must release each chunk after use.
