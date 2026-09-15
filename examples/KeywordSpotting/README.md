# Keyword spotting on the XIAO MG24 Sense

Say a keyword, read the detected word in a serial console, and watch the LED
flash. The board records audio continuously. Each prediction uses the latest
one-second recording, with 500 ms of overlap between predictions.

You need a **Seeed Studio XIAO MG24 Sense**, a USB data cable, Arduino CLI,
Python 3.10+, and a model from DeepGate. The Sense board has a microphone and
LED. No external sensor wiring is needed.

Download or clone the SDK, then open a terminal in its root directory.
Run all commands below from that directory.

## 1. Install board support

Install [Arduino CLI](https://docs.arduino.cc/arduino-cli/installation/), then
install the board package:

```sh
arduino-cli config add board_manager.additional_urls https://siliconlabs.github.io/arduino/package_arduinosilabs_index.json
arduino-cli core update-index
arduino-cli core install SiliconLabs:silabs@4.0.0
```

The build helper selects the XIAO MG24 with no radio protocol stack:
`SiliconLabs:silabs:xiao_mg24:protocol_stack=none`.

## 2. Add your model

Place these three matching files beside the sketch:

```text
examples/KeywordSpotting/
  KeywordSpotting.ino
  deepgate_model.h     preprocessing, labels, and thresholds
  model.h              compiled model interface
  libmodel.a           compiled model
```

If you already have these files, continue to step 3. Otherwise:

1. Train a `kws-dscnn` model on DeepGate. Download the `.tflite` file and
   `model_spec.json` from that training job.
2. Generate the settings header. Replace the input path with your downloaded spec:

   ```sh
   python3 tools/gen_model_meta.py /path/to/model_spec.json \
     -o examples/KeywordSpotting/deepgate_model.h
   ```

3. Build that model on DeepGate for **Seeed Studio XIAO MG24**. The target
   settings are Cortex-M33, `fpv5-sp-d16`, and hard float.
4. Extract the build download. Copy `hard/include/model.h` and
   `hard/lib/libmodel.a` beside the sketch. The `hard` directory contains the
   files for the selected float ABI.

Keep all three files from the same model. Do not edit generated labels or
thresholds. The generator can also check the source model with
`--tflite /path/to/model.tflite`. Personal model files are ignored by Git.

## 3. Build and upload

Connect the board and list its serial port:

```sh
arduino-cli board list
```

Use the port shown for your board, such as `/dev/cu.usbmodem...` on macOS or
`COM3` on Windows. Close any open serial console before uploading.

```sh
python3 tools/build_arduino.py --port /dev/your-board-port
```

The helper builds and uploads the sketch. It links the model and sets the task
stack to 32 KB for this board. Omit `--port` to compile without uploading.
See [Arduino integration](../../docs/arduino.md) for build details.

## 4. Read detections

Open any serial console on the board's port at **921600 baud**. For example:

```sh
arduino-cli monitor --port /dev/your-board-port --config baudrate=921600
```

Reset the board if you missed the ready message. The message lists the model's
keywords that the demo reports. Say one of those words and look for a detection:

```text
listening for: lights pause
heard lights (78%, threshold 53%)
```

The exact labels and thresholds depend on your model. The LED flashes when the
sketch reports a detection. The serial console needs no Python program.

## Change the action

Find the `Your action` comment in `loop()`. That block turns the LED on when a
word passes its threshold. Change it to perform your own action. At that point,
`DG_MODEL_LABELS[label]` is the detected label and `scores[label]` is its score.

Keep the action short so the main loop can process incoming audio. The LED
turns off near the start of `loop()` without a delay. If you replace the LED
action, update that part too.

## Adjust detection behavior

These settings are at the top of the sketch:

| Setting | Default | Meaning |
| --- | --- | --- |
| `prediction_interval_ms` | 500 ms | New audio needed between predictions. |
| `detection_cooldown_ms` | 1000 ms | Minimum time between reports of any label. Capture and inference continue. |
| `mic_gain` | 5.9 | Gain applied when ADC measurements become int16 audio samples. |

The first prediction waits for a full recording. Each later prediction uses
the latest one second of audio. For example, recordings from 0–1 seconds and
0.5–1.5 seconds overlap. The sketch adds incoming chunks throughout this process.

Intervals round up to a complete audio chunk. A shorter interval needs more
processing time per second. Check for audio warnings when changing it.

A threshold is the minimum score required to report a label. The sketch uses
the winning label's current score and its model threshold. It does not average
scores from earlier predictions. To choose a different threshold preset:

```sh
python3 tools/gen_model_meta.py /path/to/model_spec.json \
  -o examples/KeywordSpotting/deepgate_model.h --preset high_precision
```

Rebuild and upload after changing the header. Available presets come from the
model spec.

Background labels also compete for the highest score. When background wins,
the sketch reports nothing. `DG_MODEL_IS_BACKGROUND` comes from the model spec.
A user class named `unknown` is still a foreground class in the model. This demo
also hides `unknown` from the ready message and skips reports and LED actions
when it wins. It does not select a lower-scoring label. This choice is in
`is_demo_keyword()` in the sketch; it does not change the model's background flags.

## Follow the audio

A **sample** is one microphone measurement. A **chunk** contains 320 samples,
or 20 ms at 16 kHz. A **recording** contains 16,000 samples, or one second.
An MFCC **hop** is the distance between feature windows. The capture chunk and
MFCC hop are the same size in this example.

The main loop follows four steps:

1. Remove the oldest chunk from the recording and add the next chunk.
2. Calculate MFCC features and convert them to int8 model input.
3. Run the model and convert its output to label probabilities.
4. Check the result, print a detection, and perform the action.

`dg_mfcc_clip()` calculates features for each complete recording. Audio capture
continues during that work and inference. The LED timing is approximate because
its update waits for the main loop.

The capture code is below the main loop. TIMER0 triggers the ADC at 16 kHz.
DMA fills two buffers in turn. The callback converts each completed buffer to
int16 audio and writes one full ring-buffer chunk with `reserve()` and `commit()`.
The main loop reads chunks with `peek()` and `release()`.

The ring holds 31 queued chunks, or 620 ms of audio. The 78 MHz clock gives
4,875 timer ticks per sample. Keep the clock and power setup when using this board.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| No board port | Use a USB data cable and run `arduino-cli board list` again. |
| Missing model file or model symbols | Put all three model files beside the sketch and build with the helper. |
| No readable output | Select the board port, set 921600 baud, and reset the board. |
| Audio ring full or microphone overflow | Remove blocking work from `loop()`. Check processing time before increasing the prediction interval. |
| Missed or unwanted detections | Check the microphone level, model labels, and threshold preset with sounds from the intended environment. |

For another board, replace the capture and clock setup, update LED and serial
settings, and compile the model for that board. Keep the audio sample rate and
chunk size matched to the model. See [Arduino integration](../../docs/arduino.md)
and [preprocessing](../../docs/preprocessing.md).
