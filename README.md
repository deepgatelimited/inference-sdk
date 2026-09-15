# DeepGate inference SDK

Run a DeepGate audio model on a microcontroller. The SDK buffers microphone
samples, calculates sound features, and converts the model's input and output.

**Start with [keyword spotting on the XIAO MG24 Sense](examples/KeywordSpotting/README.md).**
The example listens continuously, prints detected words, and flashes the board's
LED. It uses overlapping one-second recordings and predicts every 500 ms.

## What you need

- A Seeed Studio XIAO MG24 Sense and a USB data cable for the supplied example.
- Arduino CLI and Python 3.10 or newer to prepare and build the example.
- A trained DeepGate model compiled for the board.
- A serial console to read detections.

Model files are not included in this repository. The example explains how to
prepare them. No Python program is needed to read detections from the board.

## SDK utilities

| Header | Purpose |
| --- | --- |
| `deepgate/dg_audio.h` | Pass complete audio chunks from a capture interrupt to the main loop. |
| `deepgate/dg_mfcc.h` | Calculate MFCC features: the sound measurements expected by the model. |
| `deepgate/dg_tensor.h` | Convert float features to int8 input, then convert int8 output to probabilities. |

The generated `deepgate_model.h` supplies model settings, labels, quantization,
and detection thresholds. Include it in the application. Keep it with the
matching `model.h` and `libmodel.a` from the same training and build process.

The SDK is C11 code with no heap allocation. It needs the C standard library
and math library (`libm`). Application code controls the microphone, prediction
interval, and response to a detection. The supplied microphone code is for the
XIAO MG24 Sense.

## Use the SDK with CMake

```cmake
set(DEEPGATE_MODEL_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/deepgate_model.h")
add_subdirectory(path/to/inference-sdk)
target_link_libraries(your_app PRIVATE deepgate)
```

For another build system, compile `src/deepgate/dg_audio.c` and
`src/deepgate/dg_mfcc.c`, add `src/` to the include path, and link `libm`.
Apply the model's preprocessing settings to both the application and SDK.
See [preprocessing](docs/preprocessing.md) for details.

## Run the tests

Contributors need C11 and C++11 compilers, CMake 3.20+, and Python 3.10+.
Run these commands from the repository root:

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s tests -p 'test_*.py'
```

Test cases are in two files:

- `tests/test_sdk.cpp`: audio buffers, MFCC reference results, and the keyword sketch.
- `tests/test_tools.py`: model metadata and build settings.

The other test files supply reference audio and board/model stubs.
The tests use synthetic audio and a fake model. They need no board, trained
model, or extra Python packages. They check preprocessing, audio buffers, and
sketch behavior. Test microphone capture and recognition on the board separately.

`tests/gen_reference.py` regenerates the saved audio and MFCC results. Only this
maintenance step needs PyTorch and torchaudio. Normal builds and tests do not.

## Reference

- [Arduino integration](docs/arduino.md): build settings and other Arduino cores.
- [Preprocessing](docs/preprocessing.md): model settings, thresholds, and memory.
