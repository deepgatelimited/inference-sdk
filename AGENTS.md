# Agent guide

Start with [README.md](README.md) and the
[Arduino example](examples/KeywordSpotting/README.md).

- Generate `deepgate_model.h` from the training job's `model_spec.json` using
  `tools/gen_model_meta.py`. The specification supplies preprocessing,
  labels, quantization, and thresholds.
- Build the model for the board's CPU, FPU, and float ABI. Place `model.h`,
  `libmodel.a`, and `deepgate_model.h` beside the sketch.
- Use `tools/build_arduino.py`, or set CMake `DEEPGATE_MODEL_HEADER`, to apply
  generated preprocessing settings consistently across the application and SDK.
- The Arduino example is a single sketch. For another board, replace its
  microphone capture and clock setup, and update LED polarity and serial settings.
- The capture interrupt fills complete audio chunks with reserve/commit. The
  main loop reads them with peek/release and updates a rolling recording.
- Prediction interval, detection cooldown, and microphone gain are at the top
  of the sketch. The default predictions use overlapping recordings every
  500 ms. Cooldown limits reports, not capture or inference.
- Use generated constants for quantization and thresholds. Only the model's
  background flags decide which labels are background.
- The portable C runtime is in `src/deepgate/`.

[Host tests](README.md#run-the-tests) verify preprocessing and integration.
