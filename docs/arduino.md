# Arduino build settings

For the XIAO MG24 Sense, follow the [example](../examples/KeywordSpotting/README.md).
`tools/build_arduino.py` builds and uploads that sketch. Use any serial console
to read the board's output after upload.

The helper selects Silicon Labs core settings, links the model archive, and
sets the task stack. These are build settings. The serial console does not
need a Python helper.

## Match the model to the board

Compile the model for the board's CPU, FPU, and float ABI. These must match the
board compiler's `-mcpu`, `-mfpu`, and `-mfloat-abi` flags. To inspect them:

```sh
arduino-cli compile --fqbn YOUR_FQBN --show-properties=expanded YOUR_SKETCH
```

The MG24 example uses Cortex-M33, `fpv5-sp-d16`, and hard float. DeepGate's
download places the files under an ABI directory, such as `hard/include/model.h`
and `hard/lib/libmodel.a`. Keep the compiled header and archive together.

## Silicon Labs core 4.0.0

This core links archives through `build.ld_libs`. The helper places the model
archive before the board's system libraries and preserves the other board
macros. It sets `ARDUINO_MAIN_TASK_STACK_SIZE` to 8192 words, or 32 KB on MG24.

To print the build command without compiling or uploading:

```sh
python3 tools/build_arduino.py --dry-run
```

Keep the library order when setting up a direct build. Inspect
`recipe.c.combine.pattern` in the core's `platform.txt` file. The
[Arduino platform specification](https://docs.arduino.cc/arduino-cli/platform-specification/)
explains these properties.

## Other cores with precompiled library support

A core can use a separate model library if its link recipe includes
`{compiler.libraries.ldflags}`. For example:

```text
DeepGateModel/
  library.properties
  src/
    model.h
    cortex-m33/
      libmodel.a
```

Use the board's `build.mcu` value for the archive directory. Set these properties:

```ini
name=DeepGateModel
version=1.0.0
author=DeepGate
maintainer=DeepGate
sentence=Compiled DeepGate model.
paragraph=Model archive and interface for a specific board.
category=Data Processing
url=https://github.com/deepgatelimited/inference-sdk
architectures=*
precompiled=true
```

Include `<model.h>` and pass both libraries to Arduino CLI:

```sh
arduino-cli compile --fqbn YOUR_FQBN \
  --library /path/to/inference-sdk --library /path/to/DeepGateModel YOUR_SKETCH
```

Cores that supply `build.fpu` and `build.float-abi` can also use archive
subdirectories for those settings. See the
[Arduino library specification](https://docs.arduino.cc/arduino-cli/library-specification/).

## Preprocessing and memory

The SDK and sketch compile separately. Apply the same model preprocessing
settings to both. The MG24 helper does this automatically. See
[preprocessing](preprocessing.md) for other build systems.

Keep large audio buffers in static storage. Check stack use for the model and
preprocessing settings. The MG24 helper reserves 32 KB, but other models can
need a different amount.
