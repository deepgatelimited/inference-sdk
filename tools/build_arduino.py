#!/usr/bin/env python3
"""Build the XIAO MG24 Sense keyword-spotting example with Arduino CLI."""

import argparse
import pathlib
import re
import shlex
import subprocess
import sys

FQBN = "SiliconLabs:silabs:xiao_mg24:protocol_stack=none"
ROOT = pathlib.Path(__file__).resolve().parent.parent


MODEL_DEFINES = (
    "DG_MFCC_SAMPLE_RATE", "DG_MFCC_N_FFT", "DG_MFCC_WIN", "DG_MFCC_HOP",
    "DG_MFCC_N_MELS", "DG_MFCC_N_COEFFS", "DG_MFCC_CLIP_HOPS", "DG_MFCC_TOP_DB",
)


def model_definitions(header: pathlib.Path) -> list[str]:
    values = dict(re.findall(r"^#define (DG_MFCC_\w+) ([0-9]+(?:\.[0-9]+)?f?)$",
                             header.read_text(), re.MULTILINE))
    missing = set(MODEL_DEFINES) - values.keys()
    if missing:
        raise ValueError(f"{header}: regenerate model metadata; missing {', '.join(sorted(missing))}")
    return [f"{name}={values[name]}" for name in MODEL_DEFINES]


def build_command(cli: str, sketch: pathlib.Path, properties: dict[str, str],
                  port: str | None = None) -> list[str]:
    macros = properties.get("build.board_specific_macros", "")
    libraries = properties.get("build.ld_libs", "")
    if not libraries or "-DARDUINO_MAIN_TASK_STACK_SIZE=" not in macros:
        raise ValueError("unsupported board properties; install SiliconLabs:silabs@4.0.0")
    macros = re.sub(r"(-DARDUINO_MAIN_TASK_STACK_SIZE=)\d+", r"\g<1>8192", macros)
    archive = sketch / "libmodel.a"
    cmd = [cli, "compile", "--fqbn", FQBN, "--library", str(ROOT),
           "--build-property", f'build.ld_libs="{archive}" {libraries}',
           "--build-property", f"build.board_specific_macros={macros}"]
    definitions = " ".join("-D" + value for value in model_definitions(sketch / "deepgate_model.h"))
    for language in ("c", "cpp"):
        key = f"compiler.{language}.extra_flags"
        cmd += ["--build-property", f"{key}={properties.get(key, '')} {definitions}".strip()]
    if port:
        cmd += ["--upload", "--port", port]
    return cmd + [str(sketch)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sketch", type=pathlib.Path, default=ROOT / "examples/KeywordSpotting")
    parser.add_argument("--build-path", type=pathlib.Path, help="directory for build output")
    parser.add_argument("--port", help="compile and upload to this port (omit to compile only)")
    parser.add_argument("--cli", default="arduino-cli", help="Arduino CLI executable")
    parser.add_argument("--dry-run", action="store_true", help="inspect board properties and print the build command")
    args = parser.parse_args()
    sketch = args.sketch.expanduser().resolve()
    missing = [name for name in (sketch.name + ".ino", "deepgate_model.h", "model.h", "libmodel.a")
               if not (sketch / name).is_file()]
    if missing:
        parser.error(f"missing in {sketch}: {', '.join(missing)}; see examples/KeywordSpotting/README.md")
    build_path = ["--build-path", str(args.build_path.expanduser().resolve())] if args.build_path else []
    try:
        result = subprocess.run([args.cli, "compile", "--fqbn", FQBN,
                                 "--show-properties=expanded", *build_path, str(sketch)],
                                check=True, capture_output=True, text=True)
        properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        cmd = build_command(args.cli, sketch, properties, args.port)
        cmd[-1:-1] = build_path
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        return exc.returncode
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
