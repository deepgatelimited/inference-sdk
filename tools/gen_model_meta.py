#!/usr/bin/env python3
"""Generate deepgate_model.h from a DeepGate model_spec.json (Python 3.10+).

The header supplies preprocessing settings, labels, quantization, and thresholds.
Use --preset to select detection behavior. --scheme selects how thresholds were calculated.
"""

import argparse
import hashlib
import json
import math
import pathlib
import sys

SPEC_VERSION = 1
# What dg_mfcc implements: torchaudio.transforms.MFCC with these fixed choices.
MFCC_FIXED = {
    "window": "hann_periodic",
    "center": True,
    "pad_mode": "reflect",
    "power": 2.0,
    "f_min": 0.0,
    "mel_scale": "htk",
    "mel_norm": None,
    "log": "amplitude_to_db",
    "top_db": 80.0,
    "dct_norm": "ortho",
}
MFCC_DEFINES = {  # spec param -> DG_MFCC_* macro
    "n_fft": "DG_MFCC_N_FFT",
    "win_length": "DG_MFCC_WIN",
    "hop_length": "DG_MFCC_HOP",
    "n_mels": "DG_MFCC_N_MELS",
    "n_mfcc": "DG_MFCC_N_COEFFS",
}


def fail(message: str) -> None:
    raise SystemExit(f"gen_model_meta: {message}")


def c_string(value: str) -> str:
    escaped = "".join(chr(byte) if 32 <= byte < 127 and byte not in (34, 92)
                      else f"\\{byte:03o}" for byte in value.encode("utf-8"))
    return '"' + escaped + '"'


def comment(value: str) -> str:
    return str(value).replace("*/", "* /").replace("\n", " ").replace("\r", " ")


def c_float(value: float) -> str:
    return f"{float(value)!r}f"


def check_spec(doc: dict) -> None:
    if not isinstance(doc, dict):
        fail("model specification must be a JSON object")
    if doc.get("spec_version") != SPEC_VERSION:
        fail(f"spec_version {doc.get('spec_version')!r} is not {SPEC_VERSION}; update the SDK")
    if doc["preprocessing"]["mode"] != "host":
        fail("preprocessing.mode is not 'host'; this SDK provides the front end, the model does not")
    kinds = [step["kind"] for step in doc["preprocessing"]["steps"]]
    if kinds != ["scale", "mfcc", "crop_frames"]:
        fail(f"preprocessing steps {kinds} are not the scale → mfcc → crop_frames pipeline dg_mfcc implements")
    scale, mfcc, crop = doc["preprocessing"]["steps"]
    if not math.isfinite(scale["factor"]) or abs(scale["factor"] - 1 / 32768) > 1e-12:
        fail(f"scale.factor {scale['factor']} is not 1/32768 (int16 PCM)")
    for key, want in MFCC_FIXED.items():
        if mfcc.get(key) != want:
            fail(f"mfcc.{key} = {mfcc.get(key)!r}; dg_mfcc implements {want!r}")
    sensor = doc["sensor"]
    if sensor["kind"] != "audio" or sensor["sample_format"] != "int16" or sensor["channels"] != 1:
        fail("dg_mfcc expects mono int16 audio")
    if mfcc["f_max"] != sensor["sample_rate"] / 2:
        fail("dg_mfcc expects f_max = sample_rate / 2")
    for key in ("sample_rate", "clip_samples"):
        if type(sensor[key]) is not int or sensor[key] <= 0:
            fail(f"sensor.{key} must be a positive integer")
    for key in MFCC_DEFINES:
        if type(mfcc[key]) is not int or mfcc[key] <= 0:
            fail(f"mfcc.{key} must be a positive integer")
    fft, hop = mfcc["n_fft"], mfcc["hop_length"]
    if hop < 2 or mfcc["win_length"] != 2 * hop:
        fail("mfcc.win_length must be twice hop_length, with hop_length >= 2")
    if fft < mfcc["win_length"] or fft & (fft - 1):
        fail("mfcc.n_fft must be a power of two at least as large as win_length")
    if not 1 <= mfcc["n_mfcc"] <= mfcc["n_mels"] <= 254:
        fail("MFCC requires 1 <= n_mfcc <= n_mels <= 254")
    hops = sensor["clip_samples"] // hop
    if hops < 2:
        fail("sensor.clip_samples must contain at least two hops")
    if sensor["clip_samples"] % mfcc["hop_length"] or crop["keep"] != [1, hops]:
        fail("dg_mfcc expects crop_frames.keep == [1, clip_samples / hop_length]")
    inputs, outputs = doc["inputs"], doc["outputs"]
    if len(inputs) != 1 or len(outputs) != 1:
        fail("this SDK renders single-input, single-output models")
    for tensor in (inputs[0], outputs[0]):
        if tensor["dtype"] != "int8" or not tensor.get("quantization"):
            fail(f"tensor {tensor['name']} must be int8 with quantization (a trained model's spec)")
        q = tensor["quantization"]
        if not math.isfinite(q["scale"]) or q["scale"] <= 0:
            fail("quantization scale must be finite and positive")
        if type(q["zero_point"]) is not int or not -128 <= q["zero_point"] <= 127:
            fail("quantization zero_point must be an int8 integer")
    if list(inputs[0]["shape"]) != [1, 1, hops - 1, mfcc["n_mfcc"]]:
        fail(f"input shape {inputs[0]['shape']} does not match the preprocessing")
    post = doc["postprocessing"]
    if post.get("kind") != "classification" or post.get("activation") != "softmax":
        fail("postprocessing must describe classification with softmax on output logits")
    if inputs[0].get("axes") != ["batch", "channel", "frame", "coefficient"]:
        fail("input axes must be [batch, channel, frame, coefficient]")
    if outputs[0].get("axes") != ["batch", "class"]:
        fail("output axes must be [batch, class]")
    labels = post.get("labels")
    if not isinstance(labels, list) or not labels or outputs[0]["shape"] != [1, len(labels)]:
        fail("postprocessing.labels must be present and match output shape [1, n_labels]")
    if any(not isinstance(label, str) or not label or "\0" in label for label in labels):
        fail("labels must be nonempty strings without NUL characters")
    if len(set(labels)) != len(labels):
        fail("labels must be unique")
    background = doc["postprocessing"].get("background_labels") or []
    if not isinstance(background, list) or any(label not in labels for label in background):
        fail("background_labels must be a list of model labels")


def choose_thresholds(doc: dict, preset: str | None, scheme: str | None) -> dict:
    """Select the required per-label model thresholds."""
    table = doc["postprocessing"].get("thresholds")
    if not table:
        fail("model thresholds are required; use a model_spec.json containing postprocessing.thresholds")
    preset = preset or table["default_preset"]
    scheme = scheme or table["default_scheme"]
    if preset not in table["presets"]:
        fail(f"preset {preset!r} is not one of {sorted(table['presets'])}")
    chosen = table["presets"][preset]
    if scheme not in chosen["schemes"]:
        fail(f"scheme {scheme!r} is not one of {sorted(chosen['schemes'])}")
    background = set(doc["postprocessing"].get("background_labels") or [])
    per_label = chosen["schemes"][scheme]["per_label"]
    for label in doc["postprocessing"]["labels"]:
        if label not in per_label and label not in background:
            fail(f"label {label!r} has no threshold and is not a background label")
    for label, entry in per_label.items():
        value = entry["threshold"]
        if not math.isfinite(value) or not 0 <= value <= 1:
            fail(f"threshold for {label!r} must be a probability between 0 and 1")
    return {
        "preset": preset,
        "scheme": scheme,
        "per_label": per_label,
        "background": background,
    }


def threshold_lines(chosen: dict, labels: list[str]) -> list[str]:
    """The rendered table: one float per label, in output order."""
    rows = []
    for label in labels:
        entry = chosen["per_label"].get(label)
        if entry is None:
            rows.append(f"    0.0f,  /* {comment(label)}: background needs no detection threshold */")
            continue
        rows.append(f"    {c_float(entry['threshold'])},  /* {comment(label)} */")
    return [
        "",
        "/* Detection thresholds in model output order. */",
        f'#define DG_MODEL_THRESHOLD_PRESET {c_string(chosen["preset"])}',
        f'#define DG_MODEL_THRESHOLD_SCHEME {c_string(chosen["scheme"])}',
        "static const float DG_MODEL_THRESHOLDS[DG_MODEL_N_LABELS] = {",
        *rows,
        "};",
    ]


def render(doc: dict, source: str, chosen: dict) -> str:
    model, sensor = doc["model"], doc["sensor"]
    mfcc = doc["preprocessing"]["steps"][1]
    labels = doc["postprocessing"]["labels"]
    in_q, out_q = doc["inputs"][0]["quantization"], doc["outputs"][0]["quantization"]
    lines = [
        f"/* Generated by tools/gen_model_meta.py from {comment(source)}. Do not edit. */",
        "/* This header sets the preprocessing values and includes the SDK. */",
        "#ifndef DEEPGATE_MODEL_H",
        "#define DEEPGATE_MODEL_H",
        "",
        f'#define DG_MODEL_ARCHITECTURE {c_string(model["key"])}',
        f'#define DG_MODEL_VERSION {c_string(model["version"])}',
        f'#define DG_MODEL_TRAINING_JOB {c_string((model.get("source") or {}).get("training_job_id") or "")}',
        "",
        f"/* Preprocessing matches {comment(mfcc.get('reference', 'mfcc'))}. */",
        f"#define DG_MFCC_SAMPLE_RATE {sensor['sample_rate']}",
        *[f"#define {macro} {mfcc[key]}" for key, macro in MFCC_DEFINES.items()],
        f"#define DG_MFCC_CLIP_HOPS {sensor['clip_samples'] // mfcc['hop_length']}",
        f"#define DG_MFCC_TOP_DB {float(mfcc['top_db'])}f",
        "",
        "#include <deepgate.h>",
        "",
        f"#define DG_MODEL_INPUT_SHAPE {{{', '.join(str(d) for d in doc['inputs'][0]['shape'])}}}",
        f"#define DG_MODEL_INPUT_SCALE {c_float(in_q['scale'])}",
        f"#define DG_MODEL_INPUT_ZERO_POINT {in_q['zero_point']}",
        f"#define DG_MODEL_OUTPUT_SCALE {c_float(out_q['scale'])}",
        f"#define DG_MODEL_OUTPUT_ZERO_POINT {out_q['zero_point']}",
        f"#define DG_MODEL_N_LABELS {len(labels)}",
        "",
        f"static const char *const DG_MODEL_LABELS[DG_MODEL_N_LABELS] = {{",
        *[f'    {c_string(label)},' for label in labels],
        "};",
        "",
        "/* Background outputs participate in classification but are not reported. */",
        "static const bool DG_MODEL_IS_BACKGROUND[DG_MODEL_N_LABELS] = {",
        *[f"    {'true' if label in chosen['background'] else 'false'},  /* {comment(label)} */" for label in labels],
        "};",
        *threshold_lines(chosen, labels),
        "",
        "#endif",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=pathlib.Path, help="model_spec.json written by the training job")
    ap.add_argument("--tflite", type=pathlib.Path, help="verify the model file matches model.artifact_sha256")
    ap.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("deepgate_model.h"))
    ap.add_argument("--preset", help="threshold preset to render (default: the spec's own)")
    ap.add_argument("--scheme", help="threshold scheme to render (default: the spec's own)")
    args = ap.parse_args()
    try:
        doc = json.loads(args.spec.read_text())
        check_spec(doc)
        chosen = choose_thresholds(doc, args.preset, args.scheme)
        if args.tflite:
            digest = hashlib.sha256(args.tflite.read_bytes()).hexdigest()
            if digest != doc["model"].get("artifact_sha256"):
                fail(f"{args.tflite} does not match the spec's artifact_sha256")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render(doc, args.spec.name, chosen))
        thresholds = f", thresholds {chosen['preset']}/{chosen['scheme']}"
        print(
            f"wrote {args.output} ({len(doc['postprocessing']['labels'])} labels, "
            f"{doc['model']['key']} {doc['model']['version']}{thresholds})"
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        fail(f"invalid or unreadable model specification: {exc}")


if __name__ == "__main__":
    main()
