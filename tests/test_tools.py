import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_arduino
import gen_model_meta as meta


def specification():
    doc = {
        "spec_version": 1,
        "model": {"key": "kws-dscnn", "version": "2.1.0"},
        "sensor": {"kind": "audio", "sample_format": "int16", "channels": 1,
                   "sample_rate": 16000, "clip_samples": 16000},
        "preprocessing": {"mode": "host", "steps": [
            {"kind": "scale", "factor": 1 / 32768},
            {"kind": "mfcc", **meta.MFCC_FIXED, "f_max": 8000, "n_fft": 1024,
             "win_length": 640, "hop_length": 320, "n_mels": 40, "n_mfcc": 10},
            {"kind": "crop_frames", "keep": [1, 50]}]},
        "inputs": [{"name": "input", "shape": [1, 1, 49, 10], "dtype": "int8",
                    "axes": ["batch", "channel", "frame", "coefficient"],
                    "quantization": {"scale": 1, "zero_point": 0}}],
        "outputs": [{"name": "output", "shape": [1, 2], "dtype": "int8",
                     "axes": ["batch", "class"],
                     "quantization": {"scale": 1, "zero_point": 0}}],
        "postprocessing": {"kind": "classification", "activation": "softmax",
                           "labels": ["_silence_", 'say "hi" */ café 🔊'],
                           "background_labels": ["_silence_"]},
    }
    labels = doc["postprocessing"]["labels"]
    doc["postprocessing"]["thresholds"] = {
        "default_preset": "balanced", "default_scheme": "shrunk",
        "presets": {"balanced": {"schemes": {"shrunk": {"per_label": {
            labels[1]: {"threshold": 0.7}
        }}}}}
    }
    return doc


class MetadataTests(unittest.TestCase):
    def test_generated_header_compiles(self):
        doc = specification()
        meta.check_spec(doc)
        header = meta.render(doc, "model_spec.json", meta.choose_thresholds(doc, None, None))
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            (path / "deepgate_model.h").write_text(header)
            (path / "main.c").write_text('#include "deepgate_model.h"\nint main(void) { return DG_MODEL_N_LABELS != 2 || DG_MODEL_INPUT_SCALE != 1.0f || DG_MODEL_INPUT_ZERO_POINT != 0 || DG_MODEL_OUTPUT_SCALE != 1.0f || DG_MODEL_OUTPUT_ZERO_POINT != 0 || !DG_MODEL_IS_BACKGROUND[0] || DG_MODEL_IS_BACKGROUND[1] || DG_MODEL_THRESHOLDS[1] != 0.7f; }\n')
            subprocess.run(["cc", "-std=c11", "-Werror", "-I", str(ROOT / "src"),
                            str(path / "main.c"), "-o", str(path / "check")], check=True)
            subprocess.run([str(path / "check")], check=True)

    def test_background_flags_preserve_model_thresholds_and_escape_comments(self):
        doc = specification()
        entry = {"threshold": 1, "recall": 1, "false_accepts_per_negative_clip": 0}
        chosen = {"preset": "balanced", "scheme": "shrunk", "background": {"_silence_"},
                  "per_label": {label: entry for label in doc["postprocessing"]["labels"]}}
        header = meta.render(doc, "model_spec.json", chosen)
        self.assertIn('true,  /* _silence_', header)
        self.assertIn('1.0f,  /* _silence_', header)
        self.assertIn('1.0f,  /* say "hi" * /', header)

    def test_background_metadata_is_explicit(self):
        doc = specification()
        doc["postprocessing"]["background_labels"] = []
        with self.assertRaisesRegex(SystemExit, "has no threshold"):
            meta.choose_thresholds(doc, None, None)
        doc["postprocessing"]["thresholds"]["presets"]["balanced"]["schemes"]["shrunk"]["per_label"]["_silence_"] = {"threshold": 0.3}
        header = meta.render(doc, "model_spec.json", meta.choose_thresholds(doc, None, None))
        self.assertIn("false,  /* _silence_", header)
        doc["postprocessing"]["background_labels"] = ["not a model output"]
        with self.assertRaisesRegex(SystemExit, "background_labels"):
            meta.check_spec(doc)

    def test_thresholds_do_not_require_validation_statistics(self):
        doc = specification()
        label = doc["postprocessing"]["labels"][1]
        doc["postprocessing"]["thresholds"] = {
            "default_preset": "balanced", "default_scheme": "shrunk",
            "presets": {"balanced": {"schemes": {"shrunk": {"per_label": {
                label: {"threshold": 0.75, "recall": None}
            }}}}}
        }
        meta.check_spec(doc)
        chosen = meta.choose_thresholds(doc, None, None)
        self.assertIn("0.75f,", meta.render(doc, "model_spec.json", chosen))

    def test_rejects_invalid_labels(self):
        for labels in ("ab", [], ["a", "a"], ["a", 2]):
            with self.subTest(labels=labels):
                doc = specification()
                doc["postprocessing"]["labels"] = labels
                with self.assertRaises(SystemExit):
                    meta.check_spec(doc)

    def test_rejects_unsupported_tensor_meaning(self):
        cases = (("postprocessing", "kind", "regression"),
                 ("postprocessing", "activation", "sigmoid"),
                 ("inputs", "axes", ["batch", "frame", "coefficient", "channel"]),
                 ("outputs", "axes", ["class", "batch"]))
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                doc = specification()
                target = doc[section][0] if section in ("inputs", "outputs") else doc[section]
                target[key] = value
                with self.assertRaises(SystemExit):
                    meta.check_spec(doc)

    def test_unknown_label_uses_explicit_background_flag(self):
        doc = specification()
        post = doc["postprocessing"]
        post["labels"] = ["_unknown_", "unknown"]
        post["background_labels"] = ["_unknown_"]
        post["thresholds"]["presets"]["balanced"]["schemes"]["shrunk"]["per_label"] = {
            "unknown": {"threshold": 0.6}
        }
        meta.check_spec(doc)
        header = meta.render(doc, "model_spec.json", meta.choose_thresholds(doc, None, None))
        self.assertIn('    "_unknown_",', header)
        self.assertIn('    "unknown",', header)
        self.assertIn('true,  /* _unknown_ */', header)
        self.assertIn('false,  /* unknown */', header)
        self.assertIn('0.6f,  /* unknown */', header)

    def test_rejects_unsupported_dimensions(self):
        for key, value in (("n_fft", 1000), ("hop_length", 0), ("win_length", 800), ("n_mels", 255)):
            with self.subTest(key=key):
                doc = specification()
                doc["preprocessing"]["steps"][1][key] = value
                with self.assertRaises(SystemExit):
                    meta.check_spec(doc)

    def test_rejects_invalid_quantization(self):
        for scale in (0, -1, float("inf"), float("nan")):
            doc = specification()
            doc["inputs"][0]["quantization"]["scale"] = scale
            with self.assertRaises(SystemExit):
                meta.check_spec(doc)

    def test_missing_model_thresholds_are_rejected(self):
        doc = specification()
        del doc["postprocessing"]["thresholds"]
        with self.assertRaisesRegex(SystemExit, "model thresholds are required"):
            meta.choose_thresholds(doc, None, None)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            spec = path / "model_spec.json"
            output = path / "deepgate_model.h"
            spec.write_text(json.dumps(doc))
            result = subprocess.run([sys.executable, str(ROOT / "tools/gen_model_meta.py"),
                                     str(spec), "-o", str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("model thresholds are required", result.stderr)
            self.assertFalse(output.exists())

    def test_malformed_spec_has_actionable_cli_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "model_spec.json"
            path.write_text('{"spec_version": 1}')
            result = subprocess.run([sys.executable, str(ROOT / "tools/gen_model_meta.py"), str(path)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("gen_model_meta:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


class ArduinoTests(unittest.TestCase):
    @patch.object(build_arduino, "model_definitions", return_value=["DG_MFCC_HOP=160"])
    def test_preserves_core_settings_and_quotes_archive(self, definitions):
        properties = {"compiler.c.extra_flags": "-DC_ONLY=1", "compiler.cpp.extra_flags": "-DCPP_ONLY=1", "build.ld_libs": "-lc -lm", "build.board_specific_macros":
                      "'-DNUM_HW_SERIAL=2' '-DARDUINO_MAIN_TASK_STACK_SIZE=2048' '-DEXTRA=1'"}
        cmd = build_arduino.build_command("arduino-cli", pathlib.Path("/tmp/my sketch"), properties)
        self.assertIn('build.ld_libs="/tmp/my sketch/libmodel.a" -lc -lm', cmd)
        self.assertIn("build.board_specific_macros='-DNUM_HW_SERIAL=2' '-DARDUINO_MAIN_TASK_STACK_SIZE=8192' '-DEXTRA=1'", cmd)
        self.assertIn("compiler.c.extra_flags=-DC_ONLY=1 -DDG_MFCC_HOP=160", cmd)
        self.assertIn("compiler.cpp.extra_flags=-DCPP_ONLY=1 -DDG_MFCC_HOP=160", cmd)
        self.assertNotIn("--upload", cmd)
        uploaded = build_arduino.build_command("arduino-cli", pathlib.Path("/tmp/sketch"), properties, "/dev/test")
        self.assertEqual(uploaded[-4:-1], ["--upload", "--port", "/dev/test"])

    def test_reads_generated_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            header = pathlib.Path(directory) / "deepgate_model.h"
            doc = specification()
            doc["preprocessing"]["steps"][1]["n_mels"] = 32
            header.write_text(meta.render(doc, "model_spec.json", meta.choose_thresholds(doc, None, None)))
            self.assertIn("DG_MFCC_N_MELS=32", build_arduino.model_definitions(header))
            header.write_text("#define DG_MFCC_HOP 160\n")
            with self.assertRaises(ValueError):
                build_arduino.model_definitions(header)

    def test_rejects_missing_board_settings(self):
        with self.assertRaises(ValueError):
            build_arduino.build_command("arduino-cli", pathlib.Path("/tmp/sketch"), {})


class CMakeTests(unittest.TestCase):
    def test_generated_settings_reach_library_and_consumer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            header = path / "deepgate_model.h"
            doc = specification()
            doc["sensor"].update(sample_rate=8000, clip_samples=8000)
            mfcc = doc["preprocessing"]["steps"][1]
            mfcc.update(n_fft=512, win_length=320, hop_length=160, n_mels=32, f_max=4000)
            meta.check_spec(doc)
            header.write_text(meta.render(doc, "model_spec.json", meta.choose_thresholds(doc, None, None)))
            (path / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.20)\nproject(config_test C)\n'
                'set(DEEPGATE_MODEL_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/deepgate_model.h")\n'
                f'add_subdirectory("{ROOT}" sdk)\n'
                'add_executable(check main.c)\ntarget_link_libraries(check PRIVATE deepgate)\n')
            (path / "main.c").write_text(
                '#include <deepgate.h>\n#include <stdlib.h>\n#include <math.h>\n'
                'static dg_mfcc_t mfcc;\n'
                'int main(int argc, char **argv) {\n'
                '  dg_mfcc_init(&mfcc);\n'
                '  return argc != 2 || DG_MFCC_N_MELS != atoi(argv[1]) ||\n'
                '    DG_MFCC_HOP != 160 || DG_AUDIO_CHUNK_SAMPLES != 160 ||\n'
                '    !dg_mfcc_config_matches(DG_MFCC_CONFIG_ID) ||\n'
                '    fabsf(mfcc.fft_twiddle_real[1] - cosf(6.28318530718f / DG_MFCC_N_FFT)) > 1e-6f ||\n'
                '    fabsf(mfcc.dct_basis[0] - 1.0f / sqrtf(DG_MFCC_N_MELS)) > 1e-6f;\n}\n')
            build = path / "build"
            subprocess.run(["cmake", "-S", str(path), "-B", str(build)], check=True, capture_output=True)
            for n_mels in (32, 24):
                mfcc["n_mels"] = n_mels
                header.write_text(meta.render(doc, "model_spec.json", meta.choose_thresholds(doc, None, None)))
                subprocess.run(["cmake", "--build", str(build)], check=True, capture_output=True)
                subprocess.run([str(build / "check"), str(n_mels)], check=True)


if __name__ == "__main__":
    unittest.main()
