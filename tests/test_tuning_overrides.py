#!/usr/bin/env python3
"""tools/tuning/overrides.py: the validation of the -Dtuning_overrides option (names against the registry)."""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "tuning"))
import overrides  # noqa: E402

REGISTRY = (ROOT / "src" / "core" / "opensps_tuning.f90").read_text()


class TestOverrides(unittest.TestCase):
    def test_flags(self):
        self.assertEqual(overrides.parse("SEG_LEN=128,PT_PIECE=16", REGISTRY),
                         ["-DTUNE_SEG_LEN_OVERRIDE=128", "-DTUNE_PT_PIECE_OVERRIDE=16"])

    def test_case_and_spaces(self):
        self.assertEqual(overrides.parse(" seg_len = 64 ,", REGISTRY), ["-DTUNE_SEG_LEN_OVERRIDE=64"])

    def test_empty(self):
        self.assertEqual(overrides.parse("", REGISTRY), [])

    def test_unknown_name_is_an_error(self):
        with self.assertRaises(ValueError) as cm:
            overrides.parse("SEG_LENGTH=128", REGISTRY)      # a typo must not silently change nothing
        self.assertIn("SEG_LEN", str(cm.exception))

    def test_run_time_parameters_are_not_build_parameters(self):
        for name in ("SSP_BATCH_CHUNK", "HOST_SFH_MAX", "PINNED_CHUNK_BYTES"):
            with self.assertRaises(ValueError):
                overrides.parse(f"{name}=8", REGISTRY)       # those are set with OPENSPS_TUNE_<NAME>

    def test_bad_values(self):
        for spec in ("SEG_LEN", "SEG_LEN=", "SEG_LEN=-1", "SEG_LEN=1.5", "SEG_LEN=abc"):
            with self.assertRaises(ValueError):
                overrides.parse(spec, REGISTRY)

    def test_repeated_name(self):
        with self.assertRaises(ValueError):
            overrides.parse("SEG_LEN=64,SEG_LEN=128", REGISTRY)

    def test_every_registry_constant_is_accepted(self):
        import re
        for name in re.findall(r"^#ifdef TUNE_([A-Z0-9_]+)_OVERRIDE", REGISTRY, re.M):
            self.assertEqual(overrides.parse(f"{name}=7", REGISTRY), [f"-DTUNE_{name}_OVERRIDE=7"])


if __name__ == "__main__":
    unittest.main()
