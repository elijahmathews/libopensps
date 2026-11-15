#!/usr/bin/env python3
"""Validate the `-Dtuning_overrides` option and print the preprocessor flags it stands for.

    overrides.py src/core/opensps_tuning.f90 "SEG_LEN=128,PT_PIECE=16"

prints, one per line, `-DTUNE_SEG_LEN_OVERRIDE=128` and `-DTUNE_PT_PIECE_OVERRIDE=16`. meson.build runs it
at configure time. A name the registry does not define (a typo would otherwise change nothing and look
like a result), a repeated name, or a value that is not a non-negative integer stops the configuration with
a message. Whether a value is *legal* for its parameter (a tile that divides a block, a size that fits the
registers) is not checked here: it is what a sweep measures, and the tests of both precisions gate it.
"""
import re
import sys


def parse(spec, registry_text):
    known = set(re.findall(r"TUNE_([A-Z0-9_]+)_OVERRIDE", registry_text))
    flags, seen = [], set()
    for item in filter(None, (x.strip() for x in spec.split(","))):
        name, eq, value = item.partition("=")
        name, value = name.strip().upper(), value.strip()
        if not eq or not re.fullmatch(r"[0-9]+", value):
            raise ValueError(f"'{item}': expected NAME=<non-negative integer>")
        if name not in known:
            raise ValueError(f"'{name}' is not a compile-time tuning parameter (known: {', '.join(sorted(known))})")
        if name in seen:
            raise ValueError(f"'{name}' is given twice")
        seen.add(name)
        flags.append(f"-DTUNE_{name}_OVERRIDE={int(value)}")
    return flags


def main(argv):
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        text = open(argv[1]).read()
        print("\n".join(parse(argv[2], text)))
    except (ValueError, OSError) as e:
        print(f"tuning_overrides: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
