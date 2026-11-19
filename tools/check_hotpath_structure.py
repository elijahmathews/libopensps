#!/usr/bin/env python3
"""A ratchet on the structure of the hot paths: the preprocessor conditionals of the sources and the `stop` statements of the library
may not grow (doc/HOT_PATHS.md).

Why: a kernel written for one compiler, vendor, precision or mode as a variant of another multiplies the code that every later change
has to keep in step (each variant has its own bugs, its own device-specific workarounds, its own numbers), and a `stop` in a library
routine kills the host program of a binding. Both only ever grow by accident, so the counts are held in a committed baseline
(tools/hotpath_structure_baseline.json) and a pull request that raises one has to say so in that file, where a reviewer sees it. A
count that falls is welcome: the script says so and the baseline is lowered in the same change (`--update`).

Counted, per directory of src/ and in total, over the Fortran, C, C++ and HIP sources:
  * conditionals: lines that start with #if, #ifdef, #ifndef or #elif;
  * stops: bare `stop` statements of Fortran sources (not `error stop`).

Usage: check_hotpath_structure.py [source root] [--update]    (exit 1 when a count exceeds its baseline)
"""
import json
import re
import sys
from pathlib import Path

args = [a for a in sys.argv[1:] if not a.startswith('--')]
update = '--update' in sys.argv
root = Path(args[0]) if args else Path(__file__).resolve().parent.parent
baseline_path = root / 'tools' / 'hotpath_structure_baseline.json'

COND = re.compile(r'^\s*#\s*(if|ifdef|ifndef|elif)\b')
STOP = re.compile(r'^\s*stop\b', re.IGNORECASE)
SRC_EXT = {'.f90', '.F90', '.inc', '.c', '.h', '.cpp', '.hpp', '.cu', '.hip'}


def count():
    counts = {}
    for path in sorted((root / 'src').rglob('*')):
        if path.suffix not in SRC_EXT or not path.is_file():
            continue
        rel = path.relative_to(root / 'src')
        d = rel.parts[0] if len(rel.parts) > 1 else '.'
        c = counts.setdefault(d, {'conditionals': 0, 'stops': 0})
        for line in path.read_text(errors='replace').split('\n'):
            if COND.match(line):
                c['conditionals'] += 1
            elif path.suffix in ('.f90', '.F90', '.inc') and STOP.match(line.split('!')[0]):
                c['stops'] += 1
    counts['total'] = {'conditionals': sum(c['conditionals'] for c in counts.values()),
                       'stops': sum(c['stops'] for c in counts.values())}
    return counts


now = count()
if update:
    baseline_path.write_text(json.dumps(now, indent=2, sort_keys=True) + '\n')
    print('baseline written:', json.dumps(now['total']))
    sys.exit(0)

if not baseline_path.exists():
    print('check_hotpath_structure: no baseline (%s); run with --update' % baseline_path)
    sys.exit(1)
base = json.loads(baseline_path.read_text())
problems, lower = [], []
for d in sorted(set(now) | set(base)):
    for kind in ('conditionals', 'stops'):
        n = now.get(d, {}).get(kind, 0)
        b = base.get(d, {}).get(kind, 0)
        if n > b:
            problems.append('%s: %d %s, the baseline allows %d' % (d, n, kind, b))
        elif n < b:
            lower.append('%s: %d %s, the baseline has %d' % (d, n, kind, b))
if problems:
    print('check_hotpath_structure: %d count(s) above the baseline (tools/hotpath_structure_baseline.json)' % len(problems))
    for p in problems:
        print('  ' + p)
    print('A new vendor, precision or mode variant of a kernel should replace one or equal a named baseline bit for bit; a stop in '
          'a library routine should be a status. If the increase is deliberate, raise the baseline in this change (--update) '
          'and say why.')
    sys.exit(1)
if lower:
    print('check_hotpath_structure: below the baseline (lower it with --update in this change):')
    for p in lower:
        print('  ' + p)
else:
    print('check_hotpath_structure: at the baseline (%d conditionals, %d stops)' %
          (now['total']['conditionals'], now['total']['stops']))
