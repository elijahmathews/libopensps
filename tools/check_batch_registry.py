#!/usr/bin/env python3
"""The per-model parameter registry (src/core/opensps_batch_registry.f90) covers PARAMS_CORE.

Checks, with the Fortran sources as text:
  * every member of PARAMS_CORE (src/core/opensps_types.f90) is a registered field, and every registered field is a member;
  * the table is consistent: N_FIELDS, the names, the class / stage / kind arrays and the F_<NAME> index constants have one
    entry per field and agree in order;
  * every real or integer field has a case in registry_get and registry_set, every character field in registry_equal.

Usage: check_batch_registry.py [source root]   (exit 1 on any problem)
"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
types_src = (root / 'src/core/opensps_types.f90').read_text()
reg_src = (root / 'src/core/opensps_batch_registry.f90').read_text()
problems = []


def strip_comments(text):
    return '\n'.join(line.split('!')[0] for line in text.split('\n'))


# --- members of PARAMS_CORE (name, kind): real(WP) / integer / character
blk = types_src[types_src.index('type PARAMS_CORE'):types_src.index('end type PARAMS_CORE')]
blk = strip_comments(blk)
# join continuation lines
blk = re.sub(r'&\s*\n\s*', ' ', blk)
members = {}
for line in blk.split('\n')[1:]:
    if '::' not in line:
        continue
    decl, rest = line.split('::', 1)
    kind = 'R' if 'real' in decl else ('I' if 'integer' in decl else 'S')
    # names are the identifiers before '=' or at the end of a comma-separated item (outside parentheses)
    depth, item, items = 0, '', []
    for ch in rest:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == ',' and depth == 0:
            items.append(item)
            item = ''
        else:
            item += ch
    items.append(item)
    for it in items:
        name = it.split('=')[0].strip()
        if name:
            members[name] = kind

# --- the registry table
reg = strip_comments(reg_src)
reg_flat = re.sub(r'&\s*\n\s*', ' ', reg)


def array_of(name, quoted):
    m = re.search(r'parameter ::\s*%s\(N_FIELDS\)\s*=\s*\[(?:character\(len=\d+\) ::)?(.*?)\]' % name, reg_flat, re.S)
    if not m:
        problems.append('table %s not found' % name)
        return []
    body = m.group(1)
    if quoted:
        return re.findall(r"'([^']*)'", body)
    return [int(x) for x in re.findall(r'-?\d+', body)]


names = array_of('field_names', True)
classes = array_of('field_class', False)
stages = array_of('field_stage', False)
kinds = array_of('field_kind', False)
m = re.search(r'N_FIELDS\s*=\s*(\d+)', reg)
n_fields = int(m.group(1)) if m else -1

if not (n_fields == len(names) == len(classes) == len(stages) == len(kinds)):
    problems.append('N_FIELDS = %d but %d names, %d classes, %d stages, %d kinds' %
                    (n_fields, len(names), len(classes), len(stages), len(kinds)))
if len(set(names)) != len(names):
    problems.append('duplicate field names: ' + ', '.join(sorted(n for n in names if names.count(n) > 1)))

for n_ in sorted(set(members) - set(names)):
    problems.append('PARAMS_CORE member not registered: ' + n_)
for n_ in sorted(set(names) - set(members)):
    problems.append('registered field is not a PARAMS_CORE member: ' + n_)

# index constants F_<NAME> = position
consts = dict((n_.lower(), int(v)) for n_, v in re.findall(r'integer, parameter, public :: F_(\w+)\s*=\s*(\d+)', reg))
for i, n_ in enumerate(names, 1):
    if consts.get(n_) != i:
        problems.append('F_%s should be %d, is %s' % (n_.upper(), i, consts.get(n_)))
for c in classes:
    if c < 0 or c > 4:
        problems.append('class out of range: %d' % c)
for c in stages:
    if c < 1 or c > 9:
        problems.append('stage out of range: %d' % c)

# kinds agree with the type, and the accessors cover the fields
kind_code = {'R': 1, 'I': 2, 'S': 3}
for i, n_ in enumerate(names):
    if n_ in members and i < len(kinds) and kinds[i] != kind_code[members[n_]]:
        problems.append('kind of %s is %d, PARAMS_CORE says %s' % (n_, kinds[i], members[n_]))


def case_names(func):
    m = re.search(r'(function|subroutine) %s\b(.*?)end (function|subroutine) %s' % (func, func), reg, re.S)
    return set(x.lower() for x in re.findall(r'case \(F_(\w+)\)', m.group(2))) if m else set()


for func, need in (('registry_get', 'RI'), ('registry_set', 'RI'), ('registry_equal', 'S')):
    have = case_names(func)
    for n_, k in members.items():
        if k in need and n_ not in have:
            problems.append('%s has no case for %s' % (func, n_))

if problems:
    print('check_batch_registry: %d problem(s)' % len(problems))
    for p in problems:
        print('  ' + p)
    sys.exit(1)
print('check_batch_registry: %d fields registered, PARAMS_CORE covered' % len(names))
