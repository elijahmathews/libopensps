# Cutover to libopensps

Tools for moving this repository into `elijahmathews/libopensps` (`doc/ROADMAP.md`, "Cutover to
libopensps"). libopensps already has its own `main` (README with the logo, LICENSE.md, AGENTS.md,
CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md, assets/; Conventional Commits, squash merges,
linear history). The fork arrives as one commit on top of it, followed by the rename. The
sequence for the day itself is in [CHECKLIST.md](CHECKLIST.md).

| File | Purpose |
|---|---|
| `make_import_commit.sh <commit> <libopensps-dir>` | The import: one commit `feat: import elijahmathews/fsps at commit <hash>` on a branch `import` from libopensps's `main`; no push |
| `rename_to_opensps.sh <repo-dir>` | The rename pull request: public C surface, build artifacts and internal Fortran names to `opensps`, version 0.1.0; idempotent, does not commit |
| `rename_internal.pl` | The text rules of the rename's internal stage |
| `port_patches.sh <patch>...` | Ports fork commits (`git format-patch`) onto renamed libopensps: same rename applied, author and message kept |
| `authors.mailmap` | Identities merged when `AUTHORS` is generated (with the evidence for each) |
| `templates/CITATION.cff` | Added by the import (cites the FSPS papers and libopensps) |
| `proposals/README.md` | Sections proposed for libopensps's README (not applied: the repository's README wins) |

Both scripts print `--help`.

## The import commit

`make_import_commit.sh` exports the fork's tree with `git archive <commit>` (so nothing untracked
or ignored, such as build directories or converted data roots, can get in), creates the branch
`import` from `origin/main` of a libopensps clone, copies the tree over it and commits once:

```
feat: import elijahmathews/fsps at commit <hash>

Source: https://github.com/elijahmathews/fsps (tag fsps-final), commit <hash>,
the modernized FSPS fork. Its tree is imported without its history, except:
- left out: data/ (1.4GB), the libraries: published as data releases of
  elijahmathews/opensps-data;
- left out: tests/data/sps_ref_*.bin and tests/data/param_ref/ (59MB),
  the regression references: part of the data release;
- not imported, this repository's own files kept instead: AGENTS.md,
  CLAUDE.md, LICENSE.md, README.md;
- CI workflows run on main (they ran on master and develop);
- added: AUTHORS, CITATION.cff.

History and authorship are recorded in AUTHORS, derived from the fork's
history; the full history stays in elijahmathews/fsps.
```

The author and committer are your git identity; the commit is unsigned and carries no trailers
(`--trailer` adds some).

**libopensps's files win.** Every path that exists on `main` keeps its version there: the fork's
`README.md`, `LICENSE.md` and `AGENTS.md` (and `CLAUDE.md`, a symbolic link to `AGENTS.md` on
`main`) are not imported. The fork's LICENSE carries the same FSPS notice as libopensps's; what
its README and AGENTS.md hold that libopensps's lack is in `proposals/README.md` and in the
fork's AGENTS.md (whose build, test and correctness sections libopensps's AGENTS.md marks as
TODO), to be merged by hand. The fork's `GEMINI.md` (`@AGENTS.md`) comes along and points to
libopensps's AGENTS.md.

**Left out:**

| Path | Size | Where it goes |
|---|---|---|
| `data/` | 1.4 GB | the data releases of elijahmathews/opensps-data, fetched by `tools/fetch_data.py` |
| `tests/data/sps_ref_*.bin` (5 files) | 52 MB | data-v2: the library `references/sps` |
| `tests/data/param_ref/` (35 files) | 7.1 MB | data-v2: the library `references/param` |

The references are copied with `SHA256SUMS` to `<libopensps-dir>.references` (`--refs-dir`) so
that they can be compared with the release; `--keep-refs` imports them instead. Why they leave:
they are generated outputs that every intended physics change regenerates, and each regeneration
would add ~60 MB to the repository's history for good. In the data release they are versioned
with the data they were computed from; the regression tests find them through the data root and
fall back to `tests/data/` (PR #30).

What remains above 1 MB (the script lists it on every run): the test library's MILES fluxes
(`tests/data/fsps-test-data/.../miles/flux.bin`, 2.4 MB), which the unit suite runs on.

**CI filters:** the imported workflows ran on `master` and `develop`; the import points them at
`main` (`--ci-branch`), so that CI runs on the import pull request itself.

**AUTHORS** is generated from the history of `<commit>` (`git log --use-mailmap`, ordered by
number of commits): FSPS contributors are those with commits up to the fork point
(`--fsps-base`, the merge base with `upstream/master`, `82a873508d50`), fork contributors those
after it. Identities are merged only where they are certainly one person (`authors.mailmap`:
Charlie Conroy under six addresses; Benjamin Johnson, Nell Byler and Elijah Mathews under two
each). Two committers appear only as addresses; AUTHORS lists them as "contributors from the
Google Code era of FSPS, known only by these handles" (`tomorrowsville`, `avillaum`). Bots and AI
agents are not listed as authors; one sentence notes that the fork was written with AI agents.

## How the import lands

`main` is protected and squash-merged. Two ways:

1. **Recommended: a pull request from the `import` branch.** Squash-merging a pull request of
   one commit keeps it as one commit with its message (check the message in the merge dialog;
   GitHub fills it from the commit for one-commit pull requests). CI runs on the pull request
   (the imported workflows now trigger on `main`), the protection rules stay in force, and the
   import is reviewable like any other change. Set the repository variable `FSPS_DATA_RELEASE`
   first, so that CI fetches the data and references.
2. **A direct push** of the commit to `main` by you, bypassing the protection: the commit
   arrives byte-for-byte as built (author, date, message), but without CI before it lands.

The rename follows as its own pull request: `refactor!: rename the public C API and library to
opensps`, with a `BREAKING CHANGE:` footer (CHECKLIST.md, step 5).

## Rename scope

**Full rename** (decided): the public surface (stage 1) and the internal names (stage 2, on by
default; `--no-internal` skips it).

Stage 1, the public surface:

| Renamed | From | To |
|---|---|---|
| Library version and soname | 3.2.0, `libfsps.so.3` | 0.1.0, `libopensps.so.0` (`--version X.Y.Z`; `--keep-version`) |
| Header | `include/fsps.h` | `include/opensps.h` |
| pkg-config | `fsps.pc` | `opensps.pc` |
| C symbols (all exported `bind(C)` names and header functions) | `fsps_*` | `opensps_*` |
| Header macros (status codes, API version, guard, legacy switch) | `FSPS_*` | `OPENSPS_*` |
| Meson | project `fsps`, `library('fsps')`, `fsps_dep`, `override_dependency('fsps')` | `opensps`, `opensps_dep` |
| Packaging | `fsps.spec`, `libfsps3`, `libfsps-dev` | `opensps.spec`, `libopensps0`, `libopensps-dev` |
| Docs and tests | C tests, `doc/FSPS_C_API.md`, mentions in docs | `opensps_*`, `doc/C_API.md` |
| CI | branch filters | `[main]` (`--branch`; already done by the import) |

Stage 2, the internal names (`rename_internal.pl`):

| Renamed | From | To |
|---|---|---|
| Source files (58) | `src/**/fsps_*.f90`, `src/core/fsps_*.c`, `tests/**/test_fsps_*.f90`, `tests/test_fsps.f90`, `tools/fsps_data/fsps_data.py`, ... | `opensps_*`, `test_opensps_*` |
| Fortran modules, submodules, procedures, types, variables | `fsps_*` (also after `_`: `test_fsps_io_mod`) | `opensps_*` |
| Fortran named constants | `FSPS_ERR_UNKNOWN_INT_PARAM`, ... | `OPENSPS_*` |
| Message tags | `[FSPS_IO]`, `[FSPS_INIT]`, `[FSPS-C]`, ... | `[OPENSPS_IO]`, `[OPENSPS-C]`, ... |
| Data tools and test library | `tools/fsps_data/`, `tests/data/fsps-test-data/` | `tools/opensps_data/`, `tests/data/opensps-test-data/` |
| Meson | source lists, unit test modules (`substring(5)` of `fsps_` becomes `substring(8)`) | |

Fortran code lines that grow beyond 132 characters are split at a comma with `&` (4 lines in
the current tree). The two renamed directories were cheap (references in meson.build, CI, the
Python tools and docs) and keep the tree free of `fsps` names except where they name legacy
FSPS or the past; the data inside `tests/data` is not edited (its manifests record
`"converter": "tools/fsps_data"` as provenance, which stays true for that data).

**Kept, on purpose** (what `fsps` remains after the rename, and why):

- **Environment variables** `SPS_HOME`, `FSPS_DATA_HOME`, `FSPS_DATA_FORMAT`, `FSPS_TEST_*`,
  `FSPS_STRESS_*`, `FSPS_SCENARIO_BATCH`, `FSPS_NO_PIN`, ... and the CI variable
  `FSPS_DATA_RELEASE`: existing setups and CI keep working; `OPENSPS_DATA_HOME`, the preferred
  data root, is read first (PR #29).
- **Data directory names** (`share/fsps`, `~/.local/share/fsps`, `$XDG_DATA_HOME/fsps`) and the
  legacy data-format aliases the readers still accept (`fsps-data.json`, `fsps-data-release`).
- **Build switches** `FSPS_PINNED`, `FSPS_CUBLAS`, `FSPS_CUBLAS_MATH` (preprocessor).
- **`FSPS_VERSION`** and the "FSPS Version: 3.2" banner: the FSPS version the physics follows.
- **Upstream FSPS**: URLs (`cconroy20/fsps`), the word FSPS in prose, citations, `python-fsps`,
  `FSPS_URL` in the data provenance tool; **elijahmathews/fsps** and `fsps-final` where the
  import's origin is named.
- **`tools/legacy_parity`, `tools/benchmark_legacy`**: they build and run legacy FSPS.
- **`include/fsps.h`**, the deprecated compatibility header (stage 1).
- **`tests/data`** contents (data and their provenance).

After the rename, `nm -D` lists no symbol containing `fsps`. Every exported symbol is a C
function `opensps_*` or a module symbol `__opensps_*_MOD_*`. The batch of one, formerly an
external procedure exported as `fsps_scenario_batch_of_one_`, is a submodule procedure since
fork PR #31.

**Compatibility:** `include/fsps.h` becomes a deprecated header that includes `opensps.h` and
`#define`s each old function and macro name to the new one, generated once from the header as
it was before the rename and frozen afterwards. It is installed alongside `opensps.h`. It costs
one file and keeps C code written against the fork compiling; there is no binary
compatibility (the soname changes anyway). Remove it at the next major version.

**Rerunnable:** the script finds the public names in the sources (bind names, header
declarations, header macros), not in a fixed list, and the internal rules are patterns. A run
on a renamed tree changes nothing, and a run on a tree that gained `fsps_*` names (a ported
patch) renames just those. `--check-only` lists what still carries an old public name.

**Blame:** the rename touches ~3,600 lines in 199 files. `.git-blame-ignore-revs` (written by
the rename, honored by GitHub and by `git config blame.ignoreRevsFile .git-blame-ignore-revs`)
lists commits that blame skips. Squash merges give the rename its final hash only on `main`,
so the hash is recorded after the merge, in a one-line follow-up pull request:
`rename_to_opensps.sh --record-blame-ignore <commit> .` (CHECKLIST.md, step 5).

**Versions:** libopensps starts at 0.1.0, soname `libopensps.so.0`. The inherited 3.2.0 and
`.so.3` were libfsps's, and a new library name starts its own numbering. Before 1.0, an
incompatible change raises the minor version and keeps soname 0; from 1.0 on, the soname follows
the major version. The RPM and Debian package versions follow (`libopensps0`). The C API version
(`OPENSPS_API_VERSION_*`, `opensps_get_driver_version`, now 1.2) stays separate: it describes the
interface the bindings check, changes only when entry points do, and would otherwise jump with
every library release.

## Porting a patch from this repository after the rename

`port_patches.sh` does it: run in a libopensps clone, on the branch that should receive the
commits.

```sh
git -C ~/Code/fsps-fork-clone format-patch -o /tmp/port <fork-commit-range>
cd ~/Code/libopensps && git switch -c port/<topic> origin/main
tools/cutover/port_patches.sh --prefix "fix: " /tmp/port/*.patch   # --prefix: a Conventional type
```

For each patch it applies `git am` on top of the import commit (where the tree is the fork's),
runs the same rename on the trees before and after the patch, and applies the difference to the
branch (`git apply -3`), committing with the patch's author, date and message plus
`Ported-from: elijahmathews/fsps@<hash>`. The patch thus arrives renamed exactly as the code it
touches was, including split long lines. Tested on fork PR #31 (the batch-of-one submodule): it
applied cleanly to the renamed tree as `src/physics/opensps_csp.f90` and
`src/physics/opensps_scenario_batch.f90`, and the suites pass on the result.
